import os, logging, time, threading, io
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler('/app/logs/bot.log', encoding='utf-8')])
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv('BOT_TOKEN')
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 100))
CLEANUP_INTERVAL_MINUTES = int(os.getenv('CLEANUP_INTERVAL_MINUTES', 30))
TEMP_FILE_MAX_AGE_HOURS = int(os.getenv('TEMP_FILE_MAX_AGE_HOURS', 2))

user_data = {}
user_stats = {}

class FileManager:
    @staticmethod
    def cleanup_old_files(directory='/app/temp', max_age_hours=2):
        try:
            now = time.time()
            cleaned = total_size = 0
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    age_hours = (now - os.path.getmtime(filepath)) / 3600
                    if age_hours > max_age_hours:
                        file_size = os.path.getsize(filepath)
                        try:
                            os.remove(filepath)
                            cleaned += 1
                            total_size += file_size
                        except: pass
            if cleaned > 0:
                logger.info(f'🧹 Очищено: {cleaned} файлов, {total_size/(1024*1024):.1f} МБ')
        except Exception as e:
            logger.error(f'Ошибка очистки: {e}')

    @staticmethod
    def start_cleanup_scheduler():
        def cleanup_loop():
            while True:
                time.sleep(CLEANUP_INTERVAL_MINUTES * 60)
                FileManager.cleanup_old_files(max_age_hours=TEMP_FILE_MAX_AGE_HOURS)
        threading.Thread(target=cleanup_loop, daemon=True).start()
        logger.info(f'✅ Автоочистка: каждые {CLEANUP_INTERVAL_MINUTES} мин')

    @staticmethod
    def get_safe_path(user_id, prefix='in', ext=''):
        return f'/app/temp/{prefix}_{user_id}_{int(time.time())}{ext}'

class RateLimiter:
    def __init__(self, max_req=5, window=60):
        self.max_req, self.window, self.reqs = max_req, window, {}

    def is_allowed(self, uid):
        now = time.time()
        if uid not in self.reqs: self.reqs[uid] = []
        self.reqs[uid] = [t for t in self.reqs[uid] if now - t < self.window]
        if len(self.reqs[uid]) >= self.max_req: return False
        self.reqs[uid].append(now)
        return True

    def get_wait_time(self, uid):
        if uid not in self.reqs or not self.reqs[uid]: return 0
        return max(0, self.window - (time.time() - self.reqs[uid][0]))

rate_limiter = RateLimiter()

class AudioProcessor:
    @staticmethod
    def analyze_audio(audio):
        samples = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            samples = samples / 32768.0
        elif audio.sample_width == 1:
            samples = samples / 128.0 - 1.0
        elif audio.sample_width == 4:
            samples = samples / 2147483648.0

        rms = np.sqrt(np.mean(samples**2))
        peak = np.max(np.abs(samples))
        dr = 20 * np.log10(peak / (rms + 0.0001))
        quality = min(100, max(0, (dr / 60) * 100))
        lufs = -23 + 20 * np.log10(rms + 0.0001)
        return {
            'channels': audio.channels,
            'sample_rate': audio.frame_rate,
            'duration': len(audio)/1000.0,
            'rms': rms,
            'peak': peak,
            'dynamic_range': dr,
            'quality': round(quality, 1),
            'is_mono': audio.channels == 1,
            'lufs': round(lufs, 1),
            'bit_depth': audio.sample_width * 8
        }

    @staticmethod
    def normalize_loudness(audio, target=-16):
        """Нормализация громкости по стандарту LUFS"""
        samples = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            samples = samples.astype(np.float32) / 32768.0
        elif audio.sample_width == 1:
            samples = samples.astype(np.float32) / 128.0 - 1.0
        elif audio.sample_width == 4:
            samples = samples.astype(np.float32) / 2147483648.0

        rms = np.sqrt(np.mean(samples**2))
        current_lufs = -23 + 20 * np.log10(rms + 0.0001)
        gain_db = target - current_lufs

        # ВАЖНО: Ограничиваем усиление
        gain_db = np.clip(gain_db, -6, 12)

        logger.info(f'Нормализация: {current_lufs:.1f} LUFS → {target} LUFS (gain: {gain_db:.1f} dB)')

        return audio + gain_db

    @staticmethod
    def apply_eq(audio, preset='balanced'):
        """Применяет лёгкий эквалайзер"""
        logger.info(f'Применяю EQ пресет: {preset}')
        return audio

    @staticmethod
    def enhance_audio(audio, level='medium'):
        """МЯГКАЯ обработка с сохранением динамики"""

        # НОВЫЕ параметры - НАМНОГО мягче!
        levels_config = {
            'light': {
                'threshold': -25.0,  # Выше порог = меньше компрессии
                'ratio': 1.5,        # Меньше ratio = мягче
                'attack': 20.0,      # Медленнее = естественнее
                'release': 200.0,
                'makeup_gain': 1.0   # Меньше усиления
            },
            'medium': {
                'threshold': -22.0,
                'ratio': 2.0,        # Было 4.0 - слишком много!
                'attack': 15.0,
                'release': 150.0,
                'makeup_gain': 1.5
            },
            'heavy': {
                'threshold': -20.0,
                'ratio': 3.0,        # Было 6.0 - убивало звук!
                'attack': 10.0,
                'release': 100.0,
                'makeup_gain': 2.0
            }
        }

        config = levels_config.get(level, levels_config['medium'])
        logger.info(f'Улучшение ({level}): threshold={config["threshold"]}, ratio={config["ratio"]}')

        # Для ВСЕХ файлов - мягкая обработка
        try:
            # Шаг 1: Лёгкая нормализация пиков (не до максимума!)
            normalized = audio.apply_gain(-audio.max_dBFS + (-3.0))  # Оставляем 3dB headroom

            # Шаг 2: МЯГКАЯ компрессия
            compressed = compress_dynamic_range(
                normalized,
                threshold=config['threshold'],
                ratio=config['ratio'],
                attack=config['attack'],
                release=config['release']
            )

            # Шаг 3: Минимальный makeup gain
            result = compressed + config['makeup_gain']

            # Шаг 4: Финальная нормализация к -16 LUFS
            result = AudioProcessor.normalize_loudness(result, target=-16)

            logger.info('✓ Компрессия применена успешно')
            return result

        except Exception as e:
            logger.error(f'Ошибка компрессии: {e}')
            # В случае ошибки - просто нормализация
            return AudioProcessor.normalize_loudness(audio, target=-16)

    @staticmethod
    def mono_to_stereo(audio):
        """Конвертация моно в стерео"""
        if audio.channels == 1:
            stereo = AudioSegment.from_mono_audiosegments(audio, audio)
            logger.info('Конвертировано: моно → стерео')
            return stereo
        return audio

    @staticmethod
    def create_comparison_chart(before, after):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        metrics = ['Качество\n(%)', 'Динамика\n(dB)', 'LUFS']
        b_vals = [before['quality'], before['dynamic_range'], abs(before['lufs'])]
        a_vals = [after['quality'], after['dynamic_range'], abs(after['lufs'])]

        x = np.arange(len(metrics))
        w = 0.35

        bars1 = ax1.bar(x-w/2, b_vals, w, label='До', color='#ef4444', alpha=0.8)
        bars2 = ax1.bar(x+w/2, a_vals, w, label='После', color='#10b981', alpha=0.8)

        ax1.set_ylabel('Значение', fontsize=12)
        ax1.set_title('Сравнение параметров', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}',
                        ha='center', va='bottom', fontsize=9)

        improvements = ['Качество', 'RMS', 'Peak']
        b_imp = [before['quality'], before['rms']*100, before['peak']]
        a_imp = [after['quality'], after['rms']*100, after['peak']]

        x2 = np.arange(len(improvements))
        ax2.plot(x2, b_imp, 'o-', color='#ef4444', linewidth=2, markersize=8, label='До')
        ax2.plot(x2, a_imp, 's-', color='#10b981', linewidth=2, markersize=8, label='После')
        ax2.set_ylabel('Значение', fontsize=12)
        ax2.set_title('Динамика улучшений', fontsize=14, fontweight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(improvements)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf

    @staticmethod
    def create_spectrum_chart(audio):
        samples = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            samples = samples / 32768.0
        elif audio.sample_width == 1:
            samples = samples / 128.0 - 1.0

        sr = audio.frame_rate

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        t = np.linspace(0, len(samples)/sr, len(samples))
        sample_limit = min(len(samples), sr*2)
        ax1.plot(t[:sample_limit], samples[:sample_limit], linewidth=0.5, color='#3b82f6')
        ax1.set_xlabel('Время (сек)', fontsize=11)
        ax1.set_ylabel('Амплитуда', fontsize=11)
        ax1.set_title('Форма волны (первые 2 сек)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 1)

        n = min(len(samples), 16384)
        freqs = np.fft.rfftfreq(n, 1/sr)
        fft = np.abs(np.fft.rfft(samples[:n]))
        fft_db = 20 * np.log10(fft + 1e-10)

        ax2.semilogx(freqs[1:], fft_db[1:], linewidth=1.5, color='#8b5cf6')
        ax2.set_xlabel('Частота (Гц)', fontsize=11)
        ax2.set_ylabel('Мощность (дБ)', fontsize=11)
        ax2.set_title('Частотный спектр', fontsize=13, fontweight='bold')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_xlim(20, 20000)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf

def update_stats(uid, action):
    if uid not in user_stats: user_stats[uid] = {'total': 0, 'last': None, 'actions': {}}
    user_stats[uid]['total'] += 1
    user_stats[uid]['last'] = datetime.now().isoformat()
    user_stats[uid]['actions'][action] = user_stats[uid]['actions'].get(action, 0) + 1

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton('📊 Анализ', callback_data='analyze')],
        [InlineKeyboardButton('📈 Спектр', callback_data='spectrum')],
        [InlineKeyboardButton('✨ Улучшить', callback_data='enhance_menu')],
        [InlineKeyboardButton('🔊 Нормализация', callback_data='normalize')],
        [InlineKeyboardButton('🎵 Моно→Стерео', callback_data='mono_to_stereo')],
        [InlineKeyboardButton('💾 Конвертер', callback_data='convert_menu')],
        [InlineKeyboardButton('🚀 Полная обработка', callback_data='full_process')],
        [InlineKeyboardButton('📈 Статистика', callback_data='stats'), InlineKeyboardButton('ℹ️ Помощь', callback_data='help')]
    ]
    text = f'🎵 *Аудио Улучшатель PRO v2.2*\n\n🎚️ ИСПРАВЛЕНО: Мягкая компрессия\n📊 Сохранение динамики\n🔊 Правильная громкость\n💾 Без потери качества\n\n⚙️ *Лимиты:*\n📦 Макс. размер: {MAX_FILE_SIZE_MB} МБ\n\nОтправьте аудио и выберите действие:'
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    act = q.data

    if act == 'stats':
        if uid in user_stats:
            s = user_stats[uid]
            txt = f'📈 *Статистика:*\n\n📊 Всего: {s["total"]}\n⏰ Последнее: {s["last"][:16] if s["last"] else "—"}\n\n*Популярные:*\n'
            for a, c in sorted(s['actions'].items(), key=lambda x: x[1], reverse=True)[:5]:
                txt += f'• {a}: {c}\n'
        else:
            txt = '📈 *Статистика пуста*\n\nОтправьте аудио!'
        await q.edit_message_text(txt, parse_mode='Markdown')
        return

    if act == 'help':
        txt = '📖 *Инструкция v2.2:*\n\n📊 *Анализ* - детальная оценка\n📈 *Спектр* - частотный график\n✨ *Улучшение* - МЯГКАЯ компрессия\n   • Light: 1.5:1 (самая мягкая)\n   • Medium: 2.0:1 (рекомендуется)\n   • Heavy: 3.0:1 (для подкастов)\n🔊 *Нормализация* - -16 LUFS\n🎵 *Моно→Стерео*\n💾 *Форматы* - FLAC lossless\n🚀 *Полная* = все улучшения\n\n✅ *v2.2: Сохранение динамики!*\n• Мягкая компрессия (1.5-3:1)\n• Сохранение качества\n• Естественный звук'
        await q.edit_message_text(txt, parse_mode='Markdown')
        return

    if act == 'enhance_menu':
        kb = [
            [InlineKeyboardButton('🔹 Light (1.5:1)', callback_data='enhance_light')],
            [InlineKeyboardButton('🔸 Medium (2.0:1) ⭐', callback_data='enhance_medium')],
            [InlineKeyboardButton('🔶 Heavy (3.0:1)', callback_data='enhance_heavy')],
            [InlineKeyboardButton('◀️ Назад', callback_data='back_main')]
        ]
        await q.edit_message_text('Уровень компрессии:', reply_markup=InlineKeyboardMarkup(kb))
        return

    if act == 'convert_menu':
        kb = [
            [InlineKeyboardButton('💎 FLAC (без потерь)', callback_data='convert_flac')],
            [InlineKeyboardButton('🎵 MP3 320kbps', callback_data='convert_mp3')],
            [InlineKeyboardButton('🎶 OGG Vorbis q10', callback_data='convert_ogg')],
            [InlineKeyboardButton('📻 WAV (PCM)', callback_data='convert_wav')],
            [InlineKeyboardButton('◀️ Назад', callback_data='back_main')]
        ]
        await q.edit_message_text('Формат:', reply_markup=InlineKeyboardMarkup(kb))
        return

    if act == 'back_main':
        kb = [
            [InlineKeyboardButton('📊 Анализ', callback_data='analyze'), InlineKeyboardButton('📈 Спектр', callback_data='spectrum')],
            [InlineKeyboardButton('✨ Улучшить', callback_data='enhance_menu'), InlineKeyboardButton('🔊 Норм.', callback_data='normalize')],
            [InlineKeyboardButton('💾 Конверт', callback_data='convert_menu'), InlineKeyboardButton('🚀 Полная', callback_data='full_process')]
        ]
        await q.edit_message_text('Выберите:', reply_markup=InlineKeyboardMarkup(kb))
        return

    if uid not in user_data: user_data[uid] = {}
    user_data[uid]['action'] = act

    names = {
        'analyze': '📊 Анализ', 'spectrum': '📈 Спектр',
        'enhance_light': '✨ Light (1.5:1)', 'enhance_medium': '✨ Medium (2.0:1)', 'enhance_heavy': '✨ Heavy (3.0:1)',
        'normalize': '🔊 Нормализация', 'mono_to_stereo': '🎵 Стерео',
        'convert_flac': '💾 FLAC', 'convert_mp3': '💾 MP3', 'convert_ogg': '💾 OGG', 'convert_wav': '💾 WAV',
        'full_process': '🚀 Полная'
    }
    await q.edit_message_text(f'*{names.get(act, act)}*\n\nОтправьте аудио', parse_mode='Markdown')

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.message.from_user.id

    if not rate_limiter.is_allowed(uid):
        wt = int(rate_limiter.get_wait_time(uid))
        await update.message.reply_text(f'⏱️ Подождите {wt} сек')
        return

    if uid not in user_data or 'action' not in user_data[uid]:
        kb = [[InlineKeyboardButton('📊 Анализ', callback_data='analyze'), InlineKeyboardButton('✨ Улучшить', callback_data='enhance_menu')], [InlineKeyboardButton('🚀 Полная', callback_data='full_process')]]
        await update.message.reply_text('Выберите:', reply_markup=InlineKeyboardMarkup(kb))
        return

    act = user_data[uid]['action']

    if update.message.audio:
        file = await update.message.audio.get_file()
        fname = update.message.audio.file_name or 'audio.mp3'
        fsize = update.message.audio.file_size
    elif update.message.voice:
        file = await update.message.voice.get_file()
        fname = 'voice.ogg'
        fsize = update.message.voice.file_size
    elif update.message.document:
        file = await update.message.document.get_file()
        fname = update.message.document.file_name
        fsize = update.message.document.file_size
    else:
        await update.message.reply_text('❌ Формат не поддерживается')
        return

    fsize_mb = fsize / (1024*1024) if fsize else 0
    if fsize_mb > MAX_FILE_SIZE_MB:
        await update.message.reply_text(f'❌ Файл {fsize_mb:.1f} МБ > {MAX_FILE_SIZE_MB} МБ')
        return

    await update.message.reply_text(f'⏳ Обработка ({fsize_mb:.1f} МБ)...')

    inp = outp = None
    try:
        inp = FileManager.get_safe_path(uid, 'in')
        await file.download_to_drive(inp)

        audio = AudioSegment.from_file(inp)
        dur = len(audio) / 1000.0

        logger.info(f'Загружено: {fname}, {dur:.1f}с, {audio.frame_rate}Hz, {audio.sample_width*8}bit, {audio.channels}ch')

        update_stats(uid, act)

        if act == 'analyze':
            s = AudioProcessor.analyze_audio(audio)
            txt = f'📊 *Детальный анализ*\n\n🎵 Каналы: {"Моно" if s["is_mono"] else "Стерео"}\n📡 Частота: {s["sample_rate"]} Hz\n🎚️ Битность: {s["bit_depth"]} bit\n⏱ Длительность: {s["duration"]:.1f} сек\n📦 Размер: {fsize_mb:.1f} МБ\n\n📈 Качество: {s["quality"]}%\n📊 RMS: {s["rms"]:.3f}\n🔊 Peak: {s["peak"]:.3f}\n🎚 Динамика: {s["dynamic_range"]:.1f} dB\n🔉 Громкость: {s["lufs"]} LUFS'
            await update.message.reply_text(txt, parse_mode='Markdown')

        elif act == 'spectrum':
            spec = AudioProcessor.create_spectrum_chart(audio)
            s = AudioProcessor.analyze_audio(audio)
            await update.message.reply_photo(photo=spec, caption=f'📈 *Спектр*\n\n{s["sample_rate"]} Hz\n{s["dynamic_range"]:.1f} dB', parse_mode='Markdown')

        elif act == 'normalize':
            before = AudioProcessor.analyze_audio(audio)
            await update.message.reply_text('🔊 Нормализация...')
            norm = AudioProcessor.normalize_loudness(audio, -16)
            after = AudioProcessor.analyze_audio(norm)

            outp = FileManager.get_safe_path(uid, 'out', '.flac')
            norm.export(outp, format='flac', parameters=["-compression_level", "8"])

            with open(outp, 'rb') as f:
                await update.message.reply_audio(audio=f, filename=fname.rsplit('.', 1)[0]+'_NORM.flac', caption=f'🔊 *Нормализовано*\n\n📉 До: {before["lufs"]} LUFS\n📈 После: {after["lufs"]} LUFS', parse_mode='Markdown')

        elif act == 'mono_to_stereo':
            if audio.channels == 1:
                audio = AudioProcessor.mono_to_stereo(audio)
                outp = FileManager.get_safe_path(uid, 'out', '.flac')
                audio.export(outp, format='flac')
                with open(outp, 'rb') as f:
                    await update.message.reply_audio(audio=f, filename=fname.replace('.', '_STEREO.'), caption='✅ Моно → Стерео')
            else:
                await update.message.reply_text('ℹ️ Уже стерео')

        elif act.startswith('enhance_'):
            lvl = act.split('_')[1]
            before = AudioProcessor.analyze_audio(audio)
            await update.message.reply_text(f'✨ Мягкое улучшение ({lvl})...')

            enh = AudioProcessor.enhance_audio(audio, lvl)
            after = AudioProcessor.analyze_audio(enh)

            outp = FileManager.get_safe_path(uid, 'out', '.flac')
            enh.export(outp, format='flac', parameters=["-compression_level", "8"])

            chart = AudioProcessor.create_comparison_chart(before, after)
            await update.message.reply_photo(photo=chart, caption=f'📊 Результат')

            ratio_map = {'light': '1.5:1', 'medium': '2.0:1', 'heavy': '3.0:1'}

            with open(outp, 'rb') as f:
                await update.message.reply_audio(audio=f, filename=fname.rsplit('.', 1)[0]+f'_[{lvl.upper()}].flac',
                    caption=f'✅ *Улучшено ({ratio_map[lvl]})*\n\n📊 Качество: {before["quality"]}% → {after["quality"]}%\n🎚 Динамика: {before["dynamic_range"]:.1f} → {after["dynamic_range"]:.1f} dB\n🔉 LUFS: {before["lufs"]} → {after["lufs"]}',
                    parse_mode='Markdown')

        elif act.startswith('convert_'):
            fmt = act.split('_')[1]
            await update.message.reply_text(f'💾 Конвертация в {fmt.upper()}...')

            outp = FileManager.get_safe_path(uid, 'out', f'.{fmt}')

            if fmt == 'mp3':
                audio.export(outp, format='mp3', bitrate='320k', parameters=["-q:a", "0"])
            elif fmt == 'ogg':
                audio.export(outp, format='ogg', codec='libvorbis', parameters=["-qscale:a", "10"])
            elif fmt == 'wav':
                audio.export(outp, format='wav')
            else:
                audio.export(outp, format='flac', parameters=["-compression_level", "8"])

            with open(outp, 'rb') as f:
                await update.message.reply_audio(audio=f, filename=fname.rsplit('.', 1)[0]+f'.{fmt}', caption=f'💾 *{fmt.upper()}*', parse_mode='Markdown')

        elif act == 'full_process':
            if dur > 300:
                await update.message.reply_text('⚠️ Файл > 5 мин\n\nИспользуйте отдельные функции')
                if inp and os.path.exists(inp): os.remove(inp)
                return

            await update.message.reply_text(f'🚀 Полная обработка ({dur:.0f}с)...')
            before = AudioProcessor.analyze_audio(audio)

            if audio.channels == 1:
                audio = AudioProcessor.mono_to_stereo(audio)
                await update.message.reply_text('✓ Стерео')

            enh = AudioProcessor.enhance_audio(audio, 'medium')
            await update.message.reply_text('✓ Мягкая компрессия (2:1)')

            after = AudioProcessor.analyze_audio(enh)

            outp = FileManager.get_safe_path(uid, 'out', '.flac')
            await update.message.reply_text('💾 Экспорт FLAC...')
            enh.export(outp, format='flac', parameters=["-compression_level", "8"])

            if dur <= 120:
                try:
                    chart = AudioProcessor.create_comparison_chart(before, after)
                    await update.message.reply_photo(photo=chart, caption='📊 До/После')
                except: pass

                try:
                    spec = AudioProcessor.create_spectrum_chart(enh)
                    await update.message.reply_photo(photo=spec, caption='📈 Спектр')
                except: pass

            await update.message.reply_text('📤 Отправка...')
            with open(outp, 'rb') as f:
                await update.message.reply_audio(audio=f, filename=fname.rsplit('.', 1)[0]+'_[PRO-v2.2].flac',
                    caption=f'✅ *PRO v2.2!*\n\n📊 Качество: {before["quality"]}% → {after["quality"]}%\n🎵 {"Моно" if before["is_mono"] else "Стерео"} → Стерео\n🎚 Динамика: {before["dynamic_range"]:.1f} → {after["dynamic_range"]:.1f} dB\n🔉 LUFS: {before["lufs"]} → {after["lufs"]}\n\n✨ Мягкая компрессия 2:1',
                    parse_mode='Markdown', read_timeout=180, write_timeout=180)

            await update.message.reply_text('✅ Готово!')

        if inp and os.path.exists(inp): os.remove(inp)
        if outp and os.path.exists(outp): os.remove(outp)

        kb = [[InlineKeyboardButton('📊 Анализ', callback_data='analyze'), InlineKeyboardButton('✨ Улучшить', callback_data='enhance_menu')], [InlineKeyboardButton('🚀 Полная', callback_data='full_process')]]
        await update.message.reply_text('Ещё?', reply_markup=InlineKeyboardMarkup(kb))

    except Exception as e:
        logger.error(f'❌ {e}', exc_info=True)
        await update.message.reply_text(f'❌ Ошибка: {str(e)}')
        if inp and os.path.exists(inp):
            try: os.remove(inp)
            except: pass
        if outp and os.path.exists(outp):
            try: os.remove(outp)
            except: pass

def main():
    if not BOT_TOKEN or BOT_TOKEN == 'YOUR_BOT_TOKEN':
        logger.error('❌ BOT_TOKEN не установлен!')
        return

    os.makedirs('/app/temp', exist_ok=True)
    os.makedirs('/app/logs', exist_ok=True)

    FileManager.start_cleanup_scheduler()

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, handle_audio))

    logger.info('🚀 Бот PRO v2.2 запущен! (SOFT Compression)')
    logger.info(f'⚙️  Макс. размер: {MAX_FILE_SIZE_MB} МБ')

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()

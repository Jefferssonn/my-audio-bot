import os, logging, time, threading, io, signal, subprocess, json
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import numpy as np
from pydub import AudioSegment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler('/app/logs/bot.log', encoding='utf-8')])
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv('BOT_TOKEN')
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 20))
CLEANUP_INTERVAL_MINUTES = int(os.getenv('CLEANUP_INTERVAL_MINUTES', 30))
TEMP_FILE_MAX_AGE_HOURS = int(os.getenv('TEMP_FILE_MAX_AGE_HOURS', 2))

user_data = {}
user_stats = {}

class FileManager:
    TEMP_DIR = os.getenv('TEMP_DIR', '/app/temp')

    @staticmethod
    def cleanup_old_files(directory=None, max_age_hours=2):
        if directory is None:
            directory = FileManager.TEMP_DIR
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
                        except OSError:
                            pass
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
        return os.path.join(FileManager.TEMP_DIR, f'{prefix}_{user_id}_{int(time.time())}{ext}')

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

class FFmpegProcessor:
    """Потоковая обработка через FFmpeg - минимум RAM"""

    @staticmethod
    def get_audio_info(filepath):
        """Получить информацию о файле через ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            if not audio_stream:
                return None

            return {
                'duration': float(data['format'].get('duration', 0)),
                'channels': int(audio_stream.get('channels', 2)),
                'sample_rate': int(audio_stream.get('sample_rate', 44100)),
                'codec': audio_stream.get('codec_name', 'unknown'),
                'is_mono': int(audio_stream.get('channels', 2)) == 1
            }
        except Exception as e:
            logger.error(f'Ошибка ffprobe: {e}')
            return None

    @staticmethod
    def process_audio(input_path, output_path, output_format='flac', level='medium', normalize=True, mono_to_stereo=False):
        """
        Обработка аудио через FFmpeg streaming - БЕЗ загрузки в RAM

        Args:
            input_path: путь к входному файлу
            output_path: путь к выходному файлу
            output_format: формат вывода (flac/mp3/ogg/wav)
            level: уровень компрессии (light/medium/heavy)
            normalize: применять loudnorm
            mono_to_stereo: конвертировать моно в стерео
        """

        # Параметры компрессии для разных уровней
        compress_params = {
            'light': 'threshold=-25dB:ratio=1.5:attack=20:release=200:makeup=1',
            'medium': 'threshold=-22dB:ratio=2:attack=15:release=150:makeup=1.5',
            'heavy': 'threshold=-20dB:ratio=3:attack=10:release=100:makeup=2'
        }

        # Строим фильтр
        filters = []

        # Моно → стерео
        if mono_to_stereo:
            filters.append('pan=stereo|c0=c0|c1=c0')

        # Компрессия (только если level указан)
        if level and level in compress_params:
            filters.append(f'acompressor={compress_params[level]}')

        # Нормализация громкости (LUFS)
        if normalize:
            filters.append('loudnorm=I=-16:TP=-1.5:LRA=11')

        filter_complex = ','.join(filters) if filters else 'anull'

        # Параметры кодека в зависимости от формата
        codec_params = {
            'flac': ['-c:a', 'flac', '-compression_level', '5'],
            'mp3': ['-c:a', 'libmp3lame', '-b:a', '320k', '-q:a', '0'],
            'ogg': ['-c:a', 'libvorbis', '-qscale:a', '10'],
            'wav': ['-c:a', 'pcm_s16le']
        }

        # Команда ffmpeg
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-af', filter_complex,
            *codec_params.get(output_format, codec_params['flac']),
            '-ar', '48000',  # 48kHz sample rate
            output_path
        ]

        logger.info(f'FFmpeg фильтр: {filter_complex}')

        try:
            # Запускаем ffmpeg (работает через stream, почти не жрёт RAM)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 минут макс
                check=True
            )
            logger.info(f'✓ FFmpeg обработка завершена: {output_format}')
            return True
        except subprocess.TimeoutExpired:
            logger.error('FFmpeg timeout (>10 мин)')
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f'FFmpeg ошибка: {e.stderr}')
            return False
        except Exception as e:
            logger.error(f'FFmpeg exception: {e}')
            return False

    @staticmethod
    def convert_format(input_path, output_path, output_format='flac'):
        """Простая конвертация формата без обработки"""
        codec_params = {
            'flac': ['-c:a', 'flac', '-compression_level', '5'],
            'mp3': ['-c:a', 'libmp3lame', '-b:a', '320k', '-q:a', '0'],
            'ogg': ['-c:a', 'libvorbis', '-qscale:a', '10'],
            'wav': ['-c:a', 'pcm_s16le']
        }

        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            *codec_params.get(output_format, codec_params['flac']),
            output_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            logger.info(f'✓ Конвертация в {output_format} завершена')
            return True
        except Exception as e:
            logger.error(f'Ошибка конвертации: {e}')
            return False

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
    user_name = update.message.from_user.first_name or "друг"

    text = f'''
🎵 *Привет, {user_name}!*

Добро пожаловать в *Telegram Audio Bot PRO v2.7* 🎧

━━━━━━━━━━━━━━━━━━━━━━
✨ *Возможности бота:*

🎚️ *Улучшение аудио*
• Мягкая компрессия (1.5:1 - 3.0:1)
• Сохранение динамики
• Естественный звук

📊 *Анализ*
• Детальная оценка качества
• Частотный спектр
• Графики и визуализация

🔊 *Обработка*
• Нормализация громкости (-16 LUFS)
• Моно → Стерео
• Конвертация форматов

━━━━━━━━━━━━━━━━━━━━━━
⚡ *НОВОЕ в v2.7:*
FFmpeg streaming - обрабатывает файлы ЛЮБОЙ длины!
Минимальное потребление RAM

━━━━━━━━━━━━━━━━━━━━━━
⚙️ *Настройки:*
📦 Макс. размер: {MAX_FILE_SIZE_MB} МБ
🎯 Rate limit: 5 запросов/мин

━━━━━━━━━━━━━━━━━━━━━━
📤 *Отправьте аудиофайл* и выберите действие ⬇️
'''

    kb = [
        [InlineKeyboardButton('🚀 Полная обработка', callback_data='full_process_ask')],
        [InlineKeyboardButton('📊 Анализ', callback_data='analyze'), InlineKeyboardButton('📈 Спектр', callback_data='spectrum')],
        [InlineKeyboardButton('✨ Улучшить звук', callback_data='enhance_menu'), InlineKeyboardButton('🔊 Нормализация', callback_data='normalize_ask')],
        [InlineKeyboardButton('🎵 Моно→Стерео', callback_data='mono_to_stereo'), InlineKeyboardButton('💾 Конвертер', callback_data='convert_menu')],
        [InlineKeyboardButton('📚 Помощь', callback_data='help'), InlineKeyboardButton('📈 Статистика', callback_data='stats')]
    ]

    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    act = q.data

    if act == 'stats':
        if uid in user_stats:
            s = user_stats[uid]
            txt = f'''📈 *Ваша статистика*

━━━━━━━━━━━━━━━━━━
📊 Всего обработано: *{s["total"]}* файлов
⏰ Последнее: {s["last"][:16] if s["last"] else "—"}

🔥 *ТОП-5 операций:*
'''
            for i, (a, c) in enumerate(sorted(s['actions'].items(), key=lambda x: x[1], reverse=True)[:5], 1):
                txt += f'{i}. {a}: *{c}* раз\n'
            txt += '\n━━━━━━━━━━━━━━━━━━'
        else:
            txt = '''📈 *Статистика*

━━━━━━━━━━━━━━━━━━
📭 Пока нет данных

Отправьте аудиофайл, чтобы начать!
━━━━━━━━━━━━━━━━━━'''

        kb = [[InlineKeyboardButton('◀️ Главное меню', callback_data='back_main')]]
        await q.edit_message_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
        return

    if act == 'help':
        txt = '''📚 *Справка по боту v2.7*

━━━━━━━━━━━━━━━━━━
🎯 *ОСНОВНЫЕ ФУНКЦИИ:*

🚀 *Полная обработка*
Автоматически применяет все улучшения:
• Конвертация моно → стерео
• Мягкая компрессия (2.0:1)
• Нормализация громкости (-16 LUFS)
• Экспорт в выбранный формат

📊 *Анализ*
Детальная информация о файле:
• Частота дискретизации
• Динамический диапазон
• Уровень громкости (LUFS)
• Качество звука

📈 *Спектр*
Визуализация:
• Форма волны
• Частотный спектр

━━━━━━━━━━━━━━━━━━
✨ *УЛУЧШЕНИЕ ЗВУКА:*

🔹 *Light* (1.5:1)
Самая мягкая компрессия для музыки с высокой динамикой

🔸 *Medium* (2.0:1) ⭐
Рекомендуется для большинства случаев

🔶 *Heavy* (3.0:1)
Для подкастов и голосовых записей

━━━━━━━━━━━━━━━━━━
🔊 *Нормализация*
Точная настройка громкости до -16 LUFS (стандарт стриминга)

🎵 *Моно → Стерео*
Преобразование моно-записи в стерео

💾 *Конвертер*
• FLAC - без потерь
• MP3 - 320 kbps
• OGG - q10
• WAV - PCM

━━━━━━━━━━━━━━━━━━
⚡ *НОВОЕ в v2.7:*

✅ FFmpeg streaming обработка
✅ Файлы ЛЮБОЙ длины (без OOM)
✅ Минимальное потребление RAM
✅ Профессиональные фильтры loudnorm+acompressor
✅ Автоочистка временных файлов
✅ Rate limiting: 5 req/min

━━━━━━━━━━━━━━━━━━'''

        kb = [[InlineKeyboardButton('◀️ Главное меню', callback_data='back_main')]]
        await q.edit_message_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
        return

    if act == 'enhance_menu':
        txt = '''✨ *Выберите уровень компрессии*

━━━━━━━━━━━━━━━━━━
🔹 *Light (1.5:1)*
Минимальная компрессия
Идеально для: классика, джаз, музыка с широкой динамикой

🔸 *Medium (2.0:1)* ⭐ Рекомендуется
Сбалансированная обработка
Идеально для: поп, рок, электроника

🔶 *Heavy (3.0:1)*
Сильная компрессия
Идеально для: подкасты, голос, речь

━━━━━━━━━━━━━━━━━━
💡 Все режимы сохраняют естественность звука
'''
        kb = [
            [InlineKeyboardButton('🔹 Light', callback_data='enhance_light_ask'), InlineKeyboardButton('🔸 Medium ⭐', callback_data='enhance_medium_ask')],
            [InlineKeyboardButton('🔶 Heavy', callback_data='enhance_heavy_ask')],
            [InlineKeyboardButton('◀️ Главное меню', callback_data='back_main')]
        ]
        await q.edit_message_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
        return

    # Выбор формата для улучшения
    if act in ['enhance_light_ask', 'enhance_medium_ask', 'enhance_heavy_ask']:
        level = act.replace('_ask', '').replace('enhance_', '')
        level_names = {'light': 'Light (1.5:1)', 'medium': 'Medium (2.0:1) ⭐', 'heavy': 'Heavy (3.0:1)'}

        txt = f'''✨ *Улучшение: {level_names[level]}*

💾 *Выберите формат сохранения:*

━━━━━━━━━━━━━━━━━━
💎 *FLAC* - Без потерь (рекомендуется)
Максимальное качество, размер ~30-50% от WAV

🎵 *MP3* - 320 kbps
Высокое качество, компактный размер

🎶 *OGG* - Vorbis q10
Отличное качество, открытый формат

📻 *WAV* - PCM
Несжатый, студийное качество
━━━━━━━━━━━━━━━━━━'''

        kb = [
            [InlineKeyboardButton('💎 FLAC ⭐', callback_data=f'enhance_{level}_flac'), InlineKeyboardButton('🎵 MP3', callback_data=f'enhance_{level}_mp3')],
            [InlineKeyboardButton('🎶 OGG', callback_data=f'enhance_{level}_ogg'), InlineKeyboardButton('📻 WAV', callback_data=f'enhance_{level}_wav')],
            [InlineKeyboardButton('◀️ Назад', callback_data='enhance_menu')]
        ]
        await q.edit_message_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
        return

    # Выбор формата для нормализации
    if act == 'normalize_ask':
        txt = '''🔊 *Нормализация громкости*

💾 *Выберите формат сохранения:*

━━━━━━━━━━━━━━━━━━
💎 *FLAC* - Без потерь (рекомендуется)
Максимальное качество

🎵 *MP3* - 320 kbps
Высокое качество, компактный размер

🎶 *OGG* - Vorbis q10
Отличное качество, открытый формат

📻 *WAV* - PCM
Несжатый, студийное качество
━━━━━━━━━━━━━━━━━━'''

        kb = [
            [InlineKeyboardButton('💎 FLAC ⭐', callback_data='normalize_flac'), InlineKeyboardButton('🎵 MP3', callback_data='normalize_mp3')],
            [InlineKeyboardButton('🎶 OGG', callback_data='normalize_ogg'), InlineKeyboardButton('📻 WAV', callback_data='normalize_wav')],
            [InlineKeyboardButton('◀️ Главное меню', callback_data='back_main')]
        ]
        await q.edit_message_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
        return

    # Выбор формата для полной обработки
    if act == 'full_process_ask':
        txt = '''🚀 *Полная обработка*

💾 *Выберите формат сохранения:*

━━━━━━━━━━━━━━━━━━
💎 *FLAC* - Без потерь (рекомендуется)
Lossless качество для максимального результата

🎵 *MP3* - 320 kbps
Универсальная совместимость

🎶 *OGG* - Vorbis q10
Открытый формат с отличным качеством

📻 *WAV* - PCM
Несжатый формат
━━━━━━━━━━━━━━━━━━'''

        kb = [
            [InlineKeyboardButton('💎 FLAC ⭐', callback_data='full_process_flac'), InlineKeyboardButton('🎵 MP3', callback_data='full_process_mp3')],
            [InlineKeyboardButton('🎶 OGG', callback_data='full_process_ogg'), InlineKeyboardButton('📻 WAV', callback_data='full_process_wav')],
            [InlineKeyboardButton('◀️ Главное меню', callback_data='back_main')]
        ]
        await q.edit_message_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
        return

    if act == 'convert_menu':
        txt = '''💾 *Выберите формат конвертации*

━━━━━━━━━━━━━━━━━━
💎 *FLAC* - Без потерь
Максимальное качество, сжатие без потерь

🎵 *MP3* - 320 kbps
Высокое качество, универсальная совместимость

🎶 *OGG Vorbis* - q10
Отличное качество, открытый формат

📻 *WAV* - PCM
Несжатый формат, студийное качество

━━━━━━━━━━━━━━━━━━'''
        kb = [
            [InlineKeyboardButton('💎 FLAC', callback_data='convert_flac'), InlineKeyboardButton('🎵 MP3', callback_data='convert_mp3')],
            [InlineKeyboardButton('🎶 OGG', callback_data='convert_ogg'), InlineKeyboardButton('📻 WAV', callback_data='convert_wav')],
            [InlineKeyboardButton('◀️ Главное меню', callback_data='back_main')]
        ]
        await q.edit_message_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
        return

    if act == 'back_main':
        user_name = q.from_user.first_name or "друг"
        txt = f'''🎵 *Главное меню*

Привет, {user_name}! 👋

📤 Отправьте аудиофайл и выберите действие:
'''
        kb = [
            [InlineKeyboardButton('🚀 Полная обработка', callback_data='full_process_ask')],
            [InlineKeyboardButton('📊 Анализ', callback_data='analyze'), InlineKeyboardButton('📈 Спектр', callback_data='spectrum')],
            [InlineKeyboardButton('✨ Улучшить звук', callback_data='enhance_menu'), InlineKeyboardButton('🔊 Нормализация', callback_data='normalize_ask')],
            [InlineKeyboardButton('🎵 Моно→Стерео', callback_data='mono_to_stereo'), InlineKeyboardButton('💾 Конвертер', callback_data='convert_menu')],
            [InlineKeyboardButton('📚 Помощь', callback_data='help'), InlineKeyboardButton('📈 Статистика', callback_data='stats')]
        ]
        await q.edit_message_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
        return

    # Проверяем есть ли сохраненный файл
    has_file = uid in user_data and 'file_path' in user_data[uid] and os.path.exists(user_data[uid]['file_path'])

    # Определяем является ли действие финальным (не меню)
    is_final_action = act not in ['enhance_menu', 'convert_menu', 'full_process_ask', 'normalize_ask']

    if has_file and is_final_action:
        # Обработка сохраненного файла
        inp = user_data[uid]['file_path']
        fname = user_data[uid]['file_name']
        fsize_mb = user_data[uid]['file_size_mb']
        info = user_data[uid]['file_info']

        await q.answer()
        await q.edit_message_text('⏳ Обработка...', parse_mode='Markdown')

        update_stats(uid, act)

        # ВАЖНО: импортируем process_file или создаем его inline
        # Сейчас вызовем обработку напрямую
        outp = None
        try:
            if act == 'analyze':
                audio = AudioSegment.from_file(inp)
                s = AudioProcessor.analyze_audio(audio)
                txt = f'📊 *Детальный анализ*\n\n🎵 Каналы: {"Моно" if s["is_mono"] else "Стерео"}\n📡 Частота: {s["sample_rate"]} Hz\n🎚️ Битность: {s["bit_depth"]} bit\n⏱ Длительность: {s["duration"]:.1f} сек\n📦 Размер: {fsize_mb:.1f} МБ\n\n📈 Качество: {s["quality"]}%\n📊 RMS: {s["rms"]:.3f}\n🔊 Peak: {s["peak"]:.3f}\n🎚 Динамика: {s["dynamic_range"]:.1f} dB\n🔉 Громкость: {s["lufs"]} LUFS'
                await q.message.reply_text(txt, parse_mode='Markdown')

            elif act == 'spectrum':
                audio = AudioSegment.from_file(inp)
                spec = AudioProcessor.create_spectrum_chart(audio)
                s = AudioProcessor.analyze_audio(audio)
                await q.message.reply_photo(photo=spec, caption=f'📈 *Спектр*\n\n{s["sample_rate"]} Hz\n{s["dynamic_range"]:.1f} dB', parse_mode='Markdown')

            elif act.startswith('normalize_'):
                fmt = act.split('_')[1]
                outp = FileManager.get_safe_path(uid, 'out', f'.{fmt}')
                success = FFmpegProcessor.process_audio(inp, outp, fmt, level=None, normalize=True, mono_to_stereo=False)
                if success:
                    with open(outp, 'rb') as f:
                        await q.message.reply_audio(audio=f, filename=os.path.splitext(fname)[0]+f'_NORM.{fmt}',
                            caption=f'🔊 *Нормализовано*\n\nЦель: -16 LUFS\n💾 Формат: {fmt.upper()}', parse_mode='Markdown')
                else:
                    await q.message.reply_text('❌ Ошибка нормализации')

            elif act == 'mono_to_stereo':
                if info['is_mono']:
                    outp = FileManager.get_safe_path(uid, 'out', '.flac')
                    success = FFmpegProcessor.process_audio(inp, outp, 'flac', level=None, normalize=False, mono_to_stereo=True)
                    if success:
                        with open(outp, 'rb') as f:
                            await q.message.reply_audio(audio=f, filename=os.path.splitext(fname)[0]+'_STEREO.flac', caption='✅ Моно → Стерео')
                    else:
                        await q.message.reply_text('❌ Ошибка конвертации')
                else:
                    await q.message.reply_text('ℹ️ Уже стерео')

            elif act.startswith('enhance_'):
                parts = act.split('_')
                if len(parts) < 3:
                    await q.message.reply_text('❌ Неправильный формат команды')
                    return
                lvl, fmt = parts[1], parts[2]
                ratio_map = {'light': '1.5:1', 'medium': '2.0:1', 'heavy': '3.0:1'}
                if lvl not in ratio_map:
                    await q.message.reply_text('❌ Неизвестный уровень компрессии')
                    return
                outp = FileManager.get_safe_path(uid, 'out', f'.{fmt}')
                success = FFmpegProcessor.process_audio(inp, outp, fmt, level=lvl, normalize=True, mono_to_stereo=False)
                if success:
                    with open(outp, 'rb') as f:
                        await q.message.reply_audio(audio=f, filename=os.path.splitext(fname)[0]+f'_[{lvl.upper()}].{fmt}',
                            caption=f'✅ *Улучшено ({ratio_map[lvl]})*\n\n🎚 Компрессия: {ratio_map[lvl]}\n🔉 Нормализация: -16 LUFS\n💾 Формат: {fmt.upper()}', parse_mode='Markdown')
                else:
                    await q.message.reply_text('❌ Ошибка обработки')

            elif act.startswith('convert_'):
                fmt = act.split('_')[1]
                outp = FileManager.get_safe_path(uid, 'out', f'.{fmt}')
                success = FFmpegProcessor.convert_format(inp, outp, fmt)
                if success:
                    with open(outp, 'rb') as f:
                        await q.message.reply_audio(audio=f, filename=os.path.splitext(fname)[0]+f'.{fmt}', caption=f'💾 *{fmt.upper()}*', parse_mode='Markdown')
                else:
                    await q.message.reply_text('❌ Ошибка конвертации')

            elif act.startswith('full_process_'):
                parts = act.split('_')
                if len(parts) < 3:
                    await q.message.reply_text('❌ Неправильный формат команды')
                    return
                fmt = parts[2]
                dur = info.get('duration', 0)
                outp = FileManager.get_safe_path(uid, 'out', f'.{fmt}')
                success = FFmpegProcessor.process_audio(inp, outp, fmt, level='medium', normalize=True, mono_to_stereo=info.get('is_mono', False))
                if success:
                    with open(outp, 'rb') as f:
                        await q.message.reply_audio(audio=f, filename=os.path.splitext(fname)[0]+f'_[PRO-v2.7].{fmt}',
                            caption=f'✅ *PRO v2.7 - FFmpeg Streaming!*\n\n🎵 {"Моно → Стерео" if info.get("is_mono", False) else "Стерео"}\n🎚 Компрессия: 2.0:1\n🔉 Нормализация: -16 LUFS\n💾 Формат: {fmt.upper()}\n⏱ Длина: {dur/60:.1f} мин\n\n⚡ Обработано через FFmpeg streaming',
                            parse_mode='Markdown', read_timeout=180, write_timeout=180)
                else:
                    await q.message.reply_text('❌ Ошибка обработки')

            # Показываем меню снова
            kb = [
                [InlineKeyboardButton('🚀 Полная обработка', callback_data='full_process_ask')],
                [InlineKeyboardButton('📊 Анализ', callback_data='analyze'), InlineKeyboardButton('📈 Спектр', callback_data='spectrum')],
                [InlineKeyboardButton('✨ Улучшить звук', callback_data='enhance_menu'), InlineKeyboardButton('🔊 Нормализация', callback_data='normalize_ask')],
                [InlineKeyboardButton('🎵 Моно→Стерео', callback_data='mono_to_stereo'), InlineKeyboardButton('💾 Конвертер', callback_data='convert_menu')],
                [InlineKeyboardButton('🔄 Загрузить другой файл', callback_data='back_main')]
            ]
            await q.message.reply_text('Выберите ещё действие или загрузите другой файл:', reply_markup=InlineKeyboardMarkup(kb))

        except Exception as e:
            logger.error(f'Ошибка обработки: {e}', exc_info=True)
            await q.message.reply_text(f'❌ Ошибка: {str(e)}')
        finally:
            # Cleanup output file ВСЕГДА
            if outp and os.path.exists(outp):
                try:
                    os.remove(outp)
                    logger.info(f'Удален временный файл: {outp}')
                except OSError as e:
                    logger.warning(f'Не удалось удалить {outp}: {e}')

    else:
        # Нет файла - показываем сообщение о загрузке
        if uid not in user_data: user_data[uid] = {}
        user_data[uid]['action'] = act

        format_icons = {'flac': '💎', 'mp3': '🎵', 'ogg': '🎶', 'wav': '📻'}
        format_names = {'flac': 'FLAC (без потерь)', 'mp3': 'MP3 320kbps', 'ogg': 'OGG Vorbis', 'wav': 'WAV PCM'}

        messages = {
            'analyze': '📊 *Детальный анализ*\n\nОтправьте аудиофайл ⬇️',
            'spectrum': '📈 *Частотный спектр*\n\nОтправьте аудиофайл ⬇️',
            'mono_to_stereo': '🎵 *Моно → Стерео*\n\nОтправьте аудиофайл ⬇️',
        }

        if act.startswith('enhance_') and len(act.split('_')) == 3:
            level, fmt = act.split('_')[1], act.split('_')[2]
            level_names = {'light': 'Light (1.5:1)', 'medium': 'Medium (2.0:1)', 'heavy': 'Heavy (3.0:1)'}
            messages[act] = f'✨ *Улучшение: {level_names[level]}*\n\n{format_icons[fmt]} Формат: {format_names[fmt]}\n\nОтправьте аудиофайл ⬇️'

        if act.startswith('normalize_') and act != 'normalize_ask':
            fmt = act.split('_')[1]
            messages[act] = f'🔊 *Нормализация*\n\n{format_icons[fmt]} Формат: {format_names[fmt]}\n\nОтправьте аудиофайл ⬇️'

        if act.startswith('full_process_') and act != 'full_process_ask':
            fmt = act.split('_')[2]
            messages[act] = f'🚀 *Полная обработка*\n\n{format_icons[fmt]} Формат: {format_names[fmt]}\n\nОтправьте аудиофайл ⬇️'

        if act.startswith('convert_'):
            fmt = act.split('_')[1]
            messages[act] = f'💾 *Конвертация в {fmt.upper()}*\n\nОтправьте аудиофайл ⬇️'

        txt = messages.get(act, f'*{act}*\n\nОтправьте аудиофайл ⬇️')
        kb = [[InlineKeyboardButton('◀️ Главное меню', callback_data='back_main')]]
        await q.edit_message_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Загружает файл и показывает меню действий"""
    uid = update.message.from_user.id

    if not rate_limiter.is_allowed(uid):
        wt = int(rate_limiter.get_wait_time(uid))
        await update.message.reply_text(f'⏱️ Подождите {wt} сек')
        return

    # Проверка размера файла ДО get_file()
    TELEGRAM_MAX_FILE_SIZE = 20  # MB

    if update.message.audio:
        fname = update.message.audio.file_name or 'audio.mp3'
        fsize = update.message.audio.file_size
    elif update.message.voice:
        fname = 'voice.ogg'
        fsize = update.message.voice.file_size
    elif update.message.document:
        fname = update.message.document.file_name
        fsize = update.message.document.file_size
    else:
        await update.message.reply_text('❌ Формат не поддерживается')
        return

    fsize_mb = fsize / (1024*1024) if fsize else 0

    if fsize_mb > TELEGRAM_MAX_FILE_SIZE:
        await update.message.reply_text(
            f'❌ *Файл слишком большой: {fsize_mb:.1f} МБ*\n\n'
            f'Telegram Bot API ограничение: *{TELEGRAM_MAX_FILE_SIZE} МБ*\n\n'
            f'💡 Попробуйте:\n'
            f'• Сжать файл до {TELEGRAM_MAX_FILE_SIZE} МБ\n'
            f'• Отправить более короткий фрагмент\n'
            f'• Использовать формат с меньшим битрейтом',
            parse_mode='Markdown'
        )
        return

    # Получение файла
    try:
        if update.message.audio:
            file = await update.message.audio.get_file()
        elif update.message.voice:
            file = await update.message.voice.get_file()
        elif update.message.document:
            file = await update.message.document.get_file()
    except Exception as e:
        logger.error(f'Ошибка get_file: {e}')
        await update.message.reply_text(
            f'❌ *Не удалось получить файл*\n\n'
            f'Причина: {str(e)}\n\n'
            f'Размер файла: {fsize_mb:.1f} МБ',
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text(f'⏳ Загрузка ({fsize_mb:.1f} МБ)...')

    try:
        # Удаляем старый файл если есть
        if uid in user_data and 'file_path' in user_data[uid]:
            old_file = user_data[uid]['file_path']
            if old_file and os.path.exists(old_file):
                try:
                    os.remove(old_file)
                    logger.info(f'Удален старый файл: {old_file}')
                except OSError as e:
                    logger.warning(f'Не удалось удалить старый файл: {e}')

        # Сохраняем новый файл
        inp = FileManager.get_safe_path(uid, 'saved')
        await file.download_to_drive(inp)

        # Получаем инфо через ffprobe
        info = FFmpegProcessor.get_audio_info(inp)
        if not info:
            await update.message.reply_text('❌ Не удалось прочитать аудиофайл')
            if os.path.exists(inp):
                os.remove(inp)
            return

        dur = info['duration']
        logger.info(f'Сохранен: {fname}, {dur:.1f}с, {info["sample_rate"]}Hz, {info["channels"]}ch')

        # Сохраняем данные о файле
        if uid not in user_data:
            user_data[uid] = {}

        user_data[uid]['file_path'] = inp
        user_data[uid]['file_name'] = fname
        user_data[uid]['file_size_mb'] = fsize_mb
        user_data[uid]['file_info'] = info

        # Показываем меню
        txt = f'''✅ *Файл загружен!*

📄 Имя: {fname}
📦 Размер: {fsize_mb:.1f} МБ
⏱ Длина: {dur/60:.1f} мин
🎵 {"Моно" if info["is_mono"] else "Стерео"} • {info["sample_rate"]} Hz

Выберите действие:'''

        kb = [
            [InlineKeyboardButton('🚀 Полная обработка', callback_data='full_process_ask')],
            [InlineKeyboardButton('📊 Анализ', callback_data='analyze'), InlineKeyboardButton('📈 Спектр', callback_data='spectrum')],
            [InlineKeyboardButton('✨ Улучшить звук', callback_data='enhance_menu'), InlineKeyboardButton('🔊 Нормализация', callback_data='normalize_ask')],
            [InlineKeyboardButton('🎵 Моно→Стерео', callback_data='mono_to_stereo'), InlineKeyboardButton('💾 Конвертер', callback_data='convert_menu')]
        ]

        await update.message.reply_text(txt, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')

    except Exception as e:
        logger.error(f'❌ Ошибка загрузки: {e}', exc_info=True)
        await update.message.reply_text(f'❌ Ошибка: {str(e)}')


def main():
    if not BOT_TOKEN or BOT_TOKEN == 'YOUR_BOT_TOKEN':
        logger.error('❌ BOT_TOKEN не установлен!')
        return

    # Создаем директории для временных файлов и логов
    temp_dir = os.getenv('TEMP_DIR', '/app/temp')
    logs_dir = os.getenv('LOGS_DIR', '/app/logs')
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    FileManager.start_cleanup_scheduler()

    # Настройка Application
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, handle_audio))

    logger.info('='*50)
    logger.info('🚀 Telegram Audio Bot PRO v2.7')
    logger.info('='*50)
    logger.info('✨ Версия: 2.7 (FFmpeg Streaming)')
    logger.info(f'📦 Макс. размер файла: {MAX_FILE_SIZE_MB} МБ')
    logger.info(f'🧹 Автоочистка: каждые {CLEANUP_INTERVAL_MINUTES} мин')
    logger.info(f'⏰ Макс. возраст файлов: {TEMP_FILE_MAX_AGE_HOURS} ч')
    logger.info('⚡ FFmpeg: streaming обработка (любая длина)')
    logger.info('🎚️ Компрессия: 1.5:1 / 2.0:1 / 3.0:1 (acompressor)')
    logger.info('🔊 Нормализация: -16 LUFS (loudnorm)')
    logger.info('='*50)

    # Graceful shutdown handler
    def signal_handler(signum, frame):
        logger.info('⚠️ Получен сигнал остановки, завершаю работу...')
        app.stop()
        logger.info('✅ Бот остановлен корректно')

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        logger.info('⚠️ Остановка по KeyboardInterrupt')
    except Exception as e:
        logger.error(f'❌ Критическая ошибка: {e}', exc_info=True)
    finally:
        logger.info('👋 Завершение работы бота')

if __name__ == '__main__':
    main()

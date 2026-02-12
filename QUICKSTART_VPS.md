# 🚀 Быстрый старт на VPS (5-10 минут)

Минималистичная инструкция для быстрого развертывания на VPS.

## Предварительные требования

- VPS с Ubuntu 20.04+
- Root или sudo доступ
- Токен бота от @BotFather

## Шаг 1: Установка Docker и Docker Compose

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Установка Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Проверка
docker --version
docker-compose --version
```

## Шаг 2: Загрузка проекта

```bash
# Установка Git (если не установлен)
sudo apt install -y git

# Клонирование репозитория
cd ~
git clone https://github.com/yourusername/telegram-audio-bot.git MyAudioBot
cd MyAudioBot
```

## Шаг 3: Настройка

```bash
# Создание .env файла
cp .env.example .env

# Редактирование (вставьте ваш токен)
nano .env
```

Замените `YOUR_BOT_TOKEN_HERE` на ваш токен от @BotFather

Сохраните: `Ctrl+O` → `Enter` → `Ctrl+X`

## Шаг 4: Запуск

```bash
# Сборка и запуск
docker-compose build
docker-compose up -d

# Проверка логов
docker-compose logs -f
```

Нажмите `Ctrl+C` для выхода из логов.

## ✅ Готово!

Ваш бот теперь работает 24/7!

## 📊 Полезные команды

```bash
# Просмотр статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f

# Перезапуск
docker-compose restart

# Остановка
docker-compose down

# Обновление
git pull && docker-compose build && docker-compose restart
```

## 🔄 Обновление бота

```bash
cd ~/MyAudioBot
git pull origin main
docker-compose down
docker-compose build
docker-compose up -d
```

## ⚠️ Проблемы?

### Бот не запускается

```bash
# Проверьте логи
docker-compose logs

# Проверьте токен
cat .env
```

### Недостаточно памяти

Убедитесь, что у VPS минимум 1GB RAM (рекомендуется 2GB)

## 📖 Подробная документация

- [Полная инструкция для VPS](DEPLOY_VPS.md)
- [Документация проекта](README.md)

---

**🎵 Наслаждайтесь!**

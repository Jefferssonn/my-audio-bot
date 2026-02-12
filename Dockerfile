FROM python:3.11-slim

# Метаданные
LABEL maintainer="Audio Bot Developer"
LABEL version="2.4"
LABEL description="Telegram Audio Bot PRO v2.4 - File Size Check"

# Установка переменных окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements.txt и установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода бота
COPY bot.py .

# Создание директорий для временных файлов и логов
RUN mkdir -p /app/temp /app/logs && \
    chmod 755 /app/temp /app/logs

# Healthcheck для проверки работы бота
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ps aux | grep -q '[p]ython.*bot.py' || exit 1

# Запуск бота
CMD ["python", "-u", "bot.py"]

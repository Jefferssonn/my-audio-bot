#!/bin/bash

echo "========================================"
echo "🚀 Запуск Telegram Audio Bot PRO v2.3"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

if [ ! -f ".env" ]; then
    echo "❌ Файл .env не найден!"
    echo ""
    echo "Создайте файл .env на основе .env.example:"
    echo "cp .env.example .env"
    echo ""
    echo "И укажите ваш BOT_TOKEN"
    exit 1
fi

echo "📦 Запуск Docker контейнера..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Ошибка запуска!"
    echo "Проверьте:"
    echo "1. Установлен ли Docker"
    echo "2. Запущен ли Docker"
    echo "3. Правильность docker-compose.yml"
    exit 1
fi

echo ""
echo "✅ Бот успешно запущен!"
echo ""
echo "📊 Просмотр логов: ./scripts/logs.sh"
echo "🛑 Остановка: ./scripts/stop.sh"
echo "🔄 Перезапуск: ./scripts/restart.sh"
echo ""

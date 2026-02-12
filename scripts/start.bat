@echo off
chcp 65001 >nul
echo ========================================
echo 🚀 Запуск Telegram Audio Bot PRO v2.6
echo ========================================
echo.

cd /d "%~dp0\.."

if not exist ".env" (
    echo ❌ Файл .env не найден!
    echo.
    echo Создайте файл .env на основе .env.example:
    echo copy .env.example .env
    echo.
    echo И укажите ваш BOT_TOKEN
    pause
    exit /b 1
)

echo 📦 Запуск Docker контейнера...
docker-compose up -d

if errorlevel 1 (
    echo.
    echo ❌ Ошибка запуска!
    echo Проверьте:
    echo 1. Установлен ли Docker
    echo 2. Запущен ли Docker Desktop
    echo 3. Правильность docker-compose.yml
    pause
    exit /b 1
)

echo.
echo ✅ Бот успешно запущен!
echo.
echo 📊 Просмотр логов: scripts\logs.bat
echo 🛑 Остановка: scripts\stop.bat
echo 🔄 Перезапуск: scripts\restart.bat
echo.
pause

@echo off
chcp 65001 >nul
echo ========================================
echo 🛑 Остановка Telegram Audio Bot
echo ========================================
echo.

cd /d "%~dp0\.."

echo 🛑 Остановка контейнера...
docker-compose down

if errorlevel 1 (
    echo.
    echo ❌ Ошибка остановки!
    pause
    exit /b 1
)

echo.
echo ✅ Бот остановлен!
echo.
pause

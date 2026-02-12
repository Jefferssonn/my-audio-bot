@echo off
chcp 65001 >nul
echo ========================================
echo 🔄 Перезапуск Telegram Audio Bot
echo ========================================
echo.

cd /d "%~dp0\.."

echo 🔄 Перезапуск контейнера...
docker-compose restart

if errorlevel 1 (
    echo.
    echo ❌ Ошибка перезапуска!
    pause
    exit /b 1
)

echo.
echo ✅ Бот перезапущен!
echo.
echo 📊 Просмотр логов: scripts\logs.bat
echo.
pause

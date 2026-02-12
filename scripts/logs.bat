@echo off
chcp 65001 >nul
echo ========================================
echo 📊 Логи Telegram Audio Bot
echo ========================================
echo.
echo Нажмите Ctrl+C для выхода
echo.

cd /d "%~dp0\.."

docker-compose logs -f --tail=100

pause

#!/bin/bash

echo "========================================"
echo "📊 Логи Telegram Audio Bot"
echo "========================================"
echo ""
echo "Нажмите Ctrl+C для выхода"
echo ""

cd "$(dirname "$0")/.."

docker-compose logs -f --tail=100

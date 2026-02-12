# Скрипты управления

## Windows

Просто запустите `.bat` файлы двойным кликом или из командной строки:

```cmd
scripts\start.bat
scripts\stop.bat
scripts\restart.bat
scripts\logs.bat
```

## Linux/Mac

Сначала сделайте скрипты исполняемыми:

```bash
chmod +x scripts/*.sh
```

Затем запускайте:

```bash
./scripts/start.sh
./scripts/stop.sh
./scripts/restart.sh
./scripts/logs.sh
```

## Альтернатива - Docker Compose напрямую

Вы также можете использовать команды Docker Compose напрямую:

```bash
# Запуск
docker-compose up -d

# Остановка
docker-compose down

# Перезапуск
docker-compose restart

# Логи
docker-compose logs -f --tail=100

# Статус
docker-compose ps
```

# 🚀 Развертывание на VPS

Полная инструкция по развертыванию Telegram Audio Bot PRO v2.3 на VPS сервере.

## 📋 Требования

- **VPS**: Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- **RAM**: Минимум 1GB, рекомендуется 2GB+
- **CPU**: 1+ vCPU
- **Диск**: 10GB+ свободного места
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: для клонирования репозитория

## 🔧 Подготовка VPS

### 1. Подключение к VPS

```bash
ssh root@your_vps_ip
# или
ssh user@your_vps_ip
```

### 2. Обновление системы

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# CentOS/RHEL
sudo yum update -y
```

### 3. Установка Docker

#### Ubuntu/Debian:

```bash
# Удаление старых версий
sudo apt remove docker docker-engine docker.io containerd runc

# Установка зависимостей
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Добавление GPG ключа Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Добавление репозитория Docker
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Установка Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Запуск Docker
sudo systemctl start docker
sudo systemctl enable docker

# Проверка установки
sudo docker --version
```

#### CentOS/RHEL:

```bash
# Установка зависимостей
sudo yum install -y yum-utils

# Добавление репозитория Docker
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Установка Docker
sudo yum install -y docker-ce docker-ce-cli containerd.io

# Запуск Docker
sudo systemctl start docker
sudo systemctl enable docker

# Проверка установки
sudo docker --version
```

### 4. Установка Docker Compose

```bash
# Скачивание последней версии
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Добавление прав на выполнение
sudo chmod +x /usr/local/bin/docker-compose

# Проверка установки
docker-compose --version
```

### 5. Установка Git (если не установлен)

```bash
# Ubuntu/Debian
sudo apt install -y git

# CentOS/RHEL
sudo yum install -y git
```

## 📦 Установка бота

### 1. Клонирование репозитория

```bash
# Переход в домашнюю директорию
cd ~

# Клонирование из Git
git clone https://github.com/yourusername/telegram-audio-bot.git MyAudioBot

# Переход в директорию проекта
cd MyAudioBot
```

Если используете приватный репозиторий:

```bash
# С использованием SSH ключа
git clone git@github.com:yourusername/telegram-audio-bot.git MyAudioBot

# С использованием токена
git clone https://YOUR_TOKEN@github.com/yourusername/telegram-audio-bot.git MyAudioBot
```

### 2. Создание .env файла

```bash
# Копирование шаблона
cp .env.example .env

# Редактирование .env
nano .env
```

Заполните файл:

```env
BOT_TOKEN=ваш_токен_от_BotFather
MAX_FILE_SIZE_MB=100
CLEANUP_INTERVAL_MINUTES=30
TEMP_FILE_MAX_AGE_HOURS=2
```

Сохраните файл: `Ctrl+O`, затем `Enter`, затем `Ctrl+X`

### 3. Сборка и запуск

```bash
# Сборка Docker образа
sudo docker-compose build

# Запуск бота в фоновом режиме
sudo docker-compose up -d

# Проверка статуса
sudo docker-compose ps

# Просмотр логов
sudo docker-compose logs -f
```

## 🎛️ Управление ботом

### Просмотр статуса

```bash
# Статус контейнеров
sudo docker-compose ps

# Статус health check
sudo docker inspect telegram-audio-bot-pro --format='{{.State.Health.Status}}'
```

### Просмотр логов

```bash
# Все логи
sudo docker-compose logs

# Последние 100 строк
sudo docker-compose logs --tail=100

# В реальном времени
sudo docker-compose logs -f

# Логи внутри контейнера
sudo docker exec telegram-audio-bot-pro cat /app/logs/bot.log
```

### Перезапуск

```bash
# Перезапуск бота
sudo docker-compose restart

# Полная пересборка и перезапуск
sudo docker-compose down
sudo docker-compose build --no-cache
sudo docker-compose up -d
```

### Остановка

```bash
# Остановка (сохраняет данные)
sudo docker-compose down

# Остановка с удалением volumes
sudo docker-compose down -v
```

## 🔄 Обновление бота

### Обновление через Git

```bash
# Остановка бота
sudo docker-compose down

# Обновление кода
git pull origin main

# Пересборка образа
sudo docker-compose build

# Запуск
sudo docker-compose up -d

# Проверка логов
sudo docker-compose logs -f
```

### Откат к предыдущей версии

```bash
# Просмотр коммитов
git log --oneline

# Откат к конкретному коммиту
git checkout <commit_hash>

# Пересборка и запуск
sudo docker-compose down
sudo docker-compose build
sudo docker-compose up -d
```

## 🔒 Безопасность

### Файрвол

```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 22/tcp    # SSH
sudo ufw enable

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --reload
```

### Ограничение доступа к Docker socket

```bash
# Создание группы docker (если не существует)
sudo groupadd docker

# Добавление пользователя в группу
sudo usermod -aG docker $USER

# Применение изменений (нужно перелогиниться)
newgrp docker
```

### Защита .env файла

```bash
# Ограничение прав доступа
chmod 600 .env

# Проверка
ls -la .env
```

## 📊 Мониторинг

### Использование ресурсов

```bash
# Статистика контейнера
sudo docker stats telegram-audio-bot-pro

# Использование диска
sudo docker system df

# Размер volumes
sudo docker volume ls
```

### Автоматический перезапуск

Уже настроен в `docker-compose.yml`:

```yaml
restart: unless-stopped
```

Бот будет автоматически перезапускаться при:
- Падении процесса
- Перезагрузке сервера
- Ошибках

### Системный сервис (опционально)

Создайте systemd сервис для автозапуска:

```bash
sudo nano /etc/systemd/system/audio-bot.service
```

```ini
[Unit]
Description=Telegram Audio Bot
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/root/MyAudioBot
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Активация:

```bash
sudo systemctl daemon-reload
sudo systemctl enable audio-bot
sudo systemctl start audio-bot
```

## 🧹 Обслуживание

### Очистка Docker

```bash
# Удаление неиспользуемых образов
sudo docker image prune -a

# Удаление неиспользуемых volumes
sudo docker volume prune

# Полная очистка
sudo docker system prune -a --volumes
```

### Бэкапы

#### Бэкап конфигурации:

```bash
# Создание бэкапа .env
cp .env .env.backup_$(date +%Y%m%d)

# Бэкап логов
sudo docker cp telegram-audio-bot-pro:/app/logs ./logs_backup_$(date +%Y%m%d)
```

#### Автоматический бэкап (cron):

```bash
# Открыть crontab
crontab -e

# Добавить строку (бэкап каждую неделю)
0 0 * * 0 cd /root/MyAudioBot && cp .env .env.backup_$(date +\%Y\%m\%d)
```

## ❓ Проблемы и решения

### Бот не запускается

```bash
# Проверка логов
sudo docker-compose logs

# Проверка токена
cat .env | grep BOT_TOKEN

# Проверка Docker
sudo systemctl status docker
```

### Недостаточно памяти

Увеличьте лимиты в `docker-compose.yml`:

```yaml
limits:
  memory: 4G  # было 2G
```

### Ошибки прав доступа

```bash
# Проверка владельца файлов
ls -la

# Изменение владельца
sudo chown -R $USER:$USER .
```

## 📞 Поддержка

- **Документация**: [README.md](README.md)
- **Быстрый старт**: [QUICKSTART.md](QUICKSTART.md)
- **Issues**: https://github.com/yourusername/telegram-audio-bot/issues

---

**🎵 Удачного развертывания!**

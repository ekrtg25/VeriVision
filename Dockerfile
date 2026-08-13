FROM python:3.10-slim

WORKDIR /app

# Отключаем создание pyc файлов и буферизацию вывода
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Устанавливаем системные зависимости, необходимые для OpenCV и других библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем и ставим зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Запускаем сервер, используя порт, который выдаст Cloud Run (или 8080 по умолчанию)
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
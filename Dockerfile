FROM python:3.12-slim

WORKDIR /app

# Отключаем создание pyc-файлов и буферизацию вывода логов
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Задаем дефолтный порт, если Cloud Run не передаст свой
ENV PORT=8080

# Устанавливаем системные зависимости для OpenCV и сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

# Устанавливаем согласованную связку PyTorch CPU
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем остальные пакеты
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения и веса
COPY . .

# Запуск через sh -c для корректной подстановки переменной $PORT от Cloud Run
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
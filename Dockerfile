# 1. Используем легкий официальный образ Python
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# 2. Устанавливаем системные зависимости, необходимые для OpenCV и работы с изображениями
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Задаем рабочую директорию внутри контейнера
WORKDIR /app

# 4. Сначала копируем только requirements.txt (для эффективного кеширования слоев Docker)
COPY requirements.txt .

# 5. Обновляем pip и устанавливаем зависимости без создания лишнего кеша
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Копируем исходный код проекта и обученные веса моделей
COPY . /app

# 7. Открываем стандартный порт Streamlit
EXPOSE 8501

# 8. Проверка здоровья контейнера (Healthcheck)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Команда для запуска приложения
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
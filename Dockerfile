FROM python:3.12-slim

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

# Копируем requirements.txt отдельно, чтобы слой кэшировался,
# пока зависимости не меняются (ускоряет повторные сборки)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

# Ставим torch отдельно как CPU-only сборку (без этого pip по умолчанию
# может подтянуть CUDA-версию весом 2-3+ ГБ, которая тут не нужна и
# сильно раздувает образ и время сборки)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Остальные зависимости из requirements.txt
# (убедись, что там НЕТ строки "torch" без индекса — иначе pip может
# переустановить его в GPU-версии поверх уже поставленной CPU-версии)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект (server.py, index.html, статика, веса моделей)
COPY . .

# Запускаем сервер на порту, который выдаст Cloud Run (или 8080 по умолчанию)
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
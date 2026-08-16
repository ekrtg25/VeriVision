import os
from pathlib import Path
from google.cloud import storage

BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "verivision-models")  # Укажи имя своего бакета
MODELS_DIR = Path("models")

WEIGHTS_FILES = [
    "calibrated_head.pth",
    "calibrators.pkl"
]

def download_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Если все файлы уже на месте, пропускаем
    if all((MODELS_DIR / f).exists() for f in WEIGHTS_FILES):
        print("[+] Все веса уже присутствуют локально.")
        return

    print(f"[+] Подключение к GCS бакету: {BUCKET_NAME}...")
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        for filename in WEIGHTS_FILES:
            target_path = MODELS_DIR / filename
            if not target_path.exists():
                print(f"[↓] Скачивание {filename}...")
                blob = bucket.blob(filename)
                blob.download_to_filename(str(target_path))
                print(f"[OK] {filename} успешно сохранен в models/")
    except Exception as e:
        print(f"[!] Ошибка скачивания весов из Cloud Storage: {e}")
        # Если веса критичны, райзим ошибку
        raise e

if __name__ == "__main__":
    download_models()
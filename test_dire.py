import os
import glob
from src.models.dire_module import DIREAnalyzer

def main():
    print("[sys] Инициализация DIRE анализатора...")
    dire = DIREAnalyzer()

    # Берем по 5 тестовых изображений
    real_paths = glob.glob("data/ood/midjourney_v6/real/*.*")[:5]
    fake_paths = glob.glob("data/ood/midjourney_v6/fake/*.*")[:5]

    if not real_paths or not fake_paths:
        print("[error] OOD данные не найдены! Проверь директорию data/ood/midjourney_v6/")
        return

    print("\n--- 🟢 РЕАЛЬНЫЕ ФОТОГРАФИИ (Unsplash) ---")
    real_errors = []
    for path in real_paths:
        err = dire.get_reconstruction_error(path)
        real_errors.append(err)
        print(f"File: {os.path.basename(path)} | MAE Error: {err:.5f}")

    print("\n--- 🔴 ДИПФЕЙКИ (Midjourney V6) ---")
    fake_errors = []
    for path in fake_paths:
        err = dire.get_reconstruction_error(path)
        fake_errors.append(err)
        print(f"File: {os.path.basename(path)} | MAE Error: {err:.5f}")

    avg_real = sum(real_errors) / len(real_errors)
    avg_fake = sum(fake_errors) / len(fake_errors)

    print("\n==========================================")
    print(f"📊 Средняя ошибка REAL: {avg_real:.5f}")
    print(f"📊 Средняя ошибка FAKE: {avg_fake:.5f}")
    print("==========================================")

if __name__ == "__main__":
    main()
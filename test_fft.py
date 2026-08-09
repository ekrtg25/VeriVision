import glob
import os
from src.models.fft_module import FFTAnalyzer

def main():
    print("[sys] Инициализация FFT анализатора...")
    fft = FFTAnalyzer(low_freq_radius=80) # Радиус среза можно тюнить

    real_paths = glob.glob("data/ood/midjourney_v6/real/*.*")[:15]
    fake_paths = glob.glob("data/ood/midjourney_v6/fake/*.*")[:15]

    print("\n--- 🟢 РЕАЛЬНЫЕ ФОТОГРАФИИ ---")
    real_scores = []
    for path in real_paths:
        score = fft.get_high_freq_ratio(path)
        real_scores.append(score)
        print(f"{os.path.basename(path)} | High-Freq Energy: {score:.2f}%")

    print("\n--- 🔴 ДИПФЕЙКИ (Midjourney V6) ---")
    fake_scores = []
    for path in fake_paths:
        score = fft.get_high_freq_ratio(path)
        fake_scores.append(score)
        print(f"{os.path.basename(path)} | High-Freq Energy: {score:.2f}%")

    avg_real = sum(real_scores) / len(real_scores) if real_scores else 0
    avg_fake = sum(fake_scores) / len(fake_scores) if fake_scores else 0

    print("\n==========================================")
    print(f"📊 Средняя ВЧ-энергия REAL: {avg_real:.2f}%")
    print(f"📊 Средняя ВЧ-энергия FAKE: {avg_fake:.2f}%")
    print("==========================================")

if __name__ == "__main__":
    main()
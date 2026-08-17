"""
Скрипт для скачивания датасета с Hugging Face целиком в локальную папку
и последующего извлечения форензик-метрик (ELA, PRNU, FFT) с общим прогресс-баром.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.forensics import ForensicsExtractor
from src.models.fft_module import FFTSpectralExtractor


def main():
    parser = argparse.ArgumentParser(description="Download dataset & extract calibration scores")
    parser.add_argument("--repo_id", type=str, default="eddyfox8812/ai-vs-real-2k-images", help="HF Dataset Repo ID")
    parser.add_argument("--download_dir", type=str, default="data/calibration_raw", help="Локальная папка для датасета")
    parser.add_argument("--out", type=str, default="data/calibration_scores.csv", help="Путь к результирующему CSV")
    args = parser.parse_args()

    local_dir = Path(args.download_dir)
    print(f"[*] [1/2] Скачивание датасета {args.repo_id} в папку: {local_dir.resolve()}...")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        max_workers=4,
    )
    print("[+] Датасет готов к обработке!")

    print("\n[*] [2/2] Извлечение метрик ELA, PRNU, FFT...")
    forensics = ForensicsExtractor()
    fft = FFTSpectralExtractor()

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [p for p in local_dir.rglob("*") if p.suffix.lower() in valid_exts]

    if not image_files:
        print(f"[!] В папке {local_dir} не найдено изображений.")
        sys.exit(1)

    print(f"[*] Найдено изображений для анализа: {len(image_files)}")

    results = []
    for img_path in tqdm(image_files, desc="Анализ форензики"):
        try:
            parent_folder = img_path.parent.name.lower()
            
            if "real" in parent_folder or "photo" in parent_folder or "authentic" in parent_folder:
                label = 0
            elif any(k in parent_folder for k in ["fake", "ai", "generated", "synth"]):
                label = 1
            else:
                grandparent_folder = img_path.parent.parent.name.lower()
                if "real" in grandparent_folder:
                    label = 0
                elif any(k in grandparent_folder for k in ["fake", "ai", "generated", "synth"]):
                    label = 1
                else:
                    continue 

            img = Image.open(img_path).convert("RGB")
            img_np = np.asarray(img)

            ela_raw = forensics.compute_ela_score(img)
            prnu_raw = forensics.compute_prnu_residual(img_np)
            fft_raw = fft.extract_spectral_features(img_np)

            results.append({
                "filename": img_path.name,
                "ela_raw": float(ela_raw),
                "prnu_raw": float(prnu_raw),
                "fft_raw": float(fft_raw),
                "label": label
            })
        except Exception as e:
            continue

    df = pd.DataFrame(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    num_real = int((df["label"] == 0).sum())
    num_fake = int((df["label"] == 1).sum())

    print(f"\n[+] Готово! Обработано {len(df)} изображений.")
    print(f"[*] Баланс классов: Real = {num_real}, Fake/AI = {num_fake}")
    print(f"[+] CSV сохранен в: {out_path}")
    print(f"[+] Теперь запустите: python scripts/fit_calibrators.py --csv {out_path} --out models/calibrators.pkl")


if __name__ == "__main__":
    main()
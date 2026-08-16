"""
Extracts raw forensic scores (ELA, PRNU, FFT) from a dataset of real and AI images,
saving them to a CSV file for calibration via fit_calibrators.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.forensics import ForensicsExtractor
from src.models.fft_module import FFTSpectralExtractor


def process_folder(folder_path: Path, label: int, forensics: ForensicsExtractor, fft: FFTSpectralExtractor):
    results = []
    
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(folder_path.rglob(ext))
        
    for img_path in tqdm(image_paths, desc=f"Processing {folder_path.name} (Label {label})"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.asarray(img)
            
            ela_raw = forensics.compute_ela_score(img)
            prnu_raw = forensics.compute_prnu_residual(img_np)
            fft_raw = fft.extract_spectral_features(img_np)
            
            results.append({
                "filename": img_path.name,
                "ela_raw": ela_raw,
                "prnu_raw": prnu_raw,
                "fft_raw": fft_raw,
                "label": label
            })
        except Exception as e:
            print(f"[!] Error processing {img_path}: {e}")
            
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract raw scores for Platt Calibration")
    parser.add_argument("--real", required=True, help="Path to real photos directory")
    parser.add_argument("--fake", required=True, help="Path to AI-generated photos directory")
    parser.add_argument("--out", default="data/calibration_scores.csv", help="Output CSV path")
    args = parser.parse_args()

    forensics = ForensicsExtractor()
    fft = FFTSpectralExtractor()

    print("[*] Extracting features from REAL images...")
    real_data = process_folder(Path(args.real), label=0, forensics=forensics, fft=fft)

    print("\n[*] Extracting features from FAKE images...")
    fake_data = process_folder(Path(args.fake), label=1, forensics=forensics, fft=fft)

    df = pd.DataFrame(real_data + fake_data)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print(f"\n[+] Successfully saved {len(df)} records to {out_path}")


if __name__ == "__main__":
    main()
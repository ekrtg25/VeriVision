"""
VeriVision Calibration Pipeline
Trains Isotonic Regression models mapping raw forensic scores to calibrated P(AI).
"""

import os
import glob
import numpy as np
import joblib
from PIL import Image
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
from src.serving.ensemble import VeriVisionEnsemble

def extract_dataset_features(real_dir: str, fake_dir: str):
    ensemble = VeriVisionEnsemble(models_dir="models", skip_calibrator_load=True)
    real_paths = glob.glob(os.path.join(real_dir, "*.*"))
    fake_paths = glob.glob(os.path.join(fake_dir, "*.*"))
    
    X_raw = {"nn_mean": [], "ela": [], "prnu": [], "fft": [], "srm": []}
    y = []

    print("[+] Extracting features for Real images...")
    for path in tqdm(real_paths):
        try:
            feats = ensemble.extract_features_vector(Image.open(path))
            for k in X_raw.keys():
                X_raw[k].append(feats.get(k, feats.get(f"{k}_score", feats.get(f"{k}_norm", 0))))
            y.append(0)
        except Exception:
            continue

    print("[+] Extracting features for AI images...")
    for path in tqdm(fake_paths):
        try:
            feats = ensemble.extract_features_vector(Image.open(path))
            for k in X_raw.keys():
                X_raw[k].append(feats.get(k, feats.get(f"{k}_score", feats.get(f"{k}_norm", 0))))
            y.append(1)
        except Exception:
            continue

    return X_raw, np.array(y)

def train_calibrators():
    real_dir = "data/robust_v1/real"
    fake_dir = "data/robust_v1/fake"
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"[!] Dataset not found at data/robust_v1/")
        return

    X_raw, y = extract_dataset_features(real_dir, fake_dir)
    calibrators = {}

    print("\n[+] Training Isotonic Calibrators...")
    for feature in X_raw.keys():
        ir = IsotonicRegression(out_of_bounds='clip')
        X_feat = np.array(X_raw[feature])
        
        # Инвертируем PRNU, так как высокий PRNU означает Real (0)
        if feature == "prnu":
            X_feat = -X_feat
            
        ir.fit(X_feat, y)
        calibrators[feature] = ir
        print(f"  • {feature} calibrated.")

    os.makedirs("models", exist_ok=True)
    joblib.dump(calibrators, "models/calibrators.pkl")
    print("[✓] Calibrators saved to models/calibrators.pkl")

if __name__ == "__main__":
    train_calibrators()
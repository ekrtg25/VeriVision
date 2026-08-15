"""
Training Script for Meta-Classifier v3.2
Features (7): [nn_mean, nn_std, nn_max, ela_score, prnu_norm, fft_norm, expert_disagreement]
"""

import os
import glob
import random
import numpy as np
import joblib
from tqdm import tqdm
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

from src.serving.ensemble import HybridEnsembleDetector


def extract_features(detector, real_paths, fake_paths):
    X, y = [], []
    
    print(f"[sys] Извлечение 7D-вектора признаков (MoE v3.2 с DQT и Disagreement)...")
    for path in tqdm(real_paths, desc="Real Images"):
        try:
            pil_img = Image.open(path).convert("RGB")
            m = detector.extract_features_vector(pil_img)
            X.append([
                m['nn_mean'], m['nn_std'], m['nn_max'],
                m['ela_score'], m['prnu_norm'], m['fft_norm'],
                m['expert_disagreement']
            ])
            y.append(0)
        except Exception:
            continue
        
    for path in tqdm(fake_paths, desc="Fake Images"):
        try:
            pil_img = Image.open(path).convert("RGB")
            m = detector.extract_features_vector(pil_img)
            X.append([
                m['nn_mean'], m['nn_std'], m['nn_max'],
                m['ela_score'], m['prnu_norm'], m['fft_norm'],
                m['expert_disagreement']
            ])
            y.append(1)
        except Exception:
            continue
        
    return np.array(X), np.array(y)


def main():
    print("[sys] Инициализация VeriVision Ensemble v3.2...")
    detector = HybridEnsembleDetector()
    
    real_paths = glob.glob("data/robust_v1/real/*.*")
    fake_paths = glob.glob("data/robust_v1/fake/*.*")
    
    if not real_paths or not fake_paths:
        print(f"[!] Картинки не найдены в data/defactify/val/")
        return

    random.seed(42)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    
    n_samples = min(len(real_paths), len(fake_paths), 1500)
    train_real, test_real = real_paths[:int(n_samples*0.7)], real_paths[int(n_samples*0.7):n_samples]
    train_fake, test_fake = fake_paths[:int(n_samples*0.7)], fake_paths[int(n_samples*0.7):n_samples]

    print(f"\n--- Шаг 1: Извлечение признаков (Train: {len(train_real)*2}, Test: {len(test_real)*2}) ---")
    X_train, y_train = extract_features(detector, train_real, train_fake)
    X_test, y_test = extract_features(detector, test_real, test_fake)
    
    print("\n--- Шаг 2: Обучение Random Forest (7 Features, Regularized) ---")
    meta_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=4,
        min_samples_leaf=8,
        class_weight='balanced',
        random_state=42
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(meta_model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"[CV ROC-AUC]: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)
    y_prob = meta_model.predict_proba(X_test)[:, 1]
    
    print("\n==========================================")
    print("🚀 ОТЧЕТ МЕТА-АНСАМБЛЯ v3.2")
    print("==========================================")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    
    importances = meta_model.feature_importances_
    names = [
        "NN_Mean", "NN_Crop_Variance", "NN_Max",
        "ELA_Gated", "PRNU_Gated", "FFT_Spectrum",
        "Expert_Disagreement"
    ]
    print("\n[sys] Feature Importances:")
    for n, imp in zip(names, importances):
        print(f" • {n:22s}: {imp * 100:.2f}%")
        
    os.makedirs("models", exist_ok=True)
    joblib.dump(meta_model, "models/meta_classifier.pkl")
    print("\n[✓] Модель v3.2 сохранена в models/meta_classifier.pkl!")


if __name__ == "__main__":
    main()
import glob
import numpy as np
import os
import joblib
from tqdm import tqdm
from src.models.fft_module import SpectralAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

def load_dataset_from_dir(base_dir, analyzer):
    real_paths = glob.glob(os.path.join(base_dir, "real/*.*"))
    fake_paths = glob.glob(os.path.join(base_dir, "fake/*.*"))
    
    X, y = [], []
    print(f"[sys] Извлечение признаков из '{base_dir}' (Real: {len(real_paths)}, Fake: {len(fake_paths)})...")

    for path in tqdm(real_paths, desc="Обработка Real"):
        features = analyzer.get_1d_power_spectrum(path)
        if features is not None:
            X.append(features)
            y.append(0)

    for path in tqdm(fake_paths, desc="Обработка Fake"):
        features = analyzer.get_1d_power_spectrum(path)
        if features is not None:
            X.append(features)
            y.append(1)

    return np.array(X), np.array(y)

def main():
    print("[sys] Инициализация Spectral Analyzer (Defactify Dataset)...")
    analyzer = SpectralAnalyzer()
    
    train_dir = "data/defactify/train"
    val_dir = "data/defactify/val"

    X_train, y_train = load_dataset_from_dir(train_dir, analyzer)
    X_val, y_val = load_dataset_from_dir(val_dir, analyzer)

    print("\n[sys] Обучение и Platt-калибровка на Train сплите (cv=5)...")
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    
    # Внутренняя кросс-валидация защищает от переобучения
    calibrated_rf = CalibratedClassifierCV(estimator=base_rf, method='sigmoid', cv=5)
    calibrated_rf.fit(X_train, y_train)

    print("\n==========================================")
    print("🚀 ОТЧЕТ ОБУЧЕНИЯ СПЕКТРАЛЬНОГО ЭКСПЕРТА (Val Set)")
    print("==========================================")
    y_pred = calibrated_rf.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=["Real", "Fake"]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(calibrated_rf, "models/rf_spectral.pkl")
    print("\n[sys] Спектральная модель сохранена.")

if __name__ == "__main__":
    main()
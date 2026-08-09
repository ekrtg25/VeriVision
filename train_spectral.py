import glob
import numpy as np
from src.models.fft_module import SpectralAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
import os
import joblib

def main():
    print("[sys] Инициализация Spectral Analyzer...")
    analyzer = SpectralAnalyzer()

    # Берем ВСЕ доступные картинки MJv6 и реальные OOD
    real_paths = glob.glob("data/ood/midjourney_v6/real/*.*")
    fake_paths = glob.glob("data/ood/midjourney_v6/fake/*.*")

    X = []
    y = []

    print(f"[sys] Извлечение 1D спектров (Real: {len(real_paths)}, Fake: {len(fake_paths)})...")
    
    for path in real_paths:
        features = analyzer.get_1d_power_spectrum(path)
        if features is not None:
            X.append(features)
            y.append(0) # 0 = Real

    for path in fake_paths:
        features = analyzer.get_1d_power_spectrum(path)
        if features is not None:
            X.append(features)
            y.append(1) # 1 = Fake

    X = np.array(X)
    y = np.array(y)

    print("\n[sys] Обучение Random Forest (с Platt Калибровкой) на спектральных профилях...")
    # Обучаем быстрый и мощный случайный лес
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    # Оборачиваем лес в калибратор (метод 'sigmoid' - это Platt Scaling)
    rf = CalibratedClassifierCV(estimator=base_rf, method='sigmoid', cv=5)
    
    # Кросс-валидация (5 фолдов)
    scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print(f"\n📊 Средняя точность кросс-валидации (5-Fold): {np.mean(scores):.2%}")

    # Обучаем на сплите для детального отчета
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print("\n==========================================")
    print("🚀 ОТЧЕТ ОБНАРУЖЕНИЯ АНОМАЛИЙ (Test Set)")
    print("==========================================")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake (MJv6)"]))
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/rf_spectral.pkl"
    joblib.dump(rf, model_path)
    print(f"\n[sys] Модель Random Forest успешно сохранена в: {model_path}")

if __name__ == "__main__":
    main()
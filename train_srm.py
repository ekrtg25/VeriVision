import glob
import numpy as np
import os
import joblib
from src.models.srm_module import SRMAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

def main():
    print("[sys] Инициализация SRM Noise Analyzer...")
    analyzer = SRMAnalyzer()

    real_paths = glob.glob("data/ood/midjourney_v6/real/*.*")
    fake_paths = glob.glob("data/ood/midjourney_v6/fake/*.*")

    X = []
    y = []

    print(f"[sys] Извлечение шумовых признаков (Real: {len(real_paths)}, Fake: {len(fake_paths)})...")
    
    for path in real_paths:
        features = analyzer.get_noise_features(path)
        if features is not None:
            X.append(features)
            y.append(0)

    for path in fake_paths:
        features = analyzer.get_noise_features(path)
        if features is not None:
            X.append(features)
            y.append(1)

    X = np.array(X)
    y = np.array(y)

    print("\n[sys] Обучение Random Forest на паттернах матричного шума...")
    # Наш второй эксперт (Лес)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print(f"\n📊 Средняя точность кросс-валидации (5-Fold): {np.mean(scores):.2%}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print("\n==========================================")
    print("🚀 ОТЧЕТ ОБНАРУЖЕНИЯ АНОМАЛИЙ ШУМА (Test Set)")
    print("==========================================")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake (MJv6)"]))

    # Сохраняем второго эксперта
    os.makedirs("models", exist_ok=True)
    model_path = "models/rf_srm.pkl"
    joblib.dump(rf, model_path)
    print(f"\n[sys] Модель SRM сохранена в: {model_path}")

if __name__ == "__main__":
    main()
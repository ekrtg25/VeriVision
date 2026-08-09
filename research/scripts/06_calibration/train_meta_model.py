import os
import glob
import random
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from src.serving.ensemble import HybridEnsembleDetector

def extract_features(detector, real_paths, fake_paths):
    X, y = [], []
    
    print(f"[sys] Извлечение вероятностей экспертов...")
    for path in tqdm(real_paths, desc="Real Images"):
        pred = detector.predict(path)
        X.append([pred['cnn_prob'], pred['fft_prob'], pred['srm_prob']])
        y.append(0)
        
    for path in tqdm(fake_paths, desc="Fake Images"):
        pred = detector.predict(path)
        X.append([pred['cnn_prob'], pred['fft_prob'], pred['srm_prob']])
        y.append(1)
        
    return np.array(X), np.array(y)

def main():
    print("[sys] Инициализация экспертов...")
    detector = HybridEnsembleDetector()
    
    real_paths = glob.glob("data/defactify/val/real/*.*")
    fake_paths = glob.glob("data/defactify/val/fake/*.*")
    
    # Перемешиваем и берем сплит для обучения мета-модели (2000 картинок)
    random.seed(42)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    
    train_real = real_paths[:1000]
    train_fake = fake_paths[:1000]
    
    test_real = real_paths[1000:2000]
    test_fake = fake_paths[1000:2000]

    print("\n--- Шаг 1: Сбор данных для Мета-Модели ---")
    X_train, y_train = extract_features(detector, train_real, train_fake)
    X_test, y_test = extract_features(detector, test_real, test_fake)
    
    print("\n--- Шаг 2: Обучение Мета-Классификатора (Stacking) ---")
    # Логистическая регрессия найдет идеальный баланс доверия к моделям
    meta_model = LogisticRegression(class_weight='balanced')
    meta_model.fit(X_train, y_train)
    
    y_pred = meta_model.predict(X_test)
    
    print("\n==========================================")
    print("🚀 ОТЧЕТ МЕТА-АНСАМБЛЯ (Стэкинг)")
    print("==========================================")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    
    # Выводим веса, которые модель назначила экспертам
    weights = meta_model.coef_[0]
    print(f"\n[sys] Доверие мета-модели (Веса):")
    print(f" - ConvNeXt: {weights[0]:.4f}")
    print(f" - FFT:      {weights[1]:.4f}")
    print(f" - SRM:      {weights[2]:.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(meta_model, "models/meta_classifier.pkl")
    print("\n[sys] Мета-модель сохранена!")

if __name__ == "__main__":
    main()
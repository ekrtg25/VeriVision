import os
import glob
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve
from tqdm import tqdm

from src.serving.ensemble import HybridEnsembleDetector

def evaluate_dataset(engine, dataset_name, base_path, threshold=0.49):
    real_files = glob.glob(os.path.join(base_path, "real", "*.*"))
    fake_files = glob.glob(os.path.join(base_path, "fake", "*.*"))
    
    # Фильтрация по расширениям
    valid_exts = ('jpg', 'jpeg', 'png')
    real_files = [f for f in real_files if f.lower().endswith(valid_exts)]
    fake_files = [f for f in fake_files if f.lower().endswith(valid_exts)]
    
    image_paths = real_files + fake_files
    true_labels = [0] * len(real_files) + [1] * len(fake_files)
    
    if not image_paths:
        return None

    print(f"\n[sys] Evaluating {dataset_name} ({len(real_files)} Real, {len(fake_files)} Fake)...")
    
    ensemble_probs = []
    
    for path in tqdm(image_paths, desc=dataset_name):
        # Вызываем предикт без mode="uncertainty"
        result = engine.predict(path, threshold=threshold)
        ensemble_probs.append(result["ensemble_prob"])
        
    ensemble_probs = np.array(ensemble_probs)
    true_labels = np.array(true_labels)
    preds = (ensemble_probs >= threshold).astype(int)
    
    # Расчет EER (Equal Error Rate)
    fpr, tpr, _ = roc_curve(true_labels, ensemble_probs)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    metrics = {
        "Accuracy": accuracy_score(true_labels, preds),
        "F1-Score": f1_score(true_labels, preds, zero_division=0),
        "Precision": precision_score(true_labels, preds, zero_division=0),
        "Recall": recall_score(true_labels, preds, zero_division=0),
        "EER": eer
    }
    return metrics

def main():
    # Оставляем порог 0.49
    THRESHOLD = 0.49
    print("[sys] Initializing Hybrid Ensemble Detector...")
    
    # Инициализируем наш новый гибридный ансамбль
    engine = HybridEnsembleDetector()
    
    datasets = {
        "Midjourney V6": "data/ood/midjourney_v6"
    }
    
    results = {}
    
    for name, path in datasets.items():
        if os.path.exists(path):
            metrics = evaluate_dataset(engine, name, path, threshold=THRESHOLD)
            if metrics:
                results[name] = metrics
        else:
            print(f"[warn] Directory not found: {path}")

    print("\n" + "="*50)
    print("🚀 OUT-OF-DISTRIBUTION (OOD) BENCHMARK RESULTS")
    print("="*50)
    
    if not results:
        print("[error] No OOD data evaluated. Please populate data/ood/ directories.")
        return

    for name, m in results.items():
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {m['Accuracy']:.4f}")
        print(f"F1-Score:  {m['F1-Score']:.4f}")
        print(f"Precision: {m['Precision']:.4f}")
        print(f"Recall:    {m['Recall']:.4f}")
        print(f"EER:       {m['EER']:.4f}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
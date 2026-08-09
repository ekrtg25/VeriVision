import torch
import glob
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from src.serving.ensemble import HybridEnsembleDetector

def main():
    print("[sys] Initializing Calibrated Ensemble Detector...")
    engine = HybridEnsembleDetector()
    
    # Пути к валидационной выборке
    real_files = glob.glob("data/raw/val/real/*.*")
    fake_files = glob.glob("data/raw/val/fake/*.*")
    
    image_paths = real_files + fake_files
    true_labels = [0] * len(real_files) + [1] * len(fake_files)
    
    if not image_paths:
        print("[error] No validation images found!")
        return

    print(f"[sys] Running inference on {len(image_paths)} validation images...")
    
    ensemble_probs = []
    
    for path in tqdm(image_paths):
        # Используем дефолтный threshold 0.5, нас интересует только ensemble_prob
        result = engine.predict(path, mode="uncertainty")
        ensemble_probs.append(result["ensemble_prob"])
        
    ensemble_probs = np.array(ensemble_probs)
    true_labels = np.array(true_labels)
    
    print("\n[sys] Searching for Optimal Threshold (Maximizing F1-Score)...")
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = {}

    for t in thresholds:
        preds = (ensemble_probs >= t).astype(int)
        f1 = f1_score(true_labels, preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                "Precision": precision_score(true_labels, preds, zero_division=0),
                "Recall": recall_score(true_labels, preds, zero_division=0)
            }

    print("-" * 40)
    print(f"✅ BEST THRESHOLD: {best_thresh:.2f}")
    print(f"F1-Score:  {best_f1:.4f}")
    print(f"Precision: {best_metrics['Precision']:.4f} (Точность детекции фейков)")
    print(f"Recall:    {best_metrics['Recall']:.4f} (Полнота детекции фейков)")
    print("-" * 40)
    print("Now update 'threshold' in app.py with this new BEST THRESHOLD value!")

if __name__ == "__main__":
    main()
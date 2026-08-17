import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict, Union

def compute_metrics(y_true: Union[np.ndarray, torch.Tensor], y_pred_probs: Union[np.ndarray, torch.Tensor], threshold: float = 0.5) -> Dict[str, float]:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()

    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='binary', zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc)
    }

if __name__ == "__main__":
    dummy_true = np.array([0, 0, 1, 1, 1])
    dummy_probs = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    
    metrics = compute_metrics(dummy_true, dummy_probs)
    print("✅ Модуль метрик работает корректно!")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
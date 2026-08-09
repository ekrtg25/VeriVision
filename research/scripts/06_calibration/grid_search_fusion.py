import os
import glob
import numpy as np
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.serving.ensemble import HybridEnsembleDetector

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Вычисляет ECE (Expected Calibration Error).
    Чем ближе значение к 0, тем честнее модель выдает проценты уверенности.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Определяем, какие предсказания попали в текущий бин
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        if bin_lower == 0.0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def to_logit(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def fuse_probabilities(probs_list, k):
    """Применяет Confidence-Weighted Logit Fusion Клода с заданным k"""
    fused_probs = []
    for p_cnn, p_fft, p_srm in probs_list:
        l_cnn = to_logit(p_cnn)
        l_fft = to_logit(p_fft)
        l_srm = to_logit(p_srm)
        
        w_cnn = abs(l_cnn) ** k
        w_fft = abs(l_fft) ** k
        w_srm = abs(l_srm) ** k
        
        sum_w = w_cnn + w_fft + w_srm
        
        if sum_w < 1e-6:
            fused_logit = 0.0
        else:
            fused_logit = (w_cnn * l_cnn + w_fft * l_fft + w_srm * l_srm) / sum_w
            
        fused_prob = 1.0 / (1.0 + math.exp(-fused_logit))
        fused_probs.append(fused_prob)
        
    return np.array(fused_probs)

def main():
    print("[sys] Инициализация ансамбля для извлечения вероятностей...")
    # Инициализируем ансамбль (он подтянет новые обученные веса)
    detector = HybridEnsembleDetector()
    
    real_paths = glob.glob("data/ood/midjourney_v6/real/*.*")
    fake_paths = glob.glob("data/ood/midjourney_v6/fake/*.*")
    
    if not real_paths or not fake_paths:
        print("[error] OOD датасет не найден. Проверьте пути.")
        return

    # Кэш для сырых предсказаний
    results = []
    y_true = []
    
    print(f"[sys] Прогон OOD датасета (Real: {len(real_paths)}, Fake: {len(fake_paths)})...")
    
    for path in tqdm(real_paths, desc="Real Images"):
        pred = detector.predict(path)
        # Берем только сырые выходы экспертов
        results.append((pred['cnn_prob'], pred['fft_prob'], pred['srm_prob']))
        y_true.append(0)
        
    for path in tqdm(fake_paths, desc="Fake Images"):
        pred = detector.predict(path)
        results.append((pred['cnn_prob'], pred['fft_prob'], pred['srm_prob']))
        y_true.append(1)
        
    y_true = np.array(y_true)
    
    print("\n==========================================")
    print("🔍 GRID SEARCH ДЛЯ ГИПЕРПАРАМЕТРА K")
    print("==========================================")
    print(" K    | Accuracy | ECE    ")
    print("------------------------------------------")
    
    for k_val in np.arange(0.5, 3.1, 0.1):
        k_val = round(k_val, 1)
        y_prob = fuse_probabilities(results, k_val)
        y_pred_binary = (y_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred_binary)
        ece = expected_calibration_error(y_true, y_prob)
        
        print(f"{k_val:<4} | {acc:.4f}   | {ece:.4f} ")
        
    print("------------------------------------------")
    print("[sys] Проанализируй таблицу.")
    print("[sys] Идеальный 'k' — тот, где Accuracy максимален, а ECE минимален (ближе к 0).")

if __name__ == "__main__":
    main()
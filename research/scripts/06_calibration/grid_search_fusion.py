import os
import glob
import random
import numpy as np
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.serving.ensemble import HybridEnsembleDetector

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Вычисляет ECE (Expected Calibration Error).
    Чем ближе к 0, тем лучше откалибрована модель.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
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

def fuse_probabilities(probs_list, k, T=2.0, max_vote=0.6):
    """
    Применяет Confidence-Weighted Logit Fusion с Temperature Scaling и кэпированием.
    """
    fused_probs = []
    for p_cnn, p_fft, p_srm in probs_list:
        # 1. Temperature Scaling: Сглаживаем логиты, чтобы наказать излишнюю самоуверенность
        l_cnn = to_logit(p_cnn) / T
        l_fft = to_logit(p_fft) / T
        l_srm = to_logit(p_srm) / T
        
        # 2. Считаем сырые веса по сглаженным логитам
        w_cnn = abs(l_cnn) ** k
        w_fft = abs(l_fft) ** k
        w_srm = abs(l_srm) ** k
        
        weights = np.array([w_cnn, w_fft, w_srm])
        logits = np.array([l_cnn, l_fft, l_srm])
        sum_w = np.sum(weights)
        
        if sum_w < 1e-6:
            fused_logit = 0.0
        else:
            # 3. Переводим веса в доли (от 0.0 до 1.0)
            norm_weights = weights / sum_w
            
            # 4. Confidence Cap (Заглушка): никто не получает монополию
            if np.any(norm_weights > max_vote):
                # Находим эксперта-монополиста
                leader_idx = np.argmax(norm_weights)
                excess = norm_weights[leader_idx] - max_vote
                
                # Обрезаем его власть до лимита
                norm_weights[leader_idx] = max_vote
                
                # Демократично распределяем излишек голоса между остальными
                others_mask = np.arange(3) != leader_idx
                sum_others = np.sum(norm_weights[others_mask])
                
                if sum_others > 0:
                    norm_weights[others_mask] += excess * (norm_weights[others_mask] / sum_others)
                else:
                    norm_weights[others_mask] += excess / 2.0
            
            # 5. Считаем финальный взвешенный логит
            fused_logit = np.sum(norm_weights * logits)
            
        # 6. Возвращаем вероятность
        fused_prob = 1.0 / (1.0 + math.exp(-fused_logit))
        fused_probs.append(fused_prob)
        
    return np.array(fused_probs)

def main():
    print("[sys] Инициализация ансамбля для извлечения вероятностей...")
    detector = HybridEnsembleDetector()
    
    # Меняем пути на наш идеальный валидационный датасет Defactify
    real_paths = glob.glob("data/defactify/val/real/*.*")
    fake_paths = glob.glob("data/defactify/val/fake/*.*")
    
    if not real_paths or not fake_paths:
        print("[error] Валидационный датасет Defactify не найден. Проверьте пути.")
        return

    # Перемешиваем файлы, чтобы выборка была репрезентативной
    random.seed(42)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    
    # Берем по 1000 изображений каждого класса для скорости и баланса
    sample_size = 1000
    real_paths = real_paths[:sample_size]
    fake_paths = fake_paths[:sample_size]

    results = []
    y_true = []
    
    print(f"[sys] Прогон случайной подвыборки Defactify (Real: {len(real_paths)}, Fake: {len(fake_paths)})...")
    
    for path in tqdm(real_paths, desc="Real Images"):
        pred = detector.predict(path)
        results.append((pred['cnn_prob'], pred['fft_prob'], pred['srm_prob']))
        y_true.append(0)
        
    for path in tqdm(fake_paths, desc="Fake Images"):
        pred = detector.predict(path)
        results.append((pred['cnn_prob'], pred['fft_prob'], pred['srm_prob']))
        y_true.append(1)
        
    y_true = np.array(y_true)
    
    print("\n==========================================")
    print("🔍 GRID SEARCH ДЛЯ ГИПЕРПАРАМЕТРА K (Defactify Val)")
    print("==========================================")
    print(" K    | Accuracy | ECE    ")
    print("------------------------------------------")
    
    # Фиксируем T и max_vote для поиска
    T_val = 2.0
    max_vote_val = 0.60
    
    for k_val in np.arange(0.5, 3.1, 0.1):
        k_val = round(k_val, 1)
        y_prob = fuse_probabilities(results, k_val, T=T_val, max_vote=max_vote_val)
        y_pred_binary = (y_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred_binary)
        ece = expected_calibration_error(y_true, y_prob)
        
        print(f"{k_val:<4} | {acc:.4f}   | {ece:.4f} ")
        
    print("------------------------------------------")
    print("[sys] Анализ завершен.")
    print(f"[sys] Параметры: Temperature (T) = {T_val}, Confidence Cap (max_vote) = {max_vote_val}")
    print("[sys] Выбирай 'k', при котором Accuracy на пике, а ECE минимален (ближе к 0).")

if __name__ == "__main__":
    main()
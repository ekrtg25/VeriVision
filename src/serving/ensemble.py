import torch
import os
import numpy as np
import math
import joblib
from PIL import Image
from torchvision import transforms

from src.models.baseline_cnn import BaselineDetector
from src.models.fft_module import SpectralAnalyzer
from src.models.srm_module import SRMAnalyzer

# Функция перевода в логиты (как просил Клод)
def to_logit(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

class HybridEnsembleDetector:
    def __init__(self, cnn_weights="models/baseline_weights.pth", 
                 fft_weights="models/rf_spectral.pkl", 
                 srm_weights="models/rf_srm.pkl", 
                 device=None):
        
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        self.cnn = BaselineDetector(pretrained=False).to(self.device)
        if os.path.exists(cnn_weights):
            state_dict = torch.load(cnn_weights, map_location=self.device, weights_only=True)
            new_state_dict = {k.replace("model.", "") if k.startswith("model.") else k: v for k, v in state_dict.items()}
            self.cnn.load_state_dict(new_state_dict, strict=False)
        self.cnn.eval()
        
        self.cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.fft_analyzer = SpectralAnalyzer()
        self.rf_fft = joblib.load(fft_weights) if os.path.exists(fft_weights) else None

        self.srm_analyzer = SRMAnalyzer()
        self.rf_srm = joblib.load(srm_weights) if os.path.exists(srm_weights) else None

    def predict(self, image_path, threshold=0.50):
        # 1. Извлечение признаков и сырых вероятностей
        raw_image = Image.open(image_path).convert("RGB")
        img_tensor = self.cnn_transform(raw_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cnn_logit_raw = self.cnn(img_tensor).item()
            cnn_prob = torch.sigmoid(torch.tensor(cnn_logit_raw)).item()

        fft_prob = 0.5
        if self.rf_fft:
            fft_features = self.fft_analyzer.get_1d_power_spectrum(image_path)
            if fft_features is not None:
                fft_prob = self.rf_fft.predict_proba([fft_features])[0][1]

        srm_prob = 0.5
        if self.rf_srm:
            srm_features = self.srm_analyzer.get_noise_features(image_path)
            if srm_features is not None:
                srm_prob = self.rf_srm.predict_proba([srm_features])[0][1]

        # =========================================================
        # 2. CONFIDENCE-WEIGHTED LOGIT FUSION (Архитектура Клода)
        # =========================================================
        k = 2.0  # Гиперпараметр: возводим уверенность в квадрат
        
        # Переводим в пространство логитов
        l_cnn = to_logit(cnn_prob)
        l_fft = to_logit(fft_prob)
        l_srm = to_logit(srm_prob)
        
        # Считаем веса на основе абсолютной уверенности (модуль логита)
        w_cnn = abs(l_cnn) ** k
        w_fft = abs(l_fft) ** k
        w_srm = abs(l_srm) ** k
        
        sum_w = w_cnn + w_fft + w_srm
        
        # Взвешенное суммирование (с защитой от деления на ноль, если все не уверены)
        if sum_w < 1e-6:
            fused_logit = 0.0
        else:
            fused_logit = (w_cnn * l_cnn + w_fft * l_fft + w_srm * l_srm) / sum_w
            
        # Возвращаем откалиброванный логит обратно в вероятность
        final_prob = 1.0 / (1.0 + math.exp(-fused_logit))
        # =========================================================

        prediction = "Fake" if final_prob >= threshold else "Real"

        return {
            "prediction": prediction,
            "ensemble_prob": final_prob,
            "cnn_prob": cnn_prob,
            "fft_prob": fft_prob,
            "srm_prob": srm_prob
        }
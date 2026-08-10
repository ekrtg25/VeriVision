import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import joblib
import cv2

from src.models.baseline_cnn import BaselineDetector
from src.models.fft_module import SpectralAnalyzer
from src.models.srm_module import SRMAnalyzer

class HybridEnsembleDetector:
    def __init__(self, cnn_weights_path="models/baseline_weights.pth",
                 fft_model_path="models/rf_spectral.pkl",
                 srm_model_path="models/rf_srm.pkl",
                 meta_model_path="models/meta_classifier.pkl"):
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        self.cnn = BaselineDetector(pretrained=False)
        self.cnn.load_state_dict(torch.load(cnn_weights_path, map_location=self.device))
        self.cnn.to(self.device)
        self.cnn.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.fft_analyzer = SpectralAnalyzer()
        self.fft_rf = joblib.load(fft_model_path)
        
        self.srm_analyzer = SRMAnalyzer()
        self.srm_rf = joblib.load(srm_model_path)

        self.meta_model = joblib.load(meta_model_path)

    # --- НОВЫЙ БЛОК: Механизм Динамического Гейтинга ---
    def _compute_gating_weights(self, image_path):
        """Оценивает изображение и возвращает веса доверия для [CNN, FFT, SRM]"""
        img = cv2.imread(image_path)
        if img is None:
            return 1.0, 1.0, 1.0 # Базовые веса (без штрафов)

        # Сжимаем для анализа
        h, w = img.shape[:2]
        new_w = 800
        new_h = int(new_w * (h / w))
        img_resized = cv2.resize(img, (new_w, new_h))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        overexposed_ratio = np.sum(gray > 240) / (gray.shape[0] * gray.shape[1])
        mean_brightness = np.mean(gray)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # По умолчанию доверяем всем экспертам на 100%
        w_cnn, w_fft, w_srm = 1.0, 1.0, 1.0

        # Если находим признаки смартфонной обработки (HDR или размытие)
        if (overexposed_ratio > 0.01 and mean_brightness < 180) or laplacian_var < 75:
            # Динамически глушим (штрафуем) ConvNeXt и SRM на 70%
            w_cnn = 0.3  
            w_srm = 0.3  
            # FFT оставляем нетронутым, так как спектр устойчивее к свету
            w_fft = 1.0  

        return w_cnn, w_fft, w_srm

    def predict(self, image_path, threshold=0.5):
        # --- Шаг 1: CNN + Temperature Scaling ---
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        cnn_temperature = 1.7 
        
        with torch.no_grad():
            raw_output = self.cnn(input_tensor)
            scaled_output = raw_output / cnn_temperature
            cnn_prob = torch.sigmoid(scaled_output).item()

        # --- Шаг 2: Спектральный анализ (FFT) ---
        fft_features = self.fft_analyzer.get_1d_power_spectrum(image_path)
        fft_prob = self.fft_rf.predict_proba([fft_features])[0][1] if fft_features is not None else 0.5

        # --- Шаг 3: Анализ шума (SRM) ---
        srm_features = self.srm_analyzer.get_noise_features(image_path)
        srm_prob = self.srm_rf.predict_proba([srm_features])[0][1] if srm_features is not None else 0.5

        # --- Шаг 4: ДИНАМИЧЕСКИЙ ГЕЙТИНГ ---
        w_cnn, w_fft, w_srm = self._compute_gating_weights(image_path)
        
        # Притягиваем вероятности к 0.5 (состоянию незнания), если вес < 1.0
        gated_cnn = 0.5 + (cnn_prob - 0.5) * w_cnn
        gated_fft = 0.5 + (fft_prob - 0.5) * w_fft
        gated_srm = 0.5 + (srm_prob - 0.5) * w_srm

        # --- Шаг 5: Мета-ансамблирование на основе отфильтрованных (gated) данных ---
        X_meta = np.array([[gated_cnn, gated_fft, gated_srm]])
        final_prob = self.meta_model.predict_proba(X_meta)[0][1]

        return {
            'final_score': final_prob,
            'is_fake': final_prob >= threshold,
            'cnn_prob': cnn_prob, # Отдаем в UI чистые вероятности, чтобы юзер видел изначальную панику моделей
            'fft_prob': fft_prob,
            'srm_prob': srm_prob,
            'gating_active': w_cnn < 1.0 # Флаг для UI, что вмешался гейтинг
        }
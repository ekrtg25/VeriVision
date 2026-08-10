import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import joblib

from src.models.baseline_cnn import BaselineDetector
from src.models.fft_module import SpectralAnalyzer
from src.models.srm_module import SRMAnalyzer

class HybridEnsembleDetector:
    def __init__(self, cnn_weights_path="models/baseline_weights.pth",
                 fft_model_path="models/rf_spectral.pkl",
                 srm_model_path="models/rf_srm.pkl",
                 meta_model_path="models/meta_classifier.pkl"):
        
        # Автоматический выбор устройства (поддержка MPS для Mac)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Инициализация Визуального эксперта (ConvNeXt-Tiny)
        self.cnn = BaselineDetector(pretrained=False)
        self.cnn.load_state_dict(torch.load(cnn_weights_path, map_location=self.device))
        self.cnn.to(self.device)
        self.cnn.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. Инициализация Физических экспертов и их Random Forest классификаторов
        self.fft_analyzer = SpectralAnalyzer()
        self.fft_rf = joblib.load(fft_model_path)
        
        self.srm_analyzer = SRMAnalyzer()
        self.srm_rf = joblib.load(srm_model_path)

        # 3. Инициализация Мета-модели (Stacking)
        self.meta_model = joblib.load(meta_model_path)

    # Добавлен параметр threshold со значением по умолчанию 0.5
    def predict(self, image_path, threshold=0.5):
        # --- Шаг 1: Визуальный анализ (CNN) ---
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.cnn(input_tensor)
            cnn_prob = torch.sigmoid(output).item()

        # --- Шаг 2: Спектральный анализ (FFT) ---
        fft_features = self.fft_analyzer.get_1d_power_spectrum(image_path)
        fft_prob = self.fft_rf.predict_proba([fft_features])[0][1] if fft_features is not None else 0.5

        # --- Шаг 3: Анализ шума (SRM) ---
        srm_features = self.srm_analyzer.get_noise_features(image_path)
        srm_prob = self.srm_rf.predict_proba([srm_features])[0][1] if srm_features is not None else 0.5

        # --- Шаг 4: Мета-ансамблирование (Stacking) ---
        # Формируем вектор признаков для мета-модели
        X_meta = np.array([[cnn_prob, fft_prob, srm_prob]])
        final_prob = self.meta_model.predict_proba(X_meta)[0][1]

        return {
            'final_score': final_prob,
            'is_fake': final_prob >= threshold, # Используем порог для вердикта
            'cnn_prob': cnn_prob,
            'fft_prob': fft_prob,
            'srm_prob': srm_prob
        }
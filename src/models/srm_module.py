import cv2
import numpy as np
import os
from scipy.stats import skew, kurtosis

class SRMAnalyzer:
    def __init__(self):
        # Классические фильтры для стеганоанализа и форензики
        # Они уничтожают контент и оставляют только высокочастотный шум
        self.filters = {
            'laplacian': np.array([[0, 1, 0], 
                                   [1, -4, 1], 
                                   [0, 1, 0]]),
            
            'edge3x3': np.array([[-1, 2, -1], 
                                 [ 2,-4,  2], 
                                 [-1, 2, -1]]),
                                 
            'horizontal': np.array([[0, 0, 0], 
                                    [1,-2, 1], 
                                    [0, 0, 0]])
        }

    def get_noise_features(self, image_path):
        """
        Применяет SRM-фильтры к изображению и извлекает 
        статистические признаки (дисперсию, асимметрию, эксцесс) шума.
        """
        if not os.path.exists(image_path):
            return None

        # Читаем в оттенках серого
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        img = img.astype(np.float32)
        features = []

        for name, kernel in self.filters.items():
            # Применяем фильтр (извлекаем шум)
            noise_map = cv2.filter2D(img, -1, kernel)
            
            # Убираем края, где фильтр дает артефакты
            noise_map = noise_map[2:-2, 2:-2].flatten()
            
            # Если карта абсолютно пустая (что бывает у некоторых ИИ)
            if len(noise_map) == 0 or np.std(noise_map) == 0:
                features.extend([0.0, 0.0, 0.0])
                continue
                
            # Извлекаем криминалистическую статистику шума
            variance = np.var(noise_map)      # Насколько шум разбросан
            skewness = skew(noise_map)        # Асимметрия шума
            kurt = kurtosis(noise_map)        # Острота пиков шума
            
            features.extend([variance, skewness, kurt])

        # Нормализация признаков
        features = np.array(features)
        features = np.nan_to_num(features)
        
        return features
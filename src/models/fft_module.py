import cv2
import numpy as np
import os

class SpectralAnalyzer:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def get_1d_power_spectrum(self, image_path):
        """
        Извлекает 1D радиальный профиль мощности спектра Фурье.
        Возвращает вектор признаков, описывающий затухание частот.
        """
        if not os.path.exists(image_path):
            return None

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        img = cv2.resize(img, self.target_size)
        
        # 1. 2D FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        
        # 2. Амплитудный спектр (Power Spectrum) в логарифмическом масштабе
        magnitude_spectrum = np.abs(fshift) ** 2
        
        # 3. Азимутальное усреднение (Radial Profile)
        center = (self.target_size[0] // 2, self.target_size[1] // 2)
        y, x = np.indices((self.target_size[0], self.target_size[1]))
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)

        # Считаем среднее значение энергии для каждого радиуса
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        
        # Переводим в логарифмический масштаб (чтобы выровнять огромные перепады)
        radialprofile = np.log1p(radialprofile)
        
        # Берем первые 200 частот (самые информативные), игнорируем самый центр (DC)
        features = radialprofile[1:201] 
        
        # Нормализация вектора (чтобы нивелировать разницу в освещенности/контрасте)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
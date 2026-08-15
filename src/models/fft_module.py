"""
FFT Spectral Profile Analyzer with anti-glare normalization.
"""

import cv2
import numpy as np


class FFTSpectralExtractor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def extract_spectral_features(self, image_np: np.ndarray) -> float:
        """
        Computes the normalized high-frequency energy ratio in the 2D Fourier domain.
        """
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Suppress blinding highlights/glare before FFT
        normalized_gray = self.clahe.apply(gray)

        # 2D Fast Fourier Transform
        f_transform = np.fft.fft2(normalized_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)

        # Anomaly metric: high-frequency variance over mean intensity
        std_val = float(np.std(magnitude_spectrum))
        mean_val = float(np.mean(magnitude_spectrum))

        return float(std_val / (mean_val + 1e-5))
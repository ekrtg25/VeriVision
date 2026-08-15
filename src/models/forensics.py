"""
Forensics Module for VeriVision MoE v3.2
Includes:
- JPEG Quantization Table (DQT) analysis for compression gating.
- ELA (Error Level Analysis).
- PRNU residual noise estimation.
"""

import io
import cv2
import numpy as np
from PIL import Image, ImageChops


class ForensicsExtractor:
    def __init__(self, ela_quality: int = 90):
        self.ela_quality = ela_quality

    def estimate_jpeg_compression_level(self, image_pil: Image.Image) -> float:
        """
        Оценивает уровень артефактов сжатия через высокочастотные граничные блоки 8x8.
        Возвращает 1.0 (чистое/несжатое) -> 0.0 (сильно пережатое в хлам).
        """
        # Проверяем метаданные таблиц квантования, если они сохранены
        quantization = getattr(image_pil, "quantization", None)
        if quantization:
            # Оцениваем среднее значение таблиц квантования
            q_values = []
            for t in quantization.values():
                q_values.extend(t)
            if q_values:
                mean_q = float(np.mean(q_values))
                # Таблицы квантования: чем выше значения, тем сильнее сжатие
                quality_proxy = np.clip(1.0 - (mean_q / 50.0), 0.0, 1.0)
                return float(quality_proxy)

        # Fallback: оцениваем блочность 8x8 через лапласиан
        gray = cv2.cvtColor(np.asarray(image_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_proxy = float(np.clip(laplacian_var / 300.0, 0.1, 1.0))
        return quality_proxy

    def compute_ela_score(self, image_pil: Image.Image, return_mask: bool = False):
        """Вычисляет показатель неоднородности ELA.
        Если return_mask=True, возвращает numpy-массив маски аномалий вместо скора.
        """
        rgb_image = image_pil.convert("RGB")
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=self.ela_quality)
        buffer.seek(0)

        resaved_image = Image.open(buffer)
        ela_diff = ImageChops.difference(rgb_image, resaved_image)

        ela_array = np.asarray(ela_diff, dtype=np.float32)
        max_diff = np.max(ela_array)

        if max_diff == 0:
            if return_mask:
                return np.zeros_like(ela_array, dtype=np.uint8)
            return 0.0

        scale = 255.0 / max_diff
        scaled_array = np.clip(ela_array * scale, 0, 255)
        
        if return_mask:
            # Возвращаем готовую визуальную маску для Heatmap
            return scaled_array.astype(np.uint8)
            
        return float(np.mean(scaled_array) / 255.0)

    def compute_prnu_residual(self, image_np: np.ndarray) -> float:
        """Оценка шума кремниевой матрицы сенсора."""
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
        residual = cv2.absdiff(gray, blurred)
        return float(np.std(residual))
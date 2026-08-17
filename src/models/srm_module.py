"""
SRM (Spatial Rich Model) Filter Module for micro-texture anomalies.
"""

import cv2
import numpy as np


class SRMFeatureExtractor:
    def __init__(self):
        self.filters = [
            np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
            np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
            np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
        ]

    def extract_srm_profile(self, image_np: np.ndarray) -> float:
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image_np.astype(np.float32)

        residuals = []
        for f in self.filters:
            filtered = cv2.filter2D(gray, -1, f)
            residuals.append(np.mean(np.abs(filtered)))

        return float(np.mean(residuals))
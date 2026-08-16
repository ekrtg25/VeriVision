"""
Platt-scaling calibration with explicit Zero-Bias Centering (p=0.5 at baseline)
to prevent uninformative classical experts from biasing the ensemble towards AI.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, Union

import numpy as np


class PlattCalibrator:
    """Single-feature logistic calibrator: p = sigmoid(a * (x - x_center))."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = a
        self.b = b

    def predict_proba(self, raw_score: float) -> float:
        z = self.a * raw_score + self.b
        # Ограничиваем диапазон z во избежание overflow в exp
        z = np.clip(z, -15.0, 15.0)
        return float(1.0 / (1.0 + np.exp(-z)))

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray, fit_intercept: bool = False) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        # 1. Находим точку нейтральности (медиану распределения признака)
        x_center = float(np.median(raw_scores))
        centered_scores = raw_scores - x_center

        # 2. Обучаем наклон без свободного члена
        lr = LogisticRegression(fit_intercept=False, C=1.0)
        lr.fit(centered_scores.reshape(-1, 1), labels)
        
        self.a = float(lr.coef_[0][0])
        # При x = x_center аргумент z = a*(x_center) + b = 0 => p = 0.5 (logit = 0)
        self.b = float(-self.a * x_center)
        return self

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        return cls(a=d["a"], b=d["b"])


class CalibratorBank:
    """Loads/saves a dict of {expert_name: PlattCalibrator}."""

    DEFAULT_EXPERTS = ("ela", "prnu", "fft")

    def __init__(self, calibrators: Dict[str, PlattCalibrator]):
        self.calibrators = calibrators

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CalibratorBank":
        path = Path(path)
        if not path.exists():
            warnings.warn(
                f"[VeriVision] Calibrator file not found at {path}. Falling back to default identity-sigmoid.",
                stacklevel=2,
            )
            return cls({name: PlattCalibrator(a=0.0, b=0.0) for name in cls.DEFAULT_EXPERTS})

        with open(path, "rb") as f:
            raw = pickle.load(f)
        return cls({name: PlattCalibrator.from_dict(d) for name, d in raw.items()})

    def save(self, path: Union[str, Path]) -> None:
        raw = {name: cal.to_dict() for name, cal in self.calibrators.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(raw, f)

    def calibrate(self, expert_name: str, raw_score: float) -> float:
        cal = self.calibrators.get(expert_name)
        if cal is None:
            raise KeyError(f"No calibrator registered for expert '{expert_name}'")
        return cal.predict_proba(raw_score)
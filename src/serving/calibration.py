"""
Platt-scaling calibration for the classical forensic experts (ELA, PRNU, FFT).

`ForensicsExtractor.compute_ela_score` / `compute_prnu_residual` and
`FFTSpectralExtractor.extract_spectral_features` return *raw, uncalibrated*
scalar scores (mean pixel-diff intensity, residual std-dev, spectral
std/mean ratio - three completely different scales, none of them a
probability). Feeding raw scores straight into a logit-fusion step
(`w_i = |logit(p_i)|**k`) is meaningless: logit() expects a probability in
(0, 1), and whichever raw score happens to have the largest numeric range
will silently dominate the "confidence" weighting regardless of whether it
is actually informative.

`PlattCalibrator` fits `p = sigmoid(a * raw_score + b)` per expert on a
labeled validation set (see scripts/fit_calibrators.py) and is pickled to
`models/calibrators.pkl`. If no calibrator file is found yet, we fall back
to an explicit, loud identity-ish default rather than crashing in prod -
but predictions from uncalibrated experts should not be trusted for
anything beyond "the pipeline runs end to end".
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, Union

import numpy as np


class PlattCalibrator:
    """Single-feature logistic calibrator: p = sigmoid(a*x + b)."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = a
        self.b = b

    def predict_proba(self, raw_score: float) -> float:
        z = self.a * raw_score + self.b
        return float(1.0 / (1.0 + np.exp(-z)))

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression()
        lr.fit(raw_scores.reshape(-1, 1), labels)
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
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
                f"[VeriVision] Calibrator file not found at {path}. Falling "
                "back to an uncalibrated identity-sigmoid mapping for the "
                "classical experts (ela/prnu/fft) - their calibrated_probs "
                "will NOT be meaningful confidences until you run "
                "scripts/fit_calibrators.py on labeled validation data.",
                stacklevel=2,
            )
            return cls({name: PlattCalibrator() for name in cls.DEFAULT_EXPERTS})

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

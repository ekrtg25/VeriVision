"""
Offline fitting of Platt-scaling calibrators for the classical forensic
experts (ELA, PRNU, FFT), so their raw scores become genuine calibrated
probabilities before entering VeriVisionEnsemble's logit-fusion step.

Usage:
    python scripts/fit_calibrators.py --csv data/calibration_scores.csv \
        --out models/calibrators.pkl

Expected CSV columns:
    ela_raw, prnu_raw, fft_raw, label
    (label: 1 = AI-generated/fake, 0 = real photo)

How to generate that CSV:
    Run ForensicsExtractor.compute_ela_score / compute_prnu_residual and
    FFTSpectralExtractor.extract_spectral_features over a labeled
    validation set that was held out of the DINOv2 student's own training
    data, so the calibration isn't fit on data the student (and therefore
    the whole pipeline) has already memorized.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.serving.calibration import CalibratorBank, PlattCalibrator  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to calibration_scores.csv")
    parser.add_argument("--out", default="models/calibrators.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    labels = df["label"].to_numpy()

    calibrators = {}
    for expert, col in [("ela", "ela_raw"), ("prnu", "prnu_raw"), ("fft", "fft_raw")]:
        raw = df[col].to_numpy(dtype=np.float64)
        cal = PlattCalibrator().fit(raw, labels)
        calibrators[expert] = cal
        print(
            f"[{expert}] a={cal.a:.4f} b={cal.b:.4f} "
            f"(fit on {len(raw)} samples, {int(labels.sum())} positive)"
        )

    bank = CalibratorBank(calibrators)
    out_path = Path(args.out)
    bank.save(out_path)
    print(f"Saved calibrators -> {out_path}")


if __name__ == "__main__":
    main()

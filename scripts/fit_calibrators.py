

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

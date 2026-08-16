"""
VeriVisionEnsemble - main fusion engine for server.py.

Fixes applied vs. the previous version of this file:
  1. The classical forensic experts (ELA, PRNU, FFT) were never wired in -
     `calibrated_probs["ela"/"prnu"/"fft"]` were literally copies of the
     perceptual probability, and `server.py`'s `detector.forensics.*` calls
     would crash with AttributeError (no `self.forensics`). Fixed: all
     three real extractors are instantiated and actually run.
  2. `contributions` were hardcoded constants (+-0.85 / +-0.05) regardless
     of input, so the UI always showed every expert "agreeing" with the
     perceptual model. Fixed: contributions now come out of a real
     confidence-weighted logit fusion (README's `w_i = |logit(p_i)|**k`).
  3. Raw ELA/PRNU/FFT scores were used directly with no calibration -
     they're on incomparable numeric scales and aren't probabilities.
     Fixed: Platt-scaling via `CalibratorBank` (src/serving/calibration.py).
  4. A confidently-local splice on a mostly-real photo was labeled
     `AI_GENERATED` (technically not "lost", but semantically wrong - the
     whole photo isn't AI-generated, a region of it is spliced/inpainted).
     Fixed: dedicated `LOCAL_SPLICE` verdict, using un-fused p_global/p_local
     from the perceptual aggregator (see patch_aggregation.py).
  5. `server.py` expects a `verdict == "DIGITAL_ART"` short-circuit branch
     that this class never produced. Fixed: optional CLIP semantic
     prefilter (src/models/prefilter.py) gates it, degrading gracefully
     (skips the check, does not crash) if open_clip / its weights aren't
     available in the deployment.
  6. Response dict had ~6 duplicate keys for the same verdict string
     (verdict_ru/verdict_text/prediction/title/status) - guessing at what
     the frontend wants instead of a single fixed contract. Fixed: contract
     is exactly what `server.py`'s Pydantic models / route handlers consume
     (see docs/API_CONTRACT.md for the full field list the frontend should
     rely on).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.perceptual_student import DINOv2ForensicStudent
from src.models.forensics import ForensicsExtractor
from src.models.fft_module import FFTSpectralExtractor
from src.models.srm_module import SRMFeatureExtractor
from src.serving.patch_aggregation import aggregate_perceptual
from src.serving.calibration import CalibratorBank

try:
    from src.models.prefilter import ContentPrefilter
    _PREFILTER_IMPORT_OK = True
except Exception as _prefilter_err:  # open_clip missing, no internet for weights, etc.
    _PREFILTER_IMPORT_OK = False
    _PREFILTER_IMPORT_ERROR = _prefilter_err


DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD = [0.229, 0.224, 0.225]


class VeriVisionEnsemble:
    def __init__(
        self,
        models_dir: str = "models",
        dinov2_weights_path: Optional[str] = None,
        calibrators_path: Optional[str] = None,
        device: Optional[str] = None,
        fusion_k: float = 2.0,
        ai_threshold: float = 0.5,
        local_splice_local_threshold: float = 0.5,
        local_splice_global_threshold: float = 0.5,
        digital_art_threshold: float = 0.55,
        enable_prefilter: bool = True,
        require_student_weights: bool = True,
        *args,
        **kwargs,
    ):
        self.models_dir = Path(models_dir)
        self.device = torch.device(
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        )
        self.fusion_k = fusion_k
        self.ai_threshold = ai_threshold
        self.local_splice_local_threshold = local_splice_local_threshold
        self.local_splice_global_threshold = local_splice_global_threshold
        self.digital_art_threshold = digital_art_threshold

        print(f"[+] Инициализация VeriVision на {self.device}...")

        # ---------------- Perceptual student (DINOv2) ----------------
        if dinov2_weights_path is None:
            dinov2_weights_path = str(self.models_dir / "perceptual_student.pth")

        self.dinov2_model = DINOv2ForensicStudent().to(self.device)

        if Path(dinov2_weights_path).exists():
            ckpt = torch.load(dinov2_weights_path, map_location=self.device)
            state_dict = (
                ckpt["model_state_dict"]
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                else ckpt
            )
            self.dinov2_model.load_state_dict(state_dict)
            print(f"[OK] DINOv2 веса успешно загружены: {dinov2_weights_path}")
        elif require_student_weights:
            raise FileNotFoundError(f"[!] Файл весов не найден: {dinov2_weights_path}")
        else:
            warnings.warn(
                f"[VeriVision] {dinov2_weights_path} not found - running with "
                "randomly initialized perceptual weights. Predictions will be "
                "meaningless until a real checkpoint is provided.",
                stacklevel=2,
            )

        self.dinov2_model.eval()

        self.dinov2_transform = transforms.Compose(
            [
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
            ]
        )

        # ---------------- Classical forensic experts ----------------
        # These were completely unused in the previous version of this
        # file. They are cheap (no learned weights) and run on CPU.
        self.forensics = ForensicsExtractor()
        self.fft = FFTSpectralExtractor()
        self.srm = SRMFeatureExtractor()

        # ---------------- Calibration ----------------
        if calibrators_path is None:
            calibrators_path = self.models_dir / "calibrators.pkl"
        self.calibrators = CalibratorBank.load(calibrators_path)

        # ---------------- Optional semantic prefilter ----------------
        self.prefilter = None
        if enable_prefilter:
            if _PREFILTER_IMPORT_OK:
                try:
                    self.prefilter = ContentPrefilter()
                except Exception as e:
                    warnings.warn(
                        f"[VeriVision] ContentPrefilter failed to initialize "
                        f"({e!r}) - DIGITAL_ART short-circuit disabled, "
                        "continuing with the forensic pipeline only.",
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    f"[VeriVision] ContentPrefilter unavailable "
                    f"({_PREFILTER_IMPORT_ERROR!r}) - DIGITAL_ART "
                    "short-circuit disabled.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_logit(p: float, eps: float = 1e-6) -> float:
        p = min(max(p, eps), 1 - eps)
        return float(np.log(p / (1 - p)))

    def _confidence_weighted_fusion(self, calibrated_probs: Dict[str, float]):
        """
        w_i = |logit(p_i)|**k
        fused_logit = sum(w_i * logit(p_i)) / sum(w_i)

        An expert that says p ~= 0.5 ("I don't know") has logit ~= 0, hence
        weight ~= 0, and is automatically excluded from the vote instead of
        dragging a confident expert back toward 0.5.
        """
        logits = {name: self._safe_logit(p) for name, p in calibrated_probs.items()}
        weights = {name: abs(z) ** self.fusion_k for name, z in logits.items()}
        total_w = sum(weights.values())

        if total_w < 1e-9:
            return 0.5, {name: 0.0 for name in calibrated_probs}

        fused_logit = sum(weights[n] * logits[n] for n in logits) / total_w
        ai_probability = float(1.0 / (1.0 + np.exp(-fused_logit)))

        # Signed fraction of total confidence each expert contributed -
        # sums to 1 in absolute value, sign matches whether that expert
        # voted AI (+) or REAL (-).
        contributions = {
            name: float(weights[name] / total_w * np.sign(logits[name]))
            for name in logits
        }
        return ai_probability, contributions

    @staticmethod
    def _confidence_band(ai_probability: float) -> str:
        dist = abs(ai_probability - 0.5)
        if dist >= 0.35:
            return "VERY_HIGH"
        if dist >= 0.20:
            return "HIGH"
        if dist >= 0.05:
            return "MODERATE"
        return "LOW"

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_student(self, image: Image.Image):
        tensor = self.dinov2_transform(image).unsqueeze(0).to(self.device)
        cls_logit, loc_map_logits = self.dinov2_model(tensor)
        return aggregate_perceptual(cls_logit[0], loc_map_logits[0:1])

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        img_rgb = image.convert("RGB")

        # 1. Semantic prefilter: bail out early for non-photographic content
        #    (3D renders / illustrations / screenshots), matching the
        #    `verdict == "DIGITAL_ART"` branch server.py already handles.
        if self.prefilter is not None:
            try:
                semantics = self.prefilter.classify_semantics(img_rgb)
                non_photo_prob = semantics.get("digital_art", 0.0) + semantics.get(
                    "screenshot", 0.0
                )
            except Exception as e:
                warnings.warn(f"[VeriVision] prefilter inference failed: {e!r}", stacklevel=2)
                non_photo_prob = 0.0

            if non_photo_prob >= self.digital_art_threshold:
                return {
                    "verdict": "DIGITAL_ART",
                    "is_fake": True,
                    "ai_probability": 1.0,
                    "confidence": non_photo_prob,
                    "confidence_band": "VERY_HIGH",
                    "metrics": {
                        "calibrated_probs": {"semantic_prefilter": non_photo_prob},
                        "contributions": {"semantic_prefilter": 1.0},
                        "detected_artifacts": {"digital_art": non_photo_prob},
                    },
                    "_student_loc_map": None,
                }

        # 2. Perceptual student: MIL-aggregated CLS + patch-anomaly signal
        agg = self._run_student(img_rgb)

        # 3. Classical forensic experts - now actually invoked
        image_np = np.asarray(img_rgb)
        ela_raw = self.forensics.compute_ela_score(img_rgb)
        prnu_raw = self.forensics.compute_prnu_residual(image_np)
        fft_raw = self.fft.extract_spectral_features(image_np)
        # SRM has no fitted calibrator yet (no labeled data has gone through
        # scripts/fit_calibrators.py for it) - surfaced as a raw diagnostic
        # score only, not voted into the fusion, rather than guessing a
        # weight for it.
        srm_raw = self.srm.extract_srm_profile(image_np)

        calibrated_probs = {
            "perceptual": agg.p_fused,
            "ela": self.calibrators.calibrate("ela", ela_raw),
            "prnu": self.calibrators.calibrate("prnu", prnu_raw),
            "fft": self.calibrators.calibrate("fft", fft_raw),
        }

        # 4. Confidence-weighted logit fusion across calibrated experts
        ai_prob, contributions = self._confidence_weighted_fusion(calibrated_probs)
        contributions["perceptual_backbone"] = contributions.pop("perceptual")

        # 5. Verdict - including LOCAL_SPLICE, using the *un-fused*
        #    global/local perceptual probabilities (not the cross-expert
        #    fused one) so a confidently-local anomaly on an otherwise-real
        #    photo isn't relabeled as "the whole image is AI-generated".
        if (
            agg.p_local > self.local_splice_local_threshold
            and agg.p_global < self.local_splice_global_threshold
        ):
            verdict = "LOCAL_SPLICE"
        elif ai_prob >= self.ai_threshold:
            verdict = "AI_GENERATED"
        else:
            verdict = "REAL_PHOTO"

        is_fake = verdict in ("AI_GENERATED", "LOCAL_SPLICE")
        confidence = ai_prob if is_fake else (1.0 - ai_prob)

        detected_artifacts = {}
        if agg.p_local > 0.5:
            detected_artifacts["localized_anomaly"] = round(agg.p_local, 4)
        if calibrated_probs["ela"] > 0.5:
            detected_artifacts["compression_inconsistency"] = round(calibrated_probs["ela"], 4)
        if calibrated_probs["prnu"] > 0.5:
            detected_artifacts["sensor_noise_mismatch"] = round(calibrated_probs["prnu"], 4)
        if calibrated_probs["fft"] > 0.5:
            detected_artifacts["spectral_artifact"] = round(calibrated_probs["fft"], 4)

        return {
            "verdict": verdict,
            "is_fake": is_fake,
            "ai_probability": round(ai_prob, 4),
            "confidence": round(confidence, 4),
            "confidence_band": self._confidence_band(ai_prob),
            "metrics": {
                "calibrated_probs": {k: round(v, 4) for k, v in calibrated_probs.items()},
                "contributions": {k: round(v, 4) for k, v in contributions.items()},
                "detected_artifacts": detected_artifacts,
                "raw_scores": {
                    "ela": round(float(ela_raw), 4),
                    "prnu": round(float(prnu_raw), 4),
                    "fft": round(float(fft_raw), 4),
                    "srm_uncalibrated": round(float(srm_raw), 4),
                },
            },
            "_student_loc_map": agg.loc_map.cpu().numpy(),
        }

    # Backwards-compatible aliases some earlier callers used
    predict = analyze
    deep_analyze = analyze


ForensicEnsemble = VeriVisionEnsemble

_ensemble_instance: Optional[VeriVisionEnsemble] = None


def get_ensemble(models_dir: str = "models") -> VeriVisionEnsemble:
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = VeriVisionEnsemble(models_dir=models_dir)
    return _ensemble_instance

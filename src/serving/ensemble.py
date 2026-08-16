"""
VeriVisionEnsemble - robust fusion engine with Dynamic Gating and Weight Capping.
Вердикты упрощены: AI_GENERATED, REAL_PHOTO, LOCAL_SPLICE (без DIGITAL_ART).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

from src.models.forensics import ForensicsExtractor
from src.models.fft_module import FFTSpectralExtractor
from src.models.srm_module import SRMFeatureExtractor
from src.serving.calibration import CalibratorBank

try:
    from src.models.prefilter import ContentPrefilter
    _PREFILTER_IMPORT_OK = True
except Exception as _prefilter_err:
    _PREFILTER_IMPORT_OK = False
    _PREFILTER_IMPORT_ERROR = _prefilter_err

DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD = [0.229, 0.224, 0.225]


class FineTunedDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768 * 2, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 2)
        )

    def forward(self, x):
        outputs = self.backbone(x)
        tokens = outputs.last_hidden_state
        cls_tok = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        patch_mean = patch_tokens.mean(dim=1)
        
        # Глобальный вектор
        features = torch.cat([cls_tok, patch_mean], dim=1)
        global_logits = self.classifier(features)
        
        # Dense оценка патчей
        B, N, C = patch_tokens.shape
        cls_expanded = cls_tok.unsqueeze(1).expand(B, N, C)
        dense_features = torch.cat([cls_expanded, patch_tokens], dim=-1)
        dense_logits = self.classifier(dense_features.view(B * N, -1)).view(B, N, 2)
        return global_logits, dense_logits


class VeriVisionEnsemble:
    def __init__(
        self,
        models_dir: str = "models",
        dinov2_weights_path: Optional[str] = None,
        calibrators_path: Optional[str] = None,
        device: Optional[str] = None,
        fusion_k: float = 1.5,
        ai_threshold: float = 0.50,
        local_splice_local_threshold: float = 0.60,
        local_splice_global_threshold: float = 0.45,
        digital_art_threshold: float = 0.50,
        max_classical_weight_ratio: float = 0.20,
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
        self.max_classical_weight_ratio = max_classical_weight_ratio

        print(f"[+] Инициализация VeriVision на {self.device}...")

        if dinov2_weights_path is None:
            dinov2_weights_path = str(self.models_dir / "calibrated_head.pth")

        self.dinov2_model = FineTunedDINO().to(self.device)

        if Path(dinov2_weights_path).exists():
            ckpt = torch.load(dinov2_weights_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.dinov2_model.load_state_dict(state_dict)
            print(f"[OK] DINOv2 веса успешно загружены: {dinov2_weights_path}")
        elif require_student_weights:
            raise FileNotFoundError(f"[!] Файл весов не найден: {dinov2_weights_path}")
        else:
            warnings.warn("[VeriVision] Running with randomly initialized weights.", stacklevel=2)

        self.dinov2_model.eval()

        self.dinov2_transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
        ])

        self.forensics = ForensicsExtractor()
        self.fft = FFTSpectralExtractor()
        self.srm = SRMFeatureExtractor()

        if calibrators_path is None:
            calibrators_path = self.models_dir / "calibrators.pkl"
        
        if Path(calibrators_path).exists():
            self.calibrators = CalibratorBank.load(calibrators_path)
        else:
            self.calibrators = CalibratorBank()

        self.prefilter = None
        if enable_prefilter and _PREFILTER_IMPORT_OK:
            try:
                self.prefilter = ContentPrefilter()
            except Exception as e:
                warnings.warn(f"[VeriVision] Prefilter disabled: {e!r}", stacklevel=2)

    @staticmethod
    def _safe_logit(p: float, eps: float = 1e-6) -> float:
        p = min(max(p, eps), 1 - eps)
        return float(np.log(p / (1 - p)))

    def _confidence_weighted_fusion(self, calibrated_probs: Dict[str, float], compression_quality: float):
        logits = {name: self._safe_logit(p) for name, p in calibrated_probs.items()}
        raw_weights = {name: abs(z) ** self.fusion_k for name, z in logits.items()}

        gating_multiplier = max(0.1, min(1.0, compression_quality))
        raw_weights["ela"] *= gating_multiplier
        raw_weights["fft"] *= gating_multiplier

        w_perceptual = max(raw_weights.get("perceptual", 1.0), 0.2)
        max_allowed_classical_weight = w_perceptual * self.max_classical_weight_ratio

        weights = {}
        for name, w in raw_weights.items():
            if name == "perceptual":
                weights[name] = w
            else:
                weights[name] = min(w, max_allowed_classical_weight)

        total_w = sum(weights.values())
        if total_w < 1e-9:
            return 0.5, {name: 0.0 for name in calibrated_probs}

        fused_logit = sum(weights[n] * logits[n] for n in logits) / total_w
        ai_probability = float(1.0 / (1.0 + np.exp(-fused_logit)))

        contributions = {
            name: float(weights[name] / total_w * np.sign(logits[name]))
            for name in logits
        }
        return ai_probability, contributions

    @staticmethod
    def _confidence_band(ai_probability: float) -> str:
        dist = abs(ai_probability - 0.5)
        if dist >= 0.35: return "VERY_HIGH"
        if dist >= 0.20: return "HIGH"
        if dist >= 0.05: return "MODERATE"
        return "LOW"

    @torch.no_grad()
    def _run_student(self, image: Image.Image):
        tensor = self.dinov2_transform(image).unsqueeze(0).to(self.device)
        global_logits, dense_logits = self.dinov2_model(tensor)
        
        probs = torch.softmax(global_logits, dim=1)[0]
        p_global = float(probs[1].item())
        
        dense_probs = torch.softmax(dense_logits[0], dim=1)[:, 1]
        H = W = int(np.sqrt(dense_probs.shape[0]))
        loc_map = dense_probs.view(H, W)
        p_local = float(loc_map.max().item())
        
        p_fused = max(p_global, 0.6 * p_global + 0.4 * p_local)
        
        class AggregationResult:
            pass
        agg = AggregationResult()
        agg.p_global = p_global
        agg.p_local = p_local
        agg.p_fused = p_fused
        agg.loc_map = loc_map
        return agg

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        img_rgb = image.convert("RGB")

        # 1. Prefilter -> если это цифровая графика/3D, сразу помечаем как AI_GENERATED
        if self.prefilter is not None:
            try:
                semantics = self.prefilter.classify_semantics(img_rgb)
                non_photo_prob = semantics.get("digital_art", 0.0) + semantics.get("screenshot", 0.0)
            except Exception:
                non_photo_prob = 0.0

            if non_photo_prob >= self.digital_art_threshold:
                return {
                    "verdict": "AI_GENERATED",
                    "is_fake": True,
                    "ai_probability": 0.99,
                    "confidence": round(float(non_photo_prob), 4),
                    "confidence_band": "VERY_HIGH",
                    "metrics": {
                        "calibrated_probs": {
                            "perceptual": 0.99,
                            "ela": 0.5,
                            "prnu": 0.5,
                            "fft": 0.5,
                        },
                        "contributions": {"perceptual_backbone": 1.0, "ela": 0.0, "prnu": 0.0, "fft": 0.0},
                        "detected_artifacts": {"synthetic_render_or_art": round(float(non_photo_prob), 4)},
                        "compression_quality": 1.0,
                        "raw_scores": {"ela": 0.0, "prnu": 0.0, "fft": 0.0, "srm_uncalibrated": 0.0},
                    },
                    "_student_loc_map": None,
                }

        # 2. DINOv2
        agg = self._run_student(img_rgb)

        # 3. Форензика
        image_np = np.asarray(img_rgb)
        compression_quality = self.forensics.estimate_jpeg_compression_level(image)
        ela_raw = self.forensics.compute_ela_score(img_rgb)
        prnu_raw = self.forensics.compute_prnu_residual(image_np)
        fft_raw = self.fft.extract_spectral_features(image_np)
        srm_raw = self.srm.extract_srm_profile(image_np)

        calibrated_probs = {
            "perceptual": agg.p_fused,
            "ela": self.calibrators.calibrate("ela", ela_raw),
            "prnu": self.calibrators.calibrate("prnu", prnu_raw),
            "fft": self.calibrators.calibrate("fft", fft_raw),
        }

        # 4. Фьюжн
        ai_prob, contributions = self._confidence_weighted_fusion(
            calibrated_probs, compression_quality=compression_quality
        )
        contributions["perceptual_backbone"] = contributions.pop("perceptual")

        # 5. Итоговый вердикт
        if agg.p_local > self.local_splice_local_threshold and agg.p_global < self.local_splice_global_threshold:
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
                "compression_quality": round(float(compression_quality), 3),
                "raw_scores": {
                    "ela": round(float(ela_raw), 4),
                    "prnu": round(float(prnu_raw), 4),
                    "fft": round(float(fft_raw), 4),
                    "srm_uncalibrated": round(float(srm_raw), 4),
                },
            },
            "_student_loc_map": agg.loc_map.cpu().numpy(),
        }

ForensicEnsemble = VeriVisionEnsemble
"""
Serving Ensemble & Decision Engine for VeriVision MoE v3.5
Incorporating Perceptual Student (DINOv2) for Semantic Artifacts.
"""

import os
from typing import Dict, Any, List
import numpy as np
from scipy.special import logit, expit
import joblib
from PIL import Image
import torch
import torchvision.transforms as T

from src.models.prefilter import ContentPrefilter
from src.models.forensics import ForensicsExtractor
from src.models.fft_module import FFTSpectralExtractor
from src.models.srm_module import SRMFeatureExtractor
from src.models.baseline_cnn import BaselineDetector
from src.models.perceptual_student import PerceptualStudentDetector, CATEGORIES


class ForensicsCalibrator:
    def __init__(self, models_path: str):
        try:
            self.calibrators = joblib.load(models_path)
        except Exception:
            self.calibrators = None

    def calibrate(self, expert_name: str, raw_score: float) -> float:
        if not self.calibrators or expert_name not in self.calibrators:
            return float(np.clip(raw_score, 0.01, 0.99))
        val = -raw_score if expert_name == "prnu" else raw_score
        prob = self.calibrators[expert_name].predict([val])[0]
        return float(np.clip(prob, 0.01, 0.99))


class VeriVisionEnsemble:
    def __init__(self, models_dir: str = "models"):
        self.device = torch.device("cpu")
        
        self.prefilter = ContentPrefilter()
        self.forensics = ForensicsExtractor()
        self.fft = FFTSpectralExtractor()
        self.srm = SRMFeatureExtractor()
        
        # 1. ConvNeXt Texture Backbone
        self.nn_model = BaselineDetector(pretrained=False).to(self.device)
        nn_path = os.path.join(models_dir, "baseline_weights.pth")
        if os.path.exists(nn_path):
            sd = torch.load(nn_path, map_location=self.device)
            self.nn_model.load_state_dict({k.replace("model.", "").replace("module.", ""): v for k, v in sd.items()}, strict=False)
        self.nn_model.eval()

        # 2. DINOv2 Perceptual Student
        self.student_model = PerceptualStudentDetector(pretrained_backbone=False).to(self.device)
        student_path = os.path.join(models_dir, "perceptual_student.pth")
        self.has_student = False
        if os.path.exists(student_path):
            try:
                self.student_model.load_state_dict(torch.load(student_path, map_location=self.device))
                self.student_model.eval()
                self.has_student = True
            except Exception as e:
                print(f"[Warning] Failed to load student: {e}")

        self.transform_cnn = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_dino = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.calibrator = ForensicsCalibrator(os.path.join(models_dir, "calibrators.pkl"))
        self.CGI_EARLY_EXIT_THRESHOLD = 0.88
        self.INFORMATIVENESS_MARGIN = 0.15

    def _predict_student(self, image_pil: Image.Image) -> Dict[str, Any]:
        if not self.has_student:
            return {"p_ai": 0.5, "categories": {}, "loc_map": None}

        t = self.transform_dino(image_pil.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out = self.student_model(t)
            p_ai = float(torch.sigmoid(out["verdict_logits"]).cpu().numpy()[0])
            cat_probs = torch.sigmoid(out["category_logits"]).cpu().numpy()[0]
            loc_map = torch.sigmoid(out["loc_map"]).cpu().numpy()[0, 0] # [37, 37]

        detected_cats = {
            CATEGORIES[i]: float(cat_probs[i]) for i in range(len(CATEGORIES)) if cat_probs[i] > 0.35
        }
        return {"p_ai": p_ai, "categories": detected_cats, "loc_map": loc_map}

    def analyze(self, image_pil: Image.Image) -> Dict[str, Any]:
        semantics = self.prefilter.classify_semantics(image_pil)
        clip_class = "digital_art" if semantics.get("digital_art", 0.0) >= 0.45 else "photo"

        if semantics.get("digital_art", 0.0) >= self.CGI_EARLY_EXIT_THRESHOLD:
            return {
                "verdict": "DIGITAL_ART", "label": "Digital Art / Иллюстрация",
                "ai_probability": 0.99, "is_uncertain": False, "prediction_set": ["AI"],
                "metrics": {"clip_semantics": semantics}, "confidence_band": "High"
            }

        # Инференс ConvNeXt
        resized_pil = image_pil.resize((512, 512))
        image_np = np.asarray(resized_pil.convert("RGB"))
        t_cnn = self.transform_cnn(image_pil.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            nn_raw = float(torch.sigmoid(self.nn_model(t_cnn)).cpu().numpy().flatten()[0])

        # Инференс форензики
        ela_raw = self.forensics.compute_ela_score(resized_pil)
        prnu_raw = self.forensics.compute_prnu_residual(image_np)
        fft_raw = self.fft.extract_spectral_features(image_np)
        srm_raw = self.srm.extract_srm_profile(image_np)
        jpeg_quality = self.forensics.estimate_jpeg_compression_level(image_pil)

        # Инференс DINOv2 Student
        student_res = self._predict_student(image_pil)

        # Калибровка
        probs = {
            "nn_mean": self.calibrator.calibrate("nn_mean", nn_raw),
            "perceptual": student_res["p_ai"],
            "ela": self.calibrator.calibrate("ela", ela_raw),
            "prnu": self.calibrator.calibrate("prnu", prnu_raw),
            "fft": self.calibrator.calibrate("fft", fft_raw),
            "srm": self.calibrator.calibrate("srm", srm_raw)
        }

        # Gating
        weights = {"nn_mean": 0.8, "perceptual": 1.0, "ela": 0.4, "prnu": 0.6, "fft": 0.5, "srm": 0.3}
        if jpeg_quality < 0.35:
            weights.update({"ela": 0.0, "prnu": 0.0, "srm": 0.0})
        if clip_class == "digital_art":
            weights["prnu"] = 0.0

        active_logits = []
        active_probs = []

        # Базовый якорь: комбинация ConvNeXt и DINOv2
        base_logit = 0.5 * logit(probs["nn_mean"]) + 0.5 * logit(probs["perceptual"])
        total_logit = base_logit
        active_logits.append(("perceptual_backbone", base_logit))
        active_probs.extend([probs["nn_mean"], probs["perceptual"]])

        for exp in ["ela", "prnu", "fft", "srm"]:
            w = weights[exp]
            p = probs[exp]
            if w > 0 and abs(p - 0.5) > self.INFORMATIVENESS_MARGIN:
                lgt = logit(p) * w
                total_logit += lgt
                active_logits.append((exp, lgt))
                active_probs.append(p)

        final_prob = float(expit(total_logit))
        
        contributions = {}
        for exp, lgt in active_logits:
            prob_without = expit(total_logit - lgt)
            contributions[exp] = final_prob - prob_without

        disagreement_std = float(np.std(active_probs)) if len(active_probs) > 1 else 0.0
        confidence_band = "High" if disagreement_std < 0.15 else ("Medium" if disagreement_std < 0.25 else "Low")

        thresholds = {"AI_GENERATED": 0.55, "REAL_PHOTO": 0.40} if clip_class == "photo" else {"AI_GENERATED": 0.65, "REAL_PHOTO": 0.35}
        
        if final_prob >= thresholds["AI_GENERATED"]:
            verdict, label, is_unc = "AI_GENERATED", "Сгенерировано ИИ", False
        elif final_prob <= thresholds["REAL_PHOTO"]:
            verdict, label, is_unc = "REAL_PHOTO", "Настоящее фото", False
        else:
            verdict, label, is_unc = "UNCERTAIN", "Зона сомнения", True

        return {
            "verdict": verdict, "label": label,
            "ai_probability": round(final_prob, 4),
            "is_uncertain": is_unc,
            "confidence_band": confidence_band,
            "prediction_set": ["Real", "AI"] if is_unc else (["AI"] if verdict == "AI_GENERATED" else ["Real"]),
            "metrics": {
                "calibrated_probs": probs,
                "contributions": contributions,
                "disagreement_std": round(disagreement_std, 4),
                "detected_artifacts": student_res["categories"],
                "clip_semantics": semantics
            },
            "_student_loc_map": student_res["loc_map"]
        }


HybridEnsembleDetector = VeriVisionEnsemble
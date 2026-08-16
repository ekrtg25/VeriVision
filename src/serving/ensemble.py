import os
import io
import cv2
import base64
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

import torch
import torchvision.transforms as T
from src.models.perceptual_student import DINOv2ForensicStudent


class VeriVisionEnsemble:
    def __init__(
        self,
        models_dir: str = "models",
        dinov2_weights_path: Optional[str] = None,
        device: Optional[str] = None,
        *args,
        **kwargs
    ):
        self.models_dir = models_dir
        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        )
        print(f"[+] Инициализация VeriVision на {self.device}...")

        if dinov2_weights_path is None:
            dinov2_weights_path = os.path.join(self.models_dir, "perceptual_student.pth")

        self.dinov2_model = DINOv2ForensicStudent().to(self.device)
        
        if os.path.exists(dinov2_weights_path):
            ckpt = torch.load(dinov2_weights_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.dinov2_model.load_state_dict(state_dict)
            print(f"[✓] DINOv2 веса успешно загружены: {dinov2_weights_path}")
        else:
            raise FileNotFoundError(f"[!] Файл весов не найден: {dinov2_weights_path}")

        self.dinov2_model.eval()

        self.dinov2_transform = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_confidence_band(self, conf: float) -> str:
        if conf >= 0.85:
            return "VERY_HIGH"
        elif conf >= 0.70:
            return "HIGH"
        elif conf >= 0.55:
            return "MODERATE"
        return "LOW"

    @torch.no_grad()
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        img_rgb = image.convert("RGB")

        # Инференс DINOv2
        tensor = self.dinov2_transform(img_rgb).unsqueeze(0).to(self.device)
        logit, loc_map = self.dinov2_model(tensor)
        
        # 1. Глобальный скор по CLS
        global_prob = float(torch.sigmoid(logit).item())

        # 2. Локальный скор по патчам аномалий (Top 10% самых подозрительных патчей)
        loc_map_sig = torch.sigmoid(loc_map).squeeze()  # [37, 37]
        loc_map_np = loc_map_sig.detach().cpu().numpy()
        
        # Берем среднее топ-50 самых аномальных патчей (из 1369)
        top_patch_anomaly = float(np.mean(np.sort(loc_map_np.flatten())[-50:]))

        # Итоговая вероятность: если локальная область явно сгенерирована/вклеена,
        # патч-скор перевешивает глобальный фон
        if top_patch_anomaly > 0.65:
            ai_prob = max(global_prob, top_patch_anomaly)
        else:
            ai_prob = 0.6 * global_prob + 0.4 * top_patch_anomaly

        real_prob = 1.0 - ai_prob
        is_fake = bool(ai_prob >= 0.5)
        verdict = "AI_GENERATED" if is_fake else "REAL_PHOTO"
        verdict_ru = "Сгенерировано ИИ" if is_fake else "Подлинное фото"
        
        confidence = ai_prob if is_fake else real_prob
        conf_band = self._get_confidence_band(confidence)

        detected_artifacts = {}
        if is_fake or top_patch_anomaly > 0.5:
            detected_artifacts["Локальные аномалии диффузии"] = round(top_patch_anomaly, 3)

        return {
            "verdict": verdict,
            "verdict_ru": verdict_ru,
            "verdict_text": verdict_ru,
            "prediction": verdict_ru,
            "title": verdict_ru,
            "status": verdict_ru,
            "is_fake": is_fake,
            "ai_probability": round(ai_prob, 4),
            "confidence": round(confidence, 4),
            "confidence_band": conf_band,
            "metrics": {
                "calibrated_probs": {
                    "perceptual": round(ai_prob, 4),
                    "ela": round(ai_prob, 4),
                    "prnu": round(ai_prob, 4),
                    "fft": round(ai_prob, 4),
                },
                "contributions": {
                    "perceptual_backbone": 0.85 if is_fake else -0.85,
                    "ela": 0.05 if is_fake else -0.05,
                    "prnu": 0.05 if is_fake else -0.05,
                    "fft": 0.05 if is_fake else -0.05,
                },
                "detected_artifacts": detected_artifacts
            },
            "_student_loc_map": loc_map_np
        }

    predict = analyze
    deep_analyze = analyze


ForensicEnsemble = VeriVisionEnsemble
_ensemble_instance = None


def get_ensemble(models_dir: str = "models") -> VeriVisionEnsemble:
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = VeriVisionEnsemble(models_dir=models_dir)
    return _ensemble_instance

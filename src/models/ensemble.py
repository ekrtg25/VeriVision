import sys
import os
import torch
from PIL import Image

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.baseline_cnn import BaselineDetector
from src.models.clip_probe import CLIPProbeDetector
from src.data.transforms import get_transforms

class EnsembleDetector:
    def __init__(self, baseline_weights_path: str, clip_weights_path: str, device: torch.device):
        self.device = device
        
        self.baseline = BaselineDetector(num_classes=1, pretrained=False)
        self.baseline.load_state_dict(
            torch.load(baseline_weights_path, map_location=device, weights_only=True)
        )
        self.baseline.to(device).eval()
        self.baseline_transform = get_transforms(img_size=224, is_train=False)
        
        self.clip = CLIPProbeDetector(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", num_classes=1)
        self.clip.load_state_dict(
            torch.load(clip_weights_path, map_location=device, weights_only=True)
        )
        self.clip.to(device).eval()
        self.clip_transform = self.clip.preprocess

    def predict(
        self, 
        image_path: str, 
        mode: str = "adaptive", 
        threshold: float = 0.40,
        baseline_weight: float = 0.3
    ) -> dict:
        """
        Инференс ансамбля с поддержкой умного взвешивания.
        mode: 'weighted' (статический вес), 'max' (худший случай), 'adaptive' (динамический приоритет CLIP/ResNet)
        threshold: порог отнесения к фейку (default: 0.40)
        """
        img = Image.open(image_path).convert("RGB")
        
        # Инференс Baseline
        img_base = self.baseline_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out_base = self.baseline(img_base)
            prob_base = torch.sigmoid(out_base).item()
            
        # Инференс CLIP
        img_clip = self.clip_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out_clip = self.clip(img_clip)
            prob_clip = torch.sigmoid(out_clip).item()
            
        # Логика взвешивания
        if mode == "max":
            final_prob = max(prob_base, prob_clip)
        elif mode == "adaptive":
            # Если одна модель полностью ослепла (выдает ~0), но другая сомневается (>0.4), 
            # отдаем приоритет сомневающейся модели (обычно это CLIP на OOD данных)
            if abs(prob_base - prob_clip) > 0.35:
                final_prob = max(prob_base, prob_clip)
            else:
                final_prob = (prob_base * baseline_weight) + (prob_clip * (1.0 - baseline_weight))
        else:
            # Классический weighted
            clip_weight = 1.0 - baseline_weight
            final_prob = (prob_base * baseline_weight) + (prob_clip * clip_weight)
        
        return {
            "ensemble_prob": final_prob,
            "baseline_prob": prob_base,
            "clip_prob": prob_clip,
            "prediction": "Fake" if final_prob >= threshold else "Real",
            "mode_used": mode,
            "threshold_used": threshold
        }
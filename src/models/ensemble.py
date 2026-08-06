# src/models/ensemble.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import clip
import os

from src.models.baseline_cnn import BaselineDetector

class EnsembleDetector:
    def __init__(self, baseline_weights_path="models/baseline_weights.pth", clip_weights_path="models/clip_weights.pth", device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. ConvNeXt
        self.baseline = BaselineDetector(pretrained=False).to(self.device)
        if os.path.exists(baseline_weights_path):
            state_dict = torch.load(baseline_weights_path, map_location=self.device, weights_only=True)
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith("model."):
                    new_state_dict[f"model.{k}"] = v
                else:
                    new_state_dict[k] = v
            self.baseline.load_state_dict(new_state_dict)
        self.baseline.eval()

        # 2. CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        
        self.clip_probe = nn.Linear(512, 1).to(self.device)
        if os.path.exists(clip_weights_path):
            clip_state_dict = torch.load(clip_weights_path, map_location=self.device, weights_only=True)
            
            # Убираем префикс "fc." если веса сохранялись через класс NormalizedCLIPProbe
            clean_clip_dict = {}
            for k, v in clip_state_dict.items():
                clean_key = k.replace("fc.", "")
                clean_clip_dict[clean_key] = v
                
            self.clip_probe.load_state_dict(clean_clip_dict)
        self.clip_probe.eval()

        self.cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, mode="uncertainty", threshold=0.35, baseline_weight=0.5, clip_weight=0.5):
        raw_image = Image.open(image_path).convert("RGB")
        
        # Инференс ConvNeXt
        img_cnn = self.cnn_transform(raw_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cnn_logit = self.baseline(img_cnn)
            cnn_prob = torch.sigmoid(cnn_logit).item()

        # Инференс CLIP с L2-нормализацией
        img_clip = self.clip_preprocess(raw_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(img_clip).float()
            features = F.normalize(features, p=2, dim=-1)
            clip_logit = self.clip_probe(features)
            clip_prob = torch.sigmoid(clip_logit).item()

        # Uncertainty-Weighted Fusion
        if mode == "uncertainty":
            cnn_cert = abs(cnn_prob - 0.5)
            clip_cert = abs(clip_prob - 0.5)
            
            total_cert = cnn_cert + clip_cert
            if total_cert < 1e-6:
                ensemble_prob = (cnn_prob + clip_prob) / 2.0
            else:
                ensemble_prob = (cnn_prob * cnn_cert + clip_prob * clip_cert) / total_cert
        elif mode == "max":
            ensemble_prob = max(cnn_prob, clip_prob)
        else:
            ensemble_prob = (cnn_prob * baseline_weight + clip_prob * clip_weight) / (baseline_weight + clip_weight)

        prediction = "Fake" if ensemble_prob >= threshold else "Real"

        return {
            "prediction": prediction,
            "ensemble_prob": ensemble_prob,
            "baseline_prob": cnn_prob,
            "clip_prob": clip_prob,
            "mode_used": mode,
            "threshold_used": threshold
        }
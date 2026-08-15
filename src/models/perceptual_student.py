"""
VeriVision Perceptual Artifact Student Model
Backbone: DINOv2-Base
Heads:
1. Verdict Head (CLS token -> P(AI))
2. Category Multi-Label Head (CLS token -> 7 artifact categories)
3. Localization Head (Patch tokens -> 2D Spatial Heatmap)
"""

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2Config

CATEGORIES = [
    "anatomy",
    "text_ocr",
    "lighting_shadow",
    "texture_repetition",
    "material_plausibility",
    "geometry_perspective",
    "other"
]

class PerceptualStudentDetector(nn.Module):
    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        if pretrained_backbone:
            self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        else:
            config = Dinov2Config.from_pretrained("facebook/dinov2-base")
            self.backbone = Dinov2Model(config)
            
        hidden_dim = self.backbone.config.hidden_size # 768 для DINOv2-Base
        self.num_categories = len(CATEGORIES)

        # 1. Verdict Head (Soft-label AI probability)
        self.verdict_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        # 2. Category Multi-Label Head
        self.category_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_categories)
        )

        # 3. Patch Localization Head (1x1 Conv / Linear mapping per token to 1 channel)
        self.loc_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        # pixel_values: [B, 3, H, W], H, W должны быть кратны 14 (например, 518x518 или 224x224)
        outputs = self.backbone(pixel_values=pixel_values)
        
        cls_token = outputs.last_hidden_state[:, 0, :] # [B, 768]
        patch_tokens = outputs.last_hidden_state[:, 1:, :] # [B, N_patches, 768]

        # Вычисление голов
        verdict_logits = self.verdict_head(cls_token) # [B, 1]
        category_logits = self.category_head(cls_token) # [B, 7]
        
        loc_logits = self.loc_head(patch_tokens) # [B, N_patches, 1]
        
        # Решейп патчей в 2D сетку
        B, N, _ = loc_logits.shape
        grid_size = int(math.sqrt(N))
        loc_map = loc_logits.permute(0, 2, 1).view(B, 1, grid_size, grid_size) # [B, 1, grid, grid]

        return {
            "verdict_logits": verdict_logits.squeeze(-1),
            "category_logits": category_logits,
            "loc_map": loc_map
        }
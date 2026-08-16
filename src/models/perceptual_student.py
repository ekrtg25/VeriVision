import math
from typing import Tuple
import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2Config


class DINOv2ForensicStudent(nn.Module):
    """
    DINOv2-Base Perceptual Student:
    - Global CLS classifier for synthetic detection
    - Dense patch-anomaly head for spatial artifact localization
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if pretrained:
            self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        else:
            config = Dinov2Config.from_pretrained("facebook/dinov2-base")
            self.backbone = Dinov2Model(config)

        hidden_dim = self.backbone.config.hidden_size  # 768

        # 1. Global image classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # 2. Patch-level anomaly head (37x37 grid for 518px)
        self.patch_anomaly = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        logits = self.classifier(cls_token).squeeze(-1)
        patch_scores = self.patch_anomaly(patch_tokens)

        B, N, _ = patch_scores.shape
        grid = int(math.sqrt(N))
        loc_map = patch_scores.permute(0, 2, 1).view(B, 1, grid, grid)

        return logits, loc_map

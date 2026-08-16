import torch
import torch.nn as nn
from transformers import AutoModel


class FineTunedDINO(nn.Module):

  def __init__(self, backbone=None):
    super().__init__()
    self.backbone = (
        backbone
        if backbone is not None
        else AutoModel.from_pretrained("facebook/dinov2-base")
    )

    # Вход 1536 = 768 (CLS) + 768 (Mean Patch)
    self.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(768 * 2, 384),
        nn.BatchNorm1d(384),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(384, 2),
    )

  def forward(self, x):
    outputs = self.backbone(x)
    tokens = outputs.last_hidden_state
    cls_tok = tokens[:, 0]
    patch_mean = tokens[:, 1:].mean(dim=1)
    features = torch.cat([cls_tok, patch_mean], dim=1)
    return self.classifier(features)
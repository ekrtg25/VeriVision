import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
from transformers import Dinov2Model
import matplotlib.pyplot as plt
import cv2


class DINOv2ForensicStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        hidden_dim = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self.patch_anomaly = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        logits = self.classifier(cls_token).squeeze(-1)
        patch_scores = self.patch_anomaly(patch_tokens) # [B, 1369, 1]

        B, N, _ = patch_scores.shape
        grid = int(np.sqrt(N))
        loc_map = patch_scores.permute(0, 2, 1).view(B, 1, grid, grid)
        return logits, loc_map


def generate_heatmap(img_path: str, output_path: str = "artifact_heatmap.jpg"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DINOv2ForensicStudent().to(device)
    model.load_state_dict(torch.load("models/perceptual_student.pth", map_location=device))
    model.eval()

    orig_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(orig_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit, loc_map = model(tensor)
        prob = torch.sigmoid(logit).item()
        
        # Преобразуем 37x37 маску в размер оригинала
        heatmap = torch.sigmoid(loc_map).squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (orig_w, orig_h))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint8 = np.uint8(255 * heatmap)

        # Накладываем JET colormap
        color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        orig_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig_cv, 0.6, color_map, 0.4, 0)

        cv2.imwrite(output_path, overlay)
        print(f"[✓] AI Probability: {prob:.2%}")
        print(f"[✓] Тепловая карта сохранена в: {output_path}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/robust_v1/fake/fake_00925.jpg"
    generate_heatmap(target)

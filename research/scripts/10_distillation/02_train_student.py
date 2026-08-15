"""
Training script for DINOv2 Student Model.
Multi-task loss:
- Soft BCE for Verdict (distillation from teacher)
- Focal Loss for Multi-label Categories
- Pixel/Patch-level BCE for Spatial Localization Heatmap
"""

import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.perceptual_student import PerceptualStudentDetector, CATEGORIES


class DistillationDataset(Dataset):
    def __init__(self, json_path: str, img_size: int = 518):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.img_size = img_size
        self.cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
        self.grid_size = img_size // 14

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.transform(img)

        soft_label = torch.tensor(item["soft_label"], dtype=torch.float32)

        # Категории
        cat_vec = torch.zeros(len(CATEGORIES), dtype=torch.float32)
        # Маска локализации
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for art in item.get("artifacts", []):
            if art["category"] in self.cat_to_idx:
                cat_vec[self.cat_to_idx[art["category"]]] = 1.0

            bbox = art["bbox_normalized"] # [ymin, xmin, ymax, xmax]
            y1 = int(np.clip(bbox[0] * self.grid_size, 0, self.grid_size - 1))
            x1 = int(np.clip(bbox[1] * self.grid_size, 0, self.grid_size - 1))
            y2 = int(np.clip(bbox[2] * self.grid_size, 0, self.grid_size))
            x2 = int(np.clip(bbox[3] * self.grid_size, 0, self.grid_size))
            mask[y1:max(y1+1, y2), x1:max(x1+1, x2)] = 1.0

        return {
            "pixel_values": pixel_values,
            "soft_label": soft_label,
            "category_vec": cat_vec,
            "loc_mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        }


def train_student():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[+] Обучение на устройстве: {device}")

    dataset = DistillationDataset("data/distillation_dataset.json", img_size=518)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = PerceptualStudentDetector(pretrained_backbone=True).to(device)

    # Дифференциальный Learning Rate: Backbone обучается тоньше, головы быстрее
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5, "weight_decay": 0.01},
        {"params": model.verdict_head.parameters(), "lr": 1e-4},
        {"params": model.category_head.parameters(), "lr": 1e-4},
        {"params": model.loc_head.parameters(), "lr": 1e-4}
    ])

    bce_soft = nn.BCEWithLogitsLoss()
    bce_cat = nn.BCEWithLogitsLoss()
    bce_loc = nn.BCEWithLogitsLoss()

    model.train()
    epochs = 5

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            imgs = batch["pixel_values"].to(device)
            soft_labels = batch["soft_label"].to(device)
            cat_targets = batch["category_vec"].to(device)
            loc_targets = batch["loc_mask"].to(device)

            optimizer.zero_grad()
            preds = model(imgs)

            l_verdict = bce_soft(preds["verdict_logits"], soft_labels)
            l_cat = bce_cat(preds["category_logits"], cat_targets)
            l_loc = bce_loc(preds["loc_map"], loc_targets)

            loss = l_verdict + 0.5 * l_cat + 0.5 * l_loc
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] — Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "models/perceptual_student.pth")
    print("[✓] Веса модели сохранены в models/perceptual_student.pth")


if __name__ == "__main__":
    train_student()
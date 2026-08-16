"""
Direct DINOv2 Classifier & Attention Visualizer Training.
Trains directly on data/robust_v1/real and fake folders.
"""

import os
import glob
import math
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import Dinov2Model, Dinov2Config
from tqdm import tqdm


class DirectImageDataset(Dataset):
    def __init__(self, data_dir: str = "data/robust_v1", img_size: int = 518):
        self.img_size = img_size
        self.samples = []
        
        reals = glob.glob(f"{data_dir}/real/*.*")
        fakes = glob.glob(f"{data_dir}/fake/*.*")
        
        for p in reals:
            self.samples.append((p, 0.0))
        for p in fakes:
            self.samples.append((p, 1.0))
            
        print(f"[+] Датасет собран: {len(reals)} real, {len(fakes)} fake (Всего: {len(self.samples)})")

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


class DirectDINOv2Detector(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        else:
            config = Dinov2Config.from_pretrained("facebook/dinov2-base")
            self.backbone = Dinov2Model(config)
            
        hidden_dim = self.backbone.config.hidden_size # 768

        # Детектор на CLS-токене
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Patch-anomaly head
        self.patch_anomaly = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        logits = self.classifier(cls_token).squeeze(-1)
        patch_scores = self.patch_anomaly(patch_tokens)
        
        B, N, _ = patch_scores.shape
        grid = int(math.sqrt(N))
        loc_map = patch_scores.permute(0, 2, 1).view(B, 1, grid, grid)

        return {"logits": logits, "loc_map": loc_map}


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[+] Обучение на устройстве: {device}")

    dataset = DirectImageDataset(data_dir="data/robust_v1", img_size=518)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    model = DirectDINOv2Detector(pretrained=True).to(device)
    
    # Замораживаем первые слои DINOv2 для стабильности, тюним последние блоки и головы
    for param in model.backbone.embeddings.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.encoder.layer[-4:].parameters(), "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
        {"params": model.patch_anomaly.parameters(), "lr": 1e-4}
    ], weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()
    epochs = 4
    
    Path("models").mkdir(exist_ok=True)
    model.train()

    print("\n--- СТАРТ ОБУЧЕНИЯ DINOv2 ---")
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(out["logits"]) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{total_loss/len(loader):.4f}", "acc": f"{correct/total:.2%}"})

    torch.save(model.state_dict(), "models/perceptual_student.pth")
    print(f"\n[✓] Модель успешно обучена и сохранена в models/perceptual_student.pth")


if __name__ == "__main__":
    train()
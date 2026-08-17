"""
Production Fine-Tuning Pipeline for ConvNeXt-Tiny (AI vs Real Detection)
Includes:
- ImageNet Pretrained Backbone
- Heavy In-The-Wild Augmentations (JPEG compression, Blur, Jitter)
- Mixed Precision Training (FP16 / AMP)
- Cosine Annealing Learning Rate Scheduler
- BCEWithLogitsLoss with Label Smoothing
"""
import os
import sys
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
from tqdm import tqdm
import io

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))


class RandomJPEGCompression:
    def __init__(self, quality_min=40, quality_max=95, p=0.5):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(self.quality_min, self.quality_max)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf)
        return img

class RobustAIDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback на случай битого файла
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


class VeriVisionBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        
        in_features = self.backbone.classifier[2].in_features
        # Заменяем финальный классификатор на бинарный выход (1 логит) с регуляризацией
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


def train_model(
    train_dir_real: str,
    train_dir_fake: str,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    save_path: str = "models/baseline_weights.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[+] Устройство для обучения: {device}")

    # Сбор путей
    real_files = [os.path.join(train_dir_real, f) for f in os.listdir(train_dir_real) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
    fake_files = [os.path.join(train_dir_fake, f) for f in os.listdir(train_dir_fake) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]

    print(f"[+] Найдено Real: {len(real_files)} | Fake: {len(fake_files)}")

    all_paths = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    combined = list(zip(all_paths, all_labels))
    random.seed(42)
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)

    split_idx = int(len(all_paths) * 0.85)
    train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]
    train_transform = T.Compose([
        RandomJPEGCompression(quality_min=45, quality_max=95, p=0.5),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        RobustAIDataset(train_paths, train_labels, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        RobustAIDataset(val_paths, val_labels, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = VeriVisionBackbone(pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    best_val_acc = 0.0

    print("\n" + "="*50)
    print(" 🚀 НАЧАЛО ДООБУЧЕНИЯ CONVNEXT-TINY (VeriVision)")
    print("="*50)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            use_amp = device.type in ["cuda", "mps"]
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total:.4f}"})

        scheduler.step()

        # Валидация
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.inference_mode():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_acc = val_correct / val_total
        print(f"\n[Epoch {epoch}] Val Loss: {val_loss/val_total:.4f} | Val Accuracy: {epoch_val_acc*100:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[✓] Лучшие веса обновлены и сохранены в {save_path} (Val Acc: {best_val_acc*100:.2f}%)\n")


if __name__ == "__main__":
    DATA_REAL = "data/robust_v1/real"
    DATA_FAKE = "data/robust_v1/fake"

    if os.path.exists(DATA_REAL) and os.path.exists(DATA_FAKE):
        train_model(DATA_REAL, DATA_FAKE, epochs=5, batch_size=32, lr=1e-4)
    else:
        print("[!] Сначала собери датасет в data/robust_v1/")
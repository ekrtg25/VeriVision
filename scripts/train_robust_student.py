import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel
import pillow_heif

pillow_heif.register_heif_opener()


class RobustCorpusDataset(Dataset):
    def __init__(self, split_dir: Path, transform=None):
        self.samples = []
        self.transform = transform
        exts = {".jpg", ".jpeg", ".png", ".webp"}

        # 0 = Real, 1 = Fake / AI
        phone_real = split_dir / "real" / "phone"
        if phone_real.exists():
            for p in phone_real.glob("*"):
                if p.suffix.lower() in exts:
                    self.samples.append((p, 0, "phone"))

        web_real = split_dir / "real" / "web_hf"
        if web_real.exists():
            for p in web_real.glob("*"):
                if p.suffix.lower() in exts:
                    self.samples.append((p, 0, "web"))

        fake_dir = split_dir / "fake"
        if fake_dir.exists():
            for p in fake_dir.glob("*"):
                if p.suffix.lower() in exts:
                    self.samples.append((p, 1, "fake"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, origin = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), origin


class StudentDINOClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        outputs = self.backbone(x)
        # Извлекаем CLS токен [batch_size, 768]
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Устройство: {device}")

    base_dir = Path("data/training_corpus")
    train_transform, val_transform = get_transforms()

    train_ds = RobustCorpusDataset(base_dir / "train", transform=train_transform)
    val_ds = RobustCorpusDataset(base_dir / "val", transform=val_transform)

    # Вес для фото с телефона 2.5x
    weights = [2.5 if origin == "phone" else 1.0 for _, _, origin in train_ds.samples]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    print(f"[*] Train сэмплов: {len(train_ds)} (Phone weights x2.5)")
    print(f"[*] Val сэмплов:   {len(val_ds)}")

    print("[*] Загрузка DINOv2 из Hugging Face (facebook/dinov2-base)...")
    backbone = AutoModel.from_pretrained("facebook/dinov2-base")
    model = StudentDINOClassifier(backbone).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    epochs = 8
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.classifier.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model.backbone(imgs)
                feats = outputs.last_hidden_state[:, 0]
            preds_logits = model.classifier(feats)

            loss = criterion(preds_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = preds_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            pbar.set_postfix({"Loss": f"{total_loss/total:.4f}", "Acc": f"{correct/total*100:.2f}%"})

        scheduler.step()

        # Валидация
        model.eval()
        v_correct, v_phone_correct, v_phone_total, v_total = 0, 0, 0, 0

        with torch.no_grad():
            for imgs, labels, origins in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds_logits = model(imgs)
                preds = preds_logits.argmax(dim=1)

                v_correct += (preds == labels).sum().item()
                v_total += imgs.size(0)

                for p, l, orig in zip(preds, labels, origins):
                    if orig == "phone":
                        v_phone_total += 1
                        if p == l:
                            v_phone_correct += 1

        val_acc = v_correct / v_total * 100
        phone_acc = (v_phone_correct / v_phone_total * 100) if v_phone_total > 0 else 0

        print(f"[*] Epoch {epoch}: Total Val Acc = {val_acc:.2f}% | Phone Real Acc = {phone_acc:.2f}% ({v_phone_correct}/{v_phone_total})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("models").mkdir(exist_ok=True)
            torch.save(model.classifier.state_dict(), "models/calibrated_head.pth")
            print(f"[+] Сохранена лучшая модель: models/calibrated_head.pth ({val_acc:.2f}%)")

    print(f"\n[+] Обучение завершено. Итоговая точность: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train()
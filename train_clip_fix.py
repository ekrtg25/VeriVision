import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import clip
import torch.nn.functional as F

class NormalizedCLIPProbe(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        # Отказываемся от глубоких слоев с Dropout, классическая L2-Linear Probe работающая лучше всего
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Обязательная L2 Нормализация признаков CLIP
        x = F.normalize(x, p=2, dim=-1)
        return self.fc(x)

class HFCLIPDataset(Dataset):
    def __init__(self, hf_split, preprocess):
        self.split = hf_split
        self.preprocess = preprocess

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        item = self.split[idx]
        image = item['image'].convert("RGB")
        image_tensor = self.preprocess(image)
        label = float(item['label'])
        return image_tensor, torch.tensor(label, dtype=torch.float32)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sys] Training Fixed CLIP Probe on device: {device}")

    # Загружаем CLIP ViT-B/32 (или "ViT-L/14" если позволяет GPU)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # Замораживаем бэкбон полностью
    for param in clip_model.parameters():
        param.requires_grad = False

    print("[data] Loading dataset...")
    dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset")
    train_dataset = HFCLIPDataset(dataset['train'], preprocess=preprocess)
    val_dataset = HFCLIPDataset(dataset['test'], preprocess=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    probe = NormalizedCLIPProbe(input_dim=512).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    # Высокий LR (1e-3) специально для единичного Linear Probe
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    epochs = 5
    best_acc = 0.0

    for epoch in range(epochs):
        probe.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)

            with torch.no_grad():
                features = clip_model.encode_image(images).float()

            optimizer.zero_grad()
            outputs = probe(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)

        # Валидация
        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.unsqueeze(1).to(device)
                features = clip_model.encode_image(images).float()
                preds = (torch.sigmoid(probe(features)) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(probe.state_dict(), "models/clip_weights.pth")
            print(f"[sys] ✨ Saved new best CLIP probe weights! Val Acc: {val_acc*100:.2f}%")

if __name__ == "__main__":
    main()
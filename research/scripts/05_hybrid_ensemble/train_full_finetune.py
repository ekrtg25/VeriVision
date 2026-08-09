import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.baseline_cnn import BaselineDetector

# Хардкорные аугментации (сжатие, размытие, шум)
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.ImageCompression(quality_range=(35, 90), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.GaussNoise(std_range=(0.05, 0.25), p=1.0),
    ], p=0.7),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class CombinedDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 1. Собираем основной датасет из data/raw/train
        raw_real = glob.glob("data/raw/train/real/*.[jJ][pP][gG]") + glob.glob("data/raw/train/real/*.[pP][nN][gG]")
        raw_fake = glob.glob("data/raw/train/fake/*.[jJ][pP][gG]") + glob.glob("data/raw/train/fake/*.[pP][nN][gG]")
        
        # 2. Собираем новые Hard Examples из data/finetune
        ft_real = glob.glob("data/finetune/real/*.[jJ][pP][gG]") + glob.glob("data/finetune/real/*.[pP][nN][gG]")
        ft_fake = glob.glob("data/finetune/fake/*.[jJ][pP][gG]") + glob.glob("data/finetune/fake/*.[pP][nN][gG]")
        
        all_real = raw_real + ft_real
        all_fake = raw_fake + ft_fake
        
        for p in all_real:
            self.image_paths.append(p)
            self.labels.append(0.0)
            
        for p in all_fake:
            self.image_paths.append(p)
            self.labels.append(1.0)
            
        print(f"[data] Loaded REAL: {len(all_real)} (raw: {len(raw_real)}, finetune: {len(ft_real)})")
        print(f"[data] Loaded FAKE: {len(all_fake)} (raw: {len(raw_fake)}, finetune: {len(ft_fake)})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_np = torch.as_tensor(transforms.functional.pil_to_tensor(image)).permute(1, 2, 0).numpy()
        
        if self.transform:
            augmented = self.transform(image=image_np)
            image = augmented['image']

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sys] Device detected: {device}")

    dataset = CombinedDataset(transform=train_transform)
    if len(dataset) == 0:
        print("[error] Данные не найдены!")
        return

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    weights_path = "models/baseline_weights.pth"
    model = BaselineDetector(num_classes=1, pretrained=False)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"[model] Loaded checkpoint from {weights_path}")

    model.to(device)

    # Замораживаем всё по умолчанию
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем layer3, layer4 для глубокого улавливания микротекстур и аномалий
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'layer3'):
            for param in model.backbone.layer3.parameters():
                param.requires_grad = True
        if hasattr(model.backbone, 'layer4'):
            for param in model.backbone.layer4.parameters():
                param.requires_grad = True

    # Размораживаем всю голову классификатора снаружи бэкбона
    for name, child in model.named_children():
        if name != 'backbone':
            for param in child.parameters():
                param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    # Увеличиваем learning rate до 3e-5 для более активной адаптации слоев
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=1e-2)

    epochs = 12  # Увеличиваем количество эпох
    print(f"[train] Starting Full Fine-Tuning for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, batch_labels in dataloader:
            images = images.to(device)
            batch_labels = batch_labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    output_path = "models/baseline_weights.pth"
    torch.save(model.state_dict(), output_path)
    print(f"[sys] Combined training finished! Saved to {output_path}")

if __name__ == "__main__":
    main()
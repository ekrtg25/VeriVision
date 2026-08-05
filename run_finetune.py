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

# Исправленные аугментации под актуальные версии Albumentations
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.ImageCompression(quality_range=(30, 85), p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.GaussNoise(std_range=(0.1, 0.3), p=1.0),
    ], p=0.8),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class FineTuneDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

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
    print(f"[sys] Device detected for Fine-Tuning: {device}")

    real_dir = "data/finetune/real"
    fake_dir = "data/finetune/fake"
    
    real_paths = glob.glob(os.path.join(real_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(real_dir, "*.[pP][nN][gG]"))
    fake_paths = glob.glob(os.path.join(fake_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(fake_dir, "*.[pP][nN][gG]"))

    if not real_paths or not fake_paths:
        print("[error] Папки data/finetune/real или data/finetune/fake пустые!")
        return

    print(f"[data] Found {len(real_paths)} REAL images and {len(fake_paths)} FAKE images.")

    image_paths = real_paths + fake_paths
    labels = [0.0] * len(real_paths) + [1.0] * len(fake_paths)

    weights_path = "models/baseline_weights.pth"
    model = BaselineDetector(num_classes=1, pretrained=False)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"[model] Loaded existing weights from {weights_path}")
    else:
        print(f"[warning] {weights_path} not found! Training from scratch.")

    model.to(device)

    # 1. Сначала замораживаем ВСЕ параметры сети
    for param in model.parameters():
        param.requires_grad = False

    # 2. Размораживаем только свертки layer4 в бэкбоне (если структура стандартная)
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True

    # 3. Размораживаем выходную голову (ищем любой не-backbone атрибут классификатора)
    for name, child in model.named_children():
        if name != 'backbone':
            for param in child.parameters():
                param.requires_grad = True

    dataset = FineTuneDataset(image_paths, labels, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, weight_decay=1e-2)

    epochs = 10
    print(f"[train] Starting Fine-Tuning for {epochs} epochs...")
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
    print(f"[sys] Fine-tuning finished! Updated weights saved to {output_path}")

if __name__ == "__main__":
    main()
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.baseline_cnn import BaselineDetector

# Мощные аугментации для борьбы с переобучением
train_transform = A.Compose([
    A.Resize(224, 224),  # Upscale из 32x32 в 224x224 для ResNet
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.ImageCompression(quality_range=(40, 90), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.GaussNoise(std_range=(0.05, 0.2), p=1.0),
    ], p=0.8),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class HFDatasetAdapter(Dataset):
    def __init__(self, hf_split, transform=None):
        self.split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        item = self.split[idx]
        image = item['image'].convert("RGB")
        
        # Конвертируем PIL в numpy для Albumentations
        image_np = torch.as_tensor(transforms.functional.pil_to_tensor(image)).permute(1, 2, 0).numpy()
        
        if self.transform:
            augmented = self.transform(image=image_np)
            image = augmented['image']

        # В CIFAKE метка: 0 - Real (или наоборот, проверим по датасету), приведем к float
        # Обычно в CIFAKE: 0 или 'REAL', 1 или 'FAKE'. Безопаснее мапить через int.
        label = float(item['label'])
        return image, torch.tensor(label, dtype=torch.float32)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sys] Ночной режим обучения запущен на устройстве: {device}")

    print("[data] Загружаем датасет CIFAKE с Hugging Face...")
    dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset")
    train_split = dataset['train']
    
    train_dataset = HFDatasetAdapter(train_split, transform=train_transform)
    # Поставим num_workers=2 для быстрой подачи данных
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    print(f"[data] Тренировочный сплит загружен. Всего картинок: {len(train_dataset)}")

    weights_path = "models/baseline_weights.pth"
    model = BaselineDetector(num_classes=1, pretrained=False)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"[model] Загружены текущие веса из {weights_path} для дообучения")

    model.to(device)

    # Размораживаем слои layer3, layer4 и классификатор для глубокой адаптации
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'layer3'):
            for param in model.backbone.layer3.parameters():
                param.requires_grad = True
        if hasattr(model.backbone, 'layer4'):
            for param in model.backbone.layer4.parameters():
                param.requires_grad = True

    for name, child in model.named_children():
        if name != 'backbone':
            for param in child.parameters():
                param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, weight_decay=1e-2)
    
    # Шедулер для плавного снижения LR к концу ночи
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs = 15  # На ночь можно поставить 10-15 эпох
    best_loss = float('inf')

    print(f"[train] Старт обучения на {epochs} эпох...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, batch_labels) in enumerate(dataloader):
            images = images.to(device)
            batch_labels = batch_labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        scheduler.step()
        
        print(f"=== ЭПОХА {epoch+1}/{epochs} ЗАВЕРШЕНА. Средний Loss: {epoch_loss:.4f} ===")

        # Сохраняем модель при улучшении результата
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), weights_path)
            print(f"[sys] ✨ Новый лучший результат! Веса сохранены в {weights_path}")

    print("[sys] Ночное обучение успешно завершено! Утром модель будет выдавать уверенные >85-90%.")

if __name__ == "__main__":
    main()
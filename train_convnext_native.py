import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # Отключаем таймауты сетевых проверок Albumentations

import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Аугментации на нативных высоком разрешении без искусственного блюра
train_transform = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class NativeImageDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Берем ТОЛЬКО нативные высококаственные снимки из raw и finetune
        real_files = glob.glob("data/raw/train/real/*.*") + glob.glob("data/finetune/real/*.*")
        fake_files = glob.glob("data/raw/train/fake/*.*") + glob.glob("data/finetune/fake/*.*")

        for p in real_files:
            if p.lower().endswith(('jpg', 'jpeg', 'png')):
                self.image_paths.append(p)
                self.labels.append(0.0)

        for p in fake_files:
            if p.lower().endswith(('jpg', 'jpeg', 'png')):
                self.image_paths.append(p)
                self.labels.append(1.0)

        print(f"[data] Loaded {len(self.image_paths)} native-resolution images (REAL: {len(real_files)}, FAKE: {len(fake_files)})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_np = torch.as_tensor(transforms.functional.pil_to_tensor(image)).permute(1, 2, 0).numpy()
        
        if self.transform:
            augmented = self.transform(image=image_np)
            image = augmented['image']

        return image, torch.tensor(self.labels[idx], dtype=torch.float32)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sys] Training ConvNeXt-Tiny on device: {device}")

    dataset = NativeImageDataset(transform=train_transform)
    if len(dataset) == 0:
        print("[error] No native dataset images found!")
        return

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Загружаем ConvNeXt-Tiny c pretrained весами
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    
    # Заменяем классификационную голову
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, 1)

    # ФАЗА 1: Замораживаем бэкбон, обучаем только голову
    for param in model.features.parameters():
        param.requires_grad = False

    model.to(device)

    # BCE Loss c Label Smoothing для исключения коллапса в 0.0/1.0
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)

    print("[train] Phase 1: Training Classifier Head (Linear Probing)...")
    model.train()
    for epoch in range(3):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.unsqueeze(1).to(device)
            
            # Label smoothing: 0 -> 0.05, 1 -> 0.95
            smoothed_labels = labels * 0.9 + 0.05

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, smoothed_labels)
            loss.backward()
            
            # Gradient Clipping против взрыва градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        print(f"Phase 1 Epoch [{epoch+1}/3] - Loss: {running_loss/len(dataset):.4f}")

    # ФАЗА 2: Размораживаем последнюю stage ConvNeXt с очень маленьким LR
    print("[train] Phase 2: Fine-Tuning Stage 3 features with low LR...")
    for param in model.features[7].parameters():  # Последняя стадия ConvNeXt
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {'params': model.features[7].parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-2)

    for epoch in range(5):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.unsqueeze(1).to(device)
            smoothed_labels = labels * 0.9 + 0.05

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, smoothed_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        print(f"Phase 2 Epoch [{epoch+1}/5] - Loss: {running_loss/len(dataset):.4f}")

    # Сохраняем обновленные веса
    output_path = "models/baseline_weights.pth"
    torch.save(model.state_dict(), output_path)
    print(f"[sys] ConvNeXt model successfully saved to {output_path}!")

if __name__ == "__main__":
    main()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.baseline_cnn import BaselineDetector

def main():
    # 1. Настройка устройства (задействуем GPU на Mac)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sys] Инициализация обучения ConvNeXt на устройстве: {device}")

    # 2. Аугментация для Train и чистый ресайз для Val
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Загрузка датасетов
    train_dir = "data/parveshiiii_ai_vs_real/train"
    val_dir = "data/parveshiiii_ai_vs_real/val"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # ImageFolder по умолчанию сортирует папки по алфавиту: 'fake' = 0, 'real' = 1.
    # Нам нужно наоборот: Real = 0, Fake = 1. Мы инвертируем метки в цикле обучения.
    print(f"[sys] Найдено классов: {train_dataset.class_to_idx}")
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 4. Инициализация модели, лосса и оптимизатора
    # Включаем pretrained=True, чтобы использовать transfer learning, а не учить с нуля
    model = BaselineDetector(pretrained=True).to(device)
    
    # Так как у нас один выходной нейрон, используем Binary Cross Entropy с логитами
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)

    epochs = 3 # 3 эпохи для fine-tuning обычно более чем достаточно
    best_val_loss = float('inf')
    os.makedirs("models", exist_ok=True)
    model_path = "models/baseline_weights.pth"

    # 5. Цикл обучения
    for epoch in range(epochs):
        print(f"\n--- Эпоха {epoch+1}/{epochs} ---")
        
        # ОБУЧЕНИЕ
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar_train = tqdm(train_loader, desc="Train")
        for images, labels in pbar_train:
            images = images.to(device)
            # Инвертируем метки: если fake=0, real=1 -> делаем fake=1, real=0
            labels = 1.0 - labels.float() 
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(outputs) >= 0.5
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            pbar_train.set_postfix({'loss': f"{loss.item():.4f}"})

        train_acc = correct_train / total_train
        
        # ВАЛИДАЦИЯ
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Val"):
                images = images.to(device)
                labels = 1.0 - labels.float()
                labels = labels.unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs) >= 0.5
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")

        # Сохраняем лучшие веса
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"🌟 Новые лучшие веса сохранены в {model_path}!")

    print("\n[sys] Обучение визуального эксперта завершено.")

if __name__ == "__main__":
    main()
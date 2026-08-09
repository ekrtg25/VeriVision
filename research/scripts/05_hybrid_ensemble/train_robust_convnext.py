import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.baseline_cnn import BaselineDetector

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sys] Инициализация РОБАСТНОГО обучения ConvNeXt на: {device}")

    # 1. Агрессивная аугментация (Защита от зубрежки)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = "data/parveshiiii_ai_vs_real/train"
    val_dir = "data/parveshiiii_ai_vs_real/val"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    batch_size = 64 # Можно взять батч больше, так как замороженная сеть ест меньше памяти
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = BaselineDetector(pretrained=True).to(device)

    # =================================================================
    # 2. LINEAR PROBING: ЗАМОРОЗКА БЭКБОНА
    # Замораживаем все слои, чтобы не сломать веса ImageNet
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем только последний слой-классификатор
    # (Обычно содержит 'fc', 'classifier' или 'head' в названии)
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in ['classifier', 'fc', 'head']):
            param.requires_grad = True
    # =================================================================

    # Передаем в оптимизатор только размороженные параметры
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    criterion = nn.BCEWithLogitsLoss()
    # Ставим learning rate чуть выше (1e-3), так как учим только один слой
    optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)

    epochs = 4
    best_val_loss = float('inf')
    os.makedirs("models", exist_ok=True)
    model_path = "models/baseline_weights.pth"

    for epoch in range(epochs):
        print(f"\n--- Эпоха {epoch+1}/{epochs} ---")
        
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        pbar_train = tqdm(train_loader, desc="Train")
        for images, labels in pbar_train:
            images = images.to(device)
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
        
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"🌟 Робастные веса сохранены в {model_path}!")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import ForensicsDataset
from src.data.transforms import get_transforms
from src.models.baseline_cnn import BaselineDetector

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        # BCEWithLogitsLoss ожидает float логиты и целевые метки формы (N, 1)
        labels = labels.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        # Считаем accuracy (сигмоида > 0.5 эквивалентна логиту > 0)
        preds = (outputs > 0).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc


def main():
    # Настройка устройства: Apple Silicon (MPS) -> CUDA -> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используем Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используем NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Используем CPU")

    # 1. Датасеты и даталоадеры
    train_transform = get_transforms(img_size=224, is_train=True)
    val_transform = get_transforms(img_size=224, is_train=False)

    train_dataset = ForensicsDataset(root_dir="data/raw/train", transform=train_transform)
    val_dataset = ForensicsDataset(root_dir="data/raw/val", transform=val_transform)

    if len(train_dataset) == 0:
        print("❌ Ошибка: В data/raw/train/ нет изображений! Добавьте картинки в real/ и fake/.")
        return

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 2. Модель, Loss и Оптимизатор
    model = BaselineDetector(num_classes=1, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 3. Тестовый цикл на 3 эпохи
    print("\n🚀 Запуск Sanity Check (3 эпохи)...")
    for epoch in range(1, 4):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Эпоха {epoch}/3 | Train Loss: {loss:.4f} | Train Acc: {acc * 100:.1f}%")

    print("\n✅ Пайплайн данных и baseline-модели работает корректно!")

if __name__ == "__main__":
    main()
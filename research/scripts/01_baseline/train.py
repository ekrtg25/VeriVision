import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import ForensicsDataset
from src.data.transforms import get_transforms
from src.models.baseline_cnn import BaselineDetector
from src.models.frequency_model import FrequencyDetector
from src.models.clip_probe import CLIPProbeDetector
from src.evaluation.metrics import compute_metrics


def get_model(model_type: str, device: torch.device):
    """Фабрика для создания нужной модели по флагу."""
    if model_type == "baseline":
        return BaselineDetector(num_classes=1, pretrained=True).to(device)
    elif model_type == "frequency":
        return FrequencyDetector(num_classes=1, pretrained=True).to(device)
    elif model_type == "clip":
        return CLIPProbeDetector(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", num_classes=1).to(device)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")


def run_epoch(model, dataloader, criterion, optimizer, device, is_train: bool = True):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_targets = []
    all_probs = []

    # Если мы в режиме валидации, отключаем градиенты
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Получаем вероятности через Сигмоиду и ЯВНО отвязываем от графа (detach)
            probs = torch.sigmoid(outputs).detach()

            all_targets.append(labels.detach().cpu())
            all_probs.append(probs.cpu())

    total_samples = len(dataloader.dataset)
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0

    # Конкатенируем и переводим в numpy без ошибок
    y_true = torch.cat(all_targets).detach().numpy()
    y_probs = torch.cat(all_probs).detach().numpy()

    metrics = compute_metrics(y_true, y_probs)
    metrics["loss"] = epoch_loss

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Обучение моделей Synthetic Media Forensics")
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "frequency", "clip"],
                        help="Выбор архитектуры: baseline | frequency | clip")
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    args = parser.parse_args()

    # Настройка устройства
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n🚀 Запуск эксперимента с моделью: [{args.model_type.upper()}] на устройстве [{device}]")

    # Для CLIP используем его родную трансформу препроцессинга
    if args.model_type == "clip":
        temp_model = CLIPProbeDetector()
        train_transform = temp_model.preprocess
        val_transform = temp_model.preprocess
    else:
        train_transform = get_transforms(img_size=224, is_train=True)
        val_transform = get_transforms(img_size=224, is_train=False)

    train_dataset = ForensicsDataset(root_dir="data/raw/train", transform=train_transform)
    val_dataset = ForensicsDataset(root_dir="data/raw/val", transform=val_transform)

    if len(train_dataset) == 0:
        print("❌ Ошибка: Нет изображений в data/raw/train")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model_type, device)
    criterion = nn.BCEWithLogitsLoss()
    
    # Для CLIP оптимизируем ТОЛЬКО обучаемые параметры (Linear Probe)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_m = run_epoch(model, val_loader, criterion, None, device, is_train=False)

        print(f"Эпоха {epoch}/{args.epochs} | "
              f"Train Loss: {train_m['loss']:.4f}, Acc: {train_m['accuracy']*100:.1f}% | "
              f"Val Loss: {val_m['loss']:.4f}, Acc: {val_m['accuracy']*100:.1f}%, ROC-AUC: {val_m['roc_auc']:.4f}")

    # === ДОБАВЛЕННЫЙ БЛОК СОХРАНЕНИЯ ВЕСОВ ===
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{args.model_type}_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Веса модели успешно сохранены в: {save_path}")
    # ==========================================

    print(f"✅ Эксперимент для [{args.model_type}] успешно завершен!\n")


if __name__ == "__main__":
    main()
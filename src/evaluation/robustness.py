import argparse
import io
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data.dataset import ForensicsDataset
from train import get_model 

def apply_jpeg(img, quality):
    """Вспомогательная функция для симуляции JPEG-сжатия через память (без сохранения на диск)."""
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def get_degradation_transforms(degradation_type: str, severity: int = 1):
    """Генерирует пайплайн искажений."""
    if degradation_type == "jpeg":
        quality = max(10, 100 - (severity * 20)) 
        return T.Compose([
            T.Resize((224, 224)),
            T.Lambda(lambda img: apply_jpeg(img, quality)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif degradation_type == "blur":
        kernel_size = 1 + (severity * 2) 
        return T.Compose([
            T.Resize((224, 224)),
            T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    elif degradation_type == "noise":
        noise_std = severity * 0.05
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: x + noise_std * torch.randn_like(x)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Неизвестный тип деградации!")

def evaluate_robustness(model, device, val_dir="data/raw/val"):
    model.eval()
    degradations = ["jpeg", "blur", "noise"]
    severities = [1, 2, 3]

    print(f"\n🛡️ Запуск тестов на устройстве: {device}")
    
    for deg in degradations:
        print(f"\n--- Искажение: {deg.upper()} ---")
        for sev in severities:
            transform = get_degradation_transforms(deg, sev)
            dataset = ForensicsDataset(root_dir=val_dir, transform=transform)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

            all_targets = []
            all_probs = []

            with torch.no_grad():
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device).float()
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    
                    all_targets.extend(labels.cpu().numpy())
                    all_probs.extend(probs)

            preds = [1 if p >= 0.5 else 0 for p in all_probs]
            acc = accuracy_score(all_targets, preds)
            auc = roc_auc_score(all_targets, all_probs)
            print(f"Уровень {sev} | Acc: {acc*100:.1f}% | ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "frequency", "clip"])
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"📥 Загрузка сохраненных весов для {args.model_type.upper()}...")
    
    model = get_model(args.model_type, device)
    weights_path = f"models/{args.model_type}_weights.pth"
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"❌ Файл весов не найден: {weights_path}.")
        exit(1)
    
    evaluate_robustness(model, device, val_dir="data/raw/val")
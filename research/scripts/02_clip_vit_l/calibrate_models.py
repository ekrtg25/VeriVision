import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import glob
import clip
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.models.baseline_cnn import BaselineDetector

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Аугментации ТОЛЬКО для ресайза/нормализации (без искажений, это же валидация)
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class ValidationDataset(Dataset):
    def __init__(self):
        self.image_paths = []
        self.labels = []
        
        # Укажи тут пути до твоей ВАЛИДАЦИОННОЙ (тестовой) выборки
        real_files = glob.glob("data/raw/val/real/*.*") 
        fake_files = glob.glob("data/raw/val/fake/*.*")

        for p in real_files:
            if p.lower().endswith(('jpg', 'jpeg', 'png')):
                self.image_paths.append(p)
                self.labels.append(0.0)

        for p in fake_files:
            if p.lower().endswith(('jpg', 'jpeg', 'png')):
                self.image_paths.append(p)
                self.labels.append(1.0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Для ConvNeXt
        image_np = torch.as_tensor(transforms.functional.pil_to_tensor(image)).permute(1, 2, 0).numpy()
        cnn_input = val_transform(image=image_np)['image']
        
        return cnn_input, img_path, torch.tensor(self.labels[idx], dtype=torch.float32)

class ModelCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        # Инициализируем температуру единицей
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def optimize_temperature(logits, labels, model_name):
    calibrator = ModelCalibrator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([calibrator.temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = criterion(calibrator(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    
    final_temp = calibrator.temperature.item()
    print(f"[{model_name}] Optimal Temperature: {final_temp:.4f}")
    return final_temp

def main():
    print("[sys] Loading models for calibration...")
    
    # Загрузка ConvNeXt
    cnn_model = BaselineDetector(pretrained=False).to(device)
    cnn_state = torch.load("models/baseline_weights.pth", map_location=device)
    cnn_state = {k.replace("model.", "") if not k.startswith("model.") else k: v for k, v in cnn_state.items()}
    cnn_model.load_state_dict(cnn_state, strict=False)
    cnn_model.eval()

    # Загрузка CLIP (ОБНОВЛЕНО НА ViT-L/14)
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    clip_probe = nn.Linear(768, 1).to(device) # Изменена размерность с 512 на 768
    clip_state = torch.load("models/clip_vit_l_weights.pth", map_location=device) # Обновлен путь к весам
    clip_state = {k.replace("fc.", ""): v for k, v in clip_state.items()}
    clip_probe.load_state_dict(clip_state, strict=False)
    clip_probe.eval()

    dataset = ValidationDataset()
    if len(dataset) == 0:
        print("[error] Validation dataset not found! Check paths.")
        return
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_cnn_logits = []
    all_clip_logits = []
    all_labels = []

    print("[sys] Collecting logits on validation set...")
    with torch.no_grad():
        for cnn_inputs, paths, labels in tqdm(dataloader):
            cnn_inputs = cnn_inputs.to(device)
            labels = labels.unsqueeze(1).to(device)
            all_labels.append(labels)

            # ConvNeXt logits
            cnn_logits = cnn_model(cnn_inputs)
            all_cnn_logits.append(cnn_logits)

            # CLIP logits (надо прогнать через preprocess по путям, так как аугментации разные)
            clip_inputs = torch.stack([clip_preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
            features = clip_model.encode_image(clip_inputs).float()
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            clip_logits = clip_probe(features)
            all_clip_logits.append(clip_logits)

    all_cnn_logits = torch.cat(all_cnn_logits)
    all_clip_logits = torch.cat(all_clip_logits)
    all_labels = torch.cat(all_labels)

    # Оптимизация температуры
    print("\n[sys] Optimizing Temperature Scaling...")
    t_cnn = optimize_temperature(all_cnn_logits, all_labels, "ConvNeXt")
    t_clip = optimize_temperature(all_clip_logits, all_labels, "CLIP")

    print("\n[sys] Save these values in src/models/ensemble.py!")

if __name__ == "__main__":
    main()
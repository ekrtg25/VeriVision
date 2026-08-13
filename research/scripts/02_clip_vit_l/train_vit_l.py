import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import clip
from tqdm import tqdm
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class TrainDataset(Dataset):
    def __init__(self, root_dir, preprocess):
        self.image_paths = []
        self.labels = []
        self.preprocess = preprocess
        
        real_files = glob.glob(f"{root_dir}/real/*.*")
        fake_files = glob.glob(f"{root_dir}/fake/*.*")
        
        valid_ext = ('jpg', 'jpeg', 'png')
        for p in real_files:
            if p.lower().endswith(valid_ext):
                self.image_paths.append(p)
                self.labels.append(0.0)
                
        for p in fake_files:
            if p.lower().endswith(valid_ext):
                self.image_paths.append(p)
                self.labels.append(1.0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(image)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.float32)

def main():
    print("[sys] Loading heavy ViT-L/14 architecture...")
    # Загружаем мощную модель
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval() # Бэкбон заморожен
    
    # Новый классификатор под размерность 768
    probe = nn.Linear(768, 1).to(device)
    
    print("[sys] Preparing DataLoaders...")
    train_dataset = TrainDataset("data/raw/train", preprocess)
    
    if len(train_dataset) == 0:
        print("[error] No training data found! Check 'data/raw/train' path.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 5
    print(f"[sys] Starting Linear Probe Training on {len(train_dataset)} images...")
    
    for epoch in range(epochs):
        probe.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)
            
            # Извлекаем фичи через ViT-L (без расчета градиентов для тяжелой части)
            with torch.no_grad():
                features = model.encode_image(inputs).float()
                # Обязательная L2-нормализация
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            
            optimizer.zero_grad()
            outputs = probe(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({"Loss": f"{running_loss/(total/32):.4f}", "Acc": f"{correct/total:.4f}"})
            
    os.makedirs("models", exist_ok=True)
    save_path = "models/clip_vit_l_weights.pth"
    torch.save(probe.state_dict(), save_path)
    print(f"\n✅ ViT-L/14 Probe saved to {save_path}")

if __name__ == "__main__":
    main()
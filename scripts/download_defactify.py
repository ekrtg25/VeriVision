import os
from datasets import load_dataset
from tqdm import tqdm

def save_split(dataset_split, split_name, base_dir="data/defactify"):
    real_dir = os.path.join(base_dir, split_name, "real")
    fake_dir = os.path.join(base_dir, split_name, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print(f"[sys] Сохранение сплита {split_name} ({len(dataset_split)} изображений)...")
    for idx, item in enumerate(tqdm(dataset_split, desc=f"Processing {split_name}")):
        img = item["Image"].convert("RGB")
        label = item["Label_A"] # 0 = Real, 1 = AI-Generated
        
        if label == 0:
            img.save(os.path.join(real_dir, f"real_{idx:06d}.jpg"), "JPEG")
        else:
            img.save(os.path.join(fake_dir, f"fake_{idx:06d}.jpg"), "JPEG")

def main():
    print("[sys] Загрузка Defactify_Image_Dataset с Hugging Face...")
    # Загружаем датасет целиком (потребует места на диске)
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")
    
    # Сохраняем Train и Validation
    save_split(ds["train"], "train")
    save_split(ds["validation"], "val")
    
    print("\n✅ Готово! Датасет Defactify загружен в 'data/defactify/'.")

if __name__ == "__main__":
    main()
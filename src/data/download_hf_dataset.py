import os
import shutil
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def download_and_distribute():
    output_dir = Path("data/raw")
    
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        for cls in ["real", "fake"]:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    print("📦 Скачиваем датасет целиком (появится стандартный прогресс-бар Hugging Face)...")
    
    ds_full = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")
    
    ds = ds_full["train"]

    print(f"\n✅ Датасет загружен. Всего изображений в архиве: {len(ds)}")
    print("⏳ Распределяем нужный объем по папкам (800 на Train, 200 на Val)...")

    counts = {
        "train": {"real": 0, "fake": 0},
        "val": {"real": 0, "fake": 0}
    }
    
    num_train = 400
    num_val = 100

    pbar = tqdm(total=(num_train + num_val) * 2, desc="Сохранение")

    # 2. Перебираем скачанный датасет и раскладываем по папкам
    for item in ds:
        img: Image.Image = item.get("image") or item.get("img")
        if img is None:
            continue

        label = item.get("label", item.get("labels", item.get("target")))
        cls_name = "fake" if label in [1, "1", "fake", "AI", "ai"] else "real"

        if counts["train"][cls_name] < num_train:
            idx = counts["train"][cls_name]
            save_path = output_dir / "train" / cls_name / f"{cls_name}_{idx:04d}.jpg"
            img.convert("RGB").save(save_path, "JPEG")
            counts["train"][cls_name] += 1
            pbar.update(1)
        elif counts["val"][cls_name] < num_val:
            idx = counts["val"][cls_name]
            save_path = output_dir / "val" / cls_name / f"{cls_name}_{idx:04d}.jpg"
            img.convert("RGB").save(save_path, "JPEG")
            counts["val"][cls_name] += 1
            pbar.update(1)

        train_done = (counts["train"]["real"] >= num_train and counts["train"]["fake"] >= num_train)
        val_done = (counts["val"]["real"] >= num_val and counts["val"]["fake"] >= num_val)

        if train_done and val_done:
            break
            
    pbar.close()

    print("\n🎉 Готово! Идеальная структура собрана:")
    print(f"  Train: {counts['train']['real']} Real, {counts['train']['fake']} Fake")
    print(f"  Val:   {counts['val']['real']} Real, {counts['val']['fake']} Fake")

if __name__ == "__main__":
    download_and_distribute()
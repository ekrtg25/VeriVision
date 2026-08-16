"""
Selective Downloader for Kaggle dataset:
Downloads only a subset of files without pulling 52 GB.
"""

import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import kagglehub

TARGET_DIR = Path("data/training_corpus")
MIN_RES = 400
random.seed(42)

def save_image_safe(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as img:
            if img.width < MIN_RES or img.height < MIN_RES:
                return False
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(dst, "JPEG", quality=92)
        return True
    except Exception:
        return False

def main():
    print("[*] Загрузка компактного среза с Kaggle...")
    
    # Скачиваем компактный проверенный срез (476 MB вместо 52 GB)
    path = kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images")
    src_dir = Path(path)
    print(f"[+] Распаковано во временный кэш: {src_dir}")

    train_fake = TARGET_DIR / "train" / "fake"
    val_fake = TARGET_DIR / "val" / "fake"
    train_real = TARGET_DIR / "train" / "real" / "web_hf"
    val_real = TARGET_DIR / "val" / "real" / "web_hf"

    fake_idx = len(list(train_fake.glob("*.jpg"))) + len(list(val_fake.glob("*.jpg")))
    real_idx = len(list(train_real.glob("*.jpg"))) + len(list(val_real.glob("*.jpg")))

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in src_dir.rglob("*") if p.suffix.lower() in valid_exts]

    added_fake, added_real = 0, 0

    for p in tqdm(files, desc="Добавление сэмплов в корпус"):
        parent_name = str(p.parent).lower()
        is_fake = any(k in parent_name for k in ["ai", "fake", "gen", "synth"])
        is_real = any(k in parent_name for k in ["real", "nature", "photo", "human"])

        if not is_fake and not is_real:
            continue

        in_val = random.random() < 0.15

        if is_fake:
            folder = val_fake if in_val else train_fake
            out_p = folder / f"kg_fake_{fake_idx:05d}.jpg"
            if save_image_safe(p, out_p):
                fake_idx += 1
                added_fake += 1
        elif is_real:
            folder = val_real if in_val else train_real
            out_p = folder / f"kg_real_{real_idx:05d}.jpg"
            if save_image_safe(p, out_p):
                real_idx += 1
                added_real += 1

    print("\n" + "=" * 55)
    print(f"[+] Успешно добавлено без скачивания 52 ГБ:")
    print(f"    - AI / Fake: {added_fake} шт")
    print(f"    - Real Web:  {added_real} шт")
    print("=" * 55)

    print("\n[i] Текущее состояние датасета:")
    for split in ["train", "val"]:
        for cls in ["fake", "real/web_hf", "real/phone"]:
            f = TARGET_DIR / split / cls
            print(f"    {split}/{cls}: {len(list(f.glob('*.jpg')))} шт")

if __name__ == "__main__":
    main()
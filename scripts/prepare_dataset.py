"""
Dataset Downloader & Sampler for VeriVision:
Combines bitmind and hayc0205 datasets for high-res AI vs Real diversity.
Distributes samples randomly into Train (85%) and Val (15%).
"""

import sys
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# Целевые объемы
TARGET_FAKE = 3500
TARGET_REAL_WEB = 2200
MIN_RESOLUTION = 400

random.seed(42)


def save_image_safe(img: Image.Image, save_path: Path, min_res: int = MIN_RESOLUTION) -> bool:
    try:
        if img.width < min_res or img.height < min_res:
            return False
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(save_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


def route_and_save(img: Image.Image, is_fake: bool, counts: dict, base_dir: Path) -> bool:
    # 85% отправляем в train, 15% в val
    in_val = random.random() < 0.15

    if is_fake:
        if counts["fake_total"] >= TARGET_FAKE:
            return False
        folder = base_dir / ("val" if in_val else "train") / "fake"
        path = folder / f"fake_{counts['fake_total']:05d}.jpg"
        if save_image_safe(img, path):
            counts["fake_total"] += 1
            if in_val:
                counts["val_fake"] += 1
            else:
                counts["train_fake"] += 1
            return True
    else:
        if counts["real_total"] >= TARGET_REAL_WEB:
            return False
        folder = base_dir / ("val" if in_val else "train") / "real" / "web_hf"
        path = folder / f"real_{counts['real_total']:05d}.jpg"
        if save_image_safe(img, path):
            counts["real_total"] += 1
            if in_val:
                counts["val_real"] += 1
            else:
                counts["train_real"] += 1
            return True
    return False


def stream_dataset(repo_name: str, base_dir: Path, counts: dict, pbar: tqdm):
    try:
        ds = load_dataset(repo_name, split="train", streaming=True)
    except Exception as e:
        print(f"\n[!] Ошибка стриминга {repo_name}: {e}")
        return

    for item in ds:
        label = item.get("label")
        img = item.get("image")
        if img is None or label is None:
            continue

        # В BitMind: 0=AI, 1=Real. В стандартных: 1=AI, 0=Real.
        if "bitmind" in repo_name:
            is_fake = (label == 0 or str(label).lower() in ["fake", "ai", "0"])
        else:
            is_fake = (label == 1 or str(label).lower() in ["fake", "ai", "1", "synthetic"])

        if route_and_save(img, is_fake, counts, base_dir):
            pbar.update(1)

        if counts["fake_total"] >= TARGET_FAKE and counts["real_total"] >= TARGET_REAL_WEB:
            break


def main():
    target_dir = Path("data/training_corpus")
    print(f"[*] Целевой каталог: {target_dir.resolve()}")

    (target_dir / "train" / "real" / "phone").mkdir(parents=True, exist_ok=True)
    (target_dir / "train" / "real" / "web_hf").mkdir(parents=True, exist_ok=True)
    (target_dir / "train" / "fake").mkdir(parents=True, exist_ok=True)
    (target_dir / "val" / "real" / "phone").mkdir(parents=True, exist_ok=True)
    (target_dir / "val" / "real" / "web_hf").mkdir(parents=True, exist_ok=True)
    (target_dir / "val" / "fake").mkdir(parents=True, exist_ok=True)

    counts = {
        "fake_total": 0, "train_fake": 0, "val_fake": 0,
        "real_total": 0, "train_real": 0, "val_real": 0,
    }

    total_target = TARGET_FAKE + TARGET_REAL_WEB
    pbar = tqdm(total=total_target, desc="Сборка сбалансированного корпуса (>=400px)")

    # 1. Источник: BitMind
    print("\n[*] [1/2] Загрузка из bitmind/AI-vs-Real-Dataset-Images-Proper...")
    stream_dataset("bitmind/AI-vs-Real-Dataset-Images-Proper", target_dir, counts, pbar)

    # 2. Источник для добора: hayc0205
    if counts["fake_total"] < TARGET_FAKE or counts["real_total"] < TARGET_REAL_WEB:
        print("\n[*] [2/2] Добор разнородных данных из hayc0205/synthetic_image_detection_dataset...")
        stream_dataset("hayc0205/synthetic_image_detection_dataset", target_dir, counts, pbar)

    pbar.close()

    print("\n" + "=" * 60)
    print("[+] Итоговая структура готова:")
    print(f"    - Fake AI:     Train = {counts['train_fake']}, Val = {counts['val_fake']}")
    print(f"    - Real Web:    Train = {counts['train_real']}, Val = {counts['val_real']}")
    print("=" * 60)
    print("\n[i] Теперь скопируй фотографии с телефона:")
    print(f"    1. ~660 шт в: {target_dir}/train/real/phone/")
    print(f"    2. ~70 шт в:  {target_dir}/val/real/phone/")


if __name__ == "__main__":
    main()
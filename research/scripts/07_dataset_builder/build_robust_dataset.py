"""
VeriVision Robust Dataset Builder v1.0
Streams curated samples from GenImage / Hugging Face and applies
in-the-wild messenger augmentations (Double JPEG, Resize, Noise).
"""

import io
import os
import random
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def apply_messenger_compression(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > 1280:
        scale = 1280.0 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

    quality = random.randint(40, 75)
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def build_curated_dataset(
    output_dir: str = "data/robust_v1", samples_per_class: int = 1500
):
    real_dir = Path(output_dir) / "real"
    fake_dir = Path(output_dir) / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[+] Инициализация стриминга GenImage (Цель: {samples_per_class} Real / {samples_per_class} Fake)..."
    )

    dataset = load_dataset(
        "TheKernel01/Tiny-GenImage", split="train", streaming=True
    )

    real_count = 0
    fake_count = 0

    pbar = tqdm(total=samples_per_class * 2, desc="Building Robust Dataset")

    for sample in dataset:
        img = sample["image"].convert("RGB")
        label = sample["label"]  # 0: Real, 1: Fake
        generator = sample.get("generator", -1)

        if random.random() < 0.35:
            img = apply_messenger_compression(img)

        if label == 0 and real_count < samples_per_class:
            save_path = real_dir / f"real_{real_count:05d}.jpg"
            img.save(save_path, "JPEG", quality=90)
            real_count += 1
            pbar.update(1)

        elif label == 1 and fake_count < samples_per_class:
            save_path = fake_dir / f"fake_{fake_count:05d}.jpg"
            img.save(save_path, "JPEG", quality=90)
            fake_count += 1
            pbar.update(1)

        if (
            real_count >= samples_per_class
            and fake_count >= samples_per_class
        ):
            break

    pbar.close()
    print(f"\n[✓] Датасет успешно собран в: {output_dir}")
    print(f" • Real images: {real_count}")
    print(f" • Fake images: {fake_count}")


if __name__ == "__main__":
    build_curated_dataset()
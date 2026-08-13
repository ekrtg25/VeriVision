import os
import requests
from datasets import load_dataset
from tqdm import tqdm
from io import BytesIO
from PIL import Image

def create_dirs(base_path):
    os.makedirs(os.path.join(base_path, "real"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "fake"), exist_ok=True)

def safe_save(img, path, quality=95):
    try:
        img.convert("RGB").save(path, quality=quality)
        return True
    except Exception as e:
        return False

def fetch_real_unsplash_images(target_dir, max_samples=100):
    print(f"\n[sys] Fetching {max_samples} REAL photos from Unsplash...")
    count = 0
    with tqdm(total=max_samples, desc="Downloading Real Photos") as pbar:
        while count < max_samples:
            try:
                # Получаем случайную высококачественную фотографию
                response = requests.get("https://picsum.photos/512", timeout=5)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img_path = os.path.join(target_dir, f"real_{count}.jpg")
                    if safe_save(img, img_path):
                        count += 1
                        pbar.update(1)
            except Exception:
                continue

def fetch_mj6_images(target_dir, max_samples=100):
    print(f"\n[sys] Fetching {max_samples} FAKE photos (Midjourney v6)...")
    try:
        ds_mj = load_dataset("brivangl/midjourney-v6-llava", split="train", streaming=True)
        mj_iter = iter(ds_mj)
        count = 0
        with tqdm(total=max_samples, desc="Downloading MJv6") as pbar:
            while count < max_samples:
                item = next(mj_iter)
                img_path = os.path.join(target_dir, f"mj6_{count}.jpg")
                if safe_save(item["image"], img_path):
                    count += 1
                    pbar.update(1)
    except Exception as e:
        print(f"[error] Midjourney v6 fetch failed: {e}")

def main():
    print("[sys] Structuring OOD directories...")
    base_ood = "data/ood"
    
    midjourney_dir = os.path.join(base_ood, "midjourney_v6")
    create_dirs(midjourney_dir)
    
    fetch_real_unsplash_images(os.path.join(midjourney_dir, "real"), 100)
    fetch_mj6_images(os.path.join(midjourney_dir, "fake"), 100)

    print("\n✅ OOD Datasets successfully generated!")
    print("Next step: Update evaluate_ood.py to test only Midjourney V6 and run it.")

if __name__ == "__main__":
    main()
"""
Bulletproof Teacher Labeling Pipeline via Local Qwen2.5-VL.
Uses format='json' with fallback parsing & instant per-item disk flushing.
"""

import os
import io
import json
import glob
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm
import ollama

MODEL_NAME = "qwen2.5vl:7b"
MAX_WORKERS = 2

PROMPT = """You are a senior digital forensics expert.
Analyze this image for generative AI visual artifacts:
- anatomy (extra fingers, unnatural eyes, distorted limbs)
- text_ocr (unreadable glyphs, nonsense text)
- lighting_shadow (conflicting light angles, impossible shadows)
- texture_repetition (waxy skin, repeating background patterns)
- geometry_perspective (broken vanishing lines)

Output strictly a JSON object with this exact structure:
{
  "verdict": "AI_GENERATED" or "REAL_PHOTO",
  "confidence": 0.85,
  "artifacts": [
    {
      "category": "anatomy",
      "location_description": "short description",
      "severity": 0.7,
      "bbox_normalized": [0.1, 0.2, 0.3, 0.4]
    }
  ]
}
If no artifacts are found, return "artifacts": [].
"""


def process_image(img_path: str) -> Optional[dict]:
    try:
        # Проверяем и ресайзим до 768px для ускорения
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail((768, 768), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                    "images": [img_bytes]
                }
            ],
            format="json",
            options={"temperature": 0.1}
        )

        content = response["message"]["content"]
        data = json.loads(content)

        verdict = data.get("verdict", "REAL_PHOTO")
        conf = float(data.get("confidence", 0.5))
        soft_label = conf if verdict == "AI_GENERATED" else (1.0 - conf)

        # Валидация списка артефактов
        clean_artifacts = []
        for art in data.get("artifacts", []):
            bbox = art.get("bbox_normalized") or art.get("bbox") or [0.0, 0.0, 1.0, 1.0]
            clean_artifacts.append({
                "category": str(art.get("category", "other")),
                "location_description": str(art.get("location_description", art.get("description", ""))),
                "severity": float(art.get("severity", 0.5)),
                "bbox_normalized": [float(x) for x in bbox[:4]]
            })

        return {
            "image_path": img_path,
            "soft_label": round(soft_label, 4),
            "verdict": verdict,
            "artifacts": clean_artifacts
        }
    except Exception as e:
        print(f"\n[Ошибка {img_path}]: {e}")
        return None


def run_labeling(
    data_dir: str = "data/robust_v1",
    output_json: str = "data/distillation_dataset.json"
):
    Path("data").mkdir(parents=True, exist_ok=True)

    existing_dataset = []
    processed_paths = set()
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_dataset = json.load(f)
                processed_paths = {item["image_path"] for item in existing_dataset}
                print(f"[i] Пропускаем {len(processed_paths)} уже обработанных файлов.")
        except Exception:
            existing_dataset = []

    all_paths = glob.glob(f"{data_dir}/real/*.*") + glob.glob(f"{data_dir}/fake/*.*")
    target_paths = [p for p in all_paths if p not in processed_paths]

    print(f"[+] Всего к разметке: {len(target_paths)} файлов.")
    dataset = existing_dataset

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_image, p): p for p in target_paths}

        for future in tqdm(as_completed(futures), total=len(target_paths), desc="Labeling"):
            res = future.result()
            if res:
                dataset.append(res)
                # Сохраняем и сбрасываем буфер на диск на КАЖДОЙ успешной картинке
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Завершено! Сохранено {len(dataset)} объектов в {output_json}")


if __name__ == "__main__":
    run_labeling()
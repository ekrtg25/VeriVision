"""Quick Pre-flight Test on 1 Real and 1 Fake image."""

import glob
import io
import json
import random
from PIL import Image
import ollama

MODEL_NAME = "qwen2.5vl:7b"
TEMPERATURE = 0.35

SYSTEM_PROMPT = """You are a world-class digital forensic scientist inspecting generative AI artifacts (Midjourney v6, Flux.1, Stable Diffusion XL, Recraft).

Conduct an exhaustive forensic audit analyzing:
1. Micro-textures: skin pores, organic hair follicles, fabric weave consistency, sub-surface scattering.
2. Lighting & Reflections: corneal specular highlights, light-source triangulation, shadow edge-falloff.
3. Anatomy & Biometrics: limb joints, dental anatomy, ear cartilage symmetry, finger digits and nails.
4. Geometry & Perspective: vanishing lines, non-Euclidean perspective flaws, melting parallel contours.
5. Physics Plausibility: gravity violations, unattached floating elements, liquid meniscus behavior.

Execute in 2 steps:
Step 1: Write a meticulous forensic critique in 'forensic_reasoning'.
Step 2: Generate final structured verdict and locate artifact bounding boxes.

Output strictly a JSON object matching this schema:
{
  "forensic_reasoning": "Exhaustive step-by-step critique of textures, illumination, and geometry...",
  "verdict": "AI_GENERATED" or "REAL_PHOTO",
  "confidence": 0.92,
  "artifacts": [
    {
      "category": "anatomy" | "text_ocr" | "lighting_shadow" | "texture_repetition" | "material_plausibility" | "geometry_perspective" | "other",
      "location_description": "Specific region description",
      "severity": 0.85,
      "bbox_normalized": [ymin, xmin, ymax, xmax]
    }
  ]
}
If no artifacts exist, set "artifacts": [].
"""


def normalize_bbox(bbox: list, img_w: int, img_h: int) -> list:
  if not bbox or len(bbox) < 4:
    return [0.0, 0.0, 1.0, 1.0]
  vals = [float(v) for v in bbox[:4]]
  if max(vals) > 1.0:
    if max(vals) <= 1000.0:
      vals = [v / 1000.0 for v in vals]
    else:
      vals = [
          vals[0] / img_h,
          vals[1] / img_w,
          vals[2] / img_h,
          vals[3] / img_w,
      ]
  return [round(min(max(v, 0.0), 1.0), 4) for v in vals]


def single_pass(img_bytes: bytes, orig_w: int, orig_h: int) -> dict:
  res = ollama.chat(
      model=MODEL_NAME,
      messages=[{
          "role": "user",
          "content": SYSTEM_PROMPT,
          "images": [img_bytes],
      }],
      format="json",
      options={"temperature": TEMPERATURE},
  )
  data = json.loads(res["message"]["content"])
  artifacts = []
  for a in data.get("artifacts", []):
    raw_b = a.get("bbox_normalized") or a.get("bbox") or [0, 0, 1, 1]
    artifacts.append({
        "category": str(a.get("category", "other")),
        "location_description": str(a.get("location_description", "")),
        "severity": float(a.get("severity", 0.5)),
        "bbox_normalized": normalize_bbox(raw_b, orig_w, orig_h),
    })
  return {
      "verdict": data.get("verdict", "REAL_PHOTO"),
      "confidence": float(data.get("confidence", 0.5)),
      "forensic_reasoning": data.get("forensic_reasoning", ""),
      "artifacts": artifacts,
  }


def test_file(path: str, label_type: str):
  print(
      f"\n==================== ТЕСТ: {label_type.upper()} ({path})"
      " ===================="
  )
  with Image.open(path) as img:
    w, h = img.size
    img_rgb = img.convert("RGB")
    img_rgb.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=92)
    b = buf.getvalue()

  print("[*] Прогон 1...")
  p1 = single_pass(b, w, h)
  print(f"    Вердикт: {p1['verdict']} (conf: {p1['confidence']})")
  print(f"    Артефактов: {len(p1['artifacts'])}")
  print(f"    Рассуждение: {p1['forensic_reasoning'][:200]}...\n")

  print("[*] Прогон 2 (Self-Consistency)...")
  p2 = single_pass(b, w, h)
  print(f"    Вердикт: {p2['verdict']} (conf: {p2['confidence']})")
  print(f"    Артефактов: {len(p2['artifacts'])}")
  print(f"    Рассуждение: {p2['forensic_reasoning'][:200]}...\n")

  s1 = (
      p1["confidence"]
      if p1["verdict"] == "AI_GENERATED"
      else (1.0 - p1["confidence"])
  )
  s2 = (
      p2["confidence"]
      if p2["verdict"] == "AI_GENERATED"
      else (1.0 - p2["confidence"])
  )
  diff = abs(s1 - s2)
  final_soft = (s1 + s2) / 2.0

  print("[✓] Итог консенсуса:")
  print(f"    Расхождение между прогонами: {diff:.3f} (порог <= 0.35)")
  print(f"    Итоговый Soft-Label: {final_soft:.4f}")
  status = (
      "ПРИНЯТО В ДАТАСЕТ"
      if diff <= 0.35
      else "ОТБРАКОВАНО (разногласие прогонов)"
  )
  print(f"    Статус отбора: {status}")


if __name__ == "__main__":
  reals = glob.glob("data/robust_v1/real/*.*")
  fakes = glob.glob("data/robust_v1/fake/*.*")

  if reals:
    test_file(random.choice(reals), "REAL_PHOTO")
  if fakes:
    test_file(random.choice(fakes), "AI_GENERATED")
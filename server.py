"""
FastAPI Server for VeriVision MoE v3.5
Main application entry point with Perceptual DINOv2 Student & Forensics Explainability.
"""

import io
import sys
import base64
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from PIL import Image, ImageChops, ImageEnhance, UnidentifiedImageError
import pillow_heif
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

pillow_heif.register_heif_opener()

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.serving.ensemble import VeriVisionEnsemble

app = FastAPI(title="VeriVision MoE v3.5", version="3.5.0")
templates = Jinja2Templates(directory="templates")

detector = VeriVisionEnsemble(models_dir="models")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


class ExpertBreakdown(BaseModel):
    name: str
    verdict_text: str
    probability: float
    contribution_percent: float
    direction: str
    heatmap_b64: Optional[str] = None


class DeepAnalysisResponse(BaseModel):
    ai_probability: float
    confidence_band: str
    verdict: str
    experts: List[ExpertBreakdown]


def validate_and_load_image(file: UploadFile, image_bytes: bytes) -> Image.Image:
    filename = file.filename or ""
    file_ext = Path(filename).suffix.lower()
    
    if not file_ext or file_ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Формат файла '{file_ext or 'неизвестный'}' не поддерживается. "
            f"Разрешенные форматы: {supported_str}"
        )

    if len(image_bytes) == 0:
        raise ValueError("Загружен пустой файл.")

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError("Размер файла превышает допустимый лимит 25 МБ.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, Exception) as err:
        raise ValueError(f"Не удалось прочитать изображение или файл поврежден: {str(err)}")


def generate_forensic_heatmaps(img: Image.Image) -> dict:
    buf_orig = io.BytesIO()
    img.save(buf_orig, "JPEG", quality=90)
    buf_orig.seek(0)
    ela_img = ImageEnhance.Brightness(
        ImageChops.difference(img, Image.open(buf_orig))
    ).enhance(15.0)

    buf_ela = io.BytesIO()
    ela_img.save(buf_ela, format="PNG")
    ela_b64 = base64.b64encode(buf_ela.getvalue()).decode("utf-8")

    gray = np.array(img.convert("L"), dtype=np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1e-6)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(mag, cmap="inferno")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buf_fft = io.BytesIO()
    plt.savefig(buf_fft, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    fft_b64 = base64.b64encode(buf_fft.getvalue()).decode("utf-8")

    return {"ela_image": ela_b64, "fft_image": fft_b64}


def generate_heatmap_overlay(original_image_np: np.ndarray, anomaly_mask: np.ndarray) -> str:
    if len(anomaly_mask.shape) == 3:
        anomaly_mask = cv2.cvtColor(anomaly_mask, cv2.COLOR_BGR2GRAY)

    mask_norm = cv2.normalize(anomaly_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)

    _, alpha_mask = cv2.threshold(mask_norm, 100, 255, cv2.THRESH_BINARY)
    alpha_mask = alpha_mask.astype(float) / 255.0
    alpha_mask = np.expand_dims(alpha_mask, axis=2)

    original_bgr = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)
    overlay = (heatmap * alpha_mask + original_bgr * (1.0 - alpha_mask)).astype(np.uint8)

    _, buffer = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buffer).decode("utf-8")


def get_expert_verbalization(expert: str, prob: float, contribution: float) -> str:
    if abs(contribution) < 0.01:
        return "Паттерн не обнаружен (сигнал неинформативен, вес аннулирован)."

    if expert == "fft":
        return (
            "Обнаружена регулярная частотная решетка генератора."
            if prob > 0.5
            else "Естественный спектральный спад частот."
        )
    elif expert == "ela":
        return (
            "Резкие перепады уровней квантования (признак локального Inpainting)."
            if prob > 0.5
            else "Равномерное квантование без монтажных стыков."
        )
    elif expert == "prnu":
        return (
            "Отсутствие консистентного сенсорного шума оптики."
            if prob > 0.5
            else "Обнаружен характерный шум матрицы камеры."
        )
    return "Характерный паттерн обнаружен."


@app.get("/", response_class=HTMLResponse)
async def index_view(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/analyze")
async def analyze_image_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        image = validate_and_load_image(file, image_bytes)
    except ValueError as val_err:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"error": str(val_err), "code": "UNSUPPORTED_FORMAT"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": f"Ошибка обработки файла: {str(e)}", "code": "PROCESSING_ERROR"},
        )

    result = detector.analyze(image)
    
    print("\n" + "=" * 45)
    print(f"FILE:           {file.filename}")
    print(f"VERDICT:        {result['verdict']} (AI Prob: {result['ai_probability']:.4f})")
    print(f"Calibrated P:   {result['metrics']['calibrated_probs']}")
    print(f"Contributions:  {result['metrics']['contributions']}")
    print("=" * 45 + "\n")

    resized_preview = image.resize((512, 512))
    result["heatmaps"] = generate_forensic_heatmaps(resized_preview)

    if "_student_loc_map" in result:
        del result["_student_loc_map"]

    return JSONResponse(status_code=200, content=result)


@app.post("/api/deep-analysis", response_model=DeepAnalysisResponse)
async def deep_analysis_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        image = validate_and_load_image(file, image_bytes)
    except ValueError as val_err:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"error": str(val_err), "code": "UNSUPPORTED_FORMAT"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": f"Ошибка обработки файла: {str(e)}", "code": "PROCESSING_ERROR"},
        )

    cv_image = np.asarray(image.resize((512, 512)))
    fusion_result = detector.analyze(image)

    metrics = fusion_result.get("metrics", {})
    contributions = metrics.get("contributions", {})
    calibrated_probs = metrics.get("calibrated_probs", {})
    artifacts = metrics.get("detected_artifacts", {})

    experts_response = []

    student_loc_map = fusion_result.get("_student_loc_map")
    student_heatmap = None
    if student_loc_map is not None and np.max(student_loc_map) > 0.35:
        try:
            mask_resized = cv2.resize((student_loc_map * 255).astype(np.uint8), (512, 512))
            student_heatmap = generate_heatmap_overlay(cv_image, mask_resized)
        except Exception:
            student_heatmap = None

    if artifacts:
        detected_list = [f"{cat} ({prob * 100:.0f}%)" for cat, prob in artifacts.items()]
        artifact_str = "Обнаружены артефакты: " + ", ".join(detected_list)
    else:
        artifact_str = "Критических структурных аномалий не обнаружено."

    p_perceptual = calibrated_probs.get("perceptual", 0.5)
    contrib_perceptual = contributions.get("perceptual_backbone", 0.0)

    experts_response.append(
        ExpertBreakdown(
            name="PERCEPTUAL (DINOv2)",
            verdict_text=artifact_str,
            probability=round(float(p_perceptual), 3),
            contribution_percent=round(float(contrib_perceptual) * 100, 1),
            direction="AI" if p_perceptual > 0.5 else "REAL",
            heatmap_b64=student_heatmap,
        )
    )

    for exp_name in ["ela", "prnu", "fft"]:
        cal_prob = calibrated_probs.get(exp_name, 0.5)
        contrib = contributions.get(exp_name, 0.0)
        direction = "AI" if contrib > 0.01 else ("REAL" if contrib < -0.01 else "NEUTRAL")

        heatmap_b64 = None
        if exp_name == "ela":
            try:
                mask = detector.forensics.compute_ela_score(image.resize((512, 512)), return_mask=True)
                if isinstance(mask, np.ndarray):
                    heatmap_b64 = generate_heatmap_overlay(cv_image, mask)
            except Exception:
                heatmap_b64 = None

        experts_response.append(
            ExpertBreakdown(
                name=exp_name.upper(),
                verdict_text=get_expert_verbalization(exp_name, cal_prob, contrib),
                probability=round(float(cal_prob), 3),
                contribution_percent=round(float(contrib) * 100, 1),
                direction=direction,
                heatmap_b64=heatmap_b64,
            )
        )

    return DeepAnalysisResponse(
        ai_probability=fusion_result["ai_probability"],
        confidence_band=fusion_result["confidence_band"],
        verdict=fusion_result["verdict"],
        experts=experts_response,
    )


@app.get("/health")
async def healthcheck():
    return {"status": "ok", "service": "VeriVision MoE v3.5"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
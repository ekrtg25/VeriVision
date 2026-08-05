import io
import time
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import Response
import cv2

# Добавляем корень проекта в sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.models.baseline_cnn import BaselineDetector
from src.models.frequency_model import FrequencyDetector
from src.models.clip_probe import CLIPProbeDetector
from src.data.transforms import get_transforms
from src.explainability.grad_cam import GradCAM, overlay_heatmap

app = FastAPI(
    title="Synthetic Media Forensics API",
    description="API для детекции AI-сгенерированных изображений и генерации Grad-CAM тепловых карт.",
    version="1.0.0"
)

# Определение устройства
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Кэш для загруженных моделей в памяти (Lazy Loading)
MODELS = {}

def get_loaded_model(model_type: str):
    """Инициализация и кэширование моделей."""
    if model_type not in MODELS:
        if model_type == "baseline":
            model = BaselineDetector(num_classes=1, pretrained=True).to(DEVICE)
        elif model_type == "frequency":
            model = FrequencyDetector(num_classes=1, pretrained=True).to(DEVICE)
        elif model_type == "clip":
            model = CLIPProbeDetector(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", num_classes=1).to(DEVICE)
        else:
            raise HTTPException(status_code=400, detail=f"Неизвестный тип модели: {model_type}")
        
        model.eval()
        MODELS[model_type] = model
        
    return MODELS[model_type]


def preprocess_image(image_bytes: bytes, model_type: str):
    """Загрузка изображения из байтов и препроцессинг."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Невалидный файл изображения")

    if model_type == "clip":
        temp_clip = get_loaded_model("clip")
        tensor_img = temp_clip.preprocess(pil_img).unsqueeze(0).to(DEVICE)
    else:
        transform = get_transforms(img_size=224, is_train=False)
        tensor_img = transform(pil_img).unsqueeze(0).to(DEVICE)

    return pil_img, tensor_img


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "device": str(DEVICE),
        "loaded_models": list(MODELS.keys())
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Query("baseline", enum=["baseline", "frequency", "clip"])
):
    start_time = time.time()
    contents = await file.read()
    _, tensor_img = preprocess_image(contents, model_type)
    
    model = get_loaded_model(model_type)
    
    with torch.no_grad():
        output = model(tensor_img)
        prob = torch.sigmoid(output).item()

    inference_time = round(time.time() - start_time, 4)
    label = "Fake" if prob >= 0.5 else "Real"

    return {
        "filename": file.filename,
        "model_type": model_type,
        "prediction": label,
        "fake_probability": round(prob, 4),
        "inference_time_seconds": inference_time
    }


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    model_type: str = Query("baseline", enum=["baseline", "frequency"])
):
    """Генерация и возврат Grad-CAM изображения."""
    contents = await file.read()
    pil_img, tensor_img = preprocess_image(contents, model_type)
    
    model = get_loaded_model(model_type)
    
    # Определяем целевой слой для Grad-CAM
    if model_type == "baseline":
        target_layer = model.backbone.layer4
    elif model_type == "frequency":
        target_layer = model.backbone.layer4

    cam = GradCAM(model, target_layer)
    heatmap = cam.generate_heatmap(tensor_img)

    orig_np = np.array(pil_img)
    _, overlay = overlay_heatmap(heatmap, orig_np)

    # Кодируем массив NumPy (RGB) в байты PNG для передачи по сети
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    success, encoded_img = cv2.imencode(".png", overlay_bgr)
    
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка кодирования изображения")

    return Response(content=encoded_img.tobytes(), media_type="image/png")
import base64
import io
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from src.serving.ensemble import HybridEnsembleDetector
from src.serving.gradcam import ResNetGradCAM

app = FastAPI(title="VeriVision API", version="2.0.0")

# Инициализация детектора при старте
detector = HybridEnsembleDetector(
    cnn_weights_path="models/baseline_weights.pth",
    fft_model_path="models/rf_spectral.pkl",
    srm_model_path="models/rf_srm.pkl",
    meta_model_path="models/meta_classifier.pkl"
)

TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.50)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    temp_file_path = TEMP_DIR / file.filename
    try:
        # Сохраняем файл для инференса
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Получаем предсказание ансамбля
        results = detector.predict(str(temp_file_path), threshold=threshold)
        
        # 2. Генерируем Grad-CAM
        try:
            grad_cam_engine = ResNetGradCAM(detector.cnn)
            heatmap_img = grad_cam_engine.generate_heatmap(str(temp_file_path))
            
            # Конвертируем PIL Image в base64 для отправки по сети
            buffered = io.BytesIO()
            heatmap_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            results["gradcam_b64"] = f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            results["gradcam_b64"] = None
            print(f"Grad-CAM Error: {e}")

        # Фикс ошибки TypeError: 'numpy.bool' object is not iterable
        return {
            'final_score': float(results['final_score']),
            'is_fake': bool(results['is_fake']),
            'cnn_prob': float(results['cnn_prob']),
            'fft_prob': float(results['fft_prob']),
            'srm_prob': float(results['srm_prob']),
            'gating_active': bool(results['gating_active']),
            'gradcam_b64': results['gradcam_b64']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_file_path.exists():
            os.remove(temp_file_path)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path("templates/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend template not found.")
    return index_path.read_text(encoding="utf-8")
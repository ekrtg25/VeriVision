import base64
import io
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import storage

from src.serving.ensemble import HybridEnsembleDetector
from src.serving.gradcam import ResNetGradCAM

GCS_BUCKET_NAME = "verivision-models-ff0e5ce4"
MODELS_DIR = Path("models")

REQUIRED_MODELS = {
    "baseline_weights.pth": MODELS_DIR / "baseline_weights.pth",
    "rf_spectral.pkl": MODELS_DIR / "rf_spectral.pkl",
    "rf_srm.pkl": MODELS_DIR / "rf_srm.pkl",
    "meta_classifier.pkl": MODELS_DIR / "meta_classifier.pkl",
}


def download_model_weights():
    """Скачивает веса моделей из Cloud Storage, если их нет локально."""
    MODELS_DIR.mkdir(exist_ok=True)
    
    missing_files = [filename for filename, filepath in REQUIRED_MODELS.items() if not filepath.exists()]
    
    if not missing_files:
        print("Все файлы весов найдены локально. Скачивание не требуется.")
        return

    print(f"Отсутствуют файлы весов: {missing_files}. Скачиваем из GCS...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)

        for filename in missing_files:
            destination_path = REQUIRED_MODELS[filename]
            blob = bucket.blob(filename)
            print(f"Скачиваем {filename} -> {destination_path}...")
            blob.download_to_filename(destination_path)
            print(f"Файл {filename} успешно загружен.")
            
    except Exception as e:
        print(f"Ошибка при скачивании весов из Cloud Storage: {e}")
        raise e


# Скачиваем веса перед инициализацией детектора
download_model_weights()

app = FastAPI(title="VeriVision API", version="2.0.0")

# Инициализация детектора при старте (модели грузятся один раз в память процесса)
detector = HybridEnsembleDetector(
    cnn_weights_path=str(REQUIRED_MODELS["baseline_weights.pth"]),
    fft_model_path=str(REQUIRED_MODELS["rf_spectral.pkl"]),
    srm_model_path=str(REQUIRED_MODELS["rf_srm.pkl"]),
    meta_model_path=str(REQUIRED_MODELS["meta_classifier.pkl"])
)

TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.50)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    suffix = Path(file.filename or "upload").suffix or ".jpg"
    temp_file_path = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = detector.predict(str(temp_file_path), threshold=threshold)

        try:
            grad_cam_engine = ResNetGradCAM(detector.cnn)
            heatmap_img = grad_cam_engine.generate_heatmap(str(temp_file_path))

            buffered = io.BytesIO()
            heatmap_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            results["gradcam_b64"] = f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            results["gradcam_b64"] = None
            print(f"Grad-CAM Error: {e}")

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
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import streamlit as st
import time
import torch
import json
from PIL import Image

# Импорты моделей и Grad-CAM
from src.models.ensemble import EnsembleDetector
from src.models.gradcam import ResNetGradCAM

# Строгая конфигурация страницы
st.set_page_config(page_title="VeriVision Engine", layout="wide", initial_sidebar_state="collapsed")

# Внедрение утилитарного CSS (Brutalism/Tech UI)
st.markdown("""
    <style>
    /* Скрываем стандартные элементы Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Принудительный моноширинный шрифт */
    html, body, [class*="css"]  {
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Строгие рамки для контейнеров */
    .stContainer {
        border: 1px solid #333333;
        padding: 15px;
    }
    
    /* Стилизация метрик и статусов */
    .status-fake {
        color: #ff4444;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .status-real {
        color: #00cc66;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .log-text {
        font-size: 0.85rem;
        color: #888888;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    """Кэшируем загрузку весов в память"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    return EnsembleDetector(
        baseline_weights_path="models/baseline_weights.pth",
        clip_weights_path="models/clip_weights.pth",
        device=device
    )

st.title("VERIVISION // FORENSIC ANALYSIS ENGINE")
st.text("SYSTEM: ONLINE | ARCHITECTURE: RESNET-50 (ONLY MODE) | XAI: GRAD-CAM")
st.markdown("---")

engine = load_engine()

# Разметка на 3 равные колонки для визуального анализа и логов
col_img, col_cam, col_analysis = st.columns([1, 1, 1.2])

temp_path = "temp_upload.jpg"

with col_img:
    st.subheader("TARGET IMAGE")
    uploaded_file = st.file_uploader("UPLOAD FILE (JPEG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, output_format="JPEG")
        image.convert("RGB").save(temp_path)

with col_cam:
    st.subheader("HEATMAP (Grad-CAM)")
    if uploaded_file is not None:
        try:
            grad_cam_engine = ResNetGradCAM(engine.baseline)
            heatmap_img = grad_cam_engine.generate_heatmap(temp_path)
            st.image(heatmap_img, caption="ResNet Layer4 Activations", use_container_width=True)
        except Exception as e:
            st.error(f"Grad-CAM Error: {e}")

with col_analysis:
    st.subheader("ANALYSIS LOG")
    log_container = st.empty()
    
    if uploaded_file is not None:
        logs = ["[sys] Image loaded into memory...", "[sys] Extracting features..."]
        log_container.text("\n".join(logs))
        time.sleep(0.2)
        
        logs.append("[model] Running Baseline ResNet architecture...")
        log_container.text("\n".join(logs))
        time.sleep(0.3)
        
        logs.append("[model] Running CLIP Vision Transformer (Bypassed)...")
        log_container.text("\n".join(logs))
        
        # Инференс ансамбля
        start_time = time.time()
        result = engine.predict(temp_path, mode="max", threshold=0.35)
        exec_time = time.time() - start_time
        
        # ВРЕМЕННЫЙ ТЕСТ: Полностью отключаем влияние CLIP, опираемся только на ResNet
        result['ensemble_prob'] = result['baseline_prob']
        if result['ensemble_prob'] >= result['threshold_used']:
            result['prediction'] = "Fake"
        else:
            result['prediction'] = "Real"

        logs.append("[xai] Generating Grad-CAM activation map...")
        logs.append("[ensemble] Single-Model Evaluation Mode (ResNet Pure)...")
        logs.append(f"[sys] Analysis complete in {exec_time:.2f}s.")
        log_container.text("\n".join(logs))
        
        st.markdown("---")
        st.subheader("RAW METRICS")
        
        formatted_json = json.dumps({
            "status": "COMPLETED",
            "execution_time_ms": int(exec_time * 1000),
            "ensemble_mode": "resnet_only (temp)",
            "decision_threshold": result['threshold_used'],
            "baseline_probability": round(result['baseline_prob'], 4),
            "clip_probability": round(result['clip_prob'], 4),
            "ensemble_probability": round(result['ensemble_prob'], 4)
        }, indent=4)
        
        st.code(formatted_json, language="json")
        
        st.markdown("---")
        st.subheader("VERDICT")
        
        fake_prob = result["ensemble_prob"] * 100
        real_prob = (1 - result["ensemble_prob"]) * 100
        
        if result['prediction'] == "Fake":
            st.markdown(f'<div class="status-fake">>> SYNTHETIC MEDIA DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-real">>> AUTHENTIC MEDIA VERIFIED</div>', unsafe_allow_html=True)

        st.markdown("#### PROBABILITY BREAKDOWN")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("🤖 FAKE (Synthetic)", f"{fake_prob:.1f}%")
        with col_res2:
            st.metric("📷 REAL (Authentic)", f"{real_prob:.1f}%")

        st.caption("Synthetic Probability Scale:")
        st.progress(float(result["ensemble_prob"]))
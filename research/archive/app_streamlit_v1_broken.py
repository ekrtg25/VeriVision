# app.py
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

from src.models.ensemble import EnsembleDetector
from src.models.gradcam import ResNetGradCAM

st.set_page_config(page_title="VeriVision Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    html, body, [class*="css"]  {
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    .stContainer {
        border: 1px solid #333333;
        padding: 15px;
    }
    
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
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    return EnsembleDetector(
        baseline_weights_path="models/baseline_weights.pth",
        clip_weights_path="models/clip_weights.pth",
        device=device
    )

st.title("VERIVISION // FORENSIC ANALYSIS ENGINE")
st.text("SYSTEM: ONLINE | ARCHITECTURE: CONVNEXT + ViT ENSEMBLE | FUSION: UNCERTAINTY-WEIGHTED")
st.markdown("---")

engine = load_engine()

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
            st.image(heatmap_img, caption="ConvNeXt Feature Activations", use_container_width=True)
        except Exception as e:
            st.error(f"Grad-CAM Error: {e}")

with col_analysis:
    st.subheader("ANALYSIS LOG")
    log_container = st.empty()
    
    if uploaded_file is not None:
        logs = ["[sys] Image loaded into memory...", "[sys] Extracting features..."]
        log_container.text("\n".join(logs))
        time.sleep(0.1)
        
        logs.append("[model] Running ConvNeXt-Tiny Architecture...")
        log_container.text("\n".join(logs))
        time.sleep(0.1)
        
        logs.append("[model] Running CLIP Vision Transformer (L2-Normalized)...")
        log_container.text("\n".join(logs))
        
        # Инференс ансамбля в режиме "uncertainty"
        start_time = time.time()
        result = engine.predict(temp_path, mode="uncertainty", threshold=0.50)
        exec_time = time.time() - start_time
        
        logs.append("[xai] Generating Grad-CAM activation map...")
        logs.append("[ensemble] Computing Uncertainty-Weighted Entropy Fusion...")
        logs.append(f"[sys] Analysis complete in {exec_time:.2f}s.")
        log_container.text("\n".join(logs))
        
        st.markdown("---")
        st.subheader("RAW METRICS")
        
        formatted_json = json.dumps({
            "status": "COMPLETED",
            "execution_time_ms": int(exec_time * 1000),
            "ensemble_mode": result['mode_used'],
            "decision_threshold": result['threshold_used'],
            "convnext_probability": round(result['baseline_prob'], 4),
            "clip_probability": round(result['clip_prob'], 4),
            "ensemble_probability": round(result['ensemble_prob'], 4)
        }, indent=4)
        
        st.code(formatted_json, language="json")
        
        st.markdown("---")
        st.subheader("VERDICT")
        
        fake_prob = result["ensemble_prob"] * 100
        real_prob = (1 - result["ensemble_prob"]) * 100
        
        if result['prediction'] == "Fake":
            st.markdown(f'<div class="status-fake">>> SYNTHETIC MEDIA DETECTED ({fake_prob:.1f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-real">>> AUTHENTIC MEDIA VERIFIED ({real_prob:.1f}%)</div>', unsafe_allow_html=True)

        st.markdown("#### PROBABILITY BREAKDOWN")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("🤖 FAKE (Synthetic)", f"{fake_prob:.1f}%")
        with col_res2:
            st.metric("📷 REAL (Authentic)", f"{real_prob:.1f}%")

        st.caption("Synthetic Probability Scale:")
        st.progress(float(result["ensemble_prob"]))
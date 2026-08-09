import streamlit as st
import time
import os
from PIL import Image
from src.models.ensemble import EnsembleDetector

# Инициализация базового движка (кешируем, чтобы не грузить веса каждый раз)
@st.cache_resource
def load_fast_engine():
    return EnsembleDetector(
        baseline_weights_path="models/baseline_weights.pth", 
        clip_weights_path="models/clip_vit_l_weights.pth"
    )

fast_engine = load_fast_engine()

def main():
    st.set_page_config(page_title="VeriVision", page_icon="👁️", layout="wide")
    
    st.title("👁️ VeriVision: AI Image Detector")
    st.markdown("Загрузи изображение, чтобы проверить, создано ли оно искусственным интеллектом.")

    # Боковая панель для настроек
    with st.sidebar:
        st.header("⚙️ Настройки сканирования")
        # Наш идеальный порог, который мы вычислили ранее
        threshold = st.slider("Threshold (Порог)", min_value=0.0, max_value=1.0, value=0.49, step=0.01)
        
        st.markdown("---")
        st.info("**Fast Scan**: Мгновенный анализ через ConvNeXt и CLIP (ViT-L/14).\n\n**Deep Scan**: Тяжелый математический анализ (DIRE + FFT) для SOTA генераторов (Midjourney, Stable Diffusion, GANs).")

    uploaded_file = st.file_uploader("Выбери изображение (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Отображаем картинку и интерфейс в две колонки
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            # Сохраняем во временный файл для движка
            temp_path = "temp_upload.jpg"
            image.save(temp_path)

        with col2:
            st.subheader("Режим анализа")
            
            # Тот самый переключатель режимов
            scan_mode = st.radio(
                "Выберите глубину проверки:",
                ["⚡ Fast Scan (Базовый ансамбль)", "🔬 Deep Forensic Scan (DIRE + FFT)"],
                index=0,
                horizontal=True
            )

            analyze_button = st.button("Начать анализ", type="primary", use_container_width=True)

            if analyze_button:
                st.markdown("---")
                
                if "Fast Scan" in scan_mode:
                    with st.spinner("Выполняется быстрый анализ..."):
                        # Запускаем наш текущий ансамбль
                        result = fast_engine.predict(temp_path, mode="uncertainty", threshold=threshold)
                        
                        st.subheader("Результат (Fast Scan)")
                        if result["prediction"] == "Fake":
                            st.error(f"🚨 ОБНАРУЖЕН ФЕЙК (Уверенность: {result['ensemble_prob']:.1%})")
                        else:
                            st.success(f"✅ РЕАЛЬНОЕ ФОТО (Уверенность: {1 - result['ensemble_prob']:.1%})")
                        
                        st.progress(result['ensemble_prob'])
                        st.caption(f"ConvNeXt: {result['baseline_prob']:.1%} | CLIP ViT-L/14: {result['clip_prob']:.1%}")

                else:
                    # Режим Deep Scan (пока с визуальными заглушками)
                    st.subheader("Результат (Deep Forensic Scan)")
                    
                    # Имитация работы DIRE (скоро заменим на реальный код)
                    with st.status("Проведение глубокого анализа...", expanded=True) as status:
                        st.write("🔄 Шаг 1/3: Инициализация базового ансамбля...")
                        fast_result = fast_engine.predict(temp_path, mode="uncertainty", threshold=threshold)
                        time.sleep(0.5)
                        
                        st.write("🧬 Шаг 2/3: Вычисление ошибки реконструкции диффузии (DIRE)...")
                        # TODO: Здесь будет вызов DIRE
                        time.sleep(1.5)
                        
                        st.write("📡 Шаг 3/3: Анализ частотного спектра на следы апсемплинга (FFT)...")
                        # TODO: Здесь будет вызов FFT
                        time.sleep(1.0)
                        
                        status.update(label="Анализ завершен!", state="complete", expanded=False)

                    # Временный мокап результата для глубокого сканирования
                    st.warning("⚠️ Deep Scan выявил аномалии. Ожидайте подключения математических модулей.")
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric(label="Базовый скор", value=f"{fast_result['ensemble_prob']:.1%}")
                    col_b.metric(label="DIRE Скоринг", value="В разработке")
                    col_c.metric(label="FFT Аномалии", value="В разработке")

        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)
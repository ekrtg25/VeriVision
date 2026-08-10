import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from src.serving.ensemble import HybridEnsembleDetector

# Кешируем загрузку всех трех моделей
@st.cache_resource
def load_engine():
    return HybridEnsembleDetector(
        cnn_weights_path="models/baseline_weights.pth",
        fft_model_path="models/rf_spectral.pkl",
        srm_model_path="models/rf_srm.pkl",
        meta_model_path="models/meta_classifier.pkl"
    )

engine = load_engine()

# --- НОВАЯ ФУНКЦИЯ: OOD-детектор (Анализатор качества фото) ---
def check_image_conditions(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []

    # Сжимаем картинку до 800px по ширине для стабильности
    h, w = img.shape[:2]
    new_w = 800
    new_h = int(new_w * (h / w))
    img_resized = cv2.resize(img, (new_w, new_h))
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    warnings = []

    # 1. Детектор настоящих бликов
    # Подняли порог до 240 (почти чистый белый)
    overexposed_pixels = np.sum(gray > 240)
    total_pixels = gray.shape[0] * gray.shape[1]
    overexposed_ratio = overexposed_pixels / total_pixels
    
    # Считаем среднюю яркость всего кадра
    mean_brightness = np.mean(gray)

    # Триггеримся ТОЛЬКО если есть выжженные пятна (>1%), 
    # НО при этом кадр в целом не является супер-светлым (mean < 180)
    if overexposed_ratio > 0.01 and mean_brightness < 180:
        warnings.append("☀️ **Сложный свет:** Обнаружены жесткие блики или пересвет (вероятно, окно/вспышка). Алгоритмы камеры могли исказить пиксели.")

    # 2. Детектор сильного размытия
    # Опустили порог до 75. Теперь среагирует только на явный смаз или расфокус
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < 75: 
        warnings.append("💨 **Сильное размытие:** Значительная часть кадра смазана. Нейросети могут ложно принять это за артефакты генерации.")

    return warnings
def main():
    st.set_page_config(page_title="VeriVision", page_icon="👁️", layout="wide")
    
    st.title("👁️ VeriVision: AI Image Detector")
    st.markdown("Гибридный детектор генеративных изображений. Анализирует пиксели, частотный спектр и матричный шум.")

    with st.sidebar:
        st.header("⚙️ Настройки")
        threshold = st.slider("Порог срабатывания (Threshold)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
        
        with st.expander("ℹ️ Как работает этот порог?"):
            st.markdown(
                "Этот ползунок управляет балансом между строгой проверкой и перестраховкой алгоритма. "
                "Он задает минимальную вероятность, при которой ансамбль выносит вердикт **Fake**.\n\n"
                "* **Высокий порог (0.70 - 0.90):** Режим высокой точности.\n"
                "* **Низкий порог (0.10 - 0.30):** Режим параноика.\n"
                "* **Золотая середина (0.50):** Мета-модель принимает сбалансированное решение."
            )
        
        st.markdown("---")
        st.info(
            "**Архитектура (Hybrid Ensemble):**\n\n"
            "🧠 **Визуальный эксперт (ConvNeXt):** Ищет артефакты пикселей и поплывшие текстуры.\n\n"
            "📡 **Спектральный эксперт (FFT):** Выявляет аномалии апсемплинга в частотной области.\n\n"
            "🔬 **Шумовой эксперт (SRM):** Анализирует синтетический матричный шум.\n\n"
            "⚖️ **Мета-модель (Stacking):** Логистическая регрессия взвешивает голоса экспертов."
        )

    uploaded_file = st.file_uploader("Загрузите изображение (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.2])
        
        temp_path = "temp_upload.jpg"
        
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Загруженное изображение", use_container_width=True)
            image.save(temp_path)

        with col2:
            st.subheader("Результат анализа")
            
            # Сначала прогоняем через наш легкий OOD-детектор
            image_warnings = check_image_conditions(temp_path)
            
            # Если нашли блики или размытие — честно предупреждаем пользователя
            if image_warnings:
                st.warning("⚠️ **ВНИМАНИЕ: Сложные условия съемки**\n\nАлгоритмы смартфона могли исказить оригинальные пиксели. **Точность детектора снижена.**")
                for w in image_warnings:
                    st.markdown(f"- {w}")
                st.markdown("---")

            # Запускаем тяжелый инференс
            with st.spinner("Просвечиваем изображение..."):
                result = engine.predict(temp_path, threshold=threshold)
            
            # Главный вердикт
            if result["is_fake"]:
                st.error(f"🚨 ОБНАРУЖЕН СГЕНЕРИРОВАННЫЙ КОНТЕНТ (Уверенность: {result['final_score']:.1%})")
            else:
                st.success(f"✅ РЕАЛЬНОЕ ФОТО (Уверенность: {1 - result['final_score']:.1%})")
            
            st.progress(float(result['final_score']))
            st.markdown("---")
            
            st.markdown("### Детализация по экспертам")
            
            c1, c2, c3 = st.columns(3)
            
            c1.metric(
                label="🧠 Визуальный (ConvNeXt)", 
                value=f"{result['cnn_prob']:.1%}",
                delta="Fake" if result['cnn_prob'] >= 0.5 else "Real",
                delta_color="inverse" if result['cnn_prob'] >= 0.5 else "normal"
            )
            
            c2.metric(
                label="📡 Спектр (FFT)", 
                value=f"{result['fft_prob']:.1%}",
                delta="Fake" if result['fft_prob'] >= 0.5 else "Real",
                delta_color="inverse" if result['fft_prob'] >= 0.5 else "normal"
            )
            
            c3.metric(
                label="🔬 Шум (SRM)", 
                value=f"{result['srm_prob']:.1%}",
                delta="Fake" if result['srm_prob'] >= 0.5 else "Real",
                delta_color="inverse" if result['srm_prob'] >= 0.5 else "normal"
            )
            
            st.caption("Если мнения экспертов расходятся, итоговое решение принимает Мета-модель (Логистическая регрессия).")

        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()
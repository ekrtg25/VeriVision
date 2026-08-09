import streamlit as st
import os
from PIL import Image
from src.models.ensemble import HybridEnsembleDetector

# Кешируем загрузку всех трех моделей, чтобы инференс был мгновенным
@st.cache_resource
def load_engine():
    return HybridEnsembleDetector(
        cnn_weights="models/baseline_weights.pth", 
        fft_weights="models/rf_spectral.pkl", 
        srm_weights="models/rf_srm.pkl"
    )

engine = load_engine()

def main():
    st.set_page_config(page_title="VeriVision", page_icon="👁️", layout="wide")
    
    st.title("👁️ VeriVision: AI Image Detector")
    st.markdown("Гибридный детектор генеративных изображений. Анализирует пиксели, частотный спектр и матричный шум.")

    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        threshold = st.slider("Порог срабатывания (Threshold)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
        
        # Интерактивная справка для пользователя
        with st.expander("ℹ️ Как работает этот порог?"):
            st.markdown(
                "Этот ползунок управляет балансом между строгой проверкой и перестраховкой алгоритма. "
                "Он задает минимальную вероятность, при которой ансамбль выносит вердикт **Fake**.\n\n"
                "* **Высокий порог (0.70 - 0.90):** Режим высокой точности (High Precision). Детектор назовет изображение фейком только при абсолютной уверенности всех алгоритмов. Защищает реальные фото от ложных обвинений, но может пропустить хитрые дипфейки.\n"
                "* **Низкий порог (0.10 - 0.30):** Режим параноика (High Recall). Система бракует всё, что кажется подозрительным хотя бы одному модулю. Блокирует 100% фейков, но будет часто ошибаться на реальных фото с тяжелой цветокоррекцией.\n"
                "* **Золотая середина (0.50):** Простое большинство голосов."
            )
        
        st.markdown("---")
        st.info(
            "**Архитектура (Hybrid Ensemble):**\n\n"
            "🧠 **Визуальный эксперт (ConvNeXt):** Ищет артефакты пикселей и поплывшие текстуры.\n\n"
            "📡 **Спектральный эксперт (FFT):** Выявляет аномалии апсемплинга в частотной области.\n\n"
            "🔬 **Шумовой эксперт (SRM):** Анализирует синтетический матричный шум."
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
            
            with st.spinner("Просвечиваем изображение..."):
                result = engine.predict(temp_path, threshold=threshold)
            
            # Главный вердикт
            if result["prediction"] == "Fake":
                st.error(f"🚨 ОБНАРУЖЕН СГЕНЕРИРОВАННЫЙ КОНТЕНТ (Уверенность: {result['ensemble_prob']:.1%})")
            else:
                st.success(f"✅ РЕАЛЬНОЕ ФОТО (Уверенность: {1 - result['ensemble_prob']:.1%})")
            
            st.progress(result['ensemble_prob'])
            st.markdown("---")
            
            st.markdown("### Детализация по экспертам")
            
            # Метрики от каждого узкого специалиста
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
            
            st.caption("Если мнения экспертов расходятся, итоговое решение принимается путем усреднения вероятностей.")

        # Очистка временного файла
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()
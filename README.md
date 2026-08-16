<div align="center">

# 👁️ VeriVision MoE v3.5
### Multi-Scale Neural & Physical Forensics Engine for AI Media Authentication

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA%20%7C%20MPS-ee4c2c.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Serving-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![DINOv2](https://img.shields.io/badge/Backbone-Meta%20DINOv2%20ViT--B%2F14-1877f2.svg?style=flat)](https://github.com/facebookresearch/dinov2)

**Мультимодальный фреймворк детекции синтетических изображений (Midjourney v6, SDXL, Flux, Ideogram, Gemini/Imagen 3) с динамическим подавлением компрессионных помех (Dynamic Gating), калибровкой Плата и попиксельной локализацией генеративных аномалий (Dense Patch XAI).**

<br/>

<!-- ==================== ГЛАВНАЯ ГИФКА РАБОТЫ СЕРВИСА ==================== -->
<!-- Положите запись экрана (5-10 сек: drag-and-drop -> инференс -> раскрытие карт) в docs/media/verivision_demo.gif -->
<p align="center">
  <img src="docs/media/verivision_demo.gif" alt="VeriVision Full Workflow Demo" width="850px" style="border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.5);" />
</p>
<p align="center">
  <em>Сквозной процесс: загрузка файла → каскадный анализ (DINOv2, ELA, FFT, PRNU) → генерация карт внимания и вербализация XAI</em>
</p>

</div>

---

## 📸 Сценарии работы и визуальные примеры (Screenshots)

Интерфейс спроектирован так, чтобы не просто выдавать бинарный вердикт, но и объяснять физическую и семантическую природу решения экспертов:

| 🤖 1. Уверенная детекция AI-генерации | 📷 2. Реальное фото (Сложный свет / Сжатие) |
| :---: | :---: |
| <img src="docs/media/screen_ai_detected.png" width="420px" alt="AI Generated Detection" /> | <img src="docs/media/screen_real_photo.png" width="420px" alt="Real Photo with Gating" /> |
| **DINOv2 подсвечивает зоны деноайзинга**, 2D FFT фиксирует гармонические сетки латентного VAE. Вердикт: `AI Generated (96.4%)`. | **Dynamic Gating глушит ELA/FFT**, предотвращая False Positive из-за пережатия в соцсетях. Вердикт: `Authentic Photo (12.1%)`. |

<br/>

| 🧩 3. Локальный монтаж (Inpainting / Face Swap) | 🔬 4. Развернутая XAI-аналитика и Heatmaps |
| :---: | :---: |
| <img src="docs/media/screen_local_splice.png" width="420px" alt="Local Inpainting Splice" /> | <img src="docs/media/screen_xai_heatmaps.png" width="420px" alt="Explainable Forensics Heatmaps" /> |
| DINOv2 находит всплеск локальной вероятности ($p_{\text{local}} > 0.7$) при низком общем балле ($p_{\text{global}} < 0.4$). Вердикт: `Local Inpainting`. | Полноразмерные тепловые карты: плотная сетка активаций DINOv2 ($37 \times 37$), градиенты ELA и 2D FFT спектрограмма. |

---

## 📑 Содержание
- [Проблематика: почему классические детекторы слепнут](#-проблематика-почему-классические-детекторы-слепнут)
- [Архитектурный дизайн MoE v3.5](#-архитектурный-дизайн-moe-v35)
  - [1. Content Semantic Prefilter](#1-content-semantic-prefilter)
  - [2. Fine-Tuned DINOv2 Dense Student (1536d)](#2-fine-tuned-dinov2-dense-student-1536d)
  - [3. Физический уровень: PRNU, ELA, 2D FFT & SRM](#3-физический-уровень-prnu-ela-2d-fft--srm)
  - [4. Dynamic Gating & Confidence Capped Fusion](#4-dynamic-gating--confidence-capped-fusion)
- [Explainable AI (XAI) & Форензик-карты](#-explainable-ai-xai--форензик-карты)
- [Журнал экспериментов и эволюция модели (9 итераций)](#-журнал-экспериментов-и-эволюция-модели-9-итераций)
- [Метрики и OOD-валидация](#-метрики-и-ood-валидация)
- [Инженерная реализация и оптимизация](#-инженерная-реализация-и-оптимизация)
- [Установка и запуск](#-установка-и-запуск)
- [Структура репозитория](#-структура-репозитория)

---

## 🎯 Проблематика: почему классические детекторы слепнут

Большинство существующих детекторов AI-изображений страдают от двух фундаментальных дефектов:

1. **Shortcut Learning (Зубрёжка артефактов датасета):** Модели на базе стандартных ResNet/EfficientNet цепляются за шум конкретной выборки. При переносе на чистые OOD-генерации (Midjourney v6, Flux) точность таких моделей падает до уровня случайного угадывания ($\approx 50\%$).
2. **Ложные срабатывания на сжатии мессенджеров:** Агрессивный JPEG-рекомпресс (Telegram, WhatsApp) разрушает высокочастотные спектры реальных фотографий. Классическая форензика (ELA, FFT) ошибочно принимает это за следы апскейлеров или монтажа.

**VeriVision MoE v3.5** устраняет эти ограничения разделением анализа на независимые модальности (семантическую и физическую) с адаптивной регулировкой доверия (Dynamic Gating).

---

## 🏛️ Архитектурный дизайн MoE v3.5

                       [Входное изображение (JPG/PNG/WEBP/HEIC)]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
         [0. Content Prefilter]                    [JPEG Quality Estimator]
     (Отсечение 3D/Digital Art)                                │
                   │                                           ▼
     ┌─────────────┴─────────────┐                 [Dynamic Gating Multiplier]
     ▼                           ▼                             │
┌──────────────────┐       ┌──────────────────┐                    │
│   DINOv2 ViT     │       │   Физический     │                    │
│  (CLS + Patches) │       │      уровень     │                    │
│   (Perceptual)   │       │ (PRNU, ELA, FFT) │                    │
└────────┬─────────┘       └────────┬─────────┘                    │
│                          │◄─────────────────────────────┘
│                          │ (Подавление веса классики при компрессии)
▼                          ▼
[Platt Calibr.]            [Platt Calibr.]
│                          │
└─────────────┬────────────┘
▼
[Confidence-Weighted & Capped Fusion]
│
├─► [Финальный вердикт: REAL / AI / SPLICE / ART]
├─► [AI Probability Score & Confidence Band]
└─► [Dense Anomaly Heatmap (37×37 -> 518×518)]


### 1. Content Semantic Prefilter
Перед запуском тяжелого инференса Zero-Shot семантический фильтр проверяет, является ли изображение цифровой иллюстрацией, 3D-рендером или скриншотом интерфейса, изолируя нерелевантный контент.

### 2. Fine-Tuned DINOv2 Dense Student (1536d)
* **Бэкбоун:** `facebook/dinov2-base` (ViT-B/14, 86M параметров), обученный на задаче DINOv2 и инвариантный к цветовым искажениям.
* **Partial Fine-Tuning:** Разморожены 2 верхних блока трансформера (`encoder.layer.10`, `encoder.layer.11`) и выходной `layernorm`. Дифференциальный LR: $1.5 \cdot 10^{-5}$ для блоков внимания и $2.0 \cdot 10^{-4}$ для головы.
* **Мультимасштабное представление:** Классификатор принимает конкатенацию CLS-токена ($768$) и среднего пространственного вектора патчей ($768$), формируя дескриптор размерностью **$1536$**.
* **Dense Patch Evaluation:** Прогон классификатора по всем $1369$ патчам ($37 \times 37$ сетка при $518\text{px}$) строит карту аномалий локального деноайзинга без накладных расходов Grad-CAM.

### 3. Физический уровень: PRNU, ELA, 2D FFT & SRM
* **PRNU Sensor Trace:** Извлечение остаточного отпечатка кремниевой матрицы фотокамеры (Photo-Response Non-Uniformity). ИИ-генераторы не обладают сенсорным шумом оптического тракта.
* **Error Level Analysis (ELA):** Анализ градиентов матриц квантования JPEG при контролируемом пересохранении (качество 90) для поиска локального Inpainting.
* **2D FFT Spectrum:** Поиск радиальных частотных сеток и гармонических пиков, оставляемых латентными декодерами VAE и апскейлерами.
* **SRM (Spatial Rich Models):** 30 базовых пространственных фильтров стеганоанализа для выявления аномалий эксцесса и асимметрии распределения шума.

### 4. Dynamic Gating & Confidence Capped Fusion
1. **Dynamic Gating:** При падении оцениваемого качества JPEG-компрессии вес ELA и FFT линейно снижается:
   $$\text{Weight}_{\text{ELA, FFT}} \leftarrow \text{Weight}_{\text{raw}} \times \max(0.1, \min(1.0, Q_{\text{comp}}))$$
2. **Weight Capping:** Максимальный вес классических экспертов ограничен $\le 20\%$ от веса DINOv2, что исключает доминирование шума над семантикой.
3. **Platt Calibration:** Сырые расстояния каждого эксперта независимо калибруются через сигмоидальную регрессию (`CalibratorBank`).

---

## 🔍 Explainable AI (XAI) & Форензик-карты

В отличие от black-box моделей, VeriVision возвращает вербализацию физического вклада каждого компонента ансамбля:
* **DINOv2 Anomaly Map:** Наложение карты активаций проблемных патчей на исходный кадр.
* **ELA Inconsistency Overlay:** Визуализация зон повторного сжатия (монтажные стыки).
* **2D FFT Log-Magnitude:** Двумерная спектрограмма распределения пространственных частот.
* **PRNU Residual:** Индикатор консистентности кремниевого отпечатка сенсора.

---

## 🧪 Журнал экспериментов и эволюция модели (9 итераций)

| # | Архитектурная гипотеза | Валидация / OOD Специфика | Итог |
|---|---|---|---|
| 1 | ConvNeXt + CLIP ViT-B/32 Linear Probing | Recall Fake 2%, EER 41% | ❌ Замороженный CLIP слеп к новым латентным диффузиям |
| 2 | Heavyweight CLIP ViT-L/14 Probe | Recall 2%, EER 49% | ❌ Увеличение параметров без адаптации признаков не работает |
| 3 | DIRE (SD 1.5 Inversion Error) | $\Delta$ ошибки Real/Fake = 0.009 (уровень шума) | ❌ Слишком медленно для инференса, сигнал размывается |
| 4 | 1D Radial FFT Profile + Random Forest | Recall Fake 84%, но Recall Real 67% | ✅ Высокий recall на фейках, но большой False Positive на реальных фото |
| 5 | Hybrid Ensemble (ConvNeXt + FFT + SRM) | Total Acc 92.5% *(на синтетической выборке N=200)* | ⚠️ Вероятности сплющивались в диапазон 45–60% из-за некорректного фьюжна |
| 6 | Platt Scaling + Logistic Stacking Meta-Model | Acc 75% | ❌ Переобучение мета-модели на малом калибровочном сете |
| 7 | Scaled Fusion на датасете AI-vs-Real (~50k) | Acc упала до 61.5%, Recall 29% | ❌ Обнаружен шорткат: сеть выучила шум трейна и перестала видеть Midjourney v6 |
| 8 | Anti-Artifact Regularization + Defactify (42k) | Acc 86.0%, Precision 0.90, F1 0.89 | ✅ Преодоление шорткатов на Midjourney/SDXL |
| **9** | **DINOv2 Partial Fine-Tuning + Dense Patches + Dynamic Gating (MoE v3.5)** | **Phone Real Acc: 80.45% \| Total Val: 79.63% \| Комплексный ансамбль: 85–87%** | 🚀 **Production-состояние системы** |

> **Ключевой инсайт эксперимента №9:** Частичная разморозка 2 верхних слоев DINOv2 на разрешении $518\times518$ с мультимасштабным пулингом `[CLS; Spatial_Mean]` позволила трансформеру одновременно видеть и глобальную композицию кадра, и микротекстуру деноайзинга.

---

## 📊 Метрики и OOD-валидация

Валидация проводилась на разнородном OOD-корпусе (смесь генераций Midjourney v6, SDXL, DALL-E 3, реальных фото с камер смартфонов Apple/Samsung/Google и веб-изображений высокой четкости).

### Сравнение этапов эволюции перцептивного блока:

| Конфигурация модели | Phone Real Acc (Специфичность) | Total Val Acc | OOD Robustness |
|---|:---:|:---:|:---:|
| DINOv2 Frozen (Linear Probe 768d) | 76.69% | 69.59% | Низкая (пропуск артефактов) |
| DINOv2 Partial Fine-Tuned (Эпоха 1) | 79.70% | 71.99% | Средняя |
| **DINOv2 Partial FT + Dense Pool (Эпоха 2 — Best)** | **80.45%** | **79.63%** | **Высокая (оптимум обобщения)** |
| DINOv2 Partial FT (Эпохи 3–5) | 68.42% *(деградация)* | 74.68% | Переобучение на шум трейна |
| **Комплексный ансамбль VeriVision MoE v3.5** | **> 85%** | **84–87%** | **Мультимодальная устойчивость** |

---

## ⚡ Инженерная реализация и оптимизация

* **Serving:** Асинхронный сервер на базе **FastAPI + Uvicorn** с эндпоинтами потокового форензик-анализа (`/api/analyze` и `/api/deep-analysis`).
* **Аппаратное ускорение:** Поддержка инференса на **NVIDIA CUDA** и **Apple Silicon (MPS)** с автоматическим fallback на CPU.
* **Mixed Precision (AMP FP16):** Использование `torch.amp` ускорило прогон ViT-B/14 в 3 раза при минимальном потреблении VRAM.
* **UI/UX:** Минималистичный дизайн (Bento Grid, JetBrains Mono типографика, Glassmorphism) на шаблонизаторе Jinja2.
* **Robust File Handling:** Валидация форматов `.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`, `.heif` (через `pillow_heif`) и защита от переполнения буфера (лимит 25 МБ).

---

## 🚀 Установка и запуск

### 1. Клонирование репозитория и окружение
```bash
git clone [https://github.com/ekrtg25/VeriVision.git](https://github.com/ekrtg25/VeriVision.git)
cd VeriVision

python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Загрузка весов моделей
Поместите обученные чекпоинты в директорию models/:

models/calibrated_head.pth — веса Fine-Tuned DINOv2 и классификатора (1536d)

models/calibrators.pkl — калибровочные сигмоиды для PRNU/ELA/FFT

3. Запуск веб-сервиса
Bash
uvicorn server:app --host 0.0.0.0 --port 8080 --reload
Интерфейс будет доступен по адресу: http://localhost:8080

📁 Структура репозитория
VeriVision/
├── server.py                   # Основной FastAPI сервер и API эндпоинты
├── templates/
│   └── index.html              # Bento UI веб-интерфейс с темной темой и XAI
├── src/
│   ├── serving/
│   │   ├── ensemble.py         # VeriVisionEnsemble: Dynamic Gating, Capped Fusion, DINOv2
│   │   └── calibration.py      # CalibratorBank (Platt Scaling)
│   ├── models/
│   │   ├── forensics.py        # Извлечение ELA, оценка качества JPEG
│   │   ├── fft_module.py       # 2D FFT спектральный анализатор
│   │   ├── srm_module.py       # SRM фильтры и расчет моментов шума
│   │   └── prefilter.py        # Semantic Content Prefilter
│   └── data/                   # Аугментации (RandomJPEGCompression, Blurs)
├── research/
│   ├── notebooks/              # Исследовательские Jupyter-ноутбуки и Kaggle скрипты
│   └── experiments_log.md      # Детальный лог гипотез и метрик
├── models/                     # Директория предобученных весов (*.pth, *.pkl)
├── docs/                       # Медиа-материалы для документации
│   └── media/                  # Скриншоты работы и демо-гифка
└── requirements.txt            # Зависимости проекта
# Experiment 18: Classical Forensics Platt Scaling Calibration & False Positive Bias Analysis

**Date:** 2026-08-16  
**Module:** `serving/calibration.py`, `serving/ensemble.py`, `scripts/calibrate_from_hf.py`, `scripts/fit_calibrators.py`  
**Dataset:** `eddyfox8812/ai-vs-real-2k-images` (2,000 samples: 1,000 Real / 1,000 AI)  
**Status:** Completed with identified False-Positive Regression on In-the-Wild Real Photos  

---

## 1. Executive Summary & Objective

In this experiment, we integrated offline Platt Scaling calibration for the three classical, rule-based forensic extractors:
1. **ELA (Error Level Analysis)** — `ForensicsExtractor.compute_ela_score`
2. **PRNU (Sensor Noise Residual)** — `ForensicsExtractor.compute_prnu_residual`
3. **FFT (2D Fourier Spectral Profile)** — `FFTSpectralExtractor.extract_spectral_features`

### Goals
- Eliminate the default uncalibrated identity sigmoid fallback and runtime startup warnings.
- Map raw heterogeneous scalar metrics onto calibrated posterior probability estimates $p_i \in (0, 1)$.
- Feed calibrated probabilities into the confidence-weighted logit fusion pipeline ($w_i = |\text{logit}(p_i)|^k$).

---

## 2. Implementation & Calibration Results

### 2.1 Dataset Extraction Pipeline
Using `scripts/calibrate_from_hf.py`, we extracted scalar scores across a balanced 50/50 dataset (2,000 images total):
- 1,000 Authentic camera photographs (`label = 0`)
- 1,000 AI-generated / synthetic images (`label = 1`)

### 2.2 Platt Scaling Fitted Parameters ($p = \sigma(a \cdot x + b)$)
Logistic regression yielded the following parameter set saved to `models/calibrators.pkl`:

| Expert | Coefficient ($a$) | Intercept ($b$) | Interpretation |
| :--- | :--- | :--- | :--- |
| **ELA** | `-8.3716` | `+0.3703` | Inverse correlation with raw diff mean on compressed web photos. |
| **PRNU** | `+0.0778` | `-0.4909` | Slight positive slope on high-frequency residual variance. |
| **FFT** | `-10.3318` | `+1.2622` | High baseline intercept ($p_{base} \approx 0.78$ at $x=0$), strong negative slope. |

---

## 3. Observed Behavior & Problem Formulation

### 3.1 Observed Metrics
- **Synthetic/AI Inputs:** Achieves $\approx 100\%$ confidence (`ai_probability >= 0.99`), producing strong and correct `AI_GENERATED` verdicts.
- **Authentic/Real Photos:** Suffers from **False Positive Bias / Uncertainty Shift**, outputting $\approx 60\%$ AI probability (`ai_probability \approx 0.60`), leading to misclassification as `AI_GENERATED` or ambiguous `LOW/MODERATE` confidence bands.

### 3.2 Root Cause Analysis

#### A. Disparity in Dataset Compression & In-the-Wild Domain Shift
- The `ai-vs-real-2k-images` dataset consists of web-scraped images where "real" photos often underwent multiple compression re-encodings (social media / web compression), while synthetic images possessed different spectral roll-off characteristics.
- When an authentic, clean camera photo or uncompressed JPEG is evaluated, its ELA and FFT spectral values deviate from the distribution seen in the calibration set.

#### B. High Base Intercept on FFT ($b = 1.2622$)
- The logistic intercept for FFT is $+1.2622$. For any image where the raw spectral metric is near zero or uninformative, the default calibrated probability output by FFT is:
  $$\sigma(1.2622) \approx 0.7794 \quad (78\% \text{ AI bias})$$
- In `_confidence_weighted_fusion`, a probability of $0.78$ produces a significant logit:
  $$\text{logit}(0.78) \approx 1.265 \implies w_{\text{fft}} = (1.265)^2 \approx 1.60$$
- Even if DINOv2 predicts $p \approx 0.15$ (Real), the biased FFT expert pulls the fused score upward.

#### C. Asymmetric Confidence Penalization in Logit Fusion ($k=2.0$)
- In confidence-weighted fusion:
  $$w_i = |\text{logit}(p_i)|^k$$
- An expert voting $p=0.78$ contributes $w \approx 1.60$, whereas a conservative DINOv2 output at $p=0.35$ only contributes $w \approx 0.38$.
- The classical experts overpower the perceptual backbone on real photos.

---

## 4. Proposed Fixes & Next Steps

1. **Temperature & Prior Re-weighting:**
   - Enforce an expert weight cap ($w_{classical} \le 0.4 \cdot w_{dinov2}$) or introduce Bayesian prior regularization in `VeriVisionEnsemble`.
2. **Holdout Calibration on Clean Camera Datasets:**
   - Fit calibrators on uncompressed/standard photo benchmarks (e.g., RAISE, Dresden, FFHQ vs Midjourney/Flux) to prevent compression artifacts from corrupting classical forensic metrics.
3. **Dynamic Gating via JPEG Quality Proxy:**
   - Incorporate `ForensicsExtractor.estimate_jpeg_compression_level()` to dynamically down-weight ELA/FFT when compression exceeds thresholds.
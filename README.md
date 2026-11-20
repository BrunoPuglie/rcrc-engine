# `rcrc-engine` (`rcrc`)

**Lightweight Geospatial Inference & Multi-Date Temporal Analysis Engine for Soybean Red Crown Rot (RCR)**

`rcrc-engine` extracts the core preprocessing, L2 Regularized Logistic Regression inference (`rcrc v3.1`), and multi-temporal pixel transition pipeline from the Red Crown Rot Classifier research framework into a clean, pip-installable Python library (`import rcrc`).

Built for **Google Colab**, **VS Code / Jupyter**, batch HPC/CLI workflows, and **R (`reticulate`)** without requiring `scikit-learn`, `joblib`, or Streamlit.

---

## Key Features

- **Zero Heavy ML Dependencies**: Evaluates L2 logistic inference directly from stored weights (`MODEL_REGISTRY["v3.1"]`) using `numpy` and `scipy.special.expit`, installing in seconds.
- **PlanetScope 8-Band Native**: Designed for `3B_AnalyticMS_SR_8b_harmonized` imagery (Band 2 Blue, Band 4 Green, Band 5 Yellow, Band 6 Red, Band 7 Red Edge, Band 8 NIR).
- **Automated Canopy Masking & Local Normalization**:
  - Clips scenes and Peak-NDVI baseline imagery to field boundaries (with optional metric inward buffer, e.g., `buffer_m=-3.0`).
  - Applies the early/peak-season **3-SD NDVI canopy mask** (`mean - 3.0 * SD`) to exclude bare soil, grassed waterways, and non-crop pixels.
  - Computes the 13 `rcrc v3.1` spectral features (NDVI, GNDVI, NDRE, EVI, CI, CIRe, CSI, plus Bands 6, 8, 7, 4, 2, 5 — excluding TDVI) with local per-field `[0, 1]` min-max normalization followed by training-set z-score standardization.
- **Cloud-Optimized GeoTIFF (COG) Export**: Outputs continuous probability `P(RCR)` and binary classification (`1 = RCR Diseased`, `0 = Healthy`, `-1 = nodata`) COGs.
- **Multi-Date Temporal Stack & Transition Tracking**:
  - Aligns chronological scenes via `rioxarray.reproject_match` and enforces a shared seasonal canopy mask.
  - Tracks pixel-level state transitions (`0`: Unaffected `0->0`, `1`: New Expansion `0->1`, `10`: Apparent Recovery `1->0`, `11`: Persistent Infection `1->1`).
  - Produces tidy `pandas.DataFrame` tables ready for LMM/GLMM disease progression analysis in R (`lme4` / `glmmTMB`) or Python (`statsmodels`).

---

## Installation

### Local / Editable Install
```bash
git clone https://github.com/BrunoPuglie/rcrc-engine.git
cd rcrc-engine
pip install -e .
```

### Google Colab
```python
!pip install -q git+https://github.com/BrunoPuglie/rcrc-engine.git
```

### R (`reticulate`)
```r
library(reticulate)
py_install("git+https://github.com/BrunoPuglie/rcrc-engine.git", pip = TRUE)
rcrc <- import("rcrc")
```

---

## Decision Threshold (`0.48`)

`rcrc v3.1` classifies a pixel as RCR-diseased when `P(RCR) >= 0.48`. The threshold is fixed as part of the model specification (`MODEL_REGISTRY["v3.1"]["threshold"]`) and is applied automatically by `predict_scene()` and `analyze_temporal_stack()`; it is not a user-tunable argument.

| Threshold | Empirical Basis (LOFO-CV, $N=2{,}921$, 15 fields) | Accuracy | Balanced Acc | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`0.48`** | Peak out-of-fold F1 & Balanced Accuracy; median in-fold LOFO tuned cutoff (`0.4796`) | 0.945 | 0.945 | 0.954 | 0.941 | 0.947 |

---

## Quickstart

### 1. Single-Date Preprocessing & Inference
```python
import rcrc

# Preprocess 8-band PlanetScope scene using Peak-NDVI 3-SD canopy mask
# Optional buffer_m=-3.0 removes ~1 pixel along field borders
ds = rcrc.preprocess_scene(
    scene_path="data/BNC1_20260808_3B_AnalyticMS_SR_8b_harmonized.tif",
    peak_ndvi_path="data/BNC1_20260701_3B_AnalyticMS_SR_8b_harmonized.tif",
    boundary_path="data/Boun_BNC1.gpkg",
    sd_mask=3.0,
    buffer_m=-3.0,
)

# Run rcrc v3.1 inference and export Cloud-Optimized GeoTIFFs
summary = rcrc.predict_scene(
    preprocessed_data=ds,
    model_version="v3.1",
    output_dir="outputs/single_date",
    prefix="BNC1_20260808",
)
print(summary)
```

### 2. Multi-Date Temporal Progression Analysis
```python
import rcrc

scenes = {
    "2026-07-27": "data/BNC1_20260727_3B_AnalyticMS_SR_8b_harmonized.tif",
    "2026-08-08": "data/BNC1_20260808_3B_AnalyticMS_SR_8b_harmonized.tif",
    "2026-08-24": "data/BNC1_20260824_3B_AnalyticMS_SR_8b_harmonized.tif",
}

results = rcrc.analyze_temporal_stack(
    scene_dict=scenes,
    peak_ndvi_path="data/BNC1_20260701_3B_AnalyticMS_SR_8b_harmonized.tif",
    boundary_path="data/Boun_BNC1.gpkg",
    model_version="v3.1",
    sd_mask=3.0,
    buffer_m=-3.0,
    field_id="BNC1",
    output_dir="outputs/temporal_BNC1",
)

progression_df = results["progression_df"]
transitions_df = results["transitions_df"]
temporal_stack = results["temporal_stack"]  # xarray.Dataset
print(progression_df[["date_iso", "days_since_start", "rcr_affected_ha", "incidence_pct", "new_expansion_ha"]])
```

### 3. Using `rcrc` from R (`reticulate` + `glmmTMB`)
```r
library(reticulate)
library(glmmTMB)

rcrc <- import("rcrc")
res <- rcrc$analyze_temporal_stack(
  scene_dict = dict(
    "2026-07-27" = "data/BNC1_20260727.tif",
    "2026-08-08" = "data/BNC1_20260808.tif",
    "2026-08-24" = "data/BNC1_20260824.tif"
  ),
  peak_ndvi_path = "data/BNC1_20260701.tif",
  boundary_path = "data/Boun_BNC1.gpkg",
  field_id = "BNC1"
)

prog_df <- py_to_r(res$progression_df)
fit <- glmmTMB(cbind(rcr_pixels, healthy_pixels) ~ days_since_start,
               family = betabinomial(), data = prog_df)
summary(fit)
```

### 4. Registering Future Model Versions (e.g., `v3.2` / `v4.0`)
```python
import rcrc

rcrc.register_model(
    version="v3.2",
    spec={
        "feature_names": rcrc.FEATURE_NAMES_V3_1,
        "scaler_mean": [...],
        "scaler_scale": [...],
        "coefficients": [...],
        "intercept": 0.0,
        "threshold": 0.48,
    },
)
```

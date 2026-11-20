<h1 align="center">rcrc-engine</h1>

<p align="center">
  Lightweight geospatial inference and temporal analysis for<br>
  <b>Soybean Red Crown Rot (RCR)</b>
</p>

<p align="center">
  <a href="https://doi.org/10.3390/rs18213839"><img src="https://img.shields.io/badge/DOI-10.3390%2Frs18213839-DC2626?style=flat-square" alt="DOI"></a>
  <img src="https://img.shields.io/badge/model-rcrc%20v3.1-3B82F6?style=flat-square" alt="model">
  <img src="https://img.shields.io/badge/python-3.9%2B-555?style=flat-square" alt="python">
  <img src="https://img.shields.io/badge/imagery-PlanetScope%208--band-555?style=flat-square" alt="imagery">
</p>

`rcrc-engine` packages the preprocessing, L2-regularized logistic inference (`rcrc v3.1`) and multi-date pixel transition pipeline of the Red Crown Rot Classifier into a pip-installable library (`import rcrc`). It runs in **Google Colab**, **VS Code / Jupyter**, **HPC / CLI** and **R (`reticulate`)**, with no `scikit-learn`, `joblib` or Streamlit.

---

## Features

| | |
| :--- | :--- |
| **Light dependencies** | Inference from stored weights (`MODEL_REGISTRY["v3.1"]`) with `numpy` and `scipy`. Installs in seconds. |
| **PlanetScope native** | `3B_AnalyticMS_SR_8b_harmonized`: Blue, Green, Yellow, Red, Red Edge, NIR. |
| **Canopy masking** | Clip to field boundary (optional `buffer_m=-3.0`), then a 3-SD Peak-NDVI mask (`mean - 3.0 * SD`) removes soil, waterways and non-crop pixels. |
| **13 spectral features** | NDVI, GNDVI, NDRE, EVI, CI, CIRe, CSI plus six bands. Per-field `[0, 1]` normalization, then z-score standardization. |
| **COG export** | Continuous `P(RCR)` and binary class rasters (`1` diseased, `0` healthy, `-1` nodata). |
| **Temporal tracking** | Aligns scenes, shares one canopy mask and tracks pixel transitions. Outputs tidy `pandas` tables for `lme4`, `glmmTMB` or `statsmodels`. |

### Transition classes

| | Code | State | Change |
| :---: | :---: | :--- | :---: |
| ![](https://img.shields.io/badge/%20%20%20%20%20-D8F3DC?style=flat-square) | `0` | Unaffected | `0 → 0` |
| ![](https://img.shields.io/badge/%20%20%20%20%20-F59E0B?style=flat-square) | `1` | New expansion | `0 → 1` |
| ![](https://img.shields.io/badge/%20%20%20%20%20-3B82F6?style=flat-square) | `10` | Apparent recovery | `1 → 0` |
| ![](https://img.shields.io/badge/%20%20%20%20%20-DC2626?style=flat-square) | `11` | Persistent infection | `1 → 1` |

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

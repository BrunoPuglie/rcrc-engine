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
| **NDVI date guidance** | Season NDVI trajectory from free Sentinel-2 (no account needed) or local PlanetScope scenes gives the peak-canopy date for the mask and the optimal detection date `t0` (80 % of peak) with a +/-5-day window. |
| **Temporal tracking** | Aligns scenes, shares one canopy mask and tracks pixel transitions. Outputs tidy `pandas` tables for `lme4`, `glmmTMB` or `statsmodels`. |

---

## Installation

### Local / Editable Install
```bash
git clone https://github.com/BrunoPuglie/rcrc-engine.git
cd rcrc-engine
pip install -e .
```

### Development with [uv](https://docs.astral.sh/uv/)
```bash
git clone https://github.com/BrunoPuglie/rcrc-engine.git
cd rcrc-engine
uv sync --extra dev      # creates .venv and installs rcrc with dev extras
uv run pytest
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

### 0. NDVI Trajectory Inspection (choose dates)
Run this before the detection steps. It finds the **peak canopy** date (used for `peak_ndvi_path`) and the **optimal detection date** `t0`, the first post-peak point where the daily NDVI trajectory reaches 80 % of peak, with a +/-5-day window around it.

```python
from pathlib import Path
import rcrc

# Free Sentinel-2 L2A from the public AWS Earth Search catalog
traj = rcrc.analyze_ndvi_phenology(
    boundary_path="data/Boun_BNC1.gpkg",
    year=2026,
    cache_csv="outputs/BNC1_2026_ndvi.csv",   # optional: reused on later runs, no download
)

# Alternative: PlanetScope scenes already on disk
# traj = rcrc.analyze_ndvi_phenology(..., source="planetscope",
#                              scenes=sorted(Path("data").glob("BNC1_*_8b_harmonized.tif")))

print(traj.summary())  

# Pick the available PlanetScope scene closest to each key date
matched = rcrc.match_scenes_to_trajectory(traj, sorted(Path("data").glob("BNC1_*.tif")))
rcrc.plot_ndvi_trajectory(traj, title="BNC1 2026", matched=matched)   # needs matplotlib
```

`matched` gives the `peak` scene for `peak_ndvi_path` and the `before` / `target` / `after` scenes for `scene_dict` in step 2.

The default source reads only the field window of the Sentinel-2 red, NIR and scene-classification Cloud-Optimized GeoTIFFs over HTTPS (one season takes seconds). It masks clouds, shadow and cirrus with the same SCL classes as rcr-scout, harmonizes values across Sentinel-2 processing baselines, and drops dates where less than half of the field is clear (`min_clear_fraction` in `rcrc.fetch_sentinel2_ndvi`). Any table with `date` and `ndvi` columns can also go straight to `rcrc.compute_ndvi_trajectory(df)`.

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

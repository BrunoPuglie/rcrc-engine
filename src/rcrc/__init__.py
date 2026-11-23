"""rcrc-engine: Lightweight Soybean Red Crown Rot (RCR) classification library.

Provides geospatial preprocessing, pure NumPy/SciPy L2 logistic inference,
multi-date temporal disease progression analysis, and seasonal NDVI
trajectory inspection to guide detection-date selection for 8-band PlanetScope
imagery (``3B_AnalyticMS_SR_8b_harmonized``).
"""

from __future__ import annotations

from rcrc.models import (
    DEFAULT_MODEL_VERSION,
    FEATURE_NAMES_V3_1,
    MODEL_REGISTRY,
    compute_probabilities_from_features,
    get_model_spec,
    list_models,
    register_model,
)
from rcrc.phenology import (
    NDVITrajectory,
    analyze_ndvi_phenology,
    compute_ndvi_trajectory,
    fetch_sentinel2_ndvi,
    match_scenes_to_trajectory,
    planetscope_ndvi_series,
    plot_ndvi_trajectory,
)
from rcrc.predict import export_cog, predict_scene
from rcrc.preprocess import (
    PLANET_8B_INDICES,
    compute_canopy_mask,
    compute_spectral_features,
    load_boundary,
    min_max_scale_canopy,
    preprocess_scene,
)
from rcrc.temporal import (
    TRANSITION_LABELS,
    TRANSITION_NEW_EXPANSION,
    TRANSITION_NODATA,
    TRANSITION_PERSISTENT,
    TRANSITION_RECOVERY,
    TRANSITION_UNAFFECTED,
    analyze_temporal_stack,
    compute_transition_matrix,
)

__version__ = "3.1.0"

__all__ = [
    "__version__",
    "DEFAULT_MODEL_VERSION",
    "FEATURE_NAMES_V3_1",
    "MODEL_REGISTRY",
    "NDVITrajectory",
    "PLANET_8B_INDICES",
    "TRANSITION_LABELS",
    "TRANSITION_NEW_EXPANSION",
    "TRANSITION_NODATA",
    "TRANSITION_PERSISTENT",
    "TRANSITION_RECOVERY",
    "TRANSITION_UNAFFECTED",
    "analyze_ndvi_phenology",
    "analyze_temporal_stack",
    "compute_canopy_mask",
    "compute_ndvi_trajectory",
    "compute_probabilities_from_features",
    "compute_spectral_features",
    "compute_transition_matrix",
    "export_cog",
    "fetch_sentinel2_ndvi",
    "get_model_spec",
    "list_models",
    "load_boundary",
    "match_scenes_to_trajectory",
    "min_max_scale_canopy",
    "planetscope_ndvi_series",
    "plot_ndvi_trajectory",
    "predict_scene",
    "preprocess_scene",
    "register_model",
]

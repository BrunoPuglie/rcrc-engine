"""rcrc-engine: Lightweight Soybean Red Crown Rot (RCR) classification library.

Provides geospatial preprocessing, pure NumPy/SciPy L2 logistic inference,
and multi-date temporal disease progression analysis for 8-band PlanetScope
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
    "PLANET_8B_INDICES",
    "TRANSITION_LABELS",
    "TRANSITION_NEW_EXPANSION",
    "TRANSITION_NODATA",
    "TRANSITION_PERSISTENT",
    "TRANSITION_RECOVERY",
    "TRANSITION_UNAFFECTED",
    "analyze_temporal_stack",
    "compute_canopy_mask",
    "compute_probabilities_from_features",
    "compute_spectral_features",
    "compute_transition_matrix",
    "export_cog",
    "get_model_spec",
    "list_models",
    "load_boundary",
    "min_max_scale_canopy",
    "predict_scene",
    "preprocess_scene",
    "register_model",
]

"""Model registry and pure NumPy/SciPy L2 logistic inference for ``rcrc``.

This module stores calibrated model parameters in ``MODEL_REGISTRY`` so
collaborators can run inference in seconds without installing heavy machine
learning frameworks (no ``scikit-learn`` or ``joblib`` dependency required).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
from scipy.special import expit

logger = logging.getLogger(__name__)

DEFAULT_MODEL_VERSION: str = "v3.1"

FEATURE_NAMES_V3_1: tuple[str, ...] = (
    "NormNDVI",
    "NormGNDVI",
    "NormNDRE",
    "NormEVI",
    "NormCI",
    "NormCIRe",
    "NormCSI",
    "Normb6",
    "Normb8",
    "Normb7",
    "Normb4",
    "Normb2",
    "Normb5",
)

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "v3.1": {
        "version": "v3.1",
        "description": (
            "Red Crown Rot Classifier (RCRC v3.1): L2 Regularized Logistic "
            "Regression trained on 8-band PlanetScope SuperDove imagery "
            "(3B_AnalyticMS_SR_8b_harmonized) across 15 soybean fields."
        ),
        "sensor": "PlanetScope SuperDove (3B_AnalyticMS_SR_8b_harmonized)",
        "algorithm": "L2 Regularized Logistic Regression (StandardScaler + LogisticRegression)",
        "training_samples": 2921,
        "training_fields": 15,
        "feature_names": FEATURE_NAMES_V3_1,
        "local_normalization": "per_field_minmax_0_1",
        "scaler_mean": (
            0.6442477233824032,   # NormNDVI
            0.5649699418007532,   # NormGNDVI
            0.5412233139335844,   # NormNDRE
            0.5583149606299213,   # NormEVI
            0.3837318384115030,   # NormCI
            0.3812408079424855,   # NormCIRe
            0.34722300581992466,  # NormCSI
            0.36865518657993845,  # Normb6 (Red)
            0.5395882574460801,   # Normb8 (NIR)
            0.47207343375556315,  # Normb7 (Red Edge)
            0.44867158507360494,  # Normb4 (Green)
            0.40700078740157475,  # Normb2 (Blue)
            0.42492543649435127,  # Normb5 (Yellow)
        ),
        "scaler_scale": (
            0.20974304013505268,  # NormNDVI
            0.21534894810312780,  # NormGNDVI
            0.22714737922136574,  # NormNDRE
            0.21595882998054605,  # NormEVI
            0.20764108428552538,  # NormCI
            0.22199130112069770,  # NormCIRe
            0.14525318010768784,  # NormCSI
            0.20268370294259525,  # Normb6 (Red)
            0.20682909794647800,  # Normb8 (NIR)
            0.20925778614128784,  # Normb7 (Red Edge)
            0.19825278081663130,  # Normb4 (Green)
            0.16301410499708402,  # Normb2 (Blue)
            0.21784437239791020,  # Normb5 (Yellow)
        ),
        "coefficients": (
            -1.5189893985455205,  # NormNDVI
            0.7818549317462044,   # NormGNDVI
            -0.7053159084901133,  # NormNDRE
            -4.3913586259622410,  # NormEVI
            -2.1604856336401626,  # NormCI
            1.2389203511571830,   # NormCIRe
            0.3022369882385244,   # NormCSI
            2.1069764660599790,   # Normb6 (Red)
            1.5564220173443883,   # Normb8 (NIR)
            1.4954661590249930,   # Normb7 (Red Edge)
            -1.2982171931109923,  # Normb4 (Green)
            0.6501833903122310,   # Normb2 (Blue)
            -2.5777541659007840,  # Normb5 (Yellow)
        ),
        "intercept": 0.9944934996292226,
        # Fixed decision threshold: maximizes LOFO-CV F1 (0.947) and Balanced
        # Accuracy (0.945); median in-fold LOFO tuned cutoff = 0.4796.
        "threshold": 0.48,
        "classes": {
            0: "Healthy / Asymptomatic",
            1: "RCR Diseased",
        },
    }
}


def list_models() -> list[str]:
    """Return sorted list of registered model version identifiers."""
    return sorted(MODEL_REGISTRY.keys())


def get_model_spec(model_version: str = DEFAULT_MODEL_VERSION) -> dict[str, Any]:
    """Retrieve a copy of the specification dictionary for a model version.

    Args:
        model_version: Model identifier key in ``MODEL_REGISTRY`` (e.g., ``"v3.1"``).

    Returns:
        Dictionary containing feature names, normalization parameters, L2
        coefficients, intercept, and decision threshold.

    Raises:
        KeyError: If ``model_version`` is not present in ``MODEL_REGISTRY``.
    """
    if model_version not in MODEL_REGISTRY:
        available = ", ".join(list_models())
        raise KeyError(
            f"Unknown model version '{model_version}'. Available models: [{available}]."
        )
    return dict(MODEL_REGISTRY[model_version])


def register_model(
    version: str,
    spec: Mapping[str, Any],
    overwrite: bool = False,
) -> None:
    """Register a new model version in ``MODEL_REGISTRY``.

    Enables collaborators or future package releases (e.g., ``"v3.2"`` or
    ``"v4.0"``) to plug in new feature sets, scaling parameters, and L2 weights
    without altering the public inference API.

    Args:
        version: Unique model version identifier (e.g., ``"v3.2"``).
        spec: Mapping containing ``feature_names``, ``coefficients``,
            ``intercept``, and ``threshold`` (the model's fixed classification
            cutoff in ``[0, 1]``). Optional ``scaler_mean``
            and ``scaler_scale`` arrays may be provided for feature
            standardization prior to the dot product.
        overwrite: Whether to replace an existing registry entry.

    Raises:
        ValueError: If the version already exists (when ``overwrite=False``) or
            if required fields / array dimensions are invalid.
    """
    if version in MODEL_REGISTRY and not overwrite:
        raise ValueError(
            f"Model version '{version}' is already registered. "
            "Pass overwrite=True to replace it."
        )

    required_keys = {"feature_names", "coefficients", "intercept", "threshold"}
    missing = required_keys - set(spec.keys())
    if missing:
        raise ValueError(f"Model spec is missing required keys: {sorted(missing)}")

    feature_names = tuple(spec["feature_names"])
    coefs = np.asarray(spec["coefficients"], dtype=np.float64)
    if coefs.ndim != 1 or len(coefs) != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} coefficients, got shape {coefs.shape}."
        )
    if not np.all(np.isfinite(coefs)):
        raise ValueError("All model coefficients must be finite numbers.")

    intercept = float(spec["intercept"])
    if not np.isfinite(intercept):
        raise ValueError("Model intercept must be a finite number.")

    threshold = float(spec["threshold"])
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")

    scaler_mean = spec.get("scaler_mean")
    scaler_scale = spec.get("scaler_scale")
    if (scaler_mean is None) != (scaler_scale is None):
        raise ValueError("Provide both scaler_mean and scaler_scale, or neither.")

    normalized_spec: dict[str, Any] = dict(spec)
    normalized_spec["version"] = version
    normalized_spec["feature_names"] = feature_names
    normalized_spec["coefficients"] = tuple(float(c) for c in coefs)
    normalized_spec["intercept"] = intercept
    normalized_spec["threshold"] = threshold

    if scaler_mean is not None and scaler_scale is not None:
        mean_arr = np.asarray(scaler_mean, dtype=np.float64)
        scale_arr = np.asarray(scaler_scale, dtype=np.float64)
        if mean_arr.shape != (len(feature_names),) or scale_arr.shape != (len(feature_names),):
            raise ValueError(
                "scaler_mean and scaler_scale must match the number of feature_names."
            )
        if np.any(scale_arr <= 0) or not np.all(np.isfinite(mean_arr)) or not np.all(np.isfinite(scale_arr)):
            raise ValueError("scaler_scale values must be strictly positive and finite.")
        normalized_spec["scaler_mean"] = tuple(float(v) for v in mean_arr)
        normalized_spec["scaler_scale"] = tuple(float(v) for v in scale_arr)

    MODEL_REGISTRY[version] = normalized_spec
    logger.info("Registered model version '%s' with %d features.", version, len(feature_names))


def compute_probabilities_from_features(
    features: np.ndarray,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> np.ndarray:
    """Compute RCR probabilities directly from normalized feature vectors.

    Applies the registered training-set standardization (when configured) and
    evaluates the logistic sigmoid of the linear L2 predictor:

    .. math::
        P(\\text{RCR}) = \\sigma\\left(\\beta_0 + \\sum_{i=1}^{p} \\beta_i
        \\frac{x_i - \\mu_i}{s_i}\\right)

    Args:
        features: 2D array of shape ``(n_samples, n_features)`` ordered
            according to ``MODEL_REGISTRY[model_version]["feature_names"]``.
        model_version: Model identifier key in ``MODEL_REGISTRY``.

    Returns:
        1D ``float32`` array of probabilities in ``[0, 1]`` of length
        ``n_samples``.

    Raises:
        ValueError: If ``features`` does not have 2 dimensions or its second
            dimension does not match the model's feature count.
    """
    spec = get_model_spec(model_version)
    n_expected = len(spec["feature_names"])

    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != n_expected:
        raise ValueError(
            f"Expected features with shape (N, {n_expected}) for model "
            f"'{model_version}', got {X.shape}."
        )

    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float32)

    scaler_mean = spec.get("scaler_mean")
    scaler_scale = spec.get("scaler_scale")
    if scaler_mean is not None and scaler_scale is not None:
        mean_arr = np.asarray(scaler_mean, dtype=np.float64)
        scale_arr = np.asarray(scaler_scale, dtype=np.float64)
        X_scaled = (X - mean_arr) / scale_arr
    else:
        X_scaled = X

    coef = np.asarray(spec["coefficients"], dtype=np.float64)
    intercept = float(spec["intercept"])
    logits = intercept + X_scaled @ coef
    probs = expit(logits)
    return probs.astype(np.float32)

"""Unit and numerical parity tests for rcrc-engine."""

from __future__ import annotations

import numpy as np
import pytest

from rcrc.models import (
    DEFAULT_MODEL_VERSION,
    FEATURE_NAMES_V3_1,
    MODEL_REGISTRY,
    compute_probabilities_from_features,
    get_model_spec,
    list_models,
    register_model,
)


def test_registry_defaults() -> None:
    assert DEFAULT_MODEL_VERSION == "v3.1"
    assert "v3.1" in list_models()
    spec = get_model_spec("v3.1")
    assert len(spec["feature_names"]) == 13
    assert spec["feature_names"] == FEATURE_NAMES_V3_1
    assert "NormTDVI" not in spec["feature_names"]
    assert "NormCSI" in spec["feature_names"]
    assert "NormNDRE" in spec["feature_names"]
    assert spec["default_threshold"] == pytest.approx(0.44)
    assert spec["optimal_f1_threshold"] == pytest.approx(0.48)


def test_register_future_model_version() -> None:
    test_version = "v9.9-test"
    register_model(
        version=test_version,
        spec={
            "feature_names": ("NormNDVI", "NormEVI"),
            "scaler_mean": (0.5, 0.5),
            "scaler_scale": (0.2, 0.2),
            "coefficients": (-1.0, -2.0),
            "intercept": 0.1,
            "default_threshold": 0.50,
        },
        overwrite=True,
    )
    probs = compute_probabilities_from_features(
        np.array([[0.5, 0.5], [0.1, 0.1]], dtype=np.float32),
        model_version=test_version,
    )
    assert probs.shape == (2,)
    assert probs[0] == pytest.approx(1.0 / (1.0 + np.exp(-0.1)), rel=1e-6)
    assert probs[1] > probs[0]
    MODEL_REGISTRY.pop(test_version, None)


def test_compute_probabilities_parity_with_v3_1_weights() -> None:
    # First row (PT_0001) of data/processed/tables/rcr_features_curated.csv (Healthy, Dise_Cali=0)
    x_pt1 = np.array(
        [[
            0.8605, 0.8351, 0.7568, 0.8079, 0.6535, 0.5132, 0.4617,
            0.1982, 0.8350, 0.3039, 0.2340, 0.3473, 0.1380,
        ]],
        dtype=np.float64,
    )
    spec = get_model_spec("v3.1")
    z = spec["intercept"] + np.sum(
        ((x_pt1[0] - np.array(spec["scaler_mean"])) / np.array(spec["scaler_scale"]))
        * np.array(spec["coefficients"])
    )
    expected = float(1.0 / (1.0 + np.exp(-z)))
    prob = compute_probabilities_from_features(x_pt1, model_version="v3.1")
    assert prob.dtype == np.float32
    assert prob[0] == pytest.approx(expected, abs=1e-6)
    assert prob[0] < 0.44


def test_spectral_features_and_minmax_scaling() -> None:
    pytest.importorskip("rioxarray")
    from rcrc.preprocess import compute_spectral_features, min_max_scale_canopy

    # Synthetic 8-band PlanetScope array (bands 1..8, DN values scaled by 10000)
    arr = np.zeros((8, 4, 4), dtype=np.float32)
    arr[1] = 400.0   # b2 Blue = 0.04
    arr[3] = 600.0   # b4 Green = 0.06
    arr[4] = 700.0   # b5 Yellow = 0.07
    arr[5] = 500.0   # b6 Red = 0.05
    arr[6] = 1200.0  # b7 Red Edge = 0.12
    arr[7] = 4000.0  # b8 NIR = 0.40

    feats = compute_spectral_features(arr, reflectance_scale=10000.0)
    assert set(feats.keys()) == {
        "NDVI", "GNDVI", "NDRE", "EVI", "CI", "CIRe", "CSI",
        "b6", "b8", "b7", "b4", "b2", "b5",
    }
    assert feats["NDVI"][0, 0] == pytest.approx((0.40 - 0.05) / (0.40 + 0.05), rel=1e-5)
    assert feats["NDRE"][0, 0] == pytest.approx((0.40 - 0.12) / (0.40 + 0.12), rel=1e-5)

    # Check local per-field min-max scaling strictly within canopy mask
    raw = np.array([[0.1, 0.5], [0.9, 99.0]], dtype=np.float32)
    mask = np.array([[True, True], [True, False]])
    scaled = min_max_scale_canopy(raw, mask)
    assert scaled[0, 0] == pytest.approx(0.0)
    assert scaled[0, 1] == pytest.approx(0.5)
    assert scaled[1, 0] == pytest.approx(1.0)
    assert np.isnan(scaled[1, 1])


def test_transition_matrix_codes() -> None:
    pytest.importorskip("rioxarray")
    from rcrc.temporal import (
        TRANSITION_NEW_EXPANSION,
        TRANSITION_NODATA,
        TRANSITION_PERSISTENT,
        TRANSITION_RECOVERY,
        TRANSITION_UNAFFECTED,
        compute_transition_matrix,
    )

    t1 = np.array([[0, 0], [1, 1], [-1, 0]], dtype=np.int8)
    t2 = np.array([[0, 1], [0, 1], [1, -1]], dtype=np.int8)
    trans = compute_transition_matrix(t1, t2)

    assert trans[0, 0] == TRANSITION_UNAFFECTED     # 0 -> 0 = 0
    assert trans[0, 1] == TRANSITION_NEW_EXPANSION  # 0 -> 1 = 1
    assert trans[1, 0] == TRANSITION_RECOVERY       # 1 -> 0 = 10
    assert trans[1, 1] == TRANSITION_PERSISTENT     # 1 -> 1 = 11
    assert trans[2, 0] == TRANSITION_NODATA         # masked
    assert trans[2, 1] == TRANSITION_NODATA         # masked

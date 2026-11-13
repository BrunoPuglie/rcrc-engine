"""Single-date RCR probability inference and Cloud-Optimized GeoTIFF export.

Computes pixel-wise Red Crown Rot probability ``P(RCR)`` and binary
classification maps directly from normalized feature stacks using the weights
stored in ``rcrc.models.MODEL_REGISTRY``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling as OverviewResampling
from rasterio.errors import DriverRegistrationError, RasterioIOError
from rasterio.transform import Affine
import rioxarray  # noqa: F401
import xarray as xr

from rcrc.models import (
    DEFAULT_MODEL_VERSION,
    compute_probabilities_from_features,
    get_model_spec,
)

logger = logging.getLogger(__name__)


def _extract_spatial_metadata(obj: xr.Dataset | xr.DataArray) -> tuple[Any, Affine, float]:
    """Extract CRS, affine transform, and pixel area in m^2 from an xarray object."""
    crs = obj.rio.crs
    transform = obj.rio.transform()
    if crs is None:
        crs_wkt = obj.attrs.get("crs_wkt")
        if crs_wkt:
            crs = rasterio.CRS.from_wkt(crs_wkt)
    if transform is None or transform == Affine.identity():
        affine_tuple = obj.attrs.get("transform_affine")
        if affine_tuple is not None:
            transform = Affine(*affine_tuple)

    pixel_area_m2 = float(obj.attrs.get("pixel_area_m2", abs(transform.a * transform.e)))
    return crs, transform, pixel_area_m2


def export_cog(
    data: np.ndarray,
    out_path: str | Path,
    crs: Any,
    transform: Affine,
    dtype: str = "float32",
    nodata: float | int = -9999.0,
    band_descriptions: Sequence[str] | None = None,
) -> Path:
    """Export a 2D or 3D NumPy array as a Cloud-Optimized GeoTIFF (COG).

    Attempts to write directly via GDAL's ``COG`` driver with ``DEFLATE``
    compression. If the local GDAL installation does not include the ``COG``
    driver, falls back to a DEFLATE-compressed, tiled GeoTIFF with internal
    overview pyramids.

    Args:
        data: 2D ``(H, W)`` or 3D ``(bands, H, W)`` array.
        out_path: Destination ``.tif`` file path.
        crs: Coordinate reference system readable by ``rasterio``.
        transform: Affine geotransform for the raster grid.
        dtype: Output NumPy/rasterio dtype string (e.g., ``"float32"``, ``"int8"``).
        nodata: Nodata sentinel value written into raster metadata.
        band_descriptions: Optional per-band description labels.

    Returns:
        Resolved ``Path`` to the written GeoTIFF.
    """
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array for COG export, got shape {arr.shape}.")

    count, height, width = arr.shape
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.floating):
        clean_arr = np.where(np.isfinite(arr), arr, nodata).astype(np_dtype)
        overview_resamp = OverviewResampling.average
    else:
        clean_arr = np.nan_to_num(arr, nan=float(nodata)).astype(np_dtype)
        overview_resamp = OverviewResampling.nearest

    cog_profile = {
        "driver": "COG",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(np_dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "DEFLATE",
        "overview_resampling": overview_resamp.name.upper(),
    }

    try:
        with rasterio.open(dest, "w", **cog_profile) as dst:
            dst.write(clean_arr)
            if band_descriptions:
                for idx, desc in enumerate(band_descriptions, start=1):
                    dst.set_band_description(idx, str(desc))
    except (DriverRegistrationError, RasterioIOError) as exc:
        logger.debug(
            "COG driver unavailable (%s); falling back to tiled GTiff with overviews.",
            exc,
        )
        gtiff_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": str(np_dtype),
            "crs": crs,
            "transform": transform,
            "nodata": nodata,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
        }
        with rasterio.open(dest, "w", **gtiff_profile) as dst:
            dst.write(clean_arr)
            if band_descriptions:
                for idx, desc in enumerate(band_descriptions, start=1):
                    dst.set_band_description(idx, str(desc))
            if min(height, width) >= 64:
                factors = [f for f in (2, 4, 8) if min(height, width) // f >= 16]
                if factors:
                    dst.build_overviews(factors, overview_resamp)
                    dst.update_tags(ns="rio_overview", resampling=overview_resamp.name)

    logger.info("Exported COG -> %s", dest)
    return dest


def predict_scene(
    preprocessed_data: xr.Dataset | xr.DataArray,
    model_version: str = DEFAULT_MODEL_VERSION,
    threshold: float = 0.44,
    output_dir: str | Path | None = None,
    prefix: str = "scene",
    return_rasters: bool = False,
) -> dict[str, Any]:
    """Run pixel-wise RCR logistic inference on a preprocessed feature stack.

    Evaluates ``P(RCR)`` over valid canopy pixels using the calibrated weights
    in ``MODEL_REGISTRY[model_version]``, applies ``threshold`` to produce binary
    classifications (``1 = RCR Diseased``, ``0 = Healthy``, ``-1 = nodata``),
    optionally exports Cloud-Optimized GeoTIFFs (COGs), and returns a summary
    metrics dictionary.

    Args:
        preprocessed_data: Output ``xarray.Dataset`` from
            :func:`rcrc.preprocess.preprocess_scene`, or a 3D ``xarray.DataArray``
            of shape ``(n_features, y, x)``.
        model_version: Registered model version key (default ``"v3.1"``).
        threshold: Classification probability cutoff in ``[0, 1]`` (default
            ``0.44``; note ``0.48`` maximizes LOFO-CV F1 and Balanced Accuracy).
        output_dir: Optional directory where probability and binary classified
            COGs are written.
        prefix: Filename prefix used when ``output_dir`` is provided.
        return_rasters: If ``True``, includes ``"prediction_ds"`` (an
            ``xarray.Dataset`` containing ``probability`` and ``classification``
            ``DataArray`` layers) in the returned dictionary.

    Returns:
        Summary dictionary containing area, incidence, probability statistics,
        and optional exported COG file paths.

    Raises:
        ValueError: If ``threshold`` is outside ``[0, 1]`` or feature dimensions
            do not match the selected model.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}.")

    spec = get_model_spec(model_version)
    expected_features = list(spec["feature_names"])

    if isinstance(preprocessed_data, xr.Dataset):
        if "features" not in preprocessed_data:
            raise ValueError("Input Dataset must contain a 'features' variable.")
        features_da = preprocessed_data["features"]
        crs, transform, pixel_area_m2 = _extract_spatial_metadata(preprocessed_data)
        mask_arr = (
            preprocessed_data["canopy_mask"].to_numpy().astype(bool)
            if "canopy_mask" in preprocessed_data
            else None
        )
    elif isinstance(preprocessed_data, xr.DataArray):
        features_da = preprocessed_data
        crs, transform, pixel_area_m2 = _extract_spatial_metadata(preprocessed_data)
        mask_arr = None
    else:
        raise TypeError(
            f"preprocessed_data must be an xarray.Dataset or xarray.DataArray, "
            f"got {type(preprocessed_data).__name__}."
        )

    if "feature" in features_da.dims:
        features_da = features_da.sel(feature=expected_features)
        feat_stack = features_da.to_numpy().astype(np.float32)
        y_dim, x_dim = features_da.dims[1], features_da.dims[2]
        y_coords = (
            features_da.coords[y_dim].to_numpy()
            if y_dim in features_da.coords
            else np.arange(feat_stack.shape[1])
        )
        x_coords = (
            features_da.coords[x_dim].to_numpy()
            if x_dim in features_da.coords
            else np.arange(feat_stack.shape[2])
        )
    else:
        feat_stack = features_da.to_numpy().astype(np.float32)
        if feat_stack.ndim != 3 or feat_stack.shape[0] != len(expected_features):
            raise ValueError(
                f"Expected {len(expected_features)} feature bands, got shape {feat_stack.shape}."
            )
        y_dim, x_dim = features_da.dims[1], features_da.dims[2]
        y_coords = (
            features_da.coords[y_dim].to_numpy()
            if y_dim in features_da.coords
            else np.arange(feat_stack.shape[1])
        )
        x_coords = (
            features_da.coords[x_dim].to_numpy()
            if x_dim in features_da.coords
            else np.arange(feat_stack.shape[2])
        )

    n_bands, height, width = feat_stack.shape
    flat_features = feat_stack.reshape(n_bands, -1).T  # (H*W, n_features)

    finite_mask = np.all(np.isfinite(flat_features), axis=1)
    if mask_arr is not None:
        valid_flat = mask_arr.reshape(-1) & finite_mask
    else:
        valid_flat = finite_mask

    prob_flat = np.full(height * width, np.nan, dtype=np.float32)
    class_flat = np.full(height * width, -1, dtype=np.int8)

    if np.any(valid_flat):
        probs = compute_probabilities_from_features(
            flat_features[valid_flat],
            model_version=model_version,
        )
        prob_flat[valid_flat] = probs
        class_flat[valid_flat] = (probs >= float(threshold)).astype(np.int8)

    prob_2d = prob_flat.reshape((height, width))
    class_2d = class_flat.reshape((height, width))

    prob_cog_path: str | None = None
    class_cog_path: str | None = None
    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        model_slug = model_version.replace(".", "_")
        thresh_slug = f"t{int(round(threshold * 100)):03d}"

        prob_file = out_root / f"{prefix}_rcrc_{model_slug}_probability.tif"
        class_file = out_root / f"{prefix}_rcrc_{model_slug}_classified_{thresh_slug}.tif"

        export_cog(
            prob_2d,
            prob_file,
            crs=crs,
            transform=transform,
            dtype="float32",
            nodata=-9999.0,
            band_descriptions=["RCR_Probability"],
        )
        export_cog(
            class_2d,
            class_file,
            crs=crs,
            transform=transform,
            dtype="int8",
            nodata=-1,
            band_descriptions=["RCR_Class (0=Healthy, 1=RCR)"],
        )
        prob_cog_path = str(prob_file)
        class_cog_path = str(class_file)

    ha_per_pixel = pixel_area_m2 / 10000.0
    total_canopy_pixels = int(np.sum(valid_flat))
    rcr_pixels = int(np.sum(class_flat == 1))
    healthy_pixels = int(np.sum(class_flat == 0))

    total_canopy_ha = round(total_canopy_pixels * ha_per_pixel, 4)
    rcr_affected_ha = round(rcr_pixels * ha_per_pixel, 4)
    incidence_pct = (
        round((rcr_pixels / total_canopy_pixels) * 100.0, 2)
        if total_canopy_pixels > 0
        else 0.0
    )

    valid_probs = prob_flat[valid_flat]
    if len(valid_probs) > 0:
        mean_prob = round(float(np.mean(valid_probs)), 4)
        median_prob = round(float(np.median(valid_probs)), 4)
        std_prob = round(float(np.std(valid_probs)), 4)
    else:
        mean_prob = float("nan")
        median_prob = float("nan")
        std_prob = float("nan")

    summary: dict[str, Any] = {
        "model_version": model_version,
        "threshold": float(threshold),
        "total_canopy_pixels": total_canopy_pixels,
        "rcr_pixels": rcr_pixels,
        "healthy_pixels": healthy_pixels,
        "total_canopy_ha": total_canopy_ha,
        "rcr_affected_ha": rcr_affected_ha,
        "incidence_pct": incidence_pct,
        "mean_probability": mean_prob,
        "median_probability": median_prob,
        "std_probability": std_prob,
        "probability_cog_path": prob_cog_path,
        "classified_cog_path": class_cog_path,
    }

    if return_rasters:
        prob_da = xr.DataArray(
            prob_2d,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
            name="probability",
        ).rio.write_nodata(np.nan)
        class_da = xr.DataArray(
            class_2d,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
            name="classification",
        ).rio.write_nodata(-1)
        valid_da = xr.DataArray(
            valid_flat.reshape((height, width)),
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
            name="valid_mask",
        )
        pred_ds = xr.Dataset(
            {
                "probability": prob_da,
                "classification": class_da,
                "valid_mask": valid_da,
            }
        )
        if crs is not None:
            pred_ds = pred_ds.rio.write_crs(crs)
        if transform is not None:
            pred_ds = pred_ds.rio.write_transform(transform)
        pred_ds.attrs.update({
            "model_version": model_version,
            "threshold": float(threshold),
            "pixel_area_m2": pixel_area_m2,
            "ha_per_pixel": ha_per_pixel,
        })
        summary["prediction_ds"] = pred_ds

    logger.info(
        "Inference (%s, t=%.2f): %.4f ha canopy, %.4f ha RCR (%.2f%%), mean P=%.4f.",
        model_version,
        threshold,
        total_canopy_ha,
        rcr_affected_ha,
        incidence_pct,
        mean_prob if np.isfinite(mean_prob) else -1.0,
    )
    return summary

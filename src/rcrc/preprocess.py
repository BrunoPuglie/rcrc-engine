"""Preprocessing, canopy masking, and local normalization for PlanetScope 8-band imagery.

Implements the RCRC v3.1 geospatial preprocessing workflow:
1. Optional inward boundary buffering and polygon clipping.
2. Early/Peak-NDVI 3-SD canopy mask generation to remove bare soil, waterways,
   and non-crop pixels.
3. Extraction of 6 surface reflectance bands and 7 vegetation indices.
4. Local per-field min-max normalization ``[0, 1]`` strictly across valid
   canopy pixels.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
import rioxarray  # noqa: F401 - registers .rio accessor on xarray objects
import shapely
from shapely.ops import unary_union
import xarray as xr

from rcrc.models import DEFAULT_MODEL_VERSION, get_model_spec

logger = logging.getLogger(__name__)

# 0-based band indices for PlanetScope SuperDove (3B_AnalyticMS_SR_8b_harmonized)
PLANET_8B_INDICES: dict[str, int] = {
    "b2": 1,  # Band 2: Blue (490 nm)
    "b4": 3,  # Band 4: Green (565 nm)
    "b5": 4,  # Band 5: Yellow (610 nm)
    "b6": 5,  # Band 6: Red (665 nm)
    "b7": 6,  # Band 7: Red Edge (705 nm)
    "b8": 7,  # Band 8: NIR (865 nm)
}

RAW_TO_NORM_FEATURE_MAP: dict[str, str] = {
    "NDVI": "NormNDVI",
    "GNDVI": "NormGNDVI",
    "NDRE": "NormNDRE",
    "EVI": "NormEVI",
    "CI": "NormCI",
    "CIRe": "NormCIRe",
    "CSI": "NormCSI",
    "b6": "Normb6",
    "b8": "Normb8",
    "b7": "Normb7",
    "b4": "Normb4",
    "b2": "Normb2",
    "b5": "Normb5",
}
NORM_TO_RAW_FEATURE_MAP: dict[str, str] = {v: k for k, v in RAW_TO_NORM_FEATURE_MAP.items()}

_ID_COLUMN_CANDIDATES: tuple[str, ...] = ("field_id", "field", "name", "id", "fid", "plot_id")


def _build_xy_coords(transform: Affine, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D pixel-center ``x`` and ``y`` coordinates from an affine transform."""
    x_coords = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    y_coords = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    return x_coords, y_coords


def _resolve_field_polygon(gdf: gpd.GeoDataFrame, field_id: str | None) -> gpd.GeoDataFrame:
    """Select the target polygon from a single- or multi-feature GeoDataFrame."""
    if gdf.empty:
        raise ValueError("Boundary GeoDataFrame contains no geometry records.")

    if len(gdf) == 1:
        return gdf.iloc[[0]].copy()

    if not field_id:
        raise ValueError(
            f"Boundary contains {len(gdf)} features; specify 'field_id' to select one."
        )

    matched_col = next(
        (col for col in gdf.columns if col.lower() in _ID_COLUMN_CANDIDATES),
        None,
    )
    if matched_col is None:
        raise ValueError(
            f"Could not locate an identifier column in boundary (columns: {list(gdf.columns)})."
        )

    series_upper = gdf[matched_col].astype(str).str.strip().str.upper()
    matches = gdf[series_upper == field_id.strip().upper()]
    if matches.empty:
        alt_id = field_id.replace("-", "_") if "-" in field_id else field_id.replace("_", "-")
        matches = gdf[series_upper == alt_id.strip().upper()]

    if matches.empty:
        raise ValueError(
            f"Field '{field_id}' not found in column '{matched_col}'."
        )
    return matches.copy()


def load_boundary(
    boundary_path: str | Path | gpd.GeoDataFrame,
    field_id: str | None = None,
    buffer_m: float = 0.0,
) -> gpd.GeoDataFrame:
    """Load, validate, and optionally buffer a field boundary polygon.

    Supports GeoPackage (``.gpkg``), Shapefile (``.shp``), GeoJSON, KML, or an
    in-memory ``geopandas.GeoDataFrame``. Strips Z coordinates, dissolves
    multipart geometries, and optionally applies a metric buffer (e.g.,
    ``buffer_m=-3.0`` for a 3 m inward buffer to remove field-edge mixed pixels).

    Args:
        boundary_path: Path to a vector boundary file or an existing
            ``GeoDataFrame``.
        field_id: Optional identifier used when ``boundary_path`` contains
            multiple field polygons.
        buffer_m: Buffer distance in meters. Negative values shrink the polygon
            inward to exclude edge effects.

    Returns:
        Single-row, 2D ``GeoDataFrame`` with a valid CRS.

    Raises:
        FileNotFoundError: If ``boundary_path`` does not exist.
        ValueError: If geometry is empty, missing CRS cannot be inferred, or
            an inward buffer collapses the polygon.
    """
    os.environ["SHAPE_RESTORE_SHX"] = "YES"

    if isinstance(boundary_path, gpd.GeoDataFrame):
        gdf = boundary_path.copy()
    else:
        path = Path(boundary_path)
        if not path.exists():
            raise FileNotFoundError(f"Boundary file not found: {path}")
        gdf = gpd.read_file(path)

    gdf = _resolve_field_polygon(gdf, field_id=field_id)

    if gdf.crs is None:
        minx, miny, maxx, maxy = gdf.total_bounds
        if abs(minx) <= 180.0 and abs(maxx) <= 180.0 and abs(miny) <= 90.0 and abs(maxy) <= 90.0:
            logger.warning("Boundary has no CRS defined; assuming EPSG:4326.")
            gdf = gdf.set_crs("EPSG:4326")
        else:
            raise ValueError("Boundary geometry has no CRS defined and bounds are not WGS84 lon/lat.")

    clean_geom_2d = unary_union(shapely.force_2d(gdf.geometry.values))
    if clean_geom_2d.is_empty:
        raise ValueError("Boundary geometry is empty after 2D conversion.")

    resolved_id = field_id or "field"
    clean_gdf = gpd.GeoDataFrame(
        {"field_id": [resolved_id]},
        geometry=[clean_geom_2d],
        crs=gdf.crs,
    )

    if buffer_m != 0.0:
        original_crs = clean_gdf.crs
        try:
            utm_crs = clean_gdf.estimate_utm_crs()
        except Exception:
            utm_crs = "EPSG:32615"
        work_gdf = clean_gdf.to_crs(utm_crs)

        buffered_geom = work_gdf.geometry.iloc[0].buffer(float(buffer_m))
        if buffered_geom.is_empty or buffered_geom.area <= 0:
            raise ValueError(
                f"Applying buffer_m={buffer_m} collapsed the field boundary to an empty geometry."
            )
        work_gdf = gpd.GeoDataFrame(
            {"field_id": [resolved_id]},
            geometry=[buffered_geom],
            crs=work_gdf.crs,
        )
        clean_gdf = work_gdf.to_crs(original_crs)
        logger.debug("Applied metric boundary buffer of %.2f m.", buffer_m)

    return clean_gdf


def _clip_raster_array(
    raster_path: str | Path,
    boundary_gdf: gpd.GeoDataFrame,
    nodata: float | int | None = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip a raster file to a boundary polygon using rasterio.mask."""
    path = Path(raster_path)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        bounds_proj = boundary_gdf.to_crs(src.crs)
        shapes = list(bounds_proj.geometry)
        fill_nodata = src.nodata if nodata is None else nodata
        clipped_img, clipped_transform = rasterio_mask(
            src,
            shapes,
            crop=True,
            filled=True,
            nodata=fill_nodata,
        )
        meta = src.meta.copy()
        meta.update({
            "height": clipped_img.shape[1],
            "width": clipped_img.shape[2],
            "transform": clipped_transform,
            "src_nodata": src.nodata,
        })
    return clipped_img, meta


def compute_canopy_mask(
    peak_ndvi_path: str | Path,
    boundary_gdf: gpd.GeoDataFrame,
    sd_mask: float = 3.0,
    reflectance_scale: float = 10000.0,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Compute the 3-SD baseline canopy mask from a Peak/Early-season scene.

    Supports either:
    - An 8-band PlanetScope SR GeoTIFF (computes NDVI from Band 6 Red and Band 8 NIR).
    - A single-band precomputed NDVI GeoTIFF.

    Pixels with ``NDVI >= mean(NDVI) - sd_mask * std(NDVI)`` are retained as
    valid crop canopy.

    Args:
        peak_ndvi_path: Path to the Peak NDVI or early-season 8-band PlanetScope
            reference scene.
        boundary_gdf: Clipped/buffered field boundary ``GeoDataFrame``.
        sd_mask: Number of standard deviations below the mean NDVI used as the
            masking cutoff (default ``3.0``).
        reflectance_scale: Scale divisor for PlanetScope surface reflectance
            DN values (default ``10000.0``).

    Returns:
        Tuple of ``(canopy_mask_bool, raster_meta, ndvi_threshold)``.

    Raises:
        ValueError: If no valid NDVI pixels remain within the boundary.
    """
    path = Path(peak_ndvi_path)
    if not path.exists():
        raise FileNotFoundError(f"Peak NDVI reference raster not found: {path}")

    with rasterio.open(path) as src:
        band_count = src.count
        src_dtype = np.dtype(src.dtypes[0])
        src_nodata = src.nodata

    if band_count >= 8:
        early_img, early_meta = _clip_raster_array(path, boundary_gdf, nodata=0)
        red_raw = early_img[PLANET_8B_INDICES["b6"]].astype(np.float32)
        nir_raw = early_img[PLANET_8B_INDICES["b8"]].astype(np.float32)

        if src_nodata is not None and not np.isnan(src_nodata):
            red_raw = np.where(red_raw == src_nodata, np.nan, red_raw)
            nir_raw = np.where(nir_raw == src_nodata, np.nan, nir_raw)

        early_red = red_raw / np.float32(reflectance_scale)
        early_nir = nir_raw / np.float32(reflectance_scale)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = early_nir + early_red
            early_ndvi = np.where(denom > 0, (early_nir - early_red) / denom, np.nan).astype(np.float32)
    elif band_count == 1:
        fill_val = np.nan if np.issubdtype(src_dtype, np.floating) else (
            src_nodata if src_nodata is not None else -9999
        )
        early_img, early_meta = _clip_raster_array(path, boundary_gdf, nodata=fill_val)
        early_ndvi = early_img[0].astype(np.float32)
        if not np.isnan(fill_val):
            early_ndvi = np.where(early_ndvi == fill_val, np.nan, early_ndvi)
        if src_nodata is not None and not np.isnan(src_nodata):
            early_ndvi = np.where(early_ndvi == src_nodata, np.nan, early_ndvi)
        early_ndvi = np.where((early_ndvi >= -1.5) & (early_ndvi <= 1.5), early_ndvi, np.nan)
    else:
        raise ValueError(
            f"Expected either an 8-band PlanetScope raster or a 1-band NDVI raster "
            f"for peak_ndvi_path, got {band_count} bands in '{path.name}'."
        )

    valid_early = early_ndvi[np.isfinite(early_ndvi)]
    if len(valid_early) == 0:
        raise ValueError(f"No valid NDVI pixels found in peak reference scene '{path.name}'.")

    ndvi_mean = float(np.mean(valid_early))
    ndvi_std = float(np.std(valid_early))
    ndvi_threshold = ndvi_mean - (float(sd_mask) * ndvi_std)
    canopy_mask_early = (early_ndvi >= ndvi_threshold) & np.isfinite(early_ndvi)

    logger.info(
        "Peak NDVI 3-SD mask (%s): mean=%.4f, std=%.4f, cutoff=%.4f, retained=%d/%d px.",
        path.name,
        ndvi_mean,
        ndvi_std,
        ndvi_threshold,
        int(np.sum(canopy_mask_early)),
        int(len(valid_early)),
    )
    return canopy_mask_early, early_meta, ndvi_threshold


def compute_spectral_features(
    scene_arr: np.ndarray,
    reflectance_scale: float = 10000.0,
    nodata_value: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute the 13 raw spectral bands and vegetation indices for RCRC v3.1.

    Uses 8-band PlanetScope (``3B_AnalyticMS_SR_8b_harmonized``) band indexing:
    Band 2 (Blue), Band 4 (Green), Band 5 (Yellow), Band 6 (Red),
    Band 7 (Red Edge), and Band 8 (NIR). Excludes TDVI.

    Args:
        scene_arr: 3D array of shape ``(bands, height, width)`` with at least 8
            PlanetScope bands.
        reflectance_scale: Divisor to convert integer DN values to surface
            reflectance ``[0, 1]`` (default ``10000.0``). If input is already
            scaled reflectance, pass ``1.0``.
        nodata_value: Optional nodata sentinel in ``scene_arr`` to replace with
            ``np.nan`` prior to computation.

    Returns:
        Dictionary mapping raw feature names (``NDVI``, ``GNDVI``, ``NDRE``,
        ``EVI``, ``CI``, ``CIRe``, ``CSI``, ``b6``, ``b8``, ``b7``, ``b4``,
        ``b2``, ``b5``) to 2D ``float32`` arrays.

    Raises:
        ValueError: If ``scene_arr`` is not 3D or has fewer than 8 bands.
    """
    if scene_arr.ndim != 3 or scene_arr.shape[0] < 8:
        raise ValueError(
            f"Expected 8-band PlanetScope array with shape (8+, H, W), got {scene_arr.shape}."
        )

    def _extract_band(idx: int) -> np.ndarray:
        arr = scene_arr[idx].astype(np.float32)
        if nodata_value is not None and not np.isnan(nodata_value):
            arr = np.where(arr == nodata_value, np.nan, arr)
        return arr / np.float32(reflectance_scale)

    b2 = _extract_band(PLANET_8B_INDICES["b2"])  # Blue
    b4 = _extract_band(PLANET_8B_INDICES["b4"])  # Green
    b5 = _extract_band(PLANET_8B_INDICES["b5"])  # Yellow
    b6 = _extract_band(PLANET_8B_INDICES["b6"])  # Red
    b7 = _extract_band(PLANET_8B_INDICES["b7"])  # Red Edge
    b8 = _extract_band(PLANET_8B_INDICES["b8"])  # NIR

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where((b8 + b6) > 0, (b8 - b6) / (b8 + b6), np.nan)
        gndvi = np.where((b8 + b4) > 0, (b8 - b4) / (b8 + b4), np.nan)
        ndre = np.where((b8 + b7) > 0, (b8 - b7) / (b8 + b7), np.nan)

        denom_evi = b8 + 6.0 * b6 - 7.5 * b2 + 1.0
        evi = np.where(denom_evi > 0, 2.5 * (b8 - b6) / denom_evi, np.nan)

        ci = np.where(b4 > 0, (b8 / b4) - 1.0, np.nan)
        cire = np.where(b7 > 0, (b8 / b7) - 1.0, np.nan)

        ratio_term = np.where(b7 > 0, b2 / b7, 0.0)
        norm_term = np.where((b8 + b7) > 0, (b8 - b7) / (b8 + b7), 0.0)
        csi = np.where((b7 > 0) & ((b8 + b7) > 0), 2.5 * ratio_term * norm_term, np.nan)

    return {
        "NDVI": ndvi.astype(np.float32),
        "GNDVI": gndvi.astype(np.float32),
        "NDRE": ndre.astype(np.float32),
        "EVI": evi.astype(np.float32),
        "CI": ci.astype(np.float32),
        "CIRe": cire.astype(np.float32),
        "CSI": csi.astype(np.float32),
        "b6": b6.astype(np.float32),
        "b8": b8.astype(np.float32),
        "b7": b7.astype(np.float32),
        "b4": b4.astype(np.float32),
        "b2": b2.astype(np.float32),
        "b5": b5.astype(np.float32),
    }


def min_max_scale_canopy(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Apply local per-field min-max scaling ``[0, 1]`` strictly over canopy pixels.

    Args:
        arr: 2D array of raw feature values.
        valid_mask: Boolean 2D mask indicating valid canopy pixels.

    Returns:
        2D ``float32`` array scaled to ``[0, 1]`` inside ``valid_mask`` and
        ``np.nan`` outside ``valid_mask``.
    """
    valid_pixels = arr[valid_mask]
    finite_pixels = valid_pixels[np.isfinite(valid_pixels)]
    if len(finite_pixels) == 0:
        return np.full_like(arr, np.nan, dtype=np.float32)

    p_min = float(np.min(finite_pixels))
    p_max = float(np.max(finite_pixels))

    if p_max == p_min:
        scaled = np.zeros_like(arr, dtype=np.float32)
    else:
        scaled = ((arr - p_min) / (p_max - p_min)).astype(np.float32)

    scaled[~valid_mask] = np.nan
    return scaled.astype(np.float32)


def _grids_match(da: xr.DataArray, ref: xr.DataArray | xr.Dataset) -> bool:
    """Check whether two rioxarray objects share identical CRS, shape, and transform."""
    try:
        return (
            da.rio.crs == ref.rio.crs
            and da.rio.height == ref.rio.height
            and da.rio.width == ref.rio.width
            and da.rio.transform().almost_equals(ref.rio.transform())
        )
    except Exception:
        return False


def preprocess_scene(
    scene_path: str | Path,
    peak_ndvi_path: str | Path,
    boundary_path: str | Path | gpd.GeoDataFrame,
    sd_mask: float = 3.0,
    buffer_m: float = 0.0,
    field_id: str | None = None,
    reference_grid: xr.DataArray | xr.Dataset | None = None,
    reflectance_scale: float = 10000.0,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> xr.Dataset:
    """Preprocess an 8-band PlanetScope scene for RCRC inference.

    Clips both the target PlanetScope scene and the Peak NDVI reference scene
    to the field boundary (with optional inward edge buffer), computes the 3-SD
    canopy mask, derives the 13 spectral features, and performs local per-field
    min-max normalization strictly across valid canopy pixels.

    Args:
        scene_path: Path to the target 8-band PlanetScope SR GeoTIFF.
        peak_ndvi_path: Path to the Peak/Early-season reference scene (8-band SR
            or 1-band NDVI GeoTIFF) used to build the 3-SD canopy mask.
        boundary_path: Path to the field polygon vector file or a ``GeoDataFrame``.
        sd_mask: Standard deviation multiplier below mean Peak NDVI for canopy
            masking (default ``3.0``).
        buffer_m: Optional metric buffer applied to the boundary polygon before
            clipping (e.g., ``-3.0`` to erode field edges by 3 m).
        field_id: Optional field identifier when ``boundary_path`` contains
            multiple polygons.
        reference_grid: Optional georeferenced ``xarray`` object to match
            spatial resolution, CRS, and extent via ``rioxarray.reproject_match``.
        reflectance_scale: Divisor for raw PlanetScope DN reflectance values
            (default ``10000.0``).
        model_version: Model key in ``MODEL_REGISTRY`` defining required feature
            names and order.

    Returns:
        Georeferenced ``xarray.Dataset`` with variables:
        - ``features``: ``(feature, y, x)`` ``float32`` normalized features.
        - ``canopy_mask``: ``(y, x)`` ``bool`` valid canopy mask.
        - ``raw_ndvi``: ``(y, x)`` ``float32`` unnormalized target NDVI.
    """
    spec = get_model_spec(model_version)
    feature_names: Sequence[str] = spec["feature_names"]

    boundary_gdf = load_boundary(boundary_path, field_id=field_id, buffer_m=buffer_m)
    canopy_mask_early, early_meta, ndvi_thresh = compute_canopy_mask(
        peak_ndvi_path=peak_ndvi_path,
        boundary_gdf=boundary_gdf,
        sd_mask=sd_mask,
        reflectance_scale=reflectance_scale,
    )

    target_img, target_meta = _clip_raster_array(scene_path, boundary_gdf, nodata=0)
    if target_img.shape[0] < 8:
        raise ValueError(
            f"Target PlanetScope scene '{Path(scene_path).name}' must have at least 8 bands, "
            f"got {target_img.shape[0]}."
        )

    target_h, target_w = target_meta["height"], target_meta["width"]
    target_transform: Affine = target_meta["transform"]
    target_crs = target_meta["crs"]
    src_nodata = target_meta.get("src_nodata")

    # Convert to float32 and set exterior/nodata pixels to NaN so optional
    # reproject_match does not bleed zero values along polygon borders.
    target_f32 = target_img.astype(np.float32)
    nir_band = target_f32[PLANET_8B_INDICES["b8"]]
    invalid_domain = ~np.isfinite(nir_band) | (nir_band <= 0)
    if src_nodata is not None and not np.isnan(src_nodata):
        invalid_domain = invalid_domain | (nir_band == src_nodata)
    target_f32[:, invalid_domain] = np.nan

    x_coords, y_coords = _build_xy_coords(target_transform, target_h, target_w)
    scene_da = xr.DataArray(
        target_f32,
        dims=("band", "y", "x"),
        coords={
            "band": np.arange(1, target_f32.shape[0] + 1),
            "y": y_coords,
            "x": x_coords,
        },
    )
    scene_da = scene_da.rio.write_crs(target_crs)
    scene_da = scene_da.rio.write_transform(target_transform)
    scene_da = scene_da.rio.write_nodata(np.nan)

    if reference_grid is not None and not _grids_match(scene_da, reference_grid):
        logger.debug("Aligning scene '%s' to reference grid via rioxarray.reproject_match.", Path(scene_path).name)
        scene_da = scene_da.rio.reproject_match(
            reference_grid,
            resampling=Resampling.bilinear,
        )
        target_h = scene_da.sizes["y"]
        target_w = scene_da.sizes["x"]
        target_transform = scene_da.rio.transform()
        target_crs = scene_da.rio.crs
        x_coords = scene_da.coords["x"].to_numpy()
        y_coords = scene_da.coords["y"].to_numpy()

    # Align early-season 3-SD mask to the final target grid
    aligned_canopy_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    reproject(
        source=canopy_mask_early.astype(np.uint8),
        destination=aligned_canopy_mask,
        src_transform=early_meta["transform"],
        src_crs=early_meta["crs"],
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )
    canopy_mask = aligned_canopy_mask.astype(bool)

    raw_features = compute_spectral_features(
        scene_da.to_numpy(),
        reflectance_scale=reflectance_scale,
        nodata_value=None,
    )
    b8 = raw_features["b8"]
    final_mask = canopy_mask & np.isfinite(b8) & (b8 > 0)

    normalized_layers: list[np.ndarray] = []
    for norm_name in feature_names:
        raw_key = NORM_TO_RAW_FEATURE_MAP.get(norm_name, norm_name)
        if raw_key not in raw_features:
            raise KeyError(
                f"Feature '{norm_name}' (raw key '{raw_key}') is not supported by preprocessing."
            )
        normalized_layers.append(min_max_scale_canopy(raw_features[raw_key], final_mask))

    norm_stack = np.stack(normalized_layers, axis=0).astype(np.float32)

    features_da = xr.DataArray(
        norm_stack,
        dims=("feature", "y", "x"),
        coords={
            "feature": list(feature_names),
            "y": y_coords,
            "x": x_coords,
        },
        name="features",
    )
    mask_da = xr.DataArray(
        final_mask.astype(bool),
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        name="canopy_mask",
    )
    raw_ndvi_arr = np.where(final_mask, raw_features["NDVI"], np.nan).astype(np.float32)
    ndvi_da = xr.DataArray(
        raw_ndvi_arr,
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        name="raw_ndvi",
    )

    ds = xr.Dataset(
        data_vars={
            "features": features_da,
            "canopy_mask": mask_da,
            "raw_ndvi": ndvi_da,
        }
    )
    ds = ds.rio.write_crs(target_crs)
    ds = ds.rio.write_transform(target_transform)
    ds["features"] = ds["features"].rio.write_nodata(np.nan)
    ds["raw_ndvi"] = ds["raw_ndvi"].rio.write_nodata(np.nan)

    pixel_area_m2 = float(abs(target_transform.a * target_transform.e))
    ds.attrs.update({
        "model_version": model_version,
        "crs_wkt": target_crs.to_wkt() if hasattr(target_crs, "to_wkt") else str(target_crs),
        "transform_affine": tuple(target_transform)[:6],
        "ndvi_mask_threshold": float(ndvi_thresh),
        "sd_mask": float(sd_mask),
        "buffer_m": float(buffer_m),
        "pixel_area_m2": pixel_area_m2,
        "ha_per_pixel": pixel_area_m2 / 10000.0,
        "scene_path": str(scene_path),
        "peak_ndvi_path": str(peak_ndvi_path),
    })

    logger.info(
        "Preprocessed scene '%s' (%dx%d px, %d valid canopy pixels, %.2f ha).",
        Path(scene_path).name,
        target_h,
        target_w,
        int(np.sum(final_mask)),
        float(np.sum(final_mask)) * (pixel_area_m2 / 10000.0),
    )
    return ds

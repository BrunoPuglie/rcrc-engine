"""Multi-date temporal disease progression and pixel transition analysis.

Aligns chronological PlanetScope scenes to a unified spatial grid and shared
seasonal canopy mask, runs RCRC inference across dates ``(t_1, ..., t_n)``,
computes pairwise pixel state transitions (``0, 1, 10, 11``), and exports tidy
longitudinal DataFrames suitable for LMM/GLMM modeling in R or Python.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr

from rcrc.models import DEFAULT_MODEL_VERSION, get_model_spec
from rcrc.predict import export_cog, predict_scene
from rcrc.preprocess import preprocess_scene

logger = logging.getLogger(__name__)

# Pixel transition codes between t_a and t_b (encoded as class_a * 10 + class_b)
TRANSITION_UNAFFECTED: int = 0       # 0 -> 0
TRANSITION_NEW_EXPANSION: int = 1    # 0 -> 1
TRANSITION_RECOVERY: int = 10        # 1 -> 0
TRANSITION_PERSISTENT: int = 11      # 1 -> 1
TRANSITION_NODATA: int = -1

TRANSITION_LABELS: dict[int, str] = {
    TRANSITION_UNAFFECTED: "Unaffected (0->0)",
    TRANSITION_NEW_EXPANSION: "New Expansion / Late Onset (0->1)",
    TRANSITION_RECOVERY: "Apparent Recovery / Canopy Regrowth (1->0)",
    TRANSITION_PERSISTENT: "Persistent Infection (1->1)",
}


def _parse_and_sort_scenes(
    scene_dict: Mapping[str, str | Path],
) -> list[tuple[str, pd.Timestamp, Path]]:
    """Validate and chronologically sort a ``{date_str: scene_path}`` mapping."""
    if not scene_dict:
        raise ValueError("scene_dict cannot be empty.")

    parsed: list[tuple[str, pd.Timestamp, Path]] = []
    for raw_key, raw_path in scene_dict.items():
        key_str = str(raw_key).strip()
        try:
            dt = pd.to_datetime(key_str)
        except Exception as exc:
            raise ValueError(
                f"Could not parse date key '{raw_key}'. Use 'YYYY-MM-DD' or 'YYYYMMDD'."
            ) from exc
        parsed.append((key_str, dt, Path(raw_path)))

    parsed.sort(key=lambda item: item[1])
    return parsed


def compute_transition_matrix(
    class_t1: np.ndarray,
    class_t2: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute pairwise pixel state transition codes between two binary maps.

    Transition codes follow ``class_t1 * 10 + class_t2`` over valid canopy
    pixels:

    - ``0``  (``0 -> 0``): Unaffected
    - ``1``  (``0 -> 1``): New Expansion / Late Onset
    - ``10`` (``1 -> 0``): Apparent Recovery / Canopy Regrowth
    - ``11`` (``1 -> 1``): Persistent Infection
    - ``-1``: Outside valid canopy mask (``nodata``)

    Args:
        class_t1: 2D binary classification array at earlier date ``t_1``.
        class_t2: 2D binary classification array at later date ``t_2``.
        valid_mask: Optional boolean mask of shared valid canopy pixels. If
            omitted, inferred as ``(class_t1 >= 0) & (class_t2 >= 0)``.

    Returns:
        2D ``int8`` transition array with values in ``{-1, 0, 1, 10, 11}``.

    Raises:
        ValueError: If array shapes do not match or valid pixels contain values
            other than ``0`` and ``1``.
    """
    arr1 = np.asarray(class_t1)
    arr2 = np.asarray(class_t2)
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Classification arrays must share identical shapes, got {arr1.shape} vs {arr2.shape}."
        )

    if valid_mask is None:
        mask = (arr1 >= 0) & (arr2 >= 0)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != arr1.shape:
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match classification shape {arr1.shape}."
            )

    if np.any(mask):
        u1 = np.unique(arr1[mask])
        u2 = np.unique(arr2[mask])
        if not set(u1.tolist()).issubset({0, 1}) or not set(u2.tolist()).issubset({0, 1}):
            raise ValueError(
                f"Binary classifications inside valid_mask must be in {{0, 1}}, "
                f"got {u1.tolist()} and {u2.tolist()}."
            )

    out = np.full(arr1.shape, TRANSITION_NODATA, dtype=np.int8)
    if np.any(mask):
        codes = (arr1[mask].astype(np.int16) * 10) + arr2[mask].astype(np.int16)
        out[mask] = codes.astype(np.int8)
    return out


def _summarize_transition(
    trans_2d: np.ndarray,
    valid_mask: np.ndarray,
    ha_per_pixel: float,
    from_date: str,
    to_date: str,
    days_elapsed: int,
    transition_type: str,
) -> dict[str, Any]:
    """Compute area and percentage breakdown for a 2D transition array."""
    total_px = int(np.sum(valid_mask))
    total_ha = round(total_px * ha_per_pixel, 4)

    counts = {
        code: int(np.sum((trans_2d == code) & valid_mask))
        for code in (
            TRANSITION_UNAFFECTED,
            TRANSITION_NEW_EXPANSION,
            TRANSITION_RECOVERY,
            TRANSITION_PERSISTENT,
        )
    }

    def _ha(px: int) -> float:
        return round(px * ha_per_pixel, 4)

    def _pct(px: int) -> float:
        return round((px / total_px) * 100.0, 2) if total_px > 0 else 0.0

    rcr_t1_ha = _ha(counts[TRANSITION_RECOVERY] + counts[TRANSITION_PERSISTENT])
    rcr_t2_ha = _ha(counts[TRANSITION_NEW_EXPANSION] + counts[TRANSITION_PERSISTENT])

    return {
        "transition_type": transition_type,
        "pair_label": f"{from_date} -> {to_date}",
        "from_date": from_date,
        "to_date": to_date,
        "days_elapsed": days_elapsed,
        "total_canopy_pixels": total_px,
        "total_canopy_ha": total_ha,
        "unaffected_pixels": counts[TRANSITION_UNAFFECTED],
        "new_expansion_pixels": counts[TRANSITION_NEW_EXPANSION],
        "recovery_pixels": counts[TRANSITION_RECOVERY],
        "persistent_pixels": counts[TRANSITION_PERSISTENT],
        "unaffected_ha": _ha(counts[TRANSITION_UNAFFECTED]),
        "new_expansion_ha": _ha(counts[TRANSITION_NEW_EXPANSION]),
        "recovery_ha": _ha(counts[TRANSITION_RECOVERY]),
        "persistent_ha": _ha(counts[TRANSITION_PERSISTENT]),
        "unaffected_pct": _pct(counts[TRANSITION_UNAFFECTED]),
        "new_expansion_pct": _pct(counts[TRANSITION_NEW_EXPANSION]),
        "recovery_pct": _pct(counts[TRANSITION_RECOVERY]),
        "persistent_pct": _pct(counts[TRANSITION_PERSISTENT]),
        "rcr_start_ha": rcr_t1_ha,
        "rcr_end_ha": rcr_t2_ha,
        "net_change_ha": round(rcr_t2_ha - rcr_t1_ha, 4),
    }


def analyze_temporal_stack(
    scene_dict: Mapping[str, str | Path],
    peak_ndvi_path: str | Path,
    boundary_path: str | Path | gpd.GeoDataFrame,
    model_version: str = DEFAULT_MODEL_VERSION,
    sd_mask: float = 3.0,
    buffer_m: float = 0.0,
    field_id: str | None = None,
    output_dir: str | Path | None = None,
    export_per_date_cogs: bool = True,
) -> dict[str, Any]:
    """Run multi-date RCR inference, alignment, and pixel transition analysis.

    Accepts a dictionary of ``{date_str: scene_path}`` captures, aligns every
    scene to a unified spatial grid (``rioxarray.reproject_match``) and shared
    seasonal 3-SD canopy mask, runs RCRC inference on each date, computes
    consecutive and start-to-end transition maps (``0, 1, 10, 11``), and
    produces tidy DataFrames ready for longitudinal LMM/GLMM analysis.

    Args:
        scene_dict: Mapping of acquisition date strings (e.g., ``"2026-07-27"``
            or ``"20260727"``) to 8-band PlanetScope GeoTIFF paths. Sorted
            chronologically automatically.
        peak_ndvi_path: Path to the Peak/Early-season reference scene used for
            the 3-SD canopy mask.
        boundary_path: Path to the field polygon vector file or a ``GeoDataFrame``.
        model_version: Registered model version key (default ``"v3.1"``).
        sd_mask: Standard deviation multiplier below mean Peak NDVI for canopy
            masking (default ``3.0``).
        buffer_m: Optional metric buffer applied to the field boundary before
            clipping (e.g., ``-3.0`` to remove edge mixed pixels).
        field_id: Optional field identifier stored in output tables and used
            when ``boundary_path`` has multiple polygons.
        output_dir: Optional directory for exporting the multi-band temporal
            transition COG, tidy progression CSVs, and per-date COGs.
        export_per_date_cogs: Whether to also write per-date probability and
            classification COGs when ``output_dir`` is provided.

    Returns:
        Dictionary containing:
        - ``"temporal_stack"``: Multi-temporal ``xarray.Dataset`` across dates.
        - ``"progression_df"``: Tidy ``pandas.DataFrame`` of metrics over time.
        - ``"transitions_df"``: Pairwise transition summary ``pandas.DataFrame``.
        - ``"exported_files"``: Dictionary of exported file paths (if any).
    """
    threshold = float(get_model_spec(model_version)["threshold"])
    ordered_scenes = _parse_and_sort_scenes(scene_dict)
    resolved_field_id = field_id or "field"

    preprocessed_list: list[xr.Dataset] = []
    reference_grid: xr.DataArray | None = None

    for idx, (date_key, dt, scene_path) in enumerate(ordered_scenes):
        logger.info(
            "Preprocessing temporal scene %d/%d (%s): %s",
            idx + 1,
            len(ordered_scenes),
            dt.strftime("%Y-%m-%d"),
            scene_path.name,
        )
        ds = preprocess_scene(
            scene_path=scene_path,
            peak_ndvi_path=peak_ndvi_path,
            boundary_path=boundary_path,
            sd_mask=sd_mask,
            buffer_m=buffer_m,
            field_id=field_id,
            reference_grid=reference_grid,
            model_version=model_version,
        )
        if reference_grid is None:
            reference_grid = ds["features"]
        preprocessed_list.append(ds)

    # Enforce an identical shared seasonal canopy mask across all dates
    shared_mask = np.ones_like(preprocessed_list[0]["canopy_mask"].to_numpy(), dtype=bool)
    for ds in preprocessed_list:
        feat_finite = np.all(np.isfinite(ds["features"].to_numpy()), axis=0)
        shared_mask &= ds["canopy_mask"].to_numpy().astype(bool) & feat_finite

    total_canopy_pixels = int(np.sum(shared_mask))
    if total_canopy_pixels == 0:
        raise ValueError("Shared seasonal canopy mask contains zero valid pixels across dates.")

    crs = preprocessed_list[0].rio.crs
    transform = preprocessed_list[0].rio.transform()
    pixel_area_m2 = float(preprocessed_list[0].attrs["pixel_area_m2"])
    ha_per_pixel = pixel_area_m2 / 10000.0
    total_canopy_ha = round(total_canopy_pixels * ha_per_pixel, 4)

    y_coords = preprocessed_list[0].coords["y"].to_numpy()
    x_coords = preprocessed_list[0].coords["x"].to_numpy()
    t0_dt = ordered_scenes[0][1]

    prob_stack: list[np.ndarray] = []
    class_stack: list[np.ndarray] = []
    ndvi_stack: list[np.ndarray] = []
    date_iso_list: list[str] = []
    progression_rows: list[dict[str, Any]] = []
    exported_files: dict[str, Any] = {"per_date_cogs": []}

    out_root = Path(output_dir) if output_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    for step_idx, ((date_key, dt, _), ds) in enumerate(zip(ordered_scenes, preprocessed_list)):
        date_iso = dt.strftime("%Y-%m-%d")
        date_slug = dt.strftime("%Y%m%d")
        date_iso_list.append(date_iso)

        # Predict using shared seasonal mask so every date shares exact footprint
        masked_ds = ds.copy()
        masked_ds["canopy_mask"] = xr.DataArray(
            shared_mask,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )

        cog_dir = (out_root / "per_date") if (out_root is not None and export_per_date_cogs) else None
        pred_res = predict_scene(
            masked_ds,
            model_version=model_version,
            output_dir=cog_dir,
            prefix=f"{resolved_field_id}_{date_slug}",
            return_rasters=True,
        )
        if cog_dir is not None:
            exported_files["per_date_cogs"].append({
                "date": date_iso,
                "probability_cog": pred_res["probability_cog_path"],
                "classified_cog": pred_res["classified_cog_path"],
            })

        pred_ds: xr.Dataset = pred_res["prediction_ds"]
        prob_2d = pred_ds["probability"].to_numpy().astype(np.float32)
        class_2d = pred_ds["classification"].to_numpy().astype(np.int8)
        raw_ndvi_2d = np.where(shared_mask, ds["raw_ndvi"].to_numpy(), np.nan).astype(np.float32)

        prob_stack.append(prob_2d)
        class_stack.append(class_2d)
        ndvi_stack.append(raw_ndvi_2d)

        valid_probs = prob_2d[shared_mask]
        q25, q75 = np.percentile(valid_probs, [25.0, 75.0])
        valid_ndvi = raw_ndvi_2d[shared_mask]
        finite_ndvi = valid_ndvi[np.isfinite(valid_ndvi)]

        rcr_px = int(pred_res["rcr_pixels"])
        healthy_px = int(pred_res["healthy_pixels"])
        rcr_ha = float(pred_res["rcr_affected_ha"])
        healthy_ha = round(healthy_px * ha_per_pixel, 4)

        progression_rows.append({
            "field_id": resolved_field_id,
            "date": date_key,
            "date_iso": date_iso,
            "days_since_start": int((dt - t0_dt).days),
            "time_step": step_idx,
            "model_version": model_version,
            "threshold": float(threshold),
            "total_canopy_pixels": total_canopy_pixels,
            "rcr_pixels": rcr_px,
            "healthy_pixels": healthy_px,
            "total_canopy_ha": total_canopy_ha,
            "rcr_affected_ha": rcr_ha,
            "healthy_ha": healthy_ha,
            "incidence_pct": float(pred_res["incidence_pct"]),
            "incidence_proportion": round(rcr_px / total_canopy_pixels, 6),
            "mean_probability": float(pred_res["mean_probability"]),
            "sd_probability": float(pred_res["std_probability"]),
            "median_probability": float(pred_res["median_probability"]),
            "iqr_probability": round(float(q75 - q25), 4),
            "mean_ndvi": round(float(np.mean(finite_ndvi)), 4) if len(finite_ndvi) > 0 else float("nan"),
        })

    # Compute pairwise transitions (consecutive + start-to-end)
    transition_arrays: list[np.ndarray] = []
    transition_pair_names: list[str] = []
    transition_rows: list[dict[str, Any]] = []

    if len(ordered_scenes) >= 2:
        for k in range(len(ordered_scenes) - 1):
            d_from = date_iso_list[k]
            d_to = date_iso_list[k + 1]
            dt_from = ordered_scenes[k][1]
            dt_to = ordered_scenes[k + 1][1]
            pair_label = f"{d_from}_to_{d_to}"

            trans_2d = compute_transition_matrix(
                class_stack[k],
                class_stack[k + 1],
                valid_mask=shared_mask,
            )
            transition_arrays.append(trans_2d)
            transition_pair_names.append(pair_label)

            summary_row = _summarize_transition(
                trans_2d=trans_2d,
                valid_mask=shared_mask,
                ha_per_pixel=ha_per_pixel,
                from_date=d_from,
                to_date=d_to,
                days_elapsed=int((dt_to - dt_from).days),
                transition_type="consecutive",
            )
            transition_rows.append(summary_row)

            # Attach transition breakdown relative to t_{k} on row k+1 of progression_df
            progression_rows[k + 1].update({
                "transition_from": d_from,
                "unaffected_ha": summary_row["unaffected_ha"],
                "new_expansion_ha": summary_row["new_expansion_ha"],
                "recovery_ha": summary_row["recovery_ha"],
                "persistent_ha": summary_row["persistent_ha"],
                "unaffected_pct": summary_row["unaffected_pct"],
                "new_expansion_pct": summary_row["new_expansion_pct"],
                "recovery_pct": summary_row["recovery_pct"],
                "persistent_pct": summary_row["persistent_pct"],
                "net_change_ha": summary_row["net_change_ha"],
            })

        # Ensure baseline t_0 row has consistent transition columns
        for col in (
            "transition_from",
            "unaffected_ha",
            "new_expansion_ha",
            "recovery_ha",
            "persistent_ha",
            "unaffected_pct",
            "new_expansion_pct",
            "recovery_pct",
            "persistent_pct",
            "net_change_ha",
        ):
            progression_rows[0].setdefault(col, None if col == "transition_from" else np.nan)

        start_to_end_2d = compute_transition_matrix(
            class_stack[0],
            class_stack[-1],
            valid_mask=shared_mask,
        )
        if len(ordered_scenes) > 2:
            transition_rows.append(
                _summarize_transition(
                    trans_2d=start_to_end_2d,
                    valid_mask=shared_mask,
                    ha_per_pixel=ha_per_pixel,
                    from_date=date_iso_list[0],
                    to_date=date_iso_list[-1],
                    days_elapsed=int((ordered_scenes[-1][1] - ordered_scenes[0][1]).days),
                    transition_type="start_to_end",
                )
            )
    else:
        start_to_end_2d = None

    time_coords = pd.to_datetime(date_iso_list)
    data_vars: dict[str, xr.DataArray] = {
        "probability": xr.DataArray(
            np.stack(prob_stack, axis=0).astype(np.float32),
            dims=("time", "y", "x"),
            coords={"time": time_coords, "y": y_coords, "x": x_coords},
        ).rio.write_nodata(np.nan),
        "classification": xr.DataArray(
            np.stack(class_stack, axis=0).astype(np.int8),
            dims=("time", "y", "x"),
            coords={"time": time_coords, "y": y_coords, "x": x_coords},
        ).rio.write_nodata(-1),
        "raw_ndvi": xr.DataArray(
            np.stack(ndvi_stack, axis=0).astype(np.float32),
            dims=("time", "y", "x"),
            coords={"time": time_coords, "y": y_coords, "x": x_coords},
        ).rio.write_nodata(np.nan),
        "canopy_mask": xr.DataArray(
            shared_mask.astype(bool),
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
        ),
    }

    if transition_arrays:
        data_vars["transitions_consecutive"] = xr.DataArray(
            np.stack(transition_arrays, axis=0).astype(np.int8),
            dims=("transition_pair", "y", "x"),
            coords={"transition_pair": transition_pair_names, "y": y_coords, "x": x_coords},
        ).rio.write_nodata(TRANSITION_NODATA)
        data_vars["transition_start_to_end"] = xr.DataArray(
            start_to_end_2d,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
        ).rio.write_nodata(TRANSITION_NODATA)

    temporal_stack = xr.Dataset(data_vars)
    temporal_stack = temporal_stack.rio.write_crs(crs)
    temporal_stack = temporal_stack.rio.write_transform(transform)
    temporal_stack.attrs.update({
        "field_id": resolved_field_id,
        "model_version": model_version,
        "threshold": float(threshold),
        "sd_mask": float(sd_mask),
        "buffer_m": float(buffer_m),
        "pixel_area_m2": pixel_area_m2,
        "ha_per_pixel": ha_per_pixel,
        "transition_codes": str(TRANSITION_LABELS),
    })

    progression_df = pd.DataFrame(progression_rows)
    transitions_df = pd.DataFrame(transition_rows)

    if out_root is not None:
        prog_csv = out_root / f"{resolved_field_id}_temporal_progression.csv"
        progression_df.to_csv(prog_csv, index=False)
        exported_files["progression_csv"] = str(prog_csv)

        if not transitions_df.empty:
            trans_csv = out_root / f"{resolved_field_id}_temporal_transitions.csv"
            transitions_df.to_csv(trans_csv, index=False)
            exported_files["transitions_csv"] = str(trans_csv)

        if transition_arrays:
            export_bands = list(transition_arrays)
            band_descs = [f"Transition_{name}" for name in transition_pair_names]
            if len(ordered_scenes) > 2 and start_to_end_2d is not None:
                export_bands.append(start_to_end_2d)
                band_descs.append(f"Transition_StartToEnd_{date_iso_list[0]}_to_{date_iso_list[-1]}")

            trans_tif = out_root / f"{resolved_field_id}_temporal_transitions.tif"
            export_cog(
                np.stack(export_bands, axis=0).astype(np.int8),
                trans_tif,
                crs=crs,
                transform=transform,
                dtype="int8",
                nodata=TRANSITION_NODATA,
                band_descriptions=band_descs,
            )
            exported_files["transitions_cog"] = str(trans_tif)

    logger.info(
        "Completed temporal stack for '%s': %d dates, %.4f ha shared canopy.",
        resolved_field_id,
        len(ordered_scenes),
        total_canopy_ha,
    )
    return {
        "temporal_stack": temporal_stack,
        "progression_df": progression_df,
        "transitions_df": transitions_df,
        "exported_files": exported_files,
    }

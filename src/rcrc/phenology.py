"""Seasonal canopy NDVI trajectory and scouting-date selection.

Port of the rcr-scout "Step 2" workflow. A field-mean NDVI time series is
turned into a continuous daily trajectory, from which the key dates of the
RCR detection workflow are derived:

- ``peak_date`` (``t_peak``): peak canopy, the date to use for the Peak-NDVI
  canopy mask (``peak_ndvi_path`` in :func:`rcrc.preprocess_scene`).
- ``target_date`` (``t_0``): first post-peak point where the trajectory is
  closest to ``target_drop_ratio * peak`` (default 80 % of peak NDVI), the
  optimal date for RCR detection.
- ``before_date`` / ``after_date``: ``t_0 -/+ window_days`` scouting window.

The NDVI series can come from:

- Sentinel-2 L2A Cloud-Optimized GeoTIFFs on the public AWS Earth Search STAC
  catalog (:func:`fetch_sentinel2_ndvi`, default, no account required);
- PlanetScope scenes already on disk (:func:`planetscope_ndvi_series`);
- any table with ``date`` and ``ndvi`` columns.
"""

from __future__ import annotations

import json
import logging
import math
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from rcrc.preprocess import PLANET_8B_INDICES, _clip_raster_array, load_boundary

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

DEFAULT_TARGET_DROP_RATIO: float = 0.80
DEFAULT_WINDOW_DAYS: int = 5

# Sentinel-2 L2A Scene Classification values masked as invalid:
# 3 cloud shadow, 8 cloud medium prob., 9 cloud high prob., 10 thin cirrus.
_S2_INVALID_SCL: tuple[int, ...] = (3, 8, 9, 10)

EARTH_SEARCH_STAC_URL: str = "https://earth-search.aws.element84.com/v1"

# GDAL options for fast anonymous reads of public COGs over HTTPS.
_COG_ENV: dict[str, Any] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "AWS_NO_SIGN_REQUEST": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}

_DATE_IN_NAME = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")


@dataclass
class NDVITrajectory:
    """Result of :func:`compute_ndvi_trajectory`.

    Attributes:
        observations: Per-date observations with ``date``, ``ndvi`` and
            ``ndvi_smooth`` (3-point rolling median) columns.
        daily: Continuous daily trajectory with ``date`` and ``ndvi_line``.
        peak_date, peak_ndvi: Peak canopy date and NDVI on the daily line.
        target_cutoff_ndvi: ``target_drop_ratio * peak_ndvi``.
        target_date, target_ndvi: Optimal detection date ``t_0`` and its NDVI.
        before_date, before_ndvi: ``t_0 - window_days``.
        after_date, after_ndvi: ``t_0 + window_days``.
        target_drop_ratio, window_days: Parameters used.
        source: Free-text description of where the observations came from.
    """

    observations: pd.DataFrame
    daily: pd.DataFrame
    peak_date: pd.Timestamp
    peak_ndvi: float
    target_cutoff_ndvi: float
    target_date: pd.Timestamp
    target_ndvi: float
    before_date: pd.Timestamp
    before_ndvi: float
    after_date: pd.Timestamp
    after_ndvi: float
    target_drop_ratio: float
    window_days: int
    source: str = "user-supplied"

    @property
    def key_dates(self) -> dict[str, pd.Timestamp]:
        """``{"peak", "before", "target", "after"}`` dates."""
        return {
            "peak": self.peak_date,
            "before": self.before_date,
            "target": self.target_date,
            "after": self.after_date,
        }

    def summary(self) -> pd.DataFrame:
        """One row per key date with its role and trajectory NDVI."""
        rows = [
            ("peak", self.peak_date, self.peak_ndvi, "Peak canopy (use for canopy mask)"),
            ("before", self.before_date, self.before_ndvi, f"t0 - {self.window_days} d"),
            ("target", self.target_date, self.target_ndvi,
             f"Optimal detection date ({self.target_drop_ratio:.0%} of peak)"),
            ("after", self.after_date, self.after_ndvi, f"t0 + {self.window_days} d"),
        ]
        df = pd.DataFrame(rows, columns=["role", "date", "ndvi_line", "description"])
        df["date_iso"] = df["date"].dt.strftime("%Y-%m-%d")
        return df[["role", "date", "date_iso", "ndvi_line", "description"]]


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def _normalize_observations(obs: pd.DataFrame | Mapping[Any, float]) -> pd.DataFrame:
    if isinstance(obs, Mapping):
        df = pd.DataFrame({"date": list(obs.keys()), "ndvi": list(obs.values())})
    else:
        if not {"date", "ndvi"}.issubset(obs.columns):
            raise ValueError("NDVI observations must have 'date' and 'ndvi' columns.")
        df = obs[["date", "ndvi"]].copy()

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")
    df = df.dropna(subset=["date", "ndvi"])
    if df.empty:
        raise ValueError("No valid NDVI observations.")
    df = df.groupby("date", as_index=False)["ndvi"].mean()
    return df.sort_values("date").reset_index(drop=True)


def compute_ndvi_trajectory(
    observations: pd.DataFrame | Mapping[Any, float],
    target_drop_ratio: float = DEFAULT_TARGET_DROP_RATIO,
    window_days: int = DEFAULT_WINDOW_DAYS,
    smooth_window: int = 3,
    source: str = "user-supplied",
) -> NDVITrajectory:
    """Build a continuous daily NDVI trajectory and derive scouting dates.

    Same algorithm as rcr-scout: same-day observations are averaged, smoothed
    with a centered rolling median (``smooth_window``), interpolated to daily
    resolution with a monotone PCHIP spline (linear when fewer than 3
    observations), and clipped to ``[0, 1]``.

    Args:
        observations: DataFrame with ``date`` and ``ndvi`` columns, or a
            ``{date: ndvi}`` mapping.
        target_drop_ratio: Fraction of peak NDVI that defines ``t_0``.
        window_days: Half-width in days of the scouting window around ``t_0``.
        smooth_window: Rolling-median window (number of observations).
            ``1`` disables smoothing.
        source: Description stored on the result.

    Returns:
        :class:`NDVITrajectory`.
    """
    if not 0.0 < target_drop_ratio < 1.0:
        raise ValueError("target_drop_ratio must be between 0 and 1.")
    if window_days < 0:
        raise ValueError("window_days must be >= 0.")

    from scipy.interpolate import PchipInterpolator

    df = _normalize_observations(observations)
    df["ndvi_smooth"] = (
        df["ndvi"].rolling(window=max(1, int(smooth_window)), center=True, min_periods=1).median()
    )

    t0 = df["date"].min()
    daily_dates = pd.date_range(t0, df["date"].max(), freq="D")
    x_obs = (df["date"] - t0).dt.days.to_numpy(dtype=float)
    y_obs = df["ndvi_smooth"].to_numpy(dtype=float)
    x_daily = (daily_dates - t0).days.to_numpy(dtype=float)

    if len(x_obs) >= 3:
        y_daily = PchipInterpolator(x_obs, y_obs)(x_daily)
    else:
        y_daily = np.interp(x_daily, x_obs, y_obs)
    y_daily = np.clip(y_daily, 0.0, 1.0)

    daily = pd.DataFrame({"date": daily_dates, "ndvi_line": y_daily})

    peak_idx = int(np.argmax(y_daily))
    peak_date = pd.Timestamp(daily_dates[peak_idx])
    peak_ndvi = float(y_daily[peak_idx])

    cutoff = float(peak_ndvi * target_drop_ratio)
    post = daily.iloc[peak_idx:]
    target_idx = int((post["ndvi_line"] - cutoff).abs().idxmin())
    target_date = pd.Timestamp(daily.loc[target_idx, "date"])
    target_ndvi = float(daily.loc[target_idx, "ndvi_line"])
    if target_ndvi > cutoff + 0.05:
        logger.warning(
            "NDVI trajectory never reaches %.0f%% of peak after %s (closest %.3f vs cutoff %.3f). "
            "The series may end too early in the season.",
            target_drop_ratio * 100, peak_date.date(), target_ndvi, cutoff,
        )

    def _ndvi_at(d: pd.Timestamp) -> float:
        return float(daily.loc[(daily["date"] - d).abs().idxmin(), "ndvi_line"])

    before_date = target_date - pd.Timedelta(days=window_days)
    after_date = target_date + pd.Timedelta(days=window_days)

    return NDVITrajectory(
        observations=df,
        daily=daily,
        peak_date=peak_date,
        peak_ndvi=peak_ndvi,
        target_cutoff_ndvi=cutoff,
        target_date=target_date,
        target_ndvi=target_ndvi,
        before_date=before_date,
        before_ndvi=_ndvi_at(before_date),
        after_date=after_date,
        after_ndvi=_ndvi_at(after_date),
        target_drop_ratio=float(target_drop_ratio),
        window_days=int(window_days),
        source=source,
    )


# ---------------------------------------------------------------------------
# NDVI sources
# ---------------------------------------------------------------------------

def _stac_search(url: str, body: dict[str, Any], timeout: float = 60.0) -> list[dict[str, Any]]:
    """POST a STAC ``/search`` request and follow ``next`` links."""
    items: list[dict[str, Any]] = []
    target: str | None = f"{url.rstrip('/')}/search"
    payload, method = dict(body), "POST"
    while target:
        data = json.dumps(payload).encode() if method == "POST" else None
        req = urllib.request.Request(
            target, data=data, method=method,
            headers={"Content-Type": "application/json", "Accept": "application/geo+json",
                     "User-Agent": "rcrc-engine"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            page = json.load(resp)
        items.extend(page.get("features", []))
        nxt = next((ln for ln in page.get("links", []) if ln.get("rel") == "next"), None)
        if nxt is None:
            break
        target = nxt["href"]
        method = nxt.get("method", "GET").upper()
        if method == "POST":
            nbody = nxt.get("body", {})
            payload = {**payload, **nbody} if nxt.get("merge") else (nbody or payload)
    return items


def _asset_scale_offset(item: dict[str, Any], key: str) -> tuple[float, float]:
    """Reflectance scale/offset for one asset.

    Earth Search sets ``earthsearch:boa_offset_applied=True`` when the
    processing-baseline >= 04.00 BOA offset (-1000 DN) has already been removed
    from the pixel values; the ``raster:bands`` offset must then not be applied
    a second time (doing so drives red reflectance negative and NDVI above 1).
    """
    bands = item["assets"][key].get("raster:bands") or [{}]
    scale = float(bands[0].get("scale", 1.0))
    offset = float(bands[0].get("offset", 0.0))
    if item["properties"].get("earthsearch:boa_offset_applied"):
        offset = 0.0
    return scale, offset


def _s2_item_field_ndvi(item: dict[str, Any], geom_wgs84: Any) -> dict[str, Any] | None:
    """Mean NDVI of the clear, in-field pixels of one Sentinel-2 L2A STAC item."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import bounds as window_bounds

    assets = item["assets"]
    with rasterio.Env(**_COG_ENV):
        with rasterio.open(assets["red"]["href"]) as red_src:
            geom = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(red_src.crs).iloc[0]
            fw = from_bounds(*geom.bounds, transform=red_src.transform)
            col0, row0 = max(0, math.floor(fw.col_off)), max(0, math.floor(fw.row_off))
            col1 = min(red_src.width, math.ceil(fw.col_off + fw.width))
            row1 = min(red_src.height, math.ceil(fw.row_off + fw.height))
            if col1 - col0 < 1 or row1 - row0 < 1:  # field outside this tile
                return None
            win = Window(col0, row0, col1 - col0, row1 - row0)
            w_transform = red_src.window_transform(win)
            w_bounds = window_bounds(win, red_src.transform)
            red_dn = red_src.read(1, window=win)
        with rasterio.open(assets["nir"]["href"]) as nir_src:
            nir_dn = nir_src.read(1, window=from_bounds(*w_bounds, transform=nir_src.transform),
                                  out_shape=red_dn.shape, resampling=Resampling.nearest)
        with rasterio.open(assets["scl"]["href"]) as scl_src:
            scl = scl_src.read(1, window=from_bounds(*w_bounds, transform=scl_src.transform),
                               out_shape=red_dn.shape, resampling=Resampling.nearest)

    in_field = geometry_mask([geom], out_shape=red_dn.shape, transform=w_transform, invert=True)
    n_field = int(in_field.sum())
    if n_field == 0:
        return None

    rs, ro = _asset_scale_offset(item, "red")
    ns, no = _asset_scale_offset(item, "nir")
    red = red_dn.astype(np.float64) * rs + ro
    nir = nir_dn.astype(np.float64) * ns + no
    denom = nir + red
    clear = (
        in_field & (red_dn > 0) & (nir_dn > 0) & (scl > 0)
        & ~np.isin(scl, _S2_INVALID_SCL) & (denom > 0)
    )
    n_clear = int(clear.sum())
    if n_clear == 0:
        return None
    ndvi = (nir[clear] - red[clear]) / denom[clear]
    return {
        "date": pd.Timestamp(item["properties"]["datetime"][:10]),
        "ndvi": float(np.mean(ndvi)),
        "clear_fraction": n_clear / n_field,
        "item_id": item["id"],
    }


def fetch_sentinel2_ndvi(
    boundary_path: str | Path | gpd.GeoDataFrame,
    year: int,
    field_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_cloud_pct: float = 70.0,
    min_clear_fraction: float = 0.5,
    stac_url: str = EARTH_SEARCH_STAC_URL,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Field-mean Sentinel-2 NDVI time series from free public data, no account needed.

    Searches the ``sentinel-2-l2a`` collection of the AWS Earth Search STAC
    catalog and reads only the field window of the red (B04), NIR (B08) and
    scene classification (SCL) Cloud-Optimized GeoTIFFs over HTTPS. Pixel values
    are harmonized across processing baselines (the 04.00 BOA offset is removed
    unless Earth Search already did so). Cloud, cloud-shadow and cirrus pixels are masked with
    the same SCL classes as rcr-scout.

    Args:
        boundary_path: Field boundary file or ``GeoDataFrame``.
        year: Season year. Default window is May 15 to Oct 15.
        field_id: Field to select when the boundary file has several polygons.
        start_date, end_date: Optional ``YYYY-MM-DD`` overrides of the window.
        max_cloud_pct: Scene-level ``eo:cloud_cover`` filter.
        min_clear_fraction: Drop dates where fewer than this fraction of the
            field's pixels are clear (partly cloudy dates give biased means).
            ``0`` keeps every date with at least one clear pixel, as rcr-scout.
        stac_url: STAC API root. Any catalog with the Earth Search
            ``sentinel-2-l2a`` asset layout (``red``, ``nir``, ``scl``) works.
        max_workers: Parallel scene reads.

    Returns:
        DataFrame with ``date``, ``ndvi``, ``clear_fraction`` and ``n_scenes``,
        one row per date.
    """
    gdf = load_boundary(boundary_path, field_id=field_id).to_crs("EPSG:4326")
    geom = gdf.geometry.iloc[0]
    start = start_date or f"{int(year)}-05-15"
    end = end_date or f"{int(year)}-10-15"

    items = _stac_search(stac_url, {
        "collections": ["sentinel-2-l2a"],
        "intersects": geom.__geo_interface__,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": float(max_cloud_pct)}},
        "limit": 100,
    })
    if not items:
        raise ValueError(f"No Sentinel-2 scenes found between {start} and {end}.")

    # The same acquisition can be published more than once (reprocessing).
    # Keep the latest processing baseline per (date, MGRS tile).
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for it in items:
        props = it["properties"]
        tile = str(props.get("s2:mgrs_tile") or props.get("grid:code") or it["id"].split("_")[1])
        key = (props["datetime"][:10], tile)
        rank = str(props.get("s2:processing_baseline", ""))
        if key not in best or rank > str(best[key]["properties"].get("s2:processing_baseline", "")):
            best[key] = it
    logger.info("Reading %d Sentinel-2 scenes from %s", len(best), stac_url)

    def _safe(it: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return _s2_item_field_ndvi(it, geom)
        except Exception as exc:  # network hiccup or missing asset: skip the scene
            logger.warning("Skipping %s: %s", it.get("id"), exc)
            return None

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        rows = [r for r in pool.map(_safe, best.values()) if r is not None]

    rows = [r for r in rows if r["clear_fraction"] >= min_clear_fraction]
    if not rows:
        raise ValueError(
            f"No Sentinel-2 dates with >= {min_clear_fraction:.0%} clear field pixels "
            f"between {start} and {end}."
        )
    return (
        pd.DataFrame(rows)
        .groupby("date", as_index=False)
        .agg(ndvi=("ndvi", "mean"), clear_fraction=("clear_fraction", "mean"),
             n_scenes=("item_id", "count"))
        .sort_values("date")
        .reset_index(drop=True)
    )


def parse_date_from_filename(path: str | Path) -> pd.Timestamp | None:
    """Extract the first ``YYYYMMDD`` date from a file name (PlanetScope style)."""
    for m in _DATE_IN_NAME.finditer(Path(path).name):
        try:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
        except ValueError:
            continue
    return None


def _as_dated_scenes(
    scenes: Mapping[Any, str | Path] | Iterable[str | Path],
) -> list[tuple[pd.Timestamp, Path]]:
    out: list[tuple[pd.Timestamp, Path]] = []
    if isinstance(scenes, Mapping):
        for key, p in scenes.items():
            out.append((pd.Timestamp(pd.to_datetime(str(key))).normalize(), Path(p)))
    else:
        for p in scenes:
            dt = parse_date_from_filename(p)
            if dt is None:
                raise ValueError(
                    f"Could not find a YYYYMMDD date in '{Path(p).name}'. "
                    "Pass a {date: path} mapping instead."
                )
            out.append((dt, Path(p)))
    out.sort(key=lambda item: item[0])
    return out


def planetscope_ndvi_series(
    scenes: Mapping[Any, str | Path] | Iterable[str | Path],
    boundary_path: str | Path | gpd.GeoDataFrame,
    field_id: str | None = None,
    buffer_m: float = 0.0,
    min_valid_fraction: float = 0.5,
) -> pd.DataFrame:
    """Field-mean NDVI from local 8-band PlanetScope scenes.

    NDVI = (NIR - Red) / (NIR + Red) using bands 8 and 6. Scenes where fewer
    than ``min_valid_fraction`` of in-field pixels are valid are skipped.

    Args:
        scenes: ``{date: path}`` mapping, or paths with ``YYYYMMDD`` in the name.
        boundary_path: Field boundary file or ``GeoDataFrame``.
        field_id: Field to select when the boundary file has several polygons.
        buffer_m: Metric buffer applied to the boundary (e.g. ``-3.0``).
        min_valid_fraction: Minimum fraction of valid in-field pixels.

    Returns:
        DataFrame with ``date``, ``ndvi``, ``valid_fraction`` and ``scene_path``.
    """
    boundary = load_boundary(boundary_path, field_id=field_id, buffer_m=buffer_m)
    red_idx, nir_idx = PLANET_8B_INDICES["b6"], PLANET_8B_INDICES["b8"]

    rows = []
    for dt, path in _as_dated_scenes(scenes):
        arr, meta = _clip_raster_array(path, boundary, nodata=0)
        if arr.shape[0] <= nir_idx:
            raise ValueError(f"{path.name} has {arr.shape[0]} bands; expected 8-band PlanetScope.")
        red = arr[red_idx].astype(np.float64)
        nir = arr[nir_idx].astype(np.float64)
        footprint = np.any(arr != 0, axis=0)
        denom = nir + red
        valid = footprint & (denom > 0)
        frac = float(valid.sum() / footprint.sum()) if footprint.any() else 0.0
        if frac < min_valid_fraction or not valid.any():
            logger.warning("Skipping %s: %.0f%% valid pixels.", path.name, frac * 100)
            continue
        ndvi = float(np.mean((nir[valid] - red[valid]) / denom[valid]))
        rows.append({"date": dt, "ndvi": ndvi, "valid_fraction": frac, "scene_path": str(path)})

    if not rows:
        raise ValueError("No usable PlanetScope scenes.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Date selection helpers
# ---------------------------------------------------------------------------

def match_scenes_to_trajectory(
    trajectory: NDVITrajectory,
    scenes: Mapping[Any, str | Path] | Iterable[str | Path],
    max_offset_days: int | None = 10,
) -> pd.DataFrame:
    """Pick the available scene closest to each key date of the trajectory.

    Use the result to fill ``peak_ndvi_path`` (role ``peak``) and the
    ``scene_dict`` of :func:`rcrc.analyze_temporal_stack` (roles ``before``,
    ``target``, ``after``).

    Args:
        trajectory: Output of :func:`compute_ndvi_trajectory`.
        scenes: ``{date: path}`` mapping, or paths with ``YYYYMMDD`` in the name.
        max_offset_days: Leave the match empty when the closest scene is
            further than this. ``None`` disables the limit.

    Returns:
        DataFrame with ``role``, ``key_date``, ``scene_date``, ``offset_days``
        and ``scene_path`` (``None`` when nothing is close enough).
    """
    dated = _as_dated_scenes(scenes)
    if not dated:
        raise ValueError("scenes cannot be empty.")
    dates = pd.DatetimeIndex([d for d, _ in dated])

    rows = []
    for role, key in trajectory.key_dates.items():
        offsets = (dates - key).days
        i = int(np.argmin(np.abs(offsets)))
        off = int(offsets[i])
        ok = max_offset_days is None or abs(off) <= max_offset_days
        rows.append({
            "role": role,
            "key_date": key,
            "scene_date": dates[i] if ok else pd.NaT,
            "offset_days": off if ok else np.nan,
            "scene_path": str(dated[i][1]) if ok else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_ndvi_trajectory(
    trajectory: NDVITrajectory,
    title: str | None = None,
    ax: "Axes | None" = None,
    matched: pd.DataFrame | None = None,
) -> "Axes":
    """Plot the NDVI trajectory with peak, ``t_0`` and the scouting window.

    Requires ``matplotlib`` (``pip install "rcrc-engine[plot]"``).

    Args:
        trajectory: Output of :func:`compute_ndvi_trajectory`.
        title: Optional axes title.
        ax: Existing axes to draw on. A new figure is created when ``None``.
        matched: Optional output of :func:`match_scenes_to_trajectory`;
            matched scene dates are drawn as tick marks on the x axis.

    Returns:
        The matplotlib ``Axes``.
    """
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            'plot_ndvi_trajectory requires matplotlib. Install with: pip install "rcrc-engine[plot]"'
        ) from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))

    tr = trajectory
    blue, orange, grey = "#13294B", "#E84A27", "#64748B"

    ax.axvspan(tr.before_date, tr.after_date, color=orange, alpha=0.10, zorder=1,
               label=f"±{tr.window_days} d window")
    ax.plot(tr.daily["date"], tr.daily["ndvi_line"], color=blue, lw=2.5, zorder=3,
            label="Daily NDVI trajectory")
    ax.scatter(tr.observations["date"], tr.observations["ndvi"], s=30, color=grey,
               edgecolor="white", zorder=4, label="Observations")
    ax.axhline(tr.target_cutoff_ndvi, color=grey, ls="--", lw=1, alpha=0.7, zorder=2,
               label=f"{tr.target_drop_ratio:.0%} of peak ({tr.target_cutoff_ndvi:.2f})")

    ax.scatter([tr.peak_date], [tr.peak_ndvi], s=90, color="#1D58A7", edgecolor="white",
               zorder=6, label=f"Peak {tr.peak_date:%Y-%m-%d} ({tr.peak_ndvi:.2f})")
    ax.scatter([tr.target_date], [tr.target_ndvi], s=110, color=orange, edgecolor="white",
               zorder=6, label=f"Target t0 {tr.target_date:%Y-%m-%d} ({tr.target_ndvi:.2f})")
    ax.scatter([tr.before_date, tr.after_date], [tr.before_ndvi, tr.after_ndvi], s=55,
               facecolor="white", edgecolor=orange, lw=2, zorder=6)

    if matched is not None and "scene_date" in matched:
        for d in matched["scene_date"].dropna():
            ax.axvline(d, ymax=0.04, color=blue, lw=2)

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("NDVI")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", color="#E5E7EB")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, ncol=2, frameon=False)
    if title:
        ax.set_title(title, loc="left")
    return ax


# ---------------------------------------------------------------------------
# One-call convenience
# ---------------------------------------------------------------------------

def analyze_ndvi_phenology(
    boundary_path: str | Path | gpd.GeoDataFrame,
    year: int | None = None,
    field_id: str | None = None,
    source: str = "sentinel2",
    scenes: Mapping[Any, str | Path] | Iterable[str | Path] | None = None,
    target_drop_ratio: float = DEFAULT_TARGET_DROP_RATIO,
    window_days: int = DEFAULT_WINDOW_DAYS,
    cache_csv: str | Path | None = None,
) -> NDVITrajectory:
    """Fetch an NDVI series and compute the scouting-date trajectory.

    Args:
        boundary_path: Field boundary file or ``GeoDataFrame``.
        year: Season year (required for ``source="sentinel2"``).
        field_id: Field to select when the boundary file has several polygons.
        source: ``"sentinel2"`` (free public Sentinel-2, no account) or
            ``"planetscope"`` (local scenes passed in ``scenes``).
        scenes: PlanetScope scenes, required for ``source="planetscope"``.
        target_drop_ratio, window_days: See :func:`compute_ndvi_trajectory`.
        cache_csv: Optional CSV path. If it exists, observations are read from
            it instead of being fetched; otherwise fetched observations are
            written to it.

    Returns:
        :class:`NDVITrajectory`.
    """
    cache = Path(cache_csv) if cache_csv else None
    if cache is not None and cache.exists():
        obs = pd.read_csv(cache, parse_dates=["date"])
        label = f"cached ({cache.name})"
    elif source == "sentinel2":
        if year is None:
            raise ValueError("year is required for source='sentinel2'.")
        obs = fetch_sentinel2_ndvi(boundary_path, year, field_id=field_id)
        label = "Sentinel-2 L2A (AWS Earth Search)"
    elif source == "planetscope":
        if scenes is None:
            raise ValueError("scenes is required for source='planetscope'.")
        obs = planetscope_ndvi_series(scenes, boundary_path, field_id=field_id)
        label = "PlanetScope (local scenes)"
    else:
        raise ValueError("source must be 'sentinel2' or 'planetscope'.")

    if cache is not None and not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        obs.to_csv(cache, index=False)

    return compute_ndvi_trajectory(
        obs, target_drop_ratio=target_drop_ratio, window_days=window_days, source=label
    )

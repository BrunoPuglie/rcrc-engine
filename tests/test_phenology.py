"""Tests for NDVI trajectory and scouting-date selection."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from rcrc.phenology import (
    compute_ndvi_trajectory,
    match_scenes_to_trajectory,
    parse_date_from_filename,
    planetscope_ndvi_series,
)


def _synthetic_season() -> pd.DataFrame:
    """Gaussian canopy curve peaking on Aug 1 (peak NDVI 0.9), sampled every 5 days."""
    dates = pd.date_range("2026-05-20", "2026-10-10", freq="5D")
    doy = (dates - pd.Timestamp("2026-08-01")).days.to_numpy(dtype=float)
    ndvi = 0.2 + 0.7 * np.exp(-((doy / 35.0) ** 2))
    return pd.DataFrame({"date": dates, "ndvi": ndvi})


def test_trajectory_key_dates() -> None:
    tr = compute_ndvi_trajectory(_synthetic_season())

    assert len(tr.daily) == (tr.daily["date"].max() - tr.daily["date"].min()).days + 1
    # 3-point rolling-median smoothing flattens the top of a 5-day series slightly
    assert abs((tr.peak_date - pd.Timestamp("2026-08-01")).days) <= 5
    assert tr.peak_ndvi == pytest.approx(0.9, abs=0.03)

    # Analytical crossing of 80 % of peak: 0.2 + 0.7 e^{-(d/35)^2} = 0.72
    d = 35.0 * np.sqrt(-np.log((0.8 * 0.9 - 0.2) / 0.7))
    expected = pd.Timestamp("2026-08-01") + pd.Timedelta(days=round(d))
    assert tr.target_date > tr.peak_date
    assert abs((tr.target_date - expected).days) <= 2
    assert tr.target_ndvi == pytest.approx(tr.target_cutoff_ndvi, abs=0.01)

    assert (tr.target_date - tr.before_date).days == 5
    assert (tr.after_date - tr.target_date).days == 5
    assert list(tr.summary()["role"]) == ["peak", "before", "target", "after"]


def test_trajectory_accepts_mapping_and_duplicate_dates() -> None:
    obs = {"2026-07-01": 0.6, "2026-07-15": 0.85, "2026-08-10": 0.7, "2026-08-30": 0.4}
    tr = compute_ndvi_trajectory(obs, target_drop_ratio=0.5, window_days=3, smooth_window=1)
    assert tr.peak_date == pd.Timestamp("2026-07-15")
    assert tr.peak_ndvi == pytest.approx(0.85)
    assert tr.target_date > tr.peak_date
    assert (tr.after_date - tr.target_date).days == 3

    dup = pd.DataFrame({"date": ["2026-07-01", "2026-07-01", "2026-07-20"], "ndvi": [0.4, 0.6, 0.8]})
    tr2 = compute_ndvi_trajectory(dup)
    assert tr2.observations["ndvi"].iloc[0] == pytest.approx(0.5)


def test_trajectory_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        compute_ndvi_trajectory(pd.DataFrame({"when": [], "ndvi": []}))
    with pytest.raises(ValueError):
        compute_ndvi_trajectory(_synthetic_season(), target_drop_ratio=1.2)


def test_parse_date_from_filename() -> None:
    assert parse_date_from_filename("BNC1_20260808_3B_AnalyticMS_SR_8b_harmonized.tif") == pd.Timestamp("2026-08-08")
    assert parse_date_from_filename("20260701_153012_44_24c4_3B.tif") == pd.Timestamp("2026-07-01")
    assert parse_date_from_filename("no_date_here.tif") is None


def test_match_scenes_to_trajectory() -> None:
    tr = compute_ndvi_trajectory(_synthetic_season())
    files = [f"F_{d:%Y%m%d}_3B.tif" for d in pd.date_range("2026-07-01", "2026-09-30", freq="3D")]
    m = match_scenes_to_trajectory(tr, files)
    assert list(m["role"]) == ["peak", "before", "target", "after"]
    assert (m["offset_days"].abs() <= 1).all()

    far = match_scenes_to_trajectory(tr, ["F_20260601_3B.tif"], max_offset_days=10)
    assert far["scene_path"].isna().all()


def _write_planet_scene(path, red: float, nir: float) -> None:
    data = np.full((8, 20, 20), 500, dtype=np.uint16)
    data[5] = red
    data[7] = nir
    with rasterio.open(
        path, "w", driver="GTiff", height=20, width=20, count=8, dtype="uint16",
        crs="EPSG:32616", transform=from_origin(400000, 4400000, 3, 3), nodata=0,
    ) as dst:
        dst.write(data)


def test_planetscope_ndvi_series(tmp_path) -> None:
    _write_planet_scene(tmp_path / "F_20260701_3B.tif", red=1000, nir=4000)  # NDVI 0.6
    _write_planet_scene(tmp_path / "F_20260801_3B.tif", red=500, nir=4500)   # NDVI 0.8
    boundary = gpd.GeoDataFrame(
        geometry=[box(400006, 4399946, 400054, 4399994)], crs="EPSG:32616"
    )
    df = planetscope_ndvi_series(sorted(tmp_path.glob("*.tif")), boundary)
    assert list(df["date"]) == [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")]
    assert df["ndvi"].tolist() == pytest.approx([0.6, 0.8])


def _stac_item(item_id: str, date: str, baseline: str, offset_applied: bool, offset: float) -> dict:
    band = [{"scale": 0.0001, "offset": offset}]
    return {
        "id": item_id,
        "properties": {
            "datetime": f"{date}T16:50:00Z",
            "s2:mgrs_tile": "16TCK",
            "s2:processing_baseline": baseline,
            "earthsearch:boa_offset_applied": offset_applied,
        },
        "assets": {k: {"href": f"{item_id}/{k}.tif", "raster:bands": band} for k in ("red", "nir")},
    }


def test_s2_offset_not_applied_twice() -> None:
    from rcrc.phenology import _asset_scale_offset

    already = _stac_item("a", "2024-07-18", "05.10", True, -0.1)
    raw = _stac_item("b", "2023-07-18", "05.09", False, -0.1)
    old = _stac_item("c", "2021-07-19", "03.01", False, 0.0)
    assert _asset_scale_offset(already, "red") == (0.0001, 0.0)
    assert _asset_scale_offset(raw, "red") == (0.0001, -0.1)
    assert _asset_scale_offset(old, "red") == (0.0001, 0.0)


def test_fetch_sentinel2_ndvi_dedup_and_clear_filter(monkeypatch) -> None:
    import rcrc.phenology as ph

    items = [
        _stac_item("S2A_16TCK_20210719_1_L2A", "2021-07-19", "03.01", False, 0.0),
        _stac_item("S2A_16TCK_20210719_3_L2A", "2021-07-19", "05.00", True, -0.1),
        _stac_item("S2B_16TCK_20210724_0_L2A", "2021-07-24", "03.01", False, 0.0),
        _stac_item("S2B_16TCK_20210729_0_L2A", "2021-07-29", "03.01", False, 0.0),
    ]
    monkeypatch.setattr(ph, "_stac_search", lambda url, body, timeout=60.0: items)
    read = []

    def fake_read(item, geom):
        read.append(item["id"])
        clear = 0.2 if item["id"].endswith("0729_0_L2A") else 1.0
        return {"date": pd.Timestamp(item["properties"]["datetime"][:10]), "ndvi": 0.8,
                "clear_fraction": clear, "item_id": item["id"]}

    monkeypatch.setattr(ph, "_s2_item_field_ndvi", fake_read)
    boundary = gpd.GeoDataFrame(geometry=[box(-88.305, 40.06, -88.299, 40.0645)], crs="EPSG:4326")
    df = ph.fetch_sentinel2_ndvi(boundary, 2021)

    assert "S2A_16TCK_20210719_1_L2A" not in read          # superseded by reprocessed item
    assert list(df["date"]) == [pd.Timestamp("2021-07-19"), pd.Timestamp("2021-07-24")]
    assert (df["n_scenes"] == 1).all()

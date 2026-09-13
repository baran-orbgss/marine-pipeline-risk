"""Tests for api.identify: feature responses are filtered to the layer's own tooltip-field
allowlist (never a raw attribute dump), and CPT channels are qc/fs/u2/qt only -- qt reported
explicitly as unavailable, never silently dropped, never used to derive CRR/CSR/FoS."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from api.identify import cpt_profile, features_geojson
from api.models import DisplaySpec, LayerDescriptor, LegendSpec, PaletteSpec, TooltipField
from shapely.geometry import LineString


def _layer_with_tooltip_fields(path: Path, fields: list[str]) -> LayerDescriptor:
    return LayerDescriptor(
        layer_id="test:layer",
        project_id="test",
        group="ANALYSIS",
        layer_type="vector",
        support_type="LINEAR_ANALYSIS",
        relative_path=str(path),
        display=DisplaySpec(
            display_name="Test",
            palette=PaletteSpec(kind="single_color", color="#000000"),
            legend=LegendSpec(title="Test", kind="single_color"),
            tooltip_fields=[TooltipField(key=f, label=f) for f in fields],
        ),
    )


def test_features_geojson_only_exposes_allowlisted_properties(tmp_path: Path) -> None:
    path = tmp_path / "route.gpkg"
    gdf = gpd.GeoDataFrame(
        {
            "chainage_m": [0.0, 25.0],
            "mobility_value": [0.1, 0.4],
            "internal_debug_column": ["secret", "secret2"],
        },
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        crs="EPSG:32631",
    )
    gdf.to_file(path, driver="GPKG")
    layer = _layer_with_tooltip_fields(path, ["chainage_m", "mobility_value"])

    result = features_geojson(layer, path)

    for feature in result["features"]:
        props = feature["properties"]
        assert set(props.keys()) == {"chainage_m", "mobility_value"}
        assert "internal_debug_column" not in props


def test_features_geojson_reprojects_to_wgs84(tmp_path: Path) -> None:
    path = tmp_path / "route.gpkg"
    gdf = gpd.GeoDataFrame(
        {"chainage_m": [0.0]},
        geometry=[LineString([(500000, 5900000), (500100, 5900100)])],
        crs="EPSG:32631",
    )
    gdf.to_file(path, driver="GPKG")
    layer = _layer_with_tooltip_fields(path, ["chainage_m"])

    result = features_geojson(layer, path)

    lon, lat = result["features"][0]["geometry"]["coordinates"][0]
    assert -10 < lon < 10
    assert 45 < lat < 65


def _write_cpt_measurements(path: Path) -> None:
    depths = [float(d) for d in range(0, 10)]
    pd.DataFrame(
        {
            "test_id": ["CPT-1"] * 10,
            "depth_bsf_m": depths,
            "qc_mpa": [1.0 + d * 0.1 for d in depths],
            "fs_kpa": [10.0 + d for d in depths],
            "u2_kpa": [5.0 + d for d in depths],
            "qt_mpa": [math.nan] * 10,
        }
    ).to_parquet(path)


def test_cpt_profile_reports_qc_fs_u2_and_qt_explicitly_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "cpt_measurements.parquet"
    _write_cpt_measurements(path)

    profile = cpt_profile(path, "CPT-1")

    assert profile is not None
    assert len(profile.depth_bsf_m) == 10
    assert profile.channels["qc"].available is True
    assert profile.channels["fs"].available is True
    assert profile.channels["u2"].available is True
    assert profile.channels["qt"].available is False
    assert profile.channels["qt"].values is None
    # No liquefaction-derived channel of any kind is representable at all -- the channel set is
    # exactly {qc, fs, u2, qt}.
    assert set(profile.channels.keys()) == {"qc", "fs", "u2", "qt"}


def test_cpt_profile_unknown_test_id_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "cpt_measurements.parquet"
    _write_cpt_measurements(path)

    assert cpt_profile(path, "CPT-does-not-exist") is None

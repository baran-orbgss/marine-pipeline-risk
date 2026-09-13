"""Tests for api.discovery: spatial metadata auto-discovery must reflect only what a file itself
declares, and must never guess CSV coordinate columns."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from api.discovery import inspect_file, inspect_parquet, reconstruct_known_tabular_geometry
from rasterio.transform import from_origin
from shapely.geometry import LineString


def _write_raster(path: Path, *, crs: str = "EPSG:32631") -> None:
    data = np.arange(100, dtype="float32").reshape(10, 10)
    transform = from_origin(500000, 5900000, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def test_raster_inspection_reports_observed_crs_and_bounds(tmp_path: Path) -> None:
    raster_path = tmp_path / "bathy.tif"
    _write_raster(raster_path)

    inspection = inspect_file(raster_path)

    assert inspection.kind == "raster"
    assert inspection.observed_crs is not None and "32631" in inspection.observed_crs
    assert inspection.band_count == 1
    assert inspection.width == 10 and inspection.height == 10
    assert inspection.bounds_native is not None
    assert inspection.bounds_wgs84 is not None
    # PL854's own working CRS: reprojected bounds must land in the North Sea, not at (0, 0).
    assert 0 < inspection.bounds_wgs84.minx < 10
    assert 50 < inspection.bounds_wgs84.miny < 60


def test_vector_inspection_reports_feature_count_and_fields(tmp_path: Path) -> None:
    gpkg_path = tmp_path / "route.gpkg"
    gdf = gpd.GeoDataFrame(
        {"chainage_m": [0.0, 25.0], "value": [1.1, 2.2]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        crs="EPSG:32631",
    )
    gdf.to_file(gpkg_path, driver="GPKG")

    inspection = inspect_file(gpkg_path)

    assert inspection.kind == "vector"
    assert inspection.feature_count == 2
    field_names = {f.name for f in inspection.fields}
    assert {"chainage_m", "value"}.issubset(field_names)


def test_csv_inspection_never_guesses_coordinate_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(
        "easting_m,northing_m,grain_size_mm\n500000,5900000,0.2\n", encoding="utf-8"
    )

    inspection = inspect_file(csv_path)

    assert inspection.kind == "tabular_unresolved"
    assert inspection.coordinate_columns_declared is False
    assert inspection.geometry_type is None
    assert any("coordinate columns not yet declared" in w for w in inspection.warnings)


def test_plain_tabular_parquet_falls_back_to_unresolved(tmp_path: Path) -> None:
    parquet_path = tmp_path / "stats.parquet"
    pd.DataFrame({"bedform_id": [1, 2], "wavelength_m": [12.0, 14.0]}).to_parquet(parquet_path)

    inspection = inspect_parquet(parquet_path)

    assert inspection.kind == "tabular_unresolved"
    assert inspection.geometry_type is None


def test_known_tabular_geometry_reconstruction_uses_verified_columns(tmp_path: Path) -> None:
    parquet_path = tmp_path / "crest_match_candidates.parquet"
    pd.DataFrame(
        {"epoch1_x_m": [500000.0, 500010.0], "epoch1_y_m": [5900000.0, 5900010.0]}
    ).to_parquet(parquet_path)

    gdf = reconstruct_known_tabular_geometry(parquet_path, "EPSG:32631")

    assert len(gdf) == 2
    assert gdf.crs.to_epsg() == 32631
    assert gdf.geometry.iloc[0].x == 500000.0


def test_known_tabular_geometry_reconstruction_raises_for_unknown_shape(tmp_path: Path) -> None:
    parquet_path = tmp_path / "unrelated.parquet"
    pd.DataFrame({"some_column": [1, 2]}).to_parquet(parquet_path)

    import pytest

    with pytest.raises(ValueError):
        reconstruct_known_tabular_geometry(parquet_path, "EPSG:32631")

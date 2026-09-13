"""Tests for api.layers: registered-asset layers are always role_established, unmatched output
files never silently become a layer, and z-order/support-type stay generic (not per-project)."""

from __future__ import annotations

import base64
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from api.layers import build_layer_catalog, catalog_extent_wgs84, decode_layer_id
from rasterio.transform import from_origin
from shapely.geometry import LineString


def _write_raster(
    path: Path, *, crs: str = "EPSG:32631", value_range: tuple[float, float] = (0.0, 50.0)
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.linspace(value_range[0], value_range[1], 100, dtype="float32").reshape(10, 10)
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


def _write_route(path: Path, *, crs: str = "EPSG:32631") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {"chainage_m": [0.0]},
        geometry=[LineString([(500000, 5900000), (500100, 5900100)])],
        crs=crs,
    )
    gdf.to_file(path, driver="GPKG")


def _write_manifest(
    path: Path, *, project_id: str, assets: list[dict], crs: str = "EPSG:32631"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "project": {"id": project_id, "name": project_id, "working_crs": crs},
                "assets": assets,
            }
        ),
        encoding="utf-8",
    )


def test_registered_bathymetry_asset_is_role_established_area_surface(api_sandbox: Path) -> None:
    output_root = api_sandbox / "data" / "processed" / "pl854"
    _write_raster(output_root / "bathymetry" / "baseline.tif")
    _write_manifest(
        api_sandbox / "configs" / "project_manifests" / "pl854.yaml",
        project_id="pl854",
        assets=[
            {
                "asset_id": "bathy",
                "category": "BATHYMETRY_RASTER",
                "evidence_role": "DERIVED",
                "path": "../../data/processed/pl854/bathymetry/baseline.tif",
                "provenance": {"source_name": "test"},
            }
        ],
    )

    catalog = build_layer_catalog("pl854")

    assert len(catalog.layers) == 1
    layer = catalog.layers[0]
    assert layer.support_type == "AREA_SURFACE"
    assert layer.group == "DATA"
    assert layer.role_established is True
    assert layer.display.legend.unit == "m"
    assert layer.tile_url_template is not None


def test_pipeline_asset_is_linear_asset_in_assets_group(api_sandbox: Path) -> None:
    output_root = api_sandbox / "data" / "processed" / "pl854"
    _write_route(output_root / "pipeline.gpkg")
    _write_manifest(
        api_sandbox / "configs" / "project_manifests" / "pl854.yaml",
        project_id="pl854",
        assets=[
            {
                "asset_id": "route",
                "category": "PIPELINE_ROUTE",
                "evidence_role": "PROJECT_GEOMETRY",
                "path": "../../data/processed/pl854/pipeline.gpkg",
                "provenance": {"source_name": "test"},
            }
        ],
    )

    catalog = build_layer_catalog("pl854")

    assert len(catalog.layers) == 1
    layer = catalog.layers[0]
    assert layer.support_type == "LINEAR_ASSET"
    assert layer.group == "ASSETS"
    assert layer.display.z_index > 0


def test_terrain_output_glob_is_role_established_and_hillshade_is_display_only(
    api_sandbox: Path,
) -> None:
    output_root = api_sandbox / "data" / "processed" / "sheringham_shoal_2020"
    _write_raster(output_root / "terrain" / "slope.tif")
    _write_raster(output_root / "terrain" / "hillshade.tif")
    _write_manifest(
        api_sandbox / "configs" / "project_manifests" / "sheringham_shoal_2020.yaml",
        project_id="sheringham_shoal_2020",
        assets=[],
    )

    catalog = build_layer_catalog("sheringham_shoal_2020")

    by_name = {layer.display.display_name: layer for layer in catalog.layers}
    assert "Slope" in by_name
    assert by_name["Slope"].role_established is True
    assert by_name["Slope"].capability_key == "terrain"
    hillshade = next(v for k, v in by_name.items() if "hillshade" in k.lower())
    assert "visualization only" in hillshade.display.legend.note.lower()
    assert hillshade.display.default_visible is False


def test_unmatched_output_file_never_becomes_a_layer(api_sandbox: Path) -> None:
    output_root = api_sandbox / "data" / "processed" / "pl854"
    (output_root / "scratch").mkdir(parents=True)
    (output_root / "scratch" / "not_a_recognised_output.tif").write_bytes(b"not a real raster")
    _write_manifest(
        api_sandbox / "configs" / "project_manifests" / "pl854.yaml", project_id="pl854", assets=[]
    )

    catalog = build_layer_catalog("pl854")

    assert catalog.layers == []


def test_extent_is_union_of_layer_bounds(api_sandbox: Path) -> None:
    output_root = api_sandbox / "data" / "processed" / "pl854"
    _write_raster(output_root / "terrain" / "slope.tif")
    _write_manifest(
        api_sandbox / "configs" / "project_manifests" / "pl854.yaml", project_id="pl854", assets=[]
    )

    catalog = build_layer_catalog("pl854")
    extent = catalog_extent_wgs84(catalog)

    assert extent is not None
    assert extent.crs == "EPSG:4326"


def test_unknown_project_returns_empty_catalog(api_sandbox: Path) -> None:
    catalog = build_layer_catalog("does-not-exist")
    assert catalog.layers == []


def test_output_layer_id_containing_a_nested_path_is_opaquely_encoded_in_its_urls(
    api_sandbox: Path,
) -> None:
    # Regression guard: an "output" layer's id embeds its relative output path verbatim (e.g.
    # "sheringham_shoal_2020:output:terrain/slope.tif"), which genuinely contains '/'. Percent-
    # encoding alone is not a safe way to carry that in a `{layer_id}` path parameter -- uvicorn
    # percent-decodes `scope["path"]` (a %2F back into a literal '/') before Starlette matches
    # routes, so the route falls through to the static-file mount and (observed directly against a
    # live server) raises a Windows OSError trying to stat a path containing ':'. The fix is to
    # encode layer_id as an opaque base64url token -- containing neither '/' nor ':' at any
    # decoding layer -- everywhere a layer_id is embedded in a URL.
    output_root = api_sandbox / "data" / "processed" / "sheringham_shoal_2020"
    _write_raster(output_root / "terrain" / "slope.tif")
    _write_manifest(
        api_sandbox / "configs" / "project_manifests" / "sheringham_shoal_2020.yaml",
        project_id="sheringham_shoal_2020",
        assets=[],
    )

    catalog = build_layer_catalog("sheringham_shoal_2020")

    layer = next(layer for layer in catalog.layers if "/" in layer.layer_id)
    assert layer.layer_id == "sheringham_shoal_2020:output:terrain/slope.tif"
    token = layer.tile_url_template.split("/layers/")[1].split("/tiles/")[0]
    assert "/" not in token
    assert ":" not in token
    assert decode_layer_id(token) == layer.layer_id
    tilejson_token = layer.tilejson_url.split("/layers/")[1].split("/tilejson.json")[0]
    assert decode_layer_id(tilejson_token) == layer.layer_id


def test_decode_layer_id_round_trips_arbitrary_characters() -> None:
    for raw in ("simple", "proj:output:terrain/aspect.tif", "proj:output:a/b/c.gpkg"):
        token = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
        assert decode_layer_id(token) == raw


def test_decode_layer_id_rejects_a_token_that_is_not_valid_utf8() -> None:
    bogus_token = base64.urlsafe_b64encode(b"\xff\xfe").decode("ascii").rstrip("=")
    with pytest.raises(ValueError):
        decode_layer_id(bogus_token)


def test_bedform_crest_matches_become_a_point_layer_from_known_coordinate_columns(
    api_sandbox: Path,
) -> None:
    import pandas as pd

    output_root = api_sandbox / "data" / "processed" / "sheringham_shoal_2020"
    bedforms_dir = output_root / "bedforms"
    bedforms_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch1_x_m": [500000.0, 500010.0],
            "epoch1_y_m": [5900000.0, 5900010.0],
            "match_status": ["MATCHED", "MATCHED"],
            "normal_displacement_m": [1.2, 0.8],
        }
    ).to_parquet(bedforms_dir / "crest_match_candidates.parquet")
    _write_manifest(
        api_sandbox / "configs" / "project_manifests" / "sheringham_shoal_2020.yaml",
        project_id="sheringham_shoal_2020",
        assets=[],
    )

    catalog = build_layer_catalog("sheringham_shoal_2020")

    bedform_layers = [layer for layer in catalog.layers if layer.capability_key == "bedforms"]
    assert len(bedform_layers) == 1
    layer = bedform_layers[0]
    assert layer.feature_count == 2
    assert layer.role_established is True
    assert {f.key for f in layer.display.tooltip_fields} >= {
        "match_status",
        "normal_displacement_m",
    }

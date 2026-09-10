"""Offline unit tests for the MAR-027 canonical project route-reference model and cross-asset
linkage layer (`project.route_reference`, `project.model`, manifest extensions, CLI wiring).

Small synthetic operator-style inputs are generated inside the tests (tiny route GeoPackages, a
small analytical GeoTIFF, small burial CSVs) -- no large binary fixtures are committed. Test names
map to MAR-027 Section 28's required proof matrix.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydantic
import pytest
import rasterio
from pyogrio import list_layers
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import LineString

from marine_engine import cli
from marine_engine.preprocessing import chainage as chainage_primitives
from marine_engine.project import categories, route_adapter, route_reference
from marine_engine.project import manifest as project_manifest
from marine_engine.project import model as project_model
from marine_engine.project import registry as project_registry
from marine_engine.terrain import readiness as terrain_readiness

WORKING_CRS = "EPSG:32631"
# 100 m straight + a 111.803... m diagonal: deliberately NOT a multiple of any round interval.
DEFAULT_ROUTE_COORDS = [(500000.0, 6000000.0), (500100.0, 6000000.0), (500200.0, 6000050.0)]
DEFAULT_ROUTE_LENGTH_M = 100.0 + math.hypot(100.0, 50.0)

# --- synthetic fixtures ------------------------------------------------------------------------


def _write_route_gpkg(
    path: Path,
    *,
    crs: str = WORKING_CRS,
    coords: list[tuple[float, float]] | None = None,
    layer: str = "route",
) -> None:
    line = LineString(coords or DEFAULT_ROUTE_COORDS)
    gpd.GeoDataFrame([{"id": 1}], geometry=[line], crs=crs).to_file(
        path, driver="GPKG", layer=layer
    )


def _write_disconnected_route_gpkg(path: Path) -> None:
    gpd.GeoDataFrame(
        [{"id": 1}, {"id": 2}],
        geometry=[
            LineString([(500000.0, 6000000.0), (500010.0, 6000000.0)]),
            LineString([(501000.0, 6001000.0), (501010.0, 6001000.0)]),
        ],
        crs=WORKING_CRS,
    ).to_file(path, driver="GPKG", layer="route")


def _write_bathymetry_tif(
    path: Path,
    *,
    crs: str = WORKING_CRS,
    origin: tuple[float, float] = (500000.0, 6000005.0),
    pixel: tuple[float, float] = (1.0, 1.0),
) -> None:
    arr = np.linspace(-20.0, -10.0, 100).reshape(10, 10).astype("float32")
    transform = from_origin(origin[0], origin[1], pixel[0], pixel[1])
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
        dst.write(arr, 1)


def _write_burial_csv(path: Path, kp_values: list[float | str]) -> None:
    n = len(kp_values)
    pd.DataFrame(
        {
            "kp_m": kp_values,
            "easting": [500000.0 + 5.0 * i for i in range(n)],
            "northing": [6000000.0] * n,
            "depth_m": [1.0 + 0.1 * i for i in range(n)],
        }
    ).to_csv(path, index=False)


def _manifest_yaml(
    assets_yaml: str,
    *,
    working_crs: str = WORKING_CRS,
    primary_route: str | None = None,
    interval_m: float | str | None = None,
    project_id: str = "test_project",
) -> str:
    head = f"project:\n  id: {project_id}\n  name: Test Project\n  working_crs: {working_crs}\n"
    if primary_route is not None:
        head += f"primary_route_asset_id: {primary_route}\n"
    if interval_m is not None:
        head += f"route_reference:\n  interval_m: {interval_m}\n"
    if not assets_yaml:
        return head + "assets: []\n"
    return head + "assets:\n" + assets_yaml


ROUTE_A_YAML = """  - asset_id: route_a
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./route_a.gpkg
    layer: route
    provenance:
      source_name: Operator route A
      horizontal_crs_declared: EPSG:32631
      survey_epoch: "2021"
"""


def _burial_yaml(
    *,
    asset_id: str = "burial_001",
    path: str = "./burial.csv",
    evidence_role: str = "MEASURED",
    relationship_to: str | None = "route_a",
    linear_reference: bool = False,
    units_declared: str = "m",
) -> str:
    rel = ""
    if relationship_to is not None:
        rel = f"""    route_relationship:
      route_asset_id: {relationship_to}
      relationship_type: ROUTE_REFERENCED
"""
        if linear_reference:
            rel += """      linear_reference:
        basis: CANONICAL_ROUTE_FROM_GEOMETRY_START
        units: m
"""
    return f"""  - asset_id: {asset_id}
    category: BURIAL_PROFILE
    evidence_role: {evidence_role}
    path: {path}
    burial_columns:
      chainage_or_kp_column: kp_m
      x_column: easting
      y_column: northing
      measured_value_column: depth_m
{rel}    provenance:
      source_name: Operator burial survey
      horizontal_crs_declared: EPSG:32631
      units_declared: {units_declared}
      survey_epoch: "2023-05"
"""


def _raster_yaml(
    *,
    asset_id: str = "bathy_001",
    path: str = "./bathy.tif",
    evidence_role: str = "MEASURED",
    relationship_to: str | None = "route_a",
    declared_crs: str | None = "EPSG:32631",
) -> str:
    rel = ""
    if relationship_to is not None:
        rel = f"""    route_relationship:
      route_asset_id: {relationship_to}
      relationship_type: ROUTE_REFERENCED
"""
    declared = f"      horizontal_crs_declared: {declared_crs}\n" if declared_crs else ""
    return f"""  - asset_id: {asset_id}
    category: BATHYMETRY_RASTER
    evidence_role: {evidence_role}
    path: {path}
{rel}    provenance:
      source_name: Operator MBES
{declared}      vertical_datum_declared: LAT
      survey_epoch: "2024-06"
"""


def _build(manifest_path: Path):
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    summary = project_registry.register_project(m, mdir)
    model = project_model.build_canonical_project_model(m, summary)
    return m, summary, model


def _write_manifest(tmp_path: Path, text: str, name: str = "manifest.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _route_project(
    tmp_path: Path, *, interval_m: float | None = 25.0, primary: str | None = "route_a"
):
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    return _write_manifest(
        tmp_path, _manifest_yaml(ROUTE_A_YAML, primary_route=primary, interval_m=interval_m)
    )


# --- Section 28.1-4: route-reference interval validation -----------------------------------------


def test_route_reference_positive_interval_accepted():
    cfg = project_manifest.RouteReferenceConfig(interval_m=25.0)
    assert cfg.interval_m == 25.0


@pytest.mark.parametrize("bad_interval", [0.0, -25.0, float("nan"), float("inf"), float("-inf")])
def test_route_reference_zero_negative_nan_infinite_interval_rejected(bad_interval: float):
    with pytest.raises(pydantic.ValidationError):
        project_manifest.RouteReferenceConfig(interval_m=bad_interval)


def test_route_reference_section_absent_means_no_interval_is_invented(tmp_path: Path):
    manifest_path = _route_project(tmp_path, interval_m=None)
    m, _summary, model = _build(manifest_path)
    assert m.route_reference is None
    rr = model.route_reference
    assert rr.status == route_reference.ROUTE_REFERENCE_NOT_BUILT
    assert rr.configured_interval_m is None
    assert rr.grid_gdf is None
    assert any("no default grid spacing is invented" in f for f in rr.findings)


def test_route_reference_unknown_key_rejected():
    with pytest.raises(pydantic.ValidationError):
        project_manifest.RouteReferenceConfig.model_validate({"interval_m": 25.0, "spacing": 1})


# --- Section 28.5-7: relationship validation ----------------------------------------------------


def _manifest_dict(assets: list[dict], **extra) -> dict:
    return {
        "project": {"id": "x", "name": "x", "working_crs": WORKING_CRS},
        "assets": assets,
        **extra,
    }


def _asset_dict(asset_id: str, category: str, **extra) -> dict:
    return {
        "asset_id": asset_id,
        "category": category,
        "evidence_role": "MEASURED",
        "path": f"{asset_id}.bin",
        "provenance": {"source_name": asset_id},
        **extra,
    }


def test_route_relationship_to_nonexistent_asset_rejected():
    with pytest.raises(pydantic.ValidationError, match="does not match any registered asset_id"):
        project_manifest.ProjectManifest.model_validate(
            _manifest_dict(
                [
                    _asset_dict(
                        "b",
                        "BURIAL_PROFILE",
                        route_relationship={
                            "route_asset_id": "ghost",
                            "relationship_type": "ROUTE_REFERENCED",
                        },
                    )
                ]
            )
        )


def test_route_relationship_to_non_route_category_rejected():
    with pytest.raises(pydantic.ValidationError, match="not 'PIPELINE_ROUTE'"):
        project_manifest.ProjectManifest.model_validate(
            _manifest_dict(
                [
                    _asset_dict("raster", "BATHYMETRY_RASTER"),
                    _asset_dict(
                        "b",
                        "BURIAL_PROFILE",
                        route_relationship={
                            "route_asset_id": "raster",
                            "relationship_type": "ROUTE_REFERENCED",
                        },
                    ),
                ]
            )
        )


def test_route_relationship_self_reference_rejected():
    with pytest.raises(pydantic.ValidationError, match="may not reference the asset itself"):
        project_manifest.ProjectManifest.model_validate(
            _manifest_dict(
                [
                    _asset_dict(
                        "route_a",
                        "PIPELINE_ROUTE",
                        route_relationship={
                            "route_asset_id": "route_a",
                            "relationship_type": "ROUTE_REFERENCED",
                        },
                    )
                ]
            )
        )


def test_unknown_relationship_type_rejected():
    with pytest.raises(pydantic.ValidationError, match="unknown relationship_type"):
        project_manifest.RouteRelationship(route_asset_id="r", relationship_type="CROSSES")


@pytest.mark.parametrize(
    "declaration",
    [
        {"basis": "SOURCE_KP_FROM_PLATFORM", "units": "m"},
        {"basis": "CANONICAL_ROUTE_FROM_GEOMETRY_START", "units": "km"},
        {"basis": "CANONICAL_ROUTE_FROM_GEOMETRY_START", "units": "m", "origin_offset_m": 0},
    ],
)
def test_unsupported_linear_reference_declarations_rejected(declaration: dict):
    with pytest.raises(pydantic.ValidationError):
        project_manifest.LinearReferenceDeclaration.model_validate(declaration)


def test_relationship_vocabulary_is_narrow_and_explicit():
    assert frozenset({categories.ROUTE_REFERENCED}) == categories.ROUTE_RELATIONSHIP_TYPES
    assert (
        frozenset({categories.CANONICAL_ROUTE_FROM_GEOMETRY_START})
        == categories.LINEAR_REFERENCE_BASES
    )
    assert frozenset({"m"}) == categories.LINEAR_REFERENCE_UNITS
    assert (
        frozenset(
            {
                project_model.LINKED,
                project_model.LINKED_WITH_LIMITATIONS,
                project_model.UNRESOLVED,
                project_model.NOT_APPLICABLE,
            }
        )
        == project_model.ROUTE_LINKAGE_STATUSES
    )


# --- Section 28.8-9: no relationship == NOT_APPLICABLE; never inferred -------------------------


def test_asset_without_relationship_is_not_applicable_and_not_defective(tmp_path: Path):
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    _write_burial_csv(tmp_path / "burial.csv", [0.0, 50.0, 100.0])
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML + _burial_yaml(relationship_to=None),
            primary_route="route_a",
            interval_m=25.0,
        ),
    )
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.declared_route_asset_id is None
    assert burial.route_linkage_status == project_model.NOT_APPLICABLE
    assert burial.linkage_facts == {}
    # a route-less asset is not marked defective on any other axis
    assert burial.registration_status == project_registry.REGISTERED
    assert burial.readiness_status_effective == burial.readiness_status_intrinsic


def test_relationship_is_never_inferred_from_filename_category_directory_or_crs(tmp_path: Path):
    # Same directory as the route, same CRS, a filename that literally names the route asset, and
    # a burial category that obviously "belongs" to a pipeline -- still NOT_APPLICABLE.
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    _write_burial_csv(tmp_path / "route_a_burial_along_route_a.csv", [0.0, 50.0, 100.0])
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML
            + _burial_yaml(path="./route_a_burial_along_route_a.csv", relationship_to=None),
            primary_route="route_a",
            interval_m=25.0,
        ),
    )
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.route_linkage_status == project_model.NOT_APPLICABLE
    assert burial.declared_route_asset_id is None
    assert "never inferred" in burial.route_linkage_findings[0]
    # and the primary route itself is not "linked" to anything either
    route = next(link for link in model.asset_linkages if link.asset_id == "route_a")
    assert route.route_linkage_status == project_model.NOT_APPLICABLE


# --- Section 28.10-14: primary route semantics -------------------------------------------------


def test_absent_primary_route_does_not_auto_select_the_only_route(tmp_path: Path):
    manifest_path = _route_project(tmp_path, primary=None)
    _m, summary, model = _build(manifest_path)
    # exactly one perfectly usable route is registered ...
    assert len(summary.asset_results) == 1
    assert summary.asset_results[0].canonical_route_gdf is not None
    # ... and it is still never promoted to primary
    pr = model.route_reference.primary_route
    assert pr.status == route_reference.PRIMARY_ROUTE_NOT_DECLARED
    assert pr.primary_route_asset_id is None
    assert model.route_reference.status == route_reference.ROUTE_REFERENCE_NOT_BUILT
    assert model.route_reference.grid_gdf is None
    assert any("none is auto-selected" in f for f in pr.findings)


def test_valid_primary_route_produces_canonical_route_reference(tmp_path: Path):
    manifest_path = _route_project(tmp_path)
    _m, _summary, model = _build(manifest_path)
    rr = model.route_reference
    assert rr.primary_route.status == route_reference.PRIMARY_ROUTE_AVAILABLE
    assert rr.primary_route.primary_route_asset_id == "route_a"
    assert rr.status == route_reference.ROUTE_REFERENCE_BUILT
    assert rr.findings == []
    assert rr.grid_gdf is not None
    assert rr.configured_interval_m == 25.0
    assert rr.route_length_m == pytest.approx(DEFAULT_ROUTE_LENGTH_M)
    assert list(rr.grid_gdf.columns) == list(route_reference.ROUTE_REFERENCE_COLUMNS)
    assert set(rr.grid_gdf["route_asset_id"]) == {"route_a"}
    assert set(rr.grid_gdf["project_id"]) == {"test_project"}


def test_unusable_primary_route_produces_controlled_finding_no_crash(tmp_path: Path):
    _write_disconnected_route_gpkg(tmp_path / "route_a.gpkg")
    manifest_path = _write_manifest(
        tmp_path, _manifest_yaml(ROUTE_A_YAML, primary_route="route_a", interval_m=25.0)
    )
    _m, summary, model = _build(manifest_path)
    reg = summary.asset_results[0].registration
    assert reg.readiness_status_effective == route_adapter.NOT_READY
    pr = model.route_reference.primary_route
    assert pr.status == route_reference.PRIMARY_ROUTE_UNUSABLE
    assert pr.primary_route_asset_id == "route_a"  # never swapped for another route
    assert pr.findings
    assert model.route_reference.status == route_reference.ROUTE_REFERENCE_NOT_BUILT
    assert model.route_reference.grid_gdf is None


def test_unusable_primary_route_is_never_replaced_by_another_usable_route(tmp_path: Path):
    _write_disconnected_route_gpkg(tmp_path / "route_a.gpkg")
    _write_route_gpkg(tmp_path / "route_b.gpkg")
    route_b_yaml = ROUTE_A_YAML.replace("route_a", "route_b")
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(ROUTE_A_YAML + route_b_yaml, primary_route="route_a", interval_m=25.0),
    )
    _m, summary, model = _build(manifest_path)
    assert summary.asset_results[1].canonical_route_gdf is not None  # route_b is usable
    pr = model.route_reference.primary_route
    assert pr.status == route_reference.PRIMARY_ROUTE_UNUSABLE
    assert pr.primary_route_asset_id == "route_a"
    assert model.route_reference.grid_gdf is None


def test_primary_route_pointing_at_non_route_category_is_controlled_finding(tmp_path: Path):
    _write_bathymetry_tif(tmp_path / "bathy.tif")
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(
            _raster_yaml(relationship_to=None), primary_route="bathy_001", interval_m=25.0
        ),
    )
    _m, _summary, model = _build(manifest_path)
    pr = model.route_reference.primary_route
    assert pr.status == route_reference.PRIMARY_ROUTE_UNUSABLE
    assert any("not 'PIPELINE_ROUTE'" in f for f in pr.findings)
    assert model.route_reference.grid_gdf is None


def test_conflicting_route_crs_produces_no_route_reference_grid(tmp_path: Path):
    _write_route_gpkg(tmp_path / "route_a.gpkg", crs="EPSG:32631")
    conflicting_yaml = ROUTE_A_YAML.replace(
        "horizontal_crs_declared: EPSG:32631", "horizontal_crs_declared: EPSG:32632"
    )
    manifest_path = _write_manifest(
        tmp_path, _manifest_yaml(conflicting_yaml, primary_route="route_a", interval_m=25.0)
    )
    _m, summary, model = _build(manifest_path)
    reg = summary.asset_results[0].registration
    assert reg.conflicts
    assert reg.readiness_status_intrinsic == route_adapter.READY
    assert reg.readiness_status_effective == route_adapter.NOT_READY
    pr = model.route_reference.primary_route
    assert pr.status == route_reference.PRIMARY_ROUTE_UNUSABLE
    assert any("declared-vs-observed conflict" in f for f in pr.findings)
    assert model.route_reference.grid_gdf is None


@pytest.mark.parametrize("bad_working_crs", ["NOT_A_REAL_CRS", "EPSG:4326"])
def test_invalid_or_geographic_project_working_crs_produces_no_route_reference_grid(
    tmp_path: Path, bad_working_crs: str
):
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML, working_crs=bad_working_crs, primary_route="route_a", interval_m=25.0
        ),
    )
    _m, summary, model = _build(manifest_path)  # controlled: never an unhandled CRSError
    assert summary.working_crs_findings
    assert model.working_crs_findings == summary.working_crs_findings
    assert model.route_reference.primary_route.status == route_reference.PRIMARY_ROUTE_UNUSABLE
    assert model.route_reference.status == route_reference.ROUTE_REFERENCE_NOT_BUILT
    assert model.route_reference.grid_gdf is None


# --- Section 28.15-23: route-reference grid invariants -------------------------------------------


def _grid(tmp_path: Path, *, interval_m: float = 25.0, coords=None) -> gpd.GeoDataFrame:
    _write_route_gpkg(tmp_path / "route_a.gpkg", coords=coords)
    manifest_path = _write_manifest(
        tmp_path, _manifest_yaml(ROUTE_A_YAML, primary_route="route_a", interval_m=interval_m)
    )
    _m, _summary, model = _build(manifest_path)
    assert model.route_reference.grid_gdf is not None
    return model.route_reference.grid_gdf


def test_chainage_begins_exactly_at_zero(tmp_path: Path):
    gdf = _grid(tmp_path)
    assert gdf["chainage_m"].iloc[0] == 0.0
    assert gdf["station_index"].iloc[0] == 0
    assert gdf["fraction_along_route"].iloc[0] == 0.0
    assert gdf.geometry.iloc[0].coords[0] == DEFAULT_ROUTE_COORDS[0]


def test_route_terminus_always_included(tmp_path: Path):
    gdf = _grid(tmp_path)
    assert gdf["chainage_m"].iloc[-1] == pytest.approx(DEFAULT_ROUTE_LENGTH_M, abs=1e-9)
    assert bool(gdf["is_terminal"].iloc[-1]) is True
    assert gdf["is_terminal"].sum() == 1
    assert gdf["fraction_along_route"].iloc[-1] == 1.0
    end = gdf.geometry.iloc[-1]
    assert (end.x, end.y) == pytest.approx(DEFAULT_ROUTE_COORDS[-1], abs=1e-6)


def test_chainage_strictly_increasing_with_no_duplicates(tmp_path: Path):
    gdf = _grid(tmp_path)
    values = gdf["chainage_m"].to_numpy()
    assert (values[1:] > values[:-1]).all()
    assert not gdf["chainage_m"].duplicated().any()
    assert not gdf["station_index"].duplicated().any()
    assert list(gdf["station_index"]) == list(range(len(gdf)))


def test_identical_input_produces_identical_output(tmp_path: Path):
    manifest_path = _route_project(tmp_path)
    _m1, _s1, model1 = _build(manifest_path)
    _m2, _s2, model2 = _build(manifest_path)
    pd.testing.assert_frame_equal(
        pd.DataFrame(model1.route_reference.grid_gdf.drop(columns="geometry")),
        pd.DataFrame(model2.route_reference.grid_gdf.drop(columns="geometry")),
    )
    assert model1.route_reference.grid_gdf.geometry.equals(model2.route_reference.grid_gdf.geometry)
    assert project_model.build_canonical_project_model_dict(
        model1
    ) == project_model.build_canonical_project_model_dict(model2)


def test_non_divisible_route_length_retains_terminal_residual_point(tmp_path: Path):
    gdf = _grid(tmp_path, interval_m=50.0)
    # 211.803... m at 50 m -> 0, 50, 100, 150, 200 regular + the exact residual terminus
    assert list(gdf["chainage_m"].iloc[:5]) == [0.0, 50.0, 100.0, 150.0, 200.0]
    assert len(gdf) == 6
    assert gdf["chainage_m"].iloc[-1] == pytest.approx(DEFAULT_ROUTE_LENGTH_M, abs=1e-9)
    assert gdf["chainage_m"].iloc[-1] - 200.0 == pytest.approx(11.80339887498948, abs=1e-9)
    assert list(gdf["is_terminal"]) == [False, False, False, False, False, True]


def test_exactly_divisible_route_length_has_no_extra_terminal_point(tmp_path: Path):
    gdf = _grid(tmp_path, interval_m=25.0, coords=[(500000.0, 6000000.0), (500100.0, 6000000.0)])
    assert list(gdf["chainage_m"]) == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert bool(gdf["is_terminal"].iloc[-1]) is True


def test_output_crs_equals_project_working_crs(tmp_path: Path):
    gdf = _grid(tmp_path)
    assert gdf.crs is not None
    assert CRS.from_user_input(gdf.crs) == CRS.from_user_input(WORKING_CRS)


def test_grid_crs_follows_working_crs_even_when_source_route_crs_differs(tmp_path: Path):
    # route delivered in ED50 / UTM 31N, project works in WGS84 / UTM 31N -> the canonical route
    # (MAR-026) is already in the working CRS, and so is the grid.
    coords = [(500000.0, 6000000.0), (500100.0, 6000000.0), (500200.0, 6000050.0)]
    _write_route_gpkg(tmp_path / "route_a.gpkg", crs="EPSG:23031", coords=coords)
    yaml_text = ROUTE_A_YAML.replace(
        "horizontal_crs_declared: EPSG:32631", "horizontal_crs_declared: EPSG:23031"
    )
    manifest_path = _write_manifest(
        tmp_path, _manifest_yaml(yaml_text, primary_route="route_a", interval_m=25.0)
    )
    _m, _summary, model = _build(manifest_path)
    gdf = model.route_reference.grid_gdf
    assert gdf is not None
    assert CRS.from_user_input(gdf.crs) == CRS.from_user_input(WORKING_CRS)


def test_source_geometry_order_remains_direction_basis(tmp_path: Path):
    forward = _grid(tmp_path, coords=DEFAULT_ROUTE_COORDS)
    reverse_dir = tmp_path / "reverse"
    reverse_dir.mkdir()
    reverse = _grid(reverse_dir, coords=list(reversed(DEFAULT_ROUTE_COORDS)))
    for gdf in (forward, reverse):
        assert set(gdf["chainage_origin_basis"]) == {route_adapter.SOURCE_GEOMETRY_ORDER}
    # chainage 0 is the source geometry's own first vertex -- flipping the source flips the grid
    assert forward.geometry.iloc[0].coords[0] == DEFAULT_ROUTE_COORDS[0]
    assert reverse.geometry.iloc[0].coords[0] == DEFAULT_ROUTE_COORDS[-1]
    assert route_reference.build_project_route_reference is not None
    source = inspect.getsource(route_reference)
    for forbidden in ("pipe_name", "from_installation", "to_installation", "landfall"):
        assert forbidden not in source  # direction never inferred from names/metadata here


def test_kp_label_is_display_only_and_does_not_mutate_numeric_chainage(tmp_path: Path):
    coords = [(500000.0, 6000000.0), (500123.456789, 6000000.0)]
    gdf = _grid(tmp_path, interval_m=50.0, coords=coords)
    exact_length = LineString(coords).length
    terminal = gdf.iloc[-1]
    assert terminal["chainage_m"] == exact_length  # unrounded, bit-identical to the geometry
    assert terminal["kp_label"] == chainage_primitives.format_kp_label(exact_length)
    assert terminal["kp_label"] == "KP 0+123.46"
    assert abs(terminal["chainage_m"] - 123.46) > 1e-6  # label rounding never fed back
    assert gdf["kp_label"].iloc[0] == "KP 0+000"
    assert gdf["kp_label"].iloc[1] == "KP 0+050"
    assert pd.api.types.is_string_dtype(gdf["kp_label"])
    assert pd.api.types.is_float_dtype(gdf["chainage_m"])


def test_grid_reuses_accepted_mar004_chainage_primitives(tmp_path: Path):
    gdf = _grid(tmp_path, interval_m=25.0)
    line = LineString(DEFAULT_ROUTE_COORDS)
    accepted = chainage_primitives.compute_chainage_stations(line, 25.0)
    assert [s.chainage_m for s in accepted.stations] == list(gdf["chainage_m"])
    assert [s.is_terminal for s in accepted.stations] == list(gdf["is_terminal"])
    source = inspect.getsource(route_reference)
    assert "compute_chainage_stations" in source
    assert "format_kp_label" in source
    assert "interpolate(" not in source  # no re-implemented linear referencing


def test_grid_vocabulary_never_calls_points_supports():
    for module in (route_reference, project_model):
        source = inspect.getsource(module).lower()
        assert "pipe_support" not in source
        assert "support_location" not in source
        assert "route_reference" in source
    assert "NOT PIPELINE SUPPORTS" in route_reference.ROUTE_REFERENCE_GRID_DISCLAIMER


# --- Section 28.24-28: evidence / readiness separation --------------------------------------------


def _full_project(tmp_path: Path, *, burial_linear_reference: bool = False):
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    _write_bathymetry_tif(tmp_path / "bathy.tif")
    _write_burial_csv(tmp_path / "burial.csv", [0.0, 50.0, 100.0, 150.0])
    return _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML
            + _raster_yaml(evidence_role="DERIVED")
            + _burial_yaml(
                evidence_role="SOURCE_INTERPRETED", linear_reference=burial_linear_reference
            ),
            primary_route="route_a",
            interval_m=25.0,
        ),
    )


def test_linkage_does_not_modify_evidence_role(tmp_path: Path):
    manifest_path = _full_project(tmp_path)
    m, summary, model = _build(manifest_path)
    declared = {a.asset_id: a.evidence_role for a in m.assets}
    registered = {
        r.registration.asset_id: r.registration.evidence_role for r in summary.asset_results
    }
    linked = {link.asset_id: link.evidence_role for link in model.asset_linkages}
    assert declared == registered == linked
    assert linked["bathy_001"] == categories.DERIVED
    assert linked["burial_001"] == categories.SOURCE_INTERPRETED
    assert linked["route_a"] == categories.PROJECT_GEOMETRY


def test_linkage_does_not_modify_intrinsic_or_effective_readiness(tmp_path: Path):
    manifest_path = _full_project(tmp_path)
    m, summary, _model_unused = _build(manifest_path)
    before = {
        r.registration.asset_id: (
            r.registration.readiness_status_intrinsic,
            r.registration.readiness_status_effective,
            json.dumps(r.registration.readiness_result, sort_keys=True, default=str),
        )
        for r in summary.asset_results
    }
    model = project_model.build_canonical_project_model(m, summary)
    after = {
        r.registration.asset_id: (
            r.registration.readiness_status_intrinsic,
            r.registration.readiness_status_effective,
            json.dumps(r.registration.readiness_result, sort_keys=True, default=str),
        )
        for r in summary.asset_results
    }
    assert before == after
    for link in model.asset_linkages:
        intrinsic, effective, _ = before[link.asset_id]
        assert link.readiness_status_intrinsic == intrinsic
        assert link.readiness_status_effective == effective


def test_linkage_preserves_mar026a_intrinsic_vs_effective_distinction(tmp_path: Path):
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    _write_bathymetry_tif(tmp_path / "bathy.tif", crs="EPSG:32631")
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML + _raster_yaml(declared_crs="EPSG:32632"),
            primary_route="route_a",
            interval_m=25.0,
        ),
    )
    _m, summary, model = _build(manifest_path)
    reg = summary.asset_results[1].registration
    assert reg.readiness_status_intrinsic == terrain_readiness.READY
    assert reg.readiness_status_effective == terrain_readiness.NOT_READY
    link = next(link for link in model.asset_linkages if link.asset_id == "bathy_001")
    assert link.readiness_status_intrinsic == terrain_readiness.READY
    assert link.readiness_status_effective == terrain_readiness.NOT_READY
    assert link.conflicts == reg.conflicts
    # linkage adds its own, separate conclusion instead of overloading either readiness field
    assert link.route_linkage_status == project_model.UNRESOLVED
    assert any("does not decide which side is correct" in f for f in link.route_linkage_findings)


def test_source_interpretation_cannot_become_measured_through_linkage(tmp_path: Path):
    manifest_path = _full_project(tmp_path, burial_linear_reference=True)
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.route_linkage_status == project_model.LINKED
    assert burial.evidence_role == categories.SOURCE_INTERPRETED
    df = project_model.build_asset_linkage_df(model)
    assert (
        df.set_index("asset_id").loc["burial_001", "evidence_role"] == categories.SOURCE_INTERPRETED
    )


def test_derived_asset_cannot_become_source_evidence_through_linkage(tmp_path: Path):
    manifest_path = _full_project(tmp_path)
    _m, _summary, model = _build(manifest_path)
    raster = next(link for link in model.asset_linkages if link.asset_id == "bathy_001")
    assert raster.route_linkage_status == project_model.LINKED
    assert raster.evidence_role == categories.DERIVED
    model_dict = project_model.build_canonical_project_model_dict(model)
    assert model_dict["assets"]["bathy_001"]["evidence_role"] == categories.DERIVED


def test_model_layer_never_imports_or_reimplements_readiness_science():
    for module in (project_model, route_reference):
        source = inspect.getsource(module)
        assert "assess_bathymetry_readiness" not in source
        assert "assess_burial_profile_readiness" not in source
        assert "assess_route_readiness" not in source
        assert "terrain.readiness" not in source
        assert "burial.readiness" not in source
        assert "readiness_result" not in source  # never read or rewritten by the linkage layer


# --- Section 28.29-34: burial linear-reference semantics -----------------------------------------


def _burial_project(
    tmp_path: Path,
    kp_values: list,
    *,
    linear_reference: bool,
    route_coords=None,
    units_declared: str = "m",
):
    _write_route_gpkg(
        tmp_path / "route_a.gpkg",
        coords=route_coords or [(500000.0, 6000000.0), (500200.0, 6000000.0)],  # exactly 200 m
    )
    _write_burial_csv(tmp_path / "burial.csv", kp_values)
    return _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML
            + _burial_yaml(linear_reference=linear_reference, units_declared=units_declared),
            primary_route="route_a",
            interval_m=25.0,
        ),
    )


def test_chainage_or_kp_column_alone_is_insufficient_for_linear_reference_equivalence(
    tmp_path: Path,
):
    # values that look exactly like canonical chainage in metres over the same range ...
    manifest_path = _burial_project(
        tmp_path, [0.0, 50.0, 100.0, 150.0, 200.0], linear_reference=False
    )
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    # ... the relationship itself is representable, but numeric correlation stays unresolved
    assert burial.route_linkage_status == project_model.LINKED_WITH_LIMITATIONS
    assert burial.declared_linear_reference is None
    assert (
        burial.linkage_facts["chainage_correlation_status"]
        == project_model.CHAINAGE_CORRELATION_UNRESOLVED_NO_LINEAR_REFERENCE
    )
    for forbidden in (
        "chainage_range_overlap_fraction",
        "chainage_range_overlap_m",
        "records_within_route_chainage_range",
        "source_chainage_min_m",
    ):
        assert forbidden not in burial.linkage_facts
    assert any("left unresolved" in f for f in burial.route_linkage_findings)


def test_explicit_linear_reference_allows_deterministic_range_correlation(tmp_path: Path):
    manifest_path = _burial_project(
        tmp_path, [0.0, 50.0, 100.0, 150.0, 250.0], linear_reference=True
    )
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    facts = burial.linkage_facts
    assert facts["linear_reference_basis"] == categories.CANONICAL_ROUTE_FROM_GEOMETRY_START
    assert facts["linear_reference_units"] == "m"
    assert (
        facts["chainage_correlation_status"]
        == project_model.CHAINAGE_CORRELATION_RESOLVED_BY_EXPLICIT_LINEAR_REFERENCE
    )
    assert facts["route_length_m"] == 200.0
    assert facts["record_count"] == 5
    assert facts["records_with_numeric_chainage"] == 5
    assert facts["source_chainage_min_m"] == 0.0
    assert facts["source_chainage_max_m"] == 250.0
    assert facts["records_within_route_chainage_range"] == 4
    assert facts["records_outside_route_chainage_range"] == 1
    assert facts["chainage_range_overlap_m"] == 200.0
    assert facts["chainage_range_overlap_fraction"] == 1.0
    # determinism: same input, same facts
    _m2, _s2, model2 = _build(manifest_path)
    burial2 = next(link for link in model2.asset_linkages if link.asset_id == "burial_001")
    assert burial2.linkage_facts == facts


def test_out_of_route_chainage_values_are_reported_not_clipped(tmp_path: Path):
    manifest_path = _burial_project(tmp_path, [-10.0, 0.0, 100.0, 250.0], linear_reference=True)
    _m, summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.route_linkage_status == project_model.LINKED_WITH_LIMITATIONS
    assert burial.linkage_facts["source_chainage_min_m"] == -10.0  # retained, not clipped to 0
    assert burial.linkage_facts["source_chainage_max_m"] == 250.0  # retained, not clipped to 200
    assert burial.linkage_facts["records_outside_route_chainage_range"] == 2
    assert burial.linkage_facts["chainage_range_overlap_m"] == 200.0
    assert any("never clipped or dropped" in f for f in burial.route_linkage_findings)
    # the loaded source table itself is untouched
    df = summary.asset_results[1].burial_table_df
    assert list(df["kp_m"]) == [-10.0, 0.0, 100.0, 250.0]


def test_duplicate_chainage_rows_remain_visible(tmp_path: Path):
    manifest_path = _burial_project(tmp_path, [0.0, 50.0, 50.0, 100.0], linear_reference=True)
    _m, summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.linkage_facts["duplicate_numeric_chainage_count"] == 1
    assert burial.linkage_facts["records_with_numeric_chainage"] == 4  # not de-duplicated
    assert burial.route_linkage_status == project_model.LINKED_WITH_LIMITATIONS
    assert any("duplicate" in f and "never collapsed" in f for f in burial.route_linkage_findings)
    # the accepted intrinsic MAR-024 duplicate-KP limitation is still there, untouched
    rr = summary.asset_results[1].registration.readiness_result
    dup_check = next(c for c in rr["checks"] if c["check_id"] == "no_duplicate_kp")
    assert dup_check["passed"] is False


def test_min_max_overlap_is_not_mislabeled_as_continuous_coverage(tmp_path: Path):
    # two samples at 0 and 200 span the whole route range but prove nothing in between
    manifest_path = _burial_project(tmp_path, [0.0, 200.0], linear_reference=True)
    _m, summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    facts = burial.linkage_facts
    assert facts["chainage_range_overlap_fraction"] == 1.0
    assert not any("coverage" in key.lower() for key in facts if not key.endswith("_note"))
    assert "NOT continuous measurement coverage" in facts["chainage_range_overlap_note"]
    # the accepted intrinsic burial readiness coverage_fraction stays unknown (never overwritten)
    rr = summary.asset_results[1].registration.readiness_result
    coverage_check = next(c for c in rr["checks"] if c["check_id"] == "coverage_fraction")
    assert coverage_check["passed"] is False
    assert "unknown" in coverage_check["detail"]


def test_unresolved_burial_reference_and_sign_semantics_remain_unresolved(tmp_path: Path):
    manifest_path = _burial_project(tmp_path, [0.0, 100.0, 200.0], linear_reference=True)
    _m, summary, model = _build(manifest_path)
    reg = summary.asset_results[1].registration
    reference_check = next(
        c
        for c in reg.readiness_result["checks"]
        if c["check_id"] == "burial_reference_convention_known"
    )
    assert reference_check["passed"] is False
    assert reg.provenance_declared["measurement_reference_declared"] is None
    assert reg.provenance_declared["sign_convention_declared"] is None
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.route_linkage_status == project_model.LINKED
    for key in burial.linkage_facts:
        assert "reference_declared" not in key
        assert "sign" not in key.lower()
        assert "burial_state" not in key.lower()
    assert burial.readiness_status_intrinsic == reg.readiness_status_intrinsic


def test_units_declared_for_measured_value_never_feed_chainage_units(tmp_path: Path):
    manifest_path = _burial_project(
        tmp_path, [0.0, 100.0, 200.0], linear_reference=True, units_declared="mm"
    )
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.linkage_facts["linear_reference_units"] == "m"
    assert burial.linkage_facts["source_chainage_max_m"] == 200.0  # no unit conversion applied


def test_non_numeric_chainage_values_are_counted_not_guessed(tmp_path: Path):
    manifest_path = _burial_project(tmp_path, [0.0, "n/a", 100.0], linear_reference=True)
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.linkage_facts["record_count"] == 3
    assert burial.linkage_facts["records_with_numeric_chainage"] == 2
    assert burial.route_linkage_status == project_model.LINKED_WITH_LIMITATIONS


def test_burial_linked_to_declared_route_not_primary_route(tmp_path: Path):
    _write_route_gpkg(tmp_path / "route_a.gpkg")  # primary, 211.8 m
    _write_route_gpkg(
        tmp_path / "route_b.gpkg", coords=[(500000.0, 6000000.0), (500200.0, 6000000.0)]
    )  # declared, 200 m
    _write_burial_csv(tmp_path / "burial.csv", [0.0, 100.0, 205.0])
    route_b_yaml = ROUTE_A_YAML.replace("route_a", "route_b")
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML
            + route_b_yaml
            + _burial_yaml(relationship_to="route_b", linear_reference=True),
            primary_route="route_a",
            interval_m=25.0,
        ),
    )
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.declared_route_asset_id == "route_b"
    assert burial.linkage_facts["route_length_m"] == 200.0
    assert burial.linkage_facts["records_outside_route_chainage_range"] == 1  # 205 > 200
    assert model.route_reference.primary_route.primary_route_asset_id == "route_a"


def test_burial_referenced_to_unusable_route_is_unresolved(tmp_path: Path):
    _write_disconnected_route_gpkg(tmp_path / "route_a.gpkg")
    _write_burial_csv(tmp_path / "burial.csv", [0.0, 100.0])
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML + _burial_yaml(linear_reference=True),
            primary_route="route_a",
            interval_m=25.0,
        ),
    )
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.route_linkage_status == project_model.UNRESOLVED
    assert any("cannot provide a canonical route" in f for f in burial.route_linkage_findings)
    assert "chainage_range_overlap_fraction" not in burial.linkage_facts


def test_failed_registration_with_relationship_is_unresolved(tmp_path: Path):
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML + _burial_yaml(path="./does_not_exist.csv", linear_reference=True),
            primary_route="route_a",
            interval_m=25.0,
        ),
    )
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.registration_status == project_registry.REGISTRATION_FAILED
    assert burial.route_linkage_status == project_model.UNRESOLVED
    assert burial.declared_route_asset_id == "route_a"  # the declaration itself is preserved


def test_future_category_with_relationship_is_linked_with_limitations_only(tmp_path: Path):
    # MAR-032 gave `CPT` a readiness adapter, so this MAR-027 invariant is exercised with
    # `BOREHOLE`, which still has none -- the intent (a future category with a declared route
    # relationship is linked with limitations only) is unchanged.
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    (tmp_path / "bh.txt").write_text("stub", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML
            + """  - asset_id: bh_001
    category: BOREHOLE
    evidence_role: MEASURED
    path: ./bh.txt
    route_relationship:
      route_asset_id: route_a
      relationship_type: ROUTE_REFERENCED
    provenance:
      source_name: Borehole campaign
""",
            primary_route="route_a",
            interval_m=25.0,
        ),
    )
    _m, _summary, model = _build(manifest_path)
    bh = next(link for link in model.asset_linkages if link.asset_id == "bh_001")
    assert bh.readiness_status_effective == categories.REGISTERED_READINESS_NOT_IMPLEMENTED
    assert bh.route_linkage_status == project_model.LINKED_WITH_LIMITATIONS
    assert any("no category-specific" in f for f in bh.route_linkage_findings)


# --- Section 19: raster extent facts are never valid-data coverage ------------------------------


def test_raster_extent_intersecting_route_reports_extent_only_facts(tmp_path: Path):
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    _write_bathymetry_tif(tmp_path / "bathy.tif")  # 500000-500010 x, 5999995-6000005 y
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(ROUTE_A_YAML + _raster_yaml(), primary_route="route_a", interval_m=25.0),
    )
    _m, _summary, model = _build(manifest_path)
    raster = next(link for link in model.asset_linkages if link.asset_id == "bathy_001")
    facts = raster.linkage_facts
    assert raster.route_linkage_status == project_model.LINKED
    assert facts["raster_extent_intersects_route"] is True
    assert facts["raster_extent_route_overlap_length_m"] == pytest.approx(10.0)
    assert facts["route_length_m"] == pytest.approx(DEFAULT_ROUTE_LENGTH_M)
    assert not any("coverage" in key.lower() for key in facts if not key.endswith("_note"))
    assert "bathymetry_coverage_fraction" not in facts
    assert "NOT valid-cell" in facts["raster_extent_note"]


def test_raster_extent_not_intersecting_route_is_a_visible_limitation(tmp_path: Path):
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    _write_bathymetry_tif(tmp_path / "bathy.tif", origin=(500000.0, 6000100.0))  # y 6000090+
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(ROUTE_A_YAML + _raster_yaml(), primary_route="route_a", interval_m=25.0),
    )
    _m, _summary, model = _build(manifest_path)
    raster = next(link for link in model.asset_linkages if link.asset_id == "bathy_001")
    assert raster.route_linkage_status == project_model.LINKED_WITH_LIMITATIONS
    assert raster.linkage_facts["raster_extent_intersects_route"] is False
    assert raster.linkage_facts["raster_extent_route_overlap_length_m"] == 0.0


def test_raster_extent_in_different_observed_crs_is_placed_via_declared_no_conflict_path(
    tmp_path: Path,
):
    # Raster embedded in EPSG:4326 (declared identically, so no conflict) covering the route's
    # location (~3.0 E, ~54.1 N) -- extent placement reprojects the OBSERVED extent only.
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    _write_bathymetry_tif(
        tmp_path / "bathy.tif", crs="EPSG:4326", origin=(2.99, 54.3), pixel=(0.002, 0.03)
    )
    manifest_path = _write_manifest(
        tmp_path,
        _manifest_yaml(
            ROUTE_A_YAML + _raster_yaml(declared_crs="EPSG:4326"),
            primary_route="route_a",
            interval_m=25.0,
        ),
    )
    _m, summary, model = _build(manifest_path)
    assert summary.asset_results[1].registration.conflicts == []
    raster = next(link for link in model.asset_linkages if link.asset_id == "bathy_001")
    assert raster.linkage_facts["raster_observed_crs"] == "EPSG:4326"
    assert raster.linkage_facts["raster_extent_intersects_route"] is True
    assert raster.linkage_facts["raster_extent_route_overlap_length_m"] == pytest.approx(
        DEFAULT_ROUTE_LENGTH_M
    )


# --- Section 20: temporal semantics --------------------------------------------------------------


def test_epochs_reported_side_by_side_without_compatibility_claim(tmp_path: Path):
    manifest_path = _full_project(tmp_path, burial_linear_reference=True)
    _m, _summary, model = _build(manifest_path)
    burial = next(link for link in model.asset_linkages if link.asset_id == "burial_001")
    assert burial.asset_survey_epoch_declared == "2023-05"
    assert burial.route_survey_epoch_declared == "2021"
    assert "not assessed" in burial.linkage_facts["temporal_note"]
    model_dict = project_model.build_canonical_project_model_dict(model)
    serialized = json.dumps(model_dict).upper()
    assert "TEMPORALLY_COMPATIBLE" not in serialized
    for module in (project_model, route_reference):
        assert "TEMPORALLY_COMPATIBLE" not in inspect.getsource(module)


# --- Section 28.35-38: outputs -------------------------------------------------------------------


def _all_keys(obj) -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.append(str(key))
            keys.extend(_all_keys(value))
    elif isinstance(obj, list):
        for value in obj:
            keys.extend(_all_keys(value))
    return keys


def test_canonical_project_json_is_deterministic_and_serializable(tmp_path: Path):
    manifest_path = _full_project(tmp_path, burial_linear_reference=True)
    _m1, _s1, model1 = _build(manifest_path)
    _m2, _s2, model2 = _build(manifest_path)
    text1 = json.dumps(project_model.build_canonical_project_model_dict(model1), indent=2)
    text2 = json.dumps(project_model.build_canonical_project_model_dict(model2), indent=2)
    assert text1 == text2
    reloaded = json.loads(text1)
    assert reloaded["scientific_role"] == project_model.SCIENTIFIC_ROLE
    assert reloaded["primary_route"]["status"] == route_reference.PRIMARY_ROUTE_AVAILABLE
    assert reloaded["route_reference"]["status"] == route_reference.ROUTE_REFERENCE_BUILT
    assert (
        reloaded["route_reference"]["chainage_origin_basis"] == route_adapter.SOURCE_GEOMETRY_ORDER
    )
    assert reloaded["route_reference"]["station_count"] == 10
    assert set(reloaded["assets"]) == {"route_a", "bathy_001", "burial_001"}
    for entry in reloaded["assets"].values():
        assert set(entry) >= {
            "category",
            "evidence_role",
            "registration_status",
            "readiness_status_intrinsic",
            "readiness_status_effective",
            "declared_route_asset_id",
            "route_linkage_status",
            "route_linkage_findings",
        }
    # model metadata only -- no source table copied in, no timestamps, no machine paths
    assert "processing_timestamp" not in text1
    assert str(tmp_path) not in text1
    assert "explicit_limitations" in reloaded


def test_linkage_parquet_contains_one_row_per_asset(tmp_path: Path):
    manifest_path = _full_project(tmp_path, burial_linear_reference=True)
    _m, _summary, model = _build(manifest_path)
    df = project_model.build_asset_linkage_df(model)
    assert list(df.columns) == list(project_model.ASSET_LINKAGE_COLUMNS)
    assert len(df) == 3
    assert list(df["asset_id"]) == ["route_a", "bathy_001", "burial_001"]
    out = tmp_path / "linkage.parquet"
    df.to_parquet(out, index=False)
    back = pd.read_parquet(out)
    assert len(back) == 3
    row = back.set_index("asset_id").loc["burial_001"]
    assert row["route_linkage_status"] == project_model.LINKED
    assert json.loads(row["route_linkage_findings"]) == []
    # burial samples span 0-150 m of the 211.8 m default route: range overlap, not coverage
    assert json.loads(row["linkage_facts_json"])["chainage_range_overlap_fraction"] == (
        pytest.approx(150.0 / DEFAULT_ROUTE_LENGTH_M)
    )
    assert json.loads(row["declared_linear_reference_json"]) == {
        "basis": categories.CANONICAL_ROUTE_FROM_GEOMETRY_START,
        "units": "m",
    }
    assert back.set_index("asset_id").loc["route_a", "declared_route_asset_id"] is None or pd.isna(
        back.set_index("asset_id").loc["route_a", "declared_route_asset_id"]
    )


def test_empty_project_yields_empty_linkage_table_with_schema(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path, _manifest_yaml(""))
    _m, _summary, model = _build(manifest_path)
    df = project_model.build_asset_linkage_df(model)
    assert list(df.columns) == list(project_model.ASSET_LINKAGE_COLUMNS)
    assert df.empty
    assert model.route_reference.primary_route.status == route_reference.PRIMARY_ROUTE_NOT_DECLARED


def test_no_numeric_readiness_or_risk_score_in_output_schema_or_modules(tmp_path: Path):
    manifest_path = _full_project(tmp_path, burial_linear_reference=True)
    _m, _summary, model = _build(manifest_path)
    keys = [k.lower() for k in _all_keys(project_model.build_canonical_project_model_dict(model))]
    keys += [c.lower() for c in project_model.build_asset_linkage_df(model).columns]
    keys += [c.lower() for c in model.route_reference.grid_gdf.columns]
    for key in keys:
        assert "score" not in key
        assert "percent" not in key
        assert "project_ready" not in key
        assert "coverage_fraction" not in key
    for module in (project_model, route_reference, project_manifest, categories):
        source = inspect.getsource(module).lower()
        assert "readiness_score" not in source
        assert "risk_score" not in source
        assert "hazard_score" not in source
        assert "suitability" not in source
    assert "PROJECT_READY" not in inspect.getsource(project_model)


def test_no_universal_project_status_only_component_findings(tmp_path: Path):
    manifest_path = _full_project(tmp_path)
    _m, _summary, model = _build(manifest_path)
    model_dict = project_model.build_canonical_project_model_dict(model)
    assert "project_status" not in model_dict
    assert "overall_status" not in model_dict
    assert {"primary_route", "route_reference", "working_crs_findings", "assets"} <= set(model_dict)
    assert (
        "IS NOT A HAZARD, RISK, OR READINESS SCORE" in model_dict["route_linkage_status_disclaimer"]
    )


# --- CLI wiring and real file outputs ------------------------------------------------------------


def test_cli_registers_build_project_model():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["build-project-model", "configs/project_manifests/pl854_route_reference.yaml"]
    )
    assert args.func is cli._cmd_build_project_model


def test_cli_build_project_model_reports_invalid_manifest_cleanly(tmp_path: Path):
    manifest_path = tmp_path / "bad_manifest.yaml"
    manifest_path.write_text(
        "project: {id: x, name: x, working_crs: EPSG:32631}\nroute_reference: {interval_m: 0}\n",
        encoding="utf-8",
    )
    assert cli.main(["build-project-model", str(manifest_path)]) == 1


def test_cli_writes_route_reference_gpkg_only_when_prerequisites_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    manifest_path = _full_project(tmp_path, burial_linear_reference=True)
    assert cli.main(["build-project-model", str(manifest_path)]) == 0

    out_dir = tmp_path / "data" / "processed" / "test_project" / "project"
    model_path = out_dir / "canonical_project_model.json"
    linkage_path = out_dir / "project_asset_linkage.parquet"
    grid_path = out_dir / "project_route_reference.gpkg"
    validation_path = out_dir / "project_model_validation.json"
    assert model_path.is_file() and linkage_path.is_file() and validation_path.is_file()
    assert grid_path.is_file()

    assert [layer[0] for layer in list_layers(grid_path)] == [route_reference.ROUTE_REFERENCE_LAYER]
    grid = gpd.read_file(grid_path, layer=route_reference.ROUTE_REFERENCE_LAYER)
    assert CRS.from_user_input(grid.crs) == CRS.from_user_input(WORKING_CRS)
    assert set(grid.geom_type) == {"Point"}
    assert len(grid) == 10
    assert grid["chainage_m"].min() == 0.0
    assert grid["chainage_m"].max() == pytest.approx(DEFAULT_ROUTE_LENGTH_M, abs=1e-6)
    assert set(grid["route_asset_id"]) == {"route_a"}
    assert bool(grid.sort_values("chainage_m")["is_terminal"].iloc[-1]) is True

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert (
        validation["question_a_does_mar027_create_a_deterministic_canonical_route_reference_grid"]
        == "YES"
    )
    assert (
        validation[
            "question_b_can_an_asset_become_route_linked_without_an_explicit_manifest_relationship"
        ]
        == "NO"
    )
    assert (
        validation["question_f_does_mar027_create_a_universal_project_hazard_or_readiness_score"]
        == "NO"
    )
    assert (
        validation["question_g_are_mar020_mar024_mar026a_scientific_readiness_semantics_preserved"]
        == "YES"
    )
    assert len(pd.read_parquet(linkage_path)) == 3

    # Now break the prerequisites (no primary route): the stale grid must NOT survive.
    broken = _write_manifest(
        tmp_path,
        _manifest_yaml(ROUTE_A_YAML + _raster_yaml() + _burial_yaml(), primary_route=None),
        name="manifest_broken.yaml",
    )
    assert cli.main(["build-project-model", str(broken)]) == 0
    assert not grid_path.exists()
    model_dict = json.loads(model_path.read_text(encoding="utf-8"))
    assert model_dict["route_reference"]["status"] == route_reference.ROUTE_REFERENCE_NOT_BUILT
    assert model_dict["route_reference"]["gis_layer"] is None
    assert model_dict["primary_route"]["status"] == route_reference.PRIMARY_ROUTE_NOT_DECLARED
    assert model_dict["route_reference"]["findings"]  # explains why no grid was produced
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert (
        validation["question_a_does_mar027_create_a_deterministic_canonical_route_reference_grid"]
        == "NO"
    )


def test_build_project_readiness_output_is_unchanged_by_mar027(tmp_path: Path, monkeypatch):
    # MAR-027 is additive: the accepted MAR-026/026A command still runs and its readiness JSON
    # carries no linkage vocabulary.
    monkeypatch.chdir(tmp_path)
    manifest_path = _full_project(tmp_path, burial_linear_reference=True)
    assert cli.main(["build-project-readiness", str(manifest_path)]) == 0
    readiness = json.loads(
        (
            tmp_path / "data" / "processed" / "test_project" / "project" / "project_readiness.json"
        ).read_text(encoding="utf-8")
    )
    assert readiness["scientific_role"] == "PROJECT_STRUCTURAL_REGISTRATION_AND_PER_ASSET_READINESS"
    assert "route_linkage_status" not in json.dumps(readiness)
    assert readiness["assets"]["route_a"]["readiness_status_intrinsic"] == route_adapter.READY

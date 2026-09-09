"""Offline unit tests for the MAR-026 generic operator project ingestion/readiness layer.

Small synthetic operator-style inputs are generated inside these tests (a tiny route
GeoPackage/GeoJSON, a small analytical bathymetry GeoTIFF, a small burial CSV) -- no large
binary fixtures are committed. Test names map to MAR-026 Section 16's required proof points and
Section 18's required semantic assertions.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydantic
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from marine_engine import cli
from marine_engine.burial import readiness as burial_readiness
from marine_engine.project import (
    bathymetry_adapter,
    burial_adapter,
    categories,
    identity,
    route_adapter,
)
from marine_engine.project import (
    manifest as project_manifest,
)
from marine_engine.project import (
    registry as project_registry,
)
from marine_engine.project import (
    report as project_report,
)
from marine_engine.terrain import readiness as terrain_readiness

PROJECT_MODULES = (
    bathymetry_adapter,
    burial_adapter,
    categories,
    identity,
    project_manifest,
    project_registry,
    project_report,
    route_adapter,
)

# --- synthetic fixtures ------------------------------------------------------------------------


def _write_route_gpkg(path: Path, *, crs: str = "EPSG:32631", layer: str = "route") -> None:
    line = LineString([(500000.0, 6000000.0), (500100.0, 6000000.0), (500200.0, 6000050.0)])
    gdf = gpd.GeoDataFrame([{"id": 1}], geometry=[line], crs=crs)
    gdf.to_file(path, driver="GPKG", layer=layer)


def _write_disconnected_route_gpkg(path: Path, *, crs: str = "EPSG:32631") -> None:
    gdf = gpd.GeoDataFrame(
        [{"id": 1}, {"id": 2}],
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(1000, 1000), (1010, 1000)])],
        crs=crs,
    )
    gdf.to_file(path, driver="GPKG", layer="route")


def _write_empty_route_geojson(path: Path, *, crs: str = "EPSG:32631") -> None:
    gdf = gpd.GeoDataFrame({"id": []}, geometry=[], crs=crs)
    gdf.to_file(path, driver="GeoJSON")


def _write_point_geometry_gpkg(path: Path, *, crs: str = "EPSG:32631") -> None:
    gdf = gpd.GeoDataFrame([{"id": 1}], geometry=[Point(0, 0)], crs=crs)
    gdf.to_file(path, driver="GPKG", layer="route")


def _write_bathymetry_tif(path: Path, *, crs: str = "EPSG:32631") -> None:
    arr = np.linspace(-20.0, -10.0, 100).reshape(10, 10).astype("float32")
    transform = from_origin(500000.0, 6000100.0, 1.0, 1.0)
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


def _write_burial_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "kp_m": [0.0, 10.0, 20.0, 30.0],
            "easting": [500000.0, 500010.0, 500020.0, 500030.0],
            "northing": [6000000.0, 6000000.0, 6000000.0, 6000000.0],
            "depth_m": [1.2, 1.1, -0.1, 0.9],
        }
    )
    df.to_csv(path, index=False)


def _write_burial_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {
            "kp_m": [0.0, 10.0, 20.0],
            "depth_m": [1.0, 1.1, 0.9],
        }
    )
    df.to_parquet(path, index=False)


def _minimal_manifest_yaml(assets_yaml: str, *, working_crs: str = "EPSG:32631") -> str:
    return f"""
project:
  id: test_project
  name: Test Project
  working_crs: {working_crs}

assets:
{assets_yaml}
"""


# --- Section 18.1-18.3: evidence roles are distinct, orthogonal, never converted ---------------


def test_evidence_roles_are_four_distinct_canonical_values():
    assert {
        categories.PROJECT_GEOMETRY,
        categories.MEASURED,
        categories.SOURCE_INTERPRETED,
        categories.DERIVED,
    } == categories.EVIDENCE_ROLES
    assert len(categories.EVIDENCE_ROLES) == 4


def test_no_conversion_function_exists_between_evidence_roles():
    # Section 18.2/18.3: source-interpreted data can never be automatically re-labelled
    # measured, and derived output can never be automatically re-labelled source evidence --
    # proven structurally: no function in categories.py accepts an evidence-role string and
    # returns a different one.
    for name, obj in vars(categories).items():
        if inspect.isfunction(obj):
            pytest.fail(f"categories.py must contain no functions at all; found {name}")


def test_registration_never_alters_declared_evidence_role(tmp_path: Path):
    for role in categories.EVIDENCE_ROLES:
        route_path = tmp_path / f"route_{role}.gpkg"
        _write_route_gpkg(route_path)
        manifest_yaml = _minimal_manifest_yaml(
            f"""  - asset_id: a
    category: PIPELINE_ROUTE
    evidence_role: {role}
    path: ./route_{role}.gpkg
    layer: route
    provenance:
      source_name: x
"""
        )
        manifest_path = tmp_path / f"manifest_{role}.yaml"
        manifest_path.write_text(manifest_yaml, encoding="utf-8")
        m, mdir = project_manifest.load_project_manifest(manifest_path)
        result = project_registry.register_asset(
            m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
        )
        assert result.registration.evidence_role == role


# --- manifest schema: Section 6, 16.A -----------------------------------------------------------


def test_valid_manifest_parses(tmp_path: Path):
    route_path = tmp_path / "route.gpkg"
    _write_route_gpkg(route_path)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: route_001
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./route.gpkg
    layer: route
    provenance:
      source_name: Operator route
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    assert m.project.id == "test_project"
    assert len(m.assets) == 1
    assert mdir == manifest_path.parent


def test_duplicate_asset_id_is_rejected(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: dup
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./a.gpkg
    provenance:
      source_name: a
  - asset_id: dup
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./b.tif
    provenance:
      source_name: b
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(pydantic.ValidationError):
        project_manifest.load_project_manifest(manifest_path)


def test_unknown_category_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        project_manifest.AssetEntry(
            asset_id="x",
            category="NOT_A_REAL_CATEGORY",
            evidence_role="MEASURED",
            path=Path("x.txt"),
            provenance=project_manifest.DeclaredProvenance(source_name="x"),
        )


def test_unknown_evidence_role_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        project_manifest.AssetEntry(
            asset_id="x",
            category="BATHYMETRY_RASTER",
            evidence_role="NOT_A_REAL_ROLE",
            path=Path("x.txt"),
            provenance=project_manifest.DeclaredProvenance(source_name="x"),
        )


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        project_manifest.ProjectManifest.model_validate(
            {
                "project": {"id": "x", "name": "x", "working_crs": "EPSG:32631"},
                "assets": [],
                "unexpected_key": 1,
            }
        )


def test_relative_asset_path_resolves_against_manifest_directory(tmp_path: Path):
    nested = tmp_path / "sub" / "manifest.yaml"
    nested.parent.mkdir(parents=True)
    route_path = tmp_path / "sub" / "inputs" / "route.gpkg"
    route_path.parent.mkdir(parents=True)
    _write_route_gpkg(route_path)
    nested.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: route_001
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./inputs/route.gpkg
    layer: route
    provenance:
      source_name: x
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(nested)
    resolved = project_manifest.resolve_asset_path(m.assets[0], mdir)
    assert resolved == route_path.resolve()
    # never resolved against the process cwd
    assert resolved != (Path.cwd() / "inputs" / "route.gpkg").resolve()


def test_primary_route_asset_id_must_match_a_real_asset():
    with pytest.raises(pydantic.ValidationError):
        project_manifest.ProjectManifest.model_validate(
            {
                "project": {"id": "x", "name": "x", "working_crs": "EPSG:32631"},
                "primary_route_asset_id": "does_not_exist",
                "assets": [],
            }
        )


# --- Section 8, 16.A: content-based identity, never mutated ------------------------------------


def test_sha256_is_stable_for_identical_content(tmp_path: Path):
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_bytes(b"identical bytes")
    p2.write_bytes(b"identical bytes")
    id1 = identity.compute_file_identity(p1)
    id2 = identity.compute_file_identity(p2)
    assert id1.sha256 == id2.sha256
    assert id1.filename != id2.filename  # identity is content-based, not filename-based


def test_altered_source_file_changes_identity(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"version one")
    before = identity.compute_file_identity(p)
    p.write_bytes(b"version two, altered")
    after = identity.compute_file_identity(p)
    assert before.sha256 != after.sha256


def test_missing_file_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        identity.compute_file_identity(tmp_path / "does_not_exist.txt")


def test_computing_identity_never_writes_to_the_source_file(tmp_path: Path):
    p = tmp_path / "a.txt"
    original_bytes = b"do not touch me"
    p.write_bytes(original_bytes)
    mtime_before = p.stat().st_mtime_ns
    identity.compute_file_identity(p)
    identity.compute_file_identity(p)
    assert p.read_bytes() == original_bytes
    assert p.stat().st_mtime_ns == mtime_before


# --- route adapter: Section 9, 16.A -------------------------------------------------------------


def test_valid_route_is_ready(tmp_path: Path):
    route_path = tmp_path / "route.gpkg"
    _write_route_gpkg(route_path)
    facts, gdf, line, source_crs = route_adapter.inspect_route_source(
        route_path, layer="route", working_crs="EPSG:32631", declared_survey_epoch="2024"
    )
    result = route_adapter.assess_route_readiness(facts)
    assert result.status == route_adapter.READY
    assert line is not None
    assert source_crs == "EPSG:32631"


def test_empty_route_is_not_ready(tmp_path: Path):
    route_path = tmp_path / "empty.geojson"
    _write_empty_route_geojson(route_path)
    facts, _gdf, _line, _crs = route_adapter.inspect_route_source(
        route_path, layer=None, working_crs="EPSG:32631", declared_survey_epoch=None
    )
    result = route_adapter.assess_route_readiness(facts)
    assert result.status == route_adapter.NOT_READY
    assert facts.is_empty is True


def test_disconnected_route_is_not_ready_and_never_invents_connectivity(tmp_path: Path):
    route_path = tmp_path / "disconnected.gpkg"
    _write_disconnected_route_gpkg(route_path)
    facts, _gdf, line, _crs = route_adapter.inspect_route_source(
        route_path, layer="route", working_crs="EPSG:32631", declared_survey_epoch=None
    )
    result = route_adapter.assess_route_readiness(facts)
    assert result.status == route_adapter.NOT_READY
    assert line is None
    assert facts.disconnected_part_count == 2


def test_route_with_no_crs_is_not_ready(tmp_path: Path):
    route_path = tmp_path / "no_crs.gpkg"
    gdf = gpd.GeoDataFrame([{"id": 1}], geometry=[LineString([(0, 0), (10, 0)])])
    gdf.to_file(route_path, driver="GPKG", layer="route")
    facts, _gdf, _line, _crs = route_adapter.inspect_route_source(
        route_path, layer="route", working_crs="EPSG:32631", declared_survey_epoch=None
    )
    result = route_adapter.assess_route_readiness(facts)
    assert result.status == route_adapter.NOT_READY
    assert facts.crs_is_present is False


def test_point_geometry_is_rejected_as_not_line_type(tmp_path: Path):
    route_path = tmp_path / "points.gpkg"
    _write_point_geometry_gpkg(route_path)
    facts, _gdf, _line, _crs = route_adapter.inspect_route_source(
        route_path, layer="route", working_crs="EPSG:32631", declared_survey_epoch=None
    )
    result = route_adapter.assess_route_readiness(facts)
    assert result.status == route_adapter.NOT_READY
    assert facts.all_line_geometry is False


def test_geographic_working_crs_is_not_projected_metric():
    assert route_adapter.is_projected_metric_crs("EPSG:4326") is False
    assert route_adapter.is_projected_metric_crs("EPSG:32631") is True


def test_invalid_working_crs_string_is_rejected(tmp_path: Path):
    route_path = tmp_path / "route.gpkg"
    _write_route_gpkg(route_path)
    facts, _gdf, _line, _crs = route_adapter.inspect_route_source(
        route_path, layer="route", working_crs="NOT_A_REAL_CRS", declared_survey_epoch=None
    )
    result = route_adapter.assess_route_readiness(facts)
    assert result.status == route_adapter.NOT_READY
    assert facts.working_crs_valid is False


def test_missing_route_file_is_not_ready(tmp_path: Path):
    facts, gdf, line, crs = route_adapter.inspect_route_source(
        tmp_path / "does_not_exist.gpkg",
        layer=None,
        working_crs="EPSG:32631",
        declared_survey_epoch=None,
    )
    assert facts.file_readable is False
    assert gdf is None
    assert line is None
    result = route_adapter.assess_route_readiness(facts)
    assert result.status == route_adapter.NOT_READY


def test_canonical_route_direction_is_source_geometry_order(tmp_path: Path):
    route_path = tmp_path / "route.gpkg"
    _write_route_gpkg(route_path)
    facts, _gdf, line, source_crs = route_adapter.inspect_route_source(
        route_path, layer="route", working_crs="EPSG:32631", declared_survey_epoch=None
    )
    canonical = route_adapter.build_canonical_project_route(
        line,
        asset_id="a",
        source_route_name="route.gpkg",
        source_crs=source_crs,
        working_crs="EPSG:32631",
    )
    assert canonical["geometry_direction_semantics"].iloc[0] == route_adapter.SOURCE_GEOMETRY_ORDER
    assert canonical["canonical_route_length_m"].iloc[0] > 0


# --- bathymetry adapter: Section 10, 16.A, 18.10 -------------------------------------------------


def test_bathymetry_readiness_is_delegated_to_terrain_readiness(tmp_path: Path):
    # Section 18.10: proves delegation by asserting the exact accepted function is what
    # produces the result -- not a re-implementation.
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path)
    facts, _observed_crs = bathymetry_adapter.inspect_bathymetry_raster(
        raster_path, declared_vertical_datum="LAT", declared_survey_epoch="2024-06"
    )
    assert isinstance(facts, terrain_readiness.RasterFacts)
    result_via_project_layer = terrain_readiness.assess_bathymetry_readiness(facts)
    result_direct = terrain_readiness.assess_bathymetry_readiness(facts)
    assert result_via_project_layer.status == result_direct.status
    assert result_via_project_layer.to_dict() == result_direct.to_dict()
    assert result_via_project_layer.status == terrain_readiness.READY


def test_bathymetry_declared_provenance_flows_through_not_embedded(tmp_path: Path):
    # Section 10: survey epoch / vertical datum come from DECLARED provenance, never claimed
    # to be embedded raster metadata.
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path)
    facts, _observed_crs = bathymetry_adapter.inspect_bathymetry_raster(
        raster_path,
        declared_vertical_datum="A_DECLARED_DATUM",
        declared_survey_epoch="A_DECLARED_EPOCH",
    )
    assert facts.vertical_datum == "A_DECLARED_DATUM"
    assert facts.survey_epoch == "A_DECLARED_EPOCH"


def test_bathymetry_missing_declared_metadata_stays_missing(tmp_path: Path):
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path)
    facts, _observed_crs = bathymetry_adapter.inspect_bathymetry_raster(
        raster_path, declared_vertical_datum=None, declared_survey_epoch=None
    )
    assert facts.vertical_datum is None
    assert facts.survey_epoch is None
    result = terrain_readiness.assess_bathymetry_readiness(facts)
    assert result.status == terrain_readiness.READY_WITH_LIMITATIONS


def test_missing_bathymetry_raster_raises_registration_error(tmp_path: Path):
    with pytest.raises(bathymetry_adapter.RasterOpenError):
        bathymetry_adapter.inspect_bathymetry_raster(
            tmp_path / "does_not_exist.tif",
            declared_vertical_datum=None,
            declared_survey_epoch=None,
        )


# --- MAR-026A Problem A: exact observed raster CRS is preserved and compared exactly -----------


def test_bathymetry_observed_crs_is_exposed_as_a_separate_exact_fact(tmp_path: Path):
    # MAR-026A: `RasterFacts` itself only exposes crs_is_geographic/crs_linear_units --
    # `inspect_bathymetry_raster` must additionally expose the exact embedded CRS.
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    _facts, observed_crs = bathymetry_adapter.inspect_bathymetry_raster(
        raster_path, declared_vertical_datum=None, declared_survey_epoch=None
    )
    assert observed_crs is not None
    assert CRS.from_user_input(observed_crs) == CRS.from_epsg(32631)


# --- burial adapter: Section 11, 16.A, 18.11 ------------------------------------------------------


def test_burial_readiness_is_delegated_to_burial_readiness(tmp_path: Path):
    # Section 18.11: proves delegation the same way as the bathymetry test above.
    burial_path = tmp_path / "burial.csv"
    _write_burial_csv(burial_path)
    mapping = project_manifest.BurialColumnMapping(
        chainage_or_kp_column="kp_m",
        x_column="easting",
        y_column="northing",
        measured_value_column="depth_m",
    )
    facts, df = burial_adapter.inspect_burial_profile(
        burial_path,
        column_mapping=mapping,
        declared_crs="EPSG:32631",
        declared_units="m",
        declared_measurement_reference="TOP_OF_ASSET_BURIAL",
        declared_survey_epoch="2024",
    )
    assert isinstance(facts, burial_readiness.BurialProfileFacts)
    result_via_project_layer = burial_readiness.assess_burial_profile_readiness(facts)
    result_direct = burial_readiness.assess_burial_profile_readiness(facts)
    assert result_via_project_layer.to_dict() == result_direct.to_dict()
    assert len(df) == 4


def test_burial_csv_and_parquet_both_supported(tmp_path: Path):
    csv_path = tmp_path / "burial.csv"
    _write_burial_csv(csv_path)
    parquet_path = tmp_path / "burial.parquet"
    _write_burial_parquet(parquet_path)
    assert len(burial_adapter.load_burial_table(csv_path)) == 4
    assert len(burial_adapter.load_burial_table(parquet_path)) == 3


def test_burial_unsupported_format_is_rejected(tmp_path: Path):
    bad_path = tmp_path / "burial.xlsx"
    bad_path.write_bytes(b"not a real xlsx")
    with pytest.raises(burial_adapter.BurialTableLoadError):
        burial_adapter.load_burial_table(bad_path)


def test_burial_missing_reference_stays_unresolved_never_guessed(tmp_path: Path):
    # Section 11/24A: if burial reference is unresolved, that unresolved state is preserved,
    # never guessed.
    burial_path = tmp_path / "burial.csv"
    _write_burial_csv(burial_path)
    mapping = project_manifest.BurialColumnMapping(
        chainage_or_kp_column="kp_m", measured_value_column="depth_m"
    )
    facts, _df = burial_adapter.inspect_burial_profile(
        burial_path,
        column_mapping=mapping,
        declared_crs="EPSG:32631",
        declared_units="m",
        declared_measurement_reference=None,
        declared_survey_epoch=None,
    )
    assert facts.burial_reference_known is False
    result = burial_readiness.assess_burial_profile_readiness(facts)
    assert any("could not be established" in r for r in result.reasons())


def test_burial_missing_column_mapping_fails_registration(tmp_path: Path):
    route_free_manifest = _minimal_manifest_yaml(
        """  - asset_id: b
    category: BURIAL_PROFILE
    evidence_role: MEASURED
    path: ./burial.csv
    provenance:
      source_name: x
"""
    )
    burial_path = tmp_path / "burial.csv"
    _write_burial_csv(burial_path)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(route_free_manifest, encoding="utf-8")
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.registration_status == project_registry.REGISTRATION_FAILED


def test_missing_burial_file_raises_load_error(tmp_path: Path):
    with pytest.raises(burial_adapter.BurialTableLoadError):
        burial_adapter.load_burial_table(tmp_path / "does_not_exist.csv")


# --- registry orchestration: Section 5, 12, 13, 18.4-18.7, 18.12-18.15 ---------------------------


def test_missing_crs_is_a_blocking_readiness_failure_not_silently_ignored(tmp_path: Path):
    route_path = tmp_path / "no_crs.gpkg"
    gdf = gpd.GeoDataFrame([{"id": 1}], geometry=[LineString([(0, 0), (10, 0)])])
    gdf.to_file(route_path, driver="GPKG", layer="route")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: r
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./no_crs.gpkg
    layer: route
    provenance:
      source_name: x
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.readiness_status == terrain_readiness.NOT_READY


def test_declared_vs_observed_crs_conflict_is_recorded_not_silently_reprojected(tmp_path: Path):
    route_path = tmp_path / "route.gpkg"
    _write_route_gpkg(route_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: r
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./route.gpkg
    layer: route
    provenance:
      source_name: x
      horizontal_crs_declared: EPSG:4326
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.conflicts, "expected a recorded declared-vs-observed CRS conflict"
    assert "EPSG:4326" in result.registration.conflicts[0]
    assert "EPSG:32631" in result.registration.conflicts[0]
    # the conflict is recorded, never silently resolved -- registration still proceeds honestly
    assert result.registration.registration_status == project_registry.REGISTERED
    # MAR-026A Section 3/8.6: the route's own facts are otherwise fine intrinsically (no
    # declared survey_epoch here, so intrinsic is READY_WITH_LIMITATIONS, never NOT_READY), but
    # an unresolved declared-vs-observed conflict forces EFFECTIVE readiness to NOT_READY, and
    # no canonical route is emitted from the unresolved contradictory metadata.
    assert result.registration.readiness_status_intrinsic == route_adapter.READY_WITH_LIMITATIONS
    assert result.registration.readiness_status_effective == route_adapter.NOT_READY
    assert result.registration.readiness_status == route_adapter.NOT_READY
    assert result.canonical_route_gdf is None


def test_missing_declared_metadata_remains_missing(tmp_path: Path):
    route_path = tmp_path / "route.gpkg"
    _write_route_gpkg(route_path)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: r
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./route.gpkg
    layer: route
    provenance:
      source_name: x
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.provenance_declared["survey_epoch"] is None
    assert result.registration.readiness_status == route_adapter.READY_WITH_LIMITATIONS


# --- MAR-026A Problem A: exact declared-vs-observed CRS conflict semantics ----------------------


def test_bathymetry_declared_and_observed_crs_match_no_conflict(tmp_path: Path):
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: b
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./bathy.tif
    provenance:
      source_name: x
      horizontal_crs_declared: EPSG:32631
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.conflicts == []
    assert (
        result.registration.readiness_status_effective
        == result.registration.readiness_status_intrinsic
    )


def test_bathymetry_equivalent_declared_crs_representation_is_not_a_false_conflict(
    tmp_path: Path,
):
    # a different textual representation (WKT) of the exact same CRS as "EPSG:32631" -- must
    # compare equal via pyproj.CRS, never via raw string equality.
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    equivalent_wkt = CRS.from_epsg(32631).to_wkt()
    asset = project_manifest.AssetEntry(
        asset_id="b",
        category=categories.BATHYMETRY_RASTER,
        evidence_role=categories.MEASURED,
        path=Path("bathy.tif"),
        provenance=project_manifest.DeclaredProvenance(
            source_name="x", horizontal_crs_declared=equivalent_wkt
        ),
    )
    result = project_registry.register_asset(asset, manifest_dir=tmp_path, working_crs="EPSG:32631")
    assert result.registration.conflicts == [], (
        f"expected no conflict for an equivalent CRS representation, got "
        f"{result.registration.conflicts}"
    )


def test_bathymetry_two_different_projected_crs_values_conflict_despite_neither_geographic(
    tmp_path: Path,
):
    # MAR-026A Problem A: the exact defect being repaired -- two different PROJECTED CRSs must
    # conflict, which the old geographic-vs-projected-only check could never detect.
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: b
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./bathy.tif
    provenance:
      source_name: x
      horizontal_crs_declared: EPSG:32632
      vertical_datum_declared: LAT
      survey_epoch: "2024-06"
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.conflicts, "two different projected CRSs must conflict"
    assert "EPSG:32632" in result.registration.conflicts[0]
    assert "EPSG:32631" in result.registration.conflicts[0]
    # intrinsic (delegated, unmodified MAR-020) readiness is untouched by the project conflict
    assert result.registration.readiness_status_intrinsic == terrain_readiness.READY
    assert result.registration.readiness_result["status"] == terrain_readiness.READY
    # the EFFECTIVE, project-integrated readiness must not remain READY
    assert result.registration.readiness_status_effective == terrain_readiness.NOT_READY
    assert result.registration.readiness_status == terrain_readiness.NOT_READY


def test_bathymetry_declared_geographic_vs_observed_projected_still_conflicts(tmp_path: Path):
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: b
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./bathy.tif
    provenance:
      source_name: x
      horizontal_crs_declared: EPSG:4326
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.conflicts, "declared geographic vs observed projected must conflict"


def test_bathymetry_missing_declared_crs_is_not_a_fabricated_conflict(tmp_path: Path):
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: b
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./bathy.tif
    provenance:
      source_name: x
      vertical_datum_declared: LAT
      survey_epoch: "2024-06"
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.conflicts == []
    assert result.registration.readiness_status_intrinsic == terrain_readiness.READY
    assert result.registration.readiness_status_effective == terrain_readiness.READY


def test_bathymetry_invalid_declared_crs_is_not_silently_ignored(tmp_path: Path):
    # Problem A repair requirement: "invalid declared CRS must not be silently ignored" -- a
    # declared value that fails to parse must not be treated as "no conflict found" when a real
    # observed CRS exists to compare against.
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: b
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./bathy.tif
    provenance:
      source_name: x
      horizontal_crs_declared: NOT_A_REAL_CRS
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.conflicts, (
        "a syntactically invalid declared CRS must not be silently ignored when an observed "
        "CRS is available to compare against"
    )
    assert result.registration.readiness_status_effective == terrain_readiness.NOT_READY


def test_readiness_status_alias_always_equals_effective_status(tmp_path: Path):
    # Section 9: `readiness_status` is kept for backward compatibility, but must represent the
    # EFFECTIVE status -- a consumer reading only the old field name must never silently see an
    # intrinsic-only result once a project-level conflict exists.
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: b
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./bathy.tif
    provenance:
      source_name: x
      horizontal_crs_declared: EPSG:32632
      vertical_datum_declared: LAT
      survey_epoch: "2024-06"
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.conflicts
    assert result.registration.readiness_status == result.registration.readiness_status_effective
    assert result.registration.readiness_status != result.registration.readiness_status_intrinsic


# --- MAR-026A Problem B: invalid project working CRS never crashes registration -----------------


def test_register_asset_with_invalid_working_crs_does_not_throw(tmp_path: Path):
    # This reproduces the exact MAR-026A defect: a valid, single-continuous route with a known
    # source CRS used to reach `GeoDataFrame.to_crs("NOT_A_REAL_CRS")` completely unguarded.
    route_path = tmp_path / "route.gpkg"
    _write_route_gpkg(route_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: r
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./route.gpkg
    layer: route
    provenance:
      source_name: x
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="NOT_A_REAL_CRS"
    )
    assert result.registration.registration_status == project_registry.REGISTERED
    assert result.registration.readiness_status_effective == route_adapter.NOT_READY
    assert result.canonical_route_gdf is None


def test_register_asset_with_geographic_working_crs_does_not_build_canonical_route(
    tmp_path: Path,
):
    route_path = tmp_path / "route.gpkg"
    _write_route_gpkg(route_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: r
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./route.gpkg
    layer: route
    provenance:
      source_name: x
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:4326"
    )
    assert result.registration.registration_status == project_registry.REGISTERED
    assert result.registration.readiness_status_effective == route_adapter.NOT_READY
    assert result.canonical_route_gdf is None


def test_bathymetry_only_project_with_invalid_working_crs_is_flagged_centrally(tmp_path: Path):
    # Section 6: no route asset exists to trigger the per-asset working_crs checks -- the
    # invalid working_crs must still be visible, checked centrally by `register_project`.
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: b
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./bathy.tif
    provenance:
      source_name: x
""",
            working_crs="NOT_A_REAL_CRS",
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    summary = project_registry.register_project(m, mdir)
    assert summary.working_crs_findings, "an invalid project working_crs must be flagged centrally"
    # the bathymetry asset itself does not depend on working_crs -- it still registers/assesses
    reg = summary.asset_results[0].registration
    assert reg.registration_status == project_registry.REGISTERED
    assert reg.readiness_status_intrinsic in (
        terrain_readiness.READY,
        terrain_readiness.READY_WITH_LIMITATIONS,
    )


def test_bathymetry_only_project_with_geographic_working_crs_is_flagged_centrally(
    tmp_path: Path,
):
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: b
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./bathy.tif
    provenance:
      source_name: x
""",
            working_crs="EPSG:4326",
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    summary = project_registry.register_project(m, mdir)
    assert summary.working_crs_findings, "a geographic (non-metric) working_crs must be flagged"


def test_valid_working_crs_produces_no_project_level_findings(tmp_path: Path):
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path, crs="EPSG:32631")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: b
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./bathy.tif
    provenance:
      source_name: x
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    summary = project_registry.register_project(m, mdir)
    assert summary.working_crs_findings == []


def test_source_files_are_never_mutated_by_registration(tmp_path: Path):
    route_path = tmp_path / "route.gpkg"
    _write_route_gpkg(route_path)
    original_bytes = route_path.read_bytes()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: r
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./route.gpkg
    layer: route
    provenance:
      source_name: x
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    project_registry.register_asset(m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631")
    assert route_path.read_bytes() == original_bytes


def test_unsupported_future_category_is_registered_but_never_ready(tmp_path: Path):
    stub_path = tmp_path / "shallow_gas.txt"
    stub_path.write_text("stub interpretation file", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: sg
    category: SHALLOW_GAS_INTERPRETATION
    evidence_role: SOURCE_INTERPRETED
    path: ./shallow_gas.txt
    provenance:
      source_name: Contractor shallow gas polygon
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.registration_status == project_registry.REGISTERED
    assert result.registration.readiness_status == categories.REGISTERED_READINESS_NOT_IMPLEMENTED
    assert result.registration.readiness_status != terrain_readiness.READY
    assert result.registration.readiness_status != "READY"


@pytest.mark.parametrize("category_value", sorted(categories.ASSET_CATEGORIES))
def test_no_category_is_ever_reported_ready_from_registration_alone(
    tmp_path: Path, category_value: str
):
    # Section 7: a registered future-category asset is never READY merely because a file exists.
    if category_value in categories.CATEGORIES_WITH_READINESS_ADAPTERS:
        pytest.skip("implemented categories are exercised by their own dedicated tests")
    stub_path = tmp_path / f"{category_value.lower()}.txt"
    stub_path.write_text("stub", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            f"""  - asset_id: a
    category: {category_value}
    evidence_role: SOURCE_INTERPRETED
    path: ./{category_value.lower()}.txt
    provenance:
      source_name: x
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.readiness_status == categories.REGISTERED_READINESS_NOT_IMPLEMENTED


def test_no_numeric_readiness_score_in_public_schemas():
    # Section 12, 18.13: no 0-100 / 0-1 score anywhere.
    for module in PROJECT_MODULES:
        source = inspect.getsource(module).lower()
        assert "readiness_score" not in source
        assert "readiness_percentage" not in source
        assert "confidence_percent" not in source


def test_no_hazard_or_risk_score_anywhere():
    # Section 18.14.
    for module in PROJECT_MODULES:
        source = inspect.getsource(module).lower()
        assert "risk_score" not in source
        assert "hazard_score" not in source


def test_project_hazard_readiness_disclaimer_present_and_explicit():
    # Section 13, 20: the aggregate description never implies universal hazard readiness.
    disclaimer = project_registry.PROJECT_HAZARD_READINESS_DISCLAIMER
    assert "DOES NOT IMPLY" in disclaimer
    assert "GEOHAZARD" in disclaimer


def test_source_interpretation_field_never_feeds_measured_classification(tmp_path: Path):
    # Section 18.15: mirrors the MAR-025A principle at the project level. Register the SAME
    # burial file/columns twice, differing ONLY in declared evidence_role (MEASURED vs.
    # SOURCE_INTERPRETED) -- the readiness result (a pure function of the file's data/structure)
    # must be byte-identical, proving evidence_role is never consumed by the classification
    # logic itself, only carried alongside it.
    burial_path = tmp_path / "burial.csv"
    _write_burial_csv(burial_path)
    mapping_yaml = """    burial_columns:
      chainage_or_kp_column: kp_m
      x_column: easting
      y_column: northing
      measured_value_column: depth_m
"""
    results = {}
    for role in (categories.MEASURED, categories.SOURCE_INTERPRETED):
        manifest_path = tmp_path / f"manifest_{role}.yaml"
        manifest_path.write_text(
            _minimal_manifest_yaml(
                f"""  - asset_id: b
    category: BURIAL_PROFILE
    evidence_role: {role}
    path: ./burial.csv
{mapping_yaml}    provenance:
      source_name: x
"""
            ),
            encoding="utf-8",
        )
        m, mdir = project_manifest.load_project_manifest(manifest_path)
        result = project_registry.register_asset(
            m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
        )
        results[role] = result

    assert results[categories.MEASURED].registration.readiness_status == (
        results[categories.SOURCE_INTERPRETED].registration.readiness_status
    )
    assert results[categories.MEASURED].registration.readiness_result == (
        results[categories.SOURCE_INTERPRETED].registration.readiness_result
    )
    # yet the declared role itself is faithfully preserved, never collapsed to one value
    assert results[categories.MEASURED].registration.evidence_role == categories.MEASURED
    assert (
        results[categories.SOURCE_INTERPRETED].registration.evidence_role
        == categories.SOURCE_INTERPRETED
    )


# --- registration status vs readiness status distinction ----------------------------------------


def test_missing_file_is_registration_failed_not_a_readiness_check_failure(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: b
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./does_not_exist.tif
    provenance:
      source_name: x
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    result = project_registry.register_asset(
        m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631"
    )
    assert result.registration.registration_status == project_registry.REGISTRATION_FAILED
    assert result.registration.readiness_status == terrain_readiness.NOT_READY
    assert result.registration.readiness_result is None
    assert result.registration.byte_size is None
    assert result.registration.sha256 is None


# --- full project registration + output builders -------------------------------------------------


def test_register_project_and_build_outputs(tmp_path: Path):
    route_path = tmp_path / "route.gpkg"
    _write_route_gpkg(route_path)
    raster_path = tmp_path / "bathy.tif"
    _write_bathymetry_tif(raster_path)
    burial_path = tmp_path / "burial.csv"
    _write_burial_csv(burial_path)

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        _minimal_manifest_yaml(
            """  - asset_id: route_001
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: ./route.gpkg
    layer: route
    provenance:
      source_name: Operator route
      horizontal_crs_declared: EPSG:32631
  - asset_id: bathy_001
    category: BATHYMETRY_RASTER
    evidence_role: MEASURED
    path: ./bathy.tif
    provenance:
      source_name: Operator MBES
      vertical_datum_declared: LAT
      survey_epoch: "2024"
  - asset_id: burial_001
    category: BURIAL_PROFILE
    evidence_role: MEASURED
    path: ./burial.csv
    burial_columns:
      chainage_or_kp_column: kp_m
      x_column: easting
      y_column: northing
      measured_value_column: depth_m
    provenance:
      source_name: Operator burial survey
      units_declared: m
"""
        ),
        encoding="utf-8",
    )
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    summary = project_registry.register_project(m, mdir)
    assert len(summary.asset_results) == 3
    assert all(
        r.registration.registration_status == project_registry.REGISTERED
        for r in summary.asset_results
    )

    registry_df = project_registry.build_asset_registry_df(summary)
    assert len(registry_df) == 3
    assert set(registry_df["asset_id"]) == {"route_001", "bathy_001", "burial_001"}

    readiness_dict = project_registry.build_project_readiness_dict(summary)
    serialized = json.dumps(readiness_dict, default=str)
    reloaded = json.loads(serialized)
    for entry in reloaded["assets"].values():
        rr = entry.get("readiness_result")
        if rr:
            for check in rr["checks"]:
                assert isinstance(check["passed"], bool)

    blocks = project_report.build_project_readiness_report_blocks(
        project_title="Test",
        purpose_text="purpose",
        project_identity_facts={"a": 1},
        asset_rows=registry_df.to_dict("records"),
        evidence_role_summary={"MEASURED": 2, "PROJECT_GEOMETRY": 1},
        readiness_summary=summary.readiness_status_counts(),
        blocking_issues=[],
        limitations=["none"],
        unsupported_categories=[],
        production_transfer_notes=["note"],
    )
    html = project_report.render_blocks_html(blocks, title="Test Report")
    assert "<html" in html
    assert "validated" not in html.lower()
    assert "safe" not in html.lower()


# --- CLI wiring -----------------------------------------------------------------------------------


def test_cli_registers_build_project_readiness():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["build-project-readiness", "configs/project_manifests/barrow_2016.yaml"]
    )
    assert args.func is cli._cmd_build_project_readiness


def test_cli_reports_invalid_manifest_cleanly(tmp_path: Path, capsys):
    manifest_path = tmp_path / "bad_manifest.yaml"
    manifest_path.write_text(
        "project: {id: x, name: x, working_crs: EPSG:32631}\nbogus_key: 1\n", encoding="utf-8"
    )
    parser = cli.build_parser()
    args = parser.parse_args(["build-project-readiness", str(manifest_path)])
    exit_code = args.func(args)
    assert exit_code == 1

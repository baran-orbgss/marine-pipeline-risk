"""Offline unit tests for the MAR-029 route-referenced observed seabed change evidence layer
(`change.route_evidence`, `change.route_evidence_manifest`, CLI wiring).

A small synthetic end-to-end fixture is generated inside the tests: a projected metric CRS, a
deterministic straight route, the real MAR-027 canonical route-reference grid, and an analytical
DoD raster whose cells are set so that every expected route sample can be calculated exactly (no
randomness). Test names map to MAR-029 Section 27's required proof matrix (items 1-47).

Fixture geometry (all in EPSG:32631):
  DoD raster: origin (500000, 6000050), 10 m x 10 m cells, 12 columns x 10 rows, nodata -9999.
    row 3 = +100 everywhere, row 5 = -100 everywhere (adjacent rows wildly different -> proves no
    interpolation), row 4 = ROW4_VALUES below, every other row = 7.
  Route: (500005, 6000003) -> (500145, 6000003), 140 m, interval 10 m -> 15 stations at
    x = 500005 + 10*k. Every station lies in raster row 4 (y in (6000000, 6000010]) at column k;
    columns 12-14 are outside the raster (x >= 500120). Cell centres are (500005+10k, 6000005), so
    every inside station is exactly 2.0 m from its cell centre.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
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
from marine_engine.change import dod, route_evidence, uncertainty
from marine_engine.change import route_evidence_manifest as rem
from marine_engine.project import manifest as project_manifest
from marine_engine.project import model as project_model
from marine_engine.project import registry as project_registry
from marine_engine.project import route_adapter, route_reference

WORKING_CRS = "EPSG:32631"
NODATA = -9999.0
ROW4_VALUES = [0.5, -0.75, 0.0, NODATA, 0.001, -3.2, 2.25, -0.001, 1.0, -1.0, 0.3, -0.3]
ROUTE_COORDS = [(500005.0, 6000003.0), (500145.0, 6000003.0)]
INSIDE_STATIONS = list(range(12))
OUTSIDE_STATIONS = [12, 13, 14]
NODATA_STATION = 3
ZERO_STATION = 2
EXPECTED_RAISING = [0, 4, 6, 8, 10]
EXPECTED_LOWERING = [1, 5, 7, 9, 11]
# MAR-029A: the only interpretable definition, taken from the accepted MAR-021 module itself.
CANONICAL_DEFINITION = dod.DoDResult.__dataclass_fields__["definition"].default
REVERSED_DEFINITION = "delta_bed_elevation_m = bed_elevation_epoch1_m - bed_elevation_epoch2_m"


def _f32(value: float) -> float:
    """The exact float32 value the raster stores -- 'sampled exactly' means this value."""
    return float(np.float32(value))


# --- synthetic fixtures ---------------------------------------------------------------------------


def _write_route_gpkg(
    path: Path, *, coords: list[tuple[float, float]] | None = None, crs: str = WORKING_CRS
) -> None:
    gpd.GeoDataFrame([{"id": 1}], geometry=[LineString(coords or ROUTE_COORDS)], crs=crs).to_file(
        path, driver="GPKG", layer="route"
    )


def _dod_array(*, row4: list[float] | None = None) -> np.ndarray:
    arr = np.full((10, 12), 7.0, dtype="float32")
    arr[3, :] = 100.0
    arr[5, :] = -100.0
    arr[4, :] = row4 if row4 is not None else ROW4_VALUES
    return arr


def _write_dod_tif(
    path: Path,
    *,
    crs: str | None = WORKING_CRS,
    nodata: float | None = NODATA,
    array: np.ndarray | None = None,
    count: int = 1,
    dtype: str = "float32",
    rotated: bool = False,
    tags: dict[str, str] | None = None,
) -> None:
    arr = _dod_array() if array is None else array
    transform = from_origin(500000.0, 6000050.0, 10.0, 10.0)
    if rotated:
        transform = rasterio.Affine(10.0, 0.5, 500000.0, 0.5, -10.0, 6000050.0)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        for band in range(1, count + 1):
            dst.write(arr.astype(dtype), band)
        if tags is None:
            tags = {
                "scientific_role": "MULTI_EPOCH_SEABED_CHANGE_POC",
                "definition": route_evidence.ACCEPTED_DOD_DEFINITION,
            }
        if tags:
            dst.update_tags(**tags)


def _project_yaml(
    *,
    route_asset_id: str = "route_a",
    route_path: str = "./route_a.gpkg",
    primary: str | None = "route_a",
    interval_m: float | None = 10.0,
    project_id: str = "test_project",
) -> str:
    head = f"project:\n  id: {project_id}\n  name: Test Project\n  working_crs: {WORKING_CRS}\n"
    if primary is not None:
        head += f"primary_route_asset_id: {primary}\n"
    if interval_m is not None:
        head += f"route_reference:\n  interval_m: {interval_m}\n"
    return (
        head
        + f"""assets:
  - asset_id: {route_asset_id}
    category: PIPELINE_ROUTE
    evidence_role: PROJECT_GEOMETRY
    path: {route_path}
    layer: route
    provenance:
      source_name: Operator route
      horizontal_crs_declared: {WORKING_CRS}
"""
    )


def _route_change_yaml(
    *,
    project_manifest_path: str = "./project.yaml",
    route_asset_id: str = "route_a",
    dod_path: str = "./dod.tif",
    provenance_path: str | None = "./prov.json",
    declared_crs: str | None = WORKING_CRS,
    declared_role: str | None = "MULTI_EPOCH_SEABED_CHANGE_POC",
    declared_definition: str | None = CANONICAL_DEFINITION,
) -> str:
    text = f"""project_manifest: {project_manifest_path}
route_asset_id: {route_asset_id}
dod:
  path: {dod_path}
  source_change_study_id: synthetic_two_epoch_change_study
  epoch1_survey_epoch_declared: "2018-10"
  epoch2_survey_epoch_declared: "2020-11"
"""
    if provenance_path is not None:
        text += f"  provenance_path: {provenance_path}\n"
    if declared_crs is not None:
        text += f"  horizontal_crs_declared: {declared_crs}\n"
    if declared_role is not None:
        text += f"  source_scientific_role_declared: {declared_role}\n"
    if declared_definition is not None:
        text += f'  dod_definition_declared: "{declared_definition}"\n'
    return text


def _fixture(
    tmp_path: Path,
    *,
    route_coords: list[tuple[float, float]] | None = None,
    route_asset_id: str = "route_a",
    primary: str | None = "route_a",
    interval_m: float | None = 10.0,
    dod_kwargs: dict | None = None,
    route_change_kwargs: dict | None = None,
    write_dod: bool = True,
    write_prov: bool = True,
) -> Path:
    _write_route_gpkg(tmp_path / "route_a.gpkg", coords=route_coords)
    if write_dod:
        _write_dod_tif(tmp_path / "dod.tif", **(dod_kwargs or {}))
    if write_prov:
        (tmp_path / "prov.json").write_text(
            json.dumps({"scientific_role": "MULTI_EPOCH_SEABED_CHANGE_POC"}), encoding="utf-8"
        )
    (tmp_path / "project.yaml").write_text(
        _project_yaml(route_asset_id=route_asset_id, primary=primary, interval_m=interval_m),
        encoding="utf-8",
    )
    kwargs = {"route_asset_id": route_asset_id, **(route_change_kwargs or {})}
    path = tmp_path / "route_change.yaml"
    path.write_text(_route_change_yaml(**kwargs), encoding="utf-8")
    return path


def _run(path: Path):
    return route_evidence.run_route_change_evidence(path)


def _run_and_write(tmp_path: Path, path: Path, out_name: str = "out"):
    result, model = _run(path)
    final = route_evidence.write_route_change_evidence_outputs(result, tmp_path / out_name)
    return final, model


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys(obj) -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_keys(v))
    elif isinstance(obj, list):
        for v in obj:
            keys.extend(_keys(v))
    return keys


# --- Section 27.1-5: input / provenance -----------------------------------------------------------


def test_01_explicit_route_change_relationship_required():
    with pytest.raises(pydantic.ValidationError):
        rem.RouteChangeEvidenceManifest.model_validate(
            {"project_manifest": "p.yaml", "dod": {"path": "d.tif", "source_change_study_id": "s"}}
        )  # no route_asset_id
    with pytest.raises(pydantic.ValidationError):
        rem.RouteChangeEvidenceManifest.model_validate(
            {"project_manifest": "p.yaml", "route_asset_id": "route_a"}
        )  # no dod
    with pytest.raises(pydantic.ValidationError):
        rem.RouteChangeEvidenceManifest.model_validate(
            {"project_manifest": "p.yaml", "route_asset_id": "route_a", "dod": {"path": "d.tif"}}
        )  # no source change study identity, no definition, no role
    complete = {
        "path": "d.tif",
        "source_change_study_id": "s",
        "dod_definition_declared": CANONICAL_DEFINITION,
        "source_scientific_role_declared": "MULTI_EPOCH_SEABED_CHANGE_POC",
    }
    assert rem.DoDSourceDeclaration.model_validate(complete).dod_definition_declared == (
        CANONICAL_DEFINITION
    )
    with pytest.raises(pydantic.ValidationError):
        rem.RouteChangeEvidenceManifest.model_validate(
            {
                "project_manifest": "p.yaml",
                "route_asset_id": "route_a",
                "dod": {"path": "d.tif", "source_change_study_id": "s"},
                "significance_threshold_m": 0.3,
            }
        )  # extra="forbid"
    with pytest.raises(pydantic.ValidationError):
        rem.DoDSourceDeclaration.model_validate(
            {"path": "d.tif", "source_change_study_id": "s", "auto_associate": True}
        )


def test_02_relationship_never_inferred_from_filename_or_generic_manifest(tmp_path: Path):
    # A DoD whose filename carries the project AND route name, in the same directory as the only
    # route, still yields nothing unless the route-change manifest names the primary route.
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    _write_dod_tif(tmp_path / "test_project_route_a_dod.tif")
    (tmp_path / "project.yaml").write_text(_project_yaml(), encoding="utf-8")
    path = tmp_path / "rc.yaml"
    path.write_text(
        _route_change_yaml(
            route_asset_id="route_b",
            dod_path="./test_project_route_a_dod.tif",
            provenance_path=None,
        ),
        encoding="utf-8",
    )
    result, _ = _run(path)
    assert result.status == route_evidence.ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE
    assert result.reason_code == route_evidence.ROUTE_ASSET_ID_MISMATCH
    assert result.samples is None
    # The generic ProjectManifest carries no DoD/hazard linkage concept at all.
    with pytest.raises(pydantic.ValidationError):
        project_manifest.ProjectManifest.model_validate(
            {
                "project": {"id": "x", "name": "x", "working_crs": WORKING_CRS},
                "dod": {"path": "d.tif"},
            }
        )
    source = inspect.getsource(route_evidence)
    assert "glob(" not in source
    assert ".stem" not in source
    assert "listdir" not in source


def test_03_dod_sha_changes_when_bytes_change_not_when_renamed(tmp_path: Path):
    _write_dod_tif(tmp_path / "dod.tif")
    identity_a, _ = route_evidence.inspect_dod_source(tmp_path / "dod.tif")
    changed = _dod_array()
    changed[4, 0] = 0.5000001
    _write_dod_tif(tmp_path / "dod_changed.tif", array=changed)
    identity_b, _ = route_evidence.inspect_dod_source(tmp_path / "dod_changed.tif")
    assert identity_a.sha256 != identity_b.sha256
    renamed = tmp_path / "renamed.tif"
    renamed.write_bytes((tmp_path / "dod.tif").read_bytes())
    identity_c, _ = route_evidence.inspect_dod_source(renamed)
    assert identity_c.sha256 == identity_a.sha256
    assert identity_c.filename != identity_a.filename


def test_04_dod_source_byte_identical_after_run(tmp_path: Path):
    path = _fixture(tmp_path)
    before = _sha(tmp_path / "dod.tif")
    final, _ = _run_and_write(tmp_path, path)
    assert final.available
    assert _sha(tmp_path / "dod.tif") == before
    assert final.dod_identity is not None and final.dod_identity.sha256 == before
    validation = route_evidence.build_route_change_evidence_validation(final)
    assert validation["question_h_was_the_dod_source_modified_during_the_run"] == "NO"


def test_05_declared_vs_observed_crs_kept_separate(tmp_path: Path):
    path = _fixture(tmp_path, route_change_kwargs={"declared_crs": "EPSG:32632"})
    final, _ = _run_and_write(tmp_path, path)
    assert final.status == route_evidence.ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE
    assert final.reason_code == route_evidence.DOD_DECLARED_CRS_CONFLICT
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    crs = metadata["crs_compatibility"]
    assert crs["dod_declared_crs"] == "EPSG:32632"
    assert crs["dod_observed_crs"] == "EPSG:32631"
    assert crs["declared_vs_observed_conflict"]
    assert metadata["dod_observed_facts"]["observed_crs"] == "EPSG:32631"
    assert metadata["route_change_manifest_declared"]["horizontal_crs_declared"] == "EPSG:32632"
    # No declared CRS is not a conflict; the observed one is still recorded on its own.
    path2 = _fixture(tmp_path, route_change_kwargs={"declared_crs": None})
    result2, _ = _run(path2)
    assert result2.available
    assert result2.declared_vs_observed_crs_conflict is None
    assert result2.dod_observed is not None and result2.dod_observed.observed_crs == "EPSG:32631"
    # An unparsable declared CRS is a conflict, never silently ignored.
    path3 = _fixture(tmp_path, route_change_kwargs={"declared_crs": "NOT_A_CRS"})
    result3, _ = _run(path3)
    assert result3.reason_code == route_evidence.DOD_DECLARED_CRS_CONFLICT


# --- Section 27.6-11: CRS / support ---------------------------------------------------------------


def test_06_equivalent_crs_representations_accepted(tmp_path: Path):
    wkt = CRS.from_epsg(32631).to_wkt()
    path = _fixture(
        tmp_path, dod_kwargs={"crs": wkt}, route_change_kwargs={"declared_crs": "epsg:32631"}
    )
    result, _ = _run(path)
    assert result.available, result.findings
    assert result.dod_vs_route_crs_status == "SEMANTICALLY_EQUIVALENT"
    assert result.declared_vs_observed_crs_conflict is None


def test_07_different_projected_crs_blocked(tmp_path: Path):
    path = _fixture(
        tmp_path, dod_kwargs={"crs": "EPSG:32632"}, route_change_kwargs={"declared_crs": None}
    )
    final, _ = _run_and_write(tmp_path, path)
    assert final.status == route_evidence.ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE
    assert final.reason_code == route_evidence.DOD_CRS_MISMATCH
    assert final.samples is None
    assert not final.products_written
    assert not (tmp_path / "out" / "route_observed_seabed_change.gpkg").exists()
    assert not (tmp_path / "out" / "route_observed_seabed_change.parquet").exists()


def test_08_no_raster_reprojection_or_resampling(tmp_path: Path):
    source = inspect.getsource(route_evidence)
    for forbidden in ("reproject(", "to_crs(", "Resampling", "rasterio.warp", "WarpedVRT"):
        assert forbidden not in source
    path = _fixture(tmp_path)
    before = _sha(tmp_path / "dod.tif")
    final, _ = _run_and_write(tmp_path, path)
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    assert metadata["crs_compatibility"]["reprojection_performed"] is False
    assert metadata["crs_compatibility"]["resampling_performed"] is False
    assert _sha(tmp_path / "dod.tif") == before


def test_09_single_band_analytical_raster_required(tmp_path: Path):
    path = _fixture(tmp_path, dod_kwargs={"count": 3})
    result, _ = _run(path)
    assert result.reason_code == route_evidence.DOD_NOT_SINGLE_BAND_ANALYTICAL_RASTER
    assert result.dod_observed is not None and result.dod_observed.band_count == 3
    byte_array = np.full((10, 12), 3, dtype="uint8")
    path2 = _fixture(tmp_path, dod_kwargs={"array": byte_array, "dtype": "uint8", "nodata": None})
    result2, _ = _run(path2)
    assert result2.reason_code == route_evidence.DOD_NOT_SINGLE_BAND_ANALYTICAL_RASTER
    assert result2.dod_observed is not None and result2.dod_observed.dtype == "uint8"


def test_10_outside_extent_point_becomes_unavailable(tmp_path: Path):
    result, _ = _run(_fixture(tmp_path))
    s = result.samples.set_index("station_index")
    for station in OUTSIDE_STATIONS:
        row = s.loc[station]
        assert row["sample_status"] == route_evidence.ROUTE_POINT_OUTSIDE_DOD_EXTENT
        assert pd.isna(row["delta_bed_elevation_m"])
        assert pd.isna(row["observed_change_direction"])
        assert pd.isna(row["raster_row"]) and pd.isna(row["raster_col"])
        assert pd.isna(row["route_point_to_sampled_cell_center_distance_m"])


def test_11_nodata_becomes_unavailable_never_zero(tmp_path: Path):
    result, _ = _run(_fixture(tmp_path))
    row = result.samples.set_index("station_index").loc[NODATA_STATION]
    assert row["sample_status"] == route_evidence.CHANGE_VALUE_NODATA
    assert pd.isna(row["delta_bed_elevation_m"])
    assert pd.isna(row["observed_change_direction"])
    assert row["raster_row"] == 4 and row["raster_col"] == NODATA_STATION  # support is known
    # NaN-nodata rasters (the accepted MAR-021 DoD convention) behave the same way.
    arr = _dod_array()
    arr[4, NODATA_STATION] = np.nan
    path = _fixture(tmp_path, dod_kwargs={"array": arr, "nodata": np.nan})
    result2, _ = _run(path)
    row2 = result2.samples.set_index("station_index").loc[NODATA_STATION]
    assert row2["sample_status"] == route_evidence.CHANGE_VALUE_NODATA
    assert pd.isna(row2["delta_bed_elevation_m"])
    zero = result2.samples.set_index("station_index").loc[ZERO_STATION]
    assert zero["sample_status"] == route_evidence.CHANGE_VALUE_AVAILABLE
    assert zero["delta_bed_elevation_m"] == 0.0  # a genuine zero is still available
    validation = route_evidence.build_route_change_evidence_validation(result2)
    assert validation["question_e_does_nodata_become_zero_change"] == "NO"


# --- Section 27.12-20: sampling -------------------------------------------------------------------


def test_12_positive_known_cell_sampled_exactly(tmp_path: Path):
    s = _run(_fixture(tmp_path))[0].samples.set_index("station_index")
    assert s.loc[0, "delta_bed_elevation_m"] == _f32(0.5)
    assert s.loc[6, "delta_bed_elevation_m"] == _f32(2.25)
    assert s.loc[0, "observed_change_direction"] == dod.OBSERVED_SEABED_RAISING


def test_13_negative_known_cell_sampled_exactly(tmp_path: Path):
    s = _run(_fixture(tmp_path))[0].samples.set_index("station_index")
    assert s.loc[1, "delta_bed_elevation_m"] == _f32(-0.75)
    assert s.loc[5, "delta_bed_elevation_m"] == _f32(-3.2)
    assert s.loc[5, "observed_change_direction"] == dod.OBSERVED_SEABED_LOWERING


def test_14_zero_known_cell_remains_zero_with_no_direction_label(tmp_path: Path):
    s = _run(_fixture(tmp_path))[0].samples.set_index("station_index")
    assert s.loc[ZERO_STATION, "delta_bed_elevation_m"] == 0.0
    assert s.loc[ZERO_STATION, "sample_status"] == route_evidence.CHANGE_VALUE_AVAILABLE
    assert pd.isna(s.loc[ZERO_STATION, "observed_change_direction"])


def test_15_adjacent_cells_with_very_different_values_prove_no_interpolation(tmp_path: Path):
    # Route at y = 6000000.5: still row 4 (row_f = 4.95) but only 0.5 m from the +/-100 row 5.
    coords = [(500005.0, 6000000.5), (500145.0, 6000000.5)]
    s = _run(_fixture(tmp_path, route_coords=coords))[0].samples.set_index("station_index")
    for k in INSIDE_STATIONS:
        if k == NODATA_STATION:
            continue
        assert s.loc[k, "delta_bed_elevation_m"] == _f32(ROW4_VALUES[k])
        assert s.loc[k, "raster_row"] == 4
    # Any bilinear/average blend with row 5 (-100) or row 3 (+100) would be far from these values.
    assert abs(s.loc[0, "delta_bed_elevation_m"]) < 1.0
    # A route in row 5 itself reads exactly -100 -- the neighbouring row never leaks in.
    coords5 = [(500005.0, 5999995.0), (500115.0, 5999995.0)]
    s5 = _run(_fixture(tmp_path, route_coords=coords5))[0].samples
    assert set(s5["raster_row"].dropna().astype(int)) == {5}
    assert set(s5["delta_bed_elevation_m"].dropna()) == {-100.0}


def test_16_raster_row_and_column_recorded_correctly(tmp_path: Path):
    s = _run(_fixture(tmp_path))[0].samples.set_index("station_index")
    for k in INSIDE_STATIONS:
        assert s.loc[k, "raster_row"] == 4
        assert s.loc[k, "raster_col"] == k
    # Cross-check against rasterio's own containing-cell indexing.
    with rasterio.open(tmp_path / "dod.tif") as ds:
        for k in INSIDE_STATIONS:
            assert ds.index(s.loc[k, "route_point_x"], s.loc[k, "route_point_y"]) == (4, k)


def test_17_sampled_cell_center_coordinates_correct(tmp_path: Path):
    s = _run(_fixture(tmp_path))[0].samples.set_index("station_index")
    for k in INSIDE_STATIONS:
        assert s.loc[k, "sampled_cell_center_x"] == 500005.0 + 10.0 * k
        assert s.loc[k, "sampled_cell_center_y"] == 6000005.0


def test_18_point_to_cell_center_distance_recorded(tmp_path: Path):
    s = _run(_fixture(tmp_path))[0].samples.set_index("station_index")
    for k in INSIDE_STATIONS:
        assert s.loc[k, "route_point_to_sampled_cell_center_distance_m"] == pytest.approx(2.0)
    metadata = route_evidence.build_route_change_evidence_metadata(_run(_fixture(tmp_path))[0])
    assert (
        "NOT a survey or positional uncertainty"
        in (metadata["sample_support"]["cell_center_distance_note"])
    )


def test_19_route_point_geometry_remains_canonical_route_point(tmp_path: Path):
    result, model = _run(_fixture(tmp_path))
    grid = model.route_reference.grid_gdf
    assert grid is not None
    assert list(result.samples.geometry.x) == list(grid.geometry.x)
    assert list(result.samples.geometry.y) == list(grid.geometry.y)
    inside = result.samples[
        result.samples["sample_status"] != route_evidence.ROUTE_POINT_OUTSIDE_DOD_EXTENT
    ]
    # Points are NOT snapped to the cell centres (which sit 2 m north of the route).
    assert (inside.geometry.y != inside["sampled_cell_center_y"]).all()
    assert (inside.geometry.y == 6000003.0).all()


def test_20_and_45_deterministic_identical_output(tmp_path: Path):
    path = _fixture(tmp_path)
    final_a, _ = _run_and_write(tmp_path, path, "out_a")
    final_b, _ = _run_and_write(tmp_path, path, "out_b")
    pd.testing.assert_frame_equal(
        pd.DataFrame(final_a.samples.drop(columns="geometry")),
        pd.DataFrame(final_b.samples.drop(columns="geometry")),
    )
    meta_a = json.loads(
        (tmp_path / "out_a" / "route_observed_seabed_change_metadata.json").read_text()
    )
    meta_b = json.loads(
        (tmp_path / "out_b" / "route_observed_seabed_change_metadata.json").read_text()
    )
    for meta in (meta_a, meta_b):
        meta["outputs"] = None  # the only field that legitimately differs: the output directory
    assert meta_a == meta_b
    assert _sha(tmp_path / "out_a" / "route_observed_seabed_change.parquet") == _sha(
        tmp_path / "out_b" / "route_observed_seabed_change.parquet"
    )
    lowered = [k.lower() for k in _keys(meta_a)]
    assert not any("timestamp" in k or "generated_at" in k or "run_time" in k for k in lowered)


# --- Section 27.21-31: scientific semantics -------------------------------------------------------


def test_21_22_raw_labels_reuse_existing_dod_vocabulary(tmp_path: Path):
    s = _run(_fixture(tmp_path))[0].samples.set_index("station_index")
    for k in EXPECTED_RAISING:
        assert s.loc[k, "observed_change_direction"] == dod.OBSERVED_SEABED_RAISING
    for k in EXPECTED_LOWERING:
        assert s.loc[k, "observed_change_direction"] == dod.OBSERVED_SEABED_LOWERING
    source = inspect.getsource(route_evidence)
    assert "classify_change_direction" in source
    assert 'OBSERVED_SEABED_RAISING = "' not in source  # imported, never re-defined
    assert 'OBSERVED_SEABED_LOWERING = "' not in source
    # No re-implemented sign logic: executable comparisons on the sampled value are absent.
    for forbidden in (
        "value > 0",
        "value < 0",
        "value >= 0",
        "value <= 0",
        "> 0)",
        "< 0)",
        "> 0]",
        "< 0]",
    ):
        assert forbidden not in source, forbidden


def test_23_24_25_no_erosion_deposition_or_scour_classification(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path))
    directions = set(final.samples["observed_change_direction"].dropna())
    assert directions == {dod.OBSERVED_SEABED_RAISING, dod.OBSERVED_SEABED_LOWERING}
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    for token in ("EROSION", "DEPOSITION", "SCOUR", "ACCRETION"):
        assert not any(token in c.upper() for c in final.samples.columns)
        assert not any(token in k.upper() for k in _keys(metadata))
        assert not any(
            token in str(v)
            for v in final.samples.drop(columns="geometry").to_numpy().ravel()
            if isinstance(v, str)
        )
    source = inspect.getsource(route_evidence)
    for forbidden in (
        '"EROSION"',
        '"DEPOSITION"',
        '"SCOUR"',
        "EROSION =",
        "DEPOSITION =",
        "SCOUR =",
    ):
        assert forbidden not in source


def test_26_27_no_significance_boolean_and_threshold_stays_null(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path))
    bool_columns = [c for c in final.samples.columns if final.samples[c].dtype == bool]
    assert bool_columns == ["is_terminal"]
    forbidden = ("significant_change", "significant_erosion", "significant_deposition", "exceeds")
    for c in [
        *final.samples.columns,
        *_keys(route_evidence.build_route_change_evidence_metadata(final)),
    ]:
        assert not any(f in c.lower() for f in forbidden), c
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    sig = metadata["generic_change_significance"]
    assert sig["generic_change_significance_threshold_m"] is None
    assert (
        sig["generic_change_significance_status"]
        == uncertainty.GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED
    )
    assert set(final.samples["generic_change_significance_status"]) == {
        uncertainty.GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED
    }


def test_28_29_30_no_magnitude_filter_tiny_values_preserved(tmp_path: Path):
    s = _run(_fixture(tmp_path))[0].samples.set_index("station_index")
    available = s[s["sample_status"] == route_evidence.CHANGE_VALUE_AVAILABLE]
    assert len(available) == 11  # every finite cell kept: 0.001, -0.001, 0.3, -0.3 included
    assert s.loc[4, "delta_bed_elevation_m"] == _f32(0.001)
    assert s.loc[4, "observed_change_direction"] == dod.OBSERVED_SEABED_RAISING
    assert s.loc[7, "delta_bed_elevation_m"] == _f32(-0.001)
    assert s.loc[7, "observed_change_direction"] == dod.OBSERVED_SEABED_LOWERING
    assert s.loc[10, "delta_bed_elevation_m"] == _f32(0.3)
    assert s.loc[11, "delta_bed_elevation_m"] == _f32(-0.3)
    source = inspect.getsource(route_evidence)
    # No numeric magnitude comparison anywhere (0.2 / 0.283 / 0.3 or otherwise).
    assert re.search(r"(abs\(|np\.abs\()[^\n]*[<>]", source) is None
    assert re.search(r"[<>]=?\s*0\.\d", source) is None
    assert "threshold_m=" not in source.replace("generic_change_significance_threshold_m", "")


def test_31_no_contiguous_physical_intervals_produced(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path))
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    for c in [*final.samples.columns, *_keys(metadata)]:
        lowered = c.lower()
        assert "zone" not in lowered and "segment" not in lowered
        assert "start_chainage" not in lowered and "end_chainage" not in lowered
        assert "from_kp" not in lowered and "to_kp" not in lowered
    assert len(final.samples) == final.route_reference.station_count  # one row per POINT
    source = inspect.getsource(route_evidence)
    assert "contiguous" not in source.lower().replace("no contiguous", "")
    assert "run_length" not in source and "groupby" not in source


# --- Section 27.32-36: route semantics ------------------------------------------------------------


def test_32_chainage_and_order_come_from_mar027_grid(tmp_path: Path):
    result, model = _run(_fixture(tmp_path))
    grid = model.route_reference.grid_gdf
    assert list(result.samples["station_index"]) == list(grid["station_index"])
    assert list(result.samples["chainage_m"]) == list(grid["chainage_m"])
    assert list(result.samples["kp_label"]) == list(grid["kp_label"])
    assert list(result.samples["is_terminal"]) == list(grid["is_terminal"])
    source = inspect.getsource(route_evidence)
    assert "compute_chainage_stations" not in source  # consumed via the MAR-027 grid only
    assert "interpolate(" not in source
    assert ".length" not in source


def test_33_route_direction_not_inferred_from_asset_name(tmp_path: Path):
    path = _fixture(
        tmp_path, route_asset_id="loggs_to_anglia_route", primary="loggs_to_anglia_route"
    )
    result, _ = _run(path)
    assert set(result.samples["chainage_origin_basis"]) == {route_adapter.SOURCE_GEOMETRY_ORDER}
    assert result.samples.set_index("station_index").loc[0, "route_point_x"] == 500005.0
    # Reversing the geometry vertex order moves chainage 0 to the other end -- the name is inert.
    reversed_path = _fixture(
        tmp_path,
        route_coords=list(reversed(ROUTE_COORDS)),
        route_asset_id="loggs_to_anglia_route",
        primary="loggs_to_anglia_route",
    )
    reversed_result, _ = _run(reversed_path)
    assert reversed_result.samples.set_index("station_index").loc[0, "route_point_x"] == 500145.0
    assert set(reversed_result.samples["chainage_origin_basis"]) == {
        route_adapter.SOURCE_GEOMETRY_ORDER
    }


def test_34_grid_interval_reported_as_indexing_resolution(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path))
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    assert metadata["route_reference"]["configured_interval_m"] == 10.0
    assert "indexing resolution" in metadata["route_reference"]["interval_semantics"]
    assert "not a survey accuracy" in metadata["route_reference"]["interval_semantics"]
    assert set(final.samples["route_reference_interval_m"]) == {10.0}


def test_35_sample_counts_not_mislabeled_as_route_length_coverage(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path))
    counts = route_evidence.build_route_change_evidence_metadata(final)["sample_counts"]
    assert counts == {
        "route_reference_sample_count": 15,
        "change_value_available_sample_count": 11,
        "raising_sample_count": 5,
        "lowering_sample_count": 5,
        "zero_change_sample_count": 1,
        "nodata_sample_count": 1,
        "outside_dod_extent_sample_count": 3,
        "note": route_evidence.SAMPLE_COUNT_NOTE,
    }
    for key in counts:
        if key != "note":
            assert key.endswith("_sample_count")
            for forbidden in ("coverage", "fraction", "percent", "route_length", "area"):
                assert forbidden not in key
    assert "NOT route-length fractions" in counts["note"]


def test_36_changing_interval_changes_sample_count_not_source_values(tmp_path: Path):
    before = _sha(_fixture(tmp_path).parent / "dod.tif")
    result_10, _ = _run(_fixture(tmp_path, interval_m=10.0))
    result_20, _ = _run(_fixture(tmp_path, interval_m=20.0))
    assert len(result_10.samples) == 15
    assert len(result_20.samples) == 8
    by_chainage_10 = result_10.samples.set_index("chainage_m")["delta_bed_elevation_m"]
    by_chainage_20 = result_20.samples.set_index("chainage_m")["delta_bed_elevation_m"]
    for chainage, value in by_chainage_20.items():
        other = by_chainage_10.loc[chainage]
        assert (pd.isna(value) and pd.isna(other)) or value == other
    assert _sha(tmp_path / "dod.tif") == before
    assert result_10.dod_identity.sha256 == result_20.dod_identity.sha256 == before


# --- Section 27.37-42: independence ---------------------------------------------------------------


def test_37_mar021_dod_function_unchanged():
    e1 = np.array([[1.0, 2.0], [3.0, np.nan]])
    e2 = np.array([[1.5, 1.0], [3.0, 4.0]])
    mask = np.array([[True, True], [True, False]])
    result = dod.compute_delta_bed_elevation(e1, e2, mask)
    assert result.definition == route_evidence.ACCEPTED_DOD_DEFINITION
    assert (
        result.definition
        == "delta_bed_elevation_m = bed_elevation_epoch2_m - bed_elevation_epoch1_m"
    )
    np.testing.assert_array_equal(result.delta_bed_elevation_m[0], [0.5, -1.0])
    assert result.delta_bed_elevation_m[1, 0] == 0.0 and np.isnan(
        result.delta_bed_elevation_m[1, 1]
    )
    labels = dod.classify_change_direction(np.array([0.5, -1.0, 0.0, np.nan]))
    assert list(labels) == [dod.OBSERVED_SEABED_RAISING, dod.OBSERVED_SEABED_LOWERING, None, None]
    source = inspect.getsource(route_evidence)
    assert "compute_delta_bed_elevation" not in source  # never recomputed here
    assert "epoch2 -" not in source and "elevation2 -" not in source


def test_38_mar021a_uncertainty_function_unchanged():
    derivation = uncertainty.derive_change_threshold(
        sigma_epoch1_m=0.2, sigma_epoch2_m=0.2, evidence_citation="source"
    )
    assert derivation.status == uncertainty.GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED
    assert derivation.generic_threshold_m is None
    assert derivation.nominal_accuracy_rss_reference_m == pytest.approx(0.2828427)
    source = inspect.getsource(route_evidence)
    assert "derive_change_threshold" not in source
    assert "propagate_sigma" not in source
    assert "nominal_accuracy_rss" not in source


def test_39_40_41_scour_burial_freespan_modules_not_imported_for_classification(tmp_path: Path):
    for module in (route_evidence, rem):
        source = inspect.getsource(module)
        assert (
            re.search(r"^(from|import) marine_engine\.(scour|burial|freespan)", source, re.M)
            is None
        )
        for value in vars(module).values():
            name = getattr(value, "__name__", "")
            assert not str(name).startswith(("marine_engine.scour", "marine_engine.burial"))
            assert not str(name).startswith("marine_engine.freespan")
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path))
    text = (tmp_path / "out" / "route_observed_seabed_change_metadata.json").read_text("utf-8")
    values = {str(v) for v in final.samples.drop(columns="geometry").to_numpy().ravel()}
    for token in (
        "SCOUR_",
        "EXPOSED",
        "BURIED",
        "FREE_SPAN",
        "OBSERVED_BURIAL_LOSS",
        "SUPPORT_LOSS",
    ):
        assert token not in text
        assert not any(token in v for v in values)


def test_42_project_intrinsic_and_effective_readiness_not_modified(tmp_path: Path):
    path = _fixture(tmp_path)
    m, mdir = project_manifest.load_project_manifest(tmp_path / "project.yaml")
    fresh = project_registry.register_project(m, mdir)
    result, model = _run(path)
    assert result.available
    again = project_registry.register_project(m, mdir)
    for a, b in zip(fresh.asset_results, again.asset_results, strict=True):
        assert a.registration == b.registration
        # No declared survey epoch -> the accepted adapter says READY_WITH_LIMITATIONS (as for
        # PL854); MAR-029 neither upgrades nor downgrades it.
        assert a.registration.readiness_status_intrinsic == route_adapter.READY_WITH_LIMITATIONS
        assert (
            a.registration.readiness_status_effective == a.registration.readiness_status_intrinsic
        )
    linkage = model.asset_linkages[0]
    assert (
        linkage.readiness_status_intrinsic
        == fresh.asset_results[0].registration.readiness_status_intrinsic
    )
    assert (
        linkage.readiness_status_effective
        == fresh.asset_results[0].registration.readiness_status_effective
    )
    source = inspect.getsource(route_evidence)
    assert "readiness_status" not in source  # never read, rewritten, or re-encoded here


# --- Section 27.43-47: outputs --------------------------------------------------------------------


def test_43_parquet_one_row_per_route_reference_point(tmp_path: Path):
    final, model = _run_and_write(tmp_path, _fixture(tmp_path))
    table = pd.read_parquet(tmp_path / "out" / "route_observed_seabed_change.parquet")
    assert list(table.columns) == list(route_evidence.ROUTE_CHANGE_COLUMNS)
    assert len(table) == model.route_reference.station_count == 15
    assert list(table["station_index"]) == list(range(15))
    assert table["raster_row"].dtype == "Int64" and table["raster_col"].dtype == "Int64"
    assert list(table["raster_col"].dropna().astype(int)) == INSIDE_STATIONS


def test_44_geopackage_points_in_canonical_route_crs(tmp_path: Path):
    final, model = _run_and_write(tmp_path, _fixture(tmp_path))
    gpkg = tmp_path / "out" / "route_observed_seabed_change.gpkg"
    assert [layer[0] for layer in list_layers(gpkg)] == [route_evidence.ROUTE_CHANGE_LAYER]
    g = gpd.read_file(gpkg, layer=route_evidence.ROUTE_CHANGE_LAYER)
    assert CRS.from_user_input(g.crs) == CRS.from_user_input(WORKING_CRS)
    assert CRS.from_user_input(g.crs) == CRS.from_user_input(model.route_reference.grid_gdf.crs)
    assert set(g.geom_type) == {"Point"}
    assert len(g) == 15
    assert list(g.geometry.x) == list(model.route_reference.grid_gdf.geometry.x)
    assert list(g.geometry.y) == list(model.route_reference.grid_gdf.geometry.y)
    assert g["chainage_m"].min() == 0.0 and g["chainage_m"].max() == 140.0
    assert list(g["station_index"]) == sorted(g["station_index"])
    assert g["sample_status"].value_counts().to_dict() == {
        route_evidence.CHANGE_VALUE_AVAILABLE: 11,
        route_evidence.ROUTE_POINT_OUTSIDE_DOD_EXTENT: 3,
        route_evidence.CHANGE_VALUE_NODATA: 1,
    }
    valid = g.loc[
        g["sample_status"] == route_evidence.CHANGE_VALUE_AVAILABLE, "delta_bed_elevation_m"
    ]
    assert valid.min() == _f32(-3.2) and valid.max() == _f32(2.25)
    assert list(g["raster_row"].dropna().astype(int).unique()) == [4]
    assert list(g["raster_col"].dropna().astype(int)) == INSIDE_STATIONS
    assert set(g["route_point_to_sampled_cell_center_distance_m"].dropna().round(9)) == {2.0}


def test_46_no_risk_readiness_or_hazard_percentage(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path))
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    lowered = [k.lower() for k in [*_keys(metadata), *final.samples.columns]]
    for forbidden in ("score", "risk", "hazard", "percent", "probability", "readiness", "_index_"):
        assert not any(forbidden in k for k in lowered), forbidden
    assert not any(k.endswith("_index") for k in lowered if k != "station_index")
    assert not any(
        re.fullmatch(r"fraction_of_route.*", k) for k in lowered
    )  # counts only (Section 18)
    html = (tmp_path / "out" / "route_observed_seabed_change_report.html").read_text("utf-8")
    assert "score" not in html.lower()
    assert route_evidence.DIRECTION_LABEL_DISCLAIMER in html.replace("&#x27;", "'")
    assert "NOT a survey or positional uncertainty" in html.replace("&#x27;", "'")
    assert "never 0 m" in html
    assert "2018-10" in html and "2020-11" in html


def test_47_no_stale_geopackage_after_subsequent_not_available_run(tmp_path: Path):
    good = _fixture(tmp_path)
    final, _ = _run_and_write(tmp_path, good)
    gpkg = tmp_path / "out" / "route_observed_seabed_change.gpkg"
    parquet = tmp_path / "out" / "route_observed_seabed_change.parquet"
    assert final.products_written and gpkg.is_file() and parquet.is_file()
    bad = _fixture(
        tmp_path, dod_kwargs={"crs": "EPSG:32632"}, route_change_kwargs={"declared_crs": None}
    )
    final_bad, _ = _run_and_write(tmp_path, bad)
    assert not final_bad.products_written
    assert not gpkg.exists() and not parquet.exists()
    metadata = json.loads(
        (tmp_path / "out" / "route_observed_seabed_change_metadata.json").read_text()
    )
    assert metadata["status"] == route_evidence.ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE
    assert metadata["status_reason_code"] == route_evidence.DOD_CRS_MISMATCH
    assert metadata["outputs"] == {"parquet": None, "gpkg": None, "gis_layer": None}
    assert metadata["sample_counts"]["route_reference_sample_count"] == 0
    validation = json.loads(
        (tmp_path / "out" / "route_observed_seabed_change_validation.json").read_text()
    )
    assert (
        validation["question_g_is_route_referenced_observed_change_available_for_this_run"] == "NO"
    )


# --- Section 25: controlled not-available reason codes --------------------------------------------


def test_not_available_no_primary_route(tmp_path: Path):
    result, _ = _run(_fixture(tmp_path, primary=None))
    assert result.reason_code == route_evidence.NO_PRIMARY_ROUTE
    assert result.samples is None and result.dod_identity is None


def test_not_available_route_reference_not_built(tmp_path: Path):
    result, _ = _run(_fixture(tmp_path, interval_m=None))
    assert result.reason_code == route_evidence.ROUTE_REFERENCE_NOT_BUILT
    assert any("no default grid spacing is invented" in f for f in result.findings)


def test_not_available_dod_not_found_and_not_readable(tmp_path: Path):
    result, _ = _run(_fixture(tmp_path, write_dod=False))
    assert result.reason_code == route_evidence.DOD_NOT_FOUND
    assert result.dod_identity is None
    path = _fixture(tmp_path, write_dod=False)
    (tmp_path / "dod.tif").write_text("not a raster", encoding="utf-8")
    result2, _ = _run(path)
    assert result2.reason_code == route_evidence.DOD_NOT_READABLE
    assert result2.dod_identity is not None  # identity of the unreadable bytes is still recorded


def test_not_available_source_provenance_missing(tmp_path: Path):
    result, _ = _run(_fixture(tmp_path, write_prov=False))
    assert result.reason_code == route_evidence.SOURCE_PROVENANCE_NOT_FOUND


def test_provenance_artifact_identified_by_content(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path))
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    prov = metadata["source_provenance_artifact"]
    assert prov["status"] == route_evidence.PROVENANCE_ARTIFACT_IDENTITY_CAPTURED
    assert prov["identity"]["sha256"] == _sha(tmp_path / "prov.json")
    assert prov["identity"]["byte_size"] == (tmp_path / "prov.json").stat().st_size
    assert final.route_manifest.dod.provenance_path is not None
    # Optional: without a declared provenance artefact nothing is invented.
    result2, _ = _run(_fixture(tmp_path, route_change_kwargs={"provenance_path": None}))
    assert result2.available and result2.provenance_identity is None


def test_not_available_source_role_mismatch_and_missing_crs_and_rotation(tmp_path: Path):
    path = _fixture(tmp_path, dod_kwargs={"tags": {"scientific_role": "SOMETHING_ELSE"}})
    result, _ = _run(path)
    assert result.reason_code == route_evidence.DOD_SOURCE_ROLE_MISMATCH
    # Untagged rasters are not a mismatch (nothing observed to compare against) -- but MAR-029A
    # records that both facts rest on the manifest declaration only.
    result_untagged, _ = _run(_fixture(tmp_path, dod_kwargs={"tags": {}}))
    assert result_untagged.available
    assert result_untagged.definition_evidence_basis == route_evidence.MANIFEST_DECLARED_ONLY
    assert result_untagged.source_role_evidence_basis == route_evidence.MANIFEST_DECLARED_ONLY
    result_nocrs, _ = _run(
        _fixture(tmp_path, dod_kwargs={"crs": None}, route_change_kwargs={"declared_crs": None})
    )
    assert result_nocrs.reason_code == route_evidence.DOD_CRS_MISSING
    result_rot, _ = _run(_fixture(tmp_path, dod_kwargs={"rotated": True}))
    assert result_rot.reason_code == route_evidence.DOD_ROTATED_GRID_UNSUPPORTED


def test_not_available_when_no_route_point_on_valid_support(tmp_path: Path):
    far = [(600000.0, 6100000.0), (600140.0, 6100000.0)]
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path, route_coords=far))
    assert final.reason_code == route_evidence.NO_ROUTE_POINTS_ON_VALID_DOD_SUPPORT
    assert not final.products_written
    assert not (tmp_path / "out" / "route_observed_seabed_change.gpkg").exists()
    # Fully-nodata support behaves the same way (never written as 0 m change).
    arr = _dod_array(row4=[NODATA] * 12)
    final2, _ = _run_and_write(tmp_path, _fixture(tmp_path, dod_kwargs={"array": arr}))
    assert final2.reason_code == route_evidence.NO_ROUTE_POINTS_ON_VALID_DOD_SUPPORT


def test_all_zero_samples_are_available_with_no_direction_labels(tmp_path: Path):
    arr = _dod_array(row4=[0.0] * 12)
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path, dod_kwargs={"array": arr}))
    assert final.available
    available = final.samples[
        final.samples["sample_status"] == route_evidence.CHANGE_VALUE_AVAILABLE
    ]
    assert len(available) == 12 and set(available["delta_bed_elevation_m"]) == {0.0}
    assert available["observed_change_direction"].isna().all()
    g = gpd.read_file(
        tmp_path / "out" / "route_observed_seabed_change.gpkg",
        layer=route_evidence.ROUTE_CHANGE_LAYER,
    )
    assert g["observed_change_direction"].isna().all()
    counts = route_evidence.build_route_change_evidence_metadata(final)["sample_counts"]
    assert counts["zero_change_sample_count"] == 12
    assert counts["raising_sample_count"] == counts["lowering_sample_count"] == 0


def test_paths_resolve_relative_to_route_change_manifest(tmp_path: Path, monkeypatch):
    nested = tmp_path / "analysis" / "change"
    nested.mkdir(parents=True)
    _write_route_gpkg(tmp_path / "route_a.gpkg")
    _write_dod_tif(tmp_path / "dod.tif")
    (tmp_path / "project.yaml").write_text(_project_yaml(), encoding="utf-8")
    path = nested / "rc.yaml"
    path.write_text(
        _route_change_yaml(
            project_manifest_path="../../project.yaml",
            dod_path="../../dod.tif",
            provenance_path=None,
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result, _ = _run(path)
    assert result.available, result.findings
    assert result.dod_resolved_path == (tmp_path / "dod.tif").resolve()


def test_output_scientific_role_and_metadata_facts(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path))
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    assert metadata["scientific_role"] == "ROUTE_REFERENCED_OBSERVED_SEABED_ELEVATION_CHANGE"
    assert metadata["status"] == route_evidence.ROUTE_CHANGE_EVIDENCE_AVAILABLE
    assert metadata["dod_definition"]["effective"] == route_evidence.ACCEPTED_DOD_DEFINITION
    assert metadata["dod_definition"]["declared"] == CANONICAL_DEFINITION
    assert metadata["dod_definition"]["observed_embedded"] == CANONICAL_DEFINITION
    assert metadata["dod_definition"]["evidence_basis"] == (
        route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE
    )
    assert metadata["dod_definition"]["zero_label"] is None
    obs = metadata["dod_observed_facts"]
    assert (obs["width"], obs["height"], obs["band_count"], obs["dtype"]) == (12, 10, 1, "float32")
    assert (obs["pixel_size_x"], obs["pixel_size_y"]) == (10.0, 10.0)
    assert obs["nodata"] == NODATA
    assert obs["bounds"] == [500000.0, 5999950.0, 500120.0, 6000050.0]
    assert obs["tags"]["scientific_role"] == "MULTI_EPOCH_SEABED_CHANGE_POC"
    assert metadata["dod_source_identity"]["sha256"] == _sha(tmp_path / "dod.tif")
    assert metadata["dod_source_identity"]["byte_size"] == (tmp_path / "dod.tif").stat().st_size
    assert metadata["sample_support"]["semantics"] == route_evidence.SAMPLE_SUPPORT_SEMANTICS
    assert metadata["route_reference"]["route_asset_id"] == "route_a"
    assert metadata["route_reference"]["station_count"] == 15
    assert metadata["temporal"]["epoch1_survey_epoch_declared"] == "2018-10"
    assert metadata["temporal"]["epoch2_survey_epoch_declared"] == "2020-11"
    assert (
        "not present-day, persistent, long-term, or future change" in metadata["temporal"]["note"]
    )
    assert metadata["outputs"]["gis_layer"] == route_evidence.ROUTE_CHANGE_LAYER
    assert (
        "NaN" not in (tmp_path / "out" / "route_observed_seabed_change_metadata.json").read_text()
    )


def test_accepted_scientific_modules_untouched_by_mar029():
    # The bridge reuses -- never redefines -- the accepted MAR-021 and MAR-027 modules.
    assert route_evidence.classify_change_direction is dod.classify_change_direction
    assert (
        dod.DoDResult.__dataclass_fields__["definition"].default
        == route_evidence.ACCEPTED_DOD_DEFINITION
    )
    source = inspect.getsource(route_evidence)
    assert "route_reference.build_project_route_reference" not in source  # consumed via model
    assert "def build_route_reference_grid" not in source
    assert "def classify_grid_alignment" not in source
    assert "def align_to_common_grid" not in source
    assert route_reference.ROUTE_REFERENCE_GRID_DISCLAIMER in source or (
        "ROUTE_REFERENCE_GRID_DISCLAIMER" in source
    )


# --- CLI wiring -----------------------------------------------------------------------------------


def test_cli_parser_registers_build_route_seabed_change_evidence():
    parser = cli.build_parser()
    args = parser.parse_args(["build-route-seabed-change-evidence", "x.yaml"])
    assert args.func is cli._cmd_build_route_seabed_change_evidence
    assert args.manifest == Path("x.yaml")
    # Additive: the accepted MAR-021 and MAR-027 commands remain registered and unchanged.
    assert parser.parse_args(["build-project-model", "m.yaml"]).func is cli._cmd_build_project_model
    assert (
        parser.parse_args(["build-seabed-change-poc", "c.yaml"]).func
        is cli._cmd_build_seabed_change_poc
    )


def test_cli_reports_invalid_inputs_cleanly(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("route_asset_id: route_a\n", encoding="utf-8")
    assert cli.main(["build-route-seabed-change-evidence", str(bad)]) == 1
    missing_project = tmp_path / "rc.yaml"
    missing_project.write_text(
        _route_change_yaml(project_manifest_path="./nope.yaml", provenance_path=None),
        encoding="utf-8",
    )
    assert cli.main(["build-route-seabed-change-evidence", str(missing_project)]) == 1


def test_cli_writes_outputs_under_project_change_route_dir(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = _fixture(tmp_path)
    assert cli.main(["build-route-seabed-change-evidence", str(path)]) == 0
    out = tmp_path / "data" / "processed" / "test_project" / "change_route"
    for name in (
        "route_observed_seabed_change.parquet",
        "route_observed_seabed_change.gpkg",
        "route_observed_seabed_change_metadata.json",
        "route_observed_seabed_change_validation.json",
        "route_observed_seabed_change_report.html",
    ):
        assert (out / name).is_file(), name
    validation = json.loads((out / "route_observed_seabed_change_validation.json").read_text())
    assert validation == {
        "question_a_does_mar029_reuse_the_accepted_mar021_dod_definition": "YES",
        "question_b_can_a_positive_dod_sample_be_called_definitive_deposition": "NO",
        "question_c_can_a_negative_dod_sample_be_called_definitive_erosion_or_scour": "NO",
        "question_d_does_mar029_apply_a_generic_change_significance_threshold": "NO",
        "question_e_does_nodata_become_zero_change": "NO",
        "question_f_is_the_dod_raster_interpolated_or_smoothed_during_route_sampling": "NO",
        "question_g_is_route_referenced_observed_change_available_for_this_run": "YES",
        "question_h_was_the_dod_source_modified_during_the_run": "NO",
        "question_i_dod_definition_evidence_basis": route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE,
    }
    stdout = capsys.readouterr().out
    assert "DoD DEFINITION EVIDENCE BASIS: MANIFEST_AND_EMBEDDED_TAG_AGREE" in stdout
    assert "ROUTE_CHANGE_EVIDENCE_AVAILABLE" in stdout
    assert "route_reference_sample_count: 15" in stdout
    assert "DOES NODATA BECOME ZERO CHANGE? NO" in stdout
    assert route_evidence.DIRECTION_LABEL_DISCLAIMER in stdout
    # A later not-available run through the CLI removes the stale point products (Section 25).
    bad = _fixture(
        tmp_path, dod_kwargs={"crs": "EPSG:32632"}, route_change_kwargs={"declared_crs": None}
    )
    assert cli.main(["build-route-seabed-change-evidence", str(bad)]) == 0
    assert not (out / "route_observed_seabed_change.gpkg").exists()
    assert not (out / "route_observed_seabed_change.parquet").exists()
    assert (out / "route_observed_seabed_change_metadata.json").is_file()


def test_project_model_command_output_unchanged_by_mar029(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fixture(tmp_path)
    assert cli.main(["build-project-model", str(tmp_path / "project.yaml")]) == 0
    model = json.loads(
        (
            tmp_path
            / "data"
            / "processed"
            / "test_project"
            / "project"
            / "canonical_project_model.json"
        ).read_text("utf-8")
    )
    assert model["scientific_role"] == project_model.SCIENTIFIC_ROLE
    assert model["route_reference"]["status"] == route_reference.ROUTE_REFERENCE_BUILT
    assert model["route_reference"]["station_count"] == 15
    text = json.dumps(model)
    assert "delta_bed_elevation" not in text and "ROUTE_REFERENCED_OBSERVED" not in text


# --- MAR-029A Section 15: DoD semantic provenance integrity --------------------------------------


def _semantic_fixture(tmp_path: Path, *, tags: dict[str, str] | None, **route_change_kwargs):
    return _fixture(
        tmp_path,
        dod_kwargs={"tags": tags} if tags is not None else None,
        route_change_kwargs={"provenance_path": None, **route_change_kwargs},
    )


def test_029a_01_missing_dod_definition_declared_fails_validation(tmp_path: Path):
    path = _fixture(tmp_path, route_change_kwargs={"declared_definition": None})
    with pytest.raises(pydantic.ValidationError):
        rem.load_route_change_evidence_manifest(path)
    with pytest.raises(pydantic.ValidationError):
        rem.DoDSourceDeclaration.model_validate(
            {
                "path": "d.tif",
                "source_change_study_id": "s",
                "dod_definition_declared": "   ",
                "source_scientific_role_declared": "X",
            }
        )  # blank is not a declaration


def test_029a_02_missing_source_scientific_role_declared_fails_validation(tmp_path: Path):
    path = _fixture(tmp_path, route_change_kwargs={"declared_role": None})
    with pytest.raises(pydantic.ValidationError):
        rem.load_route_change_evidence_manifest(path)
    with pytest.raises(pydantic.ValidationError):
        rem.DoDSourceDeclaration.model_validate(
            {
                "path": "d.tif",
                "source_change_study_id": "s",
                "dod_definition_declared": CANONICAL_DEFINITION,
                "source_scientific_role_declared": "",
            }
        )


def test_029a_03_canonical_declared_definition_accepted(tmp_path: Path):
    result, _ = _run(_semantic_fixture(tmp_path, tags=None))
    assert result.available
    assert result.effective_dod_definition == route_evidence.ACCEPTED_DOD_DEFINITION
    # Whitespace-only differences in the declaration are still the same definition.
    spaced = "delta_bed_elevation_m  =  bed_elevation_epoch2_m  -  bed_elevation_epoch1_m"
    result2, _ = _run(_semantic_fixture(tmp_path, tags=None, declared_definition=spaced))
    assert result2.available and result2.definition_evidence_basis == (
        route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE
    )


def test_029a_04_reversed_definition_is_controlled_not_available(tmp_path: Path):
    final, _ = _run_and_write(
        tmp_path, _semantic_fixture(tmp_path, tags={}, declared_definition=REVERSED_DEFINITION)
    )
    assert final.status == route_evidence.ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE
    assert final.reason_code == route_evidence.DOD_DECLARED_DEFINITION_UNSUPPORTED
    assert final.samples is None and not final.products_written
    assert final.effective_dod_definition is None
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    assert metadata["dod_definition"]["declared"] == REVERSED_DEFINITION
    assert metadata["dod_definition"]["effective"] is None
    assert metadata["dod_definition"]["evidence_basis"] is None
    assert "not flipped" in final.findings[-1]


@pytest.mark.parametrize(
    "declared",
    [
        "delta_bed_elevation_m = bed_elevation_epoch2_m + bed_elevation_epoch1_m",
        "depth_epoch2_m - depth_epoch1_m",
        "difference of the two surveys",
        "bed_elevation_epoch2_m - bed_elevation_epoch1_m",  # missing the named result variable
    ],
)
def test_029a_05_arbitrary_unsupported_definition_is_controlled_not_available(
    tmp_path: Path, declared: str
):
    result, _ = _run(_semantic_fixture(tmp_path, tags={}, declared_definition=declared))
    assert result.reason_code == route_evidence.DOD_DECLARED_DEFINITION_UNSUPPORTED
    assert result.samples is None


def test_029a_06_embedded_and_declared_canonical_definition_agree(tmp_path: Path):
    result, _ = _run(_semantic_fixture(tmp_path, tags=None))  # default tags carry the definition
    assert result.available
    assert result.definition_evidence_basis == route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE
    assert not any("declaration only" in f for f in result.findings)


def test_029a_07_embedded_definition_mismatch_blocks(tmp_path: Path):
    tags = {"scientific_role": "MULTI_EPOCH_SEABED_CHANGE_POC", "definition": REVERSED_DEFINITION}
    final, _ = _run_and_write(tmp_path, _semantic_fixture(tmp_path, tags=tags))
    assert final.reason_code == route_evidence.DOD_DEFINITION_MISMATCH
    assert final.samples is None and not final.products_written
    assert not (tmp_path / "out" / "route_observed_seabed_change.gpkg").exists()
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    assert metadata["dod_definition"]["declared"] == CANONICAL_DEFINITION
    assert metadata["dod_definition"]["observed_embedded"] == REVERSED_DEFINITION
    assert metadata["dod_definition"]["effective"] is None
    assert "does not choose a side" in final.findings[-1]


def test_029a_08_absent_embedded_definition_with_declaration_is_declared_only(tmp_path: Path):
    tags = {"scientific_role": "MULTI_EPOCH_SEABED_CHANGE_POC"}  # no definition tag
    final, _ = _run_and_write(tmp_path, _semantic_fixture(tmp_path, tags=tags))
    assert final.available
    assert final.definition_evidence_basis == route_evidence.MANIFEST_DECLARED_ONLY
    assert final.source_role_evidence_basis == route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    assert metadata["dod_definition"]["observed_embedded"] is None
    assert metadata["dod_definition"]["evidence_basis"] == route_evidence.MANIFEST_DECLARED_ONLY
    assert route_evidence.DEFINITION_DECLARED_ONLY_LIMITATION in metadata["explicit_limitations"]
    assert route_evidence.DEFINITION_DECLARED_ONLY_LIMITATION in final.findings
    validation = route_evidence.build_route_change_evidence_validation(final)
    assert validation["question_i_dod_definition_evidence_basis"] == (
        route_evidence.MANIFEST_DECLARED_ONLY
    )
    html = (tmp_path / "out" / "route_observed_seabed_change_report.html").read_text("utf-8")
    assert "DoD definition (observed embedded tag): absent" in html
    assert "MANIFEST_DECLARED_ONLY" in html


def test_029a_09_observed_role_matches_declaration(tmp_path: Path):
    result, _ = _run(_semantic_fixture(tmp_path, tags=None))
    assert result.source_role_evidence_basis == route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE


def test_029a_10_observed_role_differs_blocks(tmp_path: Path):
    tags = {"scientific_role": "SOMETHING_ELSE", "definition": CANONICAL_DEFINITION}
    result, _ = _run(_semantic_fixture(tmp_path, tags=tags))
    assert result.reason_code == route_evidence.DOD_SOURCE_ROLE_MISMATCH
    assert result.samples is None


def test_029a_11_observed_role_absent_with_declaration_is_declared_only(tmp_path: Path):
    tags = {"definition": CANONICAL_DEFINITION}  # no scientific_role tag
    final, _ = _run_and_write(tmp_path, _semantic_fixture(tmp_path, tags=tags))
    assert final.available
    assert final.source_role_evidence_basis == route_evidence.MANIFEST_DECLARED_ONLY
    assert final.definition_evidence_basis == route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    assert metadata["source_scientific_role"]["observed_embedded"] is None
    assert metadata["source_scientific_role"]["declared"] == "MULTI_EPOCH_SEABED_CHANGE_POC"
    assert route_evidence.ROLE_DECLARED_ONLY_LIMITATION in metadata["explicit_limitations"]


def test_029a_12_untagged_float_raster_without_explicit_definition_cannot_become_evidence(
    tmp_path: Path,
):
    # The pre-repair defect: an arbitrary untagged float raster with no declared definition. It is
    # now rejected at manifest validation -- it never reaches sampling.
    path = _semantic_fixture(tmp_path, tags={}, declared_definition=None)
    with pytest.raises(pydantic.ValidationError):
        _run(path)
    assert cli.main(["build-route-seabed-change-evidence", str(path)]) == 1
    # And with a NON-canonical declaration it is a controlled not-available, never AVAILABLE.
    result, _ = _run(_semantic_fixture(tmp_path, tags={}, declared_definition="my_diff"))
    assert result.status == route_evidence.ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE
    assert result.effective_dod_definition is None


def test_029a_13_metadata_separates_declared_observed_effective_definition(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _semantic_fixture(tmp_path, tags=None))
    block = route_evidence.build_route_change_evidence_metadata(final)["dod_definition"]
    assert set(block) >= {"declared", "observed_embedded", "effective", "evidence_basis"}
    assert block["declared"] == CANONICAL_DEFINITION
    assert block["observed_embedded"] == CANONICAL_DEFINITION
    assert block["effective"] == route_evidence.ACCEPTED_DOD_DEFINITION
    assert block["evidence_basis"] == route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE
    assert "definition" not in block  # the old unconditional field is gone


def test_029a_14_metadata_separates_declared_and_observed_role(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _semantic_fixture(tmp_path, tags=None))
    block = route_evidence.build_route_change_evidence_metadata(final)["source_scientific_role"]
    assert block == {
        "declared": "MULTI_EPOCH_SEABED_CHANGE_POC",
        "observed_embedded": "MULTI_EPOCH_SEABED_CHANGE_POC",
        "evidence_basis": route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE,
    }


def test_029a_15_provenance_hash_is_not_a_scientific_verification_claim(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _fixture(tmp_path))
    metadata = route_evidence.build_route_change_evidence_metadata(final)
    prov = metadata["source_provenance_artifact"]
    assert prov["status"] == route_evidence.PROVENANCE_ARTIFACT_IDENTITY_CAPTURED
    assert "NOT a claim that provenance was verified" in prov["note"]
    text = json.dumps(metadata).lower()
    html = (tmp_path / "out" / "route_observed_seabed_change_report.html").read_text("utf-8")
    for forbidden in ("provenance verified", "provenance validated", "scientifically verified"):
        assert forbidden not in text
        assert forbidden not in html.lower()


def test_029a_16_no_provenance_artifact_is_explicit_not_declared(tmp_path: Path):
    final, _ = _run_and_write(tmp_path, _semantic_fixture(tmp_path, tags=None))
    prov = route_evidence.build_route_change_evidence_metadata(final)["source_provenance_artifact"]
    assert prov["status"] == route_evidence.PROVENANCE_ARTIFACT_NOT_DECLARED
    assert prov["identity"] is None
    assert final.provenance_identity is None


def test_029a_17_18_19_sampling_values_unchanged_by_repair(tmp_path: Path):
    s = _run(_fixture(tmp_path))[0].samples.set_index("station_index")
    for k in INSIDE_STATIONS:
        if k == NODATA_STATION:
            assert pd.isna(s.loc[k, "delta_bed_elevation_m"])  # nodata never zero
            assert s.loc[k, "sample_status"] == route_evidence.CHANGE_VALUE_NODATA
        else:
            assert s.loc[k, "delta_bed_elevation_m"] == _f32(ROW4_VALUES[k])
    assert s.loc[4, "delta_bed_elevation_m"] == _f32(0.001)
    assert s.loc[7, "delta_bed_elevation_m"] == _f32(-0.001)
    assert s.loc[4, "observed_change_direction"] == dod.OBSERVED_SEABED_RAISING
    assert s.loc[7, "observed_change_direction"] == dod.OBSERVED_SEABED_LOWERING
    for k in OUTSIDE_STATIONS:
        assert s.loc[k, "sample_status"] == route_evidence.ROUTE_POINT_OUTSIDE_DOD_EXTENT
        assert pd.isna(s.loc[k, "delta_bed_elevation_m"])
    assert len(s) == 15


def test_029a_20_21_no_threshold_and_no_sign_conversion_introduced():
    source = inspect.getsource(route_evidence)
    assert re.search(r"[<>]=?\s*0\.\d", source) is None
    assert "threshold_m=" not in source.replace("generic_change_significance_threshold_m", "")
    # No sign conversion: the sampled value is never negated, multiplied, or reordered.
    for forbidden in ("-value", "value * -1", "-1 *", "* -1", "np.negative", "epoch1_m - bed"):
        assert forbidden not in source, forbidden
    assert "_represents_canonical_definition" in source
    assert route_evidence.ACCEPTED_DOD_DEFINITION == CANONICAL_DEFINITION
    assert route_evidence.ACCEPTED_DOD_DEFINITION.count(" = ") == 1  # one literal, from change.dod


def test_029a_semantic_helper_on_facts_without_a_route(tmp_path: Path):
    # The same gate logic is callable on a bare DoD source -- how the real Sheringham product is
    # checked without fabricating a route.
    _write_dod_tif(tmp_path / "dod.tif")
    _identity, observed = route_evidence.inspect_dod_source(tmp_path / "dod.tif")
    decl = rem.DoDSourceDeclaration(
        path=Path("dod.tif"),
        source_change_study_id="s",
        dod_definition_declared=CANONICAL_DEFINITION,
        source_scientific_role_declared="MULTI_EPOCH_SEABED_CHANGE_POC",
    )
    sem = route_evidence.assess_dod_semantic_provenance(decl, observed)
    assert sem.passed and sem.effective_definition == route_evidence.ACCEPTED_DOD_DEFINITION
    assert sem.definition_evidence_basis == route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE
    assert sem.source_role_evidence_basis == route_evidence.MANIFEST_AND_EMBEDDED_TAG_AGREE
    bad = decl.model_copy(update={"dod_definition_declared": REVERSED_DEFINITION})
    sem_bad = route_evidence.assess_dod_semantic_provenance(bad, observed)
    assert not sem_bad.passed and sem_bad.effective_definition is None
    assert sem_bad.reason_code == route_evidence.DOD_DECLARED_DEFINITION_UNSUPPORTED

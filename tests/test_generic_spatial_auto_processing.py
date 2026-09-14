"""MAR-034 Sections 42-50: adversarial tests for the generic terrain/bedform/change
auto-processing capability family (`canonicalize_bathymetry`, `terrain_derivatives`,
`bedform_morphodynamics`, `observed_multi_epoch_seabed_change`).

Synthetic rasters/vectors below are built directly via `rasterio`/`geopandas`, in the same idiom
`test_intake_orchestration.py` already uses for the CPT/GeoTIFF/GeoPackage cases -- never
hand-faked bytes for a positive case, and no committed binary fixtures.
"""

from __future__ import annotations

import inspect

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from marine_engine.intake import recognition, registry
from marine_engine.intake.declaration import AssetDeclaration
from marine_engine.orchestration import bootstrap as orch_bootstrap
from marine_engine.orchestration import context as orch_context
from marine_engine.orchestration import runtime as orch_runtime
from marine_engine.orchestration.adapters import bedforms as bedforms_adapter
from marine_engine.orchestration.adapters import change as change_adapter
from marine_engine.orchestration.adapters import terrain_canonical
from marine_engine.orchestration.capability import (
    BEDFORM_MORPHODYNAMICS,
    CANONICALIZE_BATHYMETRY,
    OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
)
from marine_engine.orchestration.declaration import DeclarationRequest
from marine_engine.orchestration.planner import (
    AVAILABLE,
    BLOCKED_AMBIGUOUS_SEMANTICS,
    BLOCKED_MISSING_INPUT,
    NOT_APPLICABLE,
)
from marine_engine.project.categories import BATHYMETRY_RASTER

orch_bootstrap.register_builtin_runtimes()

_CRS = "EPSG:32631"


@pytest.fixture(autouse=True)
def _isolate_default_output_directory(tmp_path, monkeypatch):
    """Each of these capabilities' executors defaults its output directory to a path relative to
    the process CWD (`Path("data/processed") / asset_id / ...`, mirroring the existing
    liquefaction capability's own established `out_dir` default) -- isolate the CWD per test so
    executing a capability here never writes into the real repository's `data/processed/`."""

    monkeypatch.chdir(tmp_path)


def _write_raster(path, *, data, transform, nodata=-9999.0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=_CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def _flat_bathymetry(path, *, size=40, depth=-30.0, cell_size=2.0, seed=0):
    rng = np.random.default_rng(seed)
    data = depth + rng.normal(scale=1.0, size=(size, size))
    transform = from_origin(500000, 5900000, cell_size, cell_size)
    _write_raster(path, data=data, transform=transform)
    return transform


def _recognize_asset(path, declaration=None, asset_id=None):
    fingerprint, decision = registry.inspect_and_recognize(path, asset_declaration=declaration)
    return orch_context.RecognizedAsset(
        path=path,
        fingerprint=fingerprint,
        recognition=decision,
        asset_id=asset_id or (declaration.asset_id if declaration else path.stem),
        declaration=declaration,
    )


# --- Section 42: negative generic GeoTIFF test -----------------------------------------------


def test_negative_arbitrary_geotiff_with_no_declaration_stays_unclassified(tmp_path):
    """A structurally valid, undeclared GeoTIFF must never auto-become bathymetry."""

    path = tmp_path / "arbitrary.tif"
    _flat_bathymetry(path)
    fingerprint, decision = registry.inspect_and_recognize(path)
    assert fingerprint.container_type == "GEOTIFF"
    assert decision.state != recognition.RECOGNIZED
    assert decision.recognized_role is None

    asset = orch_context.RecognizedAsset(path=path, fingerprint=fingerprint, recognition=decision)
    context = orch_context.PlanningContext(recognized_assets=(asset,))
    plans = {p.capability_id: p for p in orch_runtime.plan_all(context)}
    assert plans[CANONICALIZE_BATHYMETRY].status == NOT_APPLICABLE


# --- Section 43: minimal-declaration test ------------------------------------------------------


def test_minimal_declaration_unlocks_recognition_and_canonicalization(tmp_path):
    """The SAME structural GeoTIFF, given only the missing scientific facts, becomes recognized
    bathymetry and unlocks canonicalization -- proving the engine asks only for genuinely missing
    information."""

    path = tmp_path / "declared.tif"
    _flat_bathymetry(path)
    declaration = AssetDeclaration(
        asset_id="declared",
        semantic_role=BATHYMETRY_RASTER,
        source_sign_convention="ALREADY_ELEVATION_STYLE",
        vertical_datum="LAT",
        survey_epoch="2020",
    )
    asset = _recognize_asset(path, declaration)
    assert asset.recognition.state == recognition.RECOGNIZED
    assert asset.recognition.recognized_role == BATHYMETRY_RASTER

    context = orch_context.PlanningContext(recognized_assets=(asset,))
    plans = {p.capability_id: p for p in orch_runtime.plan_all(context)}
    assert plans[CANONICALIZE_BATHYMETRY].status == AVAILABLE

    _final_context, outcomes = orch_runtime.execute_plan(context)
    outcomes_by_id = {o.capability_id: o for o in outcomes}
    assert len(outcomes_by_id[CANONICALIZE_BATHYMETRY].manifests) == 1


# --- Section 44: sign-convention negative test -------------------------------------------------


def test_missing_sign_convention_blocks_canonicalization_never_guessed(tmp_path):
    path = tmp_path / "no_sign_convention.tif"
    _flat_bathymetry(path)
    declared_without_sign = AssetDeclaration(
        asset_id="no_sign_convention",
        semantic_role=BATHYMETRY_RASTER,
        vertical_datum="LAT",
    )
    asset = _recognize_asset(path, declared_without_sign)
    assert asset.recognition.state == recognition.RECOGNIZED  # role alone is enough to recognize

    context = orch_context.PlanningContext(recognized_assets=(asset,))
    plan = next(
        p for p in orch_runtime.plan_all(context) if p.capability_id == CANONICALIZE_BATHYMETRY
    )
    assert plan.status == BLOCKED_MISSING_INPUT
    assert any(terrain_canonical.SOURCE_SIGN_CONVENTION_REQUIRED in r for r in plan.reasons)

    _final_context, outcomes = orch_runtime.execute_plan(context)
    assert next(o for o in outcomes if o.capability_id == CANONICALIZE_BATHYMETRY).manifests == ()

    # Declaring it explicitly unblocks canonicalization -- never inferred from raster values.
    declared_with_sign = AssetDeclaration(
        asset_id="no_sign_convention",
        semantic_role=BATHYMETRY_RASTER,
        source_sign_convention="ALREADY_ELEVATION_STYLE",
        vertical_datum="LAT",
    )
    asset2 = _recognize_asset(path, declared_with_sign)
    context2 = orch_context.PlanningContext(recognized_assets=(asset2,))
    plan2 = next(
        p for p in orch_runtime.plan_all(context2) if p.capability_id == CANONICALIZE_BATHYMETRY
    )
    assert plan2.status == AVAILABLE


# --- Section 45: change negative/positive tests -------------------------------------------------


def _canonicalized_epoch_context(tmp_path, *, epochs: list[tuple[str, str | None, str | None]]):
    """`epochs`: list of (asset_id, vertical_datum, survey_epoch). Returns a PlanningContext with
    each epoch's raw bathymetry asset recognized and declared."""

    assets = []
    for asset_id, datum, epoch in epochs:
        path = tmp_path / f"{asset_id}.tif"
        _flat_bathymetry(path, seed=hash(asset_id) % 1000)
        declaration = AssetDeclaration(
            asset_id=asset_id,
            semantic_role=BATHYMETRY_RASTER,
            source_sign_convention="ALREADY_ELEVATION_STYLE",
            vertical_datum=datum,
            survey_epoch=epoch,
        )
        assets.append(_recognize_asset(path, declaration))
    return orch_context.PlanningContext(recognized_assets=tuple(assets))


def test_change_blocked_with_only_one_epoch(tmp_path):
    context = _canonicalized_epoch_context(tmp_path, epochs=[("only_epoch", "LAT", "2020")])
    _final, outcomes = orch_runtime.execute_plan(context)
    outcome = next(o for o in outcomes if o.capability_id == OBSERVED_MULTI_EPOCH_SEABED_CHANGE)
    assert outcome.plan.status == NOT_APPLICABLE
    assert outcome.manifests == ()


def test_change_blocked_with_unknown_epoch_dates(tmp_path):
    context = _canonicalized_epoch_context(
        tmp_path, epochs=[("epoch_a", "LAT", None), ("epoch_b", "LAT", None)]
    )
    _final, outcomes = orch_runtime.execute_plan(context)
    outcome = next(o for o in outcomes if o.capability_id == OBSERVED_MULTI_EPOCH_SEABED_CHANGE)
    assert outcome.plan.status == BLOCKED_MISSING_INPUT
    assert any(change_adapter.SURVEY_EPOCH_ORDER_NOT_ESTABLISHED in r for r in outcome.plan.reasons)
    assert outcome.manifests == ()


def test_change_blocked_with_incompatible_vertical_datum(tmp_path):
    context = _canonicalized_epoch_context(
        tmp_path, epochs=[("epoch_2018", "LAT", "2018"), ("epoch_2020", "MSL", "2020")]
    )
    _final, outcomes = orch_runtime.execute_plan(context)
    outcome = next(o for o in outcomes if o.capability_id == OBSERVED_MULTI_EPOCH_SEABED_CHANGE)
    assert outcome.plan.status == BLOCKED_MISSING_INPUT
    assert any("VERTICAL_DATUM_NOT_HARMONIZED" in r for r in outcome.plan.reasons)
    assert outcome.manifests == ()


def test_change_available_with_two_compatible_epochs(tmp_path):
    context = _canonicalized_epoch_context(
        tmp_path, epochs=[("epoch_2018", "LAT", "2018"), ("epoch_2020", "LAT", "2020")]
    )
    _final, outcomes = orch_runtime.execute_plan(context)
    outcome = next(o for o in outcomes if o.capability_id == OBSERVED_MULTI_EPOCH_SEABED_CHANGE)
    assert outcome.plan.status == AVAILABLE
    assert len(outcome.manifests) == 1
    manifest = outcome.manifests[0]
    assert "epoch_2018" in manifest.source_asset_ids
    assert "epoch_2020" in manifest.source_asset_ids
    assert any(reason.startswith("earlier_epoch='2018'") for reason in manifest.limitations)
    assert any(reason.startswith("later_epoch='2020'") for reason in manifest.limitations)


def test_change_never_pairs_same_declared_year_ambiguously(tmp_path):
    """Two epochs sharing the SAME declared survey_epoch year must never be silently ordered by
    filename or any other tie-break -- order between them is genuinely not established."""

    context = _canonicalized_epoch_context(
        tmp_path, epochs=[("epoch_x", "LAT", "2020"), ("epoch_y", "LAT", "2020")]
    )
    _final, outcomes = orch_runtime.execute_plan(context)
    outcome = next(o for o in outcomes if o.capability_id == OBSERVED_MULTI_EPOCH_SEABED_CHANGE)
    assert outcome.plan.status == BLOCKED_MISSING_INPUT
    assert any(change_adapter.SURVEY_EPOCH_ORDER_NOT_ESTABLISHED in r for r in outcome.plan.reasons)
    assert outcome.manifests == ()


def test_change_ambiguous_when_one_of_two_distinct_years_has_two_candidates(tmp_path):
    """Three epochs, two distinct declared years but one year shared by two assets: a genuine
    pairing ambiguity, distinct from "order not established"."""

    context = _canonicalized_epoch_context(
        tmp_path,
        epochs=[
            ("epoch_2018", "LAT", "2018"),
            ("epoch_2020_a", "LAT", "2020"),
            ("epoch_2020_b", "LAT", "2020"),
        ],
    )
    _final, outcomes = orch_runtime.execute_plan(context)
    outcome = next(o for o in outcomes if o.capability_id == OBSERVED_MULTI_EPOCH_SEABED_CHANGE)
    assert outcome.plan.status == BLOCKED_AMBIGUOUS_SEMANTICS
    assert outcome.manifests == ()


# --- Section 46: bedform negative/positive tests ------------------------------------------------


def _sinusoidal_bathymetry(path, *, wavelength_m, size=1200, cell_size=2.0, amplitude=1.0):
    x = np.arange(size) * cell_size
    xx, _yy = np.meshgrid(x, x)
    data = -30.0 + amplitude * np.sin(2 * np.pi * xx / wavelength_m)
    transform = from_origin(500000, 5900000, cell_size, cell_size)
    _write_raster(path, data=data, transform=transform)
    return transform, size, cell_size


def _canonical_terrain_context(tmp_path, *, wavelength_m, asset_id="seabed"):
    path = tmp_path / f"{asset_id}.tif"
    _sinusoidal_bathymetry(path, wavelength_m=wavelength_m)
    declaration = AssetDeclaration(
        asset_id=asset_id,
        semantic_role=BATHYMETRY_RASTER,
        source_sign_convention="ALREADY_ELEVATION_STYLE",
        vertical_datum="LAT",
        survey_epoch="2020",
    )
    asset = _recognize_asset(path, declaration)
    return orch_context.PlanningContext(recognized_assets=(asset,))


def test_bedform_missing_mandatory_tile_declaration_produces_exact_blocker(tmp_path):
    context = _canonical_terrain_context(tmp_path, wavelength_m=50.0)
    _final, outcomes = orch_runtime.execute_plan(context)
    outcome = next(o for o in outcomes if o.capability_id == BEDFORM_MORPHODYNAMICS)
    assert outcome.plan.status == BLOCKED_MISSING_INPUT
    assert any(bedforms_adapter.NO_DECLARED_BEDFORM_TILES in r for r in outcome.plan.reasons)
    assert outcome.manifests == ()


def test_bedform_sufficient_declared_tile_executes_real_detection(tmp_path):
    """A genuinely canonical-scale sand-wave wavelength (50 m) with a 2000 m declared tile must
    execute the real MAR-017/022 detection -- never a false negative for good input."""

    context = _canonical_terrain_context(tmp_path, wavelength_m=50.0)
    declaration = bedforms_adapter.bedform_declaration_adapter(
        DeclarationRequest(
            capability_id=BEDFORM_MORPHODYNAMICS,
            raw_capability_section={
                "tiles": [
                    {
                        "tile_id": "tile_A",
                        "center_x_m": 501200.0,
                        "center_y_m": 5898800.0,
                        "tile_size_m": 2000.0,
                        "crest_azimuth_deg": 0.0,
                    }
                ]
            },
        )
    )
    context = orch_context.PlanningContext(
        recognized_assets=context.recognized_assets,
        capability_declarations={BEDFORM_MORPHODYNAMICS: declaration},
    )
    _final, outcomes = orch_runtime.execute_plan(context)
    outcome = next(o for o in outcomes if o.capability_id == BEDFORM_MORPHODYNAMICS)
    assert outcome.plan.status == AVAILABLE
    assert len(outcome.manifests) >= 1


def test_bedform_coarse_scale_tile_never_falsely_reported_canonical_eligible(tmp_path):
    """A tile too small relative to its own detected wavelength must be reported ineligible --
    the SAME accepted >=3-wavelengths-across-tile rule, never silently relaxed."""

    context = _canonical_terrain_context(tmp_path, wavelength_m=400.0, asset_id="coarse_seabed")
    declaration = bedforms_adapter.bedform_declaration_adapter(
        DeclarationRequest(
            capability_id=BEDFORM_MORPHODYNAMICS,
            raw_capability_section={
                "tiles": [
                    {
                        "tile_id": "coarse_tile",
                        "center_x_m": 501200.0,
                        "center_y_m": 5898800.0,
                        "tile_size_m": 1000.0,
                        "crest_azimuth_deg": 0.0,
                    }
                ]
            },
        )
    )
    context = orch_context.PlanningContext(
        recognized_assets=context.recognized_assets,
        capability_declarations={BEDFORM_MORPHODYNAMICS: declaration},
    )
    _final, outcomes = orch_runtime.execute_plan(context)
    outcome = next(o for o in outcomes if o.capability_id == BEDFORM_MORPHODYNAMICS)
    tile_status_manifest = next(
        m for m in outcome.manifests if m.product_id.endswith("tile_validation_status")
    )
    import pandas as pd

    status_df = pd.read_parquet(tile_status_manifest.geometry_or_raster_path)
    assert not status_df["canonical_scale_eligible"].any()


# --- Section 47: registration tests --------------------------------------------------------


def test_register_builtin_runtimes_is_idempotent_and_order_independent(monkeypatch):
    from marine_engine.orchestration import capability as orch_capability

    fresh_registry: dict = {}
    fresh_runtimes: dict = {}
    monkeypatch.setattr(orch_capability, "CAPABILITY_REGISTRY", fresh_registry)
    monkeypatch.setattr(orch_runtime, "CAPABILITY_REGISTRY", fresh_registry)
    monkeypatch.setattr(orch_runtime, "CAPABILITY_RUNTIMES", fresh_runtimes)
    monkeypatch.setattr(orch_bootstrap, "_bootstrapped", False)

    # Re-populate the declarative registry too (it was wiped by the monkeypatch above), proving
    # the bootstrap itself does the real registration work regardless of prior import order.
    import importlib

    import marine_engine.orchestration.capability as capability_module

    importlib.reload(capability_module)
    monkeypatch.setattr(orch_runtime, "CAPABILITY_REGISTRY", capability_module.CAPABILITY_REGISTRY)

    orch_bootstrap.register_builtin_runtimes()
    first_call_ids = set(orch_runtime.CAPABILITY_RUNTIMES)
    assert first_call_ids == {
        "earthquake_cpt_liquefaction_triggering",
        "canonicalize_bathymetry",
        "terrain_derivatives",
        "bedform_morphodynamics",
        "observed_multi_epoch_seabed_change",
    }

    orch_bootstrap.register_builtin_runtimes()  # must be a safe no-op, never raise or duplicate
    assert set(orch_runtime.CAPABILITY_RUNTIMES) == first_call_ids


def test_duplicate_capability_runtime_registration_is_refused():
    existing = orch_runtime.CAPABILITY_RUNTIMES[CANONICALIZE_BATHYMETRY]
    with pytest.raises(ValueError, match=CANONICALIZE_BATHYMETRY):
        orch_runtime.register_capability_runtime(existing)


# --- Section 48: generic declaration test ---------------------------------------------------


def test_auto_process_has_no_hardcoded_capability_specific_declaration_parser():
    """Neither `_cmd_auto_process` nor its declaration-building helper may CONSTRUCT any
    capability-specific declaration type directly (a bare mention of the type name in an
    explanatory docstring, as this very test module's own docstrings do, is not itself a
    violation -- only an actual constructor call/import is)."""

    from marine_engine import cli as cli_module

    forbidden_constructions = (
        "EarthquakeTriggeringDeclaration(",
        "BathymetryDeclaration(",
        "BedformDeclaration(",
        "ChangeDeclaration(",
    )
    for source in (
        inspect.getsource(cli_module._cmd_auto_process),
        inspect.getsource(cli_module._cmd_plan_processing),
        inspect.getsource(cli_module._build_capability_declarations),
        inspect.getsource(cli_module._build_recognized_assets),
    ):
        for token in forbidden_constructions:
            assert token not in source, f"unexpectedly constructs {token!r}"

    dispatch_source = inspect.getsource(cli_module._build_capability_declarations)
    assert "runtime.declaration_adapter" in dispatch_source


# --- Section 49: CPT regression via the new generic declaration-adapter pathway -------------


def test_cpt_scenario_declaration_adapter_matches_legacy_behavior(tmp_path):
    from marine_engine.geotechnical import cpt_contract as contract
    from marine_engine.geotechnical import cpt_profile
    from marine_engine.orchestration import execution as orch_execution

    observations = __import__("pandas").DataFrame(
        {"idx": [1, 2], "depth": [0.0, 1.0], "qc": [1.0, 2.0], "fs": [10.0, 20.0], "u": [5.0, 10.0]}
    )
    build = cpt_profile.build_canonical_measurements(
        observations,
        source_id="evidence",
        test_id="CPT-1",
        depth=cpt_profile.DepthDeclaration("depth", "m", contract.DEPTH_BELOW_SEABED, "synthetic"),
        channels=[
            cpt_profile.ChannelDeclaration("qc", "MPa", contract.QC_MPA, "synthetic"),
            cpt_profile.ChannelDeclaration("fs", "kPa", contract.FS_KPA, "synthetic"),
            cpt_profile.ChannelDeclaration("u", "kPa", contract.U2_KPA, "synthetic"),
        ],
        observation_index_column="idx",
    )
    cpt_profile.write_canonical_cpt_measurements(
        build.measurements, tmp_path / "cpt.parquet", evidence_id="evidence"
    )

    assert (
        orch_execution.earthquake_triggering_declaration_adapter(
            DeclarationRequest(
                capability_id="earthquake_cpt_liquefaction_triggering", scenario_manifest_path=None
            )
        )
        is None
    )


# --- Section 50: PL854-style regression -----------------------------------------------------


def test_pl854_style_pipeline_linestring_never_becomes_bathymetry(tmp_path):
    path = tmp_path / "route.gpkg"
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[LineString([(0, 0), (1000, 1000)])], crs=_CRS)
    gdf.to_file(path, driver="GPKG", layer="route")
    fingerprint, decision = registry.inspect_and_recognize(path)
    assert decision.recognized_role != BATHYMETRY_RASTER
    assert decision.state != recognition.RECOGNIZED


def test_pl854_style_regional_coarse_bathymetry_never_becomes_high_res_bedform_evidence(tmp_path):
    """A genuinely regional-scale (long-wavelength) undulation must not be reported as
    canonical-scale bedform evidence merely because it was declared as bathymetry."""

    context = _canonical_terrain_context(tmp_path, wavelength_m=800.0, asset_id="regional_seabed")
    declaration = bedforms_adapter.bedform_declaration_adapter(
        DeclarationRequest(
            capability_id=BEDFORM_MORPHODYNAMICS,
            raw_capability_section={
                "tiles": [
                    {
                        "tile_id": "regional_tile",
                        "center_x_m": 501200.0,
                        "center_y_m": 5898800.0,
                        "tile_size_m": 2000.0,
                        "crest_azimuth_deg": 0.0,
                    }
                ]
            },
        )
    )
    context = orch_context.PlanningContext(
        recognized_assets=context.recognized_assets,
        capability_declarations={BEDFORM_MORPHODYNAMICS: declaration},
    )
    _final, outcomes = orch_runtime.execute_plan(context)
    outcome = next(o for o in outcomes if o.capability_id == BEDFORM_MORPHODYNAMICS)
    tile_status_manifest = next(
        m for m in outcome.manifests if m.product_id.endswith("tile_validation_status")
    )
    import pandas as pd

    status_df = pd.read_parquet(tile_status_manifest.geometry_or_raster_path)
    assert not status_df["canonical_scale_eligible"].any()

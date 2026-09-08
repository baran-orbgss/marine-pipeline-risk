"""Offline unit tests for marine_engine.bedforms (MAR-022).

Small synthetic arrays/records only -- never the real Sheringham Shoal
rasters, never network access. Lettered test names map to MAR-022
Section 29's required test list (A-S).
"""

from __future__ import annotations

import inspect

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString, Point, box

from marine_engine.bedforms import (
    contract,
    extraction,
    interpretation,
    matching,
    natural_context,
)
from marine_engine.bedforms import (
    maps as bedform_maps,
)
from marine_engine.bedforms import (
    report as bedform_report,
)
from marine_engine.morphology import sandwave_morphometry as swm

BEDFORM_MODULES = (
    contract,
    extraction,
    interpretation,
    matching,
    natural_context,
    bedform_maps,
    bedform_report,
)

PIXEL_SIZE_M = 1.0


def _code_only_source(module) -> str:
    """Mirrors test_seabed_change_poc.py's helper: strips the leading
    module docstring so disclaiming prose can't trip its own checks."""

    source = inspect.getsource(module)
    stripped = source.lstrip()
    if stripped.startswith('"""'):
        end = stripped.find('"""', 3)
        if end != -1:
            return stripped[end + 3 :]
    return source


def _make_sinusoidal_tile(size_px, wavelength_m, amplitude_m, azimuth_deg, pixel_size_m=1.0):
    y, x = np.indices((size_px, size_px)).astype(np.float64) * pixel_size_m
    crest_rad = np.radians(azimuth_deg)
    kx = np.sin(crest_rad + np.pi / 2) * 2 * np.pi / wavelength_m
    ky = np.cos(crest_rad + np.pi / 2) * 2 * np.pi / wavelength_m
    return amplitude_m * np.sin(kx * x + ky * y)


def _make_crest(bedform_id, x, y, azimuth_deg, wavelength_m):
    return {
        "bedform_id": bedform_id,
        "x": x,
        "y": y,
        "crest_azimuth_deg": azimuth_deg,
        "wavelength_m": wavelength_m,
    }


# --- A: a synthetic sinusoidal tile recovers wavelength through extraction ---------------------


def test_A_known_sinusoid_wavelength_recovered_via_extraction():
    tile = _make_sinusoidal_tile(1000, wavelength_m=150.0, amplitude_m=2.0, azimuth_deg=0.0)
    valid = np.ones_like(tile, dtype=bool)
    diag = swm.analyze_tile(tile, valid, PIXEL_SIZE_M)
    assert diag is not None
    transform = rasterio.Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 5900000.0)
    result = extraction.extract_tile_bedforms(
        tile,
        valid,
        transform,
        tile_id="t1",
        epoch="2020",
        center_x_m=500500.0,
        center_y_m=5899500.0,
        tile_size_m=1000.0,
        crest_azimuth_deg=diag["dominant_crest_azimuth_deg"],
    )
    assert result.bedform_rows
    wavelengths = [b["wavelength_m"] for b in result.bedform_rows]
    assert abs(np.median(wavelengths) - 150.0) < 25.0


# --- B: known wave height is recovered ---------------------------------------------------------


def test_B_known_wave_height_recovered_via_extraction():
    tile = _make_sinusoidal_tile(1000, wavelength_m=150.0, amplitude_m=2.5, azimuth_deg=0.0)
    valid = np.ones_like(tile, dtype=bool)
    diag = swm.analyze_tile(tile, valid, PIXEL_SIZE_M)
    assert diag is not None
    transform = rasterio.Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 5900000.0)
    result = extraction.extract_tile_bedforms(
        tile,
        valid,
        transform,
        tile_id="t1",
        epoch="2020",
        center_x_m=500500.0,
        center_y_m=5899500.0,
        tile_size_m=1000.0,
        crest_azimuth_deg=diag["dominant_crest_azimuth_deg"],
    )
    heights = [b["wave_height_m"] for b in result.bedform_rows]
    # Peak-to-trough of a pure sinusoid of amplitude 2.5 m is 5 m.
    assert abs(np.median(heights) - 5.0) < 1.2


# --- C: crest orientation drives the generated transect endpoints ------------------------------


def test_C_crest_azimuth_is_reused_verbatim_for_every_bedform_in_the_tile():
    tile = _make_sinusoidal_tile(800, wavelength_m=120.0, amplitude_m=1.5, azimuth_deg=40.0)
    valid = np.ones_like(tile, dtype=bool)
    diag = swm.analyze_tile(tile, valid, PIXEL_SIZE_M)
    assert diag is not None
    transform = rasterio.Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 5900000.0)
    result = extraction.extract_tile_bedforms(
        tile,
        valid,
        transform,
        tile_id="t1",
        epoch="2018",
        center_x_m=500400.0,
        center_y_m=5899600.0,
        tile_size_m=800.0,
        crest_azimuth_deg=diag["dominant_crest_azimuth_deg"],
    )
    assert result.crest_points
    azimuths = {c["crest_azimuth_deg"] for c in result.crest_points}
    assert azimuths == {diag["dominant_crest_azimuth_deg"]}


# --- D/E: scale-separation gate ------------------------------------------------------------------


def test_D_below_30m_is_classified_small_bedform():
    assert extraction.classify_bedform_scale(29.9) == extraction.SANDBED_SMALL_BEDFORM
    assert extraction.classify_bedform_scale(5.0) == extraction.SANDBED_SMALL_BEDFORM


def test_E_at_or_above_30m_is_classified_sand_wave_scale():
    assert extraction.classify_bedform_scale(30.0) == extraction.SANDBED_SAND_WAVE_SCALE
    assert extraction.classify_bedform_scale(85.0) == extraction.SANDBED_SAND_WAVE_SCALE


# --- F: two epochs with a known translation are matched with the correct normal displacement ---


def test_F_two_epochs_with_known_translation_recovers_normal_displacement():
    crest1 = _make_crest("e1_a", x=1000.0, y=1000.0, azimuth_deg=0.0, wavelength_m=150.0)
    crest2 = _make_crest("e2_a", x=1005.0, y=1000.0, azimuth_deg=2.0, wavelength_m=155.0)
    all_pairs, canonical = matching.match_crests_within_tile("tile1", [crest1], [crest2])
    assert len(canonical) == 1
    match = canonical[0]
    assert match["match_status"] == matching.MATCHED_HIGH_SUPPORT
    assert match["normal_displacement_m"] == pytest.approx(5.0, abs=1e-6)
    assert match["along_crest_displacement_m"] == pytest.approx(0.0, abs=1e-6)
    assert len(all_pairs) == 1


# --- G: normal-axis decomposition matches hand computation --------------------------------------


def test_G_displacement_decomposition_matches_hand_geometry():
    decomposition = matching.decompose_displacement(10.0, 0.0, reference_crest_azimuth_deg=0.0)
    assert decomposition.normal_azimuth_deg == pytest.approx(90.0)
    assert decomposition.normal_component_m == pytest.approx(10.0)
    assert decomposition.along_crest_component_m == pytest.approx(0.0, abs=1e-9)

    decomposition2 = matching.decompose_displacement(0.0, 10.0, reference_crest_azimuth_deg=0.0)
    assert decomposition2.normal_component_m == pytest.approx(0.0, abs=1e-9)
    assert decomposition2.along_crest_component_m == pytest.approx(10.0)


# --- H: rotation/orientation incompatibility rejects a candidate --------------------------------


def test_H_orientation_incompatible_pair_is_rejected():
    crest1 = _make_crest("e1_a", x=0.0, y=0.0, azimuth_deg=0.0, wavelength_m=150.0)
    crest2 = _make_crest("e2_a", x=2.0, y=2.0, azimuth_deg=50.0, wavelength_m=150.0)
    all_pairs, canonical = matching.match_crests_within_tile("tile1", [crest1], [crest2])
    assert canonical == []
    assert all_pairs[0]["rejection_reason"] == matching.REJECTED_ORIENTATION_INCOMPATIBLE
    assert all_pairs[0]["match_status"] is None


# --- I: an ambiguous nearest-neighbour case is rejected, never forced ---------------------------


def test_I_ambiguous_tied_candidates_are_rejected_not_forced():
    crest1 = _make_crest("e1_a", x=0.0, y=0.0, azimuth_deg=0.0, wavelength_m=150.0)
    crest2a = _make_crest("e2_a", x=10.0, y=0.0, azimuth_deg=0.0, wavelength_m=150.0)
    crest2b = _make_crest("e2_b", x=13.0, y=0.0, azimuth_deg=0.0, wavelength_m=150.0)
    all_pairs, canonical = matching.match_crests_within_tile("tile1", [crest1], [crest2a, crest2b])
    assert canonical == []
    statuses = {p["epoch2_crest_id"]: p["match_status"] for p in all_pairs}
    assert statuses == {
        "e2_a": matching.AMBIGUOUS_NO_CANONICAL_MATCH,
        "e2_b": matching.AMBIGUOUS_NO_CANONICAL_MATCH,
    }
    assert all(p["rejection_reason"] == matching.REJECTED_AMBIGUOUS for p in all_pairs)


# --- J: wavelength-scale inconsistency rejects a candidate --------------------------------------


def test_J_wavelength_scale_inconsistent_pair_is_rejected():
    crest1 = _make_crest("e1_a", x=0.0, y=0.0, azimuth_deg=0.0, wavelength_m=100.0)
    crest2 = _make_crest("e2_a", x=3.0, y=0.0, azimuth_deg=0.0, wavelength_m=300.0)
    all_pairs, canonical = matching.match_crests_within_tile("tile1", [crest1], [crest2])
    assert canonical == []
    assert all_pairs[0]["rejection_reason"] == matching.REJECTED_WAVELENGTH_SCALE_INCONSISTENT


# --- K: no asymmetry -> migration inference ------------------------------------------------------


def test_K_no_asymmetry_based_migration_inference():
    """`compute_apparent_displacement_rate` takes only a normal
    displacement and elapsed time -- asymmetry never enters the matching
    or rate-computation code at all."""

    params = set(inspect.signature(matching.compute_apparent_displacement_rate).parameters)
    assert "asymmetry" not in " ".join(params).lower()
    assert "asymmetry" not in _code_only_source(matching).lower()


# --- L: no DoD-forced matching --------------------------------------------------------------------


def test_L_matching_function_never_takes_a_dod_or_delta_parameter():
    params = set(inspect.signature(matching.match_crests_within_tile).parameters)
    assert params == {"tile_id", "epoch1_crests", "epoch2_crests"}
    match_source = inspect.getsource(matching.match_crests_within_tile)
    assert "dod" not in match_source.lower()
    assert "delta_bed_elevation" not in match_source.lower()


def test_L2_dod_sampling_is_a_separate_post_hoc_function_never_used_inside_matching():
    matching_source = inspect.getsource(matching.match_crests_within_tile)
    assert "sample_dod_around_matched_crest" not in matching_source


# --- M: an anthropogenic-disturbed tile is never marked natural ---------------------------------


def test_M_no_anthropogenic_tile_marked_natural():
    tile_box = box(0.0, 0.0, 100.0, 100.0)
    anthropogenic_gdf = gpd.GeoDataFrame(
        {"source_layer": ["PointFeatures"], "raw_description": ["Jackup location"]},
        geometry=[Point(50.0, 50.0)],
        crs="EPSG:32631",
    )
    result = natural_context.assess_natural_bedform_validation_status(
        tile_box, anthropogenic_context_gdf=anthropogenic_gdf, interpretation_available=True
    )
    assert result.status == natural_context.ANTHROPOGENIC_DISTURBANCE_PRESENT
    assert result.status != natural_context.NATURAL_SEABED_ELIGIBLE


def test_M2_a_tile_with_zero_intersecting_anthropogenic_features_is_natural_eligible():
    tile_box = box(0.0, 0.0, 100.0, 100.0)
    anthropogenic_gdf = gpd.GeoDataFrame(
        {"source_layer": ["PointFeatures"], "raw_description": ["Jackup location"]},
        geometry=[Point(5000.0, 5000.0)],
        crs="EPSG:32631",
    )
    result = natural_context.assess_natural_bedform_validation_status(
        tile_box, anthropogenic_context_gdf=anthropogenic_gdf, interpretation_available=True
    )
    assert result.status == natural_context.NATURAL_SEABED_ELIGIBLE


# --- N: unavailable interpretation is never silently treated as eligible ------------------------


def test_N_infrastructure_context_insufficient_when_interpretation_unavailable():
    tile_box = box(0.0, 0.0, 100.0, 100.0)
    result = natural_context.assess_natural_bedform_validation_status(
        tile_box, anthropogenic_context_gdf=None, interpretation_available=False
    )
    assert result.status == natural_context.INFRASTRUCTURE_CONTEXT_INSUFFICIENT


# --- O: an unmapped descriptor is never assumed natural or anthropogenic -----------------------


def test_O_unclassified_descriptor_is_never_assumed_natural_or_anthropogenic():
    gdf = gpd.GeoDataFrame(
        {"Descriptio": ["Unknown_Linear_Feature"]},
        geometry=[LineString([(0.0, 0.0), (10.0, 10.0)])],
        crs="EPSG:32631",
    )
    layer = interpretation.InterpretationLayer("LinearFeatures", gdf, "Descriptio")
    classified = interpretation.classify_features(
        [layer],
        category_by_normalized_descriptor={
            "sandwave crest": interpretation.NATURAL_BEDFORM_INTERPRETATION,
            "jackup location": interpretation.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
        },
    )
    assert classified.iloc[0]["interpretation_category"] == (
        interpretation.UNCLASSIFIED_INTERPRETATION_FEATURE
    )


# --- P: source interpretation is never used as detector input -----------------------------------


def test_P_extraction_and_matching_never_import_interpretation_modules():
    for module in (extraction, matching):
        code = _code_only_source(module)
        import_lines = "\n".join(
            line for line in code.splitlines() if line.strip().startswith(("import ", "from "))
        )
        assert "interpretation" not in import_lines
        assert "natural_context" not in import_lines


# --- Q: no route/risk score ------------------------------------------------------------------


def test_Q_no_risk_or_susceptibility_fields_in_bedform_engine_code():
    forbidden_patterns = (
        "risk_score",
        "susceptibility_score",
        "scour_score",
        "freespan_score",
        "route_suitability",
        "hazard_score",
    )
    for module in BEDFORM_MODULES:
        code = _code_only_source(module).lower()
        for fragment in forbidden_patterns:
            assert fragment not in code, f"{module.__name__} references {fragment!r}"


# --- R: no future-migration-prediction code path ------------------------------------------------


def test_R_no_future_migration_prediction_function_exists():
    """Uses code-shaped (never prose-shaped) forbidden patterns -- this
    package's own disclaiming strings legitimately contain the PROSE
    phrase 'future migration rate' (e.g. `APPARENT_RATE_DISCLAIMER`),
    which must never trip this check (same lesson as MAR-020/021's own
    'test_Q' fix)."""

    forbidden_code_patterns = (
        "def predict",
        "predict_migration",
        "forecast_migration",
        "migration_rate =",
        "future_crest_position",
    )
    for module in BEDFORM_MODULES:
        code = _code_only_source(module).lower()
        for fragment in forbidden_code_patterns:
            assert fragment not in code, f"{module.__name__} contains {fragment!r}"
    assert "future migration rate" in matching.APPARENT_RATE_DISCLAIMER.lower()


# --- S: generic bedforms package has no Sheringham/PL854 coordinates ---------------------------


def test_S_bedforms_package_has_no_sheringham_pl854_hardcoding():
    forbidden_imports = ("evidence_atlas", "providers.bathymetry", "pl854_")
    forbidden_identifiers = ("sheringham", "tce-1975", "tce-1986", "32631", "g201193", "g181231")
    for module in BEDFORM_MODULES:
        code = _code_only_source(module)
        import_lines = "\n".join(
            line for line in code.splitlines() if line.strip().startswith(("import ", "from "))
        )
        for fragment in forbidden_imports:
            assert fragment not in import_lines, f"{module.__name__} imports {fragment!r}"
        code_lower = code.lower()
        for fragment in forbidden_identifiers:
            assert fragment not in code_lower, f"{module.__name__} references {fragment!r}"


# --- U: one epoch2 crest is never claimed by two different epoch1 crests -----------------------


def test_U_no_epoch2_crest_is_claimed_by_two_epoch1_crests():
    crest1a = _make_crest("e1_a", x=0.0, y=0.0, azimuth_deg=0.0, wavelength_m=150.0)
    crest1b = _make_crest("e1_b", x=50.0, y=0.0, azimuth_deg=0.0, wavelength_m=150.0)
    crest2 = _make_crest("e2_x", x=5.0, y=0.0, azimuth_deg=0.0, wavelength_m=150.0)
    all_pairs, canonical = matching.match_crests_within_tile("tile1", [crest1a, crest1b], [crest2])
    assert len(canonical) == 1
    assert canonical[0]["epoch1_crest_id"] == "e1_a"
    by_epoch1 = {p["epoch1_crest_id"]: p for p in all_pairs}
    assert by_epoch1["e1_b"]["rejection_reason"] == matching.REJECTED_NOT_BEST_CANDIDATE
    assert by_epoch1["e1_b"]["match_status"] is None


# --- Contract / report smoke tests ---------------------------------------------------------------


def test_V_comparator_handles_a_null_geometry_row_in_interpretation_layer():
    """A real bug found via the real CLI run: the real LinearFeatures
    interpretation layer carries at least one row with a missing/None
    geometry, which crashed the naive per-row azimuth computation. That
    row must simply be skipped, never crash the whole comparison."""

    detected = gpd.GeoDataFrame(
        {"point_id": ["p1"], "x": [10.0], "y": [10.0], "crest_azimuth_deg": [0.0]},
        geometry=[Point(10.0, 10.0)],
        crs="EPSG:32631",
    )
    natural_interp = gpd.GeoDataFrame(
        {
            "source_layer": ["LinearFeatures", "LinearFeatures"],
            "raw_description": ["Sandwave crest", "Sandwave crest"],
        },
        geometry=[None, LineString([(0.0, 0.0), (0.0, 20.0)])],
        crs="EPSG:32631",
    )
    status, result_df = interpretation.compare_detected_crests_to_interpretation(
        detected, natural_interp
    )
    assert status == interpretation.SOURCE_BEDFORM_COMPARATOR_AVAILABLE
    assert len(result_df) == 1
    assert result_df.iloc[0]["nearest_distance_m"] == pytest.approx(10.0)


def test_contract_lists_required_and_preferred_fields():
    built = contract.build_bedform_morphodynamics_input_contract()
    assert built["required_fields_static_morphometry"]
    assert built["required_fields_multi_epoch_change"]
    assert built["strongly_preferred_fields"]


def test_report_blocks_render_to_html_without_error():
    blocks = bedform_report.build_bedform_morphodynamics_report_blocks(
        project_title="Test POC",
        source_data_facts={"a": "b"},
        canonical_support_facts={"a": "b"},
        natural_vs_anthropogenic_facts={"a": "b"},
        epoch1_morphometry_facts={"a": "b"},
        epoch2_morphometry_facts={"a": "b"},
        crest_matching_facts={"a": "b"},
        observed_displacement_facts={"a": "b"},
        dod_supporting_context_text="text",
        source_interpretation_comparison_facts={"a": "b"},
        limitations=["limit 1"],
        input_contract_summary=["field: description"],
    )
    html = bedform_report.render_blocks_html(blocks, title="Test POC")
    assert "Test POC" in html
    assert bedform_report.REQUIRED_DISCLAIMER_1 in html
    assert bedform_report.REQUIRED_DISCLAIMER_2 in html

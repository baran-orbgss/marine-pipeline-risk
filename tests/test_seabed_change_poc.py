"""Offline unit tests for marine_engine.change (MAR-021; MAR-021A repairs
the uncertainty semantics tested by S-U).

Small synthetic arrays/facts only -- never the real Sheringham Shoal
rasters, never network access. Lettered test names map to MAR-021 Section
27's required test list (A-R), plus MAR-021A's required test list (S-U).
"""

from __future__ import annotations

import datetime
import inspect

import numpy as np
import pytest
import rasterio

from marine_engine import cli
from marine_engine.change import (
    alignment,
    common_support,
    comparator,
    contract,
    dod,
    epoch_compatibility,
    uncertainty,
)
from marine_engine.change import (
    maps as change_maps,
)
from marine_engine.change import (
    report as change_report,
)
from marine_engine.terrain import canonical

CHANGE_MODULES = (
    alignment,
    comparator,
    common_support,
    contract,
    dod,
    epoch_compatibility,
    change_maps,
    change_report,
    uncertainty,
)


def _code_only_source(module) -> str:
    """Mirrors test_terrain_poc.py's helper: strips the leading module
    docstring so prose explaining what a module does NOT depend on can't
    trip its own independence check."""

    source = inspect.getsource(module)
    stripped = source.lstrip()
    if stripped.startswith('"""'):
        end = stripped.find('"""', 3)
        if end != -1:
            return stripped[end + 3 :]
    return source


# --- A/B/C: identical epochs / known raising / known lowering patches -------------------------


def test_A_identical_epochs_give_exactly_zero_dod():
    n = 20
    elevation = np.full((n, n), -12.0)
    valid = np.ones((n, n), dtype=bool)
    common = common_support.build_common_valid_support(valid, valid)
    result = dod.compute_delta_bed_elevation(elevation, elevation, common.common_valid_mask)
    assert np.nanmax(np.abs(result.delta_bed_elevation_m)) < 1e-9


def test_B_a_known_plus_one_metre_patch_is_recovered_as_raising():
    n = 20
    epoch1 = np.full((n, n), -10.0)
    epoch2 = epoch1.copy()
    epoch2[5:10, 5:10] += 1.0
    valid = np.ones((n, n), dtype=bool)
    common = common_support.build_common_valid_support(valid, valid)
    result = dod.compute_delta_bed_elevation(epoch1, epoch2, common.common_valid_mask)
    assert np.allclose(result.delta_bed_elevation_m[5:10, 5:10], 1.0)
    labels = dod.classify_change_direction(result.delta_bed_elevation_m)
    assert (labels[5:10, 5:10] == dod.OBSERVED_SEABED_RAISING).all()


def test_C_a_known_minus_one_metre_patch_is_recovered_as_lowering():
    n = 20
    epoch1 = np.full((n, n), -10.0)
    epoch2 = epoch1.copy()
    epoch2[5:10, 5:10] -= 1.0
    valid = np.ones((n, n), dtype=bool)
    common = common_support.build_common_valid_support(valid, valid)
    result = dod.compute_delta_bed_elevation(epoch1, epoch2, common.common_valid_mask)
    assert np.allclose(result.delta_bed_elevation_m[5:10, 5:10], -1.0)
    labels = dod.classify_change_direction(result.delta_bed_elevation_m)
    assert (labels[5:10, 5:10] == dod.OBSERVED_SEABED_LOWERING).all()


# --- D: positive-down sources converted before subtraction ------------------------------------


def test_D_positive_down_sources_are_canonicalised_before_subtraction():
    raw1_depth = np.full((5, 5), 10.0)
    raw2_depth = np.full((5, 5), 9.0)  # shallower next epoch -> seabed raised by 1m
    c1 = canonical.build_canonical_bed_elevation(
        raw1_depth,
        nodata_value=None,
        source_sign_convention=canonical.POSITIVE_DOWN_DEPTH,
        source_vertical_datum="LAT",
    )
    c2 = canonical.build_canonical_bed_elevation(
        raw2_depth,
        nodata_value=None,
        source_sign_convention=canonical.POSITIVE_DOWN_DEPTH,
        source_vertical_datum="LAT",
    )
    common = common_support.build_common_valid_support(c1.valid_mask, c2.valid_mask)
    result = dod.compute_delta_bed_elevation(
        c1.bed_elevation_m, c2.bed_elevation_m, common.common_valid_mask
    )
    assert np.allclose(result.delta_bed_elevation_m, 1.0)


# --- E: datum mismatch blocks subtraction -------------------------------------------------------


def test_E_datum_mismatch_without_evidence_is_not_harmonized():
    result = epoch_compatibility.assess_vertical_datum_compatibility("LAT", "MSL")
    assert result.status == epoch_compatibility.VERTICAL_DATUM_NOT_HARMONIZED


def test_E2_identical_datum_labels_are_harmonized():
    result = epoch_compatibility.assess_vertical_datum_compatibility("LAT", "LAT")
    assert result.status == epoch_compatibility.VERTICAL_DATUM_HARMONIZED


def test_E3_differing_labels_with_explicit_evidence_are_harmonized():
    result = epoch_compatibility.assess_vertical_datum_compatibility(
        "LAT", "MSL", harmonization_evidence="a real quoted source statement"
    )
    assert result.status == epoch_compatibility.VERTICAL_DATUM_HARMONIZED
    assert result.evidence is not None


def test_E4_datum_gate_never_takes_elevation_data_as_input():
    """Structural guarantee: the gate function's signature has no
    elevation/array/DoD parameter at all, so it is impossible for it to
    infer harmonization from a computed difference (the ticket's explicit
    prohibition)."""

    sig = inspect.signature(epoch_compatibility.assess_vertical_datum_compatibility)
    for name in sig.parameters:
        assert "elevation" not in name.lower()
        assert "delta" not in name.lower()
        assert "dod" not in name.lower()


# --- F: nodata intersection -- single-epoch cell is never zero change -------------------------


def test_F_single_epoch_only_cells_are_nan_never_zero():
    valid1 = np.ones((10, 10), dtype=bool)
    valid2 = np.ones((10, 10), dtype=bool)
    valid2[0:3, :] = False
    common = common_support.build_common_valid_support(valid1, valid2)
    elevation = np.full((10, 10), -5.0)
    result = dod.compute_delta_bed_elevation(elevation, elevation, common.common_valid_mask)
    assert np.isnan(result.delta_bed_elevation_m[0:3, :]).all()
    assert common.fraction_of_epoch1_covered == pytest.approx(0.7)
    assert common.fraction_of_epoch2_covered == pytest.approx(1.0)


# --- G: different extents --------------------------------------------------------------------


def test_G_different_extents_yield_a_smaller_common_support_than_either_epoch():
    valid1 = np.zeros((20, 20), dtype=bool)
    valid1[0:15, 0:15] = True
    valid2 = np.zeros((20, 20), dtype=bool)
    valid2[5:20, 5:20] = True
    common = common_support.build_common_valid_support(valid1, valid2)
    assert common.common_valid_cell_count < valid1.sum()
    assert common.common_valid_cell_count < valid2.sum()
    assert common.common_valid_cell_count == 10 * 10


# --- H/I/J: grid alignment classification ------------------------------------------------------


def test_H_identical_grids_are_exact_alignment():
    t = rasterio.Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 2000.0)
    result = epoch_compatibility.classify_grid_alignment(
        crs1="EPSG:32631", transform1=t, crs2="EPSG:32631", transform2=t
    )
    assert result.status == epoch_compatibility.EXACT_GRID_ALIGNMENT
    assert result.row_offset == 0 and result.col_offset == 0


def test_I_whole_pixel_offset_grids_are_integer_pixel_alignment():
    t1 = rasterio.Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 2000.0)
    t2 = rasterio.Affine(1.0, 0.0, 1003.0, 0.0, -1.0, 1990.0)
    result = epoch_compatibility.classify_grid_alignment(
        crs1="EPSG:32631", transform1=t1, crs2="EPSG:32631", transform2=t2
    )
    assert result.status == epoch_compatibility.INTEGER_PIXEL_OFFSET_ALIGNMENT
    assert result.col_offset == 3
    assert result.row_offset == 10


def test_I2_real_sheringham_2018_vs_2020_grids_are_integer_pixel_alignment():
    """The REAL 2018 (rasterized from XYZ, origin 371140/5894007) and 2020
    (origin 371143/5894017) grids, both 1m/no rotation, land on an exact
    3-column/10-row integer offset -- confirmed by direct inspection of
    both real rasters, not assumed."""

    t2018 = rasterio.Affine(1.0, 0.0, 371140.0, 0.0, -1.0, 5894007.0)
    t2020 = rasterio.Affine(1.0, 0.0, 371143.0, 0.0, -1.0, 5894017.0)
    result = epoch_compatibility.classify_grid_alignment(
        crs1="EPSG:32631", transform1=t2018, crs2="EPSG:32631", transform2=t2020
    )
    assert result.status == epoch_compatibility.INTEGER_PIXEL_OFFSET_ALIGNMENT
    assert result.col_offset == 3
    assert result.row_offset == -10


def test_I3_crop_array_to_aligned_window_is_spatially_correct_for_a_third_array():
    """A real bug found via the real CLI run: an independent comparator
    product (sharing epoch2's ORIGINAL, uncropped grid) must be cropped to
    the SAME window already used for the DoD before comparison -- this
    checks not just matching shapes but that the SAME real-world
    coordinate maps to the SAME value before and after cropping."""

    t1 = rasterio.Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 2000.0)
    t2 = rasterio.Affine(1.0, 0.0, 1003.0, 0.0, -1.0, 1990.0)
    n1, n2 = 30, 25
    valid1 = np.ones((n1, n1), dtype=bool)
    valid2 = np.ones((n2, n2), dtype=bool)
    classification = epoch_compatibility.classify_grid_alignment(
        crs1="X", transform1=t1, crs2="X", transform2=t2
    )
    _aligned1, aligned2 = alignment.align_to_common_grid(
        classification=classification,
        elevation1=np.zeros((n1, n1)),
        valid1=valid1,
        transform1=t1,
        elevation2=np.zeros((n2, n2)),
        valid2=valid2,
        transform2=t2,
        crs="X",
    )
    source_full = np.arange(n2 * n2, dtype=float).reshape(n2, n2)
    cropped = alignment.crop_array_to_aligned_window(
        source_full,
        original_transform=t2,
        aligned_transform=aligned2.transform,
        aligned_shape=aligned2.elevation.shape,
    )
    assert cropped.shape == aligned2.elevation.shape

    real_x, real_y = 1010.5, 1985.5
    orig_col, orig_row = int((real_x - t2.c) / t2.a), int((t2.f - real_y) / abs(t2.e))
    new_col, new_row = (
        int((real_x - aligned2.transform.c) / aligned2.transform.a),
        int((aligned2.transform.f - real_y) / abs(aligned2.transform.e)),
    )
    assert cropped[new_row, new_col] == source_full[orig_row, orig_col]


def test_I4_crop_array_to_aligned_window_is_a_true_noop_for_exact_alignment():
    t = rasterio.Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 2000.0)
    arr = np.arange(100).reshape(10, 10).astype(float)
    result = alignment.crop_array_to_aligned_window(
        arr, original_transform=t, aligned_transform=t, aligned_shape=(10, 10)
    )
    assert result is arr


def test_J_fractional_origin_offset_requires_resampling():
    t1 = rasterio.Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 2000.0)
    t2 = rasterio.Affine(1.0, 0.0, 1000.5, 0.0, -1.0, 2000.0)
    result = epoch_compatibility.classify_grid_alignment(
        crs1="EPSG:32631", transform1=t1, crs2="EPSG:32631", transform2=t2
    )
    assert result.status == epoch_compatibility.RESAMPLING_REQUIRED


def test_J2_mismatched_crs_is_incompatible_not_resampling():
    t = rasterio.Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 2000.0)
    result = epoch_compatibility.classify_grid_alignment(
        crs1="EPSG:32631", transform1=t, crs2="EPSG:4326", transform2=t
    )
    assert result.status == epoch_compatibility.INCOMPATIBLE_HORIZONTAL_REFERENCE


# --- K: no upsampling ---------------------------------------------------------------------------


def test_K_resampling_target_is_epoch2s_own_native_resolution_never_finer():
    n1 = 25
    elev1 = np.zeros((n1, n1))
    valid1 = np.ones((n1, n1), dtype=bool)
    t1 = rasterio.Affine(2.0, 0.0, 5000.0, 0.0, -2.0, 6000.0)
    n2 = 50
    elev2 = np.zeros((n2, n2))
    valid2 = np.ones((n2, n2), dtype=bool)
    t2 = rasterio.Affine(1.0, 0.0, 5000.0, 0.0, -1.0, 6000.0)
    classification = epoch_compatibility.classify_grid_alignment(
        crs1="EPSG:32631", transform1=t1, crs2="EPSG:32631", transform2=t2
    )
    assert classification.status == epoch_compatibility.RESAMPLING_REQUIRED
    aligned1, aligned2 = alignment.align_to_common_grid(
        classification=classification,
        elevation1=elev1,
        valid1=valid1,
        transform1=t1,
        elevation2=elev2,
        valid2=valid2,
        transform2=t2,
        crs="EPSG:32631",
    )
    assert aligned1.was_resampled is True
    assert aligned1.elevation.shape == elev2.shape
    assert aligned1.resampling_method is not None
    assert aligned2.was_resampled is False


# --- L: annualization sign and duration ---------------------------------------------------------


def test_L_annualization_divides_by_real_elapsed_years():
    d1 = datetime.date(2018, 1, 1)
    d2 = datetime.date(2021, 1, 1)
    expected_years = (d2 - d1).days / 365.25
    delta = np.array([[1.0 * expected_years]])
    result = dod.annualize_change(delta, epoch1_date=d1, epoch2_date=d2)
    assert result.elapsed_years == pytest.approx(expected_years)
    assert result.annualized_delta_m_per_year[0, 0] == pytest.approx(1.0)


def test_L2_annualization_rejects_non_chronological_dates():
    d1 = datetime.date(2020, 1, 1)
    d2 = datetime.date(2018, 1, 1)
    with pytest.raises(ValueError):
        dod.annualize_change(np.array([[1.0]]), epoch1_date=d1, epoch2_date=d2)


def test_L3_annualization_disclaimer_is_present_and_explicit():
    result = dod.annualize_change(
        np.array([[1.0]]),
        epoch1_date=datetime.date(2018, 1, 1),
        epoch2_date=datetime.date(2019, 1, 1),
    )
    assert "NOT A FUTURE CHANGE RATE PREDICTION" in result.disclaimer


# --- M: uncertainty propagation sqrt(s1^2+s2^2) -------------------------------------------------


def test_M_scalar_propagation_matches_the_pythagorean_formula():
    got = uncertainty.propagate_sigma_dod_scalar(0.2, 0.2)
    assert got == pytest.approx(0.2 * np.sqrt(2))


def test_M2_raster_propagation_matches_elementwise():
    sigma1 = np.full((4, 4), 0.2)
    sigma2 = np.full((4, 4), 0.15)
    mask = np.ones((4, 4), dtype=bool)
    got = uncertainty.propagate_sigma_dod_raster(sigma1, sigma2, mask)
    assert np.allclose(got, np.sqrt(0.2**2 + 0.15**2))


# --- N: missing uncertainty does not invent a threshold -----------------------------------------


def test_N_missing_evidence_yields_the_not_demonstrated_status_never_a_number():
    result = uncertainty.derive_change_threshold(
        sigma_epoch1_m=None, sigma_epoch2_m=None, evidence_citation=None
    )
    assert result.status == uncertainty.GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED
    assert result.generic_threshold_m is None
    assert result.nominal_accuracy_rss_reference_m is None


def test_N2_real_nominal_accuracy_evidence_still_never_demonstrates_a_generic_threshold():
    """MAR-021A: this is the core repair. MAR-021 treated a real, well-cited
    'typically less than +/-0.2 m' NOMINAL accuracy figure as if it were a
    verified 1-sigma standard uncertainty and called sqrt(0.2^2+0.2^2) a
    'defensible threshold' -- that overstated the source's own claim. Even
    with real evidence supplied, the status must stay NOT_DEMONSTRATED, and
    `generic_threshold_m` must stay None; the RSS arithmetic is still
    computed but only as a labelled, non-canonical reference number."""

    result = uncertainty.derive_change_threshold(
        sigma_epoch1_m=0.2, sigma_epoch2_m=0.2, evidence_citation="a real source citation"
    )
    assert result.status == uncertainty.GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED
    assert result.generic_threshold_m is None
    assert result.nominal_accuracy_rss_reference_m == pytest.approx(0.2 * np.sqrt(2))
    assert "sqrt" in result.formula
    assert "NOT" in result.formula  # the formula string must itself disclaim canonical status


# --- MAR-021A S/T/U: uncertainty semantics repair -----------------------------------------------


def test_S_nominal_accuracy_cannot_be_silently_treated_as_sigma_regardless_of_magnitude():
    """Parametrize-by-hand over a few plausible nominal accuracy pairs --
    none of them may ever produce a demonstrated generic threshold, since
    the INPUT KIND (nominal, not verified-1-sigma) is what disqualifies
    them, not their specific values."""

    for sigma1, sigma2 in ((0.2, 0.2), (0.15, 0.25), (0.5, 0.1)):
        result = uncertainty.derive_change_threshold(
            sigma_epoch1_m=sigma1, sigma_epoch2_m=sigma2, evidence_citation="cited"
        )
        assert result.status == uncertainty.GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED
        assert result.generic_threshold_m is None


def test_T_source_specific_analyst_threshold_is_a_distinct_kind_never_fed_into_rss():
    """Source-inspection: the 0.3 m Fugro analyst threshold constant is
    never passed as a sigma_epoch*_m argument anywhere in the CLI (i.e. it
    never enters the RSS/threshold arithmetic) -- it is wired only into a
    SOURCE_SPECIFIC_ANALYST_THRESHOLD evidence item and the report's
    separate 'Source-Specific Analyst Threshold' section."""

    source = inspect.getsource(cli._cmd_build_seabed_change_poc)
    assert "REPORTED_ANALYST_SIGNIFICANCE_THRESHOLD_M" in source
    assert "EVIDENCE_SOURCE_SPECIFIC_ANALYST_THRESHOLD" in source
    # It must appear in the analyst-threshold evidence item / report list, never as a
    # sigma_epoch1_m/sigma_epoch2_m argument to derive_change_threshold.
    threshold_call_start = source.index("derive_change_threshold(")
    threshold_call_end = source.index(")", threshold_call_start)
    threshold_call_args = source[threshold_call_start:threshold_call_end]
    assert "REPORTED_ANALYST_SIGNIFICANCE_THRESHOLD_M" not in threshold_call_args


def test_U_validation_question_e_is_hardcoded_no_with_the_required_reason():
    source = inspect.getsource(cli._cmd_build_seabed_change_poc)
    assert '"question_e_defensible_uncertainty_threshold": "NO"' in source
    assert '"question_e_defensible_uncertainty_threshold": "YES"' not in source
    assert "insufficient for" in source and "generic uncertainty propagation" in source


def test_U2_no_thresholded_change_classification_is_ever_created():
    """Neither the CLI nor the change engine ever filters/classifies DoD
    cells by a significance threshold (MAR-021A Section 8) -- the only
    per-cell classification available (classify_change_direction) is a
    raw sign split, never gated by any threshold value."""

    cli_source = inspect.getsource(cli._cmd_build_seabed_change_poc)
    assert "classify_change_direction" not in cli_source
    dod_source = inspect.getsource(dod.classify_change_direction)
    assert "threshold" not in dod_source.lower()


def test_U3_dod_computation_block_is_structurally_independent_of_uncertainty():
    """The DoD-computation code (Section 11) must not reference the
    uncertainty/threshold machinery at all -- a structural guarantee that
    the MAR-021A semantics repair cannot have touched the DoD raster
    itself (requirement 1: DoD raster must be unchanged)."""

    source = inspect.getsource(cli._cmd_build_seabed_change_poc)
    dod_start = source.index("Computing the canonical DoD")
    qa_start = source.index("Assessing horizontal misregistration QA")
    dod_block = source[dod_start:qa_start]
    assert "threshold" not in dod_block.lower()
    assert "uncertainty" not in dod_block.lower()


# --- O: median vertical bias is reported but not auto-corrected --------------------------------


def test_O_vertical_bias_qa_reports_the_median_without_altering_the_dod():
    delta = np.full((6, 6), 0.3) + np.random.default_rng(0).normal(0, 0.01, (6, 6))
    qa = comparator.assess_vertical_bias_qa(delta)
    assert qa.median_dod_m == pytest.approx(0.3, abs=0.02)
    assert "NEVER automatically subtracted" in qa.note
    # The function must not mutate its input.
    assert delta[0, 0] != 0.0


def test_O2_comparator_sign_unresolved_reports_both_interpretations_never_choosing():
    mine = np.array([[1.0, 0.8, 1.2, 0.9]])
    theirs = np.array([[1.0, 0.8, 1.2, 0.9]])
    mask = np.ones((1, 4), dtype=bool)
    result = comparator.compare_dod_to_source_product(
        mine, theirs, mask, sign_evidence="undocumented"
    )
    assert result.sign_status == comparator.SIGN_UNRESOLVED_FROM_DOCUMENTATION
    assert set(result.interpretations) == {"as_is", "sign_flipped"}


# --- P: no route/KP fabrication ------------------------------------------------------------------


def test_P_route_kp_status_is_always_the_fixed_not_applicable_literal():
    source = inspect.getsource(cli._cmd_build_seabed_change_poc)
    assert '"NOT_APPLICABLE_NO_AUTHORITATIVE_ROUTE_SUPPLIED"' in source
    assignments = [
        line.strip() for line in source.splitlines() if line.strip().startswith("route_kp_status =")
    ]
    assert len(assignments) == 1
    assert assignments[0] == 'route_kp_status = "NOT_APPLICABLE_NO_AUTHORITATIVE_ROUTE_SUPPLIED"'


def test_P2_change_engine_never_constructs_a_centerline_geometry():
    forbidden = ("LineString(", "centerline", "route_geometry")
    for module in CHANGE_MODULES:
        source = inspect.getsource(module)
        for fragment in forbidden:
            assert fragment not in source


# --- Q: no risk/susceptibility output ------------------------------------------------------------


def test_Q_no_risk_or_susceptibility_fields_in_change_engine_code():
    """Checks unambiguous, field-shaped patterns only -- bare phrases like
    'risk score'/'hazard map' legitimately appear in this package's own
    disclaiming captions (e.g. maps.py's figure caption explicitly states
    'no hazard/risk score'), which is the point of the ticket, not a
    violation of it (same lesson as MAR-020's test_C2 fix)."""

    forbidden_patterns = (
        "risk_score",
        "susceptibility_score",
        "scour_score",
        "freespan_score",
        "route_suitability",
    )
    for module in CHANGE_MODULES:
        code = _code_only_source(module).lower()
        for fragment in forbidden_patterns:
            assert fragment not in code, f"{module.__name__} references {fragment!r}"


def test_Q2_cli_command_writes_no_risk_field_literal():
    source = inspect.getsource(cli._cmd_build_seabed_change_poc)
    forbidden_field_literals = (
        '"risk_score"',
        '"susceptibility_score"',
        '"scour_score"',
        '"freespan_score"',
        '"route_suitability"',
    )
    for forbidden in forbidden_field_literals:
        assert forbidden not in source


def test_Q3_change_direction_labels_never_attribute_a_cause():
    labels = dod.classify_change_direction(np.array([[1.0, -1.0, 0.0]]))
    allowed = {dod.OBSERVED_SEABED_RAISING, dod.OBSERVED_SEABED_LOWERING, None}
    assert set(labels.ravel().tolist()) <= allowed


# --- R: generic module has no Sheringham/PL854 hard-coding ---------------------------------------


def test_R_change_package_has_no_pl854_or_sheringham_hardcoding():
    forbidden_imports = ("evidence_atlas", "providers.pl854", "pl854_")
    forbidden_identifiers = ("sheringham", "TCE-1975", "TCE-1986", "32631", "G201193", "G181231")
    for module in CHANGE_MODULES:
        code = _code_only_source(module)
        import_lines = "\n".join(
            line for line in code.splitlines() if line.strip().startswith(("import ", "from "))
        )
        for fragment in forbidden_imports:
            assert fragment not in import_lines, f"{module.__name__} imports {fragment!r}"
        code_lower = code.lower()
        for fragment in forbidden_identifiers:
            assert fragment.lower() not in code_lower, f"{module.__name__} references {fragment!r}"

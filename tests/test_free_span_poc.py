"""Offline unit tests for the MAR-025 generic pipeline free-span / support-loss POC.

Small synthetic records only for unit-level tests; the full hand-verified
`fs_synthetic.build_synthetic_free_span_case()` engineering validation case for
integration-level checks. Never the real NSTA network (the one acquisition test proves a cache
HIT skips the network entirely, by raising if `nsta_freespan.query_freespan_layer` is ever
called). Test names map to MAR-025 Section 32's required list.
"""

from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from marine_engine import cli
from marine_engine.freespan import (
    clearance,
    contract,
    maps,
    nsta_registry,
    reference,
    report,
    scenario,
    structural_handoff,
    support_state,
    synthetic,
)

FREESPAN_MODULES = (
    clearance,
    contract,
    maps,
    nsta_registry,
    reference,
    report,
    scenario,
    structural_handoff,
    support_state,
    synthetic,
)

# these engine modules must never carry real dataset identity/coordinates at all (maps.py and
# nsta_registry.py legitimately handle real NSTA/PL854 data and titles, so they are exempt, same
# precedent as burial/maps.py in MAR-024)
GENERIC_ENGINE_MODULES = (
    clearance,
    contract,
    reference,
    scenario,
    support_state,
    synthetic,
)

# structural_handoff.py and report.py legitimately name VIV/FATIGUE/ULS/FLS/RISK/PROBABILITY as
# things NOT implemented/produced -- excluded from the forbidden-vocabulary scan, same precedent
# as burial's exclusion of report.py's disclaimer prose.
VOCABULARY_SCAN_MODULES = (clearance, contract, reference, scenario, support_state)

# contract.py's own purpose text legitimately documents what MAR-025 does NOT produce (VIV,
# DNV-RP-F105, allowable span length, etc, mirroring structural_handoff.py/report.py) -- raw
# source-text scans for those exact phrases exclude it, same precedent as excluding disclaimer
# prose elsewhere in this project.
STRICT_NO_MENTION_MODULES = (clearance, reference, scenario, support_state)


# --- 1: centreline requires explicit pipe-bottom offset ---------------------------------------


def test_1_centreline_requires_explicit_pipe_bottom_offset():
    result = reference.normalize_pipe_bottom_elevation_m(
        -9.8, pipe_vertical_reference=reference.PIPE_CENTRELINE_ELEVATION
    )
    assert result is None


def test_1b_centreline_with_offset_normalizes_correctly():
    result = reference.normalize_pipe_bottom_elevation_m(
        -9.8,
        pipe_vertical_reference=reference.PIPE_CENTRELINE_ELEVATION,
        reference_to_pipe_bottom_offset_m=0.2,
    )
    assert result == pytest.approx(-10.0)


def test_1c_unknown_pipe_vertical_reference_is_rejected():
    with pytest.raises(ValueError):
        reference.normalize_pipe_bottom_elevation_m(
            -9.8, pipe_vertical_reference="NOT_A_REAL_REFERENCE"
        )


# --- 2: bottom-reference passes through correctly ----------------------------------------------


def test_2_bottom_reference_passes_through_unchanged():
    result = reference.normalize_pipe_bottom_elevation_m(
        -10.0, pipe_vertical_reference=reference.PIPE_BOTTOM_ELEVATION
    )
    assert result == pytest.approx(-10.0)


def test_2b_top_reference_requires_offset_too():
    assert (
        reference.normalize_pipe_bottom_elevation_m(
            -9.6, pipe_vertical_reference=reference.PIPE_TOP_ELEVATION
        )
        is None
    )
    result = reference.normalize_pipe_bottom_elevation_m(
        -9.6,
        pipe_vertical_reference=reference.PIPE_TOP_ELEVATION,
        reference_to_pipe_bottom_offset_m=0.4,
    )
    assert result == pytest.approx(-10.0)


# --- 3: positive clearance semantics correct ----------------------------------------------------


def test_3_positive_clearance_means_pipe_above_seabed_support():
    # pipe bottom higher (less negative / shallower) than the seabed support surface -> a gap
    # exists beneath the pipe -> positive clearance.
    result = clearance.compute_pipe_underside_clearance_m(-9.0, -9.6)
    assert result == pytest.approx(0.6)
    assert result > 0


def test_3b_negative_clearance_means_pipe_below_seabed_support():
    result = clearance.compute_pipe_underside_clearance_m(-10.0, -9.5)
    assert result == pytest.approx(-0.5)
    assert result < 0


def test_3c_missing_input_yields_no_clearance():
    assert clearance.compute_pipe_underside_clearance_m(None, -9.5) is None
    assert clearance.compute_pipe_underside_clearance_m(-9.5, None) is None


# --- 4: no threshold -> no categorical free span ------------------------------------------------


def test_4_no_threshold_means_no_categorical_free_span():
    state = support_state.classify_measured_support_state(0.6, classification_threshold_m=None)
    assert state == support_state.MEASURED_SUPPORT_STATE_UNRESOLVED
    assert state != support_state.MEASURED_UNSUPPORTED


def test_4b_not_defensible_status_constant_exists_and_is_distinct():
    assert (
        clearance.FREE_SPAN_CLASSIFICATION_NOT_DEFENSIBLE_NO_CLEARANCE_THRESHOLD
        != clearance.FREE_SPAN_CLASSIFICATION_DEFENSIBLE
    )


# --- 5: source/operator threshold stays separate from statistical uncertainty ------------------


def test_5_combined_one_sigma_threshold_formula():
    # 3-4-5 triangle: sqrt(0.03^2 + 0.04^2) = 0.05 exactly.
    result = clearance.compute_combined_one_sigma_clearance_threshold_m(0.03, 0.04)
    assert result == pytest.approx(0.05)


def test_5b_operator_and_statistical_threshold_provenance_are_distinct():
    operator_threshold = clearance.ClearanceClassificationThreshold(
        0.1, clearance.SOURCE_OR_OPERATOR_CLEARANCE_CLASSIFICATION_THRESHOLD
    )
    statistical_threshold = clearance.ClearanceClassificationThreshold(
        0.05, clearance.COMBINED_ONE_SIGMA_CLEARANCE_THRESHOLD
    )
    assert operator_threshold.provenance != statistical_threshold.provenance
    assert (
        clearance.SOURCE_OR_OPERATOR_CLEARANCE_CLASSIFICATION_THRESHOLD
        != clearance.COMBINED_ONE_SIGMA_CLEARANCE_THRESHOLD
    )


def test_5c_invalid_threshold_provenance_is_rejected():
    with pytest.raises(ValueError):
        clearance.ClearanceClassificationThreshold(0.1, "SOME_MADE_UP_PROVENANCE")


def test_5d_negative_sigma_is_rejected():
    with pytest.raises(ValueError):
        clearance.compute_combined_one_sigma_clearance_threshold_m(-0.1, 0.05)


# --- 6/7: known synthetic free span length / max clearance recovered ---------------------------


def test_6_known_synthetic_free_span_lengths_recovered():
    case = synthetic.build_synthetic_free_span_case()
    recovered = sorted(float(v) for v in case.measured_intervals_df["span_length_m"])
    expected = sorted(synthetic.EXPECTED_MEASURED_SPAN_LENGTHS_M)
    assert recovered == pytest.approx(expected)
    assert len(case.measured_intervals_df) == synthetic.EXPECTED_MEASURED_SPAN_COUNT


def test_7_known_synthetic_max_clearance_recovered():
    case = synthetic.build_synthetic_free_span_case()
    recovered_max = float(case.measured_intervals_df["maximum_clearance_m"].max())
    assert recovered_max == pytest.approx(synthetic.EXPECTED_MAX_CLEARANCE_M)


# --- 8: survey gap splits one apparent interval into two ----------------------------------------


def test_8_survey_gap_splits_one_apparent_interval_into_two():
    profile_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 10.0, 70.0, 80.0],
            "clearance_m": [0.6, 0.6, 0.6, 0.6],
            "measured_support_state": [support_state.MEASURED_UNSUPPORTED] * 4,
        }
    )
    intervals = support_state.extract_measured_free_span_intervals(
        profile_df,
        asset_id="TEST_ASSET",
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        classification_threshold_m=0.1,
        vertical_reference_semantics="TEST",
    )
    assert len(intervals) == 2
    assert sorted(intervals["span_length_m"].tolist()) == [10.0, 10.0]
    assert intervals["limitations"].notna().all()
    assert (intervals["limitations"] == support_state.SPAN_SPLIT_BY_MEASUREMENT_GAP).all()


def test_8b_no_gap_produces_one_interval():
    profile_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 10.0, 20.0, 30.0],
            "clearance_m": [0.6, 0.6, 0.6, 0.6],
            "measured_support_state": [support_state.MEASURED_UNSUPPORTED] * 4,
        }
    )
    intervals = support_state.extract_measured_free_span_intervals(
        profile_df,
        asset_id="TEST_ASSET",
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        classification_threshold_m=0.1,
        vertical_reference_semantics="TEST",
    )
    assert len(intervals) == 1
    assert intervals["span_length_m"].iloc[0] == pytest.approx(30.0)
    assert intervals["limitations"].iloc[0] is None


# --- 16.G: ordinary discrete interval is labelled UNSUPPORTED_SAMPLE_RUN_EXTENT ----------------


def test_16g_ordinary_interval_defaults_to_unsupported_sample_run_extent():
    profile_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 10.0, 20.0],
            "clearance_m": [0.6, 0.6, 0.6],
            "measured_support_state": [support_state.MEASURED_UNSUPPORTED] * 3,
        }
    )
    # interval_boundary_semantics is not passed -- must default, never silently assign
    # EXPLICIT_BOUNDARY_SAMPLES.
    intervals = support_state.extract_measured_free_span_intervals(
        profile_df,
        asset_id="TEST_ASSET",
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        classification_threshold_m=0.1,
        vertical_reference_semantics="TEST",
    )
    assert (
        intervals["interval_boundary_semantics"] == support_state.UNSUPPORTED_SAMPLE_RUN_EXTENT
    ).all()
    assert not (
        intervals["interval_boundary_semantics"] == support_state.EXPLICIT_BOUNDARY_SAMPLES
    ).any()


def test_16g2_unknown_boundary_semantics_is_rejected():
    profile_df = pd.DataFrame(
        {
            "chainage_m": [0.0],
            "clearance_m": [0.6],
            "measured_support_state": [support_state.MEASURED_UNSUPPORTED],
        }
    )
    with pytest.raises(ValueError):
        support_state.extract_measured_free_span_intervals(
            profile_df,
            asset_id="TEST_ASSET",
            max_measurement_gap_m=50.0,
            survey_epoch="TEST",
            classification_threshold_m=0.1,
            vertical_reference_semantics="TEST",
            interval_boundary_semantics="NOT_A_REAL_SEMANTICS_VALUE",
        )


def test_16g3_boundary_brackets_only_populated_when_requested():
    profile_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 10.0, 20.0, 30.0, 40.0],
            "clearance_m": [-0.5, 0.6, 0.6, 0.6, -0.5],
            "measured_support_state": [
                support_state.MEASURED_SUPPORTED,
                support_state.MEASURED_UNSUPPORTED,
                support_state.MEASURED_UNSUPPORTED,
                support_state.MEASURED_UNSUPPORTED,
                support_state.MEASURED_SUPPORTED,
            ],
        }
    )
    default_intervals = support_state.extract_measured_free_span_intervals(
        profile_df,
        asset_id="TEST_ASSET",
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        classification_threshold_m=0.1,
        vertical_reference_semantics="TEST",
    )
    assert default_intervals["left_boundary_bracket_start_m"].isna().all()
    assert default_intervals["right_boundary_bracket_start_m"].isna().all()

    bracketed_intervals = support_state.extract_measured_free_span_intervals(
        profile_df,
        asset_id="TEST_ASSET",
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        classification_threshold_m=0.1,
        vertical_reference_semantics="TEST",
        interval_boundary_semantics=support_state.BOUNDARY_BRACKETED_BY_ADJACENT_MEASUREMENTS,
    )
    row = bracketed_intervals.iloc[0]
    assert row["left_boundary_bracket_start_m"] == pytest.approx(0.0)
    assert row["left_boundary_bracket_end_m"] == pytest.approx(10.0)
    assert row["right_boundary_bracket_start_m"] == pytest.approx(30.0)
    assert row["right_boundary_bracket_end_m"] == pytest.approx(40.0)


# --- 16.H: synthetic exact case is explicitly labelled as exact-by-construction ----------------


def test_16h_synthetic_case_is_labelled_explicit_boundary_by_construction():
    case = synthetic.build_synthetic_free_span_case()
    assert (
        case.measured_intervals_df["interval_boundary_semantics"]
        == support_state.EXPLICIT_BOUNDARY_SAMPLES
    ).all()
    assert synthetic.SYNTHETIC_BOUNDARY_SEMANTICS_NOTE == (
        "EXPLICIT_BOUNDARY_SAMPLES_BY_SYNTHETIC_CONSTRUCTION"
    )
    summary = synthetic.summarize_synthetic_case(case)
    assert summary["boundary_semantics_note"] == synthetic.SYNTHETIC_BOUNDARY_SEMANTICS_NOTE


def test_16h2_explicit_boundary_is_never_the_generic_default():
    # the generic engine's own default must remain UNSUPPORTED_SAMPLE_RUN_EXTENT -- only the
    # synthetic case opts into EXPLICIT_BOUNDARY_SAMPLES, and only by explicitly passing it.
    import inspect as _inspect

    default = (
        _inspect.signature(support_state.extract_measured_free_span_intervals)
        .parameters["interval_boundary_semantics"]
        .default
    )
    assert default == support_state.UNSUPPORTED_SAMPLE_RUN_EXTENT


# --- 16.I: generic interval length is not described as exact without explicit semantics --------


def test_16i_span_length_semantics_field_present_and_honest():
    profile_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 10.0],
            "clearance_m": [0.6, 0.6],
            "measured_support_state": [support_state.MEASURED_UNSUPPORTED] * 2,
        }
    )
    intervals = support_state.extract_measured_free_span_intervals(
        profile_df,
        asset_id="TEST_ASSET",
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        classification_threshold_m=0.1,
        vertical_reference_semantics="TEST",
    )
    note = intervals["span_length_semantics"].iloc[0]
    assert "EXPLICIT_BOUNDARY_SAMPLES" in note
    assert "never" in note.lower()
    # the preferred, unambiguous field names are also present and agree numerically.
    assert intervals["unsupported_sample_run_extent_m"].iloc[0] == pytest.approx(
        intervals["span_length_m"].iloc[0]
    )
    assert intervals["unsupported_sample_run_start_chainage_m"].iloc[0] == pytest.approx(
        intervals["start_chainage_m"].iloc[0]
    )
    assert intervals["unsupported_sample_run_end_chainage_m"].iloc[0] == pytest.approx(
        intervals["end_chainage_m"].iloc[0]
    )


# --- 16.J: gap governance remains unchanged ------------------------------------------------------


def test_16j_gap_governance_unchanged_with_default_boundary_semantics():
    # identical scenario to test_8, but explicit about proving the (now-default,
    # UNSUPPORTED_SAMPLE_RUN_EXTENT) boundary semantics does not weaken gap governance.
    profile_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 10.0, 70.0, 80.0],
            "clearance_m": [0.6, 0.6, 0.6, 0.6],
            "measured_support_state": [support_state.MEASURED_UNSUPPORTED] * 4,
        }
    )
    intervals = support_state.extract_measured_free_span_intervals(
        profile_df,
        asset_id="TEST_ASSET",
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        classification_threshold_m=0.1,
        vertical_reference_semantics="TEST",
    )
    assert len(intervals) == 2
    assert (intervals["limitations"] == support_state.SPAN_SPLIT_BY_MEASUREMENT_GAP).all()
    assert (
        intervals["interval_boundary_semantics"] == support_state.UNSUPPORTED_SAMPLE_RUN_EXTENT
    ).all()


# --- 16.K: synthetic 3 spans / 100-10-10 m / 0.6 m / 1 new / 1 extended results unchanged -------


def test_16k_synthetic_case_numerical_results_unchanged():
    case = synthetic.build_synthetic_free_span_case()
    assert len(case.measured_intervals_df) == 3
    recovered_lengths = sorted(
        float(v) for v in case.measured_intervals_df["unsupported_sample_run_extent_m"]
    )
    assert recovered_lengths == pytest.approx([10.0, 10.0, 100.0])
    assert float(case.measured_intervals_df["maximum_clearance_m"].max()) == pytest.approx(0.6)
    assert case.scenario_result["new_span_count"] == 1
    assert case.scenario_result["extended_span_count"] == 1


# --- 9: supported/transition/unsupported states correct -----------------------------------------


def test_9_positive_clearance_beyond_threshold_is_unsupported():
    state = support_state.classify_measured_support_state(0.5, classification_threshold_m=0.1)
    assert state == support_state.MEASURED_UNSUPPORTED


def test_9b_negative_clearance_beyond_threshold_is_supported():
    state = support_state.classify_measured_support_state(-0.5, classification_threshold_m=0.1)
    assert state == support_state.MEASURED_SUPPORTED


def test_9c_clearance_within_threshold_band_is_transition():
    for clearance_value in (0.0, 0.05, -0.05, 0.1, -0.1):
        state = support_state.classify_measured_support_state(
            clearance_value, classification_threshold_m=0.1
        )
        assert state == support_state.MEASURED_SUPPORT_TRANSITION, clearance_value


def test_9d_classifier_has_no_source_interpretation_parameter():
    # MAR-025A Section 3: the generic measured-state classifier must not accept a source-
    # interpretation flag at all -- structurally, not just by convention.
    params = list(inspect.signature(support_state.classify_measured_support_state).parameters)
    assert params == ["clearance_m", "classification_threshold_m"]
    assert not any("source" in p.lower() for p in params)


def test_9e_source_interpreted_free_span_removed_from_measured_state_machine():
    assert not hasattr(support_state, "SOURCE_INTERPRETED_FREE_SPAN")
    assert "SOURCE_INTERPRETED_FREE_SPAN" not in support_state.MEASURED_SUPPORT_STATES


# --- 10 / MAR-025A 16.A-C: source interpretation is a separate, orthogonal field ---------------


def test_10a_source_flag_cannot_alter_measured_support_state():
    # Section 16.A: a clearance value that would geometrically be MEASURED_UNSUPPORTED must stay
    # MEASURED_UNSUPPORTED regardless of any source-interpretation bookkeeping -- there is no
    # parameter through which a source flag could reach this function at all (see test_9d).
    state_a = support_state.classify_measured_support_state(0.7, classification_threshold_m=0.1)
    state_b = support_state.classify_measured_support_state(0.7, classification_threshold_m=0.1)
    assert state_a == state_b == support_state.MEASURED_UNSUPPORTED


def test_10b_row_can_be_unsupported_and_source_interpreted_simultaneously():
    # Section 16.B: one row can simultaneously be geometrically unsupported AND carry source-
    # interpreted freespan evidence -- both truthfully recorded, neither erased.
    measured_state = support_state.classify_measured_support_state(
        0.7, classification_threshold_m=0.1
    )
    source_present = True
    assert measured_state == support_state.MEASURED_UNSUPPORTED
    assert source_present is True
    status = support_state.compute_geometry_vs_source_interpretation_status(
        measured_state, source_present
    )
    assert status == support_state.AGREES_UNSUPPORTED


def test_10c_row_can_be_supported_with_source_evidence_without_overwrite():
    # Section 16.C: one row can be geometrically supported while carrying source freespan
    # evidence -- a real evidence disagreement, neither value silently overwritten.
    measured_state = support_state.classify_measured_support_state(
        -0.7, classification_threshold_m=0.1
    )
    source_present = True
    assert measured_state == support_state.MEASURED_SUPPORTED
    assert source_present is True
    status = support_state.compute_geometry_vs_source_interpretation_status(
        measured_state, source_present
    )
    assert status == support_state.SOURCE_ONLY_FREESPAN_EVIDENCE


def test_10d_geometry_vs_source_status_never_alters_inputs():
    # the status function is purely descriptive -- calling it cannot change measured_support_state
    # or the source flag; re-deriving both from the same inputs always agrees.
    for clearance_value, threshold, source_present in (
        (0.7, 0.1, True),
        (0.7, 0.1, False),
        (-0.7, 0.1, True),
        (-0.7, 0.1, False),
        (0.0, 0.1, True),
        (0.5, None, True),
    ):
        state = support_state.classify_measured_support_state(
            clearance_value, classification_threshold_m=threshold
        )
        status = support_state.compute_geometry_vs_source_interpretation_status(
            state, source_present
        )
        assert status in support_state.GEOMETRY_VS_SOURCE_INTERPRETATION_STATUSES
        # recompute independently -- the status call must not have mutated anything.
        state_again = support_state.classify_measured_support_state(
            clearance_value, classification_threshold_m=threshold
        )
        assert state_again == state


def test_10e_unresolved_geometry_means_comparison_not_available():
    state = support_state.classify_measured_support_state(0.5, classification_threshold_m=None)
    assert state == support_state.MEASURED_SUPPORT_STATE_UNRESOLVED
    status = support_state.compute_geometry_vs_source_interpretation_status(state, True)
    assert status == support_state.COMPARISON_NOT_AVAILABLE
    status_no_flag = support_state.compute_geometry_vs_source_interpretation_status(state, False)
    assert status_no_flag == support_state.COMPARISON_NOT_AVAILABLE


def test_10f_no_evidence_at_all_status():
    state = support_state.classify_measured_support_state(-0.7, classification_threshold_m=0.1)
    status = support_state.compute_geometry_vs_source_interpretation_status(state, False)
    assert status == support_state.NO_FREESPAN_EVIDENCE


# --- 16.D/E: source flag inside/outside a run cannot split or create a measured span -----------


def test_16d_source_flag_inside_an_unsupported_run_does_not_split_the_measured_span():
    # MAR-025A Section 6 regression case: unsupported geometry at chainages 100, 125, 150, with
    # the middle 125 m sample ALSO carrying source-interpreted evidence in the caller's own
    # bookkeeping (irrelevant to this function, since `measured_support_state` never encodes it).
    profile_df = pd.DataFrame(
        {
            "chainage_m": [100.0, 125.0, 150.0],
            "clearance_m": [0.6, 0.6, 0.6],
            "measured_support_state": [support_state.MEASURED_UNSUPPORTED] * 3,
            "source_interpreted_free_span_present": [False, True, False],
        }
    )
    intervals = support_state.extract_measured_free_span_intervals(
        profile_df,
        asset_id="TEST_ASSET",
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        classification_threshold_m=0.1,
        vertical_reference_semantics="TEST",
    )
    assert len(intervals) == 1
    assert float(intervals["start_chainage_m"].iloc[0]) == pytest.approx(100.0)
    assert float(intervals["end_chainage_m"].iloc[0]) == pytest.approx(150.0)
    assert int(intervals["sample_count"].iloc[0]) == 3


def test_16d2_same_geometry_without_the_source_flag_produces_an_identical_interval():
    # proves the flag's presence/absence makes literally no difference to extraction.
    def _build(flagged_at_125: bool) -> pd.DataFrame:
        profile_df = pd.DataFrame(
            {
                "chainage_m": [100.0, 125.0, 150.0],
                "clearance_m": [0.6, 0.6, 0.6],
                "measured_support_state": [support_state.MEASURED_UNSUPPORTED] * 3,
                "source_interpreted_free_span_present": [False, flagged_at_125, False],
            }
        )
        return support_state.extract_measured_free_span_intervals(
            profile_df,
            asset_id="TEST_ASSET",
            max_measurement_gap_m=50.0,
            survey_epoch="TEST",
            classification_threshold_m=0.1,
            vertical_reference_semantics="TEST",
        )

    with_flag = _build(True)
    without_flag = _build(False)
    compare_cols = ["start_chainage_m", "end_chainage_m", "span_length_m", "sample_count"]
    pd.testing.assert_frame_equal(
        with_flag[compare_cols].reset_index(drop=True),
        without_flag[compare_cols].reset_index(drop=True),
    )


def test_16e_source_flag_alone_cannot_create_a_measured_span():
    # a row that is geometrically MEASURED_SUPPORTED, even with source evidence present, must
    # never appear in the extracted intervals -- extraction only ever looks at
    # measured_support_state, which this row's flag cannot touch.
    profile_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 10.0, 20.0],
            "clearance_m": [-0.5, -0.5, -0.5],
            "measured_support_state": [support_state.MEASURED_SUPPORTED] * 3,
            "source_interpreted_free_span_present": [True, True, True],
        }
    )
    intervals = support_state.extract_measured_free_span_intervals(
        profile_df,
        asset_id="TEST_ASSET",
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        classification_threshold_m=0.1,
        vertical_reference_semantics="TEST",
    )
    assert intervals.empty


def test_10c_synthetic_case_source_interpreted_field_is_preserved_and_orthogonal():
    case = synthetic.build_synthetic_free_span_case()
    source_flagged = case.profile_df[case.profile_df["source_interpreted_free_span_present"]]
    assert len(source_flagged) == 2

    # the sample inside the existing unsupported run: geometry unaffected, still counted.
    inside_run = source_flagged[abs(source_flagged["chainage_m"] - 200.0) < 1e-6].iloc[0]
    assert inside_run["measured_support_state"] == support_state.MEASURED_UNSUPPORTED
    assert (
        inside_run["geometry_vs_source_interpretation_status"] == support_state.AGREES_UNSUPPORTED
    )
    matched = [
        interval
        for _, interval in case.measured_intervals_df.iterrows()
        if interval["start_chainage_m"] <= 200.0 <= interval["end_chainage_m"]
    ]
    assert len(matched) == 1
    assert int(matched[0]["sample_count"]) == 5  # unchanged: still the full 150-250 run

    # the isolated geometrically-supported sample: never enters a measured interval.
    isolated = source_flagged[abs(source_flagged["chainage_m"] - 320.0) < 1e-6].iloc[0]
    assert isolated["measured_support_state"] == support_state.MEASURED_SUPPORTED
    assert (
        isolated["geometry_vs_source_interpretation_status"]
        == support_state.SOURCE_ONLY_FREESPAN_EVIDENCE
    )
    for _, interval in case.measured_intervals_df.iterrows():
        assert not (interval["start_chainage_m"] <= 320.0 <= interval["end_chainage_m"])


# --- 11/12: lowering scenario creates a new span and extends an existing span ------------------


def test_11_lowering_scenario_creates_expected_new_span():
    case = synthetic.build_synthetic_free_span_case()
    assert case.scenario_result["new_span_count"] == synthetic.EXPECTED_NEW_SPAN_COUNT
    new_intervals = case.scenario_result["scenario_intervals_df"]
    new_rows = new_intervals[new_intervals["vs_baseline_classification"] == scenario.NEW]
    assert len(new_rows) == 1
    assert float(new_rows["start_chainage_m"].iloc[0]) == pytest.approx(600.0)
    assert float(new_rows["end_chainage_m"].iloc[0]) == pytest.approx(610.0)


def test_12_lowering_scenario_extends_expected_existing_span():
    case = synthetic.build_synthetic_free_span_case()
    assert case.scenario_result["extended_span_count"] == synthetic.EXPECTED_EXTENDED_SPAN_COUNT
    scenario_intervals = case.scenario_result["scenario_intervals_df"]
    extended_rows = scenario_intervals[
        scenario_intervals["vs_baseline_classification"] == scenario.EXTENDED
    ]
    assert len(extended_rows) == 1
    # the baseline measured span was 150-250 -- the scenario version must be strictly larger.
    baseline_mask = (case.measured_intervals_df["start_chainage_m"] - 150.0).abs() < 1e-6
    baseline_span = case.measured_intervals_df[baseline_mask].iloc[0]
    extended_row = extended_rows.iloc[0]
    assert float(extended_row["start_chainage_m"]) <= float(baseline_span["start_chainage_m"])
    assert float(extended_row["end_chainage_m"]) >= float(baseline_span["end_chainage_m"])
    assert float(extended_row["span_length_m"]) > float(baseline_span["span_length_m"])


def test_11b_classify_scenario_interval_vs_measured_all_three_outcomes():
    measured = pd.DataFrame({"start_chainage_m": [100.0], "end_chainage_m": [200.0]})
    assert (
        scenario.classify_scenario_interval_vs_measured(100.0, 200.0, measured)
        == scenario.UNCHANGED
    )
    assert (
        scenario.classify_scenario_interval_vs_measured(90.0, 200.0, measured) == scenario.EXTENDED
    )
    assert scenario.classify_scenario_interval_vs_measured(500.0, 510.0, measured) == scenario.NEW


# --- 16.F: source flag cannot alter NEW/EXTENDED/UNCHANGED scenario classification --------------


def test_16f_source_flag_does_not_alter_scenario_classification():
    # scenario screening reads only measured_support_state and clearance -- a source-
    # interpretation column present (or absent) on the same geometry must produce byte-identical
    # scenario results.
    lowering_input = scenario.SeabedLoweringScenarioInput(0.30, scenario.ENGINEERING_SCENARIO)

    def _profile(source_flags: list[bool]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "chainage_m": [0.0, 10.0, 20.0, 30.0],
                "clearance_m": [-0.15, -0.15, 0.6, 0.6],
                "measured_support_state": [
                    support_state.MEASURED_SUPPORTED,
                    support_state.MEASURED_SUPPORTED,
                    support_state.MEASURED_UNSUPPORTED,
                    support_state.MEASURED_UNSUPPORTED,
                ],
                "source_interpreted_free_span_present": source_flags,
            }
        )

    measured_intervals = support_state.extract_measured_free_span_intervals(
        _profile([False, False, False, False]),
        asset_id="TEST_ASSET",
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        classification_threshold_m=0.1,
        vertical_reference_semantics="TEST",
    )

    result_with_flags = scenario.build_support_loss_scenario_profile(
        _profile([True, True, True, True]),
        asset_id="TEST_ASSET",
        lowering_input=lowering_input,
        classification_threshold_m=0.1,
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        vertical_reference_semantics="TEST",
        measured_intervals_df=measured_intervals,
    )
    result_without_flags = scenario.build_support_loss_scenario_profile(
        _profile([False, False, False, False]),
        asset_id="TEST_ASSET",
        lowering_input=lowering_input,
        classification_threshold_m=0.1,
        max_measurement_gap_m=50.0,
        survey_epoch="TEST",
        vertical_reference_semantics="TEST",
        measured_intervals_df=measured_intervals,
    )
    assert result_with_flags["new_span_count"] == result_without_flags["new_span_count"]
    assert result_with_flags["extended_span_count"] == result_without_flags["extended_span_count"]
    compare_cols = ["start_chainage_m", "end_chainage_m", "vs_baseline_classification"]
    pd.testing.assert_frame_equal(
        result_with_flags["scenario_intervals_df"][compare_cols].reset_index(drop=True),
        result_without_flags["scenario_intervals_df"][compare_cols].reset_index(drop=True),
    )


def test_16f2_synthetic_scenario_results_unaffected_by_source_flag_presence():
    # the synthetic case itself carries the flag on two samples (chainage 200, 320) -- confirm
    # the scenario counts match the hand-derived expectation regardless.
    case = synthetic.build_synthetic_free_span_case()
    assert case.scenario_result["new_span_count"] == synthetic.EXPECTED_NEW_SPAN_COUNT
    assert case.scenario_result["extended_span_count"] == synthetic.EXPECTED_EXTENDED_SPAN_COUNT


# --- 13: no pipe vertical motion silently introduced --------------------------------------------


def test_13_scenario_clearance_function_has_no_pipe_elevation_parameter():
    params = list(
        inspect.signature(scenario.compute_scenario_pipe_underside_clearance_m).parameters
    )
    assert params == ["current_clearance_m", "seabed_lowering_m"]
    assert not any("pipe" in p.lower() for p in params)


def test_13b_scenario_clearance_equals_current_clearance_plus_lowering():
    result = scenario.compute_scenario_pipe_underside_clearance_m(-0.15, 0.30)
    assert result == pytest.approx(0.15)


def test_13c_seabed_lowering_input_rejects_negative_magnitude():
    with pytest.raises(ValueError):
        scenario.SeabedLoweringScenarioInput(-0.1, scenario.ENGINEERING_SCENARIO)


def test_13d_seabed_lowering_input_rejects_unknown_evidence_type():
    with pytest.raises(ValueError):
        scenario.SeabedLoweringScenarioInput(0.3, "SOME_MADE_UP_EVIDENCE_TYPE")


def test_13e_no_lowering_scenario_screening_result():
    result = scenario.screen_support_loss_susceptibility(0.5, None, 0.1)
    assert result["screening_state"] == scenario.SUPPORT_LOSS_SCREENING_NOT_AVAILABLE


# --- 14: PL854 susceptibility unavailable --------------------------------------------------------


def test_14_pl854_susceptibility_and_structural_assessment_are_hardcoded_no():
    result = cli._derive_free_span_poc_validation_questions(
        pipe_bottom_normalization_demonstrated=True,
        clearance_computable=True,
        defensible_intervals_extracted=True,
        unsurveyed_gaps_prevented_from_joining=True,
        scenario_creates_and_extends_support_loss=True,
        real_nsta_evidence_ingested=True,
    )
    assert result["question_g_pl854_site_specific_free_span_susceptibility_map_defensible"] == "NO"
    assert result["question_h_structural_dnv_rp_f105_viv_fatigue_assessment_performed"] == "NO"
    assert result["question_i_future_failure_probability_produced"] == "NO"
    assert result["question_a_pipe_vertical_reference_normalized_to_pipe_bottom_elevation"] == "YES"
    assert (
        result["question_f_real_authoritative_nsta_observed_free_span_evidence_ingested"] == "YES"
    )


# --- 15: NSTA evidence never feeds synthetic susceptibility physics ----------------------------


def test_15_synthetic_case_builder_takes_no_arguments():
    params = inspect.signature(synthetic.build_synthetic_free_span_case).parameters
    assert len(params) == 0


def test_15b_synthetic_module_never_imports_nsta_machinery():
    # a bare "nsta" substring check is unsafe (the English word "constant" contains it) --
    # check the actual import graph instead: synthetic.py must never import any NSTA module.
    assert "nsta_freespan" not in synthetic.__dict__
    assert "nsta_registry" not in synthetic.__dict__
    imported_module_names = {
        getattr(v, "__name__", "") for v in vars(synthetic).values() if inspect.ismodule(v)
    }
    assert not any("nsta" in name.lower() for name in imported_module_names)


# --- 16/17/18: no VIV/fatigue/ULS/FLS, no risk score, no failure probability -------------------


def test_16_no_viv_fatigue_uls_fls_vocabulary_in_the_generic_engine():
    forbidden = ("VIV", "FATIGUE", "ULS", "FLS")
    for module in VOCABULARY_SCAN_MODULES:
        module_members = vars(module)
        for name, value in module_members.items():
            if name.isupper() and isinstance(value, str):
                for word in forbidden:
                    assert word not in value, f"{module.__name__}.{name} = {value!r}"


def test_17_no_risk_score_vocabulary_in_the_generic_engine():
    all_states = (
        support_state.MEASURED_SUPPORT_STATES
        | scenario.SUPPORT_LOSS_SUSCEPTIBILITY_STATES
        | reference.PIPE_VERTICAL_REFERENCES
        | scenario.ALLOWED_LOWERING_EVIDENCE_TYPES
        | clearance.THRESHOLD_PROVENANCE_TYPES
    )
    forbidden = ("SAFE", "UNSAFE", "HIGH RISK", "LOW RISK", "RISK_SCORE")
    for state in all_states:
        for word in forbidden:
            assert word not in state
    assert "risk_score" not in " ".join(support_state.FREE_SPAN_INTERVAL_COLUMNS)
    assert "score" not in " ".join(support_state.FREE_SPAN_INTERVAL_COLUMNS)


def test_18_no_failure_probability_vocabulary_in_the_generic_engine():
    all_states = (
        support_state.MEASURED_SUPPORT_STATES
        | scenario.SUPPORT_LOSS_SUSCEPTIBILITY_STATES
        | reference.PIPE_VERTICAL_REFERENCES
        | scenario.ALLOWED_LOWERING_EVIDENCE_TYPES
    )
    for state in all_states:
        assert "PROBABILITY" not in state
    assert "probability" not in " ".join(support_state.FREE_SPAN_INTERVAL_COLUMNS).lower()


def test_18b_no_free_span_prediction_or_ml_wording_in_generic_engine():
    for module in STRICT_NO_MENTION_MODULES:
        source = inspect.getsource(module).lower()
        assert "machine learning" not in source
        assert "allowable span" not in source


# --- 19: generic engine contains no PL854/NSTA coordinates --------------------------------------


def test_19_generic_engine_has_no_hardcoded_dataset_coordinates():
    # PL854's own real route bbox corner (configs/pl854.yaml) must never appear as a literal
    # value in the generic engine modules.
    forbidden_fragments = ("1.6516061", "53.3678022", "2.0019739", "53.3892292")
    for module in GENERIC_ENGINE_MODULES:
        source = inspect.getsource(module)
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{module.__name__} hard-codes {fragment!r}"


def test_19b_no_dataset_name_string_in_generic_engine_logic():
    for module in GENERIC_ENGINE_MODULES:
        source = inspect.getsource(module)
        assert "PL854" not in source
        assert "PL855" not in source
        assert "NSTA" not in source


# --- 20: offline rerun works (cache hit, no network) --------------------------------------------


def test_20_full_registry_cache_hit_never_touches_network(tmp_path, monkeypatch):
    def _explode(*_args, **_kwargs):
        raise AssertionError("query_freespan_layer must not be called on a cache hit")

    from marine_engine.providers import nsta_freespan

    monkeypatch.setattr(nsta_freespan, "query_freespan_layer", _explode)

    cache_dir = tmp_path / "raw" / "nsta" / "freespans_full_registry"
    cache_dir.mkdir(parents=True)
    stub_payload = {"type": "FeatureCollection", "features": []}
    for filename in nsta_registry.RAW_CACHE_FILENAMES.values():
        (cache_dir / filename).write_text(json.dumps(stub_payload), encoding="utf-8")

    results = nsta_registry.acquire_full_registry_cached(cache_dir)
    assert len(results) == 2
    assert all(info["already_cached"] for info in results.values())


# --- CLI wiring + validation-question derivation -------------------------------------------------


def test_cli_registers_build_free_span_poc():
    parser = cli.build_parser()
    args = parser.parse_args(["build-free-span-poc", "configs/pl854.yaml"])
    assert args.func is cli._cmd_build_free_span_poc


# --- contract, report, structural handoff --------------------------------------------------------


def test_input_contract_states_lowering_input_required_for_susceptibility():
    built = contract.build_free_span_input_contract()
    fields = [f["field"] for f in built["required_fields_support_loss_susceptibility"]]
    assert "seabed_lowering_input" in fields
    assert (
        built["required_fields_support_loss_susceptibility"]
        != built["required_fields_measured_geometry"]
    )


def test_structural_handoff_contract_is_explicit_not_implemented():
    built = structural_handoff.build_structural_free_span_assessment_handoff_contract()
    assert built["explicit_statement"] == "NOT IMPLEMENTED IN MAR-025"
    assert (
        built["structural_assessment_status"]
        == structural_handoff.STRUCTURAL_FREE_SPAN_ASSESSMENT_NOT_PERFORMED
    )
    assert len(built["required_structural_inputs"]) > 0


def test_report_blocks_render_to_html():
    blocks = report.build_free_span_report_blocks(
        project_title="Test",
        purpose_text="purpose",
        product_boundary_text="boundary",
        operator_input_model_facts={"a": 1},
        canonical_geometry_facts={"b": 2},
        measured_free_span_facts={"c": 3},
        gap_governance_text="gap text",
        support_loss_scenario_facts={"d": 4},
        synthetic_validation_facts={"e": 5},
        nsta_registry_facts={"f": 6},
        pl854_observed_context_facts={"g": 7},
        structural_boundary_text="structural text",
        production_transfer_contract_summary=["contract item"],
        limitations=["limitation item"],
    )
    html = report.render_blocks_html(blocks, title="Test Report")
    assert "<html" in html
    assert "DOES NOT ASSESS VIV" in html


# --- margin (Section 15) --------------------------------------------------------------------------


def test_margin_is_distance_past_the_classification_boundary():
    assert clearance.compute_support_clearance_margin_m(0.5, 0.1) == pytest.approx(0.4)
    assert clearance.compute_support_clearance_margin_m(0.05, 0.1) == pytest.approx(-0.05)
    assert clearance.compute_support_clearance_margin_m(None, 0.1) is None
    assert clearance.compute_support_clearance_margin_m(0.5, None) is None


# --- seabed canonicalization (Section 4) -------------------------------------------------------


def test_seabed_canonicalization_reuses_accepted_terrain_convention():
    import numpy as np

    already_elevation = clearance.canonicalize_seabed_support_elevation_m(
        np.array([-9.5, -10.6]), source_sign_convention=clearance.ALREADY_ELEVATION_STYLE
    )
    assert already_elevation.tolist() == pytest.approx([-9.5, -10.6])

    positive_down = clearance.canonicalize_seabed_support_elevation_m(
        np.array([9.5, 10.6]), source_sign_convention=clearance.POSITIVE_DOWN_DEPTH
    )
    assert positive_down.tolist() == pytest.approx([-9.5, -10.6])


def test_seabed_canonicalization_rejects_unknown_convention():
    import numpy as np

    with pytest.raises(ValueError):
        clearance.canonicalize_seabed_support_elevation_m(
            np.array([1.0]), source_sign_convention="MADE_UP_CONVENTION"
        )


# --- NSTA registry audit (Sections 18-20) ------------------------------------------------------


def _synthetic_nsta_records() -> list[dict]:
    from shapely.geometry import Point

    return [
        {
            "registry_layer": nsta_registry.nsta_freespan.CURRENT_REGISTRY_LAYER,
            "feature_id": "F1",
            "nsta_pipeline_number": "PL0001",
            "pipe_name": "TEST LINE",
            "freespanno": 1,
            "length_m": 12.5,
            "mxheight_m": 0.3,
            "comments": None,
            "start_date": 1500000000000,
            "end_date": None,
            "upd_date": None,
            "survey_id": "S1",
            "_geometry_wgs84": Point(1.0, 55.0),
        },
        {
            "registry_layer": nsta_registry.nsta_freespan.CURRENT_REGISTRY_LAYER,
            "feature_id": "F1",  # duplicated feature_id, deliberately
            "nsta_pipeline_number": "PL0001",
            "pipe_name": "TEST LINE",
            "freespanno": 2,
            "length_m": -1,  # invalid (non-positive)
            "mxheight_m": None,
            "comments": None,
            "start_date": None,
            "end_date": None,
            "upd_date": None,
            "survey_id": None,
            "_geometry_wgs84": None,
        },
        {
            "registry_layer": nsta_registry.nsta_freespan.REMOVED_REGISTRY_LAYER,
            "feature_id": "F2",
            "nsta_pipeline_number": None,  # missing pipeline id, deliberately
            "pipe_name": None,
            "freespanno": 3,
            "length_m": 5.0,
            "mxheight_m": 0.1,
            "comments": "test",
            "start_date": None,
            "end_date": None,
            "upd_date": None,
            "survey_id": None,
            "_geometry_wgs84": Point(2.0, 56.0),
        },
    ]


def test_nsta_registry_audit_df_and_summary_are_descriptive_only():
    records = _synthetic_nsta_records()
    audit_df = nsta_registry.build_nsta_registry_audit_df(records)
    assert len(audit_df) == 3
    summary = nsta_registry.summarize_registry_audit(audit_df)
    assert summary["total_record_count"] == 3
    assert summary["total_current_records"] == 2
    assert summary["total_removed_records"] == 1
    assert summary["unique_pipeline_id_count"] == 1
    assert summary["valid_length_m_count"] == 2  # the -1 length is invalid
    assert summary["valid_mxheight_m_count"] == 2  # record 2's None mxheight_m is the only gap
    assert summary["records_missing_pipeline_id_count"] == 1
    # no quality score anywhere in the summary
    assert not any("score" in key.lower() for key in summary)


# --- 16.L: NSTA audit distinct duplicate-ID count and duplicate-row count are not conflated ----


def test_16l_duplicated_feature_id_distinct_count_vs_records_with_duplicated_feature_id_count():
    # the fixture has feature_id "F1" appearing twice (one distinct duplicated ID, two rows) and
    # "F2" appearing once (not a duplicate at all) -- the two counts must differ and neither may
    # be silently reported as the other.
    records = _synthetic_nsta_records()
    audit_df = nsta_registry.build_nsta_registry_audit_df(records)
    summary = nsta_registry.summarize_registry_audit(audit_df)
    assert summary["duplicated_feature_id_distinct_count"] == 1  # only "F1"
    assert summary["records_with_duplicated_feature_id_count"] == 2  # both F1 rows
    assert (
        summary["duplicated_feature_id_distinct_count"]
        != summary["records_with_duplicated_feature_id_count"]
    )
    assert "duplicated_feature_id_count" not in summary  # the old, ambiguous key is gone


def test_16l2_no_duplicates_yields_zero_for_both_counts():
    records = [_synthetic_nsta_records()[2]]  # F2 only, appears once
    audit_df = nsta_registry.build_nsta_registry_audit_df(records)
    summary = nsta_registry.summarize_registry_audit(audit_df)
    assert summary["duplicated_feature_id_distinct_count"] == 0
    assert summary["records_with_duplicated_feature_id_count"] == 0


# --- 16.M: no hard-coded pipeline-count prose exists in generic registry module ----------------


def test_16m_no_hard_coded_pipeline_count_prose_in_nsta_registry_module():
    source = inspect.getsource(nsta_registry)
    assert "222" not in source


def test_16m2_no_hard_coded_pipeline_count_prose_in_maps_module():
    source = inspect.getsource(maps)
    assert "222" not in source


def test_nsta_registry_empty_records_produce_valid_empty_outputs():
    audit_df = nsta_registry.build_nsta_registry_audit_df([])
    assert audit_df.empty
    summary = nsta_registry.summarize_registry_audit(audit_df)
    assert summary["total_record_count"] == 0
    gdf = nsta_registry.build_nsta_registry_gdf([])
    assert gdf.empty


def test_select_example_pipeline_by_record_count():
    records = _synthetic_nsta_records() + [
        {**_synthetic_nsta_records()[2], "feature_id": "F3", "nsta_pipeline_number": "PL0002"}
    ]
    audit_df = nsta_registry.build_nsta_registry_audit_df(records)
    top_pipeline, top_count = nsta_registry.select_example_pipeline_by_record_count(audit_df)
    assert top_pipeline == "PL0001"
    assert top_count == 2


def test_select_example_pipeline_empty_audit_returns_none():
    empty_df = nsta_registry.build_nsta_registry_audit_df([])
    top_pipeline, top_count = nsta_registry.select_example_pipeline_by_record_count(empty_df)
    assert top_pipeline is None
    assert top_count == 0


# --- maps render without error --------------------------------------------------------------


def test_synthetic_maps_render(tmp_path):
    case = synthetic.build_synthetic_free_span_case()
    map_path = maps.render_free_span_support_map(
        route=case.route,
        profile_df=case.profile_df,
        measured_intervals_df=case.measured_intervals_df,
        scenario_intervals_df=case.scenario_result["scenario_intervals_df"],
        output_path=tmp_path / "map.png",
    )
    assert map_path.exists()
    assert map_path.stat().st_size > 0

    kp_path = maps.render_free_span_kp_view(
        profile_df=case.scenario_result["profile_df"],
        measured_intervals_df=case.measured_intervals_df,
        scenario_intervals_df=case.scenario_result["scenario_intervals_df"],
        classification_threshold_m=synthetic.CLASSIFICATION_THRESHOLD_M,
        seabed_lowering_m=case.lowering_input.seabed_lowering_m,
        output_path=tmp_path / "kp_view.png",
    )
    assert kp_path.exists()
    assert kp_path.stat().st_size > 0


def test_nsta_overview_map_renders_with_empty_registry(tmp_path):
    gdf = nsta_registry.build_nsta_registry_gdf([])
    map_path = maps.render_nsta_registry_overview_map(
        registry_gdf=gdf,
        example_pipeline_number=None,
        example_record_count=0,
        output_path=tmp_path / "nsta_map.png",
    )
    assert map_path.exists()
    assert map_path.stat().st_size > 0


# --- no free-span structural prediction anywhere (mirrors MAR-024's precedent) -----------------


def test_no_free_span_structural_prediction_wording_in_operational_modules():
    for module in STRICT_NO_MENTION_MODULES:
        source = inspect.getsource(module).lower()
        assert "dnv" not in source

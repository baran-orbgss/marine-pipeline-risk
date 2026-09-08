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
    state = support_state.classify_measured_support_state(
        0.6, is_source_interpreted_free_span=False, classification_threshold_m=None
    )
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


# --- 9: supported/transition/unsupported states correct -----------------------------------------


def test_9_positive_clearance_beyond_threshold_is_unsupported():
    state = support_state.classify_measured_support_state(
        0.5, is_source_interpreted_free_span=False, classification_threshold_m=0.1
    )
    assert state == support_state.MEASURED_UNSUPPORTED


def test_9b_negative_clearance_beyond_threshold_is_supported():
    state = support_state.classify_measured_support_state(
        -0.5, is_source_interpreted_free_span=False, classification_threshold_m=0.1
    )
    assert state == support_state.MEASURED_SUPPORTED


def test_9c_clearance_within_threshold_band_is_transition():
    for clearance_value in (0.0, 0.05, -0.05, 0.1, -0.1):
        state = support_state.classify_measured_support_state(
            clearance_value, is_source_interpreted_free_span=False, classification_threshold_m=0.1
        )
        assert state == support_state.MEASURED_SUPPORT_TRANSITION, clearance_value


# --- 10: source-interpreted span remains separate from measured inference ----------------------


def test_10_source_interpreted_span_overrides_geometric_label():
    # a clearance value that would geometrically be MEASURED_UNSUPPORTED on its own must become
    # SOURCE_INTERPRETED_FREE_SPAN when explicitly flagged -- never silently counted as a
    # geometric measured span.
    state = support_state.classify_measured_support_state(
        0.7, is_source_interpreted_free_span=True, classification_threshold_m=0.1
    )
    assert state == support_state.SOURCE_INTERPRETED_FREE_SPAN
    assert state != support_state.MEASURED_UNSUPPORTED


def test_10b_source_interpreted_sample_never_enters_measured_intervals():
    profile_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 10.0, 20.0],
            "clearance_m": [0.7, 0.7, 0.7],
            "measured_support_state": [
                support_state.SOURCE_INTERPRETED_FREE_SPAN,
                support_state.SOURCE_INTERPRETED_FREE_SPAN,
                support_state.SOURCE_INTERPRETED_FREE_SPAN,
            ],
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


def test_10c_synthetic_case_source_interpreted_sample_excluded_from_measured_intervals():
    case = synthetic.build_synthetic_free_span_case()
    source_flagged = case.profile_df[case.profile_df["is_source_interpreted_free_span"]]
    assert len(source_flagged) == 1
    source_chainage = float(source_flagged["chainage_m"].iloc[0])
    for _, interval in case.measured_intervals_df.iterrows():
        assert not (interval["start_chainage_m"] <= source_chainage <= interval["end_chainage_m"])


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
    assert summary["duplicated_feature_id_count"] == 2  # both F1 rows
    assert summary["records_missing_pipeline_id_count"] == 1
    # no quality score anywhere in the summary
    assert not any("score" in key.lower() for key in summary)


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

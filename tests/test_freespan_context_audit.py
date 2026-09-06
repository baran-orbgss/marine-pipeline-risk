"""Offline unit tests for marine_engine.validation.freespan_context_audit (MAR-015).

Small hand-built synthetic segment/event tables only -- never the real PL854
outputs, never network access. Lettered comments map to MAR-015 Section 27's
required test list (A-K, P here; L/M/N/O live in
test_freespan_context_audit_map.py alongside the map renderers they exercise).
"""

import pandas as pd
import pytest

from marine_engine.validation import freespan_context_audit as fca

# --- Shared synthetic fixtures ----------------------------------------------------------

_BASE_SECTION_DEFAULTS = {
    "start_chainage_m": 0.0,
    "end_chainage_m": 1000.0,
    "kp_start": "KP 0+000",
    "kp_end": "KP 1+000",
    "section_length_m": 1000.0,
    "observed_2018_event_count": 0,
    "observed_2018_total_length_m": 0.0,
    "observed_2018_max_length_m": 0.0,
    "observed_2018_max_height_m": 0.0,
    "observation_status": fca.NO_EVENT_STATUS,
    "tau_max_p95_sensitivity_min_pa": 0.2,
    "tau_max_p95_sensitivity_width_pa": 0.3,
    "current_reference_speed_p95_m_s": 0.3,
    "orbital_rms_p95_m_s": 0.2,
    "tau_max_p95_sensitivity_max_pa": 0.5,
    "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": 0.5,
    "p95_required_embedment_upper_class": "0.03",
    "local_relief_1000m_median_m": 0.5,
    "slope_500m_median_deg": 1.0,
    "mapped_250k_folk_class": "SAND",
    "nearest_valid_psa_d50_mm": 0.25,
}


def _make_section_df(rows: list[dict]) -> pd.DataFrame:
    records = []
    for i, overrides in enumerate(rows):
        record = dict(_BASE_SECTION_DEFAULTS)
        record["hydro_pair_id"] = f"pair_{chr(65 + i)}"
        record["segment_id"] = f"seg_{i}"
        record.update(overrides)
        records.append(record)
    return pd.DataFrame(records)


def _event_row(**overrides) -> dict:
    base = {
        "event_id": "2018-01",
        "canonical_kp_min": "KP 0+100",
        "canonical_kp_max": "KP 0+105",
        "canonical_mid_kp": "KP 0+102",
        "canonical_mid_chainage_m": 102.0,
        "source_length_m": 10.0,
        "source_height_m": 0.2,
        "source_comment": "synthetic test event",
        "asset_scope": "PL854_PL855_PIGGYBACK_CORRIDOR",
        "individual_line_attribution": "UNRESOLVED",
    }
    base.update(overrides)
    return base


_EMPTY_TEMPORAL_RELATIONSHIP_DF = pd.DataFrame(
    columns=["event_id_b", "event_id_a", "relationship_type", "survey_year_a", "source_statement"]
)


def _all_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_id": "2014-01", "source_length_m": 8.0},
            {"event_id": "2018-01", "source_length_m": 10.0},
        ]
    )


def _real_pipeline_section_df() -> pd.DataFrame:
    """Section-level table built via the REAL merge + build path (Section 4's
    shared-support-grid join), not the hand-built `_make_section_df` shortcut."""

    current_df = pd.DataFrame(
        {
            "segment_id": ["seg_0", "seg_1"],
            "current_node_id": ["cnode_0", "cnode_1"],
            "current_reference_speed_p95_m_s": [0.3, 0.9],
            "source_grid_nominal_resolution_m": [1500.0, 1500.0],
        }
    )
    wave_df = pd.DataFrame(
        {
            "segment_id": ["seg_0", "seg_1"],
            "wave_node_id": ["wnode_0", "wnode_1"],
            "orbital_rms_p95_m_s": [0.2, 0.4],
        }
    )
    combined_df = pd.DataFrame(
        {
            "segment_id": ["seg_0", "seg_1"],
            "start_chainage_m": [0.0, 1000.0],
            "end_chainage_m": [1000.0, 2000.0],
            "kp_start": ["KP 0+000", "KP 1+000"],
            "kp_end": ["KP 1+000", "KP 2+000"],
            "hydro_pair_id": ["pair_A", "pair_B"],
            "tau_max_p95_sensitivity_min_pa": [0.2, 0.6],
            "tau_max_p95_sensitivity_max_pa": [0.5, 1.2],
            "tau_max_p95_sensitivity_width_pa": [0.3, 0.6],
        }
    )
    mobility_df = pd.DataFrame(
        {
            "segment_id": ["seg_0", "seg_1"],
            "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": [0.5, 1.0],
            "mapped_250k_folk_class": ["SAND", "MUD"],
            "nearest_valid_psa_d50_mm": [0.25, 0.18],
            "nearest_valid_psa_distance_m": [50.0, 80.0],
            "nearest_valid_psa_sample_year": [1992, 1995],
        }
    )
    scour_df = pd.DataFrame(
        {
            "segment_id": ["seg_0", "seg_1"],
            "p95_required_embedment_lower_class": ["0", "0"],
            "p95_required_embedment_upper_class": ["0.03", "0.03"],
            "slope_500m_median_deg": [1.0, 1.5],
            "slope_1000m_median_deg": [1.1, 1.6],
            "tpi_1000m_median_m": [0.05, 0.06],
            "local_relief_1000m_median_m": [0.5, 0.6],
            "terrain_std_1000m_median_m": [0.2, 0.3],
        }
    )
    merged = fca.merge_all_segment_tables(
        current_segments_gdf=current_df,
        wave_segments_gdf=wave_df,
        combined_segments_gdf=combined_df,
        mobility_segments_gdf=mobility_df,
        scour_segments_gdf=scour_df,
    )
    counts_df = pd.DataFrame(
        {
            "hydro_pair_id": ["pair_A", "pair_B"],
            "freespan_2018_count": [1, 0],
            "freespan_2018_total_length_m": [9.33, 0.0],
            "freespan_2018_max_height_m": [0.13, 0.0],
            "freespan_2018_max_length_m": [9.33, 0.0],
            "any_2018_freespan": [True, False],
        }
    )
    return fca.build_section_level_context_table(merged, counts_df)


def _build_audit_pieces(section_df: pd.DataFrame):
    contrasts_by_key = {
        key: fca.compute_descriptive_contrast(section_df, key)
        for key in fca.AUDIT_TABLE_FEATURE_KEYS
        if not fca.FEATURE_BY_KEY[key].is_categorical
    }
    categorical_audits_by_key = {
        key: fca.compute_categorical_audit(section_df, key)
        for key in fca.AUDIT_TABLE_FEATURE_KEYS
        if fca.FEATURE_BY_KEY[key].is_categorical or key == "mobility_capacity"
    }
    readiness = fca.build_evidence_readiness(categorical_audits_by_key)
    return contrasts_by_key, categorical_audits_by_key, readiness


# --- A/B: 8 (here: 1) events remain corridor-level, never rewritten to a PL854-only label ---


def test_A_event_context_preserves_corridor_level_asset_scope_and_attribution():
    section_df = _make_section_df(
        [{"observation_status": fca.EVENT_PRESENT_STATUS, "observed_2018_event_count": 1}]
    )
    event_df = fca.build_event_level_context_table(
        pd.DataFrame([_event_row()]), section_df, _EMPTY_TEMPORAL_RELATIONSHIP_DF, _all_events_df()
    )
    row = event_df.iloc[0]
    assert row["asset_scope"] == "PL854_PL855_PIGGYBACK_CORRIDOR"
    assert row["individual_line_attribution"] == "UNRESOLVED"
    assert row["observation_role"] == fca.OBSERVATION_ROLE
    assert fca.OBSERVATION_ROLE == "CORRIDOR_LEVEL_OBSERVED_FREESPAN_EVENTS"


def test_B_event_context_never_creates_a_pl854_only_or_pl855_only_column_or_value():
    section_df = _make_section_df(
        [{"observation_status": fca.EVENT_PRESENT_STATUS, "observed_2018_event_count": 1}]
    )
    event_df = fca.build_event_level_context_table(
        pd.DataFrame([_event_row()]), section_df, _EMPTY_TEMPORAL_RELATIONSHIP_DF, _all_events_df()
    )
    for column in event_df.columns:
        assert "pl854_only" not in column.lower()
        assert "pl855_only" not in column.lower()
        assert "positive_label" not in column.lower()
    assert event_df.iloc[0]["asset_scope"] != "PL854"
    assert event_df.iloc[0]["asset_scope"] != "PL855"


# --- C: non-event sections are labelled with the absence-of-tabulation status, never ------
# --- a negative/control/safe/stable ground-truth label ------------------------------------


def test_C_non_event_sections_never_get_a_negative_or_control_label():
    section_df = _real_pipeline_section_df()
    assert set(section_df["observation_status"]) <= {fca.EVENT_PRESENT_STATUS, fca.NO_EVENT_STATUS}

    no_event_row = section_df[section_df["hydro_pair_id"] == "pair_B"].iloc[0]
    assert no_event_row["observation_status"] == fca.NO_EVENT_STATUS
    assert no_event_row["observation_status"] == "NO_TABULATED_2018_EVENT_IN_THIS_SUPPORT_SECTION"

    event_row = section_df[section_df["hydro_pair_id"] == "pair_A"].iloc[0]
    assert event_row["observation_status"] == fca.EVENT_PRESENT_STATUS

    forbidden = ("negative", "control", "safe", "stable")
    for token in forbidden:
        assert token not in no_event_row["observation_status"].lower()


# --- D: events sharing one hydro-pair section don't inflate the independent sample size ---


def test_D_events_sharing_one_section_count_as_one_independent_support_value():
    section_df = _make_section_df(
        [
            {"observation_status": fca.EVENT_PRESENT_STATUS, "observed_2018_event_count": 3},
            {"observation_status": fca.NO_EVENT_STATUS, "observed_2018_event_count": 0},
            {"observation_status": fca.EVENT_PRESENT_STATUS, "observed_2018_event_count": 1},
        ]
    )
    summary = fca.compute_event_independence_summary(section_df)
    assert summary["observed_event_count"] == 4
    assert summary["independent_hydrodynamic_support_section_count"] == 2
    assert summary["total_route_support_section_count"] == 3
    assert summary["events_per_support_section"] == {"pair_A": 3, "pair_C": 1}


# --- E: temporal statuses match the ticket's fixed expectations ---------------------------


def test_E_temporal_status_matches_expected_vocabulary_per_feature():
    expected = {
        "current_p95": fca.POSTDATES_2018,
        "wave_orbital_p95": fca.LONG_TERM_INCLUDES_2018,
        "combined_shear_p95": fca.POSTDATES_2018,
        "mobility_capacity": fca.POSTDATES_2018,
        "mar014_embedment_class": fca.POSTDATES_2018,
        "local_relief": fca.PREDATES_2018,
        "slope": fca.PREDATES_2018,
        "regional_folk_class": fca.REGIONAL_NOT_SINGLE_EPOCH,
        "psa_d50": fca.VARIES_BY_SAMPLE,
    }
    for key, expected_status in expected.items():
        assert fca.FEATURE_BY_KEY[key].temporal_status == expected_status, key


# --- F: support-resolution ratios are computed from the REAL registered support and -------
# --- the REAL observed event length, never a fabricated figure ---------------------------


def test_F_support_scale_to_event_length_ratio_uses_honest_registered_support():
    section_df = _make_section_df(
        [{"observation_status": fca.EVENT_PRESENT_STATUS, "observed_2018_event_count": 1}]
    )
    event_df = fca.build_event_level_context_table(
        pd.DataFrame([_event_row(source_length_m=10.0)]),
        section_df,
        _EMPTY_TEMPORAL_RELATIONSHIP_DF,
        _all_events_df(),
    )
    row = event_df.iloc[0]
    assert fca.FEATURE_BY_KEY["current_p95"].spatial_support_m == 1500.0
    assert row["current_p95_support_scale_to_event_length_ratio"] == pytest.approx(150.0)


def test_F_resolution_gap_statement_reports_the_real_ratio_never_a_fabricated_one():
    events_df = pd.DataFrame([_event_row(source_length_m=10.0)])
    statements = fca.compute_resolution_gap_statements(events_df)
    assert len(statements) == 1
    assert "150x larger" in statements[0]
    assert "10.00 m hydrodynamic" in statements[0]
    assert "10x larger than the observed span" in statements[0]
    assert "hydrodynamic forcing CONTEXT" in statements[0]


# --- G: a route-uniform feature (MAR-014's 0.03D) is classified SPATIALLY_UNIFORM ---------


def test_G_route_uniform_embedment_class_is_classified_spatially_uniform():
    section_df = _make_section_df(
        [
            {"p95_required_embedment_upper_class": "0.03"},
            {"p95_required_embedment_upper_class": "0.03"},
            {"p95_required_embedment_upper_class": "0.03"},
        ]
    )
    audit = fca.compute_categorical_audit(section_df, "mar014_embedment_class")
    assert audit["route_distinct_value_count"] == 1
    assert audit["classification"] == fca.SPATIALLY_UNIFORM
    assert fca.SPATIALLY_UNIFORM == "SPATIALLY_UNIFORM_AT_CURRENT_SUPPORT"


# --- H: an empirically two-valued feature is classified LOW_SPATIAL_CARDINALITY, even ------
# --- though the registry flags it as a continuous (non-categorical) feature ---------------


def test_H_two_valued_mobility_capacity_is_classified_low_spatial_cardinality():
    assert fca.FEATURE_BY_KEY["mobility_capacity"].is_categorical is False
    section_df = _make_section_df(
        [
            {"largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": 0.5},
            {"largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": 0.5},
            {"largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": 1.0},
        ]
    )
    # Callable despite is_categorical=False -- Section 14 names this feature explicitly.
    audit = fca.compute_categorical_audit(section_df, "mobility_capacity")
    assert audit["route_distinct_value_count"] == 2
    assert audit["classification"] == fca.LOW_SPATIAL_CARDINALITY
    assert fca.LOW_SPATIAL_CARDINALITY == "LOW_SPATIAL_CARDINALITY_AT_CURRENT_SUPPORT"


# --- I: descriptive contrast never contains a statistical-test/inference term -------------


def test_I_descriptive_contrast_contains_no_forbidden_statistical_terms():
    section_df = _make_section_df(
        [
            {
                "observation_status": fca.EVENT_PRESENT_STATUS,
                "current_reference_speed_p95_m_s": 5.0,
            },
            {"observation_status": fca.NO_EVENT_STATUS, "current_reference_speed_p95_m_s": 1.0},
            {"observation_status": fca.NO_EVENT_STATUS, "current_reference_speed_p95_m_s": 2.0},
        ]
    )
    contrast = fca.compute_descriptive_contrast(section_df, "current_p95")
    dumped = str(contrast).lower()
    forbidden = (
        "p_value",
        "pvalue",
        "p-value",
        "auc",
        "odds_ratio",
        "effect_size",
        "significance",
        "confidence_interval",
    )
    for token in forbidden:
        assert token not in dumped


# --- J: range overlap is a raw true/false comparison of the actual min/max, never ---------
# --- a binned or smoothed estimate ---------------------------------------------------------


def test_J_range_overlap_true_when_raw_ranges_actually_overlap():
    section_df = _make_section_df(
        [
            {
                "observation_status": fca.EVENT_PRESENT_STATUS,
                "current_reference_speed_p95_m_s": 5.0,
            },
            {"observation_status": fca.NO_EVENT_STATUS, "current_reference_speed_p95_m_s": 1.0},
            {"observation_status": fca.NO_EVENT_STATUS, "current_reference_speed_p95_m_s": 10.0},
        ]
    )
    contrast = fca.compute_descriptive_contrast(section_df, "current_p95")
    assert contrast["event_range"] == [5.0, 5.0]
    assert contrast["background_range"] == [1.0, 10.0]
    assert contrast["range_overlap"] is True


def test_J_range_overlap_false_when_raw_ranges_are_disjoint():
    section_df = _make_section_df(
        [
            {
                "observation_status": fca.EVENT_PRESENT_STATUS,
                "current_reference_speed_p95_m_s": 10.0,
            },
            {"observation_status": fca.NO_EVENT_STATUS, "current_reference_speed_p95_m_s": 1.0},
            {"observation_status": fca.NO_EVENT_STATUS, "current_reference_speed_p95_m_s": 2.0},
        ]
    )
    contrast = fca.compute_descriptive_contrast(section_df, "current_p95")
    assert contrast["event_range"] == [10.0, 10.0]
    assert contrast["background_range"] == [1.0, 2.0]
    assert contrast["range_overlap"] is False


# --- K: only an UNAMBIGUOUS source-stated temporal relationship is attached, never --------
# --- an inferred cross-survey match --------------------------------------------------------


def test_K_no_source_stated_relationship_leaves_fields_none_never_inferred():
    section_df = _make_section_df(
        [{"observation_status": fca.EVENT_PRESENT_STATUS, "observed_2018_event_count": 1}]
    )
    event_df = fca.build_event_level_context_table(
        pd.DataFrame([_event_row()]), section_df, _EMPTY_TEMPORAL_RELATIONSHIP_DF, _all_events_df()
    )
    row = event_df.iloc[0]
    assert row["source_stated_previous_survey_year"] is None
    assert row["source_stated_previous_event_id"] is None
    assert row["source_stated_previous_length_m"] is None
    assert row["source_stated_length_change_statement"] is None
    assert row["source_stated_2018_length_m"] == 10.0


def test_K_unambiguous_source_stated_relationship_is_copied_verbatim():
    section_df = _make_section_df(
        [{"observation_status": fca.EVENT_PRESENT_STATUS, "observed_2018_event_count": 1}]
    )
    temporal_relationship_df = pd.DataFrame(
        [
            {
                "event_id_b": "2018-01",
                "event_id_a": "2014-01",
                "relationship_type": "SOURCE_STATED_LENGTH_CHANGE",
                "survey_year_a": 2014,
                "source_statement": "Span increased from 8.0m (2014) to 10.0m (2018).",
            }
        ]
    )
    event_df = fca.build_event_level_context_table(
        pd.DataFrame([_event_row()]), section_df, temporal_relationship_df, _all_events_df()
    )
    row = event_df.iloc[0]
    assert row["source_stated_previous_survey_year"] == 2014
    assert row["source_stated_previous_event_id"] == "2014-01"
    assert row["source_stated_previous_length_m"] == 8.0
    assert row["source_stated_2018_length_m"] == 10.0
    assert row["source_stated_length_change_statement"] == (
        "Span increased from 8.0m (2014) to 10.0m (2018)."
    )


# --- P: no output schema anywhere contains a forbidden model-performance term --------------


def test_P_no_output_schema_contains_a_forbidden_scoring_or_performance_term():
    section_df = _make_section_df(
        [
            {"observation_status": fca.EVENT_PRESENT_STATUS, "observed_2018_event_count": 1},
            {"observation_status": fca.NO_EVENT_STATUS},
        ]
    )
    event_df = fca.build_event_level_context_table(
        pd.DataFrame([_event_row()]), section_df, _EMPTY_TEMPORAL_RELATIONSHIP_DF, _all_events_df()
    )
    event_independence = fca.compute_event_independence_summary(section_df)
    contrasts_by_key, categorical_audits_by_key, readiness = _build_audit_pieces(section_df)
    interpretation_answers = fca.answer_all_interpretation_questions(section_df)
    gaps = fca.compute_demonstrated_data_gaps(
        event_independence=event_independence,
        categorical_audits=list(categorical_audits_by_key.values()),
    )
    metadata = fca.build_audit_metadata(
        event_independence=event_independence,
        demonstrated_gaps=gaps,
        interpretation_answers=interpretation_answers,
    )

    forbidden = (
        "risk_score",
        "probability",
        "prediction",
        "auc",
        "accuracy",
        "true_positive",
        "false_positive",
    )
    for column in (*event_df.columns, *section_df.columns):
        for token in forbidden:
            assert token not in column.lower(), column
    for key, entry in readiness["features"].items():
        for token in forbidden:
            assert token not in str(entry).lower(), (key, token)

    assert metadata["score_created"] is False
    assert metadata["classifier_fitted"] is False
    assert metadata["model_validation_performed"] is False
    assert metadata["negative_labels_created"] is False
    assert contrasts_by_key  # sanity: the fixture actually exercised the contrast path

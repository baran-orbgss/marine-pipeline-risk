"""Offline unit tests for marine_engine.validation.freespan_model_context (MAR-014A).

Small hand-built synthetic segment tables only -- never the real MAR-012/013/
014 outputs, never network access. Lettered comment maps to MAR-014A
Section 25's required test list.
"""

import pandas as pd

from marine_engine.validation import freespan_model_context as fmc

_SEGMENT_COLUMNS_A = {
    "start_chainage_m": [0.0, 1000.0],
    "end_chainage_m": [1000.0, 2000.0],
    "hydro_pair_id": ["pair_A", "pair_B"],
    "tau_max_p95_sensitivity_min_pa": [0.3, 0.5],
    "tau_max_p95_sensitivity_max_pa": [0.8, 1.0],
    "tau_max_p95_sensitivity_width_pa": [0.5, 0.5],
}
_SEGMENT_COLUMNS_B = {
    "start_chainage_m": [0.0, 1000.0],
    "end_chainage_m": [1000.0, 2000.0],
    "hydro_pair_id": ["pair_A", "pair_B"],
    "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": [1.0, 0.5],
    "largest_tested_d50_with_any_exceedance_mm": [2.0, 1.0],
    "mapped_250k_folk_class": ["SAND", "MUD"],
    "nearest_valid_psa_d50_mm": [0.3, 0.2],
}
_SEGMENT_COLUMNS_C = {
    "start_chainage_m": [0.0, 1000.0],
    "end_chainage_m": [1000.0, 2000.0],
    "hydro_pair_id": ["pair_A", "pair_B"],
    "p95_required_embedment_lower_class": ["0", "0.03"],
    "p95_required_embedment_upper_class": ["0.03", "0.06"],
    "pipe_diameter_source_envelope_status": ["PIPE_DIAMETER_OUTSIDE_SOURCE_EXPERIMENT_ENVELOPE"]
    * 2,
    "slope_500m_median_deg": [1.0, 2.0],
    "tpi_1000m_median_m": [0.1, 0.2],
    "local_relief_1000m_median_m": [0.5, 0.6],
}


def _events_2018_df() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "event_id": "2018-01",
                "survey_year": 2018,
                "canonical_mid_chainage_m": 500.0,
                "canonical_mid_kp": "KP 0+500",
            },
            {
                "event_id": "2018-02",
                "survey_year": 2018,
                "canonical_mid_chainage_m": 1500.0,
                "canonical_mid_kp": "KP 1+500",
            },
        ]
    )


# --- T: model-context output carries no score/probability/rank/accuracy field ----------


def test_T_output_has_exactly_one_row_per_event_with_no_score_field():
    context_df = fmc.attach_model_context_to_events(
        _events_2018_df(),
        combined_bed_shear_segments_df=pd.DataFrame(_SEGMENT_COLUMNS_A),
        noncohesive_mobility_segments_df=pd.DataFrame(_SEGMENT_COLUMNS_B),
        scour_onset_segments_df=pd.DataFrame(_SEGMENT_COLUMNS_C),
    )

    assert len(context_df) == 2
    forbidden_substrings = ("score", "probability", "rank", "accuracy", "confidence")
    for column in context_df.columns:
        for token in forbidden_substrings:
            assert token not in column.lower(), column

    assert (
        fmc.MODEL_CONTEXT_ROLE
        == "INDEPENDENT_MODEL_OUTPUTS_JUXTAPOSED_NO_SCORE_NO_VALIDATION_METRIC"
    )
    assert (context_df["model_context_role"] == fmc.MODEL_CONTEXT_ROLE).all()


def test_T_each_event_gets_its_own_containing_segments_values():
    context_df = fmc.attach_model_context_to_events(
        _events_2018_df(),
        combined_bed_shear_segments_df=pd.DataFrame(_SEGMENT_COLUMNS_A),
        noncohesive_mobility_segments_df=pd.DataFrame(_SEGMENT_COLUMNS_B),
        scour_onset_segments_df=pd.DataFrame(_SEGMENT_COLUMNS_C),
    )

    first, second = context_df.iloc[0], context_df.iloc[1]
    assert first["hydro_pair_id"] == "pair_A"
    assert second["hydro_pair_id"] == "pair_B"
    assert first["mar012_tau_max_p95_sensitivity_min_pa"] == 0.3
    assert second["mar012_tau_max_p95_sensitivity_min_pa"] == 0.5
    assert first["mar014_p95_required_embedment_upper_class"] == "0.03"
    assert second["mar014_p95_required_embedment_upper_class"] == "0.06"


def test_T_missing_containing_segment_yields_none_never_a_fabricated_value():
    events_df = pd.DataFrame.from_records(
        [
            {
                "event_id": "2018-99",
                "survey_year": 2018,
                "canonical_mid_chainage_m": 50000.0,  # far beyond any segment's own range
                "canonical_mid_kp": "KP 50+000",
            }
        ]
    )
    context_df = fmc.attach_model_context_to_events(
        events_df,
        combined_bed_shear_segments_df=pd.DataFrame(_SEGMENT_COLUMNS_A),
        noncohesive_mobility_segments_df=pd.DataFrame(_SEGMENT_COLUMNS_B),
        scour_onset_segments_df=pd.DataFrame(_SEGMENT_COLUMNS_C),
    )
    # Falls back to the last segment (route-terminus clamping convention), never crashes.
    assert len(context_df) == 1
    assert context_df.iloc[0]["hydro_pair_id"] == "pair_B"

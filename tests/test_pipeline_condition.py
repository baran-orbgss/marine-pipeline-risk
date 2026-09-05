"""Offline unit tests for marine_engine.scour.pipeline_condition (MAR-014, Sections 28-30).

Never touches real files besides a tmp_path write; never network access.
"""

import json
from pathlib import Path

from marine_engine.scour import pipeline_condition


def test_benchmark_records_exact_required_fields():
    benchmark = pipeline_condition.build_2018_condition_benchmark()

    assert benchmark["reported_line_length_m"] == 24000
    assert benchmark["majority_buried_at_least_m"] == 0.6
    assert benchmark["majority_burial_statement_is_qualitative"] is True
    assert benchmark["free_span_count"] == 8
    assert benchmark["total_free_span_length_m"] == 97
    assert benchmark["max_free_span_height_m"] == 0.4
    assert benchmark["max_free_span_length_m"] == 23.2
    assert benchmark["exposed_section_count"] == 19
    assert benchmark["total_exposed_length_m"] == 519
    assert benchmark["longest_exposed_section_m"] == 87
    assert benchmark["survey_year"] == 2018
    assert benchmark["evidence_role"] == "HISTORICAL_AGGREGATE_CONDITION_BENCHMARK"
    assert benchmark["spatial_kp_locations_available_as_machine_readable_data"] is False
    assert benchmark["benchmark_scope"] == "PL854/PL855 export / methanol line"


def test_benchmark_preserves_source_internal_inconsistency():
    benchmark = pipeline_condition.build_2018_condition_benchmark()
    assert benchmark["source_internal_consistency_warning"] is True
    assert "23.2" in benchmark["source_internal_consistency_note"]
    assert "Narrative" in benchmark["source_internal_consistency_note"]
    # The narrative's own "no more than 10 m" claim is never used to silently
    # override the structured table value.
    assert benchmark["max_free_span_length_m"] == 23.2


# --- AD: 2018 benchmark cannot be spatially distributed to route segments -------------


def test_AD_benchmark_carries_no_spatial_or_kp_fields():
    """No per-event KP/chainage/lat/lon fields exist anywhere in the benchmark --
    it can never be spatially distributed to PL854 route segments. The one
    legitimate field naming "kp" documents the ABSENCE of machine-readable KP
    data, so key names (never the free-text values) are checked here."""

    benchmark = pipeline_condition.build_2018_condition_benchmark()

    def _all_keys(obj) -> list[str]:
        keys: list[str] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                keys.append(key)
                keys.extend(_all_keys(value))
        elif isinstance(obj, list):
            for item in obj:
                keys.extend(_all_keys(item))
        return keys

    keys_lower = [k.lower() for k in _all_keys(benchmark)]
    forbidden_key_tokens = (
        "chainage",
        "kp_start",
        "kp_end",
        "kp_position",
        "latitude",
        "longitude",
    )
    for token in forbidden_key_tokens:
        assert not any(token in key for key in keys_lower), token

    assert benchmark["spatial_kp_locations_available_as_machine_readable_data"] is False
    assert benchmark["historical_condition_used_for_spatial_calibration"] is False


def test_AD_benchmark_never_used_for_calibration():
    benchmark = pipeline_condition.build_2018_condition_benchmark()
    assert benchmark["historical_condition_used_for_spatial_calibration"] is False


def test_interpretation_statement_present():
    benchmark = pipeline_condition.build_2018_condition_benchmark()
    assert (
        benchmark["interpretation"]["statement"]
        == pipeline_condition.BENCHMARK_INTERPRETATION_STATEMENT
    )
    assert "NOT A MODEL OF THE DEEPLY BURIED MAJORITY" in benchmark["interpretation"]["statement"]


def test_write_2018_condition_benchmark_produces_valid_json(tmp_path: Path):
    output_path = tmp_path / "anglia_2018_condition_benchmark.json"
    result_path = pipeline_condition.write_2018_condition_benchmark(output_path)

    assert result_path == output_path
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["evidence_role"] == "HISTORICAL_AGGREGATE_CONDITION_BENCHMARK"

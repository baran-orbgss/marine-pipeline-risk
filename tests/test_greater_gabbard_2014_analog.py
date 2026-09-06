"""Offline unit tests for marine_engine.analogs.greater_gabbard_2014
(MAR-017C).

Never a live network call, never the real downloaded archive -- these
tests exercise the pure, offline parts of the FINAL analog orchestration.
Lettered comments map to MAR-017C Section 22's required test list.
"""

import inspect

import pandas as pd
import pytest

from marine_engine.analogs import greater_gabbard_2014 as gg

_FORBIDDEN_OUTPUT_TERMS = (
    "freespan_probability",
    "susceptibility",
    "risk",
    "migration_rate",
    "scour_prediction",
    "ml_",
)


def _infrastructure_df(*, turbines=(), rock_protection=()) -> pd.DataFrame:
    rows = []
    for tid, x, y in turbines:
        rows.append(
            {
                "infrastructure_id": tid,
                "infrastructure_type": "turbine_or_substation_foundation",
                "center_x_m": x,
                "center_y_m": y,
                "min_x_m": x - 1.0,
                "min_y_m": y - 1.0,
                "max_x_m": x + 1.0,
                "max_y_m": y + 1.0,
                "is_substation": "SUB" in tid.upper(),
            }
        )
    for rid, min_x, min_y, max_x, max_y in rock_protection:
        rows.append(
            {
                "infrastructure_id": rid,
                "infrastructure_type": "rock_concrete_protection",
                "center_x_m": (min_x + max_x) / 2.0,
                "center_y_m": (min_y + max_y) / 2.0,
                "min_x_m": min_x,
                "min_y_m": min_y,
                "max_x_m": max_x,
                "max_y_m": max_y,
                "is_substation": False,
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(gg.INFRASTRUCTURE_CONTEXT_COLUMNS))
    return pd.DataFrame(rows)


# --- A: support preflight occurs before spectral processing ---------------------------------


def test_A_cli_runs_the_canonical_support_preflight_before_any_tile_search():
    from marine_engine import cli

    source = inspect.getsource(cli._cmd_build_greater_gabbard_sandwave_validation)
    preflight_pos = source.index("run_canonical_support_preflight")
    tile_search_pos = source.index("build_tile_candidates")
    assert preflight_pos < tile_search_pos


def test_A_build_tile_candidates_is_never_called_when_early_stopped():
    from marine_engine import cli

    source = inspect.getsource(cli._cmd_build_greater_gabbard_sandwave_validation)
    _before, _, rest = source.partition("if early_stop_status is not None:")
    early_stop_body, _, else_body = rest.partition("else:")
    assert "build_tile_candidates" not in early_stop_body
    assert "build_tile_candidates" in else_body


# --- B: no below-1000 m canonical fallback ---------------------------------------------------


def test_B_preflight_and_tile_search_delegate_to_the_canonical_only_engine_search():
    """Never a custom cascade below the engine's own 2000 m -> 1000 m
    `swm.find_valid_tiles` -- this ticket introduces no new tile-size
    logic (Section 13)."""

    preflight_source = inspect.getsource(gg.run_canonical_support_preflight)
    assert "swm.find_valid_tiles" in preflight_source

    build_source = inspect.getsource(gg.build_tile_candidates)
    assert "swm.find_valid_tiles" in build_source
    assert "exploratory" not in build_source.lower()
    assert "exploratory" not in preflight_source.lower()


def test_B_preflight_covers_both_foundation_and_corridor_gridded_candidates():
    """A real archive-inspection gap (initially missed entirely) found a
    SECOND properly-gridded candidate class (`Corridors/GRIDDED 0.5x0.5`,
    152 real files at a different native resolution than the 144
    `Foundations/GRIDDED 0.25x0.25` grids) -- the preflight must cover
    BOTH, never just the class discovered first."""

    preflight_source = inspect.getsource(gg.run_canonical_support_preflight)
    assert "list_all_canonical_candidates" in preflight_source


def test_B_resolve_candidate_raises_for_an_unknown_id(tmp_path):
    with pytest.raises(FileNotFoundError):
        gg._resolve_candidate(tmp_path, "NO_SUCH_CANDIDATE")


# --- C: anthropogenic tile is not marked natural-seabed eligible ---------------------------


def test_C_a_tile_containing_a_turbine_foundation_is_anthropogenic_not_natural():
    infra = _infrastructure_df(turbines=[("GAA01", 500.0, 500.0)])
    result = gg.assess_natural_seabed_eligibility(
        tile_center_x_m=500.0,
        tile_center_y_m=500.0,
        tile_size_m=1000.0,
        infrastructure_df=infra,
        cable_segments=[],
    )
    assert result["status"] == gg.ANTHROPOGENIC_DISTURBANCE_PRESENT


def test_C_a_tile_overlapping_rock_protection_is_anthropogenic_not_natural():
    infra = _infrastructure_df(rock_protection=[("RP01", 400.0, 400.0, 600.0, 600.0)])
    result = gg.assess_natural_seabed_eligibility(
        tile_center_x_m=500.0,
        tile_center_y_m=500.0,
        tile_size_m=1000.0,
        infrastructure_df=infra,
        cable_segments=[],
    )
    assert result["status"] == gg.ANTHROPOGENIC_DISTURBANCE_PRESENT


def test_C_a_tile_far_from_all_infrastructure_is_natural_seabed_eligible():
    infra = _infrastructure_df(turbines=[("GAA01", 500000.0, 500000.0)])
    result = gg.assess_natural_seabed_eligibility(
        tile_center_x_m=0.0,
        tile_center_y_m=0.0,
        tile_size_m=1000.0,
        infrastructure_df=infra,
        cable_segments=[],
    )
    assert result["status"] == gg.NATURAL_SEABED_VALIDATION_ELIGIBLE
    assert result["nearest_turbine_distance_m"] == pytest.approx((2 * 500000.0**2) ** 0.5, rel=1e-6)


# --- D: natural eligibility is independent of spectral ranking ------------------------------


def test_D_a_spectrally_best_but_anthropogenic_tile_is_never_selected():
    """The best-ranked tile by every descriptive spectral metric must
    still be excluded from detailed validation if it is NOT
    NATURAL_SEABED_VALIDATION_ELIGIBLE -- eligibility is a hard pre-filter
    applied BEFORE ranking, never something a good spectral score can
    override."""

    df = pd.DataFrame(
        [
            {
                "tile_id": "best_but_anthropogenic",
                "tile_size_m": 2000.0,
                "center_x_m": 0.0,
                "center_y_m": 0.0,
                "directional_concentration": 0.99,
                "spectral_peak_to_median_power_ratio": 100.0,
                "dominant_wavelength_m": 500.0,  # 2000/500 = 4.0x -- spectrally eligible
                "meets_3_wavelengths_across_tile": True,
                "natural_seabed_eligibility_status": gg.ANTHROPOGENIC_DISTURBANCE_PRESENT,
                "selected_for_detailed_validation": False,
            },
            {
                "tile_id": "worse_but_natural",
                "tile_size_m": 2000.0,
                "center_x_m": 50000.0,
                "center_y_m": 50000.0,
                "directional_concentration": 0.4,
                "spectral_peak_to_median_power_ratio": 2.0,
                "dominant_wavelength_m": 500.0,
                "meets_3_wavelengths_across_tile": True,
                "natural_seabed_eligibility_status": gg.NATURAL_SEABED_VALIDATION_ELIGIBLE,
                "selected_for_detailed_validation": False,
            },
        ]
    )
    result = gg.select_detailed_validation_tiles(df)
    selected_ids = result[result["selected_for_detailed_validation"]]["tile_id"].tolist()
    assert selected_ids == ["worse_but_natural"]


# --- E: infrastructure-missing case does not silently become natural -----------------------


def test_E_zero_infrastructure_context_is_reported_as_insufficient_never_natural():
    empty_infra = _infrastructure_df()
    result = gg.assess_natural_seabed_eligibility(
        tile_center_x_m=0.0,
        tile_center_y_m=0.0,
        tile_size_m=1000.0,
        infrastructure_df=empty_infra,
        cable_segments=[],
    )
    assert result["status"] == gg.INFRASTRUCTURE_CONTEXT_INSUFFICIENT
    assert result["status"] != gg.NATURAL_SEABED_VALIDATION_ELIGIBLE


# --- F: three-wavelength eligibility remains strict ------------------------------------------


def test_F_a_natural_but_spectrally_ineligible_tile_is_never_selected():
    df = pd.DataFrame(
        [
            {
                "tile_id": "natural_but_2.9x",
                "tile_size_m": 2900.0,
                "center_x_m": 0.0,
                "center_y_m": 0.0,
                "directional_concentration": 0.99,
                "spectral_peak_to_median_power_ratio": 50.0,
                "dominant_wavelength_m": 1000.0,  # 2900/1000 = 2.9x -- ineligible
                "meets_3_wavelengths_across_tile": False,
                "natural_seabed_eligibility_status": gg.NATURAL_SEABED_VALIDATION_ELIGIBLE,
                "selected_for_detailed_validation": False,
            }
        ]
    )
    result = gg.select_detailed_validation_tiles(df)
    assert result[result["selected_for_detailed_validation"]].empty


# --- G: 30 m gate remains active -------------------------------------------------------------


def test_G_the_30m_wavelength_gate_remains_active_for_the_third_analog():
    from marine_engine.morphology import sandwave_morphometry as swm

    bedforms = [
        {"wavelength_m": 21.1, "wave_height_m": 0.3},
        {"wavelength_m": 45.0, "wave_height_m": 1.2},
    ]
    canonical, rejected_count = swm.apply_canonical_wavelength_gate(bedforms)
    assert [b["wavelength_m"] for b in canonical] == [45.0]
    assert rejected_count == 1


# --- H: validation cannot pass without a natural-seabed eligible tile -----------------------


def test_H_validation_cannot_pass_without_a_natural_seabed_eligible_tile():
    status, _reason = gg.derive_canonical_real_validation_status(
        canonical_tile_count=1,
        any_meets_3_wavelengths=True,
        any_natural_seabed_eligible=False,
        successful_transect_count=3,
        canonical_bedform_count=5,
    )
    assert status == gg.NO_NATURAL_SEABED_VALIDATION_TILE


def test_H_all_criteria_including_natural_seabed_eligibility_yields_validated():
    status, _reason = gg.derive_canonical_real_validation_status(
        canonical_tile_count=1,
        any_meets_3_wavelengths=True,
        any_natural_seabed_eligible=True,
        successful_transect_count=3,
        canonical_bedform_count=3,
    )
    assert status == gg.CANONICAL_REAL_DATA_VALIDATED


# --- I: validation cannot pass with <3 bedforms ----------------------------------------------


@pytest.mark.parametrize("bedform_count", [0, 1, 2])
def test_I_fewer_than_3_bedforms_never_yields_a_passing_status(bedform_count):
    status, _reason = gg.derive_canonical_real_validation_status(
        canonical_tile_count=1,
        any_meets_3_wavelengths=True,
        any_natural_seabed_eligible=True,
        successful_transect_count=3,
        canonical_bedform_count=bedform_count,
    )
    assert status == gg.INSUFFICIENT_CANONICAL_BEDFORMS


# --- J: analog values never enter PL854 outputs ----------------------------------------------


def test_J_analog_only_flags_are_fixed_and_correct():
    assert gg.ANALOG_ONLY_FLAGS == {
        "pl854_evidence": False,
        "pl854_feature_input": False,
        "pl854_validation_input": False,
        "method_development_analog_only": True,
    }
    assert gg.SCIENTIFIC_ROLE == "HIGH_RESOLUTION_SANDBED_MORPHOMETRY_METHOD_DEVELOPMENT_ANALOG"


def test_J_every_canonical_table_schema_carries_the_analog_only_flags():
    for columns in (gg.TILE_SPECTRAL_COLUMNS, gg.TRANSECT_COLUMNS, gg.INDIVIDUAL_BEDFORM_COLUMNS):
        for flag_name in gg.ANALOG_ONLY_FLAGS:
            assert flag_name in columns
        assert "scientific_role" in columns


def test_J_transfer_contract_states_pl854_flags_false_regardless_of_outcome():
    import json

    for status in (gg.CANONICAL_REAL_DATA_VALIDATED, gg.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT):
        contract = gg.build_pipeline_transfer_contract(
            canonical_real_validation_status=status,
            canonical_real_validation_reason="test reason",
        )
        assert contract["pl854_evidence"] is False
        assert contract["PL854_real_data_validation_passed"] is False
        assert contract["method_development_analog_only"] is True
        json.dumps(contract, default=str)


def test_J_greater_gabbard_outputs_are_never_written_under_a_pl854_study_directory():
    from marine_engine import cli

    source = inspect.getsource(cli._cmd_build_greater_gabbard_sandwave_validation)
    assert '"analogs"' in source
    assert '"greater_gabbard_2014"' in source
    assert "pipeline_id.lower()" not in source


# --- K: cross-analog summary contains method-validation fields only -------------------------


def test_K_cross_analog_summary_columns_contain_no_forbidden_term():
    from marine_engine import cli

    assert "natural_seabed_eligible_count" in cli.CROSS_ANALOG_SUMMARY_COLUMNS
    for column in cli.CROSS_ANALOG_SUMMARY_COLUMNS:
        for token in _FORBIDDEN_OUTPUT_TERMS:
            assert token not in column.lower(), column


def test_K_no_canonical_column_schema_contains_a_forbidden_term():
    all_columns = (
        list(gg.CANONICAL_SUPPORT_PREFLIGHT_COLUMNS)
        + list(gg.INFRASTRUCTURE_CONTEXT_COLUMNS)
        + list(gg.TILE_SPECTRAL_COLUMNS)
        + list(gg.TRANSECT_COLUMNS)
        + list(gg.INDIVIDUAL_BEDFORM_COLUMNS)
    )
    for column in all_columns:
        for token in _FORBIDDEN_OUTPUT_TERMS:
            assert token not in column.lower(), column


# --- L: third failed analog sets NO_FURTHER_OPEN_ANALOG_SEARCH_PLANNED ----------------------


def test_L_all_three_analogs_failing_closes_analog_hunting():
    decision = gg.build_final_mar017_family_decision(
        hhw_status=gg.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT,
        idrbnr_status=gg.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT,
        greater_gabbard_status=gg.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT,
    )
    assert decision["analog_search_status"] == "NO_FURTHER_OPEN_ANALOG_SEARCH_PLANNED"
    assert decision["canonical_real_data_validation_passed"] is False
    assert decision["real_analog_validation_attempts"] == 3


def test_L_greater_gabbard_passing_closes_validated_not_search_planned():
    decision = gg.build_final_mar017_family_decision(
        hhw_status=gg.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT,
        idrbnr_status=gg.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT,
        greater_gabbard_status=gg.CANONICAL_REAL_DATA_VALIDATED,
    )
    assert decision["analog_search_status"] == "CLOSED_VALIDATED"
    assert decision["canonical_real_data_validation_passed"] is True
    assert decision["validated_dataset"] == "GREATER_GABBARD_2014"

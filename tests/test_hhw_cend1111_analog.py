"""Offline unit tests for marine_engine.analogs.hhw_cend1111 (MAR-017).

Never a live network call, never the real downloaded ZIP -- these tests
exercise the pure, offline parts of the analog-only orchestration and the
reusable engine's dataset-agnostic boundary. Lettered comments map to
MAR-017 Section 29's required test list (M-O here; A-L live in
test_sandwave_morphometry.py alongside the reusable engine).
"""

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import LineString

from marine_engine.analogs import hhw_cend1111 as hhw
from marine_engine.morphology import sandwave_morphometry as swm

_FORBIDDEN_OUTPUT_TERMS = (
    "freespan_probability",
    "susceptibility",
    "risk",
    "migration_rate",
)


# --- M: HHW coordinates/identifiers cannot enter PL854 canonical feature outputs -----------


def test_M_reusable_engine_never_hard_codes_an_hhw_identifier():
    """Section 25: the reusable engine module must never know an HHW-
    specific grid name/coordinate -- verified by source inspection, the
    same pattern already established for prior tickets' anti-pattern
    checks in this project."""

    source = inspect.getsource(swm)
    _, _, body = source.partition('"""')
    _, _, body = body.partition('"""')  # drop the module docstring, which itself names this rule
    forbidden_identifiers = ("hhw", "asciito", "gsf", "cend", "jncc", "haisborough")
    for token in forbidden_identifiers:
        assert token not in body.lower(), token


def test_M_hhw_outputs_are_never_written_under_a_pl854_study_directory():
    """The CLI orchestration must construct every MAR-017 output path from
    `analogs/hhw_cend1111`, never from the PL854 `pipeline_id`-derived
    study directory every other command uses."""

    from marine_engine import cli

    source = inspect.getsource(cli._cmd_build_analog_sandwave_morphometry)
    assert '"analogs"' in source
    assert '"hhw_cend1111"' in source
    assert "pipeline_id.lower()" not in source


# --- N: pl854_evidence (and the other analog-only flags) are always false/true as fixed ----


def test_N_analog_only_flags_are_fixed_and_correct():
    assert hhw.ANALOG_ONLY_FLAGS == {
        "pl854_evidence": False,
        "pl854_feature_input": False,
        "pl854_validation_input": False,
        "method_development_analog_only": True,
    }
    assert hhw.SCIENTIFIC_ROLE == "HIGH_RESOLUTION_SANDBED_MORPHOMETRY_METHOD_DEVELOPMENT_ANALOG"


def test_N_every_canonical_table_schema_carries_the_analog_only_flags():
    for columns in (
        hhw.TILE_SPECTRAL_COLUMNS,
        hhw.TRANSECT_COLUMNS,
        hhw.INDIVIDUAL_BEDFORM_COLUMNS,
    ):
        for flag_name in hhw.ANALOG_ONLY_FLAGS:
            assert flag_name in columns
        assert "scientific_role" in columns


def test_N_pipeline_transfer_contract_states_pl854_evidence_is_false():
    contract = hhw.build_pipeline_transfer_contract(
        hhw_canonical_2d_validation_status=hhw.HHW_CANONICAL_2D_TILE_VALIDATION_NOT_SUPPORTED,
        hhw_detailed_bedform_validation_status=hhw.NOT_CANONICALLY_VALIDATED,
    )
    assert contract["pl854_evidence"] is False
    assert contract["pl854_feature_input"] is False
    assert contract["pl854_validation_input"] is False
    assert contract["method_development_analog_only"] is True
    assert contract["scientific_role"] == hhw.SCIENTIFIC_ROLE
    # Must be JSON-serialisable as the CLI writes it verbatim.
    json.dumps(contract, default=str)


# --- O: no output schema anywhere contains a forbidden downstream-modelling term -----------


def test_O_no_canonical_column_schema_contains_a_forbidden_term():
    all_columns = (
        list(hhw.TILE_INVENTORY_COLUMNS)
        + list(hhw.TILE_SPECTRAL_COLUMNS)
        + list(hhw.TRANSECT_COLUMNS)
        + list(hhw.INDIVIDUAL_BEDFORM_COLUMNS)
    )
    for column in all_columns:
        for token in _FORBIDDEN_OUTPUT_TERMS:
            assert token not in column.lower(), column


def test_O_pipeline_transfer_contract_contains_no_forbidden_term():
    contract = hhw.build_pipeline_transfer_contract(
        hhw_canonical_2d_validation_status=hhw.HHW_CANONICAL_2D_TILE_VALIDATION_NOT_SUPPORTED,
        hhw_detailed_bedform_validation_status=hhw.NOT_CANONICALLY_VALIDATED,
    )
    dumped = json.dumps(contract, default=str).lower()
    for token in _FORBIDDEN_OUTPUT_TERMS:
        assert token not in dumped, token


def test_O_reusable_engine_source_contains_no_forbidden_term():
    source = inspect.getsource(swm).lower()
    for token in _FORBIDDEN_OUTPUT_TERMS:
        assert token not in source, token


# --- Shared: the generic transect generator + morphometry engine are exercised end-to-end --
# --- on a synthetic tile (no live network / no real HHW download) -------------------------


def test_transect_and_bedform_flow_on_a_synthetic_tile_never_touches_hhw():
    """A lightweight, fully offline sanity check that `generate_cross_
    crest_transects` (the reusable engine) produces geometry usable by
    ordinary shapely code, with no HHW-specific assumption anywhere."""

    endpoints = swm.generate_cross_crest_transects(
        center_x_m=500000.0, center_y_m=5900000.0, tile_size_m=250.0, crest_azimuth_deg=45.0
    )
    assert len(endpoints) == 3
    lines = [LineString([p1, p2]) for p1, p2 in endpoints]
    lengths = [line.length for line in lines]
    # Transect length is kept within the tile's own footprint (Section 14) -- twice the
    # 0.5*tile_size half-length used by the reusable engine.
    assert all(length == pytest.approx(250.0, rel=0.05) for length in lengths)


def test_select_canonical_top_tiles_never_falls_back_to_an_ineligible_tile():
    """MAR-017A Section 5 directly fixes the review's Issue B: the OLD
    `select_top_tiles` fell back to an ineligible pool whenever nothing
    met the strict >=3-wavelengths condition, which is exactly how HHW's
    real 250 m tiles ended up in a canonical validation output. Neither
    tile below meets the condition, so NONE may be selected."""

    df = pd.DataFrame(
        [
            {
                "tile_id": "t1",
                "tile_size_m": 2000.0,
                "directional_concentration": 0.9,
                "spectral_peak_to_median_power_ratio": 5.0,
                "dominant_wavelength_m": 1000.0,  # 2000/1000 = 2.0x -- ineligible
                "meets_3_wavelengths_across_tile": False,
                "rank_selected_top3": False,
                "canonical_validation_eligible": True,
                "reason": None,
            },
            {
                "tile_id": "t2",
                "tile_size_m": 2000.0,
                "directional_concentration": 0.5,
                "spectral_peak_to_median_power_ratio": 2.0,
                "dominant_wavelength_m": 900.0,  # 2000/900 = 2.22x -- also ineligible
                "meets_3_wavelengths_across_tile": False,
                "rank_selected_top3": False,
                "canonical_validation_eligible": True,
                "reason": None,
            },
        ]
    )
    ranked = hhw.select_canonical_top_tiles(df, top_n=1)
    assert ranked[ranked["rank_selected_top3"]].empty


# --- MAR-017A Section 15's own required test list (A-L) -- B/I/J/K/L land here since they ---
# --- exercise this analog-orchestration module; A/C/D/E/F/G/H live in ----------------------
# --- test_sandwave_morphometry.py alongside the reusable engine functions they test --------


def test_mar017a_B_zero_canonical_tiles_yields_empty_well_formed_output_not_fake_rows():
    empty_df = hhw.build_tile_spectral_table(Path("unused.zip"), "unused_grid", [], canonical=True)
    assert empty_df.empty
    assert list(empty_df.columns) == list(hhw.TILE_SPECTRAL_COLUMNS)


def test_mar017a_I_exploratory_tiles_carry_noncanonical_semantics_through_selection():
    df = pd.DataFrame(
        [
            {
                "tile_id": "t1",
                "tile_size_m": 250.0,
                "directional_concentration": 0.9,
                "spectral_peak_to_median_power_ratio": 5.0,
                "dominant_wavelength_m": 100.0,
                "meets_3_wavelengths_across_tile": False,
                "rank_selected_top3": False,
                "canonical_validation_eligible": False,
                "reason": hhw.BELOW_MINIMUM_SPATIAL_SUPPORT,
            },
            {
                "tile_id": "t2",
                "tile_size_m": 250.0,
                "directional_concentration": 0.5,
                "spectral_peak_to_median_power_ratio": 2.0,
                "dominant_wavelength_m": 90.0,
                "meets_3_wavelengths_across_tile": False,
                "rank_selected_top3": False,
                "canonical_validation_eligible": False,
                "reason": hhw.BELOW_MINIMUM_SPATIAL_SUPPORT,
            },
        ]
    )
    selected = hhw.select_exploratory_top_tiles(df, top_n=1)
    # The lenient exploratory ranking must never erase or "promote" the noncanonical stamp.
    assert selected["canonical_validation_eligible"].eq(False).all()
    assert selected["reason"].eq(hhw.BELOW_MINIMUM_SPATIAL_SUPPORT).all()
    assert selected[selected["rank_selected_top3"]]["tile_id"].tolist() == ["t1"]


def test_mar017a_J_transfer_readiness_keeps_question_a_and_b_independent():
    validated_contract = hhw.build_pipeline_transfer_contract(
        hhw_canonical_2d_validation_status=hhw.CANONICALLY_VALIDATED,
        hhw_detailed_bedform_validation_status=hhw.CANONICALLY_VALIDATED,
    )
    not_validated_contract = hhw.build_pipeline_transfer_contract(
        hhw_canonical_2d_validation_status=hhw.HHW_CANONICAL_2D_TILE_VALIDATION_NOT_SUPPORTED,
        hhw_detailed_bedform_validation_status=hhw.NOT_CANONICALLY_VALIDATED,
    )
    # Question A (generic engine readiness) never changes with the real-HHW validation outcome.
    assert validated_contract["transfer_readiness"]["answer_A"] == "YES"
    assert not_validated_contract["transfer_readiness"]["answer_A"] == "YES"
    # Question B (real-HHW validation) DOES change, and is never collapsed into question A.
    assert validated_contract["transfer_readiness"]["answer_B"] == "YES"
    assert not_validated_contract["transfer_readiness"]["answer_B"] == "NO"


def test_mar017a_K_cli_never_writes_exploratory_dataframes_into_canonical_outputs():
    from marine_engine import cli

    source = inspect.getsource(cli._cmd_build_analog_sandwave_morphometry)
    tile_write_line = next(
        line
        for line in source.splitlines()
        if "tile_spectral_path" in line and ".to_parquet" in line
    )
    assert "canonical_spectral_df" in tile_write_line
    transect_write_line = next(
        line for line in source.splitlines() if "transect_path" in line and ".to_parquet" in line
    )
    assert "canonical_transect_df" in transect_write_line
    bedform_write_line = next(
        line for line in source.splitlines() if "bedform_path" in line and ".to_parquet" in line
    )
    assert "canonical_bedform_df" in bedform_write_line


def test_mar017a_L_pl854_evidence_flags_remain_false_after_the_repair():
    contract = hhw.build_pipeline_transfer_contract(
        hhw_canonical_2d_validation_status=hhw.HHW_CANONICAL_2D_TILE_VALIDATION_NOT_SUPPORTED,
        hhw_detailed_bedform_validation_status=hhw.NOT_CANONICALLY_VALIDATED,
    )
    assert contract["pl854_evidence"] is False
    assert contract["pl854_feature_input"] is False
    assert contract["pl854_validation_input"] is False
    assert contract["method_development_analog_only"] is True

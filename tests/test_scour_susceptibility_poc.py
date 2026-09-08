"""Offline unit tests for the MAR-023 generic scour susceptibility POC.

Small synthetic records only -- never the real PL854/Sheringham data, never
network access (the one acquisition test proves a cache HIT skips the
network entirely, by raising if `requests.get` is ever called). Test names
map to MAR-023 Section 22's required list.
"""

from __future__ import annotations

import inspect
import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from marine_engine import cli
from marine_engine.providers.bathymetry import sheringham_shoal_2024
from marine_engine.scour import (
    contract,
    observed_evidence,
    observed_evidence_map,
    poc_report,
    scour_onset,
    susceptibility,
    susceptibility_map,
)

# --- synthetic fixtures --------------------------------------------------------------------


def _mobility_rows(
    hydro_pair_id: str,
    *,
    ratios_and_statuses: list[tuple[float | None, str | None]],
    tested_d50_mm: float = 0.16,
    porosity_scenario: float = 0.40,
) -> list[dict]:
    return [
        {
            "hydro_pair_id": hydro_pair_id,
            "tested_d50_mm": tested_d50_mm,
            "porosity_scenario": porosity_scenario,
            "minimum_tested_embedment_ratio_suppressing_onset": ratio,
            "required_embedment_status": status,
        }
        for ratio, status in ratios_and_statuses
    ]


def _single_scenario_mobility_df(hydro_pair_id: str, ratios: list[float]) -> pd.DataFrame:
    rows = _mobility_rows(
        hydro_pair_id,
        ratios_and_statuses=[(r, scour_onset.REQUIRED_EMBEDMENT_IDENTIFIED) for r in ratios],
    )
    return pd.DataFrame(rows)


SECTION = {
    "section_id": 0,
    "hydro_pair_id": "HP1",
    "start_chainage_m": 0.0,
    "end_chainage_m": 1000.0,
}


# --- A: actual embedment and critical embedment remain separate ----------------------------


def test_A_actual_and_critical_embedment_remain_separate():
    # every timestep requires e/D=0.10 to suppress onset
    mobility_df = _single_scenario_mobility_df("HP1", [0.10] * 20)
    row, _detail = susceptibility.compute_section_scour_susceptibility(
        section_mobility_df=mobility_df,
        actual_embedment_m=0.05 * 0.3048,
        diameter_m=0.3048,
        evidence_type=susceptibility.EVIDENCE_TYPE_SITE_SPECIFIC,
    )
    assert row["actual_embedment_ratio"] == pytest.approx(0.05)
    assert row["critical_embedment_ratio_p95"] == pytest.approx(0.10)
    assert row["actual_embedment_ratio"] != row["critical_embedment_ratio_p95"]


# --- B: negative protection margin maps to below-critical screening ------------------------


def test_B_negative_margin_maps_to_below_critical():
    mobility_df = _single_scenario_mobility_df("HP1", [0.10] * 20)
    row, _detail = susceptibility.compute_section_scour_susceptibility(
        section_mobility_df=mobility_df,
        actual_embedment_m=0.05 * 0.3048,
        diameter_m=0.3048,
        evidence_type=susceptibility.EVIDENCE_TYPE_SITE_SPECIFIC,
    )
    assert row["embedment_protection_margin_p95_e_over_D"] < 0
    assert row["screening_state"] == susceptibility.BELOW_CRITICAL_EMBEDMENT_SCREENING


# --- C: positive protection margin never maps to SAFE ---------------------------------------


def test_C_positive_margin_never_maps_to_safe():
    mobility_df = _single_scenario_mobility_df("HP1", [0.03] * 20)
    row, _detail = susceptibility.compute_section_scour_susceptibility(
        section_mobility_df=mobility_df,
        actual_embedment_m=0.15 * 0.3048,
        diameter_m=0.3048,
        evidence_type=susceptibility.EVIDENCE_TYPE_SITE_SPECIFIC,
    )
    assert row["embedment_protection_margin_p95_e_over_D"] > 0
    assert row["screening_state"] == susceptibility.AT_OR_ABOVE_CRITICAL_EMBEDMENT_SCREENING
    for state in susceptibility.SCREENING_STATES:
        assert "SAFE" not in state
        assert "ACCEPTABLE" not in state
        assert "RISK" not in state


# --- D: exceedance fraction is computed from valid timesteps only --------------------------


def test_D_exceedance_fraction_uses_valid_timesteps_only():
    rows = _mobility_rows(
        "HP1",
        ratios_and_statuses=[
            (0.10, scour_onset.REQUIRED_EMBEDMENT_IDENTIFIED),
            (0.10, scour_onset.REQUIRED_EMBEDMENT_IDENTIFIED),
            (0.03, scour_onset.REQUIRED_EMBEDMENT_IDENTIFIED),
            (None, None),  # missing input data -- must be excluded, never a 0 or a trigger
            (None, None),
        ],
    )
    mobility_df = pd.DataFrame(rows)
    valid_count, triggered_count, fraction = susceptibility.compute_scenario_exceedance(
        mobility_df, actual_embedment_ratio=0.05
    )
    assert valid_count == 3
    assert triggered_count == 2  # the two 0.10 rows exceed 0.05; the 0.03 row does not
    assert fraction == pytest.approx(2 / 3)


# --- E: exceedance fraction is never named probability --------------------------------------


def test_E_exceedance_fraction_never_named_probability():
    assert "NOT_A_PROBABILITY" in susceptibility.EXCEEDANCE_FRACTION_SEMANTICS
    disclaimers = " ".join(susceptibility.EXCEEDANCE_FRACTION_DISCLAIMERS)
    assert "NOT A PROBABILITY" in disclaimers
    assert "NOT A FAILURE PROBABILITY" in disclaimers
    assert "NOT A RETURN-PERIOD METRIC" in disclaimers


# --- F: missing actual embedment blocks site-specific susceptibility -----------------------


def test_F_missing_actual_embedment_blocks_site_specific_result():
    mobility_df = _single_scenario_mobility_df("HP1", [0.10] * 20)
    summary_df, _detail_df = susceptibility.build_susceptibility_tables(
        pipeline_id="PL854",
        sections=[SECTION],
        mobility_df=mobility_df,
        diameter_m=0.3048,
        actual_embedment_m_by_section_id=None,
        evidence_type=susceptibility.EVIDENCE_TYPE_NO_PROFILE,
    )
    row = summary_df.iloc[0]
    assert row["actual_embedment_ratio"] is None
    assert (
        row["screening_state"]
        == susceptibility.SITE_SPECIFIC_SCOUR_SUSCEPTIBILITY_NOT_AVAILABLE_NO_EMBEDMENT_PROFILE
    )
    assert row["embedment_protection_margin_p95_e_over_D"] is None
    assert row["scour_onset_screening_exceedance_fraction"] is None


# --- G: PL854 remains scenario-only (structural) --------------------------------------------


def _cli_source() -> str:
    return inspect.getsource(cli._cmd_build_scour_susceptibility_poc)


def test_G_pl854_site_specific_call_passes_no_actual_embedment_profile():
    source = _cli_source()
    assert "actual_embedment_m_by_section_id=None" in source
    assert "susceptibility.EVIDENCE_TYPE_NO_PROFILE" in source


def test_G2_pl854_scenario_envelope_uses_tested_ratios_not_a_real_profile():
    source = _cli_source()
    assert "build_scenario_envelope_table" in source
    assert "TESTED_EMBEDMENT_SCENARIOS" in inspect.getsource(susceptibility)


# --- H: Marini domain/extrapolation status is preserved -------------------------------------


def test_H_marini_domain_extrapolation_status_preserved():
    mobility_df = _single_scenario_mobility_df("HP1", [0.10] * 20)
    row, _detail = susceptibility.compute_section_scour_susceptibility(
        section_mobility_df=mobility_df,
        actual_embedment_m=0.05 * scour_onset.PIPELINE_DIAMETER_M,
        diameter_m=scour_onset.PIPELINE_DIAMETER_M,
        evidence_type=susceptibility.EVIDENCE_TYPE_SITE_SPECIFIC,
    )
    assert row["method_domain_status"] == scour_onset.PIPE_DIAMETER_OUTSIDE_SOURCE_ENVELOPE
    assert scour_onset.RESEARCH_SCREENING_EXTRAPOLATION in row["limitations"]


# --- I: no scour depth is predicted (structural) ---------------------------------------------


def test_I_no_scour_depth_field_anywhere_in_generic_schema():
    all_columns = " ".join(susceptibility.GENERIC_OUTPUT_COLUMNS) + " ".join(
        susceptibility.SENSITIVITY_DETAIL_COLUMNS
    )
    assert "depth" not in all_columns.lower()
    source = inspect.getsource(susceptibility)
    assert "scour_depth" not in source
    assert "erosion_depth" not in source


# --- J: no risk score exists (structural) -----------------------------------------------------


def test_J_no_risk_score_field_or_vocabulary():
    all_columns = " ".join(susceptibility.GENERIC_OUTPUT_COLUMNS)
    assert "risk_score" not in all_columns
    assert "score" not in all_columns
    for state in susceptibility.SCREENING_STATES:
        assert "LOW RISK" not in state
        assert "MEDIUM RISK" not in state
        assert "HIGH RISK" not in state


# --- K: observed Sheringham scour never enters pipeline physics (structural) ----------------


def test_K_observed_evidence_module_never_imported_by_physics_engine():
    onset_source = inspect.getsource(scour_onset)
    susceptibility_source = inspect.getsource(susceptibility)
    assert "observed_evidence" not in onset_source
    assert "observed_evidence" not in susceptibility_source
    assert "Sheringham" not in onset_source
    assert "XOCEAN" not in onset_source


def test_K2_source_interpretation_not_used_to_tune_physics():
    onset_source = inspect.getsource(scour_onset)
    assert "requests" not in onset_source
    assert "shapefile" not in onset_source.lower()


# --- L: asset-type mismatch is explicit -------------------------------------------------------


def test_L_asset_physics_mismatch_is_explicit():
    assert observed_evidence.ASSET_PHYSICS_NOT_EQUIVALENT_TO_PIPELINE_SCOUR_MODEL
    assert "NOT" in observed_evidence.ASSET_PHYSICS_DISCLAIMER
    assert "validation dataset" in observed_evidence.ASSET_PHYSICS_DISCLAIMER


# --- M: source interpretation is not used to seed/tune the detector ------------------------


def test_M_classification_never_infers_scour_without_explicit_source_descriptor():
    gdf = gpd.GeoDataFrame(
        {"Descriptio": ["Exposure", "Boulder"], "geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:32631",
    )
    layer = observed_evidence.InterpretationLayer(
        layer_name="test_layer", gdf=gdf, description_column="Descriptio"
    )
    evidence_gdf = observed_evidence.build_observed_evidence_table(
        layer,
        category_by_normalized_descriptor={
            "exposure": observed_evidence.SOURCE_INTERPRETED_EXPOSURE_EVIDENCE,
            "boulder": observed_evidence.SEABED_OBJECT_CONTEXT,
        },
        survey_epoch="2024",
        source_id_column="missing_column",
    )
    # neither real descriptor literally says "scour" -- neither may be classified as such
    assert (
        observed_evidence.SOURCE_INTERPRETED_OBSERVED_SCOUR_EVIDENCE
        not in evidence_gdf["interpretation_category"].tolist()
    )
    summary = observed_evidence.summarize_observed_evidence(evidence_gdf)
    assert summary["explicit_scour_feature_count"] == 0
    assert (
        summary["morphometry_status"]
        == observed_evidence.OBSERVED_SCOUR_MORPHOMETRY_NOT_AVAILABLE_FROM_SOURCE_PACKAGE
    )


def test_M2_unmapped_descriptor_is_unclassified_never_assumed():
    gdf = gpd.GeoDataFrame(
        {"Descriptio": ["Something Unrecognised"], "geometry": [Point(0, 0)]}, crs="EPSG:32631"
    )
    layer = observed_evidence.InterpretationLayer(
        layer_name="test_layer", gdf=gdf, description_column="Descriptio"
    )
    evidence_gdf = observed_evidence.build_observed_evidence_table(
        layer, category_by_normalized_descriptor={}, survey_epoch="2024", source_id_column="x"
    )
    assert (
        evidence_gdf["interpretation_category"].iloc[0]
        == observed_evidence.UNCLASSIFIED_INTERPRETATION_FEATURE
    )


# --- N: synthetic operator route produces a valid scour map (Section 17/22) ----------------


def test_N_synthetic_operator_route_produces_a_valid_susceptibility_map(tmp_path):
    route = LineString([(0, 0), (0, 2000)])
    sections = [
        {
            "section_id": 0,
            "hydro_pair_id": "HP1",
            "start_chainage_m": 0.0,
            "end_chainage_m": 1000.0,
        },
        {
            "section_id": 1,
            "hydro_pair_id": "HP2",
            "start_chainage_m": 1000.0,
            "end_chainage_m": 2000.0,
        },
    ]
    mobility_df = pd.concat(
        [
            _single_scenario_mobility_df("HP1", [0.03] * 10),
            _single_scenario_mobility_df("HP2", [0.15] * 10),
        ],
        ignore_index=True,
    )
    diameter_m = 0.5
    summary_df, _detail_df = susceptibility.build_susceptibility_tables(
        pipeline_id="SYNTH-001",
        sections=sections,
        mobility_df=mobility_df,
        diameter_m=diameter_m,
        actual_embedment_m_by_section_id={0: 0.10 * diameter_m, 1: 0.10 * diameter_m},
        evidence_type=susceptibility.EVIDENCE_TYPE_SITE_SPECIFIC,
    )
    assert set(summary_df["screening_state"]) <= {
        susceptibility.AT_OR_ABOVE_CRITICAL_EMBEDMENT_SCREENING,
        susceptibility.BELOW_CRITICAL_EMBEDMENT_SCREENING,
    }
    assert summary_df["embedment_protection_margin_p95_e_over_D"].notna().all()

    segments_gdf = gpd.GeoDataFrame(
        {
            "segment_id": [0, 1],
            "geometry": [
                LineString([(0, 0), (0, 1000)]),
                LineString([(0, 1000), (0, 2000)]),
            ],
        },
        crs="EPSG:32631",
    )
    merged = segments_gdf.merge(summary_df, left_on="segment_id", right_on="section_id")

    output_path = tmp_path / "synthetic_susceptibility_map.png"
    result_path = susceptibility_map.render_pipeline_scour_susceptibility_map(
        segments_gdf=merged,
        route=route,
        working_crs="EPSG:32631",
        output_path=output_path,
        diameter_m=diameter_m,
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_N2_synthetic_operator_evidence_map_renders(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "interpretation_category": [
                observed_evidence.SOURCE_INTERPRETED_EXPOSURE_EVIDENCE,
                observed_evidence.SEABED_OBJECT_CONTEXT,
            ],
            "geometry": [Point(0, 0), Point(10, 10)],
        },
        crs="EPSG:32631",
    )
    output_path = tmp_path / "synthetic_evidence_map.png"
    result_path = observed_evidence_map.render_observed_scour_evidence_map(
        evidence_gdf=gdf, output_path=output_path
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0


# --- O: no hard-coded PL854/Sheringham coordinates in generic engine (structural) ----------
#
# Dataset NAMES appearing in documentation/comments/default title strings are established
# precedent in this codebase -- the already-accepted MAR-014 `scour_onset.py` itself
# documents "PL854/NSTA pipeline evidence" in `PIPELINE_DIAMETER_SOURCE`, and
# `scour_onset_map.py`'s own title defaults to "PL854 -- ...". What must never appear is a
# literal COORDINATE tied to one specific dataset -- that would mean the engine could not
# actually run against a different operator's route/site without editing its source.


def test_O_generic_engine_has_no_hardcoded_dataset_coordinates():
    # PL854's real route bbox corner (configs/pl854.yaml) and the real 2024 Sheringham
    # Shoal target bounds corner (confirmed via direct shapefile inspection this ticket).
    forbidden_coordinate_fragments = ("1.6516061", "53.3678022", "371689.24", "5868466.74")
    for module in (susceptibility, susceptibility_map, observed_evidence, observed_evidence_map):
        source = inspect.getsource(module)
        for fragment in forbidden_coordinate_fragments:
            assert fragment not in source, f"{module.__name__} hard-codes {fragment!r}"


# --- P: offline rerun works after acquisition (cache hit, no network) ----------------------


def test_P_interpretation_data_cache_hit_never_touches_network(tmp_path, monkeypatch):
    def _explode(*_args, **_kwargs):
        raise AssertionError("requests.get must not be called on a cache hit")

    monkeypatch.setattr(sheringham_shoal_2024.requests, "get", _explode)

    raw_dir = tmp_path / "raw"
    extracted_dir = raw_dir / "interpretation_data"
    extracted_dir.mkdir(parents=True)
    for ext in (".shp", ".dbf", ".shx"):
        (extracted_dir / f"{sheringham_shoal_2024.TARGETS_LAYER_NAME}{ext}").write_bytes(b"stub")
    sidecar = sheringham_shoal_2024._sidecar_path(extracted_dir)
    sidecar.write_text(
        json.dumps(
            {
                "package_bytes": 12345,
                "package_sha256": "deadbeef",
                "retrieved_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    acquisition = sheringham_shoal_2024.download_sheringham_shoal_2024_interpretation_data(raw_dir)
    assert acquisition.already_cached is True
    assert acquisition.package_bytes == 12345


# --- validation-question derivation (Section 21) --------------------------------------------


def test_validation_questions_D_G_H_are_hardcoded_no():
    result = cli._derive_scour_poc_validation_questions(
        margin_computed_when_actual_embedment_provided=True,
        exceedance_fraction_computable=True,
        pl854_scenario_envelope_produced=True,
        real_scour_evidence_ingested=True,
    )
    assert result["question_d_pl854_has_enough_data_for_site_specific_susceptibility"] == "NO"
    assert result["question_g_sheringham_used_as_pipeline_physics_validation"] == "NO"
    assert result["question_h_future_scour_depth_prediction_made"] == "NO"
    assert result["question_a_generic_engine_operator_input_ready"] == "YES"
    assert result["question_b_margin_computed_when_actual_embedment_provided"] == "YES"
    assert result["question_e_pl854_scenario_envelope_produced"] == "YES"
    assert result["question_f_real_source_interpreted_scour_evidence_ingested"] == "YES"


# --- scenario envelope table shape (Section 16) ----------------------------------------------


def test_scenario_envelope_never_collapses_the_five_scenarios():
    mobility_df = _single_scenario_mobility_df("HP1", [0.06] * 20)
    summary_df, _detail_df = susceptibility.build_scenario_envelope_table(
        pipeline_id="PL854",
        sections=[SECTION],
        mobility_df=mobility_df,
        diameter_m=0.3048,
    )
    assert len(summary_df) == len(susceptibility.TESTED_EMBEDMENT_SCENARIOS)
    assert set(summary_df["tested_embedment_scenario_ratio"]) == set(
        susceptibility.TESTED_EMBEDMENT_SCENARIOS
    )
    # a smaller scenario embedment must never show a LOWER-or-equal exceedance fraction than a
    # larger one (monotonicity the underlying physics guarantees)
    ordered = summary_df.sort_values("tested_embedment_scenario_ratio")
    fractions = ordered["scour_onset_screening_exceedance_fraction"].tolist()
    assert fractions == sorted(fractions, reverse=True)


# --- censoring behaviour (onset persists at max tested embedment) --------------------------


def test_censored_scenario_counts_as_exceeding_when_actual_is_within_envelope():
    mobility_df = _mobility_rows(
        "HP1",
        ratios_and_statuses=[(None, scour_onset.ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT)] * 10,
    )
    mobility_df = pd.DataFrame(mobility_df)
    stats = susceptibility.compute_scenario_critical_embedment_stats(mobility_df)
    assert stats.p95_censored is True
    assert stats.p95_ratio is None

    valid_count, triggered_count, fraction = susceptibility.compute_scenario_exceedance(
        mobility_df, actual_embedment_ratio=0.10
    )
    assert valid_count == 10
    assert triggered_count == 10
    assert fraction == pytest.approx(1.0)


def test_censored_p95_forces_outside_method_experimental_envelope_state():
    mobility_df = pd.DataFrame(
        _mobility_rows(
            "HP1",
            ratios_and_statuses=[(None, scour_onset.ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT)] * 10,
        )
    )
    row, _detail = susceptibility.compute_section_scour_susceptibility(
        section_mobility_df=mobility_df,
        actual_embedment_m=0.10 * 0.3048,
        diameter_m=0.3048,
        evidence_type=susceptibility.EVIDENCE_TYPE_SITE_SPECIFIC,
    )
    assert row["screening_state"] == susceptibility.OUTSIDE_METHOD_EXPERIMENTAL_ENVELOPE
    assert row["embedment_protection_margin_p95_e_over_D"] is None


def test_actual_embedment_above_tested_envelope_is_outside_method_envelope():
    mobility_df = _single_scenario_mobility_df("HP1", [0.06] * 20)
    row, _detail = susceptibility.compute_section_scour_susceptibility(
        section_mobility_df=mobility_df,
        actual_embedment_m=0.20 * 0.3048,  # above the largest tested ratio, 0.15
        diameter_m=0.3048,
        evidence_type=susceptibility.EVIDENCE_TYPE_SITE_SPECIFIC,
    )
    assert row["screening_state"] == susceptibility.OUTSIDE_METHOD_EXPERIMENTAL_ENVELOPE


# --- regression: a scenario embedment must never "exceed" its own identical critical class -
#
# Real bug found via visual inspection of the actual PL854 scenario-envelope figure (a
# tested-scenario line that should have been strictly LOWER exceedance than a smaller
# embedment scenario was instead identical to it): `actual_embedment_m = scenario_ratio *
# diameter_m` followed by `actual_embedment_ratio = actual_embedment_m / diameter_m` does not
# always round-trip to the exact same float (confirmed: 0.03 * 0.3048 / 0.3048 ==
# 0.029999999999999995 != 0.03 for PL854's real diameter), which without a tolerance made the
# 0.03 D scenario spuriously "exceed" a real timestep whose required embedment was exactly
# the identical tested class 0.03.


def test_scenario_embedment_never_spuriously_exceeds_its_own_identical_tested_class():
    diameter_m = 0.3048
    # every timestep needs exactly e/D=0.03 to suppress onset -- the SAME class this
    # scenario's own actual embedment will be set to.
    mobility_df = _single_scenario_mobility_df("HP1", [0.03] * 100)
    scenario_actual_m = 0.03 * diameter_m
    # confirm this reproduces the real, concrete float round-trip imprecision first, so this
    # test cannot silently stop testing the real failure mode if float behaviour ever changes.
    assert scenario_actual_m / diameter_m != 0.03

    valid_count, triggered_count, fraction = susceptibility.compute_scenario_exceedance(
        mobility_df, actual_embedment_ratio=scenario_actual_m / diameter_m
    )
    assert valid_count == 100
    assert triggered_count == 0
    assert fraction == pytest.approx(0.0)


def test_scenario_envelope_exceedance_is_monotonic_across_real_shaped_mixed_ratios():
    # a real-shaped mix: most timesteps need no embedment at all, some need exactly 0.03 --
    # the exact pattern found in the real PL854 data that exposed the round-trip bug.
    ratios = [0.00] * 700 + [0.03] * 300
    mobility_df = _single_scenario_mobility_df("HP1", ratios)
    fractions = []
    for scenario_ratio in susceptibility.TESTED_EMBEDMENT_SCENARIOS:
        actual_embedment_m = scenario_ratio * 0.3048
        _valid, _triggered, fraction = susceptibility.compute_scenario_exceedance(
            mobility_df, actual_embedment_ratio=actual_embedment_m / 0.3048
        )
        fractions.append(fraction)
    assert fractions == sorted(fractions, reverse=True)
    # the 0.03 D scenario must show ZERO exceedance (every timestep needs AT MOST 0.03,
    # never strictly more) -- not the same 30% as the 0.00 D scenario.
    assert fractions[0] == pytest.approx(0.30)
    assert fractions[1] == pytest.approx(0.0)


# --- report/contract smoke tests --------------------------------------------------------------


def test_report_blocks_render_to_html():
    blocks = poc_report.build_scour_poc_report_blocks(
        project_title="Test",
        purpose_text="purpose",
        pipeline_method_facts={"a": 1},
        required_inputs=["x"],
        actual_vs_critical_facts={"b": 2},
        exceedance_fraction_facts={"c": 3},
        pl854_scenario_facts={"d": 4},
        pl854_limitations=["limit"],
        sheringham_evidence_facts={"e": 5},
        asset_physics_mismatch_text="mismatch",
        production_transfer_contract_summary=["contract"],
        not_predicted=["future scour depth"],
    )
    html = poc_report.render_blocks_html(blocks, title="Test Report")
    assert "<html" in html
    assert "PIPELINE SCOUR SUSCEPTIBILITY REQUIRES ACTUAL PIPELINE EMBEDMENT." in html


def test_input_contract_states_no_profile_means_no_result():
    built = contract.build_scour_susceptibility_input_contract()
    joined = " ".join(built["notes"])
    assert "NO ACTUAL EMBEDMENT PROFILE" in joined
    assert "NO SITE-SPECIFIC SCOUR SUSCEPTIBILITY RESULT" in joined


# --- Section 23: CLI wiring -------------------------------------------------------------------


def test_cli_registers_build_scour_susceptibility_poc():
    parser = cli.build_parser()
    args = parser.parse_args(["build-scour-susceptibility-poc", "configs/pl854.yaml"])
    assert args.func is cli._cmd_build_scour_susceptibility_poc


def test_provider_download_url_matches_confirmed_real_package():
    assert sheringham_shoal_2024.INTERPRETATION_DATA_URL.endswith(
        "122_3974_Interpretation%20Data.zip"
    )
    assert sheringham_shoal_2024.TARGETS_LAYER_NAME == (
        "00965-REA-ENG-BATH_SheringhamShoal_Targets_Rev01"
    )

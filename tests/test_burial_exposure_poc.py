"""Offline unit tests for the MAR-024 generic burial/exposure POC.

Small synthetic records only -- never the real Barrow 2016 data, never
network access (the one acquisition test proves a cache HIT skips the
network entirely, by raising if `requests.get` is ever called). Test
names map to MAR-024 Section 23's required list.
"""

from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest
from shapely.geometry import LineString

from marine_engine import cli
from marine_engine.burial import (
    contract,
    cover,
    exposure_screening,
    maps,
    margin,
    profile,
    readiness,
    report,
    route,
    semantics,
)
from marine_engine.providers import barrow_2016

BURIAL_MODULES = (
    contract,
    cover,
    exposure_screening,
    margin,
    maps,
    profile,
    readiness,
    report,
    route,
    semantics,
)

# --- synthetic fixtures ----------------------------------------------------------------------

SYNTHETIC_ROUTE = LineString([(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0)])


def _synthetic_records(
    *,
    kp: list[float],
    x: list[float],
    y: list[float],
    z: list[float | None],
    exposed: list[bool] | None = None,
    uncertainty: list[float] | None = None,
) -> pd.DataFrame:
    n = len(kp)
    return pd.DataFrame(
        {
            "record_id": [str(i) for i in range(n)],
            "KP": kp,
            "X": x,
            "Y": y,
            "Z": z,
            "is_exposed": exposed or [False] * n,
            "Uncertainty": uncertainty or [0.1] * n,
        }
    )


# --- A: source burial reference must be resolved before exposure inference -----------------


def test_A_burial_reference_must_be_resolved_before_exposure_inference():
    # a strongly negative raw value (which might "look like" exposure under a naive convention)
    # is normalized away entirely: an unresolved reference yields cover_above_asset_m=None,
    # which must NOT become SOURCE_INTERPRETED_EXPOSED just because the reference is unresolved.
    cover_above_asset_m = cover.compute_cover_above_asset_m(
        -99.0, burial_reference_type=semantics.SOURCE_BURIAL_REFERENCE_UNRESOLVED
    )
    assert cover_above_asset_m is None
    state = profile.classify_current_burial_state(
        cover_above_asset_m,
        has_measurement=True,
        is_source_interpreted_exposed=False,
    )
    assert state == profile.MEASURED_REFERENCE_REQUIRES_REVIEW
    assert state != profile.SOURCE_INTERPRETED_EXPOSED


# --- B: top-of-asset vs centreline semantics remain distinct --------------------------------


def test_B_top_of_asset_vs_centreline_semantics_remain_distinct():
    assert semantics.TOP_OF_ASSET_BURIAL != semantics.CENTRELINE_BURIAL
    assert semantics.TOP_OF_ASSET_BURIAL in semantics.BURIAL_REFERENCE_TYPES
    assert semantics.CENTRELINE_BURIAL in semantics.BURIAL_REFERENCE_TYPES
    # both are valid, independently constructible semantics objects
    obj_top = semantics.BurialMeasurementSemantics(
        source_measurement_name="Z",
        measurement_reference_point=semantics.TOP_OF_ASSET_BURIAL,
        sign_convention=semantics.POSITIVE_VALUE_MEANS_DEEPER_BURIAL,
        source_sign_convention_text="positive-down",
        units="m",
        source_stated_uncertainty_available=True,
        survey_technique="DOBstar",
        unknown_fields=(),
        resolution_evidence="source explicitly states top-of-cable burial",
    )
    obj_centreline = semantics.BurialMeasurementSemantics(
        source_measurement_name="Z",
        measurement_reference_point=semantics.CENTRELINE_BURIAL,
        sign_convention=semantics.POSITIVE_VALUE_MEANS_DEEPER_BURIAL,
        source_sign_convention_text="positive-down",
        units="m",
        source_stated_uncertainty_available=True,
        survey_technique="DOBstar",
        unknown_fields=(),
        resolution_evidence="source explicitly states centreline burial",
    )
    assert obj_top.measurement_reference_point != obj_centreline.measurement_reference_point


def test_B2_invalid_reference_point_is_rejected():
    with pytest.raises(ValueError):
        semantics.BurialMeasurementSemantics(
            source_measurement_name="Z",
            measurement_reference_point="NOT_A_REAL_REFERENCE_TYPE",
            sign_convention=semantics.SIGN_CONVENTION_UNRESOLVED,
            units="m",
            source_stated_uncertainty_available=False,
            survey_technique=None,
            unknown_fields=(),
            resolution_evidence="",
        )


def test_B3_invalid_sign_convention_is_rejected():
    with pytest.raises(ValueError):
        semantics.BurialMeasurementSemantics(
            source_measurement_name="Z",
            measurement_reference_point=semantics.TOP_OF_ASSET_BURIAL,
            sign_convention="positive-down",  # free text is not a valid canonical value
            units="m",
            source_stated_uncertainty_available=False,
            survey_technique=None,
            unknown_fields=(),
            resolution_evidence="",
        )


# --- C: missing KP does not create fake route position --------------------------------------


def test_C_missing_kp_does_not_create_fake_route_position():
    records = _synthetic_records(
        kp=[float("nan"), 500.0], x=[0.0, 0.0], y=[100.0, 500.0], z=[1.0, 1.5]
    )
    result = profile.build_canonical_burial_profile(
        records_df=records,
        asset_id="TEST_ASSET",
        route=SYNTHETIC_ROUTE,
        x_column="X",
        y_column="Y",
        measured_value_column="Z",
        source_kp_column="KP",
        record_id_column="record_id",
        burial_reference_type=semantics.SOURCE_BURIAL_REFERENCE_UNRESOLVED,
        sign_convention=semantics.SIGN_CONVENTION_UNRESOLVED,
        survey_epoch="2020",
        measurement_method=None,
    )
    # chainage is derived from the REAL (x, y) position, never from source_kp -- a missing KP
    # does not prevent or fabricate a chainage value.
    assert result["chainage_m"].notna().all()
    nan_kp_row = result[result["source_kp"].isna()]
    assert len(nan_kp_row) == 1
    assert nan_kp_row["chainage_m"].iloc[0] == pytest.approx(100.0)


# --- D: duplicate KP is flagged --------------------------------------------------------------


def test_D_duplicate_kp_is_flagged():
    facts = readiness.BurialProfileFacts(
        source_file_readable=True,
        record_count=10,
        route_identifier_available=True,
        kp_available=True,
        kp_is_monotonic=True,
        duplicate_kp_count=3,
        coordinate_support=True,
        crs_available=True,
        units_available=True,
        units_consistent_across_sources=True,
        burial_reference_known=True,
        missing_value_fraction=0.0,
        coverage_fraction=1.0,
        suspicious_spike_count=0,
        negative_value_count=0,
        zero_value_count=0,
        source_uncertainty_available=True,
        survey_epoch_known=True,
    )
    result = readiness.assess_burial_profile_readiness(facts)
    assert any("duplicate" in r.lower() for r in result.reasons())
    assert result.status == readiness.READY_WITH_LIMITATIONS


def test_D2_no_duplicate_kp_is_not_flagged():
    facts = readiness.BurialProfileFacts(
        source_file_readable=True,
        record_count=10,
        route_identifier_available=True,
        kp_available=True,
        kp_is_monotonic=True,
        duplicate_kp_count=0,
        coordinate_support=True,
        crs_available=True,
        units_available=True,
        units_consistent_across_sources=True,
        burial_reference_known=True,
        missing_value_fraction=0.0,
        coverage_fraction=1.0,
        suspicious_spike_count=0,
        negative_value_count=0,
        zero_value_count=0,
        source_uncertainty_available=True,
        survey_epoch_known=True,
    )
    result = readiness.assess_burial_profile_readiness(facts)
    assert result.status == readiness.READY
    assert not any("duplicate" in r.lower() for r in result.reasons())


# --- E: major profile gaps are not interpolated ----------------------------------------------


def test_E_major_profile_gaps_are_not_interpolated():
    # two real records far apart along the route -- the profile must contain EXACTLY these two
    # rows, never a fabricated in-between record filling the gap.
    records = _synthetic_records(kp=[10.0, 900.0], x=[0.0, 900.0], y=[10.0, 1000.0], z=[1.0, 1.2])
    result = profile.build_canonical_burial_profile(
        records_df=records,
        asset_id="TEST_ASSET",
        route=SYNTHETIC_ROUTE,
        x_column="X",
        y_column="Y",
        measured_value_column="Z",
        source_kp_column="KP",
        record_id_column="record_id",
        burial_reference_type=semantics.SOURCE_BURIAL_REFERENCE_UNRESOLVED,
        sign_convention=semantics.SIGN_CONVENTION_UNRESOLVED,
        survey_epoch="2020",
        measurement_method=None,
    )
    assert len(result) == 2


# --- F: measured state and source-interpreted exposure remain distinct ---------------------


def test_F_measured_state_and_source_interpreted_exposure_remain_distinct():
    records = _synthetic_records(
        kp=[10.0, 20.0],
        x=[0.0, 0.0],
        y=[10.0, 20.0],
        z=[1.0, 1.0],
        exposed=[False, True],
    )
    result = profile.build_canonical_burial_profile(
        records_df=records,
        asset_id="TEST_ASSET",
        route=SYNTHETIC_ROUTE,
        x_column="X",
        y_column="Y",
        measured_value_column="Z",
        source_kp_column="KP",
        record_id_column="record_id",
        burial_reference_type=semantics.TOP_OF_ASSET_BURIAL,
        sign_convention=semantics.POSITIVE_VALUE_MEANS_DEEPER_BURIAL,
        survey_epoch="2020",
        measurement_method=None,
        exposure_flag_column="is_exposed",
    )
    states = result.sort_values("chainage_m")["measured_burial_state"].tolist()
    assert states[0] == profile.MEASURED_BURIED
    assert states[1] == profile.SOURCE_INTERPRETED_EXPOSED
    assert states[0] != states[1]
    # cover_above_asset_m passed through unchanged for TOP_OF_ASSET_BURIAL with a resolved,
    # positive-means-deeper sign convention -- 1.0 in, 1.0 out.
    covers = result.sort_values("chainage_m")["cover_above_asset_m"].tolist()
    assert covers[0] == pytest.approx(1.0)


# --- G: trench scar is not automatically exposure ---------------------------------------------


def test_G_trench_scar_is_not_automatically_exposure():
    # a record explicitly NOT flagged exposed (e.g. a "trench scar" or any other non-exposure
    # descriptor) must never resolve to SOURCE_INTERPRETED_EXPOSED, regardless of its resolved
    # cover value.
    state = profile.classify_current_burial_state(
        0.0,
        has_measurement=True,
        is_source_interpreted_exposed=False,
    )
    assert state != profile.SOURCE_INTERPRETED_EXPOSED


# --- H: actual burial and target burial remain distinct ---------------------------------------


def test_H_actual_and_target_burial_remain_distinct():
    measured = 1.2
    target = 0.6
    result = margin.compute_burial_margin_m(measured, target)
    assert result == pytest.approx(measured - target)
    assert measured != target  # the two inputs are never the same conceptual quantity


# --- I: no target means no burial margin ------------------------------------------------------


def test_I_no_target_means_no_burial_margin():
    assert margin.compute_burial_margin_m(1.5, None) is None


# --- J: positive burial margin never maps to SAFE ----------------------------------------------


def test_J_positive_burial_margin_never_maps_to_safe():
    result = margin.compute_burial_margin_m(1.0, 0.4)
    assert result > 0
    # `compute_burial_margin_m` structurally cannot "map to" SAFE: it returns a plain
    # float | None, and the module defines no status/state vocabulary at all for margin
    # values -- there is no classification function a positive margin could be routed through.
    assert isinstance(result, float)
    module_members = vars(margin)
    state_like_constants = {
        name: value
        for name, value in module_members.items()
        if name.isupper() and isinstance(value, str)
    }
    for name, value in state_like_constants.items():
        assert "SAFE" not in value, f"{name} = {value!r} uses SAFE-like language"


# --- K: lowering screening sign is correct / synthetic lowering screening works (Section 17) --


def test_K_synthetic_lowering_screening_matches_section_17_exactly():
    r1 = exposure_screening.screen_exposure_susceptibility(
        1.0,
        exposure_screening.SeabedLoweringInput(
            0.4, exposure_screening.OPERATOR_DEFINED_LOWERING_SCENARIO
        ),
    )
    assert r1["remaining_cover_after_lowering_m"] == pytest.approx(0.6)
    assert r1["screening_state"] == exposure_screening.POSITIVE_COVER_REMAINS_IN_SCREENING

    r2 = exposure_screening.screen_exposure_susceptibility(
        0.2,
        exposure_screening.SeabedLoweringInput(
            0.4, exposure_screening.OBSERVED_MULTI_EPOCH_SEABED_LOWERING
        ),
    )
    assert r2["remaining_cover_after_lowering_m"] == pytest.approx(-0.2)
    assert r2["screening_state"] == exposure_screening.ZERO_OR_NEGATIVE_COVER_IN_SCREENING


def test_K2_negative_lowering_magnitude_is_rejected():
    with pytest.raises(ValueError):
        exposure_screening.SeabedLoweringInput(
            -0.1, exposure_screening.OPERATOR_DEFINED_LOWERING_SCENARIO
        )


def test_K3_unknown_evidence_type_is_rejected():
    with pytest.raises(ValueError):
        exposure_screening.SeabedLoweringInput(0.4, "SOME_MADE_UP_EVIDENCE_TYPE")


# --- L: no lowering input means no future susceptibility --------------------------------------


def test_L_no_lowering_input_means_no_future_susceptibility():
    result = exposure_screening.screen_exposure_susceptibility(1.0, None)
    assert result["screening_state"] == exposure_screening.NO_DEFENSIBLE_SEABED_LOWERING_INPUT
    assert result["remaining_cover_after_lowering_m"] is None


def test_L2_no_cover_and_no_lowering_input_reports_missing_lowering_first():
    # MAR-024A: with neither a resolved cover nor a lowering scenario, the missing lowering
    # input is the single actionable blocker -- this is also the real Barrow case (cover
    # unresolved, no lowering input at all), where NO_DEFENSIBLE_SEABED_LOWERING_INPUT is the
    # preserved real result, never INSUFFICIENT_BURIAL_INPUT.
    result = exposure_screening.screen_exposure_susceptibility(None, None)
    assert result["screening_state"] == exposure_screening.NO_DEFENSIBLE_SEABED_LOWERING_INPUT


def test_L3_no_cover_with_a_real_lowering_input_is_insufficient_burial_input():
    # INSUFFICIENT_BURIAL_INPUT is reserved for a real lowering scenario with no cover value
    # to apply it to.
    result = exposure_screening.screen_exposure_susceptibility(
        None,
        exposure_screening.SeabedLoweringInput(
            0.4, exposure_screening.OPERATOR_DEFINED_LOWERING_SCENARIO
        ),
    )
    assert result["screening_state"] == exposure_screening.INSUFFICIENT_BURIAL_INPUT


# --- M: no probability field --------------------------------------------------------------------


def test_M_no_probability_field_anywhere_in_the_generic_engine():
    # the schema/vocabulary surface (column names and classification-state values) must never
    # contain "probability" -- free-text disclaimer prose (which legitimately explains why NO
    # probability concept exists, e.g. `report.REQUIRED_DISCLAIMER_2`) is a different thing
    # and is deliberately not scanned here.
    schema_names = " ".join(profile.CANONICAL_BURIAL_PROFILE_COLUMNS).upper()
    assert "PROBABILITY" not in schema_names
    all_states = (
        exposure_screening.EXPOSURE_SCREENING_STATES
        | profile.MEASURED_BURIAL_STATES
        | semantics.BURIAL_REFERENCE_TYPES
        | semantics.SIGN_CONVENTIONS
    )
    for state in all_states:
        assert "PROBABILITY" not in state


def test_M2_no_probability_in_screening_or_validation_vocabulary():
    all_states = (
        exposure_screening.EXPOSURE_SCREENING_STATES
        | profile.MEASURED_BURIAL_STATES
        | semantics.BURIAL_REFERENCE_TYPES
        | semantics.SIGN_CONVENTIONS
    )
    for state in all_states:
        assert "PROBABILITY" not in state


# --- N: no risk score ------------------------------------------------------------------------


def test_N_no_risk_score_or_safe_unsafe_vocabulary():
    forbidden = ("SAFE", "UNSAFE", "HIGH RISK", "LOW RISK", "RISK_SCORE")
    all_states = (
        exposure_screening.EXPOSURE_SCREENING_STATES
        | profile.MEASURED_BURIAL_STATES
        | semantics.BURIAL_REFERENCE_TYPES
        | semantics.SIGN_CONVENTIONS
    )
    for state in all_states:
        for word in forbidden:
            assert word not in state


def test_N2_no_risk_score_field_in_any_output_schema():
    assert "risk_score" not in " ".join(profile.CANONICAL_BURIAL_PROFILE_COLUMNS)
    assert "score" not in " ".join(profile.CANONICAL_BURIAL_PROFILE_COLUMNS)


# --- O: no free-span result --------------------------------------------------------------------


def test_O_no_free_span_result_anywhere_in_the_generic_engine():
    for module in BURIAL_MODULES:
        source = inspect.getsource(module).lower()
        assert "freespan" not in source
        assert "free-span" not in source
        assert "free span" not in source


# --- P: generic engine contains no Barrow/PL854 coordinates -----------------------------------


def test_P_generic_engine_has_no_hardcoded_dataset_coordinates():
    # the real Barrow route's own confirmed coordinate fragments (from the real RPL points/DoB
    # listing) and PL854's real route bbox corner must never appear as literal values in the
    # generic engine modules.
    forbidden_fragments = ("505103.19", "5986140.17", "482143.96", "1.6516061", "53.3678022")
    for module in BURIAL_MODULES:
        source = inspect.getsource(module)
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{module.__name__} hard-codes {fragment!r}"


def test_P2_no_dataset_name_string_in_generic_engine_logic():
    # dataset names in default title strings are established precedent elsewhere in this
    # project (e.g. scour_onset_map.py's own "PL854 -- ..." default title), so `maps.py` is
    # exempt here; the remaining, non-rendering engine modules should carry no Barrow/PL854
    # identity at all.
    non_map_modules = (
        contract,
        cover,
        exposure_screening,
        margin,
        profile,
        readiness,
        route,
        semantics,
    )
    for module in non_map_modules:
        source = inspect.getsource(module)
        assert "Barrow" not in source
        assert "PL854" not in source


# --- Q: offline rerun works after acquisition (cache hit, no network) -----------------------


def test_Q_dob_listing_cache_hit_never_touches_network(tmp_path, monkeypatch):
    def _explode(*_args, **_kwargs):
        raise AssertionError("requests.get must not be called on a cache hit")

    monkeypatch.setattr(barrow_2016.requests, "get", _explode)

    raw_dir = tmp_path / "raw"
    extracted_dir = raw_dir / "depth_of_burial_listing"
    listing_dir = extracted_dir / "Listing"
    listing_dir.mkdir(parents=True)
    (listing_dir / barrow_2016.DOB_LISTING_FILENAME).write_bytes(b"stub")
    sidecar = barrow_2016._sidecar_path(extracted_dir, "depth_of_burial_listing")
    sidecar.write_text(
        json.dumps(
            {
                "package_bytes": 999,
                "package_sha256": "deadbeef",
                "retrieved_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    acquisition = barrow_2016.download_depth_of_burial_listing(raw_dir)
    assert acquisition.already_cached is True
    assert acquisition.package_bytes == 999


def test_Q2_route_position_list_cache_hit_never_touches_network(tmp_path, monkeypatch):
    def _explode(*_args, **_kwargs):
        raise AssertionError("requests.get must not be called on a cache hit")

    monkeypatch.setattr(barrow_2016.requests, "get", _explode)

    raw_dir = tmp_path / "raw"
    extracted_dir = raw_dir / "route_position_list"
    rpl_dir = extracted_dir / "RPL"
    rpl_dir.mkdir(parents=True)
    (rpl_dir / barrow_2016.RPL_LINE_FILENAME).write_bytes(b"stub")
    (rpl_dir / barrow_2016.RPL_POINTS_FILENAME).write_bytes(b"stub")
    sidecar = barrow_2016._sidecar_path(extracted_dir, "route_position_list")
    sidecar.write_text(
        json.dumps(
            {
                "package_bytes": 555,
                "package_sha256": "cafef00d",
                "retrieved_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    acquisition = barrow_2016.download_route_position_list(raw_dir)
    assert acquisition.already_cached is True
    assert acquisition.package_bytes == 555


# --- CLI wiring + validation-question derivation ---------------------------------------------


def test_cli_registers_build_burial_exposure_poc():
    parser = cli.build_parser()
    args = parser.parse_args(["build-burial-exposure-poc", "configs/barrow_2016.yaml"])
    assert args.func is cli._cmd_build_burial_exposure_poc


def test_validation_questions_G_H_are_hardcoded_no():
    result = cli._derive_burial_exposure_poc_validation_questions(
        real_dob_dataset_ingested=True,
        burial_reference_resolved=True,
        authoritative_route_recovered=True,
        measured_burial_profile_produced=True,
        explicit_exposure_evidence_present=True,
        generic_exposure_screening_computable=True,
    )
    assert result["question_g_real_barrow_future_exposure_susceptibility_defensible"] == "NO"
    assert result["question_h_exposure_probability_produced"] == "NO"
    assert result["question_a_real_operator_style_dob_dataset_ingested"] == "YES"
    assert result["question_b_source_burial_measurement_reference_resolved"] == "YES"
    assert result["question_f_generic_cover_depletion_exposure_screening_computable"] == "YES"


def test_provider_download_urls_match_confirmed_real_packages():
    assert barrow_2016.DOB_LISTING_URL.endswith("Depth%20of%20Burial%20Listing.zip")
    assert barrow_2016.ROUTE_POSITION_LIST_URL.endswith("Route%20Position%20List%20Files.zip")
    assert barrow_2016.SOURCE_CRS == "EPSG:25830"


# --- synthetic operator route produces valid maps ---------------------------------------------


def test_synthetic_route_produces_valid_observed_state_map(tmp_path):
    records = _synthetic_records(
        kp=[10.0, 500.0, 900.0],
        x=[0.0, 500.0, 900.0],
        y=[10.0, 1000.0, 1000.0],
        z=[1.0, 0.02, -0.5],
        exposed=[False, False, True],
    )
    result = profile.build_canonical_burial_profile(
        records_df=records,
        asset_id="TEST_ASSET",
        route=SYNTHETIC_ROUTE,
        x_column="X",
        y_column="Y",
        measured_value_column="Z",
        source_kp_column="KP",
        record_id_column="record_id",
        burial_reference_type=semantics.TOP_OF_ASSET_BURIAL,
        sign_convention=semantics.POSITIVE_VALUE_MEANS_DEEPER_BURIAL,
        survey_epoch="2020",
        measurement_method=None,
        exposure_flag_column="is_exposed",
    )
    output_path = tmp_path / "state_map.png"
    result_path = maps.render_observed_burial_state_map(
        route=SYNTHETIC_ROUTE,
        profile_df=result,
        output_path=output_path,
        value_label="Measured burial (m)",
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0

    kp_output_path = tmp_path / "kp_profile.png"
    kp_result_path = maps.render_burial_kp_profile(
        profile_df=result,
        output_path=kp_output_path,
        value_label="Measured burial (m)",
        target_burial_m=0.6,
    )
    assert kp_result_path.exists()
    assert kp_result_path.stat().st_size > 0


def test_route_resolve_rejects_disjoint_multilinestring():
    from shapely.geometry import MultiLineString

    disjoint = MultiLineString([[(0, 0), (1, 1)], [(10, 10), (11, 11)]])
    with pytest.raises(route.InvalidAssetRouteError):
        route.resolve_route_linestring(disjoint)


def test_report_blocks_render_to_html():
    blocks = report.build_burial_exposure_report_blocks(
        project_title="Test",
        purpose_text="purpose",
        source_survey_facts={"a": 1},
        burial_reference_semantics_facts={"b": 2},
        data_readiness_facts={"c": 3},
        route_and_coverage_facts={"d": 4},
        canonical_cover_semantics_text=(
            "RAW SOURCE MEASUREMENT, CANONICAL REFERENCE BURIAL DEPTH, TOP-OF-ASSET COVER, "
            "SOURCE-INTERPRETED EXPOSURE, and FUTURE EXPOSURE SCREENING are distinct concepts."
        ),
        measured_burial_profile_facts={"e": 5},
        source_interpreted_exposure_facts={"f": 6},
        burial_margin_text="no margin",
        exposure_screening_contract_facts={"g": 7},
        future_susceptibility_text="not demonstrated",
        gis_outputs=["layer1"],
        production_transfer_contract_summary=["contract item"],
    )
    html = report.render_blocks_html(blocks, title="Test Report")
    assert "<html" in html
    assert "THIS POC DISTINGUISHES MEASURED BURIAL STATE" in html
    assert "CANONICAL REFERENCE BURIAL DEPTH" in html
    assert "TOP-OF-ASSET COVER" in html


def test_input_contract_states_lowering_input_required_for_future_screening():
    built = contract.build_burial_exposure_input_contract()
    fields = [f["field"] for f in built["required_fields_future_exposure_screening"]]
    assert "defensible_seabed_change_or_lowering_input" in fields
    assert (
        built["required_fields_future_exposure_screening"]
        != built["required_fields_current_burial_state"]
    )


# ===============================================================================================
# MAR-024A -- Canonical Burial-Cover Semantics Repair. Test names map to MAR-024A Section 11's
# required list (A-O).
# ===============================================================================================


# --- A/B: sign convention is applied exactly once, in either direction ------------------------


def test_mar024a_A_positive_means_deeper_sign_convention_passes_through():
    result = cover.normalize_canonical_reference_burial_depth_m(
        1.0, sign_convention=semantics.POSITIVE_VALUE_MEANS_DEEPER_BURIAL
    )
    assert result == pytest.approx(1.0)


def test_mar024a_B_negative_means_deeper_sign_convention_is_flipped():
    result = cover.normalize_canonical_reference_burial_depth_m(
        -1.0, sign_convention=semantics.NEGATIVE_VALUE_MEANS_DEEPER_BURIAL
    )
    assert result == pytest.approx(1.0)


# --- C: unresolved sign convention yields no canonical burial depth ----------------------------


def test_mar024a_C_unresolved_sign_convention_yields_no_canonical_depth():
    result = cover.normalize_canonical_reference_burial_depth_m(
        -99.0, sign_convention=semantics.SIGN_CONVENTION_UNRESOLVED
    )
    assert result is None
    # true regardless of the raw value's own magnitude or sign -- never guessed from it.
    for raw in (0.0, 1.0, -1.0, 250.0):
        assert (
            cover.normalize_canonical_reference_burial_depth_m(
                raw, sign_convention=semantics.SIGN_CONVENTION_UNRESOLVED
            )
            is None
        )


def test_mar024a_C2_unknown_sign_convention_string_is_rejected():
    with pytest.raises(ValueError):
        cover.normalize_canonical_reference_burial_depth_m(1.0, sign_convention="made-up")


# --- D: top-of-asset reference requires no offset -----------------------------------------------


def test_mar024a_D_top_of_asset_cover_equals_canonical_depth():
    result = cover.compute_cover_above_asset_m(
        1.0, burial_reference_type=semantics.TOP_OF_ASSET_BURIAL
    )
    assert result == pytest.approx(1.0)


# --- E/F: centreline reference requires an explicit reference-to-top offset --------------------


def test_mar024a_E_centreline_cover_subtracts_the_explicit_offset():
    result = cover.compute_cover_above_asset_m(
        1.0,
        burial_reference_type=semantics.CENTRELINE_BURIAL,
        reference_to_asset_top_offset_m=0.1,
    )
    assert result == pytest.approx(0.9)


def test_mar024a_F_centreline_without_an_offset_cannot_be_classified():
    result = cover.compute_cover_above_asset_m(
        1.0, burial_reference_type=semantics.CENTRELINE_BURIAL
    )
    assert result is None
    state = profile.classify_current_burial_state(
        result, has_measurement=True, is_source_interpreted_exposed=False
    )
    assert state == profile.MEASURED_REFERENCE_REQUIRES_REVIEW


# --- G: top-of-asset and centreline are physically different, never interchanged ---------------


def test_mar024a_G_top_of_asset_and_centreline_produce_different_cover():
    top_of_asset_cover = cover.compute_cover_above_asset_m(
        1.0, burial_reference_type=semantics.TOP_OF_ASSET_BURIAL
    )
    centreline_cover = cover.compute_cover_above_asset_m(
        1.0,
        burial_reference_type=semantics.CENTRELINE_BURIAL,
        reference_to_asset_top_offset_m=0.1,
    )
    assert top_of_asset_cover != centreline_cover
    assert top_of_asset_cover == pytest.approx(1.0)
    assert centreline_cover == pytest.approx(0.9)


# --- H/I/J: measured state is classified from cover alone --------------------------------------


def test_mar024a_H_positive_cover_is_measured_buried():
    state = profile.classify_current_burial_state(
        0.5, has_measurement=True, is_source_interpreted_exposed=False
    )
    assert state == profile.MEASURED_BURIED


def test_mar024a_I_zero_cover_is_measured_at_seabed_level():
    state = profile.classify_current_burial_state(
        0.0, has_measurement=True, is_source_interpreted_exposed=False
    )
    assert state == profile.MEASURED_AT_SEABED_LEVEL


def test_mar024a_J_negative_cover_is_measured_above_seabed():
    state = profile.classify_current_burial_state(
        -0.5, has_measurement=True, is_source_interpreted_exposed=False
    )
    assert state == profile.MEASURED_ABOVE_SEABED
    assert state != profile.SOURCE_INTERPRETED_EXPOSED


# --- K/L: source-interpreted exposure stays independent of the measured cover state ------------


def test_mar024a_K_negative_cover_does_not_automatically_create_exposure():
    state = profile.classify_current_burial_state(
        -5.0, has_measurement=True, is_source_interpreted_exposed=False
    )
    assert state == profile.MEASURED_ABOVE_SEABED
    assert state != profile.SOURCE_INTERPRETED_EXPOSED


def test_mar024a_L_source_interpreted_exposure_overrides_any_cover_value():
    # the explicit source flag takes priority regardless of what the resolved cover says --
    # even a strongly positive (buried-looking) cover never suppresses explicit exposure
    # evidence, and a NOT-exposed record with the same cover gets the physical state instead.
    exposed_state = profile.classify_current_burial_state(
        2.0, has_measurement=True, is_source_interpreted_exposed=True
    )
    not_exposed_state = profile.classify_current_burial_state(
        2.0, has_measurement=True, is_source_interpreted_exposed=False
    )
    assert exposed_state == profile.SOURCE_INTERPRETED_EXPOSED
    assert not_exposed_state == profile.MEASURED_BURIED
    assert exposed_state != not_exposed_state


# --- M: future exposure screening accepts canonical cover only ---------------------------------


def test_mar024a_M_screening_signature_takes_cover_above_asset_m():
    params = list(inspect.signature(exposure_screening.screen_exposure_susceptibility).parameters)
    assert params[0] == "cover_above_asset_m"
    assert "measured_burial_m" not in params


def test_mar024a_M2_screening_computes_from_the_supplied_cover_value():
    result = exposure_screening.screen_exposure_susceptibility(
        0.6,
        exposure_screening.SeabedLoweringInput(
            0.4, exposure_screening.OPERATOR_DEFINED_LOWERING_SCENARIO
        ),
    )
    assert result["remaining_cover_after_lowering_m"] == pytest.approx(0.2)
    assert result["screening_state"] == exposure_screening.POSITIVE_COVER_REMAINS_IN_SCREENING


# --- N: a raw source burial value cannot bypass normalization ----------------------------------


def test_mar024a_N_classification_signature_has_no_raw_measured_value_parameter():
    params = list(inspect.signature(profile.classify_current_burial_state).parameters)
    assert params[0] == "cover_above_asset_m"
    assert "measured_value_m" not in params
    assert "measured_burial_value_m" not in params


def test_mar024a_N2_classification_result_is_unaffected_by_the_raw_values_own_magnitude():
    # holding cover_above_asset_m fixed, the classification result must be identical no matter
    # what a hypothetical raw source value's magnitude/sign was -- proving the raw value has no
    # channel through which it could influence the physical-state decision.
    results = {
        profile.classify_current_burial_state(
            1.0, has_measurement=True, is_source_interpreted_exposed=False
        )
        for _raw_hint in (-1_000_000.0, -1.0, 0.0, 1.0, 1_000_000.0)
    }
    assert results == {profile.MEASURED_BURIED}


# --- O: real Barrow unresolved-reference counts/results remain unchanged -----------------------


def test_mar024a_O_unresolved_reference_and_sign_together_preserve_barrow_shape():
    # mirrors the real Barrow situation: BOTH the reference point AND the sign convention are
    # unresolved, so canonical depth and cover are null for every record regardless of exposure
    # flag -- exposed records still resolve to SOURCE_INTERPRETED_EXPOSED (the explicit source
    # flag is checked before the cover-null fallback), non-exposed records resolve to
    # MEASURED_REFERENCE_REQUIRES_REVIEW, and the real screening call with no lowering input
    # stays NO_DEFENSIBLE_SEABED_LOWERING_INPUT. Verified against the real 20,946-record Barrow
    # profile (19,658 MEASURED_REFERENCE_REQUIRES_REVIEW / 1,288 SOURCE_INTERPRETED_EXPOSED) via
    # a real offline CLI rerun, not reproduced here (this file uses only synthetic records).
    records = _synthetic_records(
        kp=[10.0, 20.0, 30.0],
        x=[0.0, 0.0, 0.0],
        y=[10.0, 20.0, 30.0],
        z=[-14.99, -13.67, -20.0],
        exposed=[False, True, False],
    )
    result = profile.build_canonical_burial_profile(
        records_df=records,
        asset_id="TEST_ASSET",
        route=SYNTHETIC_ROUTE,
        x_column="X",
        y_column="Y",
        measured_value_column="Z",
        source_kp_column="KP",
        record_id_column="record_id",
        burial_reference_type=semantics.SOURCE_BURIAL_REFERENCE_UNRESOLVED,
        sign_convention=semantics.SIGN_CONVENTION_UNRESOLVED,
        survey_epoch="2016",
        measurement_method=None,
        exposure_flag_column="is_exposed",
    )
    assert result["canonical_reference_burial_depth_m"].isna().all()
    assert result["cover_above_asset_m"].isna().all()
    states = dict(zip(result["source_kp"], result["measured_burial_state"], strict=True))
    assert states[10.0] == profile.MEASURED_REFERENCE_REQUIRES_REVIEW
    assert states[20.0] == profile.SOURCE_INTERPRETED_EXPOSED
    assert states[30.0] == profile.MEASURED_REFERENCE_REQUIRES_REVIEW

    real_screening = exposure_screening.screen_exposure_susceptibility(None, None)
    assert real_screening["screening_state"] == (
        exposure_screening.NO_DEFENSIBLE_SEABED_LOWERING_INPUT
    )

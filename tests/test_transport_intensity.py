"""Offline unit tests for marine_engine.sediment.transport_intensity (MAR-030).

Small hand-built synthetic DataFrames only -- never real PL854 data, never
network access. Numbered comments map to MAR-030 Section 27's required test
matrix. Synthetic MAR-013 rows are built THROUGH the accepted MAR-013
helpers (`compute_mobility_ratio`, `classify_incipient_motion_status`) so
the source semantics under test are MAR-013's own, not re-invented here.

MAR-030A: `build_transport_intensity_3hourly` now enforces the MAR-013 source
contract (every row's role, the exact canonical nine-scenario vocabulary,
mm/m unit identity, unique source keys, per-timestamp scenario completeness,
status consistency), so every fixture handed to the builder is a complete
nine-scenario table (`_canonical_rows`); single-scenario `_mobility_rows`
frames are used only where a function is exercised directly.
"""

import ast
import inspect
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from marine_engine.sediment import noncohesive_mobility as ncm
from marine_engine.sediment import transport_intensity as ti

WORKING_CRS = "EPSG:32631"

# Section 26 synthetic core: explicit mobility ratios and their expected intensity.
CORE_RATIOS = [np.nan, 0.0, 0.5, 0.999, 1.0, 1.001, 1.25, 2.0, 5.0]
CORE_EXPECTED_INTENSITY = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.001, 0.25, 1.0, 4.0]


def _mobility_rows(
    ratios: list[float],
    *,
    pair_id: str = "pair_A",
    d50_mm: float = 0.25,
    tau_cr: float = 0.2,
    start: str = "2024-01-01T00:00:00Z",
) -> pd.DataFrame:
    """MAR-013-shaped rows whose mobility ratio equals `ratios` exactly.

    `tau_max = ratio * tau_cr` so `tau_max / tau_cr` reproduces the ratio; a NaN
    ratio is encoded the MAR-013 way -- `tau_cr = 0` makes the ratio undefined.
    """

    n = len(ratios)
    ratios_arr = np.asarray(ratios, dtype=float)
    tau_cr_arr = np.where(np.isnan(ratios_arr), 0.0, tau_cr)
    tau_max_arr = np.where(np.isnan(ratios_arr), 0.05, ratios_arr * tau_cr)
    mobility_ratio = ncm.compute_mobility_ratio(tau_max_arr, tau_cr_arr)
    status = ncm.classify_incipient_motion_status(mobility_ratio)
    times = pd.date_range(start, periods=n, freq="3h", tz="UTC")
    return pd.DataFrame(
        {
            "hydro_pair_id": pair_id,
            "current_node_id": f"current_{pair_id}",
            "wave_node_id": f"wave_{pair_id}",
            "time_utc": times,
            "tested_d50_mm": d50_mm,
            "tested_d50_m": d50_mm / 1000.0,
            "tau_max_grain_skin_pa": tau_max_arr,
            "tau_critical_pa": tau_cr_arr,
            "critical_shields_parameter": 0.04,
            "mobility_ratio": mobility_ratio,
            "incipient_motion_status": status,
            "scientific_role": ncm.SCIENTIFIC_ROLE,
        }
    )


def _all_scenario_rows(ratios_by_d50: dict[float, list[float]] | None = None) -> pd.DataFrame:
    """One block per tested D50 scenario (all nine), same pair, same timestamps."""

    frames = []
    for i, d50_mm in enumerate(sorted(ncm.TESTED_D50_SCENARIOS_MM)):
        ratios = (
            ratios_by_d50[d50_mm]
            if ratios_by_d50 is not None
            else [0.5 + 0.1 * i, 1.0, 1.5 + 0.2 * i, 3.0 - 0.1 * i]
        )
        frames.append(_mobility_rows(ratios, d50_mm=d50_mm, tau_cr=0.1 * (i + 1)))
    return pd.concat(frames, ignore_index=True)


def _canonical_rows(
    ratios: list[float],
    *,
    pair_id: str = "pair_A",
    tau_cr: float = 0.2,
    start: str = "2024-01-01T00:00:00Z",
) -> pd.DataFrame:
    """A contract-complete MAR-013 table: the SAME ratio series for all nine scenarios.

    The first block (0.063 mm) carries `ratios` in order, so `.loc[i]` and the
    first stats group refer to the same ratio as a single-scenario frame would.
    `tau_cr` is scaled per scenario so the stresses differ between scenarios as
    they would in MAR-013, while the ratio -- and hence the intensity -- is fixed.
    """

    frames = [
        _mobility_rows(ratios, pair_id=pair_id, d50_mm=d50_mm, tau_cr=tau_cr * (i + 1), start=start)
        for i, d50_mm in enumerate(sorted(ncm.TESTED_D50_SCENARIOS_MM))
    ]
    return pd.concat(frames, ignore_index=True)


def _contract_one_pair() -> dict:
    return ti.validate_mobility_source(_all_scenario_rows())


def _empty_contract() -> dict:
    return ti.validate_mobility_source(
        pd.DataFrame(columns=list(ncm.NONCOHESIVE_MOBILITY_3HOURLY_COLUMNS))
    )


# --- Formula (tests 1-7, Section 26) -------------------------------------------------


def test_26_core_intensity_matches_expected_within_floating_precision():
    intensity = ti.compute_relative_excess_shields_intensity(np.array(CORE_RATIOS))
    for got, expected in zip(intensity, CORE_EXPECTED_INTENSITY, strict=True):
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected, abs=1e-12)


def test_26_signed_stage_is_m_minus_one_for_finite_values():
    stage = ti.compute_relative_shields_stage(np.array(CORE_RATIOS))
    assert np.isnan(stage[0])
    finite = np.array(CORE_RATIOS[1:])
    assert stage[1:] == pytest.approx(finite - 1.0, abs=1e-12)
    assert stage[1] == pytest.approx(-1.0)
    assert stage[3] < 0  # M=0.999 -> negative stage


def test_1_below_threshold_intensity_is_zero():
    assert ti.compute_relative_excess_shields_intensity(np.array([0.0, 0.5, 0.999]))[
        0:3
    ].tolist() == [
        0.0,
        0.0,
        0.0,
    ]


def test_2_exactly_at_threshold_intensity_is_zero_and_stage_zero():
    assert ti.compute_relative_excess_shields_intensity(np.array([1.0]))[0] == 0.0
    assert ti.compute_relative_shields_stage(np.array([1.0]))[0] == 0.0


def test_3_above_threshold_intensity_is_m_minus_one():
    m = np.array([1.25, 2.0, 5.0, 8.97])
    assert ti.compute_relative_excess_shields_intensity(m) == pytest.approx(m - 1.0)


def test_4_undefined_mobility_remains_null_never_zero():
    out = ti.compute_relative_excess_shields_intensity(np.array([np.nan, np.inf, -np.inf]))
    assert np.isnan(out).all()
    stage = ti.compute_relative_shields_stage(np.array([np.nan, np.inf]))
    assert np.isnan(stage).all()


def test_5_intensity_never_negative():
    m = np.linspace(0.0, 3.0, 301)
    out = ti.compute_relative_excess_shields_intensity(m)
    assert (out >= 0.0).all()


def test_6_no_clipping_upper_bound():
    m = np.array([10.0, 1e3, 1e6])
    assert ti.compute_relative_excess_shields_intensity(m) == pytest.approx(m - 1.0)


def test_7_tiny_excess_preserved_not_rounded_away():
    out = ti.compute_relative_excess_shields_intensity(np.array([1.0 + 1e-9]))
    assert out[0] > 0.0
    assert out[0] == pytest.approx(1e-9, rel=1e-3)


# --- Algebraic QA (tests 8-10, Section 14) -------------------------------------------


def test_8_mobility_ratio_agrees_with_tau_max_over_tau_cr_on_mar013_rows():
    df = _all_scenario_rows()
    ti.verify_mobility_ratio_consistency(df)  # must not raise
    recomputed = df["tau_max_grain_skin_pa"] / df["tau_critical_pa"]
    assert np.allclose(df["mobility_ratio"], recomputed, rtol=1e-12, atol=0)


def test_9_inconsistent_source_row_triggers_controlled_qa_failure():
    df = _canonical_rows([0.5, 1.5, 2.0])
    df.loc[1, "mobility_ratio"] = 1.5 * 1.01  # 1% off its own stresses
    with pytest.raises(ti.MobilityRatioConsistencyError):
        ti.verify_mobility_ratio_consistency(df)
    with pytest.raises(ti.MobilityRatioConsistencyError):
        ti.build_transport_intensity_3hourly(df)


def test_9_finite_ratio_where_tau_cr_is_invalid_is_also_a_qa_failure():
    df = _mobility_rows([0.5, 1.5])
    df.loc[0, "tau_critical_pa"] = 0.0  # ratio stays finite -> inconsistent definedness
    with pytest.raises(ti.MobilityRatioConsistencyError):
        ti.verify_mobility_ratio_consistency(df)


def test_9_null_ratio_where_stresses_are_valid_is_a_qa_failure():
    df = _mobility_rows([0.5, 1.5])
    df.loc[1, "mobility_ratio"] = np.nan
    with pytest.raises(ti.MobilityRatioConsistencyError):
        ti.verify_mobility_ratio_consistency(df)


def test_9_qa_tolerance_accepts_floating_point_noise_only():
    df = _mobility_rows([1.5])
    df.loc[0, "mobility_ratio"] = 1.5 * (1.0 + 1e-13)
    ti.verify_mobility_ratio_consistency(df)  # noise-level difference accepted
    df.loc[0, "mobility_ratio"] = 1.5 * (1.0 + 1e-6)
    with pytest.raises(ti.MobilityRatioConsistencyError):
        ti.verify_mobility_ratio_consistency(df)


def test_10_invalid_tau_cr_cannot_produce_finite_intensity():
    df = _canonical_rows([np.nan, 2.0])  # first row encoded with tau_cr = 0
    assert df.loc[0, "tau_critical_pa"] == 0.0
    out = ti.build_transport_intensity_3hourly(df)
    assert np.isnan(out.loc[0, "relative_excess_shields_intensity"])
    assert np.isnan(out.loc[0, "relative_shields_stage"])
    assert out.loc[1, "relative_excess_shields_intensity"] == pytest.approx(1.0)


def test_negative_tau_cr_cannot_produce_finite_intensity():
    df = _canonical_rows([2.0])
    df.loc[0, "tau_critical_pa"] = -0.2
    df.loc[0, "mobility_ratio"] = ncm.compute_mobility_ratio(
        df["tau_max_grain_skin_pa"].to_numpy(), df["tau_critical_pa"].to_numpy()
    )[0]
    df.loc[0, "incipient_motion_status"] = None
    assert np.isnan(df.loc[0, "mobility_ratio"])
    out = ti.build_transport_intensity_3hourly(df)
    assert np.isnan(out.loc[0, "relative_excess_shields_intensity"])


# --- Source validation (Section 11) ----------------------------------------------------


@pytest.mark.parametrize(
    "column", ["mobility_ratio", "tau_critical_pa", "tau_max_grain_skin_pa", "hydro_pair_id"]
)
def test_missing_required_scientific_column_fails_cleanly(column: str):
    df = _mobility_rows([0.5, 1.5]).drop(columns=[column])
    with pytest.raises(ti.TransportIntensitySchemaError, match=column):
        ti.build_transport_intensity_3hourly(df)


def test_foreign_source_scientific_role_is_rejected():
    df = _mobility_rows([0.5, 1.5])
    df["scientific_role"] = "COMBINED_WAVE_CURRENT_BED_SHEAR_SENSITIVITY"
    with pytest.raises(ti.TransportIntensitySourceRoleError):
        ti.build_transport_intensity_3hourly(df)


def test_mixed_source_roles_are_rejected():
    df = _mobility_rows([0.5, 1.5])
    df.loc[0, "scientific_role"] = "SOMETHING_ELSE"
    with pytest.raises(ti.TransportIntensitySourceRoleError):
        ti.build_transport_intensity_3hourly(df)


def test_empty_source_yields_empty_output_with_full_schema():
    empty = pd.DataFrame(columns=list(ncm.NONCOHESIVE_MOBILITY_3HOURLY_COLUMNS))
    out = ti.build_transport_intensity_3hourly(empty)
    assert out.empty
    assert list(out.columns) == list(ti.TRANSPORT_INTENSITY_3HOURLY_COLUMNS)


# --- MAR-013 preservation (tests 11-17) ------------------------------------------------


def test_11_to_15_identity_timestamp_d50_and_status_preserved_row_for_row():
    df = _all_scenario_rows()
    out = ti.build_transport_intensity_3hourly(df)

    assert len(out) == len(df)  # 11
    assert out["hydro_pair_id"].tolist() == df["hydro_pair_id"].tolist()  # 12
    assert (out["time_utc"].to_numpy() == df["time_utc"].to_numpy()).all()  # 13
    assert out["tested_d50_mm"].tolist() == df["tested_d50_mm"].tolist()  # 14
    assert out["tested_d50_m"].tolist() == df["tested_d50_m"].tolist()
    assert out["incipient_motion_status"].tolist() == df["incipient_motion_status"].tolist()  # 15
    assert (out["mobility_ratio"].to_numpy() == df["mobility_ratio"].to_numpy()).all()
    assert (out["tau_max_grain_skin_pa"].to_numpy() == df["tau_max_grain_skin_pa"].to_numpy()).all()
    assert (out["tau_critical_pa"].to_numpy() == df["tau_critical_pa"].to_numpy()).all()


def test_15_exact_threshold_keeps_mar013_above_or_at_status_with_zero_excess():
    df = _canonical_rows([1.0])
    out = ti.build_transport_intensity_3hourly(df)
    assert out.loc[0, "incipient_motion_status"] == ncm.ABOVE_OR_AT_THRESHOLD
    assert out.loc[0, "relative_excess_shields_intensity"] == 0.0


def test_16_scientific_role_changed_only_for_the_new_derived_product():
    df = _all_scenario_rows()
    out = ti.build_transport_intensity_3hourly(df)
    assert set(out["scientific_role"]) == {ti.SCIENTIFIC_ROLE}
    assert set(out["source_scientific_role"]) == {ncm.SCIENTIFIC_ROLE}
    assert ti.SCIENTIFIC_ROLE != ncm.SCIENTIFIC_ROLE
    assert ti.SCIENTIFIC_ROLE == (
        "NONCOHESIVE_RELATIVE_EXCESS_SHIELDS_TRANSPORT_POTENTIAL_INTENSITY"
    )
    # The source frame itself is untouched.
    assert set(df["scientific_role"]) == {ncm.SCIENTIFIC_ROLE}


def test_16_builder_does_not_mutate_its_input():
    df = _all_scenario_rows()
    before = df.copy(deep=True)
    ti.build_transport_intensity_3hourly(df)
    pd.testing.assert_frame_equal(df, before)


def test_17_dependency_is_one_way_and_mar013_constants_are_reused_not_copied():
    mar013_source = inspect.getsource(ncm)
    assert "transport_intensity" not in mar013_source
    assert ti.TESTED_D50_SCENARIOS_MM is ncm.TESTED_D50_SCENARIOS_MM
    assert ti.GRAIN_SIZE_SCENARIO_SEMANTICS is ncm.GRAIN_SIZE_SCENARIO_SEMANTICS
    assert ti.SOURCE_SCIENTIFIC_ROLE is ncm.SCIENTIFIC_ROLE


def test_17_mar030_module_defines_no_stress_or_threshold_physics():
    mar030_source = inspect.getsource(ti)
    for forbidden in (
        "compute_soulsby_whitehouse",
        "compute_critical_shear_stress",
        "compute_wave_friction",
        "compute_current_friction_velocity",
        "compute_soulsby_max_combined",
        "combined_bed_shear",
        "z0_skin",
    ):
        assert forbidden not in mar030_source, forbidden


def test_output_schema_matches_constant():
    out = ti.build_transport_intensity_3hourly(_all_scenario_rows())
    assert list(out.columns) == list(ti.TRANSPORT_INTENSITY_3HOURLY_COLUMNS)
    assert set(out["transport_intensity_support_semantics"]) == {
        ti.TRANSPORT_INTENSITY_SUPPORT_SEMANTICS
    }


# --- Statistics (tests 18-23, Sections 15-16) -------------------------------------------


def test_18_19_summaries_computed_from_timestamp_level_intensity_not_percentile_minus_one():
    # Ratios chosen so that p95(max(M-1,0)) != max(p95(M)-1, 0) is NOT the point --
    # rather, the MEAN of clipped intensity differs from mean(M)-1 (the shortcut).
    ratios = [0.2, 0.4, 0.6, 1.2, 1.4]
    df = _canonical_rows(ratios)
    out = ti.build_transport_intensity_3hourly(df)
    stats = ti.compute_transport_intensity_stats(out)
    row = stats.iloc[0]

    intensity = np.maximum(np.array(ratios) - 1.0, 0.0)
    assert row["relative_excess_intensity_mean"] == pytest.approx(intensity.mean())
    assert row["relative_excess_intensity_mean"] != pytest.approx(np.mean(ratios) - 1.0)
    assert row["relative_excess_intensity_p50"] == pytest.approx(np.percentile(intensity, 50))
    assert row["relative_excess_intensity_p90"] == pytest.approx(np.percentile(intensity, 90))
    assert row["relative_excess_intensity_p95"] == pytest.approx(np.percentile(intensity, 95))
    assert row["relative_excess_intensity_p99"] == pytest.approx(np.percentile(intensity, 99))
    assert row["relative_excess_intensity_max"] == pytest.approx(0.4)
    assert row["relative_excess_intensity_mean_when_positive"] == pytest.approx(0.3)


def test_19_p50_of_clipped_series_differs_from_clipped_p50_of_ratio():
    # p50 of M is 0.6 -> shortcut max(0.6-1,0)=0 -- agrees. Use a case where they differ:
    ratios = [0.5, 0.9, 1.1, 1.3, 1.5, 0.7]
    df = _canonical_rows(ratios)
    stats = ti.compute_transport_intensity_stats(ti.build_transport_intensity_3hourly(df))
    intensity = np.maximum(np.array(ratios) - 1.0, 0.0)
    assert stats.iloc[0]["relative_excess_intensity_mean"] == pytest.approx(intensity.mean())
    # The shortcut mean(M)-1 = 0.0 while the true mean intensity is 0.15.
    assert stats.iloc[0]["relative_excess_intensity_mean"] == pytest.approx(0.15)


def test_20_exact_m_equals_one_counts_at_or_above_but_contributes_zero_positive_excess():
    df = _canonical_rows([0.5, 1.0, 1.0, 1.5])
    stats = ti.compute_transport_intensity_stats(ti.build_transport_intensity_3hourly(df))
    row = stats.iloc[0]
    assert row["valid_intensity_timestamp_count"] == 4
    assert row["at_or_above_incipient_motion_count"] == 3
    assert row["strict_positive_excess_count"] == 1  # 21: strict > 1 only
    assert row["at_or_above_incipient_motion_fraction"] == pytest.approx(0.75)
    assert row["strict_positive_excess_fraction"] == pytest.approx(0.25)
    assert row["relative_excess_intensity_mean_when_positive"] == pytest.approx(0.5)


def test_22_23_denominator_is_valid_timestamps_and_missing_rows_are_not_zero_observations():
    df = _canonical_rows([np.nan, np.nan, 0.5, 1.5])
    out = ti.build_transport_intensity_3hourly(df)
    stats = ti.compute_transport_intensity_stats(out)
    row = stats.iloc[0]
    assert row["valid_intensity_timestamp_count"] == 2  # not 4
    assert row["at_or_above_incipient_motion_count"] == 1
    assert row["at_or_above_incipient_motion_fraction"] == pytest.approx(0.5)  # 1/2, not 1/4
    assert row["strict_positive_excess_fraction"] == pytest.approx(0.5)
    assert row["relative_excess_intensity_mean"] == pytest.approx(0.25)  # (0 + 0.5)/2, not /4
    assert row["fraction_denominator"] == ti.FRACTION_DENOMINATOR
    assert "valid" in ti.FRACTION_DENOMINATOR_DESCRIPTION
    for word in ("probability", "annual", "return", "project life", "route"):
        assert word not in ti.FRACTION_DENOMINATOR_DESCRIPTION


def test_all_null_group_reports_zero_valid_and_null_fractions_not_zero():
    df = _canonical_rows([np.nan, np.nan])
    stats = ti.compute_transport_intensity_stats(ti.build_transport_intensity_3hourly(df))
    row = stats.iloc[0]
    assert row["valid_intensity_timestamp_count"] == 0
    assert row["at_or_above_incipient_motion_fraction"] is None or pd.isna(
        row["at_or_above_incipient_motion_fraction"]
    )
    assert row["relative_excess_intensity_p95"] is None or pd.isna(
        row["relative_excess_intensity_p95"]
    )


def test_stats_schema_matches_constant_and_one_row_per_pair_x_d50():
    df = pd.concat(
        [_all_scenario_rows(), _all_scenario_rows().assign(hydro_pair_id="pair_B")],
        ignore_index=True,
    )
    stats = ti.compute_transport_intensity_stats(ti.build_transport_intensity_3hourly(df))
    assert list(stats.columns) == list(ti.TRANSPORT_INTENSITY_STATS_COLUMNS)
    assert len(stats) == 2 * 9
    assert not stats.duplicated(subset=["hydro_pair_id", "tested_d50_mm"]).any()
    assert set(stats["scientific_role"]) == {ti.SCIENTIFIC_ROLE}
    assert set(stats["grain_size_scenario_semantics"]) == {ncm.GRAIN_SIZE_SCENARIO_SEMANTICS}


def test_stats_empty_input():
    empty = pd.DataFrame(columns=list(ti.TRANSPORT_INTENSITY_3HOURLY_COLUMNS))
    stats = ti.compute_transport_intensity_stats(empty)
    assert stats.empty
    assert list(stats.columns) == list(ti.TRANSPORT_INTENSITY_STATS_COLUMNS)


def test_mar013_cross_check_agrees_with_accepted_mar013_stats():
    df = _all_scenario_rows()
    mar013_stats = ncm.compute_noncohesive_mobility_stats(df)
    out = ti.build_transport_intensity_3hourly(df)
    stats = ti.compute_transport_intensity_stats(out)
    compared = ti.cross_check_against_mobility_stats(stats, mar013_stats)
    assert compared == 9
    merged = stats.merge(mar013_stats, on=["hydro_pair_id", "tested_d50_mm"])
    assert (
        merged["at_or_above_incipient_motion_fraction"] == merged["threshold_exceedance_fraction"]
    ).all()


def test_mar013_cross_check_disagreement_is_a_controlled_failure():
    df = _all_scenario_rows()
    mar013_stats = ncm.compute_noncohesive_mobility_stats(df)
    stats = ti.compute_transport_intensity_stats(ti.build_transport_intensity_3hourly(df))
    tampered = mar013_stats.copy()
    tampered.loc[0, "threshold_exceedance_count"] += 1
    with pytest.raises(ti.MobilityStatsCrossCheckError):
        ti.cross_check_against_mobility_stats(stats, tampered)


def test_status_column_disagreeing_with_ratio_is_a_controlled_failure():
    df = _canonical_rows([0.5, 1.5])
    out = ti.build_transport_intensity_3hourly(df)
    out.loc[1, "incipient_motion_status"] = ncm.BELOW_THRESHOLD  # tampered downstream
    with pytest.raises(ti.MobilityStatsCrossCheckError):
        ti.compute_transport_intensity_stats(out)


# --- D50 semantics (tests 24-28) -------------------------------------------------------


def test_24_25_26_all_nine_scenarios_separate_no_preferred_no_aggregate():
    stats = ti.compute_transport_intensity_stats(
        ti.build_transport_intensity_3hourly(_all_scenario_rows())
    )
    assert sorted(stats["tested_d50_mm"].unique().tolist()) == sorted(ncm.TESTED_D50_SCENARIOS_MM)
    assert len(stats) == 9
    lowered = [c.lower() for c in ti.TRANSPORT_INTENSITY_STATS_COLUMNS]
    lowered += [c.lower() for c in ti.TRANSPORT_INTENSITY_SEGMENTS_COLUMNS]
    lowered += [c.lower() for c in ti.TRANSPORT_INTENSITY_3HOURLY_COLUMNS]
    for forbidden in (
        "preferred_d50",
        "default_d50",
        "local_d50",
        "best_d50",
        "actual_d50",
        "overall",
        "weighted",
        "score",
        "mean_across_d50",
        "nearest_valid_psa",
        "folk",
    ):
        assert not any(forbidden in c for c in lowered), forbidden


def test_27_28_module_never_converts_folk_or_interpolates_psa():
    """AST scan: no identifier, attribute, or column key in the module's CODE touches
    Folk-class, PSA, or interpolation quantities (docstring/limitation wording may
    legitimately mention them as things NOT done)."""

    tree = ast.parse(inspect.getsource(ti))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
        elif isinstance(node, ast.arg):
            names.add(node.arg.lower())
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                names.add(node.slice.value.lower())
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.add(key.value.lower())
    for forbidden in ("folk", "psa", "interpolat", "kriging", "idw", "voronoi", "nearest_valid"):
        hits = sorted(n for n in names if forbidden in n)
        # Metadata keys that DECLARE the absence of these operations are the only allowance.
        hits = [h for h in hits if not h.endswith("_applied")]
        assert hits == [], (forbidden, hits)


# --- No rate / causality (tests 29-37) --------------------------------------------------


def test_29_to_37_no_rate_flux_direction_or_risk_fields_anywhere():
    forbidden = (
        "q_b",
        "q_t",
        "qb_",
        "qt_",
        "kg_s",
        "m3_s",
        "kg/s",
        "m3/s",
        "bedload",
        "suspended",
        "total_load",
        "direction",
        "erosion",
        "deposition",
        "scour",
        "probability",
        "risk",
        "score",
        "class",
    )
    all_columns = [
        *ti.TRANSPORT_INTENSITY_3HOURLY_COLUMNS,
        *ti.TRANSPORT_INTENSITY_STATS_COLUMNS,
        *ti.TRANSPORT_INTENSITY_SEGMENTS_COLUMNS,
    ]
    lowered = [c.lower() for c in all_columns]
    for term in forbidden:
        assert not any(term in c for c in lowered), term


def test_no_intensity_classes_exist_in_module():
    src = inspect.getsource(ti)
    for term in ('"LOW"', '"MODERATE"', '"HIGH"', '"VERY_HIGH"', "traffic"):
        assert term not in src, term


def test_metadata_not_computed_claims_are_all_explicitly_false():
    md = ti.build_transport_intensity_metadata(
        outputs={"x": "y"},
        row_count=36,
        hydro_pair_count=1,
        cross_checked_group_count=9,
        source_contract=_contract_one_pair(),
    )
    for key in (
        "transport_rate_computed",
        "bedload_flux_computed",
        "suspended_load_computed",
        "total_load_computed",
        "net_transport_direction_computed",
        "erosion_deposition_prediction_computed",
        "scour_computed",
        "risk_score_computed",
        "probability_computed",
        "van_rijn_1984_transport_rate_formula_applied",
        "preferred_d50_assigned",
        "aggregation_across_d50_scenarios",
        "intensity_classes_defined",
        "bgs_folk_to_numeric_d50_mapping_applied",
        "psa_d50_interpolation_applied",
        "continuous_pipeline_d50_field_created",
        "site_specific_sediment_transport_intensity_along_route",
        "mar_012_or_mar_013_formulations_changed",
    ):
        assert md[key] is False, key
    assert md["units"] == "dimensionless"
    assert md["scientific_role"] == ti.SCIENTIFIC_ROLE
    assert md["source_scientific_role"] == ncm.SCIENTIFIC_ROLE
    assert md["stress_basis"] == "MAR-013 tau_max_grain_skin_pa"
    assert "tau_critical_pa" in md["threshold_basis"]
    assert "Soulsby-Whitehouse" in md["threshold_basis"]
    assert md["tested_d50_scenarios_mm"] == list(ncm.TESTED_D50_SCENARIOS_MM)
    assert md["grain_size_scenario_semantics"] == ncm.GRAIN_SIZE_SCENARIO_SEMANTICS


def test_metadata_van_rijn_wording_and_references_are_exact():
    md = ti.build_transport_intensity_metadata(
        outputs={},
        row_count=0,
        hydro_pair_count=0,
        cross_checked_group_count=0,
        source_contract=_empty_contract(),
    )
    assert md["van_rijn_relation"] == (
        "The MAR-030 intensity is algebraically equivalent to the relative-excess "
        "structure of the Van Rijn transport-stage parameter when formed from "
        "consistent grain-related and critical stresses. It is used here only as a "
        "dimensionless forcing-exceedance diagnostic. The Van Rijn bed-load "
        "transport-rate formula is NOT applied."
    )
    dois = {r["doi"] for r in md["references"]}
    assert dois == {
        "10.1061/(ASCE)0733-9429(1984)110:10(1431)",
        "10.1061/(ASCE)0733-9429(2007)133:6(649)",
        "10.1016/S0378-3839(98)00013-1",
    }
    assert "representative wave cycle" in md["combined_wave_current_limitation"]
    assert "PEAK" in md["combined_wave_current_limitation"]
    assert md["fraction_denominator_description"] == (
        "fraction of valid contemporaneous matched timestamps"
    )


# --- GIS (tests 38-42, Section 19) -----------------------------------------------------


def _mar013_segments(pair_ids: list[str | None]) -> gpd.GeoDataFrame:
    """Accepted-MAR-013-shaped capacity segments: contiguous, one per hydro pair run."""

    records = []
    geoms = []
    for i, pair_id in enumerate(pair_ids):
        x0, x1 = 1000.0 * i, 1000.0 * (i + 1)
        records.append(
            {
                "pipeline_id": "PLX",
                "segment_id": i,
                "start_chainage_m": x0,
                "end_chainage_m": x1,
                "kp_start": f"KP {i}+000",
                "kp_end": f"KP {i + 1}+000",
                "hydro_pair_id": pair_id,
                "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": 0.5,
                "nearest_valid_psa_d50_mm": 0.21,
                "mapped_250k_folk_class": "gS",
                "scientific_role": ncm.SCIENTIFIC_ROLE,
            }
        )
        geoms.append(LineString([(400000.0 + x0, 5900000.0), (400000.0 + x1, 5900000.0)]))
    return gpd.GeoDataFrame(records, geometry=geoms, crs=WORKING_CRS)


def _two_pair_stats() -> pd.DataFrame:
    df = pd.concat(
        [_all_scenario_rows(), _all_scenario_rows().assign(hydro_pair_id="pair_B")],
        ignore_index=True,
    )
    return ti.compute_transport_intensity_stats(ti.build_transport_intensity_3hourly(df))


def test_38_39_40_scenario_layer_is_segment_x_d50_with_inherited_geometry():
    segments = _mar013_segments(["pair_A", "pair_B"])
    stats = _two_pair_stats()
    gdf = ti.build_transport_intensity_segments(segments, stats)

    assert len(gdf) == 2 * 9  # 39: each physical segment appears nine times
    assert list(gdf.columns) == [*ti.TRANSPORT_INTENSITY_SEGMENTS_COLUMNS, "geometry"]
    assert gdf.crs == segments.crs
    for _, seg in segments.iterrows():
        scenario_rows = gdf[gdf["segment_id"] == seg["segment_id"]]
        assert sorted(scenario_rows["tested_d50_mm"].tolist()) == sorted(
            ncm.TESTED_D50_SCENARIOS_MM
        )  # 38
        assert all(g.equals(seg.geometry) for g in scenario_rows.geometry)  # 40
        assert (scenario_rows["start_chainage_m"] == seg["start_chainage_m"]).all()
        assert (scenario_rows["kp_start"] == seg["kp_start"]).all()

    # Attributes come from the matching pair x D50 statistic, never from another scenario.
    by_key = stats.set_index(["hydro_pair_id", "tested_d50_mm"])
    for _, row in gdf.iterrows():
        expected = by_key.loc[(row["hydro_pair_id"], row["tested_d50_mm"])]
        assert row["relative_excess_intensity_p95"] == pytest.approx(
            expected["relative_excess_intensity_p95"]
        )
        assert row["at_or_above_incipient_motion_fraction"] == pytest.approx(
            expected["at_or_above_incipient_motion_fraction"]
        )


def test_41_gis_fields_never_imply_actual_local_d50():
    gdf = ti.build_transport_intensity_segments(_mar013_segments(["pair_A"]), _two_pair_stats())
    lowered = [c.lower() for c in gdf.columns]
    for forbidden in ("nearest_valid_psa", "folk", "largest_tested", "actual", "local_d50"):
        assert not any(forbidden in c for c in lowered), forbidden
    assert set(gdf["grain_size_scenario_semantics"]) == {ncm.GRAIN_SIZE_SCENARIO_SEMANTICS}
    assert set(gdf["scientific_role"]) == {ti.SCIENTIFIC_ROLE}


def test_segment_without_hydro_pair_keeps_nine_features_with_null_statistics():
    gdf = ti.build_transport_intensity_segments(
        _mar013_segments(["pair_A", None]), _two_pair_stats()
    )
    orphan = gdf[gdf["segment_id"] == 1]
    assert len(orphan) == 9
    assert orphan["hydro_pair_id"].isna().all()
    assert orphan["relative_excess_intensity_p95"].isna().all()
    assert orphan["valid_intensity_timestamp_count"].isna().all()


def test_segments_missing_required_mar013_column_fails_cleanly():
    segments = _mar013_segments(["pair_A"]).drop(columns=["start_chainage_m"])
    with pytest.raises(ti.TransportIntensitySchemaError, match="start_chainage_m"):
        ti.build_transport_intensity_segments(segments, _two_pair_stats())


def test_segments_empty_input():
    empty = gpd.GeoDataFrame(columns=[*ti._MOBILITY_SEGMENT_REQUIRED], geometry=[], crs=WORKING_CRS)
    gdf = ti.build_transport_intensity_segments(empty, _two_pair_stats())
    assert gdf.empty
    assert list(gdf.columns) == [*ti.TRANSPORT_INTENSITY_SEGMENTS_COLUMNS, "geometry"]


def test_segments_gpkg_roundtrip_preserves_nine_scenarios_per_segment(tmp_path: Path):
    gdf = ti.build_transport_intensity_segments(
        _mar013_segments(["pair_A", "pair_B"]), _two_pair_stats()
    )
    path = ti.write_transport_intensity_segments_gpkg(gdf, tmp_path / "ti.gpkg")
    back = gpd.read_file(path, layer="noncohesive_transport_intensity_segments")
    assert len(back) == 18
    assert back.crs.to_epsg() == 32631
    assert set(back.geom_type) == {"LineString"}
    assert back.groupby("segment_id")["tested_d50_mm"].nunique().tolist() == [9, 9]


def test_42_visualization_renders_all_nine_scenarios_without_choosing_one(tmp_path: Path):
    gdf = ti.build_transport_intensity_segments(
        _mar013_segments(["pair_A", "pair_B", None]), _two_pair_stats()
    )
    png = ti.render_transport_intensity_scenario_matrix(gdf, output_path=tmp_path / "m.png")
    assert png.exists() and png.stat().st_size > 0
    assert ti.MATRIX_VISUAL_QUANTITY == "relative_excess_intensity_p95"
    src = inspect.getsource(ti.render_transport_intensity_scenario_matrix)
    assert "d50_values = sorted(TESTED_D50_SCENARIOS_MM)" in src
    assert "not observed local sediment assignments" in src
    # No scenario selection logic: no MAR-013 capacity field, no argmax/idxmax over D50,
    # and no identifier naming a preferred scenario (the footer's "no preferred D50"
    # wording is a negative statement, checked separately below).
    for forbidden in ("largest_tested", "idxmax", "argmax", "preferred_d50", "PREFERRED"):
        assert forbidden not in src, forbidden
    assert "no preferred" in src  # footer negative statement (split across literals)


def test_visualization_handles_empty_layer(tmp_path: Path):
    empty = gpd.GeoDataFrame(
        columns=[*ti.TRANSPORT_INTENSITY_SEGMENTS_COLUMNS], geometry=[], crs=WORKING_CRS
    )
    png = ti.render_transport_intensity_scenario_matrix(empty, output_path=tmp_path / "e.png")
    assert png.stat().st_size > 0


# --- Determinism (tests 43-44) ----------------------------------------------------------


def test_43_identical_input_gives_identical_tabular_output():
    df = _all_scenario_rows()
    out1 = ti.build_transport_intensity_3hourly(df)
    out2 = ti.build_transport_intensity_3hourly(df.copy(deep=True))
    pd.testing.assert_frame_equal(out1, out2)
    stats1 = ti.compute_transport_intensity_stats(out1)
    stats2 = ti.compute_transport_intensity_stats(out2)
    pd.testing.assert_frame_equal(stats1, stats2)
    seg = _mar013_segments(["pair_A"])
    g1 = ti.build_transport_intensity_segments(seg, stats1)
    g2 = ti.build_transport_intensity_segments(seg, stats2)
    pd.testing.assert_frame_equal(pd.DataFrame(g1), pd.DataFrame(g2))


def test_44_metadata_deterministic_apart_from_outputs_and_counts():
    contract = _contract_one_pair()
    a = ti.build_transport_intensity_metadata(
        outputs={"p": "one"},
        row_count=36,
        hydro_pair_count=1,
        cross_checked_group_count=9,
        source_contract=contract,
    )
    b = ti.build_transport_intensity_metadata(
        outputs={"p": "two"},
        row_count=36,
        hydro_pair_count=1,
        cross_checked_group_count=9,
        source_contract=dict(contract),
    )
    a.pop("outputs")
    b.pop("outputs")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --- Report wording ---------------------------------------------------------------------


def test_report_states_the_mandatory_negatives(capsys: pytest.CaptureFixture[str]):
    df = _all_scenario_rows()
    out = ti.build_transport_intensity_3hourly(df)
    stats = ti.compute_transport_intensity_stats(out)
    gdf = ti.build_transport_intensity_segments(_mar013_segments(["pair_A"]), stats)
    ti.print_transport_intensity_report(
        intensity_df=out, stats_df=stats, segments_gdf=gdf, cross_checked_group_count=9
    )
    text = capsys.readouterr().out
    assert "SEDIMENT TRANSPORT RATE / BEDLOAD / SUSPENDED LOAD     = NO" in text
    assert "NET TRANSPORT DIRECTION                                = NO" in text
    assert "PREFERRED / ASSIGNED ACTUAL D50                        = NO" in text
    assert "SITE-SPECIFIC SEDIMENT TRANSPORT INTENSITY ALONG ROUTE = NO" in text
    assert "GENERIC TESTED-D50 TRANSPORT-POTENTIAL INTENSITY       = YES" in text
    assert "Van Rijn bed-load transport-rate formula is NOT applied" in text
    assert "fraction of valid contemporaneous matched timestamps" in text
    assert "scenarios are NOT combined" in text


# =======================================================================================
# MAR-030A -- MAR-013 source contract integrity (numbered per the MAR-030A Section 14 matrix)
# =======================================================================================


def _two_pairs(ratios: list[float] | None = None) -> pd.DataFrame:
    ratios = ratios or [0.5, 1.0, 1.5, 3.0]
    return pd.concat(
        [_canonical_rows(ratios, pair_id="pair_A"), _canonical_rows(ratios, pair_id="pair_B")],
        ignore_index=True,
    )


def _stats_pair(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(MAR-030 stats, MAR-013 stats) for the same accepted-shaped source."""

    mar030 = ti.compute_transport_intensity_stats(ti.build_transport_intensity_3hourly(df))
    mar013 = ncm.compute_noncohesive_mobility_stats(df)
    return mar030, mar013


def _set_d50(df: pd.DataFrame, row: int, d50_mm: float) -> None:
    df.loc[row, "tested_d50_mm"] = d50_mm
    df.loc[row, "tested_d50_m"] = d50_mm / 1000.0


def test_30a_exception_hierarchy_is_one_controlled_family():
    for exc in (
        ti.TransportIntensitySchemaError,
        ti.TransportIntensitySourceRoleError,
        ti.TransportIntensityScenarioContractError,
        ti.TransportIntensitySourceKeyError,
        ti.IncipientMotionStatusConsistencyError,
        ti.MobilityRatioConsistencyError,
        ti.MobilityStatsCrossCheckError,
    ):
        assert issubclass(exc, ti.TransportIntensityError)


# --- Scientific role (1-6) ---------------------------------------------------------------


def test_30a_01_every_correct_role_passes():
    record = ti.validate_mobility_source(_canonical_rows([0.5, 1.5]))
    assert record["scientific_role_all_rows_verified"] is True
    assert record["source_scientific_role_required"] == ncm.SCIENTIFIC_ROLE
    assert record["source_row_count"] == 18
    assert record["source_hydro_pair_count"] == 1
    assert record["source_hydro_pair_timestamp_count"] == 2
    assert record["source_hydro_pair_d50_group_count"] == 9


def test_30a_02_foreign_role_fails():
    df = _canonical_rows([0.5, 1.5])
    df["scientific_role"] = "SOULSBY_ALGEBRAIC_WAVE_CURRENT_BED_SHEAR_SENSITIVITY"
    with pytest.raises(ti.TransportIntensitySourceRoleError, match="SOULSBY_ALGEBRAIC"):
        ti.validate_mobility_source(df)


def test_30a_03_mixed_correct_and_foreign_fails():
    df = _canonical_rows([0.5, 1.5])
    df.loc[3, "scientific_role"] = "SOMETHING_ELSE"
    with pytest.raises(ti.TransportIntensitySourceRoleError, match="SOMETHING_ELSE"):
        ti.build_transport_intensity_3hourly(df)


def test_30a_04_all_null_role_fails():
    df = _canonical_rows([0.5, 1.5])
    df["scientific_role"] = None
    with pytest.raises(ti.TransportIntensitySourceRoleError, match="18 null/missing"):
        ti.validate_mobility_source(df)


def test_30a_05_mixed_correct_and_null_fails_and_never_asserts_a_source_role():
    df = _canonical_rows([0.5, 1.5]).astype({"scientific_role": object})
    df.loc[0, "scientific_role"] = None
    df.loc[7, "scientific_role"] = np.nan
    with pytest.raises(ti.TransportIntensitySourceRoleError, match="2 null/missing"):
        ti.validate_mobility_source(df)
    # The dropna() escape hatch is gone: no derived table exists to carry an
    # asserted source_scientific_role for rows whose source role is missing.
    with pytest.raises(ti.TransportIntensitySourceRoleError):
        ti.build_transport_intensity_3hourly(df)
    assert "dropna" not in inspect.getsource(ti._verify_source_scientific_role)


def test_30a_06_empty_string_role_fails():
    df = _canonical_rows([0.5, 1.5])
    df.loc[5, "scientific_role"] = ""
    with pytest.raises(ti.TransportIntensitySourceRoleError, match=r"\[''\]"):
        ti.validate_mobility_source(df)


# --- Scenario vocabulary (7-12) -----------------------------------------------------------


def test_30a_07_exact_canonical_nine_scenarios_pass():
    record = ti.validate_mobility_source(_canonical_rows([0.5, 1.5]))
    assert record["exact_scenario_set_verified"] is True
    assert record["canonical_tested_d50_scenarios_mm"] == list(ncm.TESTED_D50_SCENARIOS_MM)
    assert ti.TESTED_D50_SCENARIOS_MM is ncm.TESTED_D50_SCENARIOS_MM  # reused, not copied


def test_30a_08_one_missing_scenario_fails():
    df = _canonical_rows([0.5, 1.5])
    df = df[df["tested_d50_mm"] != 16.0].reset_index(drop=True)
    with pytest.raises(ti.TransportIntensityScenarioContractError, match=r"missing \[16.0\]"):
        ti.build_transport_intensity_3hourly(df)


def test_30a_09_unexpected_32mm_scenario_fails():
    extra = _mobility_rows([0.5, 1.5], d50_mm=32.0, tau_cr=5.0)
    df = pd.concat([_canonical_rows([0.5, 1.5]), extra], ignore_index=True)
    with pytest.raises(ti.TransportIntensityScenarioContractError, match=r"unexpected \[32.0\]"):
        ti.build_transport_intensity_3hourly(df)


def test_30a_09b_nearly_canonical_value_is_not_reinterpreted_as_a_scenario():
    df = _canonical_rows([0.5, 1.5])
    df.loc[0, "tested_d50_mm"] = 0.063 * (1 + 1e-9)  # not the serialized MAR-013 constant
    df.loc[0, "tested_d50_m"] = df.loc[0, "tested_d50_mm"] / 1000.0
    with pytest.raises(ti.TransportIntensityScenarioContractError, match="unexpected"):
        ti.validate_mobility_source(df)


def test_30a_10_null_d50_fails():
    df = _canonical_rows([0.5, 1.5])
    df.loc[4, "tested_d50_mm"] = np.nan
    with pytest.raises(
        ti.TransportIntensityScenarioContractError, match="null/non-finite tested_d50_mm"
    ):
        ti.validate_mobility_source(df)


def test_30a_11_non_finite_d50_fails():
    df = _canonical_rows([0.5, 1.5])
    df.loc[4, "tested_d50_mm"] = np.inf
    with pytest.raises(
        ti.TransportIntensityScenarioContractError, match="null/non-finite tested_d50_mm"
    ):
        ti.validate_mobility_source(df)


def test_30a_12_duplicate_scenario_within_pair_timestamp_fails():
    df = _canonical_rows([0.5, 1.5])
    # Row 2 is (pair_A, t0, 0.125 mm); relabel it as 0.063 mm -> t0 carries 0.063 twice
    # and lacks 0.125, while the GLOBAL vocabulary is still complete via t1.
    assert df.loc[2, "tested_d50_mm"] == 0.125
    _set_d50(df, 2, 0.063)
    assert set(df["tested_d50_mm"]) == set(ncm.TESTED_D50_SCENARIOS_MM)
    with pytest.raises(
        ti.TransportIntensitySourceKeyError, match=ti.DUPLICATE_MAR013_MOBILITY_SOURCE_KEY
    ):
        ti.validate_mobility_source(df)


# --- Unit identity (13-15) ---------------------------------------------------------------


def test_30a_13_correct_mm_m_values_pass():
    df = _canonical_rows([0.5, 1.5])
    assert (df["tested_d50_m"] == df["tested_d50_mm"] / 1000.0).all()
    record = ti.validate_mobility_source(df)
    assert record["d50_unit_consistency_verified"] is True
    # Serialization-noise-level differences are accepted; nothing is overwritten.
    noisy = df.copy()
    noisy["tested_d50_m"] = noisy["tested_d50_m"] * (1 + 1e-13)
    ti.validate_mobility_source(noisy)


def test_30a_14_contradictory_mm_vs_m_fails():
    df = _canonical_rows([0.5, 1.5])
    row = int(df.index[df["tested_d50_mm"] == 0.25][0])
    df.loc[row, "tested_d50_m"] = 0.000500  # 0.250 mm <-> 0.000500 m
    with pytest.raises(
        ti.TransportIntensityScenarioContractError, match="tested_d50_m inconsistent"
    ):
        ti.validate_mobility_source(df)
    assert df.loc[row, "tested_d50_m"] == 0.000500  # never silently recomputed


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_30a_15_null_or_non_finite_d50_m_fails(bad: float):
    df = _canonical_rows([0.5, 1.5])
    df.loc[6, "tested_d50_m"] = bad
    with pytest.raises(
        ti.TransportIntensityScenarioContractError, match="null/non-finite tested_d50_m"
    ):
        ti.validate_mobility_source(df)


# --- Source keys (16-17) -----------------------------------------------------------------


def test_30a_16_duplicate_source_key_fails_and_is_never_aggregated():
    df = _canonical_rows([0.5, 1.5])
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)  # exact duplicate of one row
    with pytest.raises(ti.TransportIntensitySourceKeyError) as info:
        ti.build_transport_intensity_3hourly(df)
    assert ti.DUPLICATE_MAR013_MOBILITY_SOURCE_KEY in str(info.value)
    assert "never aggregated" in str(info.value)
    for forbidden in (
        "drop_duplicates(subset=list(SOURCE_KEY_COLUMNS))",
        'keep="first"',
        'keep="last"',
    ):
        assert forbidden not in inspect.getsource(ti._verify_source_key_uniqueness)


def test_30a_16b_null_key_component_fails():
    df = _canonical_rows([0.5, 1.5]).astype({"hydro_pair_id": object})
    df.loc[1, "hydro_pair_id"] = None
    with pytest.raises(ti.TransportIntensitySourceKeyError, match="null hydro_pair_id"):
        ti.validate_mobility_source(df)
    df = _canonical_rows([0.5, 1.5])
    df.loc[1, "time_utc"] = pd.NaT
    with pytest.raises(ti.TransportIntensitySourceKeyError, match="null hydro_pair_id"):
        ti.validate_mobility_source(df)


def test_30a_17_scenario_completeness_checked_independently_per_pair_timestamp():
    df = _two_pairs()
    complete = ti.validate_mobility_source(df)
    assert complete["per_timestamp_scenario_completeness_verified"] is True
    assert complete["source_hydro_pair_timestamp_count"] == 8
    # Remove ONE row: pair_B at its second timestamp loses the 0.5 mm scenario. The global
    # vocabulary is still complete and every key is unique -- only completeness catches it.
    t1 = df["time_utc"].sort_values().unique()[1]
    victim = df.index[
        (df["hydro_pair_id"] == "pair_B") & (df["time_utc"] == t1) & (df["tested_d50_mm"] == 0.5)
    ]
    assert len(victim) == 1
    broken = df.drop(index=victim).reset_index(drop=True)
    assert set(broken["tested_d50_mm"]) == set(ncm.TESTED_D50_SCENARIOS_MM)
    assert not broken.duplicated(subset=list(ti.SOURCE_KEY_COLUMNS)).any()
    with pytest.raises(ti.TransportIntensityScenarioContractError) as info:
        ti.build_transport_intensity_3hourly(broken)
    assert "1 hydro_pair_id x time_utc group(s)" in str(info.value)
    assert "pair_B" in str(info.value)
    # pair_A alone is untouched and still passes.
    ti.validate_mobility_source(broken[broken["hydro_pair_id"] == "pair_A"].reset_index(drop=True))


# --- Incipient-motion status QA (18-21) -----------------------------------------------------


def test_30a_18_below_threshold_status_passes():
    df = _canonical_rows([0.2, 0.5, 0.999])
    assert set(df["incipient_motion_status"]) == {ncm.BELOW_THRESHOLD}
    ti.verify_incipient_motion_status_consistency(df)
    assert ti.validate_mobility_source(df)["incipient_motion_status_consistency_verified"] is True


def test_30a_19_above_or_at_threshold_status_passes():
    df = _canonical_rows([1.0, 1.001, 5.0])
    assert set(df["incipient_motion_status"]) == {ncm.ABOVE_OR_AT_THRESHOLD}
    ti.verify_incipient_motion_status_consistency(df)


def test_30a_20_mismatched_finite_status_fails_and_is_not_corrected():
    df = _canonical_rows([0.5, 1.5])
    df.loc[1, "incipient_motion_status"] = ncm.BELOW_THRESHOLD  # ratio 1.5 says ABOVE_OR_AT
    with pytest.raises(ti.IncipientMotionStatusConsistencyError, match="never corrected"):
        ti.build_transport_intensity_3hourly(df)
    assert df.loc[1, "incipient_motion_status"] == ncm.BELOW_THRESHOLD
    df = _canonical_rows([0.5, 1.5])
    df.loc[0, "incipient_motion_status"] = ncm.ABOVE_OR_AT_THRESHOLD  # ratio 0.5 says BELOW
    with pytest.raises(ti.IncipientMotionStatusConsistencyError):
        ti.validate_mobility_source(df)
    # MAR-013's own classifier is what is re-applied -- no new threshold in MAR-030A.
    assert "classify_incipient_motion_status" in inspect.getsource(
        ti.verify_incipient_motion_status_consistency
    )


def test_30a_21_undefined_ratio_with_non_null_status_fails_and_vice_versa():
    df = _canonical_rows([np.nan, 1.5])
    assert pd.isna(df.loc[0, "incipient_motion_status"])
    ti.verify_incipient_motion_status_consistency(df)  # null status for undefined ratio: OK
    df.loc[0, "incipient_motion_status"] = ncm.BELOW_THRESHOLD
    with pytest.raises(ti.IncipientMotionStatusConsistencyError):
        ti.validate_mobility_source(df)
    df = _canonical_rows([0.5, 1.5]).astype({"incipient_motion_status": object})
    df.loc[1, "incipient_motion_status"] = None  # finite ratio with a missing status
    with pytest.raises(ti.IncipientMotionStatusConsistencyError):
        ti.validate_mobility_source(df)


# --- Stats cross-check (22-30) -------------------------------------------------------------


def test_30a_22_exact_key_set_passes():
    mar030, mar013 = _stats_pair(_two_pairs())
    assert ti.cross_check_against_mobility_stats(mar030, mar013) == 18


def test_30a_23_mar030_missing_group_fails_even_though_intersection_agrees():
    mar030, mar013 = _stats_pair(_two_pairs())
    trimmed = mar030[~((mar030["hydro_pair_id"] == "pair_B") & (mar030["tested_d50_mm"] == 0.5))]
    assert len(trimmed) == 17
    # An inner merge alone would have passed: every intersecting group agrees.
    merged = trimmed.merge(mar013, on=["hydro_pair_id", "tested_d50_mm"], how="inner")
    assert (merged["valid_intensity_timestamp_count"] == merged["valid_count"]).all()
    with pytest.raises(ti.MobilityStatsCrossCheckError, match="only in MAR-013"):
        ti.cross_check_against_mobility_stats(trimmed, mar013)


def test_30a_24_mar030_extra_group_fails():
    mar030, mar013 = _stats_pair(_two_pairs())
    extra = mar030.iloc[[0]].assign(tested_d50_mm=32.0)
    with pytest.raises(ti.MobilityStatsCrossCheckError, match="only in MAR-030"):
        ti.cross_check_against_mobility_stats(pd.concat([mar030, extra], ignore_index=True), mar013)


def test_30a_25_mar013_missing_group_fails():
    mar030, mar013 = _stats_pair(_two_pairs())
    trimmed = mar013[~((mar013["hydro_pair_id"] == "pair_A") & (mar013["tested_d50_mm"] == 16.0))]
    with pytest.raises(ti.MobilityStatsCrossCheckError, match="only in MAR-030"):
        ti.cross_check_against_mobility_stats(mar030, trimmed)


def test_30a_26_mar013_extra_group_fails():
    mar030, mar013 = _stats_pair(_two_pairs())
    extra = mar013.iloc[[0]].assign(hydro_pair_id="pair_C")
    with pytest.raises(ti.MobilityStatsCrossCheckError, match="only in MAR-013"):
        ti.cross_check_against_mobility_stats(mar030, pd.concat([mar013, extra], ignore_index=True))


def test_30a_27_duplicate_mar013_stats_key_fails():
    mar030, mar013 = _stats_pair(_two_pairs())
    duplicated = pd.concat([mar013, mar013.iloc[[4]]], ignore_index=True)
    with pytest.raises(
        ti.MobilityStatsCrossCheckError, match="MAR-013 statistics carry duplicated"
    ):
        ti.cross_check_against_mobility_stats(mar030, duplicated)


def test_30a_28_duplicate_mar030_stats_key_fails():
    mar030, mar013 = _stats_pair(_two_pairs())
    duplicated = pd.concat([mar030, mar030.iloc[[4]]], ignore_index=True)
    with pytest.raises(
        ti.MobilityStatsCrossCheckError, match="MAR-030 statistics carry duplicated"
    ):
        ti.cross_check_against_mobility_stats(duplicated, mar013)


def test_30a_29_equal_keys_but_unequal_valid_count_fails():
    mar030, mar013 = _stats_pair(_two_pairs())
    tampered = mar013.copy()
    tampered.loc[2, "valid_count"] -= 1
    with pytest.raises(ti.MobilityStatsCrossCheckError, match="counts disagree"):
        ti.cross_check_against_mobility_stats(mar030, tampered)


def test_30a_30_equal_keys_but_unequal_exceedance_count_fails():
    mar030, mar013 = _stats_pair(_two_pairs())
    tampered = mar013.copy()
    tampered.loc[2, "threshold_exceedance_count"] += 1
    with pytest.raises(ti.MobilityStatsCrossCheckError, match="counts disagree"):
        ti.cross_check_against_mobility_stats(mar030, tampered)


def test_30a_cross_check_one_sided_empty_is_a_key_set_failure_not_zero():
    mar030, mar013 = _stats_pair(_two_pairs())
    empty030 = pd.DataFrame(columns=list(ti.TRANSPORT_INTENSITY_STATS_COLUMNS))
    empty013 = pd.DataFrame(columns=list(ncm.NONCOHESIVE_MOBILITY_STATS_COLUMNS))
    with pytest.raises(ti.MobilityStatsCrossCheckError):
        ti.cross_check_against_mobility_stats(empty030, mar013)
    with pytest.raises(ti.MobilityStatsCrossCheckError):
        ti.cross_check_against_mobility_stats(mar030, empty013)
    assert ti.cross_check_against_mobility_stats(empty030, empty013) == 0


# --- GIS (31-32) ---------------------------------------------------------------------------


def test_30a_31_canonical_14_x_9_pl854_like_structure_remains_supported():
    pair_ids = [f"pair_{i:02d}" for i in range(14)]
    df = pd.concat([_canonical_rows([0.5, 1.5], pair_id=p) for p in pair_ids], ignore_index=True)
    record = ti.validate_mobility_source(df)
    assert record["source_hydro_pair_count"] == 14
    assert record["source_hydro_pair_d50_group_count"] == 126
    mar030, mar013 = _stats_pair(df)
    assert len(mar030) == 126
    assert ti.cross_check_against_mobility_stats(mar030, mar013) == 126
    gdf = ti.build_transport_intensity_segments(_mar013_segments(pair_ids), mar030)
    assert len(gdf) == 126
    assert gdf.groupby("segment_id")["tested_d50_mm"].nunique().tolist() == [9] * 14
    assert gdf["relative_excess_intensity_p95"].notna().all()
    assert gdf["valid_intensity_timestamp_count"].notna().all()


def test_30a_32_missing_scenario_statistics_for_supported_segment_fails_not_silent_na():
    segments = _mar013_segments(["pair_A", "pair_B"])
    stats = _two_pair_stats()
    trimmed = stats[~((stats["hydro_pair_id"] == "pair_B") & (stats["tested_d50_mm"] == 0.5))]
    with pytest.raises(ti.TransportIntensityScenarioContractError) as info:
        ti.build_transport_intensity_segments(segments, trimmed)
    assert "pair_B" in str(info.value) and "[0.5]" in str(info.value)
    # A supported segment whose pair has NO statistics at all fails the same way.
    with pytest.raises(ti.TransportIntensityScenarioContractError):
        ti.build_transport_intensity_segments(segments, stats[stats["hydro_pair_id"] == "pair_A"])
    with pytest.raises(ti.TransportIntensityScenarioContractError):
        ti.build_transport_intensity_segments(
            segments, pd.DataFrame(columns=list(ti.TRANSPORT_INTENSITY_STATS_COLUMNS))
        )
    # Duplicate statistics keys never resolve to one feature silently.
    with pytest.raises(ti.TransportIntensityScenarioContractError, match="duplicated"):
        ti.build_transport_intensity_segments(
            segments, pd.concat([stats, stats.iloc[[0]]], ignore_index=True)
        )
    # Accepted MAR-013 semantics preserved: a segment with NO hydro-pair support keeps
    # nine features with null statistics (see test_segment_without_hydro_pair_...).
    gdf = ti.build_transport_intensity_segments(_mar013_segments(["pair_A", None]), stats)
    assert len(gdf) == 18


# --- Regression (33-36) ----------------------------------------------------------------------


def test_30a_33_formula_remains_exactly_max_m_minus_one_zero():
    m = np.array([np.nan, 0.0, 0.5, 0.999, 1.0, 1.001, 1.25, 2.0, 5.0])
    out = ti.compute_relative_excess_shields_intensity(m)
    expected = np.where(np.isfinite(m), np.maximum(m - 1.0, 0.0), np.nan)
    assert np.array_equal(out, expected, equal_nan=True)
    src = inspect.getsource(ti.compute_relative_excess_shields_intensity)
    assert "np.maximum(stage, 0.0)" in src
    assert ti.INTENSITY_DEFINITION.startswith(
        "relative_excess_shields_intensity = max(mobility_ratio - 1, 0)"
    )


def test_30a_34_null_mobility_remains_null_through_the_contract_checked_builder():
    df = _canonical_rows([np.nan, 1.5])
    out = ti.build_transport_intensity_3hourly(df)
    null_rows = out[out["mobility_ratio"].isna()]
    assert len(null_rows) == 9  # one undefined timestamp per scenario
    assert null_rows["relative_excess_shields_intensity"].isna().all()
    assert null_rows["relative_shields_stage"].isna().all()


def test_30a_35_no_rate_flux_direction_field_introduced():
    columns = [
        *ti.TRANSPORT_INTENSITY_3HOURLY_COLUMNS,
        *ti.TRANSPORT_INTENSITY_STATS_COLUMNS,
        *ti.TRANSPORT_INTENSITY_SEGMENTS_COLUMNS,
        *_contract_one_pair().keys(),
    ]
    for term in ("rate", "flux", "direction", "bedload", "suspended", "q_b", "kg_s", "m3_s"):
        assert not any(term in c.lower() for c in columns), term
    assert list(ti.TRANSPORT_INTENSITY_3HOURLY_COLUMNS) == [
        "hydro_pair_id",
        "current_node_id",
        "wave_node_id",
        "time_utc",
        "tested_d50_mm",
        "tested_d50_m",
        "tau_max_grain_skin_pa",
        "tau_critical_pa",
        "critical_shields_parameter",
        "mobility_ratio",
        "incipient_motion_status",
        "relative_shields_stage",
        "relative_excess_shields_intensity",
        "transport_intensity_support_semantics",
        "scientific_role",
        "source_scientific_role",
    ]


def test_30a_36_source_input_remains_unmodified_on_pass_and_on_every_failure():
    good = _canonical_rows([np.nan, 0.5, 1.5])
    before = good.copy(deep=True)
    ti.validate_mobility_source(good)
    ti.build_transport_intensity_3hourly(good)
    pd.testing.assert_frame_equal(good, before)

    failing = []
    df = _canonical_rows([0.5, 1.5]).astype({"scientific_role": object})
    df.loc[0, "scientific_role"] = None
    failing.append(df)
    df = _canonical_rows([0.5, 1.5])
    df.loc[0, "tested_d50_m"] = 0.5
    failing.append(df)
    df = _canonical_rows([0.5, 1.5])
    failing.append(pd.concat([df, df.iloc[[0]]], ignore_index=True))
    df = _canonical_rows([0.5, 1.5])
    df.loc[1, "incipient_motion_status"] = ncm.BELOW_THRESHOLD
    failing.append(df)
    for df in failing:
        before = df.copy(deep=True)
        with pytest.raises(ti.TransportIntensityError):
            ti.build_transport_intensity_3hourly(df)
        pd.testing.assert_frame_equal(df, before)


# --- Metadata / report -----------------------------------------------------------------------


def test_30a_metadata_source_contract_section_is_machine_readable_and_evidence_based():
    contract = ti.validate_mobility_source(_two_pairs())
    md = ti.build_transport_intensity_metadata(
        outputs={},
        row_count=72,
        hydro_pair_count=2,
        cross_checked_group_count=18,
        source_contract=contract,
    )
    section = md["source_contract"]
    assert section["source_scientific_role_required"] == ncm.SCIENTIFIC_ROLE
    assert section["canonical_tested_d50_scenarios_mm"] == list(ncm.TESTED_D50_SCENARIOS_MM)
    for flag in (
        "scientific_role_all_rows_verified",
        "exact_scenario_set_verified",
        "d50_unit_consistency_verified",
        "source_key_uniqueness_verified",
        "per_timestamp_scenario_completeness_verified",
        "incipient_motion_status_consistency_verified",
        "mar013_stats_key_set_match_verified",
    ):
        assert section[flag] is True, flag
    assert section["mar013_stats_groups_cross_checked"] == 18
    assert "not a new scientific validation" in section["source_contract_semantics"]
    json.dumps(md)  # serializable

    # The key-set flag is never asserted when the cross-check did not cover every group.
    with pytest.raises(ti.MobilityStatsCrossCheckError):
        ti.build_transport_intensity_metadata(
            outputs={},
            row_count=72,
            hydro_pair_count=2,
            cross_checked_group_count=17,
            source_contract=contract,
        )
    empty = ti.build_transport_intensity_metadata(
        outputs={},
        row_count=0,
        hydro_pair_count=0,
        cross_checked_group_count=0,
        source_contract=_empty_contract(),
    )
    assert empty["source_contract"]["scientific_role_all_rows_verified"] is None
    assert empty["source_contract"]["mar013_stats_key_set_match_verified"] is None


def test_30a_report_prints_the_source_contract(capsys: pytest.CaptureFixture[str]):
    df = _all_scenario_rows()
    out = ti.build_transport_intensity_3hourly(df)
    stats = ti.compute_transport_intensity_stats(out)
    gdf = ti.build_transport_intensity_segments(_mar013_segments(["pair_A"]), stats)
    ti.print_transport_intensity_report(
        intensity_df=out,
        stats_df=stats,
        segments_gdf=gdf,
        cross_checked_group_count=9,
        source_contract=ti.validate_mobility_source(df),
    )
    text = capsys.readouterr().out
    assert "MAR-013 source contract (MAR-030A: source/integration integrity)" in text
    assert "scientific_role_all_rows_verified" in text
    assert "per_timestamp_scenario_completeness_verified" in text
    assert "mar013_stats_key_set_match_verified" in text
    assert "= YES (9 group(s))" in text

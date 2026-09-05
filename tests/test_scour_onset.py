"""Offline unit tests for marine_engine.scour.scour_onset (MAR-014).

Small hand-built synthetic DataFrames only -- never the real PL854 route or
real Copernicus data, never network access. Lettered comments map to
MAR-014 Section 40's required test list.
"""

import numpy as np
import pandas as pd
import pytest

from marine_engine.metocean.combined_bed_shear import UnreconciledHydroNodeError, build_hydro_pairs
from marine_engine.scour import scour_onset as so

WORKING_CRS = "EPSG:32631"


# --- A: Marini wave friction factor exact against hand calculation ------------------


def test_A_marini_wave_friction_factor_matches_hand_calculation():
    a_x = 0.15
    d50_m = 0.00025
    f_w = so.compute_marini_wave_friction_factor(np.array([a_x]), d50_m)
    expected = 0.04 * (a_x / (2.5 * d50_m)) ** (-0.25)
    assert f_w[0] == pytest.approx(expected)


def test_marini_wave_friction_factor_undefined_for_zero_excursion():
    f_w = so.compute_marini_wave_friction_factor(np.array([0.0]), 0.00025)
    assert np.isnan(f_w[0])


# --- B: theta_w exact against hand calculation ---------------------------------------


def test_B_marini_wave_shields_parameter_matches_hand_calculation():
    d50_m = 0.00025
    u_star_w = 0.02
    tau_w = so.compute_marini_wave_shear_stress_pa(np.array([u_star_w]))
    theta_w = so.compute_marini_wave_shields_parameter(tau_w, d50_m)

    s = so.RHO_SEDIMENT_KG_M3 / so.RHO_WATER_KG_M3
    expected_tau_w = so.RHO_WATER_KG_M3 * u_star_w**2
    expected_theta_w = expected_tau_w / (so.RHO_WATER_KG_M3 * so.GRAVITY_M_S2 * (s - 1.0) * d50_m)

    assert tau_w[0] == pytest.approx(expected_tau_w)
    assert theta_w[0] == pytest.approx(expected_theta_w)


def test_zero_wave_excursion_gives_zero_theta_w_never_infinity():
    """Do not let a_x=0 generate infinities (Section 14)."""

    a_x = so.compute_wave_semi_excursion_m(np.array([0.0]), np.array([np.nan]))
    f_w = so.compute_marini_wave_friction_factor(a_x, 0.00025)
    u_star_w = so.compute_marini_wave_friction_velocity_m_s(f_w, np.array([0.0]))
    tau_w = so.compute_marini_wave_shear_stress_pa(u_star_w)
    theta_w = so.compute_marini_wave_shields_parameter(tau_w, 0.00025)

    assert a_x[0] == pytest.approx(0.0)
    assert u_star_w[0] == pytest.approx(0.0)
    assert tau_w[0] == pytest.approx(0.0)
    assert theta_w[0] == pytest.approx(0.0)
    assert np.isfinite(theta_w[0])


# --- C: wave-only branch ---------------------------------------------------------------


def test_C_wave_only_kc_alpha_beta():
    uc_perp = np.array([0.0])
    uw_perp = np.array([0.2])
    t_rep = np.array([6.0])
    branch = so.classify_kc_branch(uc_perp, uw_perp)
    assert branch[0] == so.WAVE_ONLY

    kc = so.compute_kc_marini(uc_perp, uw_perp, t_rep, 0.3048, branch)
    assert kc[0] == pytest.approx(uw_perp[0] * t_rep[0] / 0.3048)

    alpha = so.compute_alpha(uc_perp, uw_perp)
    assert alpha[0] == pytest.approx(0.0)

    beta = so.compute_beta(alpha)
    assert beta[0] == pytest.approx(1.0)


# --- D: current-only branch -------------------------------------------------------------


def test_D_current_only_uses_sumer_constants_no_divide_by_zero():
    uc_perp = np.array([0.3])
    uw_perp = np.array([0.0])
    t_rep = np.array([6.0])
    branch = so.classify_kc_branch(uc_perp, uw_perp)
    assert branch[0] == so.CURRENT_ONLY

    kc = so.compute_kc_marini(uc_perp, uw_perp, t_rep, 0.3048, branch)
    assert np.isnan(kc[0])  # never evaluated for current-only

    theta_w = np.array([0.0])
    x, a, b = so.compute_marini_ab_coefficients(kc, theta_w, branch)
    assert np.isnan(x[0])  # X never forced through KC=NaN/theta=0
    assert a[0] == pytest.approx(0.025)
    assert b[0] == pytest.approx(0.5)
    assert np.isfinite(a[0])
    assert np.isfinite(b[0])

    alpha = so.compute_alpha(uc_perp, uw_perp)
    beta = so.compute_beta(alpha)
    u_equivalent = so.compute_equivalent_onset_velocity_metric_m_s(uc_perp, uw_perp, beta, branch)
    assert u_equivalent[0] == pytest.approx(uc_perp[0])


# --- E: calm ---------------------------------------------------------------------------


def test_E_calm_gives_no_onset_forcing():
    uc_perp = np.array([0.0])
    uw_perp = np.array([0.0])
    branch = so.classify_kc_branch(uc_perp, uw_perp)
    assert branch[0] == so.CALM

    alpha = so.compute_alpha(uc_perp, uw_perp)
    beta = so.compute_beta(alpha)  # alpha is NaN, beta will be NaN -- fine, unused for CALM
    u_equivalent = so.compute_equivalent_onset_velocity_metric_m_s(uc_perp, uw_perp, beta, branch)
    assert u_equivalent[0] == pytest.approx(0.0)

    omega_forcing = so.compute_omega_forcing(u_equivalent, 0.3048, 0.40)
    assert omega_forcing[0] == pytest.approx(0.0)
    status = so.classify_scour_onset_status(np.array([omega_forcing[0] / 1.0]))
    assert status[0] == so.SCOUR_ONSET_NOT_REACHED


# --- F: combined KC Eq. 30 against hand calculation ------------------------------------


def test_F_combined_kc_matches_hand_calculation():
    uc_perp = np.array([0.2])
    uw_perp = np.array([0.15])
    t_rep = np.array([6.0])
    diameter_m = 0.3048
    branch = so.classify_kc_branch(uc_perp, uw_perp)
    assert branch[0] == so.COMBINED

    kc = so.compute_kc_marini(uc_perp, uw_perp, t_rep, diameter_m, branch)
    expected = (uw_perp[0] * t_rep[0] / diameter_m) * (
        -0.25 + 1.25 * np.exp((uc_perp[0] / uw_perp[0]) ** 0.87)
    )
    assert kc[0] == pytest.approx(expected)


# --- G: alpha bounded 0..1 ---------------------------------------------------------------


def test_G_alpha_bounded_0_to_1():
    uc_perp = np.array([0.0, 0.1, 0.3, 0.5])
    uw_perp = np.array([0.2, 0.1, 0.0, 0.0])
    alpha = so.compute_alpha(uc_perp, uw_perp)
    valid = alpha[np.isfinite(alpha)]
    assert (valid >= 0.0).all()
    assert (valid <= 1.0).all()


# --- H: beta polynomial exact ------------------------------------------------------------


def test_H_beta_polynomial_exact():
    alpha = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    beta = so.compute_beta(alpha)
    expected = 6.0 * alpha**3 - 16.0 * alpha**2 + 10.0 * alpha + 1.0
    assert beta == pytest.approx(expected)


# --- I: beta stays positive for representative alpha values ----------------------------


def test_I_beta_positive_for_representative_alpha():
    alpha = np.linspace(0.0, 1.0, 21)
    beta = so.compute_beta(alpha)
    assert (beta > 0).all()


# --- J: Marini a/b exact against hand calculation --------------------------------------


def test_J_marini_a_b_match_hand_calculation():
    x = 0.5
    a = so.compute_marini_a(np.array([x]))
    b = so.compute_marini_b(np.array([x]))
    assert a[0] == pytest.approx(0.025 * (1.0 - np.exp(-14.0 * x)))
    assert b[0] == pytest.approx(0.5 + np.exp(-2.9 * x))


# --- K: larger embedment increases Omega threshold --------------------------------------


def test_K_larger_embedment_increases_omega_threshold():
    a = np.array([0.02])
    b = np.array([0.6])
    thresholds = [
        so.compute_omega_threshold(a, b, e)[0] for e in sorted(so.TESTED_EMBEDMENT_RATIOS)
    ]
    assert all(t2 > t1 for t1, t2 in zip(thresholds, thresholds[1:], strict=False))


# --- L: onset condition is Omega_forcing >= Omega_threshold -----------------------------


def test_L_onset_condition_boundary():
    status = so.classify_scour_onset_status(np.array([0.5, 1.0, 1.5, np.nan]))
    assert status.tolist() == [
        so.SCOUR_ONSET_NOT_REACHED,
        so.SCOUR_ONSET_REACHED,
        so.SCOUR_ONSET_REACHED,
        None,
    ]


# --- M/N: pipeline-normal projection ----------------------------------------------------


def test_M_flow_parallel_to_pipe_gives_zero_normal_component():
    perp, angle = so.compute_perpendicular_component(
        np.array([0.5]), np.array([30.0]), np.array([30.0])
    )
    assert perp[0] == pytest.approx(0.0, abs=1e-9)
    assert angle[0] == pytest.approx(0.0)


def test_N_flow_perpendicular_to_pipe_gives_full_magnitude():
    perp, angle = so.compute_perpendicular_component(
        np.array([0.5]), np.array([30.0]), np.array([120.0])
    )
    assert perp[0] == pytest.approx(0.5)
    assert angle[0] == pytest.approx(90.0)


# --- O: wave directions 180 apart give identical projected orbital amplitude -----------


def test_O_directions_180_apart_give_identical_projection():
    perp_a, _ = so.compute_perpendicular_component(
        np.array([0.4]), np.array([50.0]), np.array([10.0])
    )
    perp_b, _ = so.compute_perpendicular_component(
        np.array([0.4]), np.array([230.0]), np.array([10.0])
    )
    assert perp_a[0] == pytest.approx(perp_b[0])


# --- P: route tangent is local geometry, not whole-route chord ------------------------


def test_P_route_tangent_is_local_not_whole_route_chord():
    from shapely.geometry import LineString

    # An L-shaped route: whole-route chord bearing (start->end) differs
    # sharply from the LOCAL tangent near the bend.
    route = LineString([(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0)])
    near_bend_bearing = so.compute_local_tangent_bearing_deg(route, 999.0, epsilon_m=2.0)
    whole_route_chord_bearing = float(np.degrees(np.arctan2(1000.0 - 0.0, 1000.0 - 0.0)) % 360.0)
    assert near_bend_bearing != pytest.approx(whole_route_chord_bearing, abs=5.0)
    # Local tangent just past the bend should read close to due-north (0 deg).
    past_bend_bearing = so.compute_local_tangent_bearing_deg(route, 1005.0, epsilon_m=2.0)
    assert past_bend_bearing == pytest.approx(0.0, abs=1.0)


# --- Q: current pipe-top reconstruction uses z_top = D - e ------------------------------


def test_Q_pipe_top_height_uses_d_minus_e():
    diameter_m = 0.3048
    for embedment_ratio in so.TESTED_EMBEDMENT_RATIOS:
        z_top = so.compute_pipe_top_height_m(diameter_m, embedment_ratio)
        assert z_top == pytest.approx(diameter_m - embedment_ratio * diameter_m)
        assert z_top > 0


def test_pipe_top_height_requires_positive_result():
    with pytest.raises(ValueError):
        so.compute_pipe_top_height_m(0.3048, 1.0)


def test_current_reconstruction_matches_log_profile_scale_factor():
    """Uc_top should equal U_ref scaled by the same log-profile ratio used to invert
    u_star_c in the first place -- a consistency check on the two-step formula."""

    z0_skin = 0.00002083  # 0.25mm/12
    z_ref = 3.0
    u_ref = 0.4
    u_star_c = so.compute_current_friction_velocity_m_s(
        np.array([u_ref]), z0_skin, target_height_m=z_ref
    )
    z_top = so.compute_pipe_top_height_m(0.3048, 0.06)
    uc_top = so.reconstruct_current_at_height_m_s(u_star_c, z_top, z0_skin)

    expected_ratio = np.log((z_top + z0_skin) / z0_skin) / np.log((z_ref + z0_skin) / z0_skin)
    assert uc_top[0] == pytest.approx(u_ref * expected_ratio)


# --- R: same D50 controls z0_skin and Marini wave Shields -------------------------------


def test_R_same_d50_drives_both_z0_skin_and_theta_w():
    d50_mm = 0.25
    d50_m = d50_mm / 1000.0
    z0_skin = so.compute_z0_skin_m(d50_m)
    assert z0_skin == pytest.approx(d50_m / 12.0)

    a_x = so.compute_wave_semi_excursion_m(np.array([0.2]), np.array([6.0]))
    f_w = so.compute_marini_wave_friction_factor(a_x, d50_m)
    expected_f_w = 0.04 * (a_x[0] / (2.5 * d50_m)) ** (-0.25)
    assert f_w[0] == pytest.approx(expected_f_w)


# --- S/T/U: fixed scenario sets ----------------------------------------------------------


def test_S_only_three_d50_scenarios():
    assert so.TESTED_D50_SCENARIOS_MM == (0.160, 0.250, 0.480)


def test_T_only_three_porosity_scenarios():
    assert so.TESTED_POROSITY_SCENARIOS == (0.35, 0.40, 0.45)


def test_U_only_five_tested_embedments():
    assert so.TESTED_EMBEDMENT_RATIOS == (0.00, 0.03, 0.06, 0.10, 0.15)


# --- Full-row builder helpers ------------------------------------------------------------


def _current_hourly_df(
    *,
    node_id: str = "current_A",
    times: list[pd.Timestamp] | None = None,
    speed_m_s: float = 0.4,
    height_above_model_bed_m: float = 3.0,
    height_above_model_bed_valid: bool = True,
    current_direction_to_deg: float = 45.0,
) -> pd.DataFrame:
    times = times or [pd.Timestamp("2025-01-01T00:00", tz="UTC")]
    rows = []
    for t in times:
        rows.append(
            {
                "current_node_id": node_id,
                "time_utc": t,
                "uo_m_s": speed_m_s,
                "vo_m_s": 0.0,
                "current_speed_m_s": speed_m_s,
                "current_direction_to_deg": current_direction_to_deg,
                "height_above_model_bed_m": height_above_model_bed_m,
                "height_above_model_bed_valid": height_above_model_bed_valid,
                "model_bathymetry_m": 27.0,
            }
        )
    return pd.DataFrame(rows)


def _wave_3hourly_df(
    *,
    node_id: str = "wave_A",
    times: list[pd.Timestamp] | None = None,
    uw: float = 0.25,
    t_rep: float = 6.5,
    wave_direction_to_deg: float = 200.0,
) -> pd.DataFrame:
    times = times or [pd.Timestamp("2025-01-01T00:00", tz="UTC")]
    rows = []
    for t in times:
        rows.append(
            {
                "wave_node_id": node_id,
                "time_utc": t,
                "wave_orbital_velocity_equivalent_amplitude_m_s": uw,
                "equivalent_peak_period_from_tz_s": t_rep,
                "wave_mean_direction_to_deg": wave_direction_to_deg,
            }
        )
    return pd.DataFrame(rows)


def _hydro_pairs_df(*, current_node_id="current_A", wave_node_id="wave_A") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hydro_pair_id": f"{current_node_id}__{wave_node_id}",
                "current_node_id": current_node_id,
                "wave_node_id": wave_node_id,
            }
        ]
    )


_TANGENT_BEARING = {"current_A__wave_A": 30.0}


def test_builder_column_schema_matches_constant():
    result, _diag = so.build_scour_onset_embedment_3hourly(
        _current_hourly_df(), _wave_3hourly_df(), _hydro_pairs_df(), _TANGENT_BEARING
    )
    assert list(result.columns) == list(so.SCOUR_ONSET_EMBEDMENT_3HOURLY_COLUMNS)


def test_builder_empty_inputs():
    result, diag = so.build_scour_onset_embedment_3hourly(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    )
    assert result.empty
    assert diag is None
    assert list(result.columns) == list(so.SCOUR_ONSET_EMBEDMENT_3HOURLY_COLUMNS)


def test_builder_emits_nine_scenario_rows_per_timestamp():
    result, _diag = so.build_scour_onset_embedment_3hourly(
        _current_hourly_df(), _wave_3hourly_df(), _hydro_pairs_df(), _TANGENT_BEARING
    )
    assert len(result) == 9  # 3 D50 x 3 porosity, single timestamp


# --- N (Section 40): zero current/wave special cases remain physically valid ----------


def test_N_zero_current_row_has_valid_onset_margins_and_null_direction():
    result, _diag = so.build_scour_onset_embedment_3hourly(
        _current_hourly_df(speed_m_s=0.0), _wave_3hourly_df(), _hydro_pairs_df(), _TANGENT_BEARING
    )
    row = result.iloc[0]
    assert pd.isna(row["current_direction_to_deg"])
    assert np.isfinite(row["onset_margin_eD_000"])


def test_N_zero_wave_row_has_valid_onset_margins():
    result, _diag = so.build_scour_onset_embedment_3hourly(
        _current_hourly_df(), _wave_3hourly_df(uw=0.0), _hydro_pairs_df(), _TANGENT_BEARING
    )
    row = result.iloc[0]
    assert np.isfinite(row["onset_margin_eD_000"])


def test_calm_row_never_reaches_onset():
    result, _diag = so.build_scour_onset_embedment_3hourly(
        _current_hourly_df(speed_m_s=0.0),
        _wave_3hourly_df(uw=0.0),
        _hydro_pairs_df(),
        _TANGENT_BEARING,
    )
    assert (result["onset_status_eD_000"] == so.SCOUR_ONSET_NOT_REACHED).all()
    assert result["onset_margin_eD_000"].to_numpy() == pytest.approx(0.0)


# --- L: current/wave exact-time join only -----------------------------------------------


def test_L_only_the_shared_exact_timestamp_survives_the_join():
    t0 = pd.Timestamp("2025-01-01T00:00", tz="UTC")
    t1 = pd.Timestamp("2025-01-01T03:00", tz="UTC")
    t2 = pd.Timestamp("2025-01-01T06:00", tz="UTC")
    current_df = pd.concat(
        [
            _current_hourly_df(times=[t0]),
            _current_hourly_df(times=[t1]),
            _current_hourly_df(times=[t2]),
        ],
        ignore_index=True,
    )
    wave_df = _wave_3hourly_df(times=[t1])
    result, _diag = so.build_scour_onset_embedment_3hourly(
        current_df, wave_df, _hydro_pairs_df(), _TANGENT_BEARING
    )
    assert set(result["time_utc"].unique()) == {t1}
    assert len(result) == 9


def test_unmatched_timestamps_produce_no_fabricated_rows():
    t0 = pd.Timestamp("2025-01-01T00:00", tz="UTC")
    t5 = pd.Timestamp("2025-01-01T15:00", tz="UTC")
    result, _diag = so.build_scour_onset_embedment_3hourly(
        _current_hourly_df(times=[t0]),
        _wave_3hourly_df(times=[t5]),
        _hydro_pairs_df(),
        _TANGENT_BEARING,
    )
    assert result.empty


# --- M: coordinate reconciliation used, node string equality not assumed --------------


def test_M_hydro_pairs_reused_from_mar012_coordinate_based():
    current_nodes = pd.DataFrame(
        {"node_id": ["current_XYZ"], "longitude": [1.7], "latitude": [53.37]}
    )
    wave_nodes = pd.DataFrame({"node_id": ["wave_ABC"], "longitude": [1.7], "latitude": [53.37]})
    pairs = build_hydro_pairs(current_nodes, wave_nodes, working_crs=WORKING_CRS)
    assert pairs.iloc[0]["current_node_id"] == "current_XYZ"
    assert pairs.iloc[0]["wave_node_id"] == "wave_ABC"


def test_unreconcilable_coordinates_still_hard_fail_when_reused():
    current_nodes = pd.DataFrame(
        {
            "node_id": ["current_A", "current_B"],
            "longitude": [1.666667, 1.696970],
            "latitude": [53.364861, 53.364861],
        }
    )
    wave_nodes = pd.DataFrame(
        {"node_id": ["wave_A"], "longitude": [1.666667], "latitude": [53.364861]}
    )
    with pytest.raises(UnreconciledHydroNodeError):
        build_hydro_pairs(current_nodes, wave_nodes, working_crs=WORKING_CRS)


# --- V: required embedment 0 when zero embedment already suppresses onset -------------


def test_V_required_class_zero_when_already_suppressed():
    margins = {
        0.00: np.array([0.5]),
        0.03: np.array([0.4]),
        0.06: np.array([0.3]),
        0.10: np.array([0.2]),
        0.15: np.array([0.1]),
    }
    ratio, metres, status = so.determine_required_embedment(margins, 0.3048)
    assert ratio[0] == pytest.approx(0.0)
    assert metres[0] == pytest.approx(0.0)
    assert status[0] == so.REQUIRED_EMBEDMENT_IDENTIFIED


# --- W: onset persists through 0.15 -----------------------------------------------------


def test_W_onset_persists_at_max_tested_embedment():
    margins = {
        0.00: np.array([5.0]),
        0.03: np.array([4.0]),
        0.06: np.array([3.0]),
        0.10: np.array([2.0]),
        0.15: np.array([1.5]),
    }
    ratio, metres, status = so.determine_required_embedment(margins, 0.3048)
    assert np.isnan(ratio[0])
    assert np.isnan(metres[0])
    assert status[0] == so.ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT


def test_missing_margins_leave_status_null_not_persists():
    margins = {e: np.array([np.nan]) for e in so.TESTED_EMBEDMENT_RATIOS}
    ratio, _metres, status = so.determine_required_embedment(margins, 0.3048)
    assert np.isnan(ratio[0])
    assert status[0] is None


# --- X: 95% required embedment from onset FRACTIONS, not paired percentiles -----------


def test_X_coverage_derivation_uses_onset_fractions_directly():
    fractions = {0.00: 0.20, 0.03: 0.11, 0.06: 0.06, 0.10: 0.04, 0.15: 0.01}
    value, status = so._embedment_required_for_coverage(fractions, 0.05)
    assert value == pytest.approx(0.10)
    assert status == so.REQUIRED_EMBEDMENT_IDENTIFIED


def test_X_coverage_above_tested_range_when_never_satisfied():
    fractions = dict.fromkeys(so.TESTED_EMBEDMENT_RATIOS, 0.50)
    value, status = so._embedment_required_for_coverage(fractions, 0.05)
    assert value is None
    assert status == so.ABOVE_TESTED_EMBEDMENT_RANGE


def test_stats_column_schema_matches_constant():
    result, _diag = so.build_scour_onset_embedment_3hourly(
        _current_hourly_df(), _wave_3hourly_df(), _hydro_pairs_df(), _TANGENT_BEARING
    )
    stats = so.compute_scour_onset_embedment_stats(result)
    assert list(stats.columns) == list(so.SCOUR_ONSET_EMBEDMENT_STATS_COLUMNS)
    assert len(stats) == 9


def test_stats_completeness_never_exceeds_100_pct():
    times = [
        pd.Timestamp("2025-01-01T00:00", tz="UTC") + pd.Timedelta(hours=3 * i) for i in range(4)
    ]
    current_df = pd.concat([_current_hourly_df(times=[t]) for t in times], ignore_index=True)
    wave_df = pd.concat([_wave_3hourly_df(times=[t]) for t in times], ignore_index=True)
    result, _diag = so.build_scour_onset_embedment_3hourly(
        current_df, wave_df, _hydro_pairs_df(), _TANGENT_BEARING
    )
    stats = so.compute_scour_onset_embedment_stats(result)
    assert (stats["completeness_pct"] <= 100.0001).all()


# --- Y: embedment monotonicity QA catches an injected violation ------------------------


def test_Y_monotonicity_check_passes_clean_data():
    result, _diag = so.build_scour_onset_embedment_3hourly(
        _current_hourly_df(), _wave_3hourly_df(), _hydro_pairs_df(), _TANGENT_BEARING
    )
    violation_count, _is_monotonic = so.compute_embedment_monotonicity_violations(result)
    assert violation_count == 0
    assert so.raise_if_embedment_monotonicity_violated(result) == 0


def test_Y_monotonicity_check_catches_injected_violation():
    result, _diag = so.build_scour_onset_embedment_3hourly(
        _current_hourly_df(), _wave_3hourly_df(), _hydro_pairs_df(), _TANGENT_BEARING
    )
    corrupted = result.copy()
    corrupted.loc[0, "onset_margin_eD_015"] = corrupted.loc[0, "onset_margin_eD_000"] + 10.0

    violation_count, is_monotonic = so.compute_embedment_monotonicity_violations(corrupted)
    assert violation_count == 1
    assert not is_monotonic.iloc[0]

    with pytest.raises(so.EmbedmentMonotonicityViolationError):
        so.raise_if_embedment_monotonicity_violated(corrupted)


# --- Z: pipe-diameter applicability flag is false for D=0.3048 -------------------------


def test_Z_pipe_diameter_outside_source_envelope():
    applicability = so.compute_applicability_diagnostics(None, pd.DataFrame())
    assert applicability["within_source_pipe_diameter_envelope"] is False
    assert applicability["overall_direct_source_envelope_match"] is False

    result, diag = so.build_scour_onset_embedment_3hourly(
        _current_hourly_df(), _wave_3hourly_df(), _hydro_pairs_df(), _TANGENT_BEARING
    )
    applicability_real = so.compute_applicability_diagnostics(diag, result)
    assert applicability_real["within_source_pipe_diameter_envelope"] is False


def test_applicability_empty_input():
    applicability = so.compute_applicability_diagnostics(None, pd.DataFrame())
    assert applicability["projected_uc_source_range_fraction"] is None


# --- AA: BGS Folk does not enter numeric physics ----------------------------------------


def test_AA_builder_never_reads_folk_or_predictive_columns():
    current_df = _current_hourly_df()
    wave_df = _wave_3hourly_df()
    assert not any("folk" in c.lower() for c in current_df.columns)
    assert not any("predictive" in c.lower() for c in current_df.columns)
    assert not any("folk" in c.lower() for c in wave_df.columns)
    result, _diag = so.build_scour_onset_embedment_3hourly(
        current_df, wave_df, _hydro_pairs_df(), _TANGENT_BEARING
    )
    assert not any("folk" in c.lower() for c in result.columns)
    assert not any("predictive" in c.lower() for c in result.columns)


# --- AB: PSA D50 does not propagate into route physics ----------------------------------


def test_AB_builder_signature_has_no_sediment_evidence_parameter():
    import inspect

    sig = inspect.signature(so.build_scour_onset_embedment_3hourly)
    assert "psa" not in "".join(sig.parameters.keys()).lower()
    assert "sediment_evidence" not in "".join(sig.parameters.keys()).lower()


# --- AC: MAR-007 morphology does not alter onset calculation ---------------------------


def test_AC_builder_signature_has_no_morphology_parameter():
    import inspect

    sig = inspect.signature(so.build_scour_onset_embedment_3hourly)
    assert "morphology" not in "".join(sig.parameters.keys()).lower()


def test_AC_morphology_role_is_legacy_context_only():
    assert so.MORPHOLOGY_ROLE == "LEGACY_REGIONAL_CONTEXT_ONLY"


# --- Sensitivity envelope: never a D50/porosity average --------------------------------


def test_envelope_never_averages_scenarios():
    stats_df = pd.DataFrame(
        {
            "hydro_pair_id": ["P1"] * 9,
            "embedment_ratio_required_for_95pct_state_coverage": [
                0.00,
                0.03,
                0.06,
                0.10,
                0.15,
                None,
                0.03,
                0.06,
                0.10,
            ],
        }
    )
    envelope = so.compute_sensitivity_envelope(stats_df)
    row = envelope.iloc[0]
    assert row["p95_required_embedment_lower_ratio"] == pytest.approx(0.00)
    assert row["p95_required_embedment_upper_class"] == f">{max(so.TESTED_EMBEDMENT_RATIOS):g}"
    assert pd.isna(row["p95_required_embedment_upper_ratio"])


def test_envelope_empty_input():
    result = so.compute_sensitivity_envelope(pd.DataFrame())
    assert result.empty
    assert list(result.columns) == list(so.SENSITIVITY_ENVELOPE_COLUMNS)


# --- AH: no output schema contains forbidden downstream-physics terms -----------------


def test_AH_no_output_schema_contains_forbidden_downstream_terms():
    forbidden = (
        "equilibrium_scour_depth",
        "erosion_rate",
        "exposure_probability",
        "freespan_probability",
        "fatigue",
        "risk_score",
    )
    all_columns = [
        *so.SCOUR_ONSET_EMBEDMENT_3HOURLY_COLUMNS,
        *so.SCOUR_ONSET_EMBEDMENT_STATS_COLUMNS,
        *so.SENSITIVITY_ENVELOPE_COLUMNS,
    ]
    columns_lower = [c.lower() for c in all_columns]
    for term in forbidden:
        assert not any(term in column for column in columns_lower), term

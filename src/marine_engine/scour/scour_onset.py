"""Pipeline scour-onset embedment screening (MAR-014).

Scope -- read before touching this module
--------------------------------------------
MAR-013 established that PL854's hydrodynamic environment is capable of
reaching noncohesive sediment incipient-motion conditions. This module
asks the PIPELINE-SPECIFIC follow-on question: for a shallowly embedded
pipeline under the observed wave/current forcing, what TESTED embedment is
required to suppress the empirical onset-of-scour condition? Canonical
scientific role: `PIPELINE_SCOUR_ONSET_EMBEDMENT_SCREENING`. This is a
scour-ONSET screening product -- never predicted scour depth, erosion
depth, burial, exposure, free-span geometry, probability of failure, or
pipeline risk.

Scientific model: Marini et al. (2024), never substituted (Section 2)
------------------------------------------------------------------------
Marini, F., Postacchini, M., Pizzigalli, C., Badalini, M., Corvaro, S., &
Brocchini, M. (2024). On the onset of pipeline scouring: Reconciling waves
and currents forcing. Coastal Engineering, 190, 104507. DOI:
10.1016/j.coastaleng.2024.104507. Generalises the Sumer et al. (2001)
tunnel-scour-onset criterion to wave + current conditions. This module
never substitutes an older Zang-only onset equation and never performs
independent literature research.

Critical applicability limitation -- never hidden (Section 3)
-------------------------------------------------------------------
The source experimental datasets cover pipeline diameters D=0.05-0.10 m;
PL854's `PIPELINE_DIAMETER_M` (0.3048 m) is OUTSIDE that envelope. Every
output and map states `PIPE_DIAMETER_OUTSIDE_SOURCE_EXPERIMENT_ENVELOPE`
and `RESEARCH_SCREENING_EXTRAPOLATION_NOT_CALIBRATED_PL854_PREDICTION` --
this is never called a validated field prediction.

Discrete screening, never continuous extrapolation (Sections 5-8, 20)
-------------------------------------------------------------------------
Three D50 scenarios (0.160/0.250/0.480 mm, the Marini/Zang COMBINED-flow
calibration envelope only -- never all nine MAR-013 scenarios), three
porosity scenarios (0.35/0.40/0.45), and five tested embedment ratios
(0.00/0.03/0.06/0.10/0.15) are FIXED sensitivity dimensions, never a
preferred/canonical value, never averaged, never derived from BGS Folk
class or PSA D50 interpolation. The published onset equation is evaluated
ONLY at these five explicit embedment scenarios -- never solved for an
unconstrained continuous critical embedment, and never extrapolated above
e/D=0.15 (a persisting onset there is flagged
`ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT`, never assigned a numeric value).

Pipeline-normal projection: an explicit screening extension (Section 12)
------------------------------------------------------------------------------
The Marini/Zang laboratory experiments are effectively 2D/codirectional
relative to the pipe; PL854 experiences oblique current and wave
directions. `PIPELINE_NORMAL_2D_SCREENING_PROJECTION` projects the local
current and wave orbital amplitude onto the route-normal direction using
the LOCAL route tangent (never a whole-route chord) -- an explicit,
disclosed screening approximation, never a validated oblique-flow model
(`oblique_flow_extension_directly_validated_by_source_experiments =
false`).

Grain-related current reconstruction at pipe-top height (Section 13)
-------------------------------------------------------------------------
Reuses MAR-013's exact `z0_skin = d50/12` roughness convention and
MAR-012's log-profile friction-velocity inversion
(`combined_bed_shear.compute_current_friction_velocity_m_s`), but
reconstructs the current at the TESTED pipe-top height `z_top = D - e`
(which varies per embedment scenario) rather than at a fixed 1 m target --
never MAR-010's five roughness-sensitivity scenarios.

Marini wave Shields parameter: the model's OWN definition (Section 14)
-------------------------------------------------------------------------
Never substitutes MAR-013's `mobility_ratio`, MAR-012's combined stress,
or the Soulsby-Whitehouse critical Shields parameter into `theta` here --
Marini's `a(KC, theta)` calibration uses its own wave-Shields formulation,
computed independently in this module.

Four flow branches, each explicit (Sections 15, 18-19)
-----------------------------------------------------------
`COMBINED`/`WAVE_ONLY`/`CURRENT_ONLY`/`CALM`. The generalised KC (Eq. 30)
and Marini a/b coefficients are only evaluated for `COMBINED`/`WAVE_ONLY`;
`CURRENT_ONLY` uses the original Sumer et al. constants (a=0.025, b=0.5)
directly rather than forcing the generalised equations through
`theta=0`/`KC=infinity` (a genuine 0*infinity indeterminate form); `CALM`
carries no onset forcing at all.
"""

from typing import Any

import numpy as np
import pandas as pd

from marine_engine.metocean.combined_bed_shear import (
    compute_current_friction_velocity_m_s,
    fold_wave_current_axis_angle_deg,
)
from marine_engine.sediment.noncohesive_mobility import compute_z0_skin_m

# --- Fixed scientific constants (Sections 2-7 -- do not change) ---------------------

GRAVITY_M_S2 = 9.80665
RHO_WATER_KG_M3 = 1027.0
RHO_SEDIMENT_KG_M3 = 2650.0
VON_KARMAN_KAPPA = 0.40

SCIENTIFIC_ROLE = "PIPELINE_SCOUR_ONSET_EMBEDMENT_SCREENING"
SOURCE_MODEL_CITATION = (
    "Marini, F., Postacchini, M., Pizzigalli, C., Badalini, M., Corvaro, S., & "
    "Brocchini, M. (2024). On the onset of pipeline scouring: Reconciling waves and "
    "currents forcing. Coastal Engineering, 190, 104507."
)
SOURCE_MODEL_DOI = "10.1016/j.coastaleng.2024.104507"

PIPELINE_DIAMETER_M = 0.3048
PIPELINE_DIAMETER_SOURCE = "canonical PL854/NSTA pipeline evidence: 12 inch = 304.8 mm"

TESTED_D50_SCENARIOS_MM: tuple[float, ...] = (0.160, 0.250, 0.480)
D50_SCENARIO_SEMANTICS = "COMBINED_FLOW_CALIBRATION_ENVELOPE_D50_SCREENING_SCENARIOS"

TESTED_POROSITY_SCENARIOS: tuple[float, ...] = (0.35, 0.40, 0.45)
POROSITY_SCENARIO_SEMANTICS = "SANDY_BED_POROSITY_SENSITIVITY_SCENARIOS_NOT_SITE_MEASUREMENTS"

TESTED_EMBEDMENT_RATIOS: tuple[float, ...] = (0.00, 0.03, 0.06, 0.10, 0.15)
EMBEDMENT_SCENARIO_SEMANTICS = (
    "DISCRETE_TESTED_EMBEDMENT_SCREENING_SCENARIOS_NOT_ACTUAL_ROUTE_EMBEDMENT"
)

# Source experimental envelopes (Sections 3, 24) -- for applicability
# diagnostics only, never a reason to discard results.
SOURCE_DIAMETER_RANGE_M: tuple[float, float] = (0.05, 0.10)
SOURCE_CURRENT_ONLY_D50_RANGE_MM: tuple[float, float] = (0.18, 1.25)
SOURCE_WAVE_ONLY_D50_RANGE_MM: tuple[float, float] = (0.18, 0.18)
SOURCE_COMBINED_D50_RANGE_MM: tuple[float, float] = (0.16, 0.48)
SOURCE_UC_RANGE_M_S: tuple[float, float] = (0.13, 0.45)
SOURCE_UW_RANGE_M_S: tuple[float, float] = (0.07, 0.36)
SOURCE_KC_RANGE: tuple[float, float] = (1.6, 48.0)
SOURCE_EMBEDMENT_RANGE_OVERALL: tuple[float, float] = (0.01, 0.15)
SOURCE_EMBEDMENT_RANGE_COMBINED: tuple[float, float] = (0.03, 0.15)

PIPE_DIAMETER_OUTSIDE_SOURCE_ENVELOPE = "PIPE_DIAMETER_OUTSIDE_SOURCE_EXPERIMENT_ENVELOPE"
RESEARCH_SCREENING_EXTRAPOLATION = (
    "RESEARCH_SCREENING_EXTRAPOLATION_NOT_CALIBRATED_PL854_PREDICTION"
)

# KC/onset flow branches (Sections 15, 18-19)
COMBINED = "COMBINED"
WAVE_ONLY = "WAVE_ONLY"
CURRENT_ONLY = "CURRENT_ONLY"
CALM = "CALM"

SCOUR_ONSET_REACHED = "SCOUR_ONSET_CONDITION_REACHED"
SCOUR_ONSET_NOT_REACHED = "SCOUR_ONSET_CONDITION_NOT_REACHED"

REQUIRED_EMBEDMENT_IDENTIFIED = "REQUIRED_EMBEDMENT_IDENTIFIED"
ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT = "ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT"
ABOVE_TESTED_EMBEDMENT_RANGE = "ABOVE_TESTED_EMBEDMENT_RANGE"

MORPHOLOGY_ROLE = "LEGACY_REGIONAL_CONTEXT_ONLY"
MORPHOLOGY_SOURCE_ACQUISITION_YEARS = "1991-1992"


def _sediment_relative_density(
    rho_sediment: float = RHO_SEDIMENT_KG_M3, rho_water: float = RHO_WATER_KG_M3
) -> float:
    return rho_sediment / rho_water


class ScourOnsetCompletenessError(Exception):
    """More valid matched timestamps exist than the expected regular-cadence count allows."""


def _completeness_pct(valid_count: int, expected_count: int) -> float | None:
    if not expected_count:
        return None
    if valid_count > expected_count:
        raise ScourOnsetCompletenessError(
            f"{valid_count} valid matched timestamps exceeds the expected regular-cadence "
            f"count of {expected_count} -- completeness must never exceed 100%"
        )
    return 100.0 * valid_count / expected_count


# --- Pipeline-normal projection (Section 12) ------------------------------------------


def compute_local_tangent_bearing_deg(route, chainage_m: float, *, epsilon_m: float = 1.0) -> float:
    """Local route tangent bearing at `chainage_m`, from the TRUE curved geometry.

    Uses a numerical tangent (points at `chainage_m` +/- `epsilon_m`) --
    never a straight chord across the whole route. Degrees clockwise from
    true north, matching `current.compute_current_direction_to_deg`'s own
    convention exactly (so folding against a current/wave direction later
    is directly comparable).
    """

    total_length_m = route.length
    back_chainage = max(0.0, chainage_m - epsilon_m)
    forward_chainage = min(total_length_m, chainage_m + epsilon_m)
    if forward_chainage <= back_chainage:
        back_chainage, forward_chainage = 0.0, total_length_m

    start_point = route.interpolate(back_chainage)
    end_point = route.interpolate(forward_chainage)
    dx = end_point.x - start_point.x
    dy = end_point.y - start_point.y
    return float(np.degrees(np.arctan2(dx, dy)) % 360.0)


def compute_perpendicular_component(
    magnitude: np.ndarray, direction_deg: np.ndarray, tangent_bearing_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The pipeline-normal component of a directed flow, plus the folded axis angle used.

    The route tangent, like the wave axis, is an UNDIRECTED line (a pipe
    has no forward/backward preference) -- `fold_wave_current_axis_angle_deg`
    (reused directly from MAR-012) folds the raw direction difference to an
    acute angle in [0, 90] exactly as it does for the wave-current axis;
    `sin` of that folded angle is identical to `sin` of the raw minimal
    0..180 difference, so this is mathematically exact, never an
    approximation. Flow parallel to the tangent (angle=0) projects to
    zero; flow perpendicular to it (angle=90) projects to the full
    magnitude, unchanged (Section 12, tests M/N). A genuinely zero
    magnitude (e.g. zero current) projects to exactly zero regardless of
    direction -- direction is null exactly when magnitude is zero (the
    same zero-speed convention as MAR-012/013), and `0 * NaN` is `NaN` in
    IEEE 754, not `0`, so this is handled by an explicit branch rather than
    relying on bare arithmetic.
    """

    magnitude = np.asarray(magnitude, dtype=float)
    angle_deg = fold_wave_current_axis_angle_deg(direction_deg, tangent_bearing_deg)
    is_zero_magnitude = magnitude == 0
    with np.errstate(invalid="ignore"):
        general = np.abs(magnitude * np.sin(np.radians(angle_deg)))
    perpendicular = np.where(is_zero_magnitude, 0.0, general)
    return perpendicular, angle_deg


# --- Current reconstruction at tested pipe-top height (Section 13) ------------------


def compute_pipe_top_height_m(diameter_m: float, embedment_ratio: float) -> float:
    """`z_top = D * (1 - e/D)`. Requires a strictly positive result (Section 13)."""

    z_top = diameter_m * (1.0 - embedment_ratio)
    if not (z_top > 0):
        raise ValueError(
            f"embedment_ratio={embedment_ratio:g} on diameter_m={diameter_m:g} gives a "
            f"non-positive pipe-top height z_top={z_top:g} -- refusing to evaluate the "
            "log-profile there"
        )
    return z_top


def reconstruct_current_at_height_m_s(
    u_star_c: np.ndarray, z_target_m: float, z0_skin_m: float, *, kappa: float = VON_KARMAN_KAPPA
) -> np.ndarray:
    """`U(z_target) = u_star_c/kappa * ln((z_target+z0_skin)/z0_skin)` -- the SAME log-profile,
    evaluated at the tested pipe-top height rather than the reference sample's own height."""

    u_star_c = np.asarray(u_star_c, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (u_star_c / kappa) * np.log((z_target_m + z0_skin_m) / z0_skin_m)


# --- Flow-branch classification (Sections 15, 18-19) ----------------------------------


def classify_kc_branch(uc_perp: np.ndarray, uw_perp: np.ndarray) -> np.ndarray:
    uc_perp = np.asarray(uc_perp, dtype=float)
    uw_perp = np.asarray(uw_perp, dtype=float)
    branch = np.full(uc_perp.shape, None, dtype=object)
    branch[(uc_perp == 0) & (uw_perp == 0)] = CALM
    branch[(uw_perp == 0) & (uc_perp > 0)] = CURRENT_ONLY
    branch[(uc_perp == 0) & (uw_perp > 0)] = WAVE_ONLY
    branch[(uc_perp > 0) & (uw_perp > 0)] = COMBINED
    return branch


# --- Marini wave Shields parameter (Section 14) ---------------------------------------


def compute_wave_semi_excursion_m(uw_perp: np.ndarray, t_rep: np.ndarray) -> np.ndarray:
    """`a_x = Uw_perp*T/(2*pi)`; exactly 0 for Uw_perp==0 (never `0*NaN`)."""

    uw_perp = np.asarray(uw_perp, dtype=float)
    t_rep = np.asarray(t_rep, dtype=float)
    is_calm_wave = uw_perp == 0
    with np.errstate(invalid="ignore"):
        general = uw_perp * t_rep / (2.0 * np.pi)
    return np.where(is_calm_wave, 0.0, general)


def compute_marini_wave_friction_factor(a_x: np.ndarray, d50_m: float) -> np.ndarray:
    """`f_w = 0.04*(a_x/(2.5*d50))^(-0.25)`, defined only for `a_x > 0` (Section 14)."""

    a_x = np.asarray(a_x, dtype=float)
    eligible = a_x > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.04 * np.power(np.where(eligible, a_x, np.nan) / (2.5 * d50_m), -0.25)


def compute_marini_wave_friction_velocity_m_s(
    f_w_marini: np.ndarray, uw_perp: np.ndarray
) -> np.ndarray:
    """`u_star_w = sqrt(f_w/2) * Uw_perp`; exactly 0 for Uw_perp==0 (never `0*NaN`)."""

    f_w_marini = np.asarray(f_w_marini, dtype=float)
    uw_perp = np.asarray(uw_perp, dtype=float)
    is_calm_wave = uw_perp == 0
    with np.errstate(invalid="ignore"):
        general = np.sqrt(f_w_marini / 2.0) * uw_perp
    return np.where(is_calm_wave, 0.0, general)


def compute_marini_wave_shear_stress_pa(
    u_star_w: np.ndarray, rho_water: float = RHO_WATER_KG_M3
) -> np.ndarray:
    """`tau_w_marini = rho_water * u_star_w^2`."""

    return rho_water * np.asarray(u_star_w, dtype=float) ** 2


def compute_marini_wave_shields_parameter(
    tau_w_marini: np.ndarray,
    d50_m: float,
    *,
    rho_water: float = RHO_WATER_KG_M3,
    rho_sediment: float = RHO_SEDIMENT_KG_M3,
    g: float = GRAVITY_M_S2,
) -> np.ndarray:
    """`theta_w_marini = tau_w_marini / [rho_water*g*(s-1)*d50]`, the MODEL'S OWN wave
    Shields definition -- never MAR-012/013's combined stress or mobility ratio."""

    s = _sediment_relative_density(rho_sediment, rho_water)
    tau_w_marini = np.asarray(tau_w_marini, dtype=float)
    return tau_w_marini / (rho_water * g * (s - 1.0) * d50_m)


# --- Combined Keulegan-Carpenter number (Section 15) -----------------------------------


def compute_kc_marini(
    uc_perp: np.ndarray,
    uw_perp: np.ndarray,
    t_rep: np.ndarray,
    diameter_m: float,
    branch: np.ndarray,
) -> np.ndarray:
    """Marini et al. Eq. 30, evaluated only for COMBINED/WAVE_ONLY branches.

    CURRENT_ONLY/CALM never evaluate this (it would require dividing by
    zero and `exp(infinity)`, a genuine 0*infinity indeterminate form) --
    `kc_marini` stays null there; the current-only branch uses the
    original Sumer et al. constants directly instead (Section 18).
    """

    uc_perp = np.asarray(uc_perp, dtype=float)
    uw_perp = np.asarray(uw_perp, dtype=float)
    t_rep = np.asarray(t_rep, dtype=float)
    kc = np.full(uc_perp.shape, np.nan)

    is_wave_only = branch == WAVE_ONLY
    kc[is_wave_only] = uw_perp[is_wave_only] * t_rep[is_wave_only] / diameter_m

    is_combined = branch == COMBINED
    with np.errstate(over="ignore", invalid="ignore"):
        ratio_power = np.power(uc_perp[is_combined] / uw_perp[is_combined], 0.87)
        kc[is_combined] = (uw_perp[is_combined] * t_rep[is_combined] / diameter_m) * (
            -0.25 + 1.25 * np.exp(ratio_power)
        )
    return kc


# --- Combined velocity ratio / equivalent onset velocity (Section 16) ---------------


def compute_alpha(uc_perp: np.ndarray, uw_perp: np.ndarray) -> np.ndarray:
    """`alpha = Uc_perp / (Uc_perp+Uw_perp)`, defined only where the sum is positive."""

    uc_perp = np.asarray(uc_perp, dtype=float)
    uw_perp = np.asarray(uw_perp, dtype=float)
    total = uc_perp + uw_perp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, uc_perp / total, np.nan)


def compute_beta(alpha: np.ndarray) -> np.ndarray:
    """`beta = 6*alpha^3 - 16*alpha^2 + 10*alpha + 1`."""

    alpha = np.asarray(alpha, dtype=float)
    return 6.0 * alpha**3 - 16.0 * alpha**2 + 10.0 * alpha + 1.0


def compute_equivalent_onset_velocity_metric_m_s(
    uc_perp: np.ndarray, uw_perp: np.ndarray, beta: np.ndarray, branch: np.ndarray
) -> np.ndarray:
    """Never named `critical_velocity` -- an observed state has not necessarily reached
    onset (Section 16). CURRENT_ONLY uses `Uc_perp` directly (Section 18); CALM is 0."""

    uc_perp = np.asarray(uc_perp, dtype=float)
    uw_perp = np.asarray(uw_perp, dtype=float)
    beta = np.asarray(beta, dtype=float)
    total = uc_perp + uw_perp
    u_equivalent = np.full(uc_perp.shape, np.nan)

    u_equivalent[branch == CALM] = 0.0
    is_current_only = branch == CURRENT_ONLY
    u_equivalent[is_current_only] = uc_perp[is_current_only]

    is_general = (branch == WAVE_ONLY) | (branch == COMBINED)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_equivalent[is_general] = np.sqrt(total[is_general] ** 2 / beta[is_general])
    return u_equivalent


# --- Marini a/b onset coefficients (Sections 17-18) ------------------------------------


def compute_marini_x(kc_marini: np.ndarray, theta_w_marini: np.ndarray) -> np.ndarray:
    """`X = KC^0.69 * theta_w^0.70` -- only meaningful for COMBINED/WAVE_ONLY."""

    kc_marini = np.asarray(kc_marini, dtype=float)
    theta_w_marini = np.asarray(theta_w_marini, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.power(kc_marini, 0.69) * np.power(theta_w_marini, 0.70)


def compute_marini_a(x: np.ndarray) -> np.ndarray:
    """`a = 0.025*[1-exp(-14*X)]`. Never clamped to a convenient value."""

    x = np.asarray(x, dtype=float)
    return 0.025 * (1.0 - np.exp(-14.0 * x))


def compute_marini_b(x: np.ndarray) -> np.ndarray:
    """`b = 0.5 + exp(-2.9*X)`. Never clamped to a convenient value."""

    x = np.asarray(x, dtype=float)
    return 0.5 + np.exp(-2.9 * x)


def compute_marini_ab_coefficients(
    kc_marini: np.ndarray, theta_w_marini: np.ndarray, branch: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, a, b): general Marini formula for COMBINED/WAVE_ONLY; the original Sumer et al.
    current-only constants (a=0.025, b=0.5) for CURRENT_ONLY, never forced through the
    generalised equations (Section 18); null for all three in CALM."""

    x = np.full(branch.shape, np.nan)
    a = np.full(branch.shape, np.nan)
    b = np.full(branch.shape, np.nan)

    is_general = (branch == WAVE_ONLY) | (branch == COMBINED)
    x[is_general] = compute_marini_x(kc_marini[is_general], theta_w_marini[is_general])
    a[is_general] = compute_marini_a(x[is_general])
    b[is_general] = compute_marini_b(x[is_general])

    is_current_only = branch == CURRENT_ONLY
    a[is_current_only] = 0.025
    b[is_current_only] = 0.5

    return x, a, b


# --- Scour-onset condition (Section 19) -------------------------------------------------


def compute_omega_forcing(
    u_equivalent: np.ndarray,
    diameter_m: float,
    porosity: float,
    *,
    rho_sediment: float = RHO_SEDIMENT_KG_M3,
    rho_water: float = RHO_WATER_KG_M3,
    g: float = GRAVITY_M_S2,
) -> np.ndarray:
    """`Omega_forcing = u_equivalent^2 / [g*D*(s-1)*(1-n)]` -- `D` is the PIPE diameter,
    never a grain diameter."""

    s = _sediment_relative_density(rho_sediment, rho_water)
    u_equivalent = np.asarray(u_equivalent, dtype=float)
    return u_equivalent**2 / (g * diameter_m * (s - 1.0) * (1.0 - porosity))


def compute_omega_threshold(a: np.ndarray, b: np.ndarray, embedment_ratio: float) -> np.ndarray:
    """`Omega_threshold = a * exp(9*(e/D)^b)`."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a * np.exp(9.0 * np.power(embedment_ratio, b))


def classify_scour_onset_status(onset_margin: np.ndarray) -> np.ndarray:
    """`SCOUR_ONSET_CONDITION_REACHED` iff `Omega_forcing/Omega_threshold >= 1`."""

    onset_margin = np.asarray(onset_margin, dtype=float)
    status = np.full(onset_margin.shape, None, dtype=object)
    valid = np.isfinite(onset_margin)
    status[valid & (onset_margin >= 1.0)] = SCOUR_ONSET_REACHED
    status[valid & (onset_margin < 1.0)] = SCOUR_ONSET_NOT_REACHED
    return status


# --- Discrete required embedment (Section 20) -------------------------------------------


def determine_required_embedment(
    margins_by_embedment_ratio: dict[float, np.ndarray], diameter_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ratio, metres, status): the smallest tested e/D suppressing onset, per row.

    Never extrapolates above the largest tested embedment (0.15) -- a
    persisting onset there is flagged `ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT`,
    with a null numeric ratio, rather than an extrapolated value. Rows with
    no valid margin at any tested embedment (e.g. missing input data) are
    left null/status-less rather than conflated with either outcome.
    """

    sorted_ratios = sorted(TESTED_EMBEDMENT_RATIOS)
    length = len(next(iter(margins_by_embedment_ratio.values())))
    ratio = np.full(length, np.nan)
    status = np.full(length, None, dtype=object)
    resolved = np.zeros(length, dtype=bool)
    any_valid = np.zeros(length, dtype=bool)

    for embedment_ratio in sorted_ratios:
        margin = np.asarray(margins_by_embedment_ratio[embedment_ratio], dtype=float)
        valid = np.isfinite(margin)
        any_valid |= valid
        not_reached = valid & (margin < 1.0)
        newly_resolved = not_reached & ~resolved
        ratio[newly_resolved] = embedment_ratio
        status[newly_resolved] = REQUIRED_EMBEDMENT_IDENTIFIED
        resolved |= newly_resolved

    persists = any_valid & ~resolved
    status[persists] = ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT

    metres = ratio * diameter_m
    return ratio, metres, status


# --- Canonical 3-hourly output (Section 21) -------------------------------------------

_EMBEDMENT_COLUMN_SUFFIX: dict[float, str] = {
    0.00: "000",
    0.03: "003",
    0.06: "006",
    0.10: "010",
    0.15: "015",
}

SCOUR_ONSET_EMBEDMENT_3HOURLY_COLUMNS = (
    # Identity
    "hydro_pair_id",
    "current_node_id",
    "wave_node_id",
    "time_utc",
    # Pipeline
    "pipeline_diameter_m",
    "route_tangent_bearing_deg",
    # Sediment
    "tested_d50_mm",
    "tested_d50_m",
    "z0_skin_m",
    "porosity_scenario",
    "sediment_relative_density",
    # Hydrodynamic source
    "reference_current_speed_m_s",
    "reference_current_height_m",
    "current_direction_to_deg",
    "wave_orbital_amplitude_m_s",
    "representative_wave_period_s",
    "wave_direction_to_deg",
    # Projection/context
    "current_axis_to_route_angle_deg",
    "wave_axis_to_route_angle_deg",
    # Per-embedment onset margin/status
    "onset_margin_eD_000",
    "onset_margin_eD_003",
    "onset_margin_eD_006",
    "onset_margin_eD_010",
    "onset_margin_eD_015",
    "onset_status_eD_000",
    "onset_status_eD_003",
    "onset_status_eD_006",
    "onset_status_eD_010",
    "onset_status_eD_015",
    # Required embedment
    "minimum_tested_embedment_ratio_suppressing_onset",
    "minimum_tested_embedment_m_suppressing_onset",
    "required_embedment_status",
    # Provenance
    "scientific_role",
)

_WAVE_SIDE_RENAME = {
    "wave_orbital_velocity_equivalent_amplitude_m_s": "wave_orbital_amplitude_m_s",
    "equivalent_peak_period_from_tz_s": "representative_wave_period_s",
    "wave_mean_direction_to_deg": "wave_direction_to_deg",
}


def build_scour_onset_embedment_3hourly(
    current_hourly_df: pd.DataFrame,
    wave_3hourly_df: pd.DataFrame,
    hydro_pairs_df: pd.DataFrame,
    tangent_bearing_by_pair_id: dict[str, float],
    *,
    diameter_m: float = PIPELINE_DIAMETER_M,
) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    """LONG format: one row per `hydro_pair_id x time_utc x tested_d50_mm x porosity_scenario`.

    For every tested embedment, only compact `onset_margin_eD_*`/
    `onset_status_eD_*` columns are stored -- never a 5x row fan-out
    (Section 21). Each embedment scenario still requires its OWN complete
    recomputation of the current-at-pipe-top chain (since `z_top = D - e`
    changes the log-profile reconstruction height), computed once per D50
    scenario and reused across the three porosity scenarios (Omega_forcing's
    `(1-n)` term is the only porosity-dependent piece).

    Returns `(table, reference_diagnostics)` -- the second element carries
    the un-embedded, first-tested-D50 `Uc_perp`/`Uw_perp`/`kc_marini`/
    `branch`/axis-angle arrays used only for the Section 24-25 applicability
    and projection QA statistics (`None` if there is no matched data).
    """

    if current_hourly_df.empty or wave_3hourly_df.empty or hydro_pairs_df.empty:
        return pd.DataFrame(columns=list(SCOUR_ONSET_EMBEDMENT_3HOURLY_COLUMNS)), None

    current_side = current_hourly_df.merge(
        hydro_pairs_df[["current_node_id", "wave_node_id", "hydro_pair_id"]],
        on="current_node_id",
        how="inner",
    )
    wave_side = wave_3hourly_df.rename(columns=_WAVE_SIDE_RENAME)[
        [
            "wave_node_id",
            "time_utc",
            "wave_orbital_amplitude_m_s",
            "representative_wave_period_s",
            "wave_direction_to_deg",
        ]
    ]
    base = current_side.merge(wave_side, on=["wave_node_id", "time_utc"], how="inner")
    if base.empty:
        return pd.DataFrame(columns=list(SCOUR_ONSET_EMBEDMENT_3HOURLY_COLUMNS)), None

    n_rows = len(base)
    tangent_bearing = base["hydro_pair_id"].map(tangent_bearing_by_pair_id).to_numpy(dtype=float)

    u_ref = base["current_speed_m_s"].to_numpy(dtype=float)
    height_valid = base["height_above_model_bed_valid"].fillna(False).to_numpy(dtype=bool)
    z_ref_raw = base["height_above_model_bed_m"].to_numpy(dtype=float)
    z_ref_eligible = height_valid & np.isfinite(z_ref_raw) & (z_ref_raw > 0)
    z_ref = np.where(z_ref_eligible, z_ref_raw, np.nan)

    raw_current_dir = base["current_direction_to_deg"].to_numpy(dtype=float)
    speed_valid = np.isfinite(u_ref) & (u_ref > 0)
    current_dir = np.where(speed_valid, raw_current_dir, np.nan)

    uw = base["wave_orbital_amplitude_m_s"].to_numpy(dtype=float)
    t_rep = base["representative_wave_period_s"].to_numpy(dtype=float)
    wave_dir = base["wave_direction_to_deg"].to_numpy(dtype=float)

    uw_perp, wave_axis_angle = compute_perpendicular_component(uw, wave_dir, tangent_bearing)

    identity = {
        "hydro_pair_id": base["hydro_pair_id"].to_numpy(),
        "current_node_id": base["current_node_id"].to_numpy(),
        "wave_node_id": base["wave_node_id"].to_numpy(),
        "time_utc": base["time_utc"].to_numpy(),
        "pipeline_diameter_m": diameter_m,
        "route_tangent_bearing_deg": tangent_bearing,
        "reference_current_speed_m_s": u_ref,
        "reference_current_height_m": z_ref_raw,
        "current_direction_to_deg": current_dir,
        "wave_orbital_amplitude_m_s": uw,
        "representative_wave_period_s": t_rep,
        "wave_direction_to_deg": wave_dir,
        "wave_axis_to_route_angle_deg": wave_axis_angle,
    }

    scenario_frames = []
    reference_diagnostics: dict[str, np.ndarray] | None = None
    for d50_mm in TESTED_D50_SCENARIOS_MM:
        d50_m = d50_mm / 1000.0
        z0_skin = float(compute_z0_skin_m(d50_m))

        u_star_c = compute_current_friction_velocity_m_s(u_ref, z0_skin, target_height_m=z_ref)
        a_x = compute_wave_semi_excursion_m(uw_perp, t_rep)
        f_w_marini = compute_marini_wave_friction_factor(a_x, d50_m)
        u_star_w = compute_marini_wave_friction_velocity_m_s(f_w_marini, uw_perp)
        tau_w_marini = compute_marini_wave_shear_stress_pa(u_star_w)
        theta_w_marini = compute_marini_wave_shields_parameter(tau_w_marini, d50_m)

        current_axis_angle = None
        per_embedment: dict[float, dict[str, np.ndarray]] = {}
        for embedment_ratio in TESTED_EMBEDMENT_RATIOS:
            z_top = compute_pipe_top_height_m(diameter_m, embedment_ratio)
            uc_top = reconstruct_current_at_height_m_s(u_star_c, z_top, z0_skin)
            uc_perp, current_axis_angle = compute_perpendicular_component(
                uc_top, current_dir, tangent_bearing
            )
            branch = classify_kc_branch(uc_perp, uw_perp)
            kc_marini = compute_kc_marini(uc_perp, uw_perp, t_rep, diameter_m, branch)
            alpha = compute_alpha(uc_perp, uw_perp)
            beta = compute_beta(alpha)
            u_equivalent = compute_equivalent_onset_velocity_metric_m_s(
                uc_perp, uw_perp, beta, branch
            )
            _x, a_coef, b_coef = compute_marini_ab_coefficients(kc_marini, theta_w_marini, branch)
            omega_forcing_no_porosity = u_equivalent**2 / (
                GRAVITY_M_S2 * diameter_m * (_sediment_relative_density() - 1.0)
            )
            omega_threshold = compute_omega_threshold(a_coef, b_coef, embedment_ratio)
            per_embedment[embedment_ratio] = {
                "omega_forcing_no_porosity": omega_forcing_no_porosity,
                "omega_threshold": omega_threshold,
            }

            # Reference applicability/projection diagnostics (Sections 24-25):
            # captured once, at the FIRST tested D50 and zero embedment (the
            # natural un-embedded reference case) -- never written to the
            # compact 3-hourly parquet schema itself (Section 21's exact
            # column list), used only for route-wide QA reporting.
            if (
                reference_diagnostics is None
                and d50_mm == TESTED_D50_SCENARIOS_MM[0]
                and (embedment_ratio == 0.00)
            ):
                reference_diagnostics = {
                    "uc_perp": uc_perp,
                    "uw_perp": uw_perp,
                    "kc_marini": kc_marini,
                    "branch": branch,
                    "current_axis_angle_deg": current_axis_angle,
                    "wave_axis_angle_deg": wave_axis_angle,
                }

        for porosity in TESTED_POROSITY_SCENARIOS:
            margins_by_embedment: dict[float, np.ndarray] = {}
            row_data: dict[str, Any] = {
                **identity,
                "current_axis_to_route_angle_deg": current_axis_angle,
                "tested_d50_mm": d50_mm,
                "tested_d50_m": d50_m,
                "z0_skin_m": z0_skin,
                "porosity_scenario": porosity,
                "sediment_relative_density": _sediment_relative_density(),
            }
            for embedment_ratio in TESTED_EMBEDMENT_RATIOS:
                pe = per_embedment[embedment_ratio]
                omega_forcing = pe["omega_forcing_no_porosity"] / (1.0 - porosity)
                # CALM rows have `u_equivalent == 0` (so `omega_forcing == 0`
                # exactly) but never compute Marini a/b at all (Section 19's
                # own calm branch), so `omega_threshold` is NaN there --
                # explicitly force the margin to 0 (never reached) rather
                # than propagate `0 / NaN == NaN` (Section 19: "For calm
                # flow: NOT_REACHED").
                with np.errstate(invalid="ignore", divide="ignore"):
                    general_margin = omega_forcing / pe["omega_threshold"]
                margin = np.where(omega_forcing == 0, 0.0, general_margin)
                margins_by_embedment[embedment_ratio] = margin
                suffix = _EMBEDMENT_COLUMN_SUFFIX[embedment_ratio]
                row_data[f"onset_margin_eD_{suffix}"] = margin
                row_data[f"onset_status_eD_{suffix}"] = classify_scour_onset_status(margin)

            required_ratio, required_metres, required_status = determine_required_embedment(
                margins_by_embedment, diameter_m
            )
            row_data["minimum_tested_embedment_ratio_suppressing_onset"] = required_ratio
            row_data["minimum_tested_embedment_m_suppressing_onset"] = required_metres
            row_data["required_embedment_status"] = required_status
            row_data["scientific_role"] = SCIENTIFIC_ROLE

            scenario_frames.append(pd.DataFrame(row_data, index=range(n_rows)))

    result = pd.concat(scenario_frames, ignore_index=True)
    return result[list(SCOUR_ONSET_EMBEDMENT_3HOURLY_COLUMNS)], reference_diagnostics


# --- Embedment monotonicity QA (Section 23) ---------------------------------------------


class EmbedmentMonotonicityViolationError(Exception):
    """A row's onset margin increased with embedment -- never silently reordered/repaired."""


def compute_embedment_monotonicity_violations(
    mobility_df: pd.DataFrame, *, tolerance: float = 1e-9
) -> tuple[int, pd.Series]:
    """Per-row check: `onset_margin_eD_000 >= ... >= onset_margin_eD_015` (within tolerance).

    Returns `(violation_count, is_monotonic_per_row)`. A tight numerical
    tolerance absorbs floating-point noise only -- a genuine violation
    beyond it indicates a real inconsistency in the calculation, never
    silently reordered or repaired (Section 23).
    """

    if mobility_df.empty:
        return 0, pd.Series(dtype=bool)

    sorted_suffixes = [_EMBEDMENT_COLUMN_SUFFIX[e] for e in sorted(TESTED_EMBEDMENT_RATIOS)]
    columns = [f"onset_margin_eD_{suffix}" for suffix in sorted_suffixes]
    values = mobility_df[columns].to_numpy(dtype=float)

    is_monotonic = np.ones(len(mobility_df), dtype=bool)
    for i in range(values.shape[1] - 1):
        current_col, next_col = values[:, i], values[:, i + 1]
        both_valid = np.isfinite(current_col) & np.isfinite(next_col)
        violates = both_valid & (next_col > current_col + tolerance)
        is_monotonic &= ~violates

    violation_count = int((~is_monotonic).sum())
    return violation_count, pd.Series(is_monotonic, index=mobility_df.index)


def raise_if_embedment_monotonicity_violated(
    mobility_df: pd.DataFrame, *, tolerance: float = 1e-9
) -> int:
    """Hard-fails on any genuine violation -- the model's own structure (Omega_threshold
    increases with e/D; Omega_forcing cannot increase with embedment) guarantees
    monotonicity, so an observed violation indicates an unresolved defect (Section 23)."""

    violation_count, _ = compute_embedment_monotonicity_violations(mobility_df, tolerance=tolerance)
    if violation_count:
        raise EmbedmentMonotonicityViolationError(
            f"{violation_count} row(s) have a non-monotonic onset-margin sequence across "
            "tested embedments (increasing embedment made the onset condition easier to "
            "satisfy) -- this is not expected under the model's own structure and has not "
            "been silently reordered or repaired"
        )
    return violation_count


# --- Per-pair / per-D50 / per-porosity statistics (Section 22) -------------------------

SCOUR_ONSET_EMBEDMENT_STATS_COLUMNS = (
    "hydro_pair_id",
    "tested_d50_mm",
    "porosity_scenario",
    "overlap_start_time_utc",
    "overlap_end_time_utc",
    "matched_count",
    "completeness_pct",
    "onset_fraction_eD_000",
    "onset_fraction_eD_003",
    "onset_fraction_eD_006",
    "onset_fraction_eD_010",
    "onset_fraction_eD_015",
    "onset_pct_eD_000",
    "onset_pct_eD_003",
    "onset_pct_eD_006",
    "onset_pct_eD_010",
    "onset_pct_eD_015",
    "embedment_ratio_required_for_90pct_state_coverage",
    "embedment_ratio_required_for_90pct_state_coverage_status",
    "embedment_ratio_required_for_95pct_state_coverage",
    "embedment_ratio_required_for_95pct_state_coverage_status",
    "embedment_ratio_required_for_99pct_state_coverage",
    "embedment_ratio_required_for_99pct_state_coverage_status",
)


def _embedment_required_for_coverage(
    onset_fraction_by_ratio: dict[float, float | None], max_onset_fraction: float
) -> tuple[float | None, str]:
    """Smallest tested e/D whose onset fraction is `<= max_onset_fraction`.

    Never a continuous quantile interpolation between tested embedment
    levels (Section 22) -- discrete tested scenarios only.
    """

    for embedment_ratio in sorted(TESTED_EMBEDMENT_RATIOS):
        fraction = onset_fraction_by_ratio.get(embedment_ratio)
        if fraction is not None and fraction <= max_onset_fraction:
            return embedment_ratio, REQUIRED_EMBEDMENT_IDENTIFIED
    return None, ABOVE_TESTED_EMBEDMENT_RANGE


def compute_scour_onset_embedment_stats(mobility_df: pd.DataFrame) -> pd.DataFrame:
    """Per `hydro_pair_id x tested_d50_mm x porosity_scenario` descriptive statistics."""

    if mobility_df.empty:
        return pd.DataFrame(columns=list(SCOUR_ONSET_EMBEDMENT_STATS_COLUMNS))

    sorted_ratios = sorted(TESTED_EMBEDMENT_RATIOS)
    records = []
    for (pair_id, d50_mm, porosity), group in mobility_df.groupby(
        ["hydro_pair_id", "tested_d50_mm", "porosity_scenario"]
    ):
        start = group["time_utc"].min()
        end = group["time_utc"].max()
        expected_count = int(round((end - start).total_seconds() / (3 * 3600.0))) + 1
        matched_count = int(len(group))

        record: dict[str, Any] = {
            "hydro_pair_id": pair_id,
            "tested_d50_mm": d50_mm,
            "porosity_scenario": porosity,
            "overlap_start_time_utc": start,
            "overlap_end_time_utc": end,
            "matched_count": matched_count,
            "completeness_pct": _completeness_pct(matched_count, expected_count),
        }

        onset_fraction_by_ratio: dict[float, float | None] = {}
        for embedment_ratio in sorted_ratios:
            suffix = _EMBEDMENT_COLUMN_SUFFIX[embedment_ratio]
            margin = group[f"onset_margin_eD_{suffix}"].dropna()
            fraction = float((margin >= 1.0).sum() / len(margin)) if len(margin) else None
            onset_fraction_by_ratio[embedment_ratio] = fraction
            record[f"onset_fraction_eD_{suffix}"] = fraction
            record[f"onset_pct_eD_{suffix}"] = 100.0 * fraction if fraction is not None else None

        for label, max_fraction in (("90pct", 0.10), ("95pct", 0.05), ("99pct", 0.01)):
            value, status = _embedment_required_for_coverage(onset_fraction_by_ratio, max_fraction)
            record[f"embedment_ratio_required_for_{label}_state_coverage"] = value
            record[f"embedment_ratio_required_for_{label}_state_coverage_status"] = status

        records.append(record)
    return pd.DataFrame(records, columns=list(SCOUR_ONSET_EMBEDMENT_STATS_COLUMNS))


# --- Source calibration / applicability QA (Section 24) --------------------------------


def _fraction_within_range(values: np.ndarray, value_range: tuple[float, float]) -> float | None:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if not len(valid):
        return None
    return float(((valid >= value_range[0]) & (valid <= value_range[1])).sum() / len(valid))


def compute_applicability_diagnostics(
    reference_diagnostics: dict[str, np.ndarray] | None,
    mobility_df: pd.DataFrame,
    *,
    diameter_m: float = PIPELINE_DIAMETER_M,
) -> dict[str, Any]:
    """Fractions of projected combined-flow states within the source Marini/Zang
    experimental envelopes -- `APPLICABILITY_DIAGNOSTIC_NOT_CONFIDENCE_SCORE` (Section 24).

    `within_source_pipe_diameter_envelope`/`overall_direct_source_envelope_match` are
    always False for PL854 -- results are never discarded merely for this.
    Uc/Uw/KC fractions use the reference (un-embedded, first tested D50)
    projected diagnostics captured by the 3-hourly builder.
    """

    within_pipe_diameter_envelope = bool(
        SOURCE_DIAMETER_RANGE_M[0] <= diameter_m <= SOURCE_DIAMETER_RANGE_M[1]
    )

    empty = {
        "within_source_pipe_diameter_envelope": within_pipe_diameter_envelope,
        "overall_direct_source_envelope_match": False,
        "projected_uc_source_range_fraction": None,
        "projected_uw_source_range_fraction": None,
        "kc_source_range_fraction": None,
        "d50_source_range_fraction": None,
        "embedment_source_range_fraction": None,
    }
    if mobility_df.empty or not reference_diagnostics:
        return empty

    is_combined = reference_diagnostics["branch"] == COMBINED
    uc_fraction = _fraction_within_range(
        reference_diagnostics["uc_perp"][is_combined], SOURCE_UC_RANGE_M_S
    )
    uw_fraction = _fraction_within_range(
        reference_diagnostics["uw_perp"][is_combined], SOURCE_UW_RANGE_M_S
    )
    kc_fraction = _fraction_within_range(
        reference_diagnostics["kc_marini"][is_combined], SOURCE_KC_RANGE
    )

    d50_values = mobility_df["tested_d50_mm"].dropna()
    d50_fraction = (
        float(
            (
                (d50_values >= SOURCE_COMBINED_D50_RANGE_MM[0])
                & (d50_values <= SOURCE_COMBINED_D50_RANGE_MM[1])
            ).sum()
            / len(d50_values)
        )
        if len(d50_values)
        else None
    )

    return {
        "within_source_pipe_diameter_envelope": within_pipe_diameter_envelope,
        "overall_direct_source_envelope_match": False,
        "projected_uc_source_range_fraction": uc_fraction,
        "projected_uw_source_range_fraction": uw_fraction,
        "kc_source_range_fraction": kc_fraction,
        "d50_source_range_fraction": d50_fraction,
        "embedment_source_range_fraction": float(
            sum(
                1
                for e in TESTED_EMBEDMENT_RATIOS
                if SOURCE_EMBEDMENT_RANGE_COMBINED[0] <= e <= SOURCE_EMBEDMENT_RANGE_COMBINED[1]
            )
            / len(TESTED_EMBEDMENT_RATIOS)
        ),
    }


# --- Pipeline-normal projection QA (Section 25) -----------------------------------------


def compute_pipeline_normal_projection_qa(
    reference_diagnostics: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    """Angle-to-tangent and projection-ratio percentiles -- descriptive QA only, never
    converted into a confidence score (Section 25).

    The projection ratio `Uc_perp/Uc_top` (or `Uw_perp/Uw`) is mathematically
    identical to `abs(sin(angle_to_tangent))`, so it is derived directly
    from the stored axis-angle diagnostics rather than needing separate
    stored magnitude columns.
    """

    empty = {
        "current_axis_angle_median_deg": None,
        "current_axis_angle_p05_deg": None,
        "current_axis_angle_p95_deg": None,
        "wave_axis_angle_median_deg": None,
        "wave_axis_angle_p05_deg": None,
        "wave_axis_angle_p95_deg": None,
        "current_projection_ratio_median": None,
        "current_projection_ratio_p05": None,
        "current_projection_ratio_p95": None,
        "wave_projection_ratio_median": None,
        "wave_projection_ratio_p05": None,
        "wave_projection_ratio_p95": None,
    }
    if not reference_diagnostics:
        return empty

    current_angle = pd.Series(reference_diagnostics["current_axis_angle_deg"]).dropna()
    wave_angle = pd.Series(reference_diagnostics["wave_axis_angle_deg"]).dropna()
    current_ratio = (
        np.abs(np.sin(np.radians(current_angle))) if len(current_angle) else current_angle
    )
    wave_ratio = np.abs(np.sin(np.radians(wave_angle))) if len(wave_angle) else wave_angle

    def _stats(series: pd.Series) -> tuple[float | None, float | None, float | None]:
        if not len(series):
            return None, None, None
        return (
            float(series.median()),
            float(series.quantile(0.05)),
            float(series.quantile(0.95)),
        )

    current_angle_median, current_angle_p05, current_angle_p95 = _stats(current_angle)
    wave_angle_median, wave_angle_p05, wave_angle_p95 = _stats(wave_angle)
    current_ratio_median, current_ratio_p05, current_ratio_p95 = _stats(current_ratio)
    wave_ratio_median, wave_ratio_p05, wave_ratio_p95 = _stats(wave_ratio)

    return {
        "current_axis_angle_median_deg": current_angle_median,
        "current_axis_angle_p05_deg": current_angle_p05,
        "current_axis_angle_p95_deg": current_angle_p95,
        "wave_axis_angle_median_deg": wave_angle_median,
        "wave_axis_angle_p05_deg": wave_angle_p05,
        "wave_axis_angle_p95_deg": wave_angle_p95,
        "current_projection_ratio_median": current_ratio_median,
        "current_projection_ratio_p05": current_ratio_p05,
        "current_projection_ratio_p95": current_ratio_p95,
        "wave_projection_ratio_median": wave_ratio_median,
        "wave_projection_ratio_p05": wave_ratio_p05,
        "wave_projection_ratio_p95": wave_ratio_p95,
    }


# --- Sensitivity envelope across D50 x porosity (Section 26) ---------------------------

SENSITIVITY_ENVELOPE_COLUMNS = (
    "hydro_pair_id",
    "p95_required_embedment_lower_class",
    "p95_required_embedment_lower_ratio",
    "p95_required_embedment_upper_class",
    "p95_required_embedment_upper_ratio",
)


def _embedment_class_label(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f">{max(TESTED_EMBEDMENT_RATIOS):g}"
    return f"{value:g}"


def _embedment_class_sort_key(value: float | None) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("inf")
    return value


def compute_sensitivity_envelope(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Per hydro_pair_id: the discrete-class RANGE of `embedment_ratio_required_for_
    95pct_state_coverage` across all 3 D50 x 3 porosity scenarios (Section 26).

    Never a D50/porosity-averaged "best estimate" -- min/max class only,
    using the discrete ordering `0 < 0.03 < 0.06 < 0.10 < 0.15 < >0.15`.
    """

    if stats_df.empty:
        return pd.DataFrame(columns=list(SENSITIVITY_ENVELOPE_COLUMNS))

    records = []
    for pair_id, group in stats_df.groupby("hydro_pair_id"):
        normalized = [
            None if pd.isna(v) else float(v)
            for v in group["embedment_ratio_required_for_95pct_state_coverage"].tolist()
        ]
        if not normalized:
            continue
        lower = min(normalized, key=_embedment_class_sort_key)
        upper = max(normalized, key=_embedment_class_sort_key)
        records.append(
            {
                "hydro_pair_id": pair_id,
                "p95_required_embedment_lower_class": _embedment_class_label(lower),
                "p95_required_embedment_lower_ratio": lower,
                "p95_required_embedment_upper_class": _embedment_class_label(upper),
                "p95_required_embedment_upper_ratio": upper,
            }
        )
    return pd.DataFrame(records, columns=list(SENSITIVITY_ENVELOPE_COLUMNS))

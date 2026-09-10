"""Slope-instability screening mechanics (MAR-031 Sections 4-5, 13-16).

Pure array/scalar functions. Two strictly separated products:

A. Terrain-derived normalized undrained strength demand (real-data product; no soil parameters):

       normalized_undrained_strength_demand = sin(alpha) * cos(alpha)

   Derivation (undrained translational infinite slope, Baeten et al. 2014):
       tau_d = gamma_prime * z * sin(alpha) * cos(alpha)
       FS    = s_u / tau_d
   With N_u = s_u / (gamma_prime * z):  FS = N_u / (sin(alpha) * cos(alpha)); at FS = 1 the
   required ratio is N_u_required = sin(alpha) * cos(alpha). The demand is therefore the
   normalized undrained shear-strength ratio s_u / (gamma_prime * z) required to reach limit
   equilibrium under the simplified geometry. It is NOT actual soil strength, NOT a factor of
   safety, NOT a probability of failure, NOT a susceptibility score, and NOT a landslide risk.

B. Optional explicit hypothetical geotechnical scenario (never a default, never inferred):

       tau_driving_pa    = gamma_prime_n_m3 * slip_surface_depth_m * sin(alpha) * cos(alpha)
       factor_of_safety  = undrained_shear_strength_pa / tau_driving_pa    (tau_driving_pa > 0)

   Internal units are SI (Pa, N/m3, m). kPa / kN/m3 inputs are converted explicitly, once.

Model states are descriptive only (MODEL_FS_BELOW_1 / MODEL_FS_AT_1 / MODEL_FS_ABOVE_1 /
NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR / NOT_EVALUABLE). There is no safe/unsafe or low/
high verdict vocabulary, no design factor (e.g. FS >= 1.5), no minimum slope threshold, no slope
hazard class anywhere in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- Section 11: accepted scenario vocabulary (the ONLY accepted values) --------------------------

MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE = (
    "COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE"
)
PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO = "USER_DECLARED_HYPOTHETICAL_SCENARIO"

# --- Section 16: descriptive model states ---------------------------------------------------------

MODEL_FS_BELOW_1 = "MODEL_FS_BELOW_1"
MODEL_FS_AT_1 = "MODEL_FS_AT_1"
MODEL_FS_ABOVE_1 = "MODEL_FS_ABOVE_1"
NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR = "NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR"
NOT_EVALUABLE = "NOT_EVALUABLE"

MODEL_STATES: tuple[str, ...] = (
    MODEL_FS_BELOW_1,
    MODEL_FS_AT_1,
    MODEL_FS_ABOVE_1,
    NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR,
    NOT_EVALUABLE,
)

# Integer codes for a compact model-state raster; the legend is written into metadata alongside
# any state raster so the codes are never interpreted without their names.
MODEL_STATE_CODES: dict[str, int] = {
    NOT_EVALUABLE: 0,
    NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR: 1,
    MODEL_FS_BELOW_1: 2,
    MODEL_FS_AT_1: 3,
    MODEL_FS_ABOVE_1: 4,
}

# The mechanical limit-equilibrium boundary. It is the ONLY reference value in this module and it
# is a definition of the model (FS = s_u / tau_d = 1), not an engineering acceptance criterion.
FS_LIMIT_EQUILIBRIUM = 1.0
# Floating-point equality tolerance for "FS is exactly at limit equilibrium". A NUMERICAL
# tolerance for round-off in sin/cos/division, not a physical band around FS = 1.
FS_AT_1_RELATIVE_TOLERANCE = 1e-9

# Explicit unit conversions (Section 4): kPa -> Pa, kN/m3 -> N/m3.
PA_PER_KPA = 1000.0
N_PER_KN = 1000.0

# Geometrically supported slope range for the infinite-slope idealization, in degrees.
SLOPE_DEG_MIN = 0.0
SLOPE_DEG_MAX = 90.0


class SlopeStabilityInputError(ValueError):
    """A scenario parameter or slope input is outside the supported domain. Never clipped,
    defaulted, or inferred -- the caller must supply a valid explicit value."""


# --- Slope input validation -----------------------------------------------------------------------


def _validated_slope_deg(slope_deg: np.ndarray | float) -> np.ndarray:
    """Return a float64 READ-ONLY view (no copy when the input is already float64, so a
    hundreds-of-millions-of-cells raster is not duplicated merely to be checked). NaN propagates
    (unmeasured / nodata cells). Any finite value outside [0, 90] degrees is a geometric
    contradiction for a slope MAGNITUDE and fails explicitly -- it is never wrapped, clipped, or
    absolute-valued. Callers never write into the returned array; the input is never modified."""

    slope = np.asarray(slope_deg, dtype=np.float64)
    if np.isinf(slope).any():
        raise SlopeStabilityInputError("slope_deg contains infinite values")
    with np.errstate(invalid="ignore"):  # NaN compares False on both sides -> passes through
        out_of_range = (slope < SLOPE_DEG_MIN) | (slope > SLOPE_DEG_MAX)
    if out_of_range.any():
        bad = slope[out_of_range]
        raise SlopeStabilityInputError(
            f"slope_deg contains {int(out_of_range.sum())} finite value(s) outside "
            f"[{SLOPE_DEG_MIN:g}, {SLOPE_DEG_MAX:g}] degrees (e.g. {float(bad.flat[0])!r}); "
            "a slope magnitude outside this range is not geometrically supported"
        )
    return slope


def _sin_cos_product(slope: np.ndarray) -> np.ndarray:
    """sin(alpha) * cos(alpha) with a bounded working set: exactly two new full-size arrays
    (alpha reused in place for cos), so a full native-resolution raster stays tractable."""

    alpha = np.asarray(np.radians(slope), dtype=np.float64)  # asarray: 0-d scalars stay arrays
    product = np.asarray(np.sin(alpha), dtype=np.float64)
    np.cos(alpha, out=alpha)
    product *= alpha
    del alpha
    return product


# --- Section 5: normalized undrained strength demand (terrain-only product) -----------------------


def compute_normalized_strength_demand(slope_deg: np.ndarray | float) -> np.ndarray:
    """normalized_undrained_strength_demand = sin(alpha) * cos(alpha), dimensionless.

    Exact identities: 0 deg -> 0; 45 deg -> 0.5 (the maximum); 90 deg -> 0. NaN in -> NaN out.
    Always >= 0 on the supported range. No empirical coefficient, no calibration, no class.
    """

    slope = _validated_slope_deg(slope_deg)
    demand = _sin_cos_product(slope)
    if np.any(demand < 0.0):  # pragma: no cover -- mathematically unreachable on [0, 90]
        raise AssertionError("normalized strength demand must be non-negative on [0, 90] degrees")
    return demand


# --- Section 11 / 13: explicit hypothetical scenario ----------------------------------------------


def _require_finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise SlopeStabilityInputError(f"{name} must be a finite number > 0, got {value!r}")
    try:
        as_float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SlopeStabilityInputError(
            f"{name} must be a finite number > 0, got {value!r}"
        ) from exc
    if not math.isfinite(as_float) or as_float <= 0.0:
        raise SlopeStabilityInputError(
            f"{name} must be a finite number > 0 (zero, negative, NaN and infinity are rejected; "
            f"nothing is clipped or defaulted), got {value!r}"
        )
    return as_float


@dataclass(frozen=True)
class UndrainedInfiniteSlopeScenario:
    """One explicit, user-declared hypothetical geotechnical scenario (Sections 11-13).

    Every field is REQUIRED. `material_model` and `parameter_basis` must equal the single accepted
    value each -- MAR-031 does not accept a scenario as measured site geotechnical truth, and does
    not accept any other material model. Parameters are validated as finite and > 0; nothing is
    clipped, defaulted, or inferred (not from literature, mapped sediment classes,
    particle-size data, morphology, or water depth)."""

    scenario_id: str
    material_model: str
    parameter_basis: str
    undrained_shear_strength_kpa: float
    submerged_unit_weight_kn_m3: float
    slip_surface_depth_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise SlopeStabilityInputError("scenario_id must be a non-empty string")
        if self.material_model != MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE:
            raise SlopeStabilityInputError(
                f"material_model must be "
                f"{MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE!r} (the only "
                f"model implemented), got {self.material_model!r}"
            )
        if self.parameter_basis != PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO:
            raise SlopeStabilityInputError(
                f"parameter_basis must be "
                f"{PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO!r} (MAR-031 does not accept "
                f"measured site geotechnical truth), got {self.parameter_basis!r}"
            )
        object.__setattr__(
            self,
            "undrained_shear_strength_kpa",
            _require_finite_positive(
                "undrained_shear_strength_kpa", self.undrained_shear_strength_kpa
            ),
        )
        object.__setattr__(
            self,
            "submerged_unit_weight_kn_m3",
            _require_finite_positive(
                "submerged_unit_weight_kn_m3", self.submerged_unit_weight_kn_m3
            ),
        )
        object.__setattr__(
            self,
            "slip_surface_depth_m",
            _require_finite_positive("slip_surface_depth_m", self.slip_surface_depth_m),
        )

    @property
    def undrained_shear_strength_pa(self) -> float:
        return self.undrained_shear_strength_kpa * PA_PER_KPA

    @property
    def submerged_unit_weight_n_m3(self) -> float:
        return self.submerged_unit_weight_kn_m3 * N_PER_KN

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "material_model": self.material_model,
            "parameter_basis": self.parameter_basis,
            "undrained_shear_strength_kpa": self.undrained_shear_strength_kpa,
            "undrained_shear_strength_pa": self.undrained_shear_strength_pa,
            "submerged_unit_weight_kn_m3": self.submerged_unit_weight_kn_m3,
            "submerged_unit_weight_n_m3": self.submerged_unit_weight_n_m3,
            "slip_surface_depth_m": self.slip_surface_depth_m,
            "site_specific_measurement": False,
            "note": "USER_DECLARED_HYPOTHETICAL_SCENARIO -- NOT_SITE_SPECIFIC_MEASUREMENT",
        }


# --- Section 14: driving shear + scenario factor of safety ----------------------------------------


def compute_driving_shear_pa(
    slope_deg: np.ndarray | float, *, submerged_unit_weight_n_m3: float, slip_surface_depth_m: float
) -> np.ndarray:
    """tau_driving_pa = gamma_prime * z * sin(alpha) * cos(alpha), in Pa (SI inputs). Exactly 0
    at alpha = 0 (no idealized gravitational downslope shear). NaN in -> NaN out."""

    gamma = _require_finite_positive("submerged_unit_weight_n_m3", submerged_unit_weight_n_m3)
    depth = _require_finite_positive("slip_surface_depth_m", slip_surface_depth_m)
    slope = _validated_slope_deg(slope_deg)
    tau = _sin_cos_product(slope)
    tau *= gamma * depth
    return tau


@dataclass(frozen=True)
class ScenarioFactorOfSafetyResult:
    """Per-cell scenario result. `factor_of_safety` is NaN (null) wherever the idealized driving
    shear is exactly zero -- infinity is never written -- and wherever the slope is NaN. Every
    result carries its scenario identity so no output can be separated from its
    USER_DECLARED_HYPOTHETICAL_SCENARIO basis."""

    scenario_id: str
    material_model: str
    parameter_basis: str
    tau_driving_pa: np.ndarray
    factor_of_safety: np.ndarray
    model_state: np.ndarray  # object array of MODEL_STATES strings, same shape


def classify_model_state(factor_of_safety: float, tau_driving_pa: float) -> str:
    """Scalar descriptive classification against the limit-equilibrium boundary FS = 1.

    Non-finite tau -> NOT_EVALUABLE; tau == 0 -> NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR;
    non-finite FS -> NOT_EVALUABLE; else BELOW / AT / ABOVE 1 (AT within a pure floating-point
    tolerance). No safe/unsafe verdict vocabulary exists."""

    if not math.isfinite(tau_driving_pa):
        return NOT_EVALUABLE
    if tau_driving_pa == 0.0:
        return NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR
    if not math.isfinite(factor_of_safety):
        return NOT_EVALUABLE
    if math.isclose(
        factor_of_safety, FS_LIMIT_EQUILIBRIUM, rel_tol=FS_AT_1_RELATIVE_TOLERANCE, abs_tol=0.0
    ):
        return MODEL_FS_AT_1
    if factor_of_safety < FS_LIMIT_EQUILIBRIUM:
        return MODEL_FS_BELOW_1
    return MODEL_FS_ABOVE_1


def classify_model_states(factor_of_safety: np.ndarray, tau_driving_pa: np.ndarray) -> np.ndarray:
    """Vectorized `classify_model_state`, returning an object array of state strings."""

    fs = np.asarray(factor_of_safety, dtype=np.float64)
    tau = np.asarray(tau_driving_pa, dtype=np.float64)
    if fs.shape != tau.shape:
        raise SlopeStabilityInputError("factor_of_safety and tau_driving_pa shapes differ")
    state = np.full(fs.shape, NOT_EVALUABLE, dtype=object)
    tau_finite = np.isfinite(tau)
    no_shear = tau_finite & (tau == 0.0)
    evaluable = tau_finite & ~no_shear & np.isfinite(fs)
    with np.errstate(invalid="ignore"):
        at_1 = evaluable & np.isclose(
            fs, FS_LIMIT_EQUILIBRIUM, rtol=FS_AT_1_RELATIVE_TOLERANCE, atol=0.0
        )
        below_1 = evaluable & ~at_1 & (fs < FS_LIMIT_EQUILIBRIUM)
        above_1 = evaluable & ~at_1 & (fs > FS_LIMIT_EQUILIBRIUM)
    state[no_shear] = NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR
    state[below_1] = MODEL_FS_BELOW_1
    state[at_1] = MODEL_FS_AT_1
    state[above_1] = MODEL_FS_ABOVE_1
    return state


def encode_model_states(model_state: np.ndarray) -> np.ndarray:
    """Object state array -> int8 code raster per `MODEL_STATE_CODES`."""

    codes = np.full(model_state.shape, MODEL_STATE_CODES[NOT_EVALUABLE], dtype=np.int8)
    for name, code in MODEL_STATE_CODES.items():
        codes[model_state == name] = code
    return codes


def compute_scenario_factor_of_safety(
    slope_deg: np.ndarray | float, scenario: UndrainedInfiniteSlopeScenario
) -> ScenarioFactorOfSafetyResult:
    """FS = s_u / tau_driving_pa for every cell with nonzero finite driving shear; NaN (never
    infinity) where tau_driving_pa == 0 or the slope is NaN. A static, hydrostatic-only,
    undrained total-stress idealization: excess pore pressure, triggers, progressive/
    retrogressive failure, liquefaction and post-failure dynamics are NOT modelled.

    The input slope array is never modified."""

    if not isinstance(scenario, UndrainedInfiniteSlopeScenario):
        raise SlopeStabilityInputError(
            "scenario must be an explicit UndrainedInfiniteSlopeScenario -- there is no default"
        )
    tau = compute_driving_shear_pa(
        slope_deg,
        submerged_unit_weight_n_m3=scenario.submerged_unit_weight_n_m3,
        slip_surface_depth_m=scenario.slip_surface_depth_m,
    )
    fs = np.full(tau.shape, np.nan, dtype=np.float64)
    evaluable = np.isfinite(tau) & (tau > 0.0)
    fs[evaluable] = scenario.undrained_shear_strength_pa / tau[evaluable]
    state = classify_model_states(fs, tau)
    return ScenarioFactorOfSafetyResult(
        scenario_id=scenario.scenario_id,
        material_model=scenario.material_model,
        parameter_basis=scenario.parameter_basis,
        tau_driving_pa=tau,
        factor_of_safety=fs,
        model_state=state,
    )

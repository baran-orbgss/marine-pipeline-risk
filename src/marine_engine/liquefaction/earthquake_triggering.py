"""Boulanger & Idriss (2014) deterministic CPT-based earthquake liquefaction triggering (MAR-033
Sections 3-4, 7-21).

`evaluate_triggering_profile` is the single row-level orchestration function: given one canonical
CPT test's depth/qc/qt/fs/u2 series plus explicit declarations for every required input (earthquake
scenario, tip-resistance correction mode, vertical stress model, fines content, cohesionless-soil
applicability), it returns the full Section 21 profile table -- one row per input depth, fully
auditable (every intermediate term preserved), and independently gated per row: even a fully
declared scenario can still leave an individual row `NOT_EVALUABLE` (e.g. non-positive effective
stress at that specific depth).

A missing declaration (any parameter left `None`) never raises and never fabricates a default: it
makes every row `NOT_EVALUABLE` with the matching named reason (Section 24's "exact real-source
blockers" requirement). This mirrors `slope_stability.core`'s established pattern of an explicit,
non-fabricated model-state vocabulary with no safe/unsafe verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marine_engine.liquefaction import contract, cpt_normalization
from marine_engine.liquefaction.scenario import EarthquakeScenario, LiquefactionInputError
from marine_engine.liquefaction.stress import StressModel

__all__ = [
    "TipResistanceDeclaration",
    "FinesDeclaration",
    "SoilApplicabilityDeclaration",
    "TRIGGERING_PROFILE_COLUMNS",
    "GENERAL_CORRELATION_SENSITIVITY_VARIANT_TAGS",
    "stress_reduction_factor",
    "cyclic_stress_ratio",
    "expand_general_correlation_sensitivity_fines",
    "evaluate_triggering_profile",
]

TRIGGERING_PROFILE_COLUMNS: tuple[str, ...] = (
    "evidence_id",
    "test_id",
    "scenario_id",
    "depth_bsf_m",
    "tip_resistance_input_mpa",
    "tip_resistance_mode",
    "corrected_tip_resistance_mpa",
    "fs_kpa",
    "u2_kpa",
    "fines_content_percent",
    "fines_content_source",
    "sigma_v0_kpa",
    "sigma_v0_effective_kpa",
    "rd",
    "CN",
    "qcN",
    "qc1N",
    "delta_qc1N",
    "qc1Ncs",
    "CRR_7p5_1atm",
    "Csigma",
    "Ksigma",
    "MSFmax",
    "MSF",
    "CSR",
    "CRR_M_sigma",
    "FS_liq",
    "evaluation_state",
    "limitations",
    "method_id",
)


@dataclass(frozen=True)
class TipResistanceDeclaration:
    """Section 5: the accepted corrected-tip-resistance input mode. Raw qc + u2 with no declared
    area ratio is insufficient -- `area_ratio` is required (and used) only for
    QC_PLUS_U2_AND_DECLARED_AREA_RATIO; it is never inferred or assumed at a typical value."""

    mode: str
    basis: str
    area_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in contract.TIP_RESISTANCE_MODES:
            raise LiquefactionInputError(
                f"tip resistance mode must be one of {sorted(contract.TIP_RESISTANCE_MODES)}, "
                f"got {self.mode!r}"
            )
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise LiquefactionInputError(
                "TipResistanceDeclaration.basis must state the declared source"
            )
        if self.mode == contract.QC_PLUS_U2_AND_DECLARED_AREA_RATIO:
            if self.area_ratio is None:
                raise LiquefactionInputError(
                    "QC_PLUS_U2_AND_DECLARED_AREA_RATIO requires an explicit area_ratio "
                    "declaration -- it is never inferred or assumed at a typical value"
                )
            if not (0.0 < self.area_ratio < 1.0):
                raise LiquefactionInputError(
                    f"area_ratio must be in (0, 1), got {self.area_ratio!r}"
                )
        elif self.area_ratio is not None:
            raise LiquefactionInputError(
                f"area_ratio is only meaningful for QC_PLUS_U2_AND_DECLARED_AREA_RATIO, not "
                f"{self.mode!r}"
            )


@dataclass(frozen=True)
class FinesDeclaration:
    """Section 12: the accepted fines-content source. `fc_percent` is required for every source
    except CPT_ESTIMATED_FC_GENERAL_CORRELATION and GENERAL_CORRELATION_SENSITIVITY.

    CPT_ESTIMATED_FC_GENERAL_CORRELATION (the single-C_FC "expert mode") requires `c_fc` to be
    one of the three literature-defined sensitivity values (Section 14) -- never a
    silently-chosen 0.0. GENERAL_CORRELATION_SENSITIVITY (MAR-033A: the automatic mode that
    evaluates all three C_FC variants -- never one silently-chosen value) requires NEITHER
    `fc_percent` NOR `c_fc`: the three variants are resolved by
    `expand_general_correlation_sensitivity_fines`, one level above `evaluate_triggering_profile`,
    never by this declaration picking a single value.
    """

    source: str
    basis: str
    fc_percent: float | None = None
    c_fc: float | None = None

    def __post_init__(self) -> None:
        if self.source not in contract.FINES_CONTENT_SOURCES:
            raise LiquefactionInputError(
                f"fines source must be one of {sorted(contract.FINES_CONTENT_SOURCES)}, got "
                f"{self.source!r}"
            )
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise LiquefactionInputError("FinesDeclaration.basis must state the declared source")
        if self.source == contract.CPT_ESTIMATED_FC_GENERAL_CORRELATION:
            if self.c_fc not in contract.GENERAL_CORRELATION_C_FC_VALUES:
                raise LiquefactionInputError(
                    "CPT_ESTIMATED_FC_GENERAL_CORRELATION requires c_fc to be one of "
                    f"{contract.GENERAL_CORRELATION_C_FC_VALUES}, got {self.c_fc!r}"
                )
            if self.fc_percent is not None:
                raise LiquefactionInputError(
                    "fc_percent is not meaningful for CPT_ESTIMATED_FC_GENERAL_CORRELATION (it is "
                    "derived per-row from Ic); declare c_fc instead"
                )
        elif self.source == contract.GENERAL_CORRELATION_SENSITIVITY:
            if self.c_fc is not None:
                raise LiquefactionInputError(
                    "GENERAL_CORRELATION_SENSITIVITY evaluates all three literature C_FC values "
                    "automatically -- declare a single c_fc only for the semantically distinct "
                    f"CPT_ESTIMATED_FC_GENERAL_CORRELATION expert mode, got c_fc={self.c_fc!r}"
                )
            if self.fc_percent is not None:
                raise LiquefactionInputError(
                    "fc_percent is not meaningful for GENERAL_CORRELATION_SENSITIVITY (it is "
                    "derived per-row, per-variant, from Ic)"
                )
        else:
            if self.fc_percent is None:
                raise LiquefactionInputError(f"{self.source} requires an explicit fc_percent value")
            if not (contract.FC_BOUND_LOW <= self.fc_percent <= contract.FC_BOUND_HIGH):
                raise LiquefactionInputError(
                    f"fc_percent must be within [{contract.FC_BOUND_LOW}, {contract.FC_BOUND_HIGH}]"
                )


# GENERAL_CORRELATION_SENSITIVITY product-identity tags (MAR-033A): a fixed, explicit mapping over
# the three literal `contract.GENERAL_CORRELATION_C_FC_VALUES` -- never a formatted/derived string
# that could collide or drift if the literal values ever changed order.
GENERAL_CORRELATION_SENSITIVITY_VARIANT_TAGS: dict[float, str] = {
    -0.29: "C_FC_MINUS_0P29",
    0.0: "C_FC_0P00",
    0.29: "C_FC_PLUS_0P29",
}


def expand_general_correlation_sensitivity_fines(
    fines: FinesDeclaration,
) -> tuple[FinesDeclaration, ...]:
    """MAR-033A Part A item 6: expands one GENERAL_CORRELATION_SENSITIVITY declaration into the
    three literature-defined CPT_ESTIMATED_FC_GENERAL_CORRELATION variant declarations, one per
    C_FC in `contract.GENERAL_CORRELATION_C_FC_VALUES` (-0.29, 0.0, +0.29) -- never averaged,
    never reduced to one value, never converted into a probability. Each returned declaration is
    the semantically distinct single-C_FC "expert mode" (Section 7), used internally to compute
    one independent profile per variant."""

    if fines.source != contract.GENERAL_CORRELATION_SENSITIVITY:
        raise LiquefactionInputError(
            "expand_general_correlation_sensitivity_fines requires source="
            f"{contract.GENERAL_CORRELATION_SENSITIVITY!r}, got {fines.source!r}"
        )
    return tuple(
        FinesDeclaration(
            source=contract.CPT_ESTIMATED_FC_GENERAL_CORRELATION, basis=fines.basis, c_fc=c_fc
        )
        for c_fc in contract.GENERAL_CORRELATION_C_FC_VALUES
    )


@dataclass(frozen=True)
class SoilApplicabilityDeclaration:
    """Section 15: whether cohesionless-soil applicability is established, and how. A commonly
    used Ic threshold is never an invisible engine default -- `CPT_IC_SCREEN` requires an explicit
    `ic_cutoff` naming its own source."""

    established: bool
    basis_kind: str | None = None
    basis: str = ""
    ic_cutoff: float | None = None

    def __post_init__(self) -> None:
        if not self.established:
            return
        if self.basis_kind not in contract.SOIL_APPLICABILITY_BASES:
            raise LiquefactionInputError(
                f"basis_kind must be one of {sorted(contract.SOIL_APPLICABILITY_BASES)} when "
                f"applicability is established, got {self.basis_kind!r}"
            )
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise LiquefactionInputError(
                "SoilApplicabilityDeclaration.basis must state the declared source"
            )
        if self.basis_kind == contract.CPT_IC_SCREEN and self.ic_cutoff is None:
            raise LiquefactionInputError(
                "CPT_IC_SCREEN requires an explicit ic_cutoff -- a commonly used Ic threshold is "
                "never an invisible engine default"
            )


# --- Sections 7-8: cyclic stress ratio and depth-dependent stress reduction -----------------------


def stress_reduction_factor(depth_m: np.ndarray | float, moment_magnitude_mw: float) -> np.ndarray:
    """Section 8: rd = exp(alpha(z) + beta(z)*Mw). `depth_m` is metres; the sin() arguments are
    radians, matching the ticket's literal coefficients."""

    z = np.asarray(depth_m, dtype=np.float64)
    alpha = -1.012 - 1.126 * np.sin(z / 11.73 + 5.133)
    beta = 0.106 + 0.118 * np.sin(z / 11.28 + 5.142)
    return np.exp(alpha + beta * moment_magnitude_mw)


def cyclic_stress_ratio(
    pga_g: float,
    sigma_v0_kpa: np.ndarray | float,
    sigma_v0_effective_kpa: np.ndarray | float,
    rd: np.ndarray | float,
) -> np.ndarray:
    """Section 7: CSR = 0.65 * (a_max/g) * (sigma_v0 / sigma_v0') * rd, with a_max/g = pga_g."""

    sigma_v0 = np.asarray(sigma_v0_kpa, dtype=np.float64)
    sigma_eff = np.asarray(sigma_v0_effective_kpa, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.65 * pga_g * (sigma_v0 / sigma_eff) * np.asarray(rd, dtype=np.float64)


def _corrected_tip_resistance_mpa(
    qc_mpa: np.ndarray,
    qt_mpa: np.ndarray,
    u2_kpa: np.ndarray,
    declaration: TipResistanceDeclaration,
) -> tuple[np.ndarray, np.ndarray]:
    """Section 5: returns `(corrected_tip_resistance_mpa, tip_resistance_input_mpa)`. NaN
    propagates from a missing source channel -- never silently substituted."""

    if declaration.mode == contract.QT_MEASURED_OR_SOURCE_CORRECTED:
        return qt_mpa, qt_mpa
    if declaration.mode == contract.QC_EXPLICITLY_DECLARED_AREA_CORRECTED:
        return qc_mpa, qc_mpa
    if declaration.mode == contract.QC_PLUS_U2_AND_DECLARED_AREA_RATIO:
        ar = declaration.area_ratio
        with np.errstate(invalid="ignore"):
            qt_computed = qc_mpa + (1.0 - ar) * (u2_kpa / 1000.0)
        return qt_computed, qc_mpa
    raise AssertionError("unreachable: mode validated in TipResistanceDeclaration.__post_init__")


def _crr_7p5_1atm(qc1ncs: np.ndarray) -> np.ndarray:
    """Section 16. Coefficients are literal; never modified, never mixed with another curve.

    At a qc1Ncs far outside the method's calibrated range (e.g. from a near-zero effective
    stress a few millimetres below the seabed) the quartic term can overflow `exp` to `+inf` --
    a legitimate IEEE-754 result, suppressed here so it does not print a spurious warning. `+inf`
    then fails the `np.isfinite` evaluability check in `_classify_evaluation_states`, so such a
    row is reported `NOT_EVALUABLE`, never a fabricated finite FS_liq.
    """

    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(
            qc1ncs / 113.0
            + (qc1ncs / 1000.0) ** 2
            - (qc1ncs / 140.0) ** 3
            + (qc1ncs / 137.0) ** 4
            - 2.80
        )


def _ksigma(
    qc1ncs: np.ndarray, sigma_v0_effective_kpa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Section 17: returns `(Csigma, Ksigma)`. qc1Ncs is capped at 211 ONLY inside this
    expression (never elsewhere); Csigma <= 0.3; Ksigma <= 1.1."""

    bounded = np.minimum(qc1ncs, contract.QC1NCS_KSIGMA_BOUND_HIGH)
    with np.errstate(invalid="ignore", divide="ignore"):
        csigma = np.minimum(1.0 / (37.3 - 8.27 * bounded**0.264), contract.CSIGMA_MAX)
        ksigma = np.minimum(
            1.0 - csigma * np.log(sigma_v0_effective_kpa / contract.ATMOSPHERIC_PRESSURE_KPA),
            contract.KSIGMA_MAX,
        )
    return csigma, ksigma


def _msf(qc1ncs: np.ndarray, moment_magnitude_mw: float) -> tuple[np.ndarray, np.ndarray]:
    """Section 18: returns `(MSFmax, MSF)`. MSFmax <= 2.2."""

    msfmax = np.minimum(1.09 + (qc1ncs / 180.0) ** 3, contract.MSFMAX_MAX)
    msf = 1.0 + (msfmax - 1.0) * (8.64 * np.exp(-moment_magnitude_mw / 4.0) - 1.325)
    return msfmax, msf


def _classify_evaluation_states(fs_liq: np.ndarray, blocking: np.ndarray) -> np.ndarray:
    state = np.full(fs_liq.shape, contract.NOT_EVALUABLE, dtype=object)
    evaluable = ~blocking & np.isfinite(fs_liq)
    with np.errstate(invalid="ignore"):
        at_1 = evaluable & np.isclose(
            fs_liq, 1.0, rtol=contract.FS_AT_1_RELATIVE_TOLERANCE, atol=0.0
        )
        below_1 = evaluable & ~at_1 & (fs_liq < 1.0)
        above_1 = evaluable & ~at_1 & (fs_liq > 1.0)
    state[below_1] = contract.MODEL_FS_BELOW_1
    state[at_1] = contract.MODEL_FS_AT_1
    state[above_1] = contract.MODEL_FS_ABOVE_1
    return state


def evaluate_triggering_profile(
    cpt_rows: pd.DataFrame,
    *,
    evidence_id: str,
    scenario_id: str,
    scenario: EarthquakeScenario | None,
    tip_resistance: TipResistanceDeclaration | None,
    stress_model: StressModel | None,
    fines: FinesDeclaration | None,
    soil_applicability: SoilApplicabilityDeclaration | None,
    static_shear_material: bool = False,
) -> pd.DataFrame:
    """Section 21: the full row-level triggering profile for one CPT test's canonical rows.

    `cpt_rows` must carry `test_id`, `depth_bsf_m`, `qc_mpa`, `qt_mpa`, `fs_kpa`, `u2_kpa` (the
    canonical MAR-032 CPT_CANONICAL_PROFILE_V1 fields). Any declaration left `None` makes every
    row `NOT_EVALUABLE` with the matching named reason -- nothing is fabricated. A fully declared
    row can still independently become `NOT_EVALUABLE` (non-positive effective stress, a missing
    channel at that depth, a soil-applicability Ic screen failing at that depth, and so on).
    """

    n = len(cpt_rows)
    depth = cpt_rows["depth_bsf_m"].to_numpy(dtype=np.float64)
    qc = cpt_rows["qc_mpa"].to_numpy(dtype=np.float64)
    qt = cpt_rows["qt_mpa"].to_numpy(dtype=np.float64)
    fs_kpa = cpt_rows["fs_kpa"].to_numpy(dtype=np.float64)
    u2_kpa = cpt_rows["u2_kpa"].to_numpy(dtype=np.float64)

    blocking = np.zeros(n, dtype=bool)
    limitations: list[list[str]] = [[] for _ in range(n)]

    def _block_all(reason: str) -> None:
        blocking[:] = True
        for row in limitations:
            row.append(reason)

    def _block_where(mask: np.ndarray, reason: str) -> None:
        blocking[mask] = True
        for i in np.flatnonzero(mask):
            limitations[i].append(reason)

    def _note_where(mask: np.ndarray, reason: str) -> None:
        for i in np.flatnonzero(mask):
            limitations[i].append(reason)

    # --- Section 5: corrected tip resistance --------------------------------------------------
    if tip_resistance is None:
        corrected_tip_mpa = np.full(n, np.nan)
        tip_input_mpa = np.where(np.isfinite(qt), qt, qc)
        tip_mode: str | None = None
        _block_all(contract.CORRECTED_TIP_RESISTANCE_NOT_ESTABLISHED)
    else:
        corrected_tip_mpa, tip_input_mpa = _corrected_tip_resistance_mpa(
            qc, qt, u2_kpa, tip_resistance
        )
        tip_mode = tip_resistance.mode
        _block_where(
            ~np.isfinite(corrected_tip_mpa), contract.CORRECTED_TIP_RESISTANCE_NOT_ESTABLISHED
        )
    corrected_tip_kpa = corrected_tip_mpa * 1000.0

    # --- Section 9: vertical stress ------------------------------------------------------------
    if stress_model is None:
        sigma_v0 = np.full(n, np.nan)
        sigma_eff = np.full(n, np.nan)
        _block_all(contract.VERTICAL_STRESS_PROFILE_NOT_AVAILABLE)
    else:
        sigma_v0, sigma_eff = stress_model.stress_at_depths(depth)
        _block_where(
            ~np.isfinite(sigma_v0) | ~np.isfinite(sigma_eff),
            contract.MISSING_STRESS_AT_DEPTH,
        )
        nonpositive = np.isfinite(sigma_eff) & (sigma_eff <= 0.0)
        _block_where(nonpositive, contract.NONPOSITIVE_EFFECTIVE_STRESS)
    sigma_eff_safe = np.where(np.isfinite(sigma_eff) & (sigma_eff > 0.0), sigma_eff, np.nan)

    # --- Sections 14-15: Ic (shared by CPT-estimated fines and the Ic soil-applicability screen)
    need_ic = (
        fines is not None and fines.source == contract.CPT_ESTIMATED_FC_GENERAL_CORRELATION
    ) or (
        soil_applicability is not None
        and soil_applicability.established
        and soil_applicability.basis_kind == contract.CPT_IC_SCREEN
    )
    ic_result = None
    if need_ic:
        ic_result = cpt_normalization.iterate_ic_and_n(
            corrected_tip_kpa, sigma_v0, sigma_eff_safe, fs_kpa
        )

    # --- Section 12/14: fines content ----------------------------------------------------------
    if fines is None:
        fc_percent = np.full(n, np.nan)
        fc_source: str | None = None
        _block_all(contract.MISSING_FINES_OR_APPLICABILITY)
    elif fines.source == contract.GENERAL_CORRELATION_SENSITIVITY:
        # evaluate_triggering_profile computes exactly ONE profile per call. GENERAL_CORRELATION_
        # SENSITIVITY must be expanded into its three CPT_ESTIMATED_FC_GENERAL_CORRELATION variant
        # declarations (`expand_general_correlation_sensitivity_fines`) by the caller -- never
        # silently collapsed to one value here.
        raise LiquefactionInputError(
            "evaluate_triggering_profile computes exactly one profile; "
            f"{contract.GENERAL_CORRELATION_SENSITIVITY} must be expanded into three "
            f"{contract.CPT_ESTIMATED_FC_GENERAL_CORRELATION} variants via "
            "expand_general_correlation_sensitivity_fines before calling this function "
            "(see orchestration.execution for the per-variant fan-out)"
        )
    elif fines.source == contract.CPT_ESTIMATED_FC_GENERAL_CORRELATION:
        assert ic_result is not None
        # Section 14/MAR-033A: a non-converged Ic/n iteration must never be silently consumed to
        # derive fines content, even where it was mathematically `evaluable` (net resistance and
        # effective stress both positive) -- convergence is a separate, independently-gated fact.
        ic_usable = ic_result.evaluable & ic_result.converged
        fc_percent = np.where(
            ic_usable,
            cpt_normalization.estimate_fc_general_correlation(ic_result.ic, fines.c_fc),
            np.nan,
        )
        fc_source = fines.source
        _block_where(~ic_result.evaluable, contract.MISSING_FINES_OR_APPLICABILITY)
        _block_where(
            ic_result.evaluable & ~ic_result.converged, contract.IC_ITERATION_NOT_CONVERGED
        )
    else:
        fc_percent = np.full(n, fines.fc_percent)
        fc_source = fines.source

    # --- Section 15: cohesionless-soil applicability -------------------------------------------
    if soil_applicability is None or not soil_applicability.established:
        _block_all(contract.COHESIONLESS_SOIL_APPLICABILITY_NOT_ESTABLISHED)
    elif soil_applicability.basis_kind == contract.CPT_IC_SCREEN:
        assert ic_result is not None
        # Same dual evaluable/converged gate as the fines branch above: a non-converged Ic/n
        # result must not be silently used to decide cohesionless-soil applicability either.
        ic_usable = ic_result.evaluable & ic_result.converged
        _block_where(~ic_result.evaluable, contract.COHESIONLESS_SOIL_APPLICABILITY_NOT_ESTABLISHED)
        _block_where(
            ic_result.evaluable & ~ic_result.converged, contract.IC_ITERATION_NOT_CONVERGED
        )
        exceeds_cutoff = ic_usable & (ic_result.ic > soil_applicability.ic_cutoff)
        _block_where(exceeds_cutoff, contract.COHESIONLESS_SOIL_APPLICABILITY_NOT_ESTABLISHED)
    # SOURCE_ESTABLISHED: every row is treated as applicable; nothing further to gate.

    # --- Section 11: overburden normalization (NaN naturally propagates from any missing input)
    cn_result = cpt_normalization.iterate_cn_qc1ncs(corrected_tip_kpa, sigma_eff_safe, fc_percent)
    # MAR-033A: CN non-convergence is BLOCKING -- a numerical FS_liq built on a qc1Ncs the fixed
    # point never actually settled on is not an accepted model result (never MODEL_FS_*), even
    # though the intermediate CN/qc1Ncs values are still preserved in the row for auditability.
    _block_where(~cn_result.converged, contract.CN_ITERATION_NOT_CONVERGED)

    # --- Section 8 (limitation only, not blocking) ---------------------------------------------
    _note_where(
        depth > contract.RD_DEEP_EXTRAPOLATION_DEPTH_M, contract.RD_DEEP_EXTRAPOLATION_LIMITATION
    )

    # --- Sections 6-7, 16-20: scenario-dependent terms ----------------------------------------
    if scenario is None:
        rd = np.full(n, np.nan)
        csr = np.full(n, np.nan)
        msfmax = np.full(n, np.nan)
        msf = np.full(n, np.nan)
        _block_all(contract.MISSING_EARTHQUAKE_SCENARIO)
    else:
        rd = stress_reduction_factor(depth, scenario.moment_magnitude_mw)
        csr = cyclic_stress_ratio(scenario.pga_g, sigma_v0, sigma_eff_safe, rd)
        msfmax, msf = _msf(cn_result.qc1ncs, scenario.moment_magnitude_mw)
        with np.errstate(invalid="ignore"):
            nonpositive_csr = ~np.isfinite(csr) | (csr <= 0.0)
        _block_where(nonpositive_csr, contract.NONPOSITIVE_CSR)

    crr_7p5_1atm = _crr_7p5_1atm(cn_result.qc1ncs)
    csigma, ksigma = _ksigma(cn_result.qc1ncs, sigma_eff_safe)
    with np.errstate(invalid="ignore", over="ignore"):
        crr_m_sigma = crr_7p5_1atm * msf * ksigma
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        fs_liq = np.where(np.isfinite(csr) & (csr > 0.0), crr_m_sigma / csr, np.nan)

    # A row that passed every named gate above can still land on a non-finite FS_liq (e.g. CRR
    # overflow from a qc1Ncs far outside the calibrated range, typically from a near-zero
    # effective stress a few millimetres below the seabed) -- reported explicitly rather than
    # silently becoming NOT_EVALUABLE with no stated reason.
    _block_where(~blocking & ~np.isfinite(fs_liq), contract.NONFINITE_MODEL_RESULT)

    if static_shear_material:
        _note_where(np.ones(n, dtype=bool), contract.STATIC_SHEAR_OUTSIDE_MODEL_SCOPE)

    evaluation_state = _classify_evaluation_states(fs_liq, blocking)
    if static_shear_material:
        evaluation_state = np.where(
            evaluation_state == contract.NOT_EVALUABLE,
            evaluation_state,
            contract.OUTSIDE_METHOD_SUPPORT,
        )

    return pd.DataFrame(
        {
            "evidence_id": [evidence_id] * n,
            "test_id": cpt_rows["test_id"].to_numpy(),
            "scenario_id": [scenario_id] * n,
            "depth_bsf_m": depth,
            "tip_resistance_input_mpa": tip_input_mpa,
            "tip_resistance_mode": [tip_mode] * n,
            "corrected_tip_resistance_mpa": corrected_tip_mpa,
            "fs_kpa": fs_kpa,
            "u2_kpa": u2_kpa,
            "fines_content_percent": fc_percent,
            "fines_content_source": [fc_source] * n,
            "sigma_v0_kpa": sigma_v0,
            "sigma_v0_effective_kpa": sigma_eff,
            "rd": rd,
            "CN": cn_result.cn,
            "qcN": cn_result.qcn,
            "qc1N": cn_result.qc1n,
            "delta_qc1N": cn_result.delta_qc1n,
            "qc1Ncs": cn_result.qc1ncs,
            "CRR_7p5_1atm": crr_7p5_1atm,
            "Csigma": csigma,
            "Ksigma": ksigma,
            "MSFmax": msfmax,
            "MSF": msf,
            "CSR": csr,
            "CRR_M_sigma": crr_m_sigma,
            "FS_liq": fs_liq,
            "evaluation_state": evaluation_state,
            "limitations": [tuple(row) for row in limitations],
            "method_id": [contract.METHOD_ID] * n,
        },
        columns=list(TRIGGERING_PROFILE_COLUMNS),
    )

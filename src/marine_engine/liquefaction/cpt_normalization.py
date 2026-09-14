"""Boulanger & Idriss (2014) CPT resistance normalization (MAR-033 Sections 10-14).

Pure, deterministic, vectorized numpy functions -- no readiness gating and no I/O. Callers
(`earthquake_triggering.py`) are responsible for confirming required inputs exist (finite,
strictly positive effective stress, etc.) before calling; nothing here fabricates a missing
input, substitutes a default, or silently clips an out-of-domain value without reporting it back
through an explicit boolean/array result. Arrays are accepted so the real ~138k-row Sheringham
Shoal profile normalizes in one vectorized pass rather than a 138k-iteration Python loop.

Two fixed-point iterations are implemented, both with an explicit tolerance, an explicit maximum
iteration count and an explicit per-element convergence flag (Section 11):

* `iterate_cn_qc1ncs` -- the overburden-normalization fixed point CN -> qc1N -> qc1Ncs -> m -> CN.
* `iterate_ic_and_n` -- the CPT soil-behaviour-index Ic together with its stress exponent `n`,
  following Robertson (2009), "Interpretation of cone penetration tests -- a unified approach",
  Canadian Geotechnical Journal 46(11), 1337-1355 -- the procedure Boulanger & Idriss (2014)
  reference for their own Ic-based fines-content correlation (Section 14). `n` is never a fixed,
  invented constant: it is solved jointly with Ic (n = 0.381*Ic + 0.05*(sigma_v0_eff/Pa) - 0.15,
  bounded to [0.5, 1.0]) exactly as that procedure defines it.

Both iterations run for the full `max_iterations` on every element (no per-row early exit, so the
loop stays a fixed, small number of vectorized numpy passes regardless of array size) and report
convergence by comparing the final update against the one before it -- a converged fixed point is
stable, so running additional passes past convergence never changes the reported value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marine_engine.liquefaction import contract

__all__ = [
    "CnIterationResult",
    "IcIterationResult",
    "clean_sand_delta_qc1n",
    "iterate_cn_qc1ncs",
    "estimate_fc_general_correlation",
    "estimate_fc_general_correlation_sensitivity",
    "iterate_ic_and_n",
]


def clean_sand_delta_qc1n(qc1n: np.ndarray | float, fc_percent: np.ndarray | float) -> np.ndarray:
    """Section 13: the clean-sand equivalent CPT resistance correction, delta_qc1N."""

    qc1n_arr = np.asarray(qc1n, dtype=np.float64)
    fc_arr = np.asarray(fc_percent, dtype=np.float64)
    return (11.9 + qc1n_arr / 14.6) * np.exp(
        1.63 - 9.7 / (fc_arr + 2.0) - (15.7 / (fc_arr + 2.0)) ** 2
    )


@dataclass(frozen=True)
class CnIterationResult:
    """Section 11: the converged (or non-converged) overburden-normalization state, one entry per
    input element. `qc1ncs` is the UNBOUNDED value; the [21, 254] clamp in Section 11 is applied
    only inside the `m` exponent, never to the reported qc1Ncs itself."""

    cn: np.ndarray
    qcn: np.ndarray
    qc1n: np.ndarray
    delta_qc1n: np.ndarray
    qc1ncs: np.ndarray
    m: np.ndarray
    converged: np.ndarray
    tolerance: float
    max_iterations: int


def _m_exponent(qc1ncs: np.ndarray) -> np.ndarray:
    bounded = np.clip(qc1ncs, contract.QC1NCS_M_BOUND_LOW, contract.QC1NCS_M_BOUND_HIGH)
    return 1.338 - 0.249 * bounded**0.264


def iterate_cn_qc1ncs(
    qc_corrected_kpa: np.ndarray | float,
    sigma_v0_effective_kpa: np.ndarray | float,
    fc_percent: np.ndarray | float,
    *,
    tolerance: float = contract.CN_DEFAULT_TOLERANCE,
    max_iterations: int = contract.CN_DEFAULT_MAX_ITERATIONS,
) -> CnIterationResult:
    """Section 11: deterministic fixed-point iteration CN -> qc1N -> qc1Ncs -> m -> CN.

    `sigma_v0_effective_kpa` must already be confirmed strictly positive by the caller (the row
    orchestration in `earthquake_triggering.py` never calls this with a non-positive effective
    stress); a non-positive value here would only produce a non-finite CN, which then propagates
    as a non-finite (never fabricated) result.
    """

    qc = np.asarray(qc_corrected_kpa, dtype=np.float64)
    sigma_eff = np.asarray(sigma_v0_effective_kpa, dtype=np.float64)
    fc = np.asarray(fc_percent, dtype=np.float64)
    qcn = qc / contract.ATMOSPHERIC_PRESSURE_KPA

    cn = np.ones_like(qcn, dtype=np.float64)
    qc1n = cn * qcn
    delta = clean_sand_delta_qc1n(qc1n, fc)
    qc1ncs = qc1n + delta
    m = _m_exponent(qc1ncs)
    cn_before = cn

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for _ in range(max_iterations):
            cn_before = cn
            cn = np.minimum((contract.ATMOSPHERIC_PRESSURE_KPA / sigma_eff) ** m, contract.CN_MAX)
            qc1n = cn * qcn
            delta = clean_sand_delta_qc1n(qc1n, fc)
            qc1ncs = qc1n + delta
            m = _m_exponent(qc1ncs)

    converged = np.abs(cn - cn_before) < tolerance
    return CnIterationResult(
        cn=cn,
        qcn=qcn,
        qc1n=qc1n,
        delta_qc1n=delta,
        qc1ncs=qc1ncs,
        m=m,
        converged=converged,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def estimate_fc_general_correlation(ic: np.ndarray | float, c_fc: float) -> np.ndarray:
    """Section 14: FC = 80*(Ic + C_FC) - 137, bounded to [0, 100]."""

    ic_arr = np.asarray(ic, dtype=np.float64)
    fc = 80.0 * (ic_arr + c_fc) - 137.0
    return np.clip(fc, contract.FC_BOUND_LOW, contract.FC_BOUND_HIGH)


def estimate_fc_general_correlation_sensitivity(ic: np.ndarray | float) -> dict[float, np.ndarray]:
    """Section 14: the three literature-defined `GENERAL_CORRELATION_SENSITIVITY` scenarios --
    never a single silently-chosen `C_FC = 0` value passed off as site truth."""

    return {
        c_fc: estimate_fc_general_correlation(ic, c_fc)
        for c_fc in contract.GENERAL_CORRELATION_C_FC_VALUES
    }


@dataclass(frozen=True)
class IcIterationResult:
    """Section 14: the converged (or non-converged, or not-evaluable) Ic/n state, one entry per
    input element. `evaluable` is False wherever the net cone resistance
    `qc_corrected - sigma_v0` (or the effective stress) is not strictly positive -- Q and F are
    then mathematically undefined and `ic` is NaN, never a fabricated/zero-filled value."""

    evaluable: np.ndarray
    q: np.ndarray
    f: np.ndarray
    ic: np.ndarray
    n: np.ndarray
    converged: np.ndarray
    tolerance: float
    max_iterations: int


def iterate_ic_and_n(
    qc_corrected_kpa: np.ndarray | float,
    sigma_v0_kpa: np.ndarray | float,
    sigma_v0_effective_kpa: np.ndarray | float,
    fs_kpa: np.ndarray | float,
    *,
    tolerance: float = contract.IC_N_DEFAULT_TOLERANCE,
    max_iterations: int = contract.IC_N_DEFAULT_MAX_ITERATIONS,
) -> IcIterationResult:
    """Section 14: Robertson (2009) soil-behaviour index Ic, solved jointly with its stress
    exponent `n` (n = 0.381*Ic + 0.05*(sigma_v0_eff/Pa) - 0.15, bounded [0.5, 1.0])."""

    qc = np.asarray(qc_corrected_kpa, dtype=np.float64)
    sigma_v0 = np.asarray(sigma_v0_kpa, dtype=np.float64)
    sigma_eff = np.asarray(sigma_v0_effective_kpa, dtype=np.float64)
    fs = np.asarray(fs_kpa, dtype=np.float64)
    net_resistance = qc - sigma_v0

    shape = np.broadcast_shapes(net_resistance.shape, sigma_eff.shape, fs.shape)
    net_resistance = np.broadcast_to(net_resistance, shape).astype(np.float64)
    sigma_eff = np.broadcast_to(sigma_eff, shape).astype(np.float64)

    n = np.ones(shape, dtype=np.float64)
    q = np.full(shape, np.nan, dtype=np.float64)
    f = np.full(shape, np.nan, dtype=np.float64)
    ic = np.full(shape, np.nan, dtype=np.float64)
    n_before = n

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        evaluable = (net_resistance > 0.0) & (sigma_eff > 0.0)
        for _ in range(max_iterations):
            n_before = n
            q = np.where(
                evaluable,
                (net_resistance / contract.ATMOSPHERIC_PRESSURE_KPA)
                * (contract.ATMOSPHERIC_PRESSURE_KPA / sigma_eff) ** n,
                np.nan,
            )
            f = np.where(evaluable, fs / net_resistance * 100.0, np.nan)
            step_evaluable = evaluable & (q > 0.0) & (f > 0.0)
            ic = np.where(
                step_evaluable,
                np.sqrt((3.47 - np.log10(q)) ** 2 + (1.22 + np.log10(f)) ** 2),
                np.nan,
            )
            evaluable = step_evaluable
            n_new = 0.381 * ic + 0.05 * (sigma_eff / contract.ATMOSPHERIC_PRESSURE_KPA) - 0.15
            n = np.where(
                evaluable,
                np.clip(n_new, contract.IC_N_EXPONENT_MIN, contract.IC_N_EXPONENT_MAX),
                n,
            )

    converged = evaluable & (np.abs(n - n_before) < tolerance)
    return IcIterationResult(
        evaluable=evaluable,
        q=q,
        f=f,
        ic=ic,
        n=n,
        converged=converged,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

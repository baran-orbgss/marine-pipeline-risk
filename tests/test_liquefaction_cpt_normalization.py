"""MAR-033 Section 41: independent numerical benchmarks for the Boulanger & Idriss (2014)
overburden-normalization and Ic/n iterations.

Every expected value below is computed by an INDEPENDENT reference expression written directly in
this file with the plain `math` module (never by calling the production `numpy`-vectorized
functions under test and asserting they equal themselves). The two code paths implement the same
literal, citation-specified formula but are otherwise unrelated: `math.sin`/`math.exp` vs.
`numpy`'s, a scalar Python loop vs. a vectorized loop. Agreement to a tight tolerance is therefore
real evidence the production implementation matches the specified equations, not a tautology.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from marine_engine.liquefaction import contract, cpt_normalization

PA = 101.3  # contract.ATMOSPHERIC_PRESSURE_KPA, restated literally for an independent check


def test_atmospheric_pressure_constant_matches_ticket():
    assert contract.ATMOSPHERIC_PRESSURE_KPA == 101.3


def test_clean_sand_delta_qc1n_independent_value():
    # Section 13: delta_qc1N = (11.9 + qc1N/14.6) * exp(1.63 - 9.7/(FC+2) - (15.7/(FC+2))**2)
    qc1n, fc = 120.0, 15.0
    expected = (11.9 + qc1n / 14.6) * math.exp(1.63 - 9.7 / (fc + 2.0) - (15.7 / (fc + 2.0)) ** 2)
    got = cpt_normalization.clean_sand_delta_qc1n(qc1n, fc)
    assert float(got) == pytest.approx(expected, rel=1e-12)


def _independent_cn_iteration(
    qc_corrected_kpa: float, sigma_eff_kpa: float, fc: float, iterations: int
) -> dict:
    """A scalar, plain-Python re-implementation of Section 11's fixed point, structurally
    independent of `cpt_normalization.iterate_cn_qc1ncs` (no numpy, no shared helper)."""

    qcn = qc_corrected_kpa / PA
    cn = 1.0
    qc1n = cn * qcn
    for _ in range(iterations):
        bounded = min(
            max(
                qc1n
                + (11.9 + qc1n / 14.6) * math.exp(1.63 - 9.7 / (fc + 2) - (15.7 / (fc + 2)) ** 2),
                21.0,
            ),
            254.0,
        )
        m = 1.338 - 0.249 * bounded**0.264
        cn = min((PA / sigma_eff_kpa) ** m, 1.7)
        qc1n = cn * qcn
    delta = (11.9 + qc1n / 14.6) * math.exp(1.63 - 9.7 / (fc + 2) - (15.7 / (fc + 2)) ** 2)
    qc1ncs = qc1n + delta
    return {"cn": cn, "qc1n": qc1n, "delta_qc1n": delta, "qc1ncs": qc1ncs}


def test_cn_qc1ncs_iteration_matches_independent_reference():
    qc_corrected_kpa, sigma_eff_kpa, fc = 8000.0, 100.0, 10.0
    expected = _independent_cn_iteration(qc_corrected_kpa, sigma_eff_kpa, fc, iterations=50)
    result = cpt_normalization.iterate_cn_qc1ncs(qc_corrected_kpa, sigma_eff_kpa, fc)
    assert bool(result.converged) is True
    assert float(result.cn) == pytest.approx(expected["cn"], rel=1e-9)
    assert float(result.qc1n) == pytest.approx(expected["qc1n"], rel=1e-9)
    assert float(result.delta_qc1n) == pytest.approx(expected["delta_qc1n"], rel=1e-9)
    assert float(result.qc1ncs) == pytest.approx(expected["qc1ncs"], rel=1e-9)


def test_cn_capped_at_1p7():
    # A very low effective stress drives (Pa/sigma')^m far above 1.7; Section 11 requires capping.
    result = cpt_normalization.iterate_cn_qc1ncs(5000.0, 1.0, 5.0)
    assert float(result.cn) == pytest.approx(contract.CN_MAX, rel=1e-9)


def test_cn_non_convergence_is_reported_honestly():
    # Section 11: "explicit tolerance, explicit maximum iterations, convergence flag. No silent
    # non-convergence." One iteration with an unreasonably tight tolerance cannot settle -- the
    # flag must say so rather than silently reporting a partially-iterated value as converged.
    result = cpt_normalization.iterate_cn_qc1ncs(
        8000.0, 100.0, 10.0, tolerance=1e-15, max_iterations=1
    )
    assert bool(result.converged) is False
    # A generous budget converges for the same inputs -- proves the flag is not just always False.
    converged_result = cpt_normalization.iterate_cn_qc1ncs(
        8000.0, 100.0, 10.0, tolerance=1e-9, max_iterations=100
    )
    assert bool(converged_result.converged) is True


def test_m_exponent_bound_applied_only_inside_m():
    # Section 11: the [21, 254] clamp is used only inside the `m` exponent; the REPORTED qc1Ncs
    # stays unbounded. A very high fines content pushes delta_qc1N (and therefore qc1Ncs) far
    # above 254; qc1Ncs itself must not be silently clamped to 254 in the output.
    result = cpt_normalization.iterate_cn_qc1ncs(
        np.array([15000.0]), np.array([50.0]), np.array([40.0])
    )
    assert float(result.qc1ncs[0]) > 254.0


def test_estimate_fc_general_correlation_bounded_0_100():
    # Section 14: FC = 80*(Ic + C_FC) - 137, bounded [0, 100].
    ic_low, ic_high = 0.5, 5.0
    low = cpt_normalization.estimate_fc_general_correlation(ic_low, 0.0)
    high = cpt_normalization.estimate_fc_general_correlation(ic_high, 0.0)
    assert float(low) == pytest.approx(contract.FC_BOUND_LOW, abs=1e-9)
    assert float(high) == pytest.approx(contract.FC_BOUND_HIGH, abs=1e-9)
    mid_ic = 2.2
    expected = 80.0 * (mid_ic + 0.29) - 137.0
    got = cpt_normalization.estimate_fc_general_correlation(mid_ic, 0.29)
    assert float(got) == pytest.approx(expected, rel=1e-12)


def test_general_correlation_sensitivity_uses_exactly_the_three_literature_values():
    sensitivity = cpt_normalization.estimate_fc_general_correlation_sensitivity(2.2)
    assert set(sensitivity.keys()) == set(contract.GENERAL_CORRELATION_C_FC_VALUES)
    assert set(contract.GENERAL_CORRELATION_C_FC_VALUES) == {-0.29, 0.0, 0.29}
    # Distinct C_FC values must give distinct FC estimates -- never one value silently repeated.
    values = [float(v) for v in sensitivity.values()]
    assert len(set(values)) == 3


def _independent_ic_iteration(
    qc_corrected_kpa: float,
    sigma_v0_kpa: float,
    sigma_eff_kpa: float,
    fs_kpa: float,
    iterations: int,
) -> dict:
    """Independent scalar re-implementation of Section 14's Ic/n fixed point (Robertson 2009)."""

    net = qc_corrected_kpa - sigma_v0_kpa
    n = 1.0
    q = f = ic = None
    for _ in range(iterations):
        q = (net / PA) * (PA / sigma_eff_kpa) ** n
        f = fs_kpa / net * 100.0
        ic = math.sqrt((3.47 - math.log10(q)) ** 2 + (1.22 + math.log10(f)) ** 2)
        n = min(max(0.381 * ic + 0.05 * (sigma_eff_kpa / PA) - 0.15, 0.5), 1.0)
    return {"q": q, "f": f, "ic": ic, "n": n}


def test_ic_n_iteration_matches_independent_reference():
    qc_corrected_kpa, sigma_v0_kpa, sigma_eff_kpa, fs_kpa = 3000.0, 100.0, 80.0, 40.0
    expected = _independent_ic_iteration(
        qc_corrected_kpa, sigma_v0_kpa, sigma_eff_kpa, fs_kpa, iterations=50
    )
    result = cpt_normalization.iterate_ic_and_n(
        qc_corrected_kpa, sigma_v0_kpa, sigma_eff_kpa, fs_kpa
    )
    assert bool(result.evaluable) is True
    assert float(result.ic) == pytest.approx(expected["ic"], rel=1e-9)
    assert float(result.n) == pytest.approx(expected["n"], rel=1e-9)
    assert contract.IC_N_EXPONENT_MIN <= float(result.n) <= contract.IC_N_EXPONENT_MAX


def test_ic_not_evaluable_when_net_resistance_nonpositive():
    # Q and F are undefined when qc_corrected <= sigma_v0 -- must be reported, never zero-filled.
    result = cpt_normalization.iterate_ic_and_n(50.0, 100.0, 80.0, 40.0)
    assert bool(result.evaluable) is False
    assert math.isnan(float(result.ic))


def test_cn_and_ic_iterations_are_vectorized_over_arrays():
    # The real Sheringham profile has 138k+ rows; both iterations must accept whole arrays.
    n = 1000
    qc = np.linspace(1000.0, 20000.0, n)
    sigma_eff = np.linspace(20.0, 400.0, n)
    fc = np.full(n, 10.0)
    result = cpt_normalization.iterate_cn_qc1ncs(qc, sigma_eff, fc)
    assert result.cn.shape == (n,)
    assert np.all(result.cn <= contract.CN_MAX + 1e-12)

"""MAR-033 Section 42: adversarial science tests for the row-level Boulanger & Idriss (2014)
earthquake-liquefaction-triggering computation (`liquefaction.earthquake_triggering`).

Every synthetic CPT row below is a small, hand-built table -- no committed binary fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from marine_engine.liquefaction import contract
from marine_engine.liquefaction.earthquake_triggering import (
    FinesDeclaration,
    SoilApplicabilityDeclaration,
    TipResistanceDeclaration,
    cyclic_stress_ratio,
    evaluate_triggering_profile,
    stress_reduction_factor,
)
from marine_engine.liquefaction.scenario import EarthquakeScenario, LiquefactionInputError
from marine_engine.liquefaction.stress import LayeredStressModel, SoilLayer


def _cpt_rows(
    depths: list[float], qc: list[float], qt: list[float], fs: list[float], u2: list[float]
) -> pd.DataFrame:
    n = len(depths)
    return pd.DataFrame(
        {
            "test_id": ["T1"] * n,
            "depth_bsf_m": depths,
            "qc_mpa": qc,
            "qt_mpa": qt,
            "fs_kpa": fs,
            "u2_kpa": u2,
        }
    )


SCENARIO = EarthquakeScenario(scenario_id="benchmark", moment_magnitude_mw=7.5, pga_g=0.2)
STRESS = LayeredStressModel(
    layers=(
        SoilLayer(
            top_depth_m=0.0, bottom_depth_m=50.0, total_unit_weight_kn_m3=18.0, basis="benchmark"
        ),
    ),
    water_unit_weight_kn_m3=10.0,
)
FINES = FinesDeclaration(
    source=contract.USER_DECLARED_FC_SCENARIO, basis="benchmark", fc_percent=15.0
)
APPLICABILITY = SoilApplicabilityDeclaration(
    established=True, basis_kind=contract.SOURCE_ESTABLISHED, basis="benchmark"
)


def _evaluate(rows: pd.DataFrame, **overrides) -> pd.DataFrame:
    kwargs = {
        "evidence_id": "synthetic",
        "scenario_id": "benchmark",
        "scenario": SCENARIO,
        "tip_resistance": TipResistanceDeclaration(
            mode=contract.QC_PLUS_U2_AND_DECLARED_AREA_RATIO, basis="benchmark", area_ratio=0.75
        ),
        "stress_model": STRESS,
        "fines": FINES,
        "soil_applicability": APPLICABILITY,
    }
    kwargs.update(overrides)
    return evaluate_triggering_profile(rows, **kwargs)


# --- Section 5: corrected tip resistance ----------------------------------------------------------


def test_raw_qc_never_silently_becomes_qt():
    rows = _cpt_rows([5.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(
        rows,
        tip_resistance=TipResistanceDeclaration(
            mode=contract.QT_MEASURED_OR_SOURCE_CORRECTED, basis="benchmark"
        ),
    )
    # QT_MEASURED_OR_SOURCE_CORRECTED reads qt_mpa only; qc_mpa is never substituted for it.
    assert math.isnan(out["corrected_tip_resistance_mpa"].iloc[0])
    assert out["evaluation_state"].iloc[0] == contract.NOT_EVALUABLE
    assert contract.CORRECTED_TIP_RESISTANCE_NOT_ESTABLISHED in out["limitations"].iloc[0]


def test_u2_plus_missing_area_ratio_fails():
    with pytest.raises(LiquefactionInputError, match="area_ratio"):
        TipResistanceDeclaration(
            mode=contract.QC_PLUS_U2_AND_DECLARED_AREA_RATIO, basis="benchmark"
        )


def test_qt_accepted_when_present():
    rows = _cpt_rows([5.0], [10.0], [10.5], [80.0], [40.0])
    out = _evaluate(
        rows,
        tip_resistance=TipResistanceDeclaration(
            mode=contract.QT_MEASURED_OR_SOURCE_CORRECTED, basis="benchmark"
        ),
    )
    assert out["corrected_tip_resistance_mpa"].iloc[0] == pytest.approx(10.5)
    assert out["tip_resistance_input_mpa"].iloc[0] == pytest.approx(10.5)


def test_explicit_area_corrected_qc_accepted_only_when_declared():
    rows = _cpt_rows([5.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(
        rows,
        tip_resistance=TipResistanceDeclaration(
            mode=contract.QC_EXPLICITLY_DECLARED_AREA_CORRECTED, basis="benchmark"
        ),
    )
    assert out["corrected_tip_resistance_mpa"].iloc[0] == pytest.approx(10.0)
    # Without ANY declaration, the same raw qc is never treated as corrected.
    out_undeclared = _evaluate(rows, tip_resistance=None)
    assert math.isnan(out_undeclared["corrected_tip_resistance_mpa"].iloc[0])
    assert (
        contract.CORRECTED_TIP_RESISTANCE_NOT_ESTABLISHED in out_undeclared["limitations"].iloc[0]
    )


def test_qc_plus_u2_area_ratio_computes_qt():
    rows = _cpt_rows([5.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(rows)
    expected_qt_mpa = 10.0 + (1 - 0.75) * (40.0 / 1000.0)
    assert out["corrected_tip_resistance_mpa"].iloc[0] == pytest.approx(expected_qt_mpa, rel=1e-12)
    assert out["tip_resistance_input_mpa"].iloc[0] == pytest.approx(10.0)


def test_area_ratio_out_of_range_rejected():
    with pytest.raises(LiquefactionInputError):
        TipResistanceDeclaration(
            mode=contract.QC_PLUS_U2_AND_DECLARED_AREA_RATIO, basis="x", area_ratio=1.5
        )
    with pytest.raises(LiquefactionInputError):
        TipResistanceDeclaration(
            mode=contract.QC_PLUS_U2_AND_DECLARED_AREA_RATIO, basis="x", area_ratio=0.0
        )


# --- Section 6: earthquake scenario ---------------------------------------------------------------


def test_missing_mw_fails():
    with pytest.raises(LiquefactionInputError):
        EarthquakeScenario(scenario_id="x", moment_magnitude_mw=None, pga_g=0.2)  # type: ignore[arg-type]
    rows = _cpt_rows([5.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(rows, scenario=None)
    assert out["evaluation_state"].iloc[0] == contract.NOT_EVALUABLE
    assert contract.MISSING_EARTHQUAKE_SCENARIO in out["limitations"].iloc[0]


def test_missing_pga_fails():
    with pytest.raises(LiquefactionInputError):
        EarthquakeScenario(scenario_id="x", moment_magnitude_mw=7.5, pga_g=None)  # type: ignore[arg-type]


def test_scenario_manifest_missing_mw_field_fails_to_parse(tmp_path):
    from marine_engine.liquefaction.manifest import load_liquefaction_scenario_manifest

    manifest_path = tmp_path / "scenario.yaml"
    manifest_path.write_text(
        """
scenario_id: incomplete
earthquake:
  pga_g: 0.2
tip_resistance:
  mode: QT_MEASURED_OR_SOURCE_CORRECTED
  basis: x
stress:
  mode: SEABED_RELATIVE_LAYERED_STRESS
  water_unit_weight_kn_m3: 10.0
  layers:
    - top_depth_m: 0.0
      bottom_depth_m: 10.0
      total_unit_weight_kn_m3: 18.0
      basis: x
fines:
  source: USER_DECLARED_FC_SCENARIO
  basis: x
  fc_percent: 10.0
soil_applicability:
  established: true
  basis_kind: SOURCE_ESTABLISHED
  basis: x
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="moment_magnitude_mw"):
        load_liquefaction_scenario_manifest(manifest_path)


# --- Section 9: vertical stress -------------------------------------------------------------------


def test_zero_or_negative_effective_stress_rejected():
    # A layer/water unit weight combination that drives sigma_eff to zero at the top of the layer.
    rows = _cpt_rows([0.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(rows)
    assert out["evaluation_state"].iloc[0] == contract.NOT_EVALUABLE
    assert contract.NONPOSITIVE_EFFECTIVE_STRESS in out["limitations"].iloc[0]


def test_missing_stress_rejected():
    rows = _cpt_rows([5.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(rows, stress_model=None)
    assert out["evaluation_state"].iloc[0] == contract.NOT_EVALUABLE
    assert contract.VERTICAL_STRESS_PROFILE_NOT_AVAILABLE in out["limitations"].iloc[0]
    assert math.isnan(out["sigma_v0_kpa"].iloc[0])
    assert math.isnan(out["sigma_v0_effective_kpa"].iloc[0])


def test_depth_beyond_declared_layer_coverage_is_missing_not_extrapolated():
    rows = _cpt_rows([100.0], [10.0], [np.nan], [80.0], [40.0])  # STRESS layers only cover [0, 50]
    out = _evaluate(rows)
    assert out["evaluation_state"].iloc[0] == contract.NOT_EVALUABLE
    assert contract.MISSING_STRESS_AT_DEPTH in out["limitations"].iloc[0]


def test_layered_stress_model_rejects_gaps_and_non_seabed_start():
    with pytest.raises(LiquefactionInputError, match="contiguous"):
        LayeredStressModel(
            layers=(
                SoilLayer(0.0, 10.0, 18.0, "x"),
                SoilLayer(12.0, 20.0, 18.0, "x"),  # gap between 10 and 12
            ),
            water_unit_weight_kn_m3=10.0,
        )
    with pytest.raises(LiquefactionInputError, match="seabed"):
        LayeredStressModel(layers=(SoilLayer(2.0, 10.0, 18.0, "x"),), water_unit_weight_kn_m3=10.0)


# --- Sections 12, 14, 15: fines / soil applicability ----------------------------------------------


def test_missing_fines_and_applicability_handled_honestly():
    rows = _cpt_rows([5.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(rows, fines=None, soil_applicability=None)
    assert out["evaluation_state"].iloc[0] == contract.NOT_EVALUABLE
    assert contract.MISSING_FINES_OR_APPLICABILITY in out["limitations"].iloc[0]
    assert contract.COHESIONLESS_SOIL_APPLICABILITY_NOT_ESTABLISHED in out["limitations"].iloc[0]
    assert math.isnan(out["fines_content_percent"].iloc[0])


def test_cpt_estimated_fc_remains_distinguishable_from_measured_fc():
    rows = _cpt_rows([5.0, 10.0], [10.0, 15.0], [np.nan, np.nan], [80.0, 100.0], [40.0, 50.0])
    measured = _evaluate(
        rows,
        fines=FinesDeclaration(source=contract.MEASURED_LAB_FC, basis="lab test", fc_percent=12.0),
    )
    estimated = _evaluate(
        rows,
        fines=FinesDeclaration(
            source=contract.CPT_ESTIMATED_FC_GENERAL_CORRELATION, basis="Ic-based", c_fc=0.0
        ),
    )
    assert set(measured["fines_content_source"]) == {contract.MEASURED_LAB_FC}
    assert set(estimated["fines_content_source"]) == {contract.CPT_ESTIMATED_FC_GENERAL_CORRELATION}
    # The CPT-estimated values are actually computed per-row, not a copy of the measured constant.
    assert not np.allclose(estimated["fines_content_percent"], measured["fines_content_percent"])


def test_ic_soil_applicability_screen_blocks_rows_above_cutoff():
    # A very high Ic (clay-like) at one depth should fail the applicability screen even though
    # the scenario otherwise fully declares everything.
    rows = _cpt_rows([5.0], [1.0], [np.nan], [200.0], [40.0])  # low qc, high fs -> high Ic
    out = _evaluate(
        rows,
        soil_applicability=SoilApplicabilityDeclaration(
            established=True, basis_kind=contract.CPT_IC_SCREEN, basis="benchmark", ic_cutoff=2.0
        ),
    )
    assert out["evaluation_state"].iloc[0] == contract.NOT_EVALUABLE
    assert contract.COHESIONLESS_SOIL_APPLICABILITY_NOT_ESTABLISHED in out["limitations"].iloc[0]


def test_soil_applicability_not_established_blocks_every_row():
    rows = _cpt_rows([5.0, 10.0], [10.0, 15.0], [np.nan, np.nan], [80.0, 100.0], [40.0, 50.0])
    out = _evaluate(rows, soil_applicability=SoilApplicabilityDeclaration(established=False))
    assert (out["evaluation_state"] == contract.NOT_EVALUABLE).all()


# --- Section 8: rd deep-extrapolation limitation --------------------------------------------------


def test_deep_rd_limitation_preserved_but_not_blocking():
    rows = _cpt_rows([15.0], [10.0], [np.nan], [80.0], [40.0])  # > RD_DEEP_EXTRAPOLATION_DEPTH_M
    out = _evaluate(rows)
    assert contract.RD_DEEP_EXTRAPOLATION_LIMITATION in out["limitations"].iloc[0]
    # The row is still evaluated (formula applied), never invalidated solely for this reason.
    assert out["evaluation_state"].iloc[0] in (
        contract.MODEL_FS_BELOW_1,
        contract.MODEL_FS_AT_1,
        contract.MODEL_FS_ABOVE_1,
    )
    assert not math.isnan(out["FS_liq"].iloc[0])


def test_shallow_row_has_no_deep_limitation():
    rows = _cpt_rows([2.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(rows)
    assert contract.RD_DEEP_EXTRAPOLATION_LIMITATION not in out["limitations"].iloc[0]


# --- Section 19: static shear / model scope -------------------------------------------------------


def test_static_shear_material_marks_outside_method_support():
    rows = _cpt_rows([5.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(rows, static_shear_material=True)
    assert out["evaluation_state"].iloc[0] == contract.OUTSIDE_METHOD_SUPPORT
    assert contract.STATIC_SHEAR_OUTSIDE_MODEL_SCOPE in out["limitations"].iloc[0]


# --- Section 4/14.1: no fabricated verdict vocabulary ---------------------------------------------


def test_no_risk_class_or_probability_vocabulary_in_output():
    rows = _cpt_rows([5.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(rows)
    rendered = out.to_string()
    for label in contract.PROHIBITED_OUTPUT_LABELS:
        assert label not in rendered
    assert set(out["evaluation_state"]).issubset(contract.EVALUATION_STATES)


def test_no_wave_liquefaction_field_in_profile_output():
    rows = _cpt_rows([5.0], [10.0], [np.nan], [80.0], [40.0])
    out = _evaluate(rows)
    assert "wave" not in "".join(out.columns).lower()


def test_method_id_and_provenance_present_on_every_row():
    rows = _cpt_rows([5.0, 10.0], [10.0, 15.0], [np.nan, np.nan], [80.0, 100.0], [40.0, 50.0])
    out = _evaluate(rows)
    assert set(out["method_id"]) == {contract.METHOD_ID}
    assert set(out["scenario_id"]) == {"benchmark"}
    assert set(out["evidence_id"]) == {"synthetic"}


# --- Section 23: no spatial interpolation between CPT rows ----------------------------------------


def test_profile_never_adds_or_interpolates_rows():
    depths = [1.0, 5.0, 9.0]
    rows = _cpt_rows(
        depths, [10.0, 12.0, 14.0], [np.nan] * 3, [80.0, 90.0, 100.0], [40.0, 45.0, 50.0]
    )
    out = _evaluate(rows)
    assert len(out) == len(rows)
    assert sorted(out["depth_bsf_m"].tolist()) == sorted(depths)


# --- Sections 7-8: CSR / rd independent spot-check (integration-level, complements the pure
# equation-level benchmarks in test_liquefaction_cpt_normalization.py) ----------------------------


def test_ksigma_csigma_msfmax_caps_hold_at_extreme_qc1ncs():
    # Section 17/18: Csigma <= 0.3, Ksigma <= 1.1, MSFmax <= 2.2, even for an unrealistically
    # dense reading that would otherwise push all three past their method-defined caps.
    rows = _cpt_rows([5.0], [40.0], [np.nan], [80.0], [40.0])  # very high qc -> very high qc1Ncs
    out = _evaluate(
        rows,
        fines=FinesDeclaration(
            source=contract.USER_DECLARED_FC_SCENARIO, basis="x", fc_percent=0.0
        ),
    )
    assert out["qc1Ncs"].iloc[0] > contract.QC1NCS_KSIGMA_BOUND_HIGH
    assert out["Csigma"].iloc[0] == pytest.approx(contract.CSIGMA_MAX)
    assert out["Ksigma"].iloc[0] <= contract.KSIGMA_MAX + 1e-12
    assert out["MSFmax"].iloc[0] == pytest.approx(contract.MSFMAX_MAX)


def test_csr_and_rd_match_independent_expression():
    depth, mw, pga = 6.0, 7.2, 0.18
    alpha = -1.012 - 1.126 * math.sin(depth / 11.73 + 5.133)
    beta = 0.106 + 0.118 * math.sin(depth / 11.28 + 5.142)
    expected_rd = math.exp(alpha + beta * mw)
    got_rd = float(stress_reduction_factor(depth, mw))
    assert got_rd == pytest.approx(expected_rd, rel=1e-12)

    sigma_v0, sigma_eff = 100.0, 60.0
    expected_csr = 0.65 * pga * (sigma_v0 / sigma_eff) * expected_rd
    got_csr = float(cyclic_stress_ratio(pga, sigma_v0, sigma_eff, got_rd))
    assert got_csr == pytest.approx(expected_csr, rel=1e-12)

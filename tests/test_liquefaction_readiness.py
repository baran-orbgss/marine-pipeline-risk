"""MAR-033 Section 40: axis-by-axis earthquake-triggering capability readiness.

`CPT_IDENTITY` and `DEPTH_REFERENCE` delegate to `geotechnical.cpt_readiness.CptEvidenceFacts` --
these tests build that dataclass directly rather than a real parquet file, since the delegation
itself (not MAR-032's own file-reading machinery) is what this module is responsible for.
"""

from __future__ import annotations

from marine_engine.geotechnical import cpt_contract
from marine_engine.geotechnical.cpt_readiness import CptEvidenceFacts
from marine_engine.liquefaction import contract
from marine_engine.liquefaction.earthquake_triggering import (
    FinesDeclaration,
    SoilApplicabilityDeclaration,
    TipResistanceDeclaration,
)
from marine_engine.liquefaction.readiness import (
    EarthquakeTriggeringReadinessFacts,
    assess_earthquake_triggering_readiness,
    wave_liquefaction_status,
)
from marine_engine.liquefaction.scenario import EarthquakeScenario

VERIFIED_CPT_FACTS = CptEvidenceFacts(
    machine_readable_profile_available=True,
    canonical_profile_created=True,
    canonical_product_identity_verified=True,
    measured_evidence_role_verified=True,
    row_count=100,
    depth_reference=cpt_contract.DEPTH_BELOW_SEABED,
    depth_bsf_available=True,
    channels_present=(cpt_contract.QC_MPA, cpt_contract.FS_KPA, cpt_contract.U2_KPA),
)

FULLY_DECLARED_KWARGS = {
    "tip_resistance_declared": TipResistanceDeclaration(
        mode=contract.QC_PLUS_U2_AND_DECLARED_AREA_RATIO, basis="x", area_ratio=0.75
    ),
    "stress_model_declared": True,
    "stress_model_valid": True,
    "stress_model_problem": None,
    "fines_declared": FinesDeclaration(
        source=contract.USER_DECLARED_FC_SCENARIO, basis="x", fc_percent=10.0
    ),
    "soil_applicability_declared": SoilApplicabilityDeclaration(
        established=True, basis_kind=contract.SOURCE_ESTABLISHED, basis="x"
    ),
    "scenario_declared": EarthquakeScenario(scenario_id="x", moment_magnitude_mw=7.5, pga_g=0.2),
}


def test_fully_declared_scenario_is_ready():
    facts = EarthquakeTriggeringReadinessFacts(
        cpt_evidence_facts=VERIFIED_CPT_FACTS, **FULLY_DECLARED_KWARGS
    )
    result = assess_earthquake_triggering_readiness(facts)
    assert result.status == contract.READY
    for axis in result.axes:
        assert axis.status == contract.READY, axis.to_dict()


def test_unverified_cpt_evidence_blocks_cpt_identity():
    unverified = CptEvidenceFacts(canonical_product_identity_verified=False)
    facts = EarthquakeTriggeringReadinessFacts(
        cpt_evidence_facts=unverified, **FULLY_DECLARED_KWARGS
    )
    result = assess_earthquake_triggering_readiness(facts)
    assert result.status == contract.NOT_READY
    assert result.axis(contract.AXIS_CPT_IDENTITY).status == contract.NOT_READY


def test_no_tip_resistance_declaration_blocks_that_axis_only():
    kwargs = dict(FULLY_DECLARED_KWARGS)
    kwargs["tip_resistance_declared"] = None
    facts = EarthquakeTriggeringReadinessFacts(cpt_evidence_facts=VERIFIED_CPT_FACTS, **kwargs)
    result = assess_earthquake_triggering_readiness(facts)
    assert result.status == contract.NOT_READY
    assert result.axis(contract.AXIS_CORRECTED_TIP_RESISTANCE).status == contract.NOT_READY
    assert result.axis(contract.AXIS_STRESS_PROFILE).status == contract.READY


def test_declared_mode_requires_its_own_channel_present():
    kwargs = dict(FULLY_DECLARED_KWARGS)
    kwargs["tip_resistance_declared"] = TipResistanceDeclaration(
        mode=contract.QT_MEASURED_OR_SOURCE_CORRECTED, basis="x"
    )
    # VERIFIED_CPT_FACTS.channels_present has no QT_MPA -- the declared mode's channel is absent.
    facts = EarthquakeTriggeringReadinessFacts(cpt_evidence_facts=VERIFIED_CPT_FACTS, **kwargs)
    result = assess_earthquake_triggering_readiness(facts)
    axis = result.axis(contract.AXIS_CORRECTED_TIP_RESISTANCE)
    assert axis.status == contract.NOT_READY
    assert any(
        contract.CORRECTED_TIP_RESISTANCE_NOT_ESTABLISHED in r for r in axis.blocking_reasons
    )


def test_no_stress_model_blocks_stress_axis():
    kwargs = dict(FULLY_DECLARED_KWARGS)
    kwargs["stress_model_declared"] = False
    kwargs["stress_model_valid"] = False
    facts = EarthquakeTriggeringReadinessFacts(cpt_evidence_facts=VERIFIED_CPT_FACTS, **kwargs)
    result = assess_earthquake_triggering_readiness(facts)
    assert result.axis(contract.AXIS_STRESS_PROFILE).status == contract.NOT_READY


def test_no_fines_or_no_applicability_each_block_the_combined_axis():
    kwargs_no_fines = dict(FULLY_DECLARED_KWARGS)
    kwargs_no_fines["fines_declared"] = None
    facts = EarthquakeTriggeringReadinessFacts(
        cpt_evidence_facts=VERIFIED_CPT_FACTS, **kwargs_no_fines
    )
    result = assess_earthquake_triggering_readiness(facts)
    assert result.axis(contract.AXIS_FINES_OR_SOIL_APPLICABILITY).status == contract.NOT_READY

    kwargs_no_applicability = dict(FULLY_DECLARED_KWARGS)
    kwargs_no_applicability["soil_applicability_declared"] = SoilApplicabilityDeclaration(
        established=False
    )
    facts2 = EarthquakeTriggeringReadinessFacts(
        cpt_evidence_facts=VERIFIED_CPT_FACTS, **kwargs_no_applicability
    )
    result2 = assess_earthquake_triggering_readiness(facts2)
    assert result2.axis(contract.AXIS_FINES_OR_SOIL_APPLICABILITY).status == contract.NOT_READY


def test_no_scenario_blocks_both_magnitude_and_pga_axes():
    kwargs = dict(FULLY_DECLARED_KWARGS)
    kwargs["scenario_declared"] = None
    facts = EarthquakeTriggeringReadinessFacts(cpt_evidence_facts=VERIFIED_CPT_FACTS, **kwargs)
    result = assess_earthquake_triggering_readiness(facts)
    assert result.axis(contract.AXIS_EARTHQUAKE_MAGNITUDE).status == contract.NOT_READY
    assert result.axis(contract.AXIS_SEABED_PGA).status == contract.NOT_READY
    assert result.status == contract.NOT_READY


def test_static_shear_material_blocks_method_domain_axis():
    kwargs = dict(FULLY_DECLARED_KWARGS)
    kwargs["static_shear_material"] = True
    facts = EarthquakeTriggeringReadinessFacts(cpt_evidence_facts=VERIFIED_CPT_FACTS, **kwargs)
    result = assess_earthquake_triggering_readiness(facts)
    assert result.axis(contract.AXIS_METHOD_DOMAIN).status == contract.NOT_READY


def test_all_declared_axes_are_named_exactly_as_the_ticket_specifies():
    facts = EarthquakeTriggeringReadinessFacts(
        cpt_evidence_facts=VERIFIED_CPT_FACTS, **FULLY_DECLARED_KWARGS
    )
    result = assess_earthquake_triggering_readiness(facts)
    axis_names = {a.axis for a in result.axes}
    assert axis_names == set(contract.READINESS_AXES)


def test_wave_liquefaction_status_remains_a_separate_unimplemented_mechanism():
    status = wave_liquefaction_status()
    assert status["status"] == contract.NOT_EVALUABLE
    assert status["method_authority"] == "NOT_GRANTED_IN_MAR_033"
    assert status["axis"] == contract.AXIS_WAVE_LIQUEFACTION

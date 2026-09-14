"""Earthquake-triggering capability readiness (MAR-033 Section 40).

Extends the MAR-032 CPT-evidence foundation (`geotechnical.cpt_readiness`) rather than
duplicating it: `CPT_IDENTITY` and `DEPTH_REFERENCE` below read directly from the SAME
`geotechnical.cpt_readiness.CptEvidenceFacts` a caller already assembled -- nothing here
re-derives canonical-product identity or measured-evidence-role verification.

This module answers a DECLARATIVE question -- "given what has been explicitly declared, can the
earthquake-triggering capability even be attempted" -- before any row is computed. It never runs
the numeric engine (`earthquake_triggering.py`) itself, so it stays cheap even against a
138k-row CPT profile. A per-row numeric limitation that can only be known once the profile is
actually computed (e.g. CN iteration non-convergence at one specific depth) is reported by the
profile computation itself, in its own `limitations` column -- never invented here.

Mirrors the `geotechnical.cpt_readiness` axis pattern (`AxisAssessment` / overall status folded
from blocking vs. limitation reasons) rather than inventing a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marine_engine.geotechnical import cpt_contract
from marine_engine.geotechnical.cpt_readiness import CptEvidenceFacts
from marine_engine.liquefaction import contract
from marine_engine.liquefaction.earthquake_triggering import (
    FinesDeclaration,
    SoilApplicabilityDeclaration,
    TipResistanceDeclaration,
)
from marine_engine.liquefaction.scenario import EarthquakeScenario

__all__ = [
    "EarthquakeTriggeringReadinessFacts",
    "AxisAssessment",
    "EarthquakeTriggeringReadinessResult",
    "assess_earthquake_triggering_readiness",
]


@dataclass(frozen=True)
class EarthquakeTriggeringReadinessFacts:
    """Plain facts assembled by a caller (the orchestration execution layer). Defaults are the
    honest 'absent' state so a partially-declared scenario can never look more complete than it
    is."""

    cpt_evidence_facts: CptEvidenceFacts
    tip_resistance_declared: TipResistanceDeclaration | None = None
    stress_model_declared: bool = False
    stress_model_valid: bool = False
    stress_model_problem: str | None = None
    fines_declared: FinesDeclaration | None = None
    soil_applicability_declared: SoilApplicabilityDeclaration | None = None
    scenario_declared: EarthquakeScenario | None = None
    static_shear_material: bool = False


@dataclass(frozen=True)
class AxisAssessment:
    axis: str
    status: str
    blocking_reasons: tuple[str, ...] = ()
    limitation_reasons: tuple[str, ...] = ()
    facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "status": self.status,
            "blocking_reasons": list(self.blocking_reasons),
            "limitation_reasons": list(self.limitation_reasons),
            "facts": dict(self.facts),
        }


@dataclass(frozen=True)
class EarthquakeTriggeringReadinessResult:
    status: str
    axes: tuple[AxisAssessment, ...]

    def axis(self, name: str) -> AxisAssessment:
        return next(a for a in self.axes if a.axis == name)

    def reasons(self) -> list[str]:
        out: list[str] = []
        for a in self.axes:
            out.extend(f"{a.axis} ({contract.BLOCKING}): {r}" for r in a.blocking_reasons)
            out.extend(f"{a.axis} ({contract.LIMITATION}): {r}" for r in a.limitation_reasons)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "status_meaning": (
                "readiness of the EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING capability specifically "
                "(Section 40) -- not the broader CPT evidence readiness, which stays "
                "geotechnical.cpt_readiness's own, unmodified conclusion"
            ),
            "method_id": contract.METHOD_ID,
            "method_authority": contract.METHOD_AUTHORITY,
            "axes": [a.to_dict() for a in self.axes],
            "blocking_reasons": [f"{a.axis}: {r}" for a in self.axes for r in a.blocking_reasons],
            "limitation_reasons": [
                f"{a.axis}: {r}" for a in self.axes for r in a.limitation_reasons
            ],
        }


def _axis(
    axis: str, *, blocking: list[str], limitations: list[str], facts: dict[str, Any]
) -> AxisAssessment:
    if blocking:
        status = contract.NOT_READY
    elif limitations:
        status = contract.READY_WITH_LIMITATIONS
    else:
        status = contract.READY
    return AxisAssessment(
        axis=axis,
        status=status,
        blocking_reasons=tuple(blocking),
        limitation_reasons=tuple(limitations),
        facts=facts,
    )


_TIP_MODE_REQUIRED_CHANNELS: dict[str, tuple[str, ...]] = {
    contract.QT_MEASURED_OR_SOURCE_CORRECTED: (cpt_contract.QT_MPA,),
    contract.QC_PLUS_U2_AND_DECLARED_AREA_RATIO: (cpt_contract.QC_MPA, cpt_contract.U2_KPA),
    contract.QC_EXPLICITLY_DECLARED_AREA_CORRECTED: (cpt_contract.QC_MPA,),
}


def assess_earthquake_triggering_readiness(
    facts: EarthquakeTriggeringReadinessFacts,
) -> EarthquakeTriggeringReadinessResult:
    """Section 40: eight explicit axes -> one overall `EARTHQUAKE_TRIGGERING`-capability status.

    `WAVE_LIQUEFACTION` is reported separately (Section 3): earthquake and wave readiness never
    merge, and MAR-033 grants no wave-induced-liquefaction method authority at all.
    """

    axes: list[AxisAssessment] = []
    cpt = facts.cpt_evidence_facts

    # CPT_IDENTITY -- delegates entirely to the MAR-032/032A/032B measured-evidence gate.
    blocking: list[str] = []
    if not cpt.measured_cpt_profile_verified:
        blocking.append(
            "no verified MEASURED_CPT_CPTU_PROFILE evidence is available (canonical product "
            "identity and/or measured-evidence-role gate did not verify); see "
            "geotechnical.cpt_readiness for the detailed reason"
        )
    axes.append(
        _axis(
            contract.AXIS_CPT_IDENTITY,
            blocking=blocking,
            limitations=[],
            facts={"measured_cpt_profile_verified": cpt.measured_cpt_profile_verified},
        )
    )
    cpt_identity_ready = not blocking

    # CORRECTED_TIP_RESISTANCE
    blocking = []
    decl = facts.tip_resistance_declared
    if decl is None:
        blocking.append(
            f"{contract.CORRECTED_TIP_RESISTANCE_NOT_ESTABLISHED}: no tip-resistance correction "
            "mode declared"
        )
    elif cpt_identity_ready:
        required = _TIP_MODE_REQUIRED_CHANNELS[decl.mode]
        missing_channels = [c for c in required if c not in cpt.channels_present]
        if missing_channels:
            blocking.append(
                f"{contract.CORRECTED_TIP_RESISTANCE_NOT_ESTABLISHED}: declared mode {decl.mode!r} "
                f"requires channel(s) {missing_channels} not present in the verified CPT evidence"
            )
    axes.append(
        _axis(
            contract.AXIS_CORRECTED_TIP_RESISTANCE,
            blocking=blocking,
            limitations=[],
            facts={"declared_mode": decl.mode if decl is not None else None},
        )
    )

    # DEPTH_REFERENCE -- same check MAR-032 itself uses.
    blocking = []
    if cpt.depth_reference != cpt_contract.DEPTH_BELOW_SEABED or not cpt.depth_bsf_available:
        blocking.append("depth_bsf_m (depth below seabed) is not established for this evidence")
    axes.append(
        _axis(
            contract.AXIS_DEPTH_REFERENCE,
            blocking=blocking,
            limitations=[],
            facts={
                "depth_reference": cpt.depth_reference,
                "depth_bsf_available": cpt.depth_bsf_available,
            },
        )
    )

    # STRESS_PROFILE
    blocking = []
    if not facts.stress_model_declared:
        blocking.append(
            f"{contract.VERTICAL_STRESS_PROFILE_NOT_AVAILABLE}: no stress model declared"
        )
    elif not facts.stress_model_valid:
        blocking.append(
            f"{contract.VERTICAL_STRESS_PROFILE_NOT_AVAILABLE}: declared stress model is invalid"
            + (f" ({facts.stress_model_problem})" if facts.stress_model_problem else "")
        )
    axes.append(
        _axis(
            contract.AXIS_STRESS_PROFILE,
            blocking=blocking,
            limitations=[],
            facts={
                "stress_model_declared": facts.stress_model_declared,
                "stress_model_valid": facts.stress_model_valid,
            },
        )
    )

    # FINES_OR_SOIL_APPLICABILITY
    blocking = []
    if facts.fines_declared is None:
        blocking.append(
            f"{contract.MISSING_FINES_OR_APPLICABILITY}: no fines-content source declared"
        )
    if (
        facts.soil_applicability_declared is None
        or not facts.soil_applicability_declared.established
    ):
        blocking.append(
            f"{contract.COHESIONLESS_SOIL_APPLICABILITY_NOT_ESTABLISHED}: cohesionless-soil "
            "applicability is not established"
        )
    axes.append(
        _axis(
            contract.AXIS_FINES_OR_SOIL_APPLICABILITY,
            blocking=blocking,
            limitations=[],
            facts={
                "fines_source_declared": (
                    facts.fines_declared.source if facts.fines_declared is not None else None
                ),
                "soil_applicability_established": (
                    facts.soil_applicability_declared.established
                    if facts.soil_applicability_declared is not None
                    else False
                ),
            },
        )
    )

    # EARTHQUAKE_MAGNITUDE / SEABED_PGA -- Section 40 keeps them as two named axes even though,
    # in this implementation, both are populated (or both absent) by the same declared scenario.
    scenario = facts.scenario_declared
    blocking = (
        []
        if scenario is not None
        else [f"{contract.MISSING_EARTHQUAKE_SCENARIO}: moment_magnitude_mw not declared"]
    )
    axes.append(
        _axis(
            contract.AXIS_EARTHQUAKE_MAGNITUDE,
            blocking=blocking,
            limitations=[],
            facts={
                "moment_magnitude_mw": scenario.moment_magnitude_mw
                if scenario is not None
                else None
            },
        )
    )
    blocking = (
        []
        if scenario is not None
        else [f"{contract.MISSING_EARTHQUAKE_SCENARIO}: pga_g not declared"]
    )
    axes.append(
        _axis(
            contract.AXIS_SEABED_PGA,
            blocking=blocking,
            limitations=[],
            facts={
                "pga_g": scenario.pga_g if scenario is not None else None,
                "pga_reference": scenario.pga_reference if scenario is not None else None,
            },
        )
    )

    # METHOD_DOMAIN -- free-field / nearly-level applicability (Section 19).
    blocking = []
    if facts.static_shear_material:
        blocking.append(
            f"{contract.STATIC_SHEAR_OUTSIDE_MODEL_SCOPE}: sustained static shear is declared "
            "material; the free-field/nearly-level model does not cover this condition (no "
            "K-alpha correction in MAR-033)"
        )
    axes.append(
        _axis(
            contract.AXIS_METHOD_DOMAIN,
            blocking=blocking,
            limitations=[],
            facts={"static_shear_material": facts.static_shear_material},
        )
    )

    any_blocking = any(a.blocking_reasons for a in axes)
    any_limitation = any(a.limitation_reasons for a in axes)
    if any_blocking:
        overall = contract.NOT_READY
    elif any_limitation:
        overall = contract.READY_WITH_LIMITATIONS
    else:
        overall = contract.READY
    return EarthquakeTriggeringReadinessResult(status=overall, axes=tuple(axes))


def wave_liquefaction_status() -> dict[str, Any]:
    """Section 3/40: wave-induced liquefaction stays a completely separate, unimplemented
    mechanism -- MAR-033 grants it no method authority at all."""

    return {
        "axis": contract.AXIS_WAVE_LIQUEFACTION,
        "status": contract.NOT_EVALUABLE,
        "method_authority": "NOT_GRANTED_IN_MAR_033",
        "note": (
            "Wave-induced/residual-pore-pressure seabed liquefaction is a separate mechanism from "
            "earthquake-induced triggering and is not modelled by this ticket."
        ),
    }

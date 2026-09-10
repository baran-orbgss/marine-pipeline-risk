"""Explicit, non-numeric CPT evidence readiness and liquefaction-INPUT readiness (MAR-032
Sections 20-22, 35, 36).

`assess_cpt_readiness` answers: is the CPT EVIDENCE itself a usable canonical numeric profile?
`assess_liquefaction_input_readiness` answers, SEPARATELY for the earthquake-induced and the
wave/current-induced mechanisms: which evidence a future model would need is present, which is
missing, and whether any method authority exists (in MAR-032: none). Neither function computes
CSR, CRR, a factor of safety, a probability, LPI, settlement, lateral spreading or pore pressure.

A file merely existing never yields READY: `DIGITAL_PROFILE` is BLOCKING unless a canonical
profile with at least one row was actually created from machine-readable numeric source data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marine_engine.geotechnical import cpt_contract as contract

__all__ = [
    "CptEvidenceFacts",
    "AxisAssessment",
    "CptReadinessResult",
    "assess_cpt_readiness",
    "assess_liquefaction_input_readiness",
    "pl854_cpt_status",
]


@dataclass(frozen=True)
class CptEvidenceFacts:
    """Plain facts assembled by a caller (provider build or project adapter). Defaults are the
    honest 'absent' state so a partially-known asset can never look more complete than it is."""

    source_package_resolved: bool = False
    source_checksum_recorded: bool = False
    documentary_evidence_available: bool = False
    machine_readable_profile_available: bool = False
    canonical_profile_created: bool = False
    row_count: int = 0
    test_count: int = 0
    declared_test_count: int | None = None
    duplicate_observation_identity_count: int = 0
    contradictory_duplicate_depth_row_count: int = 0
    depth_order_violation_count: int = 0
    depth_reference: str = contract.DEPTH_REFERENCE_UNRESOLVED
    depth_bsf_available: bool = False
    channels_present: tuple[str, ...] = ()
    unit_unresolved_canonical_channels: tuple[str, ...] = ()
    unit_unresolved_raw_channels: tuple[str, ...] = ()
    coordinates_available: bool = False
    crs_resolved: bool = False
    crs_conflict: str | None = None
    cone_area_ratio_source_stated: bool = False
    # Section 21 -- earthquake-triggering evidence (all absent unless a caller proves otherwise)
    soil_unit_weight_or_stress_state_available: bool = False
    vertical_total_stress_basis_available: bool = False
    vertical_effective_stress_basis_available: bool = False
    earthquake_pga_available: bool = False
    earthquake_magnitude_available: bool = False
    stress_reduction_factor_method_authorized: bool = False
    # Section 22 -- wave-liquefaction evidence
    wave_forcing_available: bool = False
    water_depth_available: bool = False
    soil_hydraulic_properties_available: bool = False
    soil_compressibility_stiffness_available: bool = False
    initial_effective_stress_state_available: bool = False
    pore_pressure_response_parameters_available: bool = False
    wave_liquefaction_model_authorized: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


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
class CptReadinessResult:
    cpt_profile_status: str
    axes: tuple[AxisAssessment, ...]

    @property
    def status(self) -> str:
        """Alias used by the project registry: the CPT EVIDENCE readiness (never liquefaction)."""
        return self.cpt_profile_status

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
            "status": self.cpt_profile_status,
            "status_meaning": (
                "readiness of the CPT/CPTU EVIDENCE as a canonical numeric profile; NOT a "
                "liquefaction susceptibility, hazard or risk statement"
            ),
            "axes": [a.to_dict() for a in self.axes],
            "blocking_reasons": [f"{a.axis}: {r}" for a in self.axes for r in a.blocking_reasons],
            "limitation_reasons": [
                f"{a.axis}: {r}" for a in self.axes for r in a.limitation_reasons
            ],
        }


def _axis(
    axis: str,
    *,
    blocking: list[str],
    limitations: list[str],
    facts: dict[str, Any],
    status_override: str | None = None,
) -> AxisAssessment:
    if status_override is not None:
        status = status_override
    elif blocking:
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


def assess_cpt_readiness(facts: CptEvidenceFacts) -> CptReadinessResult:
    """Seven CPT-evidence axes -> explicit status each -> one overall CPT-evidence status."""

    axes: list[AxisAssessment] = []

    # SOURCE_PACKAGE
    blocking: list[str] = []
    limitations: list[str] = []
    if not facts.source_package_resolved:
        blocking.append("official source package not resolved")
    if not facts.source_checksum_recorded:
        blocking.append("acquired package checksum (SHA-256) not recorded")
    axes.append(
        _axis(
            contract.SOURCE_PACKAGE,
            blocking=blocking,
            limitations=limitations,
            facts={
                "source_package_resolved": facts.source_package_resolved,
                "source_checksum_recorded": facts.source_checksum_recorded,
            },
            status_override=(contract.NOT_AVAILABLE if not facts.source_package_resolved else None),
        )
    )

    # DIGITAL_PROFILE -- the gate that makes 'file exists' insufficient.
    blocking, limitations = [], []
    override: str | None = None
    if not facts.machine_readable_profile_available:
        if facts.documentary_evidence_available:
            blocking.append(
                "DOCUMENTARY_CPT_EVIDENCE_ONLY: PDF/image logs are not digitized (no OCR, no "
                "chart digitization); DIGITAL_NUMERIC_CPT_PROFILE_READY = false"
            )
        else:
            blocking.append("no machine-readable numeric CPT/CPTU source identified")
            override = contract.NOT_AVAILABLE
    elif not facts.canonical_profile_created or facts.row_count <= 0:
        blocking.append("machine-readable source present but no canonical profile row was created")
    axes.append(
        _axis(
            contract.DIGITAL_PROFILE,
            blocking=blocking,
            limitations=limitations,
            facts={
                "documentary_cpt_evidence_available": facts.documentary_evidence_available,
                "machine_readable_cpt_profile_available": facts.machine_readable_profile_available,
                "canonical_profile_created": facts.canonical_profile_created,
                "row_count": facts.row_count,
                "test_count": facts.test_count,
            },
            status_override=override,
        )
    )
    digital_ready = not blocking

    # IDENTITY
    blocking, limitations = [], []
    if facts.duplicate_observation_identity_count > 0:
        blocking.append(
            f"{facts.duplicate_observation_identity_count} rows share a duplicate observation "
            "identity (source_id, test_id, observation_index)"
        )
    if facts.contradictory_duplicate_depth_row_count > 0:
        # Distinct observation identities that share one depth value but carry different
        # readings (e.g. consecutive scans before the depth sensor moved). They are preserved as
        # separate observations -- never averaged or collapsed -- and reported as a limitation;
        # only a duplicated observation IDENTITY (above) is blocking.
        limitations.append(
            f"{facts.contradictory_duplicate_depth_row_count} rows share a depth value with "
            "another observation of the same test but carry different readings (kept as "
            "distinct observations; not averaged, not collapsed)"
        )
    if facts.declared_test_count is not None and facts.declared_test_count != facts.test_count:
        limitations.append(
            f"source-declared test count ({facts.declared_test_count}) differs from parsed "
            f"machine-readable test count ({facts.test_count}); the declaration is recorded, not "
            "enforced"
        )
    if facts.depth_order_violation_count > 0:
        limitations.append(
            f"{facts.depth_order_violation_count} depth-order violations reported (rows kept in "
            "source order; nothing re-sorted or resampled)"
        )
    axes.append(
        _axis(
            contract.IDENTITY,
            blocking=blocking,
            limitations=limitations,
            facts={
                "test_count": facts.test_count,
                "declared_test_count": facts.declared_test_count,
                "duplicate_observation_identity_count": facts.duplicate_observation_identity_count,
                "contradictory_duplicate_depth_row_count": (
                    facts.contradictory_duplicate_depth_row_count
                ),
            },
            status_override=(contract.NOT_EVALUABLE if not digital_ready else None),
        )
    )

    # DEPTH_REFERENCE
    blocking, limitations = [], []
    if facts.depth_reference != contract.DEPTH_BELOW_SEABED:
        blocking.append(
            f"{contract.DEPTH_REFERENCE_UNRESOLVED}: depth column not established as depth below "
            "seabed by source metadata; depth_bsf_m = null"
        )
    elif not facts.depth_bsf_available:
        blocking.append("depth reference declared but no finite depth_bsf_m value exists")
    axes.append(
        _axis(
            contract.DEPTH_REFERENCE,
            blocking=blocking,
            limitations=limitations,
            facts={
                "depth_reference": facts.depth_reference,
                "depth_bsf_available": facts.depth_bsf_available,
            },
            status_override=(contract.NOT_EVALUABLE if not digital_ready else None),
        )
    )

    # MEASUREMENT_SEMANTICS
    blocking, limitations = [], []
    present = set(facts.channels_present)
    if contract.QC_MPA not in present and contract.QT_MPA not in present:
        blocking.append("neither measured (qc) nor corrected (qt) cone resistance is available")
    if contract.QT_MPA not in present:
        limitations.append(
            "qt (corrected cone resistance) not provided by source; qt_mpa = null -- NOT derived "
            "from qc (no unequal-area / pore-pressure correction in MAR-032)"
        )
    if contract.FS_KPA not in present:
        limitations.append("sleeve friction fs not available")
    if contract.U2_KPA not in present:
        limitations.append("pore pressure u2 not available")
    axes.append(
        _axis(
            contract.MEASUREMENT_SEMANTICS,
            blocking=blocking,
            limitations=limitations,
            facts={"channels_present": list(facts.channels_present)},
            status_override=(contract.NOT_EVALUABLE if not digital_ready else None),
        )
    )

    # UNITS
    blocking, limitations = [], []
    if facts.unit_unresolved_canonical_channels:
        blocking.append(
            "canonical channel(s) with unknown/unconvertible source unit left null: "
            + ", ".join(facts.unit_unresolved_canonical_channels)
        )
    if facts.unit_unresolved_raw_channels:
        limitations.append(
            "raw-only channel(s) with no stated source unit preserved unnormalized: "
            + ", ".join(facts.unit_unresolved_raw_channels)
        )
    axes.append(
        _axis(
            contract.UNITS,
            blocking=blocking,
            limitations=limitations,
            facts={
                "unit_unresolved_canonical_channels": list(
                    facts.unit_unresolved_canonical_channels
                ),
                "unit_unresolved_raw_channels": list(facts.unit_unresolved_raw_channels),
            },
            status_override=(contract.NOT_EVALUABLE if not digital_ready else None),
        )
    )

    # SPATIAL_REFERENCE
    blocking, limitations = [], []
    override = None
    if facts.crs_conflict:
        blocking.append(f"declared vs source CRS conflict: {facts.crs_conflict}")
    if not facts.coordinates_available:
        limitations.append(f"{contract.SPATIAL_LOCATION_UNRESOLVED}: no test coordinates")
        override = contract.NOT_AVAILABLE
    elif not facts.crs_resolved:
        limitations.append(
            f"{contract.CRS_UNRESOLVED}: coordinates present but no CRS established (never "
            "guessed, no UTM-by-location inference)"
        )
    axes.append(
        _axis(
            contract.SPATIAL_REFERENCE,
            blocking=blocking,
            limitations=limitations,
            facts={
                "coordinates_available": facts.coordinates_available,
                "crs_resolved": facts.crs_resolved,
                "crs_conflict": facts.crs_conflict,
            },
            status_override=override if not blocking else None,
        )
    )

    # Overall CPT-evidence status: any BLOCKING -> NOT_READY; else any limitation / NOT_AVAILABLE
    # spatial -> READY_WITH_LIMITATIONS; else READY.
    any_blocking = any(a.blocking_reasons for a in axes)
    any_limitation = any(a.limitation_reasons for a in axes) or any(
        a.status == contract.NOT_AVAILABLE for a in axes if a.axis == contract.SPATIAL_REFERENCE
    )
    if any_blocking:
        overall = contract.NOT_READY
    elif any_limitation:
        overall = contract.READY_WITH_LIMITATIONS
    else:
        overall = contract.READY
    return CptReadinessResult(cpt_profile_status=overall, axes=tuple(axes))


def _availability(flag: bool) -> str:
    return contract.AVAILABLE if flag else contract.NOT_AVAILABLE


def assess_liquefaction_input_readiness(facts: CptEvidenceFacts) -> dict[str, Any]:
    """Section 21/22: two SEPARATE mechanism blocks. Each is `NOT_EVALUABLE` in MAR-032 because no
    triggering / hydro-geotechnical method authority exists, independent of how much evidence is
    present. Evidence availability is itemized so the next ticket can see exactly what is
    missing."""

    present = set(facts.channels_present)
    earthquake_items = {
        "machine_readable_cpt_profile": _availability(
            facts.machine_readable_profile_available and facts.canonical_profile_created
        ),
        "depth_below_seabed": _availability(
            facts.depth_reference == contract.DEPTH_BELOW_SEABED and facts.depth_bsf_available
        ),
        "qc_or_qt": _availability(bool({contract.QC_MPA, contract.QT_MPA} & present)),
        "qc": _availability(contract.QC_MPA in present),
        "qt": _availability(contract.QT_MPA in present),
        "sleeve_friction_fs": _availability(contract.FS_KPA in present),
        "pore_pressure_u2": _availability(contract.U2_KPA in present),
        "cone_area_ratio": (
            "AVAILABLE_SOURCE_STATED_NOT_APPLIED"
            if facts.cone_area_ratio_source_stated
            else contract.NOT_AVAILABLE
        ),
        "soil_unit_weight_or_stress_state": _availability(
            facts.soil_unit_weight_or_stress_state_available
        ),
        "vertical_total_stress_basis": _availability(facts.vertical_total_stress_basis_available),
        "vertical_effective_stress_basis": _availability(
            facts.vertical_effective_stress_basis_available
        ),
        "earthquake_pga_a_max": _availability(facts.earthquake_pga_available),
        "earthquake_magnitude": _availability(facts.earthquake_magnitude_available),
        "stress_reduction_factor_method": (
            contract.AVAILABLE
            if facts.stress_reduction_factor_method_authorized
            else contract.NOT_AUTHORIZED
        ),
    }
    earthquake_missing = [
        k
        for k in contract.EARTHQUAKE_REQUIRED_EVIDENCE
        if earthquake_items[k] in (contract.NOT_AVAILABLE, contract.NOT_AUTHORIZED)
    ]

    wave_items = {
        "wave_forcing": _availability(facts.wave_forcing_available),
        "water_depth": _availability(facts.water_depth_available),
        "soil_profile": _availability(
            facts.machine_readable_profile_available and facts.canonical_profile_created
        ),
        "soil_hydraulic_properties": _availability(facts.soil_hydraulic_properties_available),
        "soil_compressibility_stiffness_properties": _availability(
            facts.soil_compressibility_stiffness_available
        ),
        "initial_effective_stress_state": _availability(
            facts.initial_effective_stress_state_available
        ),
        "pore_pressure_response_parameters": _availability(
            facts.pore_pressure_response_parameters_available
        ),
    }
    wave_missing = [
        k for k in contract.WAVE_REQUIRED_EVIDENCE if wave_items[k] != contract.AVAILABLE
    ]

    return {
        "role": contract.LIQUEFACTION_INPUT_READINESS_ASSESSMENT,
        "status": contract.NOT_EVALUABLE,
        "mechanisms_kept_separate": True,
        "earthquake_induced": {
            "mechanism": contract.EARTHQUAKE_INDUCED_LIQUEFACTION,
            "status": contract.NOT_EVALUABLE,
            "status_code": contract.EARTHQUAKE_LIQUEFACTION_TRIGGERING_NOT_EVALUABLE,
            "method_authority": "NOT_GRANTED_IN_MAR_032",
            "required_evidence": earthquake_items,
            "missing_or_unauthorized": earthquake_missing,
            "future_method_context": [
                "youd_2001",
                "boulanger_idriss_2014",
                "robertson_wride_1998",
            ],
            "note": (
                "A future simplified framework separates seismic demand (CSR from a_max, "
                "sigma_v0, sigma'_v0, r_d) from penetration-based resistance (CRR). Neither is "
                "computed here; the dependency list only defines what evidence would be required."
            ),
        },
        "wave_current_induced": {
            "mechanism": contract.WAVE_CURRENT_INDUCED_SEABED_LIQUEFACTION,
            "status": contract.NOT_EVALUABLE,
            "status_code": contract.WAVE_INDUCED_LIQUEFACTION_NOT_EVALUABLE,
            "WAVE_INDUCED_LIQUEFACTION_MODELLED": False,
            "model_authority": (
                "GRANTED" if facts.wave_liquefaction_model_authorized else "NOT_GRANTED_IN_MAR_032"
            ),
            "required_evidence": wave_items,
            "missing": wave_missing,
            "future_method_context": ["jeng_2001"],
            "note": (
                "Wave-induced seabed liquefaction involves oscillatory/momentary and residual "
                "pore-pressure response of a porous seabed. Accepted MAR-009/011/012 wave and "
                "current forcing does NOT make it evaluable; no constitutive model is frozen here."
            ),
        },
        "not_computed": dict(contract.NOT_COMPUTED_FLAGS),
        "surface_sediment_boundary": contract.SURFACE_SEDIMENT_BOUNDARY_STATEMENT,
    }


def pl854_cpt_status() -> dict[str, Any]:
    """Section 35: PL854 has no canonical subsurface CPT/SPT profile. MAR-008 surface sediment
    evidence does not change this and no route-wide liquefaction classification exists."""

    return {
        "study": "PL854",
        "CPT_GEOTECHNICAL_PROFILE": contract.NOT_AVAILABLE,
        "EARTHQUAKE_LIQUEFACTION_TRIGGERING": contract.NOT_EVALUABLE,
        "WAVE_INDUCED_LIQUEFACTION": contract.NOT_EVALUABLE,
        "surface_sediment_substituted": False,
        "route_wide_liquefaction_classification": None,
        "note": (
            "No canonical subsurface CPT/SPT profile exists for PL854; MAR-008 surface sediment "
            "evidence is context only and does not change this."
        ),
    }

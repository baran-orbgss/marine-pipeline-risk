"""Structural free-span assessment handoff contract (MAR-025 Section 29).

Documents what a FUTURE structural/VIV/fatigue module would additionally need. Nothing in this
module implements, approximates, or encodes any part of DNV-RP-F105 or any other structural
free-span standard -- MAR-025 performs no structural assessment (Section 2).
"""

from __future__ import annotations

from typing import Any

STRUCTURAL_FREE_SPAN_ASSESSMENT_NOT_PERFORMED = "STRUCTURAL_FREE_SPAN_ASSESSMENT_NOT_PERFORMED"
GEOMETRIC_SUPPORT_CONDITION_AND_SUPPORT_LOSS_SCREENING = (
    "GEOMETRIC_SUPPORT_CONDITION_AND_SUPPORT_LOSS_SCREENING"
)

REQUIRED_STRUCTURAL_INPUTS: tuple[dict[str, str], ...] = (
    {"field": "pipe_outer_diameter_and_wall_thickness", "description": "Pipe OD / wall thickness."},
    {
        "field": "material_properties",
        "description": "Steel grade, elastic modulus, density, and other material properties.",
    },
    {
        "field": "contents_and_effective_mass",
        "description": "Pipeline contents and the effective (structural + added + contents) "
        "mass per unit length.",
    },
    {"field": "coating", "description": "Concrete/anti-corrosion coating thickness and density."},
    {
        "field": "axial_force_or_tension",
        "description": "Effective axial force / residual lay tension.",
    },
    {
        "field": "boundary_conditions_and_soil_support",
        "description": "End-span boundary conditions and soil (axial/lateral/vertical) "
        "support stiffness.",
    },
    {
        "field": "free_span_length",
        "description": "The measured free-span length (this MAR-025 POC's own output can "
        "seed this).",
    },
    {
        "field": "gap_or_height",
        "description": "The measured gap/height beneath the pipe (this MAR-025 POC's own "
        "output can seed this).",
    },
    {"field": "water_depth", "description": "Local water depth."},
    {"field": "current", "description": "Near-bed current speed/direction statistics."},
    {
        "field": "waves",
        "description": "Wave climate (significant height, period, directionality) at the span "
        "location.",
    },
    {"field": "damping", "description": "Structural and hydrodynamic damping ratios."},
    {
        "field": "waves_and_current_combined_flow_model",
        "description": "A combined flow model appropriate to the applicable standard.",
    },
    {
        "field": "structural_model",
        "description": "The beam/finite-element structural idealization of the span (e.g. "
        "pinned-pinned, fixed-fixed, or a full FE model).",
    },
    {
        "field": "applicable_licensed_standard_and_version",
        "description": "The specific structural free-span standard and version to be applied "
        "(e.g. a licensed copy of DNV-RP-F105) -- not available in this repository.",
    },
)


def build_structural_free_span_assessment_handoff_contract() -> dict[str, Any]:
    return {
        "scientific_role": "STRUCTURAL_FREE_SPAN_ASSESSMENT_HANDOFF_CONTRACT",
        "mar025_result": GEOMETRIC_SUPPORT_CONDITION_AND_SUPPORT_LOSS_SCREENING,
        "structural_assessment_status": STRUCTURAL_FREE_SPAN_ASSESSMENT_NOT_PERFORMED,
        "explicit_statement": "NOT IMPLEMENTED IN MAR-025",
        "purpose": "Records what a future structural/VIV/fatigue-life module would "
        "additionally require beyond MAR-025's geometric outputs. This is documentation of a "
        "future requirement, not an implementation -- no DNV-RP-F105 (or any other structural "
        "free-span standard) equations are encoded anywhere in this repository, since no "
        "licensed copy of such a standard is available here.",
        "required_structural_inputs": list(REQUIRED_STRUCTURAL_INPUTS),
        "mar025_outputs_available_as_seed_data": [
            "measured free-span interval geometry (span_id, start/end chainage, span length, "
            "maximum/median clearance)",
            "measured support state per sample",
            "support-loss susceptibility screening results for a tested seabed-lowering scenario",
        ],
        "notes": [
            "MAR-025 identifies / screens geometric pipe support loss. It does not assess VIV, "
            "fatigue life, ULS/FLS acceptability, or pipeline failure probability.",
        ],
    }

"""Generic pipe/seabed free-span input contract (MAR-025 Section 16).

Zero dependency on any specific project or dataset. Documentation only, not a validator --
mirrors the established `contract.py` shape used throughout this project (see
`marine_engine.burial.contract`).
"""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS_MEASURED_GEOMETRY: tuple[dict[str, str], ...] = (
    {
        "field": "route_or_chainage",
        "description": "The real authoritative pipeline route/chainage, never an invented "
        "centreline.",
    },
    {
        "field": "pipe_vertical_profile",
        "description": "Real, source-measured pipe elevation values along the route -- never "
        "assumed or interpolated over unsurveyed gaps.",
    },
    {
        "field": "pipe_vertical_reference",
        "description": "What the pipe elevation number means (PIPE_CENTRELINE_ELEVATION, "
        "PIPE_BOTTOM_ELEVATION, PIPE_TOP_ELEVATION, or OTHER_EXPLICIT_PIPE_REFERENCE). Never "
        "inferred.",
    },
    {
        "field": "pipe_geometry_or_reference_offset",
        "description": "Required unless the pipe vertical reference is already "
        "PIPE_BOTTOM_ELEVATION: the explicit `reference_to_pipe_bottom_offset_m` (e.g. D/2 for "
        "a circular pipe centreline), or the pipe geometry (outer diameter) needed to derive "
        "it. Never assumed circular or of any particular size.",
    },
    {
        "field": "seabed_support_elevation_profile",
        "description": "Real seabed elevation representative of vertical support beneath the "
        "pipe, in the project's one accepted canonical elevation convention (higher = "
        "shallower) -- never a raw positive-down depth subtracted without conversion.",
    },
    {
        "field": "crs",
        "description": "A defined, projected, metric horizontal CRS for every measurement.",
    },
    {
        "field": "survey_epoch",
        "description": "The acquisition date or date range of the survey.",
    },
    {
        "field": "sample_support_or_spacing",
        "description": "The along-route sample spacing/support of both the pipe and seabed "
        "profiles, needed to derive a defensible maximum measurement-gap threshold for "
        "interval continuity (Section 10).",
    },
    {
        "field": "nodata_or_missing_semantics",
        "description": "How missing/nodata values are represented in the source, so a missing "
        "measurement is never silently treated as zero clearance.",
    },
)

REQUIRED_FIELDS_CATEGORICAL_FREE_SPAN: tuple[dict[str, str], ...] = (
    {
        "field": "clearance_classification_threshold",
        "description": "A defensible clearance-significance threshold (a combined 1-sigma "
        "vertical uncertainty, or a stated operator/source QC threshold). Without one, only "
        "continuous clearance is retained -- "
        "FREE_SPAN_CLASSIFICATION_NOT_DEFENSIBLE_NO_CLEARANCE_THRESHOLD, never an invented "
        "0.05 m / 0.1 m default.",
    },
)

REQUIRED_FIELDS_SUPPORT_LOSS_SUSCEPTIBILITY: tuple[dict[str, str], ...] = (
    {
        "field": "seabed_lowering_input",
        "description": "A defensible seabed-lowering magnitude for the tested scenario.",
    },
    {
        "field": "lowering_evidence_type",
        "description": "One of MEASURED_REPEAT_MBES_LOWERING, ENGINEERING_SCENARIO, or "
        "EXTERNALLY_SUPPLIED_MORPHODYNAMIC_SCENARIO. Never derived from scour-onset screening, "
        "sediment mobility, sand-wave asymmetry, or a single bathymetry epoch.",
    },
    {
        "field": "lowering_epoch_or_scenario_semantics",
        "description": "Whether the lowering input is an observed multi-epoch measurement or a "
        "labelled engineering/operator scenario, and its own epoch/provenance.",
    },
)

STRONGLY_PREFERRED_FIELDS: tuple[dict[str, str], ...] = (
    {
        "field": "vertical_uncertainty",
        "description": "Source-stated 1-sigma vertical uncertainty for the pipe and seabed "
        "profiles separately, enabling a combined statistical clearance threshold.",
    },
    {
        "field": "source_interpreted_free_spans",
        "description": "Operator-supplied evidence explicitly identifying free-span sections -- "
        "preserved as SOURCE_INTERPRETED_FREE_SPAN, never derived from clearance alone.",
    },
    {
        "field": "high_resolution_mbes",
        "description": "High-resolution multibeam bathymetry for spatial context and, where "
        "genuinely applicable (repeat surveys), deriving a defensible seabed-lowering input.",
    },
    {
        "field": "sss_profiler_or_rov_condition_evidence",
        "description": "Side-scan sonar, sub-bottom profiler, or ROV condition evidence for "
        "independent corroboration of measured support state.",
    },
)


def build_free_span_input_contract() -> dict[str, Any]:
    return {
        "scientific_role": "GENERIC_PIPELINE_FREE_SPAN_SUPPORT_LOSS_INPUT_CONTRACT",
        "purpose": "Defines what the future OrbGSS Marine Module requires from an operator "
        "before a real measured pipe/seabed profile can be turned into current free-span "
        "geometry, and separately what is additionally required before any support-loss "
        "susceptibility screening can be produced. This contract does NOT cover structural "
        "free-span integrity assessment (DNV-RP-F105 / VIV / fatigue / ULS / FLS) -- see "
        "`structural_handoff.py` for that separate, unimplemented boundary.",
        "required_fields_measured_geometry": list(REQUIRED_FIELDS_MEASURED_GEOMETRY),
        "required_fields_categorical_free_span": list(REQUIRED_FIELDS_CATEGORICAL_FREE_SPAN),
        "required_fields_support_loss_susceptibility": list(
            REQUIRED_FIELDS_SUPPORT_LOSS_SUSCEPTIBILITY
        ),
        "strongly_preferred_fields": list(STRONGLY_PREFERRED_FIELDS),
        "notes": [
            "This contract is documentation of requirements, not an implemented validator.",
            "Measured/observed free-span geometry, support-loss susceptibility screening, and "
            "source-reported/observed free-span evidence are three distinct concepts -- "
            "measured geometry can be produced from real source data alone; susceptibility "
            "screening requires a separately-defensible seabed-lowering input and is never "
            "inferred from a single survey epoch; source-reported evidence is never treated as "
            "validation of either.",
            "No VIV fatigue life, ULS/FLS utilization, allowable span length, structural "
            "acceptance, future failure probability, or 0-100 risk score is ever produced by "
            "this contract's consumers.",
        ],
    }

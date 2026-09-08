"""Generic burial/exposure input contract (MAR-024 Section 20).

Zero dependency on any specific project or dataset. Documentation only,
not a validator -- mirrors the established `contract.py` shape used
throughout this project.
"""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS_CURRENT_STATE: tuple[dict[str, str], ...] = (
    {
        "field": "authoritative_linear_asset_route",
        "description": "The real source route/route-position geometry for the asset -- never "
        "an invented centreline.",
    },
    {
        "field": "chainage_or_kp",
        "description": "A chainage/KP reference for each measurement, preserved exactly as "
        "the source states it alongside a route-derived chainage.",
    },
    {
        "field": "measured_burial_values",
        "description": "Real, source-measured burial/depth values -- never assumed or "
        "interpolated over unsurveyed gaps.",
    },
    {
        "field": "burial_measurement_reference",
        "description": "What the source's own burial/depth number actually means (top of "
        "asset, centreline, or another source-specific reference). Without this resolved, "
        "current burial state is reported as MEASURED_REFERENCE_REQUIRES_REVIEW, never "
        "silently assumed.",
    },
    {
        "field": "source_sign_convention",
        "description": "What a positive vs. negative raw value means -- resolved to exactly "
        "one of POSITIVE_VALUE_MEANS_DEEPER_BURIAL, NEGATIVE_VALUE_MEANS_DEEPER_BURIAL, or "
        "explicitly SIGN_CONVENTION_UNRESOLVED (MAR-024A). A numeric 'depth of burial' column "
        "is NOT analysis-ready until both its sign and its reference point are resolved -- an "
        "unresolved sign never bypasses normalization to drive classification.",
    },
    {
        "field": "reference_to_asset_top_offset",
        "description": "Required only when the burial-measurement reference point is "
        "CENTRELINE_BURIAL or OTHER_SOURCE_SPECIFIC_REFERENCE: the explicit, never-assumed "
        "offset converting the source reference depth into cover above the top of the asset. "
        "May be derived from operator-supplied asset geometry (e.g. diameter / 2 for a "
        "circular cross-section) but is never itself assumed by the generic engine. "
        "TOP_OF_ASSET_BURIAL requires no offset.",
    },
    {
        "field": "asset_geometry",
        "description": "Operator-supplied asset cross-section geometry (e.g. diameter), "
        "needed only to derive `reference_to_asset_top_offset` for a CENTRELINE_BURIAL "
        "reference -- never assumed to be circular or of any particular size by the generic "
        "engine.",
    },
    {
        "field": "units",
        "description": "The measurement units for burial values and chainage/KP -- checked "
        "for consistency across every source file used, never assumed equal.",
    },
    {
        "field": "survey_epoch",
        "description": "The acquisition date or date range of the survey.",
    },
    {
        "field": "crs_and_spatial_association",
        "description": "A defined, projected, metric horizontal CRS and real (x, y) "
        "coordinates for each measurement.",
    },
)

REQUIRED_FIELDS_FUTURE_EXPOSURE_SCREENING: tuple[dict[str, str], ...] = (
    {
        "field": "defensible_seabed_change_or_lowering_input",
        "description": "EITHER a defensible observed multi-epoch seabed-lowering product "
        "(OBSERVED_MULTI_EPOCH_SEABED_LOWERING) OR an explicitly-labelled operator scenario "
        "(OPERATOR_DEFINED_LOWERING_SCENARIO). Without one of these, no future exposure "
        "screening result is produced -- current burial/exposure state alone never implies "
        "future susceptibility.",
    },
)

STRONGLY_PREFERRED_FIELDS: tuple[dict[str, str], ...] = (
    {
        "field": "survey_uncertainty",
        "description": "Source-stated measurement uncertainty per record.",
    },
    {
        "field": "high_resolution_mbes",
        "description": "High-resolution multibeam bathymetry for spatial context and, where "
        "genuinely applicable, deriving a defensible seabed-lowering input.",
    },
    {
        "field": "interpreted_exposure_features",
        "description": "Operator-supplied GIS/tabular interpretation explicitly identifying "
        "exposed sections -- preserved as SOURCE_INTERPRETED_EXPOSURE_EVIDENCE, never derived "
        "from an unclear numeric convention.",
    },
    {
        "field": "sediment_or_morphology_context",
        "description": "Sediment and seabed-morphology context for the route corridor.",
    },
)


def build_burial_exposure_input_contract() -> dict[str, Any]:
    return {
        "scientific_role": "GENERIC_LINEAR_ASSET_BURIAL_EXPOSURE_STATE_INPUT_CONTRACT",
        "purpose": "Defines what the future OrbGSS Marine Module requires from an operator "
        "before a real measured depth-of-burial profile can be turned into a current "
        "burial/exposure state, and separately what is additionally required before any "
        "future exposure-susceptibility screening can be produced.",
        "required_fields_current_burial_state": list(REQUIRED_FIELDS_CURRENT_STATE),
        "required_fields_future_exposure_screening": list(
            REQUIRED_FIELDS_FUTURE_EXPOSURE_SCREENING
        ),
        "strongly_preferred_fields": list(STRONGLY_PREFERRED_FIELDS),
        "notes": [
            "This contract is documentation of requirements, not an implemented validator.",
            "Observed/measured burial state, source-interpreted exposure evidence, and future "
            "exposure susceptibility are three distinct concepts -- current burial state can "
            "be produced from real source data alone; future exposure susceptibility requires "
            "a separately-defensible seabed-lowering input and is never inferred from a single "
            "survey epoch.",
            "A numeric 'depth of burial' column is NOT analysis-ready until both its sign "
            "convention and its measurement reference point are explicitly resolved (MAR-024A) "
            "-- an unresolved sign convention or reference point yields no canonical reference "
            "burial depth and no cover-above-asset value, never a guessed classification.",
        ],
    }

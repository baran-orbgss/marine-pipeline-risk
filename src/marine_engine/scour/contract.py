"""Generic pipeline scour susceptibility input contract (MAR-023 Section 14).

Zero dependency on any specific project or dataset. Documentation only, not
a validator -- mirrors `marine_engine.bedforms.contract`'s established shape.
"""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS: tuple[dict[str, str], ...] = (
    {
        "field": "route_or_pipeline_sections",
        "description": "The pipeline route geometry, divided into sections with a stable "
        "section identifier and chainage range.",
    },
    {
        "field": "pipe_diameter_m",
        "description": "The pipeline's outer diameter.",
    },
    {
        "field": "actual_embedment_profile",
        "description": "Operator-supplied, route-specific, MEASURED actual embedment -- never "
        "a typical/assumed value. Without this, no site-specific scour susceptibility result "
        "can be produced (Section 6): NO ACTUAL EMBEDMENT PROFILE = NO SITE-SPECIFIC SCOUR "
        "SUSCEPTIBILITY RESULT.",
    },
    {
        "field": "sediment_grain_size_property",
        "description": "The grain-size property the accepted Marini et al. (2024) onset "
        "method requires (D50).",
    },
    {
        "field": "accepted_near_bed_current_forcing",
        "description": "Near-bed current speed/direction time series, reconciled to the route "
        "the same way MAR-009B/MAR-012 already established.",
    },
    {
        "field": "accepted_wave_forcing",
        "description": "Wave orbital velocity amplitude, representative period, and direction "
        "time series, reconciled the same way MAR-011A/MAR-012 already established.",
    },
    {
        "field": "horizontal_crs",
        "description": "A defined, projected, metric horizontal coordinate reference system.",
    },
    {
        "field": "forcing_epoch_provenance",
        "description": "The acquisition window and source of the current/wave forcing record.",
    },
)

STRONGLY_PREFERRED_FIELDS: tuple[dict[str, str], ...] = (
    {
        "field": "high_resolution_bathymetry",
        "description": "High-resolution analytical bathymetry for spatial context and, where "
        "a second epoch exists, observed seabed change context around the route.",
    },
    {
        "field": "observed_scour_interpretation",
        "description": "Operator-supplied GIS interpretation of observed scour/exposure "
        "evidence near the asset -- kept as an independent evidence layer, never used to tune "
        "the pipeline scour-onset physics (Section 12).",
    },
    {
        "field": "route_specific_sediment_samples",
        "description": "Measured sediment samples along the route, rather than the fixed "
        "Marini/Zang calibration-envelope D50 scenarios used for screening.",
    },
    {
        "field": "observed_near_bed_metocean",
        "description": "Measured (rather than model-derived) near-bed current/wave observations "
        "for validation of the forcing record.",
    },
    {
        "field": "protection_or_rock_dump_geometry",
        "description": "As-built rock dump or other protection geometry, which changes the "
        "effective embedment/exposure state independently of natural burial.",
    },
)


def build_scour_susceptibility_input_contract() -> dict[str, Any]:
    return {
        "scientific_role": "GENERIC_PIPELINE_SCOUR_ONSET_SUSCEPTIBILITY_SCREENING_INPUT_CONTRACT",
        "purpose": "Defines what the future OrbGSS Marine Module requires from an operator "
        "before pipeline scour-onset susceptibility can be screened against the actual route, "
        "and what it requires to separately ingest and map real observed scour evidence.",
        "required_fields": list(REQUIRED_FIELDS),
        "strongly_preferred_fields": list(STRONGLY_PREFERRED_FIELDS),
        "notes": [
            "This contract is documentation of requirements, not an implemented validator.",
            "NO ACTUAL EMBEDMENT PROFILE = NO SITE-SPECIFIC SCOUR SUSCEPTIBILITY RESULT. "
            "Scenario analysis using tested embedment ratios may still be performed without one.",
            "Observed scour evidence ingestion (Section 9-13) is a separate, independent "
            "workflow from pipeline scour-onset physics screening and never validates it unless "
            "the asset physics and required engineering inputs genuinely match.",
        ],
    }

"""Generic multi-epoch seabed-change input contract (MAR-021 Section 24).

Zero dependency on any specific project or dataset. Documentation only,
not a validator (see `epoch_compatibility`/`common_support` for the actual
per-run checks).
"""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS: tuple[dict[str, str], ...] = (
    {
        "field": "epoch1_analytical_bathymetry",
        "description": "A single-band analytical elevation/depth raster or grid for the "
        "earlier epoch.",
    },
    {
        "field": "epoch2_analytical_bathymetry",
        "description": "A single-band analytical elevation/depth raster or grid for the "
        "later epoch.",
    },
    {
        "field": "horizontal_crs",
        "description": "A defined, projected, metric horizontal coordinate reference "
        "system, common to both epochs.",
    },
    {
        "field": "survey_epochs",
        "description": "The acquisition date or date range of each epoch's survey.",
    },
    {
        "field": "nodata_semantics",
        "description": "An explicit nodata/invalid-cell sentinel or mask for each epoch.",
    },
    {
        "field": "known_sign_convention",
        "description": "Whether each epoch's raw values are positive-down depth or "
        "already elevation-style.",
    },
    {
        "field": "demonstrated_vertical_datum_compatibility",
        "description": "Evidence (identical stated datum, or a defensible source-stated "
        "transformation) that both epochs share a common vertical reference -- multi-epoch "
        "bathymetry without this is NOT change-analysis-ready.",
    },
)

STRONGLY_PREFERRED_FIELDS: tuple[dict[str, str], ...] = (
    {
        "field": "survey_uncertainty_evidence",
        "description": "Per-epoch survey uncertainty / HSD / CUBE layers, or a nominal "
        "accuracy statement from the source survey report.",
    },
    {
        "field": "source_survey_report",
        "description": "The source survey/comparison report documenting acquisition, "
        "processing, and datum methodology.",
    },
    {
        "field": "infrastructure_interpretation_context",
        "description": "Existing infrastructure/interpretation context (e.g. cable "
        "routes, turbine locations) relevant to the change map, if already available.",
    },
)


def build_seabed_change_input_contract() -> dict[str, Any]:
    return {
        "scientific_role": "GENERIC_MULTI_EPOCH_SEABED_CHANGE_INPUT_CONTRACT",
        "purpose": "Defines what the future OrbGSS Marine Module requires from operator-supplied "
        "multi-epoch bathymetry before it can be processed into observed seabed-change analytics.",
        "required_fields": list(REQUIRED_FIELDS),
        "strongly_preferred_fields": list(STRONGLY_PREFERRED_FIELDS),
        "notes": [
            "Multi-epoch bathymetry without demonstrated vertical-datum compatibility is NOT "
            "change-analysis-ready -- this is a hard gate, not a soft preference.",
            "This contract is documentation of requirements, not an implemented validator; see "
            "marine_engine.change.epoch_compatibility for the actual per-run gates.",
        ],
    }

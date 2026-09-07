"""Generic operator bathymetry input contract (MAR-020 Section 14).

Zero dependency on any specific project or dataset. Defines what the
future OrbGSS Marine Module requires from an operator bathymetry upload --
documentation only, not a validator (that is `readiness.py`, which checks
a SPECIFIC file against a subset of these same concepts).
"""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS: tuple[dict[str, str], ...] = (
    {
        "field": "analytical_raster_or_grid",
        "description": "A single-band analytical elevation/depth raster or grid -- never an "
        "RGB/rendered preview image.",
    },
    {
        "field": "horizontal_crs",
        "description": "A defined horizontal coordinate reference system.",
    },
    {
        "field": "metric_coordinate_support",
        "description": "A projected, metric CRS, or a geographic CRS that can be safely "
        "transformed to one without ambiguity.",
    },
    {
        "field": "spatial_extent",
        "description": "A valid, finite spatial extent (bounds).",
    },
    {
        "field": "nodata_semantics",
        "description": "An explicit nodata/invalid-cell sentinel or mask.",
    },
    {
        "field": "horizontal_resolution",
        "description": "A known, finite native pixel/grid spacing.",
    },
    {
        "field": "survey_epoch",
        "description": "The acquisition date or date range of the survey.",
    },
)

STRONGLY_PREFERRED_FIELDS: tuple[dict[str, str], ...] = (
    {
        "field": "vertical_datum",
        "description": "The vertical reference datum (e.g. LAT, MSL, a geoid model) -- "
        "required for safe cross-epoch or cross-survey comparison.",
    },
    {
        "field": "sign_convention",
        "description": "Whether raw values are positive-down depth or already elevation-style "
        "(higher = shallower) -- required to build the canonical bed_elevation_m raster "
        "correctly.",
    },
)

SUPPORTED_CANDIDATE_FORMATS: tuple[dict[str, str], ...] = (
    {
        "format": "GeoTIFF",
        "status": "IMPLEMENTED",
        "note": "the format demonstrated end-to-end in MAR-020",
    },
    {
        "format": "ASCII grid (Esri ASCII / XYZ regular grid)",
        "status": "NOT_YET_IMPLEMENTED",
        "note": "a documented candidate for a future ticket; not implemented unless already "
        "trivial reuse of existing project code",
    },
    {
        "format": "XYZ point/grid",
        "status": "NOT_YET_IMPLEMENTED",
        "note": "a documented candidate for a future ticket; not implemented unless already "
        "trivial reuse of existing project code",
    },
)


def build_terrain_input_contract() -> dict[str, Any]:
    """The full, static generic input-contract document. Pure documentation
    of what a future operator upload requires -- no per-run data."""

    return {
        "scientific_role": "GENERIC_OPERATOR_BATHYMETRY_INPUT_CONTRACT",
        "purpose": "Defines what the future OrbGSS Marine Module requires from an operator "
        "bathymetry upload before it can be processed into canonical terrain analytics.",
        "required_fields": list(REQUIRED_FIELDS),
        "strongly_preferred_fields": list(STRONGLY_PREFERRED_FIELDS),
        "supported_candidate_formats": list(SUPPORTED_CANDIDATE_FORMATS),
        "notes": [
            "This contract is documentation of requirements, not an implemented validator; "
            "see marine_engine.terrain.readiness for the actual per-file readiness checks.",
            "Do not implement every candidate format until a real ticket requires it -- "
            "GeoTIFF is the only format actually exercised end-to-end as of MAR-020.",
        ],
    }

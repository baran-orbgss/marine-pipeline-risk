"""Generic bedform-morphodynamics input contract (MAR-022 Section 26).

Zero dependency on any specific project or dataset. Documentation only,
not a validator.
"""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS_STATIC: tuple[dict[str, str], ...] = (
    {
        "field": "high_resolution_analytical_bathymetry",
        "description": "A single-band analytical elevation/depth raster, native resolution "
        "sufficient to resolve the target bedform wavelength scale (this engine's canonical "
        "sand-wave floor is 30 m trough-to-trough).",
    },
    {
        "field": "metric_crs",
        "description": "A defined, projected, metric horizontal coordinate reference system.",
    },
    {
        "field": "known_sign_convention",
        "description": "Whether raw values are positive-down depth or already elevation-style.",
    },
    {
        "field": "native_resolution_sufficient_for_target_wavelength",
        "description": "Pixel size fine enough (relative to the target bedform wavelength) that "
        "the >=3-wavelengths-across-tile canonical eligibility rule can be met.",
    },
    {
        "field": "nodata_semantics",
        "description": "An explicit nodata/invalid-cell sentinel or mask.",
    },
)

REQUIRED_FIELDS_MULTI_EPOCH: tuple[dict[str, str], ...] = (
    {
        "field": "second_bathymetry_epoch",
        "description": "A second high-resolution analytical bathymetry epoch, same minimum "
        "requirements as the first.",
    },
    {
        "field": "demonstrated_vertical_datum_compatibility",
        "description": "Evidence that both epochs share a common vertical reference.",
    },
    {
        "field": "horizontal_co_registration",
        "description": "A demonstrated (not assumed) horizontal grid relationship between the "
        "two epochs -- exact, integer-pixel-offset, or resampled-with-method-recorded.",
    },
    {
        "field": "survey_epochs",
        "description": "The acquisition date or date range of each epoch's survey.",
    },
)

STRONGLY_PREFERRED_FIELDS: tuple[dict[str, str], ...] = (
    {
        "field": "infrastructure_interpretations",
        "description": "Operator-supplied GIS interpretation of infrastructure/disturbance "
        "context (e.g. foundation scour, cable exposures, jack-up locations, rock protection) so "
        "canonical natural-seabed validation tiles can be distinguished from anthropogenically "
        "disturbed ones.",
    },
    {
        "field": "seabed_feature_interpretations",
        "description": "Operator-supplied GIS interpretation of natural seabed features (e.g. "
        "sand-wave crests) usable as an independent source-interpretation comparator -- never as "
        "a seed for this engine's own detector.",
    },
    {
        "field": "source_survey_report",
        "description": "The source survey report documenting acquisition, processing, and "
        "accuracy methodology.",
    },
)


def build_bedform_morphodynamics_input_contract() -> dict[str, Any]:
    return {
        "scientific_role": "GENERIC_SANDBED_BEDFORM_MORPHOLOGY_AND_OBSERVED_CHANGE_INPUT_CONTRACT",
        "purpose": "Defines what the future OrbGSS Marine Module requires from operator-supplied "
        "bathymetry before it can be processed into static bedform morphometry and, where two "
        "epochs are available, observed multi-epoch crest displacement.",
        "required_fields_static_morphometry": list(REQUIRED_FIELDS_STATIC),
        "required_fields_multi_epoch_change": list(REQUIRED_FIELDS_MULTI_EPOCH),
        "strongly_preferred_fields": list(STRONGLY_PREFERRED_FIELDS),
        "notes": [
            "This contract is documentation of requirements, not an implemented validator.",
            "Static bedform geometry, observed multi-epoch change, and future migration "
            "prediction are three distinct things -- this contract, like this engine, covers "
            "only the first two.",
        ],
    }

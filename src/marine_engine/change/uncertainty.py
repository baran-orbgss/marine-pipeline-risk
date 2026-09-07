"""Generic DoD uncertainty evidence + propagation (MAR-021 Sections 12-14).

Zero dependency on any specific project or dataset. This module NEVER
invents a change-detection threshold: it only derives one when the caller
supplies real, explicit evidence (a per-epoch nominal vertical accuracy
figure, or a verified per-cell sigma raster), and always records exactly
which evidence backed the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PRECISION_PROXY_NOT_TOTAL_SURVEY_UNCERTAINTY = "PRECISION_PROXY_NOT_TOTAL_SURVEY_UNCERTAINTY"
RAW_DOD_AVAILABLE_UNCERTAINTY_THRESHOLD_NOT_DEFENSIBLE = (
    "RAW_DOD_AVAILABLE_UNCERTAINTY_THRESHOLD_NOT_DEFENSIBLE"
)
VOLUMETRIC_CHANGE_NOT_REPORTED_UNCERTAINTY_INSUFFICIENT = (
    "VOLUMETRIC_CHANGE_NOT_REPORTED_UNCERTAINTY_INSUFFICIENT"
)

REPORTED_NOMINAL_ACCURACY = (
    "REPORTED_NOMINAL_ACCURACY"  # a scalar, source-documented per-epoch figure
)
VERIFIED_PER_CELL_SIGMA = (
    "VERIFIED_PER_CELL_SIGMA"  # a spatially-resolved, unit-verified sigma raster
)
UNVERIFIED_CLASSIFIED_GRID = (
    "UNVERIFIED_CLASSIFIED_GRID"  # acquired but units/scale not established
)


@dataclass(frozen=True)
class UncertaintyEvidenceItem:
    kind: str  # one of the *_EVIDENCE constants below
    epoch: str
    description: str
    source_citation: str
    usable_for_threshold: bool


EVIDENCE_NOMINAL_ACCURACY = "NOMINAL_ACCURACY_STATEMENT"
EVIDENCE_DATUM_SQUARE_REPEATABILITY = "DATUM_SQUARE_REPEATABILITY"
EVIDENCE_CLASSIFIED_GRID_UNVERIFIED = "CLASSIFIED_GRID_UNVERIFIED_UNITS"


def build_uncertainty_evidence_inventory(items: list[UncertaintyEvidenceItem]) -> dict[str, Any]:
    return {
        "scientific_role": "GENERIC_DOD_UNCERTAINTY_EVIDENCE_INVENTORY",
        "items": [
            {
                "kind": item.kind,
                "epoch": item.epoch,
                "description": item.description,
                "source_citation": item.source_citation,
                "usable_for_threshold": item.usable_for_threshold,
            }
            for item in items
        ],
        "notes": [
            "This inventory records what uncertainty evidence was actually found -- it never "
            "assumes a classified/byte-scaled grid (e.g. an 'HSD' product with no documented "
            "scale) is directly usable as a metres-scale standard deviation without independent "
            "unit verification.",
        ],
    }


@dataclass(frozen=True)
class ThresholdDerivation:
    status: str
    threshold_m: float | None
    formula: str | None
    inputs: dict[str, float] | None
    reason: str


def derive_change_threshold(
    *,
    sigma_epoch1_m: float | None,
    sigma_epoch2_m: float | None,
    evidence_citation: str | None,
) -> ThresholdDerivation:
    """Derives a threshold ONLY from explicit, named per-epoch nominal
    accuracy figures with a real citation -- never from the DoD data
    itself, never an invented round number."""

    if sigma_epoch1_m is None or sigma_epoch2_m is None or not evidence_citation:
        return ThresholdDerivation(
            status=RAW_DOD_AVAILABLE_UNCERTAINTY_THRESHOLD_NOT_DEFENSIBLE,
            threshold_m=None,
            formula=None,
            inputs=None,
            reason="no per-epoch nominal accuracy figure (with a real source citation) was "
            "supplied for one or both epochs -- retaining the continuous DoD only, per Section "
            "14 (this is a valid POC result, not a failure)",
        )

    threshold = float(np.sqrt(sigma_epoch1_m**2 + sigma_epoch2_m**2))
    return ThresholdDerivation(
        status="THRESHOLD_DERIVED_FROM_REPORTED_NOMINAL_ACCURACY",
        threshold_m=threshold,
        formula="threshold_m = sqrt(sigma_epoch1_m^2 + sigma_epoch2_m^2)",
        inputs={"sigma_epoch1_m": sigma_epoch1_m, "sigma_epoch2_m": sigma_epoch2_m},
        reason=f"derived from source-reported per-epoch nominal vertical accuracy figures "
        f"({evidence_citation}) -- a scalar nominal figure, not a spatially-resolved per-cell "
        "propagation (see propagate_sigma_dod_raster for that case, which requires verified "
        "per-cell sigma rasters for BOTH epochs)",
    )


def propagate_sigma_dod_scalar(sigma_epoch1_m: float, sigma_epoch2_m: float) -> float:
    """sqrt(sigma1^2 + sigma2^2) for two independent nominal accuracy
    figures."""

    return float(np.sqrt(sigma_epoch1_m**2 + sigma_epoch2_m**2))


def propagate_sigma_dod_raster(
    sigma_epoch1_m: np.ndarray, sigma_epoch2_m: np.ndarray, common_valid_mask: np.ndarray
) -> np.ndarray:
    """sqrt(sigma1^2 + sigma2^2), per cell -- ONLY valid to call when both
    inputs are already-verified, unit-correct (metres, 1-sigma elevation
    uncertainty) rasters on the SAME common grid as the DoD (Section 13).
    This function does not itself verify units -- that verification must
    happen before calling it, and is recorded in the uncertainty evidence
    inventory, never silently assumed."""

    if sigma_epoch1_m.shape != sigma_epoch2_m.shape:
        raise ValueError("sigma rasters must already be on the same common grid")
    sigma_dod = np.sqrt(sigma_epoch1_m**2 + sigma_epoch2_m**2)
    return np.where(common_valid_mask, sigma_dod, np.nan)

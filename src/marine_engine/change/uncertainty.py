"""Generic DoD uncertainty evidence + propagation (MAR-021 Sections 12-14;
uncertainty SEMANTICS repaired by MAR-021A).

Zero dependency on any specific project or dataset. This module NEVER
invents a change-detection threshold, and -- as of MAR-021A -- it also
never treats a merely "nominal"/"typical" source accuracy figure as if it
were a verified 1-sigma standard uncertainty. MAR-021 originally computed
`sqrt(sigma1^2 + sigma2^2)` from the Sheringham Shoal source's own "typically
less than +/-0.2 m" statement and reported the result as a "defensible
uncertainty threshold" -- that overstated the source's own claim: the
source never established that +/-0.2 m is a 1-sigma (or any other named
confidence-level) figure, so treating it as an addable-in-quadrature sigma
was not defensible. `derive_change_threshold` below now ALWAYS reports
`GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED` for nominal-accuracy
inputs; the same RSS arithmetic is still computed and returned, but only as
`nominal_accuracy_rss_reference_m` -- explicitly labelled as a reference
number, never a canonical threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PRECISION_PROXY_NOT_TOTAL_SURVEY_UNCERTAINTY = "PRECISION_PROXY_NOT_TOTAL_SURVEY_UNCERTAINTY"

# MAR-021A: replaces the old RAW_DOD_AVAILABLE_UNCERTAINTY_THRESHOLD_NOT_DEFENSIBLE status.
# Returned whenever a GENERIC, defensible DoD significance threshold has not been demonstrated --
# which, for this project, is ALWAYS (no per-epoch value has ever been verified by its source as
# a 1-sigma or documented-equivalent-confidence standard uncertainty; a "nominal"/"typical"
# accuracy figure does not qualify on its own, however real and well-cited it is).
GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED = (
    "GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED"
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
# MAR-021A: a source's OWN analyst significance criterion (e.g. Fugro's own "changes <0.3 m were
# not considered significant") -- real, citable evidence, but scoped to that ONE source's own
# interpretation practice. It must never be repackaged as a generic OrbGSS uncertainty formula or
# threshold; it is preserved only as contextual evidence alongside the measurement-accuracy items.
SOURCE_SPECIFIC_ANALYST_SIGNIFICANCE_THRESHOLD = "SOURCE_SPECIFIC_ANALYST_SIGNIFICANCE_THRESHOLD"


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
EVIDENCE_SOURCE_SPECIFIC_ANALYST_THRESHOLD = "SOURCE_SPECIFIC_ANALYST_THRESHOLD_STATEMENT"


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
    generic_threshold_m: float | None
    nominal_accuracy_rss_reference_m: float | None
    formula: str | None
    inputs: dict[str, float] | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "generic_threshold_m": self.generic_threshold_m,
            "nominal_accuracy_rss_reference_m": self.nominal_accuracy_rss_reference_m,
            "formula": self.formula,
            "inputs": self.inputs,
            "reason": self.reason,
        }


def derive_change_threshold(
    *,
    sigma_epoch1_m: float | None,
    sigma_epoch2_m: float | None,
    evidence_citation: str | None,
) -> ThresholdDerivation:
    """MAR-021A: a GENERIC, defensible DoD significance threshold requires
    a per-epoch value the SOURCE has explicitly verified as a 1-sigma (or
    documented-equivalent-confidence) standard uncertainty. This function
    has no parameter for that because no project in this codebase has ever
    been supplied one -- a merely "nominal"/"typical" accuracy figure does
    NOT qualify on its own, however real and well-cited it is, since the
    source may never have stated what statistical confidence it
    represents. This function therefore ALWAYS returns
    `GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED` and a `None`
    `generic_threshold_m` -- this is a valid, expected POC result, not a
    failure (Section 14/MAR-021A).

    The RSS combination of the two supplied nominal figures is still
    computed and returned as `nominal_accuracy_rss_reference_m` for
    transparency (callers may want to SEE the number that MAR-021
    mistakenly called a threshold), but it is explicitly NOT a propagated
    1-sigma DoD uncertainty and NOT usable as a canonical significance
    threshold -- callers must never present it as either.
    """

    rss_reference = None
    if sigma_epoch1_m is not None and sigma_epoch2_m is not None:
        rss_reference = propagate_sigma_dod_scalar(sigma_epoch1_m, sigma_epoch2_m)

    reason = (
        "source accuracy confidence semantics are insufficient for generic uncertainty "
        "propagation -- a nominal/typical accuracy figure is not necessarily a 1-sigma "
        "standard uncertainty, and no source in this project has established that it is"
    )
    if sigma_epoch1_m is None or sigma_epoch2_m is None or not evidence_citation:
        reason += (
            "; additionally, no per-epoch accuracy figure (with a real source citation) "
            "was supplied for one or both epochs"
        )

    return ThresholdDerivation(
        status=GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED,
        generic_threshold_m=None,
        nominal_accuracy_rss_reference_m=rss_reference,
        formula="nominal_accuracy_rss_reference_m = sqrt(sigma_epoch1_m^2 + sigma_epoch2_m^2) "
        "-- NOT a propagated 1-sigma DoD uncertainty and NOT a canonical significance threshold"
        if rss_reference is not None
        else None,
        inputs={"sigma_epoch1_m": sigma_epoch1_m, "sigma_epoch2_m": sigma_epoch2_m}
        if rss_reference is not None
        else None,
        reason=reason,
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

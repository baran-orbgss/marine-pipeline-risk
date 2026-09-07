"""Generic canonical DEM-of-Difference calculation (MAR-021 Section 11).

Zero dependency on any specific project or dataset. Exactly one canonical
definition, computed exactly once:

    delta_bed_elevation_m = bed_elevation_epoch2_m - bed_elevation_epoch1_m

positive = seabed raised (deposition/fill-compatible); negative = seabed
lowered (erosion/scour-compatible) -- see `OBSERVED_SEABED_ELEVATION_CHANGE`
below, the required wording (Section 1), used verbatim wherever this
result is described. No smoothing is applied before this subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBSERVED_SEABED_ELEVATION_CHANGE = "OBSERVED_SEABED_ELEVATION_CHANGE"
OBSERVED_SEABED_RAISING = "OBSERVED_SEABED_RAISING"
OBSERVED_SEABED_LOWERING = "OBSERVED_SEABED_LOWERING"

ANNUALIZATION_DISCLAIMER = (
    "ANNUALIZED OBSERVED CHANGE IS NOT A FUTURE CHANGE RATE PREDICTION. It is the observed "
    "total change over the survey interval divided by the elapsed time -- a descriptive rate, "
    "not a forecast."
)


@dataclass(frozen=True)
class DoDResult:
    delta_bed_elevation_m: np.ndarray
    common_valid_mask: np.ndarray
    definition: str = "delta_bed_elevation_m = bed_elevation_epoch2_m - bed_elevation_epoch1_m"


def compute_delta_bed_elevation(
    bed_elevation_epoch1_m: np.ndarray,
    bed_elevation_epoch2_m: np.ndarray,
    common_valid_mask: np.ndarray,
) -> DoDResult:
    if bed_elevation_epoch1_m.shape != bed_elevation_epoch2_m.shape:
        raise ValueError(
            f"epochs must already be on a common grid (shapes {bed_elevation_epoch1_m.shape} vs "
            f"{bed_elevation_epoch2_m.shape})"
        )
    delta = np.where(common_valid_mask, bed_elevation_epoch2_m - bed_elevation_epoch1_m, np.nan)
    return DoDResult(delta_bed_elevation_m=delta, common_valid_mask=common_valid_mask)


def classify_change_direction(delta_bed_elevation_m: np.ndarray) -> np.ndarray:
    """Returns an array of dtype=object with OBSERVED_SEABED_RAISING /
    OBSERVED_SEABED_LOWERING / None (flat/NaN) per cell -- raw physical
    labels only, never an anthropogenic-vs-natural or erosion-vs-scour
    interpretation (Section 18)."""

    labels = np.full(delta_bed_elevation_m.shape, None, dtype=object)
    finite = np.isfinite(delta_bed_elevation_m)
    labels[finite & (delta_bed_elevation_m > 0)] = OBSERVED_SEABED_RAISING
    labels[finite & (delta_bed_elevation_m < 0)] = OBSERVED_SEABED_LOWERING
    return labels


@dataclass(frozen=True)
class AnnualizedChangeResult:
    annualized_delta_m_per_year: np.ndarray
    elapsed_years: float
    disclaimer: str = ANNUALIZATION_DISCLAIMER


def annualize_change(
    delta_bed_elevation_m: np.ndarray, *, epoch1_date, epoch2_date
) -> AnnualizedChangeResult:
    """Purely descriptive: total observed change / elapsed time. Requires
    real epoch dates (never assumed); raises if epoch2 is not after
    epoch1."""

    elapsed_days = (epoch2_date - epoch1_date).days
    if elapsed_days <= 0:
        raise ValueError(
            f"epoch2_date ({epoch2_date}) must be after epoch1_date ({epoch1_date}) to annualize"
        )
    elapsed_years = elapsed_days / 365.25
    return AnnualizedChangeResult(
        annualized_delta_m_per_year=delta_bed_elevation_m / elapsed_years,
        elapsed_years=elapsed_years,
    )

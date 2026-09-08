"""Canonical pipe-underside clearance and its classification threshold (MAR-025 Sections 4-7).

Zero dependency on any specific project or dataset. `seabed_support_elevation_m` must already
be in the project's one accepted canonical terrain convention (`marine_engine.terrain.canonical`
-- higher value = shallower/crestward) -- `canonicalize_seabed_support_elevation_m` performs
that conversion for a 1-D support profile, reusing the exact same accepted sign vocabulary
(`POSITIVE_DOWN_DEPTH`/`ALREADY_ELEVATION_STYLE`) rather than inventing a parallel one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from marine_engine.terrain.canonical import (
    ALREADY_ELEVATION_STYLE,
    POSITIVE_DOWN_DEPTH,
    SOURCE_SIGN_CONVENTIONS,
)

__all__ = [
    "ALREADY_ELEVATION_STYLE",
    "POSITIVE_DOWN_DEPTH",
    "SOURCE_SIGN_CONVENTIONS",
    "canonicalize_seabed_support_elevation_m",
    "compute_pipe_underside_clearance_m",
    "FREE_SPAN_CLASSIFICATION_NOT_DEFENSIBLE_NO_CLEARANCE_THRESHOLD",
    "FREE_SPAN_CLASSIFICATION_DEFENSIBLE",
    "SOURCE_OR_OPERATOR_CLEARANCE_CLASSIFICATION_THRESHOLD",
    "COMBINED_ONE_SIGMA_CLEARANCE_THRESHOLD",
    "compute_combined_one_sigma_clearance_threshold_m",
    "ClearanceClassificationThreshold",
    "compute_support_clearance_margin_m",
]


# --- Section 4: seabed support reference -- reuses the one accepted canonical convention ------


def canonicalize_seabed_support_elevation_m(
    raw_m: np.ndarray, *, source_sign_convention: str
) -> np.ndarray:
    """Section 4: convert a raw seabed support profile into the accepted canonical elevation
    convention (higher = shallower), exactly once, mirroring
    `terrain.canonical.build_canonical_bed_elevation`'s formula for a 1-D profile rather than a
    raster. Never subtracts a positive-down raw depth without this explicit conversion."""

    if source_sign_convention not in SOURCE_SIGN_CONVENTIONS:
        raise ValueError(f"unknown source sign convention: {source_sign_convention!r}")
    raw = np.asarray(raw_m, dtype=np.float64)
    if source_sign_convention == POSITIVE_DOWN_DEPTH:
        return -raw
    return raw  # ALREADY_ELEVATION_STYLE


# --- Section 5: canonical clearance -----------------------------------------------------------


def compute_pipe_underside_clearance_m(
    pipe_bottom_elevation_m: float | None, seabed_support_elevation_m: float | None
) -> float | None:
    """Section 5: `pipe_bottom_elevation_m - seabed_support_elevation_m`. Positive = the pipe
    underside is geometrically above the local seabed support surface (a gap exists beneath the
    pipe); zero = coincident; negative = the pipe underside lies below that surface (resting on
    or embedded in it). `None` if either input is unavailable -- never a fabricated clearance."""

    if pipe_bottom_elevation_m is None or seabed_support_elevation_m is None:
        return None
    return pipe_bottom_elevation_m - seabed_support_elevation_m


# --- Section 6-7: clearance classification threshold -- never a fixed magic number -----------

# Section 6's escape hatch: retained only when no defensible threshold exists at all.
FREE_SPAN_CLASSIFICATION_NOT_DEFENSIBLE_NO_CLEARANCE_THRESHOLD = (
    "FREE_SPAN_CLASSIFICATION_NOT_DEFENSIBLE_NO_CLEARANCE_THRESHOLD"
)
FREE_SPAN_CLASSIFICATION_DEFENSIBLE = "FREE_SPAN_CLASSIFICATION_DEFENSIBLE"

# Section 7: threshold provenance -- an operator/source-stated QC threshold is never relabelled
# as a statistically-combined uncertainty, and vice versa.
SOURCE_OR_OPERATOR_CLEARANCE_CLASSIFICATION_THRESHOLD = (
    "SOURCE_OR_OPERATOR_CLEARANCE_CLASSIFICATION_THRESHOLD"
)
COMBINED_ONE_SIGMA_CLEARANCE_THRESHOLD = "COMBINED_ONE_SIGMA_CLEARANCE_THRESHOLD"

THRESHOLD_PROVENANCE_TYPES = frozenset(
    {SOURCE_OR_OPERATOR_CLEARANCE_CLASSIFICATION_THRESHOLD, COMBINED_ONE_SIGMA_CLEARANCE_THRESHOLD}
)


@dataclass(frozen=True)
class ClearanceClassificationThreshold:
    """A defensible clearance-significance threshold plus where it came from (Section 6-7).
    Never constructed with an invented default -- both fields are always caller-supplied."""

    threshold_m: float
    provenance: str

    def __post_init__(self) -> None:
        if self.provenance not in THRESHOLD_PROVENANCE_TYPES:
            raise ValueError(f"unknown threshold provenance: {self.provenance!r}")
        if self.threshold_m < 0:
            raise ValueError(f"threshold_m must be >= 0 (a magnitude); got {self.threshold_m!r}")


def compute_combined_one_sigma_clearance_threshold_m(
    sigma_pipe_vertical_m: float, sigma_seabed_vertical_m: float
) -> float:
    """Section 7: `sqrt(sigma_pipe_vertical^2 + sigma_seabed_vertical^2)`. The caller must have
    already verified both inputs are genuinely stated as 1-sigma uncertainties -- this function
    never checks confidence semantics itself (they cannot be inferred from the numbers alone);
    if that cannot be verified, do not call this function at all (Section 7: "If uncertainty
    confidence semantics are unresolved: do not propagate them as sigma"), use an operator
    `SOURCE_OR_OPERATOR_CLEARANCE_CLASSIFICATION_THRESHOLD` instead."""

    if sigma_pipe_vertical_m < 0 or sigma_seabed_vertical_m < 0:
        raise ValueError("sigma values must be >= 0 (a magnitude)")
    return math.sqrt(sigma_pipe_vertical_m**2 + sigma_seabed_vertical_m**2)


# --- Section 15: physical margin, not a 0-1 score ----------------------------------------------


def compute_support_clearance_margin_m(
    clearance_m: float | None, classification_threshold_m: float | None
) -> float | None:
    """Section 15: `abs(clearance_m) - classification_threshold_m` -- how far past the
    classification boundary the clearance sits (positive = confidently classified in whichever
    direction, negative = inside the ambiguous transition band). `None` when no threshold is
    available (never a fabricated margin). A physical value in metres, never a 0-1 score."""

    if clearance_m is None or classification_threshold_m is None:
        return None
    return abs(clearance_m) - classification_threshold_m

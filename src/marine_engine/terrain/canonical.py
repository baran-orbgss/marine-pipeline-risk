"""Generic canonical terrain raster semantics (MAR-020 Section 6).

Zero dependency on any specific project or dataset. Defines exactly one
internal canonical representation, `bed_elevation_m`, where a HIGHER value
always means shallower/crestward -- regardless of whether the real source
raster's own convention was already elevation-style or was a positive-down
depth. The conversion (if any) happens exactly once, is always explicit
(never guessed from a filename), and both the source and canonical
conventions are preserved side by side in the returned record so no
provenance is ever silently lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Every raw value must be reinterpreted through exactly one of these two source conventions --
# there is no silent default; the caller must state which one applies, based on verified
# evidence (see `infer_source_sign_convention`), never a filename guess alone.
POSITIVE_DOWN_DEPTH = "POSITIVE_DOWN_DEPTH"  # raw value = depth below datum, larger = deeper
ALREADY_ELEVATION_STYLE = "ALREADY_ELEVATION_STYLE"  # raw value already: higher = shallower

SOURCE_SIGN_CONVENTIONS = frozenset({POSITIVE_DOWN_DEPTH, ALREADY_ELEVATION_STYLE})

CANONICAL_SIGN_CONVENTION_NOTE = (
    "bed_elevation_m: higher value = shallower/crestward, lower value = deeper. "
    "Converted from the verified source convention exactly once; see "
    "source_sign_convention/source_vertical_datum for the original semantics."
)


@dataclass(frozen=True)
class SignConventionEvidence:
    """What was actually checked before declaring a sign convention --
    never a bare assertion. `plausible` is a soft signal, not a hard gate:
    real data can legitimately fail a plausibility heuristic (e.g. an
    inland/very shallow site), so callers should record but not
    auto-reject on `plausible=False`."""

    declared_convention: str
    raw_min: float
    raw_max: float
    raw_mean: float
    reference_depth_range_m: tuple[float, float] | None
    plausible: bool
    note: str


def infer_source_sign_convention_evidence(
    raw_min: float,
    raw_max: float,
    raw_mean: float,
    *,
    declared_convention: str,
    reference_depth_range_m: tuple[float, float] | None = None,
) -> SignConventionEvidence:
    """A plausibility CROSS-CHECK for a convention the caller has already
    determined some other way (e.g. real raster inspection, source
    metadata) -- this function never guesses the convention itself, it
    only flags whether the declared one is consistent with the raw
    statistics, so a wrong declaration is caught rather than silently
    trusted."""

    if declared_convention not in SOURCE_SIGN_CONVENTIONS:
        raise ValueError(f"unknown source sign convention: {declared_convention!r}")

    plausible = True
    note = ""
    if declared_convention == POSITIVE_DOWN_DEPTH:
        plausible = raw_mean > 0
        note = (
            "positive-down depth declared; raw values are mostly positive, consistent"
            if plausible
            else "positive-down depth declared, but raw values are mostly NEGATIVE -- "
            "re-check the declared convention against real source documentation"
        )
    elif declared_convention == ALREADY_ELEVATION_STYLE:
        plausible = raw_max <= 0 or raw_mean < 0
        note = (
            "already-elevation-style declared; raw values are consistent with a submerged "
            "site referenced to a datum near the surface (all/mostly <= 0)"
            if plausible
            else "already-elevation-style declared, but raw values are mostly POSITIVE -- "
            "re-check the declared convention against real source documentation"
        )

    if reference_depth_range_m is not None and plausible:
        lo, hi = reference_depth_range_m
        magnitude = abs(raw_mean)
        if not (0.3 * lo <= magnitude <= 3.0 * hi):
            plausible = False
            note += (
                f"; magnitude {magnitude:.1f} is well outside the expected regional depth "
                f"context {reference_depth_range_m} -- worth independent confirmation"
            )

    return SignConventionEvidence(
        declared_convention=declared_convention,
        raw_min=raw_min,
        raw_max=raw_max,
        raw_mean=raw_mean,
        reference_depth_range_m=reference_depth_range_m,
        plausible=plausible,
        note=note,
    )


@dataclass(frozen=True)
class CanonicalTerrainRaster:
    bed_elevation_m: np.ndarray
    valid_mask: np.ndarray
    source_sign_convention: str
    source_vertical_datum: str | None
    canonical_sign_convention_note: str = CANONICAL_SIGN_CONVENTION_NOTE


def build_canonical_bed_elevation(
    raw: np.ndarray,
    *,
    nodata_value: float | None,
    source_sign_convention: str,
    source_vertical_datum: str | None,
) -> CanonicalTerrainRaster:
    """Convert a raw band into the canonical `bed_elevation_m` convention,
    exactly once, with the conversion fully documented in the return
    value. Never modifies the caller's own array in place."""

    if source_sign_convention not in SOURCE_SIGN_CONVENTIONS:
        raise ValueError(f"unknown source sign convention: {source_sign_convention!r}")

    raw = np.asarray(raw, dtype=np.float64)
    if nodata_value is None:
        valid_mask = np.isfinite(raw)
    else:
        valid_mask = np.isfinite(raw) & ~np.isclose(
            raw, nodata_value, rtol=0, atol=abs(nodata_value) * 1e-6 + 1e-6
        )

    if source_sign_convention == POSITIVE_DOWN_DEPTH:
        bed_elevation_m = np.where(valid_mask, -raw, np.nan)
    else:  # ALREADY_ELEVATION_STYLE
        bed_elevation_m = np.where(valid_mask, raw, np.nan)

    return CanonicalTerrainRaster(
        bed_elevation_m=bed_elevation_m,
        valid_mask=valid_mask,
        source_sign_convention=source_sign_convention,
        source_vertical_datum=source_vertical_datum,
    )

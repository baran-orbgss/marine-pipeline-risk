"""Canonical burial-cover normalization -- sign convention and reference-point repair
(MAR-024A Problems A and B).

Turns a raw source-reported burial/depth value into a canonical, physically
meaningful quantity in exactly two steps, each performed exactly once:

1. `normalize_canonical_reference_burial_depth_m` resolves the sign
   convention: positive means the stated reference point lies BELOW seabed,
   zero means AT seabed, negative means ABOVE seabed. Returns `None`
   whenever the sign convention is unresolved -- never a guessed sign.
2. `compute_cover_above_asset_m` resolves the reference-point offset:
   TOP_OF_ASSET_BURIAL passes the canonical depth through unchanged;
   CENTRELINE_BURIAL and OTHER_SOURCE_SPECIFIC_REFERENCE require an
   explicit, never-assumed `reference_to_asset_top_offset_m` (for a
   circular asset this may be diameter / 2, supplied by the caller -- this
   module never assumes a diameter or a circular cross-section).

`cover_above_asset_m` is the only numeric quantity this package permits to
drive measured burial-state classification (see `burial.profile`).
"""

from __future__ import annotations

import pandas as pd

from marine_engine.burial.semantics import (
    BURIAL_REFERENCE_TYPES,
    CENTRELINE_BURIAL,
    NEGATIVE_VALUE_MEANS_DEEPER_BURIAL,
    OTHER_SOURCE_SPECIFIC_REFERENCE,
    POSITIVE_VALUE_MEANS_DEEPER_BURIAL,
    SIGN_CONVENTIONS,
    SOURCE_BURIAL_REFERENCE_UNRESOLVED,
    TOP_OF_ASSET_BURIAL,
)


def normalize_canonical_reference_burial_depth_m(
    raw_value_m: float | None, *, sign_convention: str
) -> float | None:
    """MAR-024A Section 3: apply the resolved sign convention exactly once. `None` in yields
    `None` out; `SIGN_CONVENTION_UNRESOLVED` also yields `None` regardless of `raw_value_m` --
    an unresolved sign is never guessed from the raw value's own sign."""

    if sign_convention not in SIGN_CONVENTIONS:
        raise ValueError(
            f"unknown sign_convention: {sign_convention!r} -- must be one of "
            f"{sorted(SIGN_CONVENTIONS)}"
        )
    if raw_value_m is None or (isinstance(raw_value_m, float) and pd.isna(raw_value_m)):
        return None
    if sign_convention == POSITIVE_VALUE_MEANS_DEEPER_BURIAL:
        return float(raw_value_m)
    if sign_convention == NEGATIVE_VALUE_MEANS_DEEPER_BURIAL:
        return -float(raw_value_m)
    return None  # SIGN_CONVENTION_UNRESOLVED


def compute_cover_above_asset_m(
    canonical_reference_burial_depth_m: float | None,
    *,
    burial_reference_type: str,
    reference_to_asset_top_offset_m: float | None = None,
) -> float | None:
    """MAR-024A Section 4-5: convert a canonical reference burial depth (already sign-
    normalized) into cover directly above the top of the asset.

    TOP_OF_ASSET_BURIAL requires no offset. CENTRELINE_BURIAL and
    OTHER_SOURCE_SPECIFIC_REFERENCE both require an explicit
    `reference_to_asset_top_offset_m` (never assumed, e.g. never a hard-coded diameter / 2) --
    without one, no cover value is produced. SOURCE_BURIAL_REFERENCE_UNRESOLVED, or a `None`
    canonical depth from an unresolved sign convention, both yield `None` (no canonical cover
    may be generated without a resolved reference)."""

    if burial_reference_type not in BURIAL_REFERENCE_TYPES:
        raise ValueError(f"unknown burial_reference_type: {burial_reference_type!r}")
    if canonical_reference_burial_depth_m is None or (
        isinstance(canonical_reference_burial_depth_m, float)
        and pd.isna(canonical_reference_burial_depth_m)
    ):
        return None
    if burial_reference_type == SOURCE_BURIAL_REFERENCE_UNRESOLVED:
        return None
    if burial_reference_type == TOP_OF_ASSET_BURIAL:
        return canonical_reference_burial_depth_m
    # CENTRELINE_BURIAL and OTHER_SOURCE_SPECIFIC_REFERENCE both require an explicit offset.
    assert burial_reference_type in (CENTRELINE_BURIAL, OTHER_SOURCE_SPECIFIC_REFERENCE)
    if reference_to_asset_top_offset_m is None:
        return None
    return canonical_reference_burial_depth_m - reference_to_asset_top_offset_m

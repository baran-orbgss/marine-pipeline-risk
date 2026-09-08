"""Generic independent multi-epoch crest matching (MAR-022 Sections 12-16).

Zero dependency on any specific project or dataset -- no site coordinates,
no epoch years, no CRS. Operates only on caller-supplied crest records
(each a plain dict with `bedform_id`/`x`/`y`/`crest_azimuth_deg`/
`wavelength_m`) already scoped to ONE tile's own two independent
extractions (Section 12: "never match across clearly different bedform
systems" -- enforced structurally by scoping matching to a single tile's
own crest sets, never a whole-site search).

This module NEVER takes a DEM-of-Difference/elevation-change array as an
input -- matching decisions are made from crest geometry alone (spatial
proximity, orientation compatibility, wavelength-scale consistency, and
displacement along the local cross-crest normal). `sample_dod_around_
matched_crest` is a separate, clearly-named function the caller may
invoke AFTERWARD, purely to attach descriptive supporting evidence to an
already-decided match -- it can never influence which pairs matched.

No numeric confidence/probability score is ever produced; only the
three categorical statuses below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio

MATCHED_HIGH_SUPPORT = "MATCHED_HIGH_SUPPORT"
MATCHED_WITH_LIMITATIONS = "MATCHED_WITH_LIMITATIONS"
AMBIGUOUS_NO_CANONICAL_MATCH = "AMBIGUOUS_NO_CANONICAL_MATCH"
MATCH_STATUSES = frozenset(
    {MATCHED_HIGH_SUPPORT, MATCHED_WITH_LIMITATIONS, AMBIGUOUS_NO_CANONICAL_MATCH}
)

REJECTED_OUTSIDE_SEARCH_RADIUS = "OUTSIDE_SEARCH_RADIUS"
REJECTED_ALONG_CREST_OFFSET_TOO_LARGE = "ALONG_CREST_OFFSET_TOO_LARGE"
REJECTED_ORIENTATION_INCOMPATIBLE = "ORIENTATION_INCOMPATIBLE"
REJECTED_WAVELENGTH_SCALE_INCONSISTENT = "WAVELENGTH_SCALE_INCONSISTENT"
REJECTED_NOT_BEST_CANDIDATE = "NOT_BEST_CANDIDATE_FOR_ITS_ENDPOINT"
REJECTED_AMBIGUOUS = "AMBIGUOUS_MULTIPLE_EQUALLY_CLOSE_CANDIDATES"
REJECTED_NO_CANDIDATE = "NO_CANDIDATE_WITHIN_TOLERANCE"

OBSERVED_CREST_DISPLACEMENT = "OBSERVED_CREST_DISPLACEMENT"
OBSERVED_APPARENT_CREST_DISPLACEMENT_RATE = "OBSERVED_APPARENT_CREST_DISPLACEMENT_RATE"
APPARENT_RATE_DISCLAIMER = (
    "Derived from two survey epochs only. Survey dates are month-scale approximate. This is NOT "
    "a future migration rate, NOT a Knaapen predictor output, and NOT a long-term trend."
)

# Named, documented tolerances -- never magic numbers scattered through the matching logic.
# Deliberately generous relative to the ~10 m/2-year historical context noted for this site
# (Section 17) -- that number is external historical context only and must never be used to
# calibrate these tolerances; they are set wide enough to admit a real match without presupposing
# any particular migration rate.
MAX_SEARCH_RADIUS_M = 60.0
MAX_ALONG_CREST_OFFSET_M = 60.0
ORIENTATION_TOLERANCE_DEG = 30.0
WAVELENGTH_RATIO_TOLERANCE = (0.5, 2.0)
AMBIGUITY_MARGIN_M = 5.0
HIGH_SUPPORT_ORIENTATION_DEG = 10.0
HIGH_SUPPORT_WAVELENGTH_RATIO = (0.8, 1.25)
HIGH_SUPPORT_DISTANCE_FRACTION = 0.5


def _angular_difference_undirected(a_deg: float, b_deg: float) -> float:
    """Difference between two UNDIRECTED (0-180, crest-orientation-style)
    azimuths."""

    diff = abs(a_deg - b_deg) % 180.0
    return min(diff, 180.0 - diff)


@dataclass(frozen=True)
class DisplacementDecomposition:
    normal_azimuth_deg: float
    normal_component_m: float
    along_crest_component_m: float


def decompose_displacement(
    dx_m: float, dy_m: float, reference_crest_azimuth_deg: float
) -> DisplacementDecomposition:
    """Section 14: 'Define a reproducible local normal axis.' The normal
    axis is always `(reference_crest_azimuth_deg + 90) deg mod 360` --
    since a crest orientation is itself reported mod 180 (MAR-017
    convention), this normal azimuth is a deterministic, reproducible
    choice between the two geometrically-opposite normal directions, NOT
    an assertion that it points toward any particular compass direction
    (Section 14: never label it east/west/north/south)."""

    normal_azimuth_deg = (reference_crest_azimuth_deg + 90.0) % 360.0
    normal_rad = math.radians(normal_azimuth_deg)
    crest_rad = math.radians(reference_crest_azimuth_deg)
    # Azimuth convention throughout this project: 0=north(+y), 90=east(+x).
    normal_unit = (math.sin(normal_rad), math.cos(normal_rad))
    along_unit = (math.sin(crest_rad), math.cos(crest_rad))
    normal_component = dx_m * normal_unit[0] + dy_m * normal_unit[1]
    along_component = dx_m * along_unit[0] + dy_m * along_unit[1]
    return DisplacementDecomposition(normal_azimuth_deg, normal_component, along_component)


def evaluate_candidate_pair(crest1: dict[str, Any], crest2: dict[str, Any]) -> dict[str, Any]:
    """Section 13's QA fields for ONE (epoch1_crest, epoch2_crest) pair.
    Never decides match status by itself -- see `match_crests_within_tile`
    for the actual (global, ambiguity-aware) assignment."""

    dx, dy = crest2["x"] - crest1["x"], crest2["y"] - crest1["y"]
    straight_distance_m = float(math.hypot(dx, dy))
    decomposition = decompose_displacement(dx, dy, crest1["crest_azimuth_deg"])
    orientation_difference_deg = _angular_difference_undirected(
        crest1["crest_azimuth_deg"], crest2["crest_azimuth_deg"]
    )
    wavelength1, wavelength2 = crest1["wavelength_m"], crest2["wavelength_m"]
    wavelength_ratio = wavelength2 / wavelength1 if wavelength1 else float("inf")

    return {
        "epoch1_crest_id": crest1["bedform_id"],
        "epoch2_crest_id": crest2["bedform_id"],
        "straight_distance_m": straight_distance_m,
        "normal_azimuth_deg": decomposition.normal_azimuth_deg,
        "normal_displacement_m": decomposition.normal_component_m,
        "absolute_normal_displacement_m": abs(decomposition.normal_component_m),
        "along_crest_displacement_m": decomposition.along_crest_component_m,
        "orientation_difference_deg": orientation_difference_deg,
        "wavelength_1_m": wavelength1,
        "wavelength_2_m": wavelength2,
        "wavelength_ratio": wavelength_ratio,
        "wavelength_difference_m": wavelength2 - wavelength1,
    }


def _passes_base_gates(pair: dict[str, Any]) -> str | None:
    """Returns None if the pair passes every base gate, else the single
    rejection reason (checked in a fixed order so it is reproducible)."""

    if pair["straight_distance_m"] > MAX_SEARCH_RADIUS_M:
        return REJECTED_OUTSIDE_SEARCH_RADIUS
    if abs(pair["along_crest_displacement_m"]) > MAX_ALONG_CREST_OFFSET_M:
        return REJECTED_ALONG_CREST_OFFSET_TOO_LARGE
    if pair["orientation_difference_deg"] > ORIENTATION_TOLERANCE_DEG:
        return REJECTED_ORIENTATION_INCOMPATIBLE
    lo, hi = WAVELENGTH_RATIO_TOLERANCE
    if not (lo <= pair["wavelength_ratio"] <= hi):
        return REJECTED_WAVELENGTH_SCALE_INCONSISTENT
    return None


def _classify_support(pair: dict[str, Any]) -> str:
    lo, hi = HIGH_SUPPORT_WAVELENGTH_RATIO
    high_support = (
        pair["orientation_difference_deg"] <= HIGH_SUPPORT_ORIENTATION_DEG
        and lo <= pair["wavelength_ratio"] <= hi
        and pair["straight_distance_m"] <= MAX_SEARCH_RADIUS_M * HIGH_SUPPORT_DISTANCE_FRACTION
    )
    return MATCHED_HIGH_SUPPORT if high_support else MATCHED_WITH_LIMITATIONS


def match_crests_within_tile(
    tile_id: str, epoch1_crests: list[dict[str, Any]], epoch2_crests: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Section 12-14: matches crests already scoped to ONE tile. Returns
    (all_candidates, canonical_matches). `all_candidates` records every
    pair within `MAX_SEARCH_RADIUS_M` regardless of outcome (Section 13:
    every proposed pair carries a match_status/rejection_reason).
    `canonical_matches` holds only the accepted MATCHED_HIGH_SUPPORT /
    MATCHED_WITH_LIMITATIONS rows.

    Assignment is global, not a per-epoch1-crest independent nearest-
    neighbour pick (which could silently double-claim one epoch2 crest for
    two different epoch1 crests): the globally closest still-available
    gate-passing pair is considered first. If any OTHER still-available
    pair sharing either of its endpoints lies within `AMBIGUITY_MARGIN_M`,
    the entire cluster is rejected and BOTH endpoints are retired
    entirely from further matching in this tile (`AMBIGUOUS_NO_CANONICAL_
    MATCH`) -- a crest is never given a later 'second chance' at a more
    distant candidate once its closest option was ambiguous, since that
    would mean silently preferring a worse-supported pair over a
    contested closer one. Otherwise the pair is accepted and both
    endpoints are likewise retired (claimed)."""

    all_pairs: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for c1 in epoch1_crests:
        for c2 in epoch2_crests:
            pair = evaluate_candidate_pair(c1, c2)
            pair["tile_id"] = tile_id
            if pair["straight_distance_m"] > MAX_SEARCH_RADIUS_M:
                pair["match_status"] = None
                pair["rejection_reason"] = REJECTED_OUTSIDE_SEARCH_RADIUS
                all_pairs.append(pair)
                continue
            reason = _passes_base_gates(pair)
            if reason is not None:
                pair["match_status"] = None
                pair["rejection_reason"] = reason
                all_pairs.append(pair)
                continue
            candidates.append(pair)

    remaining = {id(p): p for p in candidates}
    resolved: dict[int, tuple[str | None, str | None]] = {}

    while remaining:
        best = min(remaining.values(), key=lambda p: p["straight_distance_m"])
        e1, e2 = best["epoch1_crest_id"], best["epoch2_crest_id"]
        cluster = [
            p
            for p in remaining.values()
            if (p["epoch1_crest_id"] == e1 or p["epoch2_crest_id"] == e2)
            and p["straight_distance_m"] - best["straight_distance_m"] <= AMBIGUITY_MARGIN_M
        ]
        # Every remaining pair whose endpoint is about to be retired (whether accepted or
        # thrown out as ambiguous) -- accepting `best` also removes any OTHER pair competing
        # for the same endpoint, never leaving a stale duplicate-endpoint candidate behind.
        same_endpoint = [
            p
            for p in remaining.values()
            if p["epoch1_crest_id"] == e1 or p["epoch2_crest_id"] == e2
        ]
        if len(cluster) > 1:
            for pair in same_endpoint:
                status = AMBIGUOUS_NO_CANONICAL_MATCH if pair in cluster else None
                resolved[id(pair)] = (status, REJECTED_AMBIGUOUS)
                del remaining[id(pair)]
            continue

        resolved[id(best)] = (_classify_support(best), None)
        del remaining[id(best)]
        for pair in same_endpoint:
            if id(pair) in remaining:
                resolved[id(pair)] = (None, REJECTED_NOT_BEST_CANDIDATE)
                del remaining[id(pair)]

    canonical_matches: list[dict[str, Any]] = []
    for pair in candidates:
        status, reason = resolved.get(id(pair), (None, REJECTED_NO_CANDIDATE))
        pair["match_status"] = status
        pair["rejection_reason"] = reason
        all_pairs.append(pair)
        if status in (MATCHED_HIGH_SUPPORT, MATCHED_WITH_LIMITATIONS):
            canonical_matches.append(pair)

    return all_pairs, canonical_matches


def compute_apparent_displacement_rate(normal_displacement_m: float, elapsed_years: float) -> float:
    """Section 15. Purely descriptive -- see `APPARENT_RATE_DISCLAIMER`,
    which every caller must surface alongside this value."""

    if elapsed_years <= 0:
        raise ValueError(f"elapsed_years must be positive, got {elapsed_years}")
    return normal_displacement_m / elapsed_years


def sample_dod_around_matched_crest(
    dod_array: np.ndarray,
    transform: rasterio.Affine,
    *,
    x_m: float,
    y_m: float,
    normal_azimuth_deg: float,
    flank_offset_m: float = 15.0,
) -> dict[str, Any]:
    """Section 16: OPTIONAL, purely descriptive DoD context sampled around
    an ALREADY-MATCHED crest -- called only after `match_crests_within_
    tile` has decided the match. Samples the DoD (nearest-cell) at the
    crest position and at `flank_offset_m` either side along the crest's
    own normal axis. Never requires a dipole pattern; never used to
    accept/reject a match."""

    normal_rad = math.radians(normal_azimuth_deg)
    dx, dy = math.sin(normal_rad) * flank_offset_m, math.cos(normal_rad) * flank_offset_m

    def _sample(x: float, y: float) -> float | None:
        row, col = rasterio.transform.rowcol(transform, x, y)
        if 0 <= row < dod_array.shape[0] and 0 <= col < dod_array.shape[1]:
            value = dod_array[row, col]
            return float(value) if np.isfinite(value) else None
        return None

    crest_value = _sample(x_m, y_m)
    flank_a_value = _sample(x_m + dx, y_m + dy)
    flank_b_value = _sample(x_m - dx, y_m - dy)
    magnitude = None
    if flank_a_value is not None and flank_b_value is not None:
        magnitude = abs(flank_a_value - flank_b_value)

    return {
        "dod_at_crest_m": crest_value,
        "dod_flank_a_m": flank_a_value,
        "dod_flank_b_m": flank_b_value,
        "dod_flank_difference_magnitude_m": magnitude,
        "note": "Descriptive local DoD context only -- a perfect raising/lowering dipole across "
        "the crest is NOT required, and this evidence never determines match acceptance.",
    }

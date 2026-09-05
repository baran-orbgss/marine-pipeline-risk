"""Official freespan spatial evidence recovery + canonical route reconciliation (MAR-014A).

Scope -- read before touching this module
--------------------------------------------
The Ithaca Energy (UK) Limited "Pipelines and Umbilical Comparative
Assessment" (April 2020), Appendix B Table B.1, tabulates explicit
freespan survey KP/Easting/Northing/length/height for the PL854-PL855
export/methanol corridor across 2012, 2014, and 2018 -- unlike MAR-014's
generic 2018 benchmark, these ARE spatially resolved events. Canonical
scientific role: `HISTORICAL_SPATIAL_FREESPAN_CORRIDOR_EVIDENCE`. This
module recovers that evidence, empirically determines its (undocumented)
source CRS, and reconciles it onto the TRUE canonical NSTA route -- it
never computes exposure probability, freespan probability, scour depth,
VIV, fatigue, an arbitrary risk score, or an ML validation metric.

Corridor scope, never single-line attribution (Section 2)
------------------------------------------------------------
Table B.1's own scope is the PIGGYBACKED PL854/PL855 corridor, not PL854
alone -- `asset_scope = PL854_PL855_PIGGYBACK_CORRIDOR` and
`individual_line_attribution = UNRESOLVED` on every event, always. No
event is ever silently attributed to PL854 alone.

Source CRS is undocumented -- inferred, never assumed (Sections 7-9)
-------------------------------------------------------------------------
Table B.1 gives Easting/Northing without stating a CRS/datum. This module
never claims the source states a CRS. Two candidates are evaluated
empirically by transforming each candidate's coordinates to the canonical
working CRS and measuring point-to-route distance and a source-KP-vs-
canonical-chainage linear fit; a candidate is accepted ONLY if it clears
ALL of: median/max endpoint-to-route distance within strict bounds, a
materially smaller residual than the alternative, a coherent near-linear
KP-chainage relation, and consistent orientation across the dataset
(`select_working_crs` raises `CRSReconciliationError` and refuses to
silently choose a CRS otherwise). The linear fit is a QA diagnostic only
-- it is NEVER used to place events (`project_events_to_canonical_route`
always re-projects the actual transformed coordinates onto the real
route).

Survey KP direction is verified, never assumed (Sections 11-12)
-----------------------------------------------------------------------
Source survey KP is never assumed to share MAR-004's canonical NSTA
chainage direction -- `source_survey_kp_direction_relative_to_canonical`
is derived from the empirical fit's own sign, and a source Start
coordinate is never called the canonical route start (canonical
chainage min/max are derived independently per event, never inherited
from the source's own Start/End labelling).

Route-conforming geometry, never a straight chord (Section 13)
--------------------------------------------------------------------
Each event's canonical geometry is the TRUE route substring between its
own `canonical_chainage_min_m`/`canonical_chainage_max_m` -- never a
straight line between the two (possibly noisy) transformed endpoints.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import LineString, Point
from shapely.ops import substring

from marine_engine.preprocessing.chainage import format_kp_label, project_point_to_route

# --- Fixed scientific constants (Sections 2, 7-9 -- do not change) ------------------

SCIENTIFIC_ROLE = "HISTORICAL_SPATIAL_FREESPAN_CORRIDOR_EVIDENCE"
ASSET_SCOPE = "PL854_PL855_PIGGYBACK_CORRIDOR"
INDIVIDUAL_LINE_ATTRIBUTION = "UNRESOLVED"

CANDIDATE_CRS_A_EPSG = 23031  # ED50 / UTM zone 31N -- the working hypothesis
CANDIDATE_CRS_B_EPSG = 32631  # WGS84 / UTM zone 31N -- the canonical working CRS itself
CANDIDATE_CRS_EPSG_CODES: tuple[int, ...] = (CANDIDATE_CRS_A_EPSG, CANDIDATE_CRS_B_EPSG)

SOURCE_CRS_STATUS = "CRS_INFERRED_FROM_SPATIAL_CONSISTENCY_NOT_SOURCE_STATED"

# Acceptance guard thresholds (Section 9) -- project QA heuristics, not
# physical constants.
MAX_ACCEPTABLE_MEDIAN_DISTANCE_M = 50.0
MAX_ACCEPTABLE_MAX_DISTANCE_M = 150.0
MATERIALLY_SMALLER_FACTOR = 0.5  # accepted candidate's median must be < this x the alternative's
MIN_ACCEPTABLE_R_SQUARED = 0.98

REVERSED = "REVERSED"
SAME_AS_CANONICAL = "SAME_AS_CANONICAL"

# Official Anglia A NUI position (Section 10) -- independent diagnostic
# context only, never used to move or warp source events.
ANGLIA_A_WGS84_LAT = 53.3676167
ANGLIA_A_WGS84_LON = 1.6516667


class CRSReconciliationError(Exception):
    """No candidate CRS cleared the acceptance guard -- refuses to silently choose one."""


class AngliaFreespanValidationError(Exception):
    """A gross (not rounding-scale) inconsistency was found reconciling the source evidence."""


# --- CRS candidate diagnostic (Sections 7-9) --------------------------------------------


@dataclass(frozen=True)
class CrsCandidateDiagnostics:
    """One candidate CRS's empirical fit against the true canonical route."""

    candidate_epsg: int
    endpoint_count: int
    distance_min_m: float
    distance_median_m: float
    distance_p95_m: float
    distance_max_m: float
    fit_slope: float
    fit_intercept_m: float
    fit_r_squared: float
    residual_median_m: float
    residual_p95_m: float
    residual_max_m: float
    orientation_consistent: bool


def _linear_fit_r_squared(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Simple OLS `y = slope*x + intercept`, plus R^2 and the fitted residuals."""

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r_squared, residuals


def _orientation_consistent(source_kp_m: np.ndarray, chainages_m: np.ndarray) -> bool:
    """Whether the KP-vs-chainage trend has the SAME sign in the low-KP and high-KP
    halves of the dataset -- a genuine cross-check beyond the single overall fit,
    catching a hypothetical direction flip partway through the dataset."""

    order = np.argsort(source_kp_m)
    sorted_kp = source_kp_m[order]
    sorted_chainage = chainages_m[order]
    midpoint = len(sorted_kp) // 2
    if midpoint < 2 or (len(sorted_kp) - midpoint) < 2:
        return True  # too few points to meaningfully split -- never spuriously fail

    slope_low, _, _, _ = _linear_fit_r_squared(sorted_kp[:midpoint], sorted_chainage[:midpoint])
    slope_high, _, _, _ = _linear_fit_r_squared(sorted_kp[midpoint:], sorted_chainage[midpoint:])
    return bool(np.sign(slope_low) == np.sign(slope_high))


def evaluate_crs_candidate(
    events_df: pd.DataFrame, route: LineString, working_crs: str, candidate_epsg: int
) -> CrsCandidateDiagnostics:
    """Transform every event Start/End endpoint under `candidate_epsg`, project each
    onto the TRUE canonical route, and report distance + KP-vs-chainage fit statistics.

    The route projection itself is real and independent of the fit -- the
    fit is a QA diagnostic only, never used to place events (Section 8).
    """

    eastings = np.concatenate(
        [
            events_df["source_easting_start_m"].to_numpy(),
            events_df["source_easting_end_m"].to_numpy(),
        ]
    )
    northings = np.concatenate(
        [
            events_df["source_northing_start_m"].to_numpy(),
            events_df["source_northing_end_m"].to_numpy(),
        ]
    )
    source_kp_m = np.concatenate(
        [
            events_df["source_survey_kp_start_km"].to_numpy() * 1000.0,
            events_df["source_survey_kp_end_km"].to_numpy() * 1000.0,
        ]
    )

    transformer = pyproj.Transformer.from_crs(f"EPSG:{candidate_epsg}", working_crs, always_xy=True)
    x_transformed, y_transformed = transformer.transform(eastings, northings)

    distances = np.empty(len(x_transformed))
    chainages = np.empty(len(x_transformed))
    for i, (x, y) in enumerate(zip(x_transformed, y_transformed, strict=True)):
        projection = project_point_to_route(route, Point(float(x), float(y)))
        distances[i] = projection.distance_m
        chainages[i] = projection.chainage_m

    slope, intercept, r_squared, residuals = _linear_fit_r_squared(source_kp_m, chainages)
    abs_residuals = np.abs(residuals)

    return CrsCandidateDiagnostics(
        candidate_epsg=candidate_epsg,
        endpoint_count=len(distances),
        distance_min_m=float(distances.min()),
        distance_median_m=float(np.median(distances)),
        distance_p95_m=float(np.percentile(distances, 95)),
        distance_max_m=float(distances.max()),
        fit_slope=slope,
        fit_intercept_m=intercept,
        fit_r_squared=r_squared,
        residual_median_m=float(np.median(abs_residuals)),
        residual_p95_m=float(np.percentile(abs_residuals, 95)),
        residual_max_m=float(abs_residuals.max()),
        orientation_consistent=_orientation_consistent(source_kp_m, chainages),
    )


def select_working_crs(
    diagnostics_by_epsg: dict[int, CrsCandidateDiagnostics],
) -> tuple[int, dict[str, bool]]:
    """Accept `CANDIDATE_CRS_A_EPSG` ONLY if every Section 9 guard clears; else hard-fail.

    Never silently chooses a CRS -- ambiguous or failing diagnostics raise
    `CRSReconciliationError` with the full diagnostics attached for
    external review.
    """

    accepted = diagnostics_by_epsg[CANDIDATE_CRS_A_EPSG]
    alternative = diagnostics_by_epsg[CANDIDATE_CRS_B_EPSG]

    checks = {
        "median_distance_within_50m": accepted.distance_median_m
        <= MAX_ACCEPTABLE_MEDIAN_DISTANCE_M,
        "max_distance_within_150m": accepted.distance_max_m <= MAX_ACCEPTABLE_MAX_DISTANCE_M,
        "materially_smaller_than_alternative": (
            accepted.distance_median_m < MATERIALLY_SMALLER_FACTOR * alternative.distance_median_m
        ),
        "coherent_near_linear_fit": accepted.fit_r_squared >= MIN_ACCEPTABLE_R_SQUARED,
        "orientation_consistent": accepted.orientation_consistent,
    }

    if not all(checks.values()):
        raise CRSReconciliationError(
            f"No candidate CRS cleared the acceptance guard -- checks: {checks}; "
            f"EPSG:{CANDIDATE_CRS_A_EPSG} diagnostics: {accepted}; "
            f"EPSG:{CANDIDATE_CRS_B_EPSG} diagnostics: {alternative}"
        )
    return CANDIDATE_CRS_A_EPSG, checks


def compute_anglia_a_anchor_distance_m(
    highest_source_kp_easting_m: float,
    highest_source_kp_northing_m: float,
    accepted_epsg: int,
) -> float:
    """Geodesic distance (m) from the highest-source-KP endpoint (transformed under the
    accepted CRS) to the official WGS84 Anglia A NUI position -- independent context
    only, never used to move or warp source events (Section 10)."""

    transformer = pyproj.Transformer.from_crs(f"EPSG:{accepted_epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(highest_source_kp_easting_m, highest_source_kp_northing_m)
    geod = pyproj.Geod(ellps="WGS84")
    _fwd_azimuth, _back_azimuth, distance_m = geod.inv(
        lon, lat, ANGLIA_A_WGS84_LON, ANGLIA_A_WGS84_LAT
    )
    return float(distance_m)


# --- Canonical event projection (Sections 11-14) ----------------------------------------

FREESPAN_SPATIAL_EVIDENCE_COLUMNS = (
    "event_id",
    "survey_year",
    "asset_scope",
    "individual_line_attribution",
    "scientific_role",
    "source_survey_kp_start_km",
    "source_survey_kp_end_km",
    "source_easting_start_m",
    "source_northing_start_m",
    "source_easting_end_m",
    "source_northing_end_m",
    "source_length_m",
    "source_height_m",
    "source_comment",
    "source_page",
    "source_table",
    "inferred_source_crs_epsg",
    "source_crs_status",
    "canonical_chainage_a_m",
    "canonical_chainage_b_m",
    "canonical_chainage_min_m",
    "canonical_chainage_max_m",
    "canonical_kp_min",
    "canonical_kp_max",
    "canonical_mid_chainage_m",
    "canonical_mid_kp",
    "endpoint_a_route_distance_m",
    "endpoint_b_route_distance_m",
    "projected_route_interval_length_m",
    "source_kp_difference_length_m",
    "absolute_length_difference_m",
    "relative_length_difference_pct",
)

# A gross (not rounding-scale) length mismatch -- Section 14 never forces
# equality, but a mismatch beyond this is worth surfacing loudly rather
# than silently accepting as "close enough". Public (not `_`-prefixed): the
# reconciliation metadata (Section 24) reports these thresholds directly.
GROSS_LENGTH_MISMATCH_ABSOLUTE_M = 50.0
GROSS_LENGTH_MISMATCH_RELATIVE_PCT = 100.0


def project_events_to_canonical_route(
    events_df: pd.DataFrame, route: LineString, working_crs: str, accepted_epsg: int
) -> pd.DataFrame:
    """Reconcile every source event onto the TRUE canonical NSTA route (Sections 11-14).

    Never calls the source Start the canonical route start -- canonical
    chainage min/max are derived independently, since survey and canonical
    direction differ (confirmed empirically, never hard-coded in advance).
    Never draws a straight chord: callers use `canonical_chainage_min_m`/
    `canonical_chainage_max_m` with `shapely.ops.substring` on the real
    route for geometry.
    """

    transformer = pyproj.Transformer.from_crs(f"EPSG:{accepted_epsg}", working_crs, always_xy=True)
    records = []
    for _, row in events_df.iterrows():
        xa, ya = transformer.transform(
            row["source_easting_start_m"], row["source_northing_start_m"]
        )
        xb, yb = transformer.transform(row["source_easting_end_m"], row["source_northing_end_m"])
        projection_a = project_point_to_route(route, Point(float(xa), float(ya)))
        projection_b = project_point_to_route(route, Point(float(xb), float(yb)))

        chainage_a = projection_a.chainage_m
        chainage_b = projection_b.chainage_m
        chainage_min = min(chainage_a, chainage_b)
        chainage_max = max(chainage_a, chainage_b)
        mid_chainage = (chainage_min + chainage_max) / 2.0

        projected_interval_length = chainage_max - chainage_min
        source_kp_diff_length = (
            abs(row["source_survey_kp_end_km"] - row["source_survey_kp_start_km"]) * 1000.0
        )
        absolute_length_diff = abs(row["source_length_m"] - projected_interval_length)
        relative_length_diff_pct = (
            100.0 * absolute_length_diff / row["source_length_m"]
            if row["source_length_m"] > 0
            else None
        )

        if (
            absolute_length_diff > GROSS_LENGTH_MISMATCH_ABSOLUTE_M
            and (relative_length_diff_pct or 0.0) > GROSS_LENGTH_MISMATCH_RELATIVE_PCT
        ):
            raise AngliaFreespanValidationError(
                f"event {row['event_id']}: projected route interval length "
                f"{projected_interval_length:.2f} m is grossly inconsistent with the "
                f"source reported length {row['source_length_m']:.2f} m "
                f"(absolute diff {absolute_length_diff:.2f} m, "
                f"{relative_length_diff_pct:.1f}%)"
            )

        records.append(
            {
                "event_id": row["event_id"],
                "survey_year": row["survey_year"],
                "asset_scope": ASSET_SCOPE,
                "individual_line_attribution": INDIVIDUAL_LINE_ATTRIBUTION,
                "scientific_role": SCIENTIFIC_ROLE,
                "source_survey_kp_start_km": row["source_survey_kp_start_km"],
                "source_survey_kp_end_km": row["source_survey_kp_end_km"],
                "source_easting_start_m": row["source_easting_start_m"],
                "source_northing_start_m": row["source_northing_start_m"],
                "source_easting_end_m": row["source_easting_end_m"],
                "source_northing_end_m": row["source_northing_end_m"],
                "source_length_m": row["source_length_m"],
                "source_height_m": row["source_height_m"],
                "source_comment": row.get("source_comment"),
                "source_page": row.get("source_page"),
                "source_table": row.get("source_table"),
                "inferred_source_crs_epsg": accepted_epsg,
                "source_crs_status": SOURCE_CRS_STATUS,
                "canonical_chainage_a_m": chainage_a,
                "canonical_chainage_b_m": chainage_b,
                "canonical_chainage_min_m": chainage_min,
                "canonical_chainage_max_m": chainage_max,
                "canonical_kp_min": format_kp_label(chainage_min),
                "canonical_kp_max": format_kp_label(chainage_max),
                "canonical_mid_chainage_m": mid_chainage,
                "canonical_mid_kp": format_kp_label(mid_chainage),
                "endpoint_a_route_distance_m": projection_a.distance_m,
                "endpoint_b_route_distance_m": projection_b.distance_m,
                "projected_route_interval_length_m": projected_interval_length,
                "source_kp_difference_length_m": source_kp_diff_length,
                "absolute_length_difference_m": absolute_length_diff,
                "relative_length_difference_pct": relative_length_diff_pct,
            }
        )
    return pd.DataFrame(records, columns=list(FREESPAN_SPATIAL_EVIDENCE_COLUMNS))


def classify_survey_direction(fit_slope: float) -> str:
    """`REVERSED` for a negative KP-vs-chainage slope, else `SAME_AS_CANONICAL` --
    derived from the empirical fit, never hard-coded in advance (Section 11)."""

    return REVERSED if fit_slope < 0 else SAME_AS_CANONICAL


def build_event_geometries(events_df: pd.DataFrame, route: LineString) -> list[Any]:
    """The TRUE canonical route substring for every event -- never a straight chord
    between the two transformed endpoints (Section 13)."""

    return [
        substring(
            route,
            row["canonical_chainage_min_m"],
            row["canonical_chainage_max_m"],
            normalized=False,
        )
        for _, row in events_df.iterrows()
    ]


# --- Segment event counts (Section 18) --------------------------------------------------

SEGMENT_FREESPAN_COUNTS_COLUMNS = (
    "hydro_pair_id",
    "freespan_2018_count",
    "freespan_2018_total_length_m",
    "freespan_2018_max_height_m",
    "freespan_2018_max_length_m",
    "any_2018_freespan",
)


def compute_segment_freespan_counts(
    events_2018_df: pd.DataFrame, segment_bounds: list[dict[str, Any]]
) -> pd.DataFrame:
    """Per hydro-pair segment: count/length/height of 2018 events overlapping its
    chainage range -- an event crossing a segment boundary is counted in EVERY
    segment its own [min,max] interval geometrically overlaps, never assigned
    wholly to one segment by midpoint alone (Section 18)."""

    records = []
    for segment in segment_bounds:
        seg_start = segment["start_chainage_m"]
        seg_end = segment["end_chainage_m"]
        overlapping = events_2018_df[
            (events_2018_df["canonical_chainage_min_m"] < seg_end)
            & (events_2018_df["canonical_chainage_max_m"] > seg_start)
        ]
        records.append(
            {
                "hydro_pair_id": segment["hydro_pair_id"],
                "freespan_2018_count": int(len(overlapping)),
                "freespan_2018_total_length_m": float(overlapping["source_length_m"].sum())
                if len(overlapping)
                else 0.0,
                "freespan_2018_max_height_m": float(overlapping["source_height_m"].max())
                if len(overlapping)
                else None,
                "freespan_2018_max_length_m": float(overlapping["source_length_m"].max())
                if len(overlapping)
                else None,
                "any_2018_freespan": bool(len(overlapping) > 0),
            }
        )
    return pd.DataFrame(records, columns=list(SEGMENT_FREESPAN_COUNTS_COLUMNS))

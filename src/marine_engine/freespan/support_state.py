"""Measured support state classification and free-span interval extraction
(MAR-025 Sections 8-10, repaired by MAR-025A).

Zero dependency on any specific project or dataset. MEASURED GEOMETRY and SOURCE INTERPRETATION
are orthogonal evidence dimensions (MAR-025A Section 2): `classify_measured_support_state`
depends ONLY on canonical clearance and a defensible classification threshold and cannot accept
a source-interpretation flag at all -- source-interpreted free-span evidence is tracked as its
own separate field by the caller (e.g. `source_interpreted_free_span_present`), never fed into
this classifier, so it can neither erase a valid geometric state nor split/bridge/create a
measured interval. Interval extraction (Section 9) groups ONLY contiguous `MEASURED_UNSUPPORTED`
samples, and never bridges an along-route measurement gap wider than an explicit, caller-supplied
maximum (Section 10).

A discrete sample run's extent is NOT automatically an exact physical free-span length
(MAR-025A Section 8): `interval_boundary_semantics` records how the interval's boundary was
actually established, defaulting to `UNSUPPORTED_SAMPLE_RUN_EXTENT` -- `EXPLICIT_BOUNDARY_SAMPLES`
is never silently assigned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# --- Section 8: measured support state vocabulary -- geometry only, never SAFE/UNSAFE/HIGH RISK/
# LOW RISK, and never a source-interpretation state (MAR-025A Section 3) ----------------------

MEASURED_SUPPORTED = "MEASURED_SUPPORTED"
MEASURED_UNSUPPORTED = "MEASURED_UNSUPPORTED"
MEASURED_SUPPORT_TRANSITION = "MEASURED_SUPPORT_TRANSITION"
MEASURED_SUPPORT_STATE_UNRESOLVED = "MEASURED_SUPPORT_STATE_UNRESOLVED"

MEASURED_SUPPORT_STATES = frozenset(
    {
        MEASURED_SUPPORTED,
        MEASURED_UNSUPPORTED,
        MEASURED_SUPPORT_TRANSITION,
        MEASURED_SUPPORT_STATE_UNRESOLVED,
    }
)


def classify_measured_support_state(
    clearance_m: float | None,
    *,
    classification_threshold_m: float | None,
) -> str:
    """One sample's measured support state (Section 8, MAR-025A Section 3). Depends ONLY on
    `clearance_m` and `classification_threshold_m` -- there is structurally no parameter through
    which source-interpreted evidence could influence this result."""

    if classification_threshold_m is None or clearance_m is None or pd.isna(clearance_m):
        return MEASURED_SUPPORT_STATE_UNRESOLVED
    if clearance_m > classification_threshold_m:
        return MEASURED_UNSUPPORTED
    if clearance_m < -classification_threshold_m:
        return MEASURED_SUPPORTED
    return MEASURED_SUPPORT_TRANSITION


# --- MAR-025A Section 5: geometry vs. source-interpretation agreement -- purely descriptive,
# never confidence/validation/probability, and never fed back into measured geometry -----------

AGREES_UNSUPPORTED = "AGREES_UNSUPPORTED"
SOURCE_ONLY_FREESPAN_EVIDENCE = "SOURCE_ONLY_FREESPAN_EVIDENCE"
GEOMETRY_ONLY_UNSUPPORTED_EVIDENCE = "GEOMETRY_ONLY_UNSUPPORTED_EVIDENCE"
NO_FREESPAN_EVIDENCE = "NO_FREESPAN_EVIDENCE"
COMPARISON_NOT_AVAILABLE = "COMPARISON_NOT_AVAILABLE"

GEOMETRY_VS_SOURCE_INTERPRETATION_STATUSES = frozenset(
    {
        AGREES_UNSUPPORTED,
        SOURCE_ONLY_FREESPAN_EVIDENCE,
        GEOMETRY_ONLY_UNSUPPORTED_EVIDENCE,
        NO_FREESPAN_EVIDENCE,
        COMPARISON_NOT_AVAILABLE,
    }
)


def compute_geometry_vs_source_interpretation_status(
    measured_support_state: str, source_interpreted_free_span_present: bool
) -> str:
    """MAR-025A Section 5: a purely descriptive comparison of the two orthogonal evidence
    dimensions -- read-only, never used to alter `measured_support_state` or
    `source_interpreted_free_span_present` themselves. `COMPARISON_NOT_AVAILABLE` whenever
    geometry itself could not be resolved (no defensible threshold/clearance), regardless of the
    source flag, since "agreement" is undefined without a resolved geometric answer."""

    if measured_support_state == MEASURED_SUPPORT_STATE_UNRESOLVED:
        return COMPARISON_NOT_AVAILABLE
    geometrically_unsupported = measured_support_state == MEASURED_UNSUPPORTED
    if geometrically_unsupported and source_interpreted_free_span_present:
        return AGREES_UNSUPPORTED
    if geometrically_unsupported and not source_interpreted_free_span_present:
        return GEOMETRY_ONLY_UNSUPPORTED_EVIDENCE
    if not geometrically_unsupported and source_interpreted_free_span_present:
        return SOURCE_ONLY_FREESPAN_EVIDENCE
    return NO_FREESPAN_EVIDENCE


# --- Section 9: free-span interval extraction --------------------------------------------------

EVIDENCE_TYPE_MEASURED_FREE_SPAN_INTERVAL = "MEASURED_FREE_SPAN_INTERVAL_FROM_CLASSIFIED_CLEARANCE"

# Section 10: recorded on an interval whose start or end followed an along-route measurement
# gap wider than the caller-supplied maximum -- proof that two observations were never silently
# bridged across an unsurveyed gap and called one free span.
SPAN_SPLIT_BY_MEASUREMENT_GAP = "SPAN_SPLIT_BY_MEASUREMENT_GAP"

# --- MAR-025A Sections 8-11: interval boundary semantics -- a discrete unsupported sample run's
# extent is NOT automatically an exact physical free-span length ------------------------------

EXPLICIT_BOUNDARY_SAMPLES = "EXPLICIT_BOUNDARY_SAMPLES"
UNSUPPORTED_SAMPLE_RUN_EXTENT = "UNSUPPORTED_SAMPLE_RUN_EXTENT"
BOUNDARY_BRACKETED_BY_ADJACENT_MEASUREMENTS = "BOUNDARY_BRACKETED_BY_ADJACENT_MEASUREMENTS"

INTERVAL_BOUNDARY_SEMANTICS_TYPES = frozenset(
    {
        EXPLICIT_BOUNDARY_SAMPLES,
        UNSUPPORTED_SAMPLE_RUN_EXTENT,
        BOUNDARY_BRACKETED_BY_ADJACENT_MEASUREMENTS,
    }
)

SPAN_LENGTH_SEMANTICS_NOTE = (
    "span_length_m equals unsupported_sample_run_extent_m -- see interval_boundary_semantics; "
    "never an exact physical free-span length unless interval_boundary_semantics is "
    "EXPLICIT_BOUNDARY_SAMPLES"
)

FREE_SPAN_INTERVAL_COLUMNS: tuple[str, ...] = (
    "span_id",
    "asset_id",
    "start_chainage_m",
    "end_chainage_m",
    "span_length_m",
    "span_length_semantics",
    "unsupported_sample_run_start_chainage_m",
    "unsupported_sample_run_end_chainage_m",
    "unsupported_sample_run_extent_m",
    "interval_boundary_semantics",
    "maximum_clearance_m",
    "median_clearance_m",
    "sample_count",
    "survey_epoch",
    "evidence_type",
    "classification_threshold_m",
    "vertical_reference_semantics",
    "left_boundary_bracket_start_m",
    "left_boundary_bracket_end_m",
    "right_boundary_bracket_start_m",
    "right_boundary_bracket_end_m",
    "limitations",
)


def extract_measured_free_span_intervals(
    profile_df: pd.DataFrame,
    *,
    asset_id: str,
    max_measurement_gap_m: float,
    survey_epoch: str,
    classification_threshold_m: float | None,
    vertical_reference_semantics: str,
    interval_boundary_semantics: str = UNSUPPORTED_SAMPLE_RUN_EXTENT,
    support_state_column: str = "measured_support_state",
    clearance_column: str = "clearance_m",
    chainage_column: str = "chainage_m",
    span_id_prefix: str | None = None,
) -> pd.DataFrame:
    """Section 9-10: group contiguous `MEASURED_UNSUPPORTED` samples (sorted by chainage) into
    intervals -- based ONLY on `measured_support_state`, so no source-interpretation flag the
    caller may separately track can create, terminate, bridge, or split an interval (MAR-025A
    Section 6: that state simply does not exist in `support_state_column`'s vocabulary any more).
    A sample whose state is anything else (including `MEASURED_SUPPORT_TRANSITION`) ends the
    current interval. A chainage gap exceeding `max_measurement_gap_m` between two consecutive
    unsupported samples ALSO ends the current interval, even though both samples are
    individually `MEASURED_UNSUPPORTED` -- both resulting intervals are flagged
    `SPAN_SPLIT_BY_MEASUREMENT_GAP` (Section 10).

    `interval_boundary_semantics` defaults to `UNSUPPORTED_SAMPLE_RUN_EXTENT` (MAR-025A Section
    9) -- `EXPLICIT_BOUNDARY_SAMPLES` is never silently assigned; the caller must explicitly
    state it. When `BOUNDARY_BRACKETED_BY_ADJACENT_MEASUREMENTS` is requested, the immediately
    adjacent non-unsupported sample on each side (if one exists within `max_measurement_gap_m`)
    is recorded as a bracket showing where the true threshold crossing could lie -- never
    linearly interpolated to a precise crossing point."""

    if interval_boundary_semantics not in INTERVAL_BOUNDARY_SEMANTICS_TYPES:
        raise ValueError(f"unknown interval_boundary_semantics: {interval_boundary_semantics!r}")

    prefix = span_id_prefix or asset_id
    ordered = profile_df.sort_values(chainage_column).reset_index(drop=True)
    n = len(ordered)
    is_unsupported = (ordered[support_state_column] == MEASURED_UNSUPPORTED).to_numpy()
    chainages_arr = ordered[chainage_column].astype(float).to_numpy()

    intervals: list[dict[str, Any]] = []
    run_indices: list[int] = []

    def close_run(flagged: bool) -> None:
        nonlocal run_indices
        if not run_indices:
            return
        rows = ordered.iloc[run_indices]
        chainages = rows[chainage_column].astype(float).tolist()
        clearances = rows[clearance_column].astype(float).tolist()
        start_chainage = min(chainages)
        end_chainage = max(chainages)
        extent = end_chainage - start_chainage
        limitations = [SPAN_SPLIT_BY_MEASUREMENT_GAP] if flagged else []

        left_bracket_start = left_bracket_end = None
        right_bracket_start = right_bracket_end = None
        if interval_boundary_semantics == BOUNDARY_BRACKETED_BY_ADJACENT_MEASUREMENTS:
            first_idx, last_idx = run_indices[0], run_indices[-1]
            if first_idx > 0 and not is_unsupported[first_idx - 1]:
                neighbor_chainage = float(chainages_arr[first_idx - 1])
                if (start_chainage - neighbor_chainage) <= max_measurement_gap_m:
                    left_bracket_start, left_bracket_end = neighbor_chainage, start_chainage
            if last_idx < n - 1 and not is_unsupported[last_idx + 1]:
                neighbor_chainage = float(chainages_arr[last_idx + 1])
                if (neighbor_chainage - end_chainage) <= max_measurement_gap_m:
                    right_bracket_start, right_bracket_end = end_chainage, neighbor_chainage

        intervals.append(
            {
                "span_id": f"{prefix}_SPAN_{len(intervals) + 1:03d}",
                "asset_id": asset_id,
                "start_chainage_m": start_chainage,
                "end_chainage_m": end_chainage,
                "span_length_m": extent,
                "span_length_semantics": SPAN_LENGTH_SEMANTICS_NOTE,
                "unsupported_sample_run_start_chainage_m": start_chainage,
                "unsupported_sample_run_end_chainage_m": end_chainage,
                "unsupported_sample_run_extent_m": extent,
                "interval_boundary_semantics": interval_boundary_semantics,
                "maximum_clearance_m": max(clearances),
                "median_clearance_m": float(np.median(clearances)),
                "sample_count": len(run_indices),
                "survey_epoch": survey_epoch,
                "evidence_type": EVIDENCE_TYPE_MEASURED_FREE_SPAN_INTERVAL,
                "classification_threshold_m": classification_threshold_m,
                "vertical_reference_semantics": vertical_reference_semantics,
                "left_boundary_bracket_start_m": left_bracket_start,
                "left_boundary_bracket_end_m": left_bracket_end,
                "right_boundary_bracket_start_m": right_bracket_start,
                "right_boundary_bracket_end_m": right_bracket_end,
                "limitations": "; ".join(limitations) or None,
            }
        )
        run_indices = []

    this_run_gap_split = False
    prev_chainage: float | None = None
    for i in range(n):
        if is_unsupported[i]:
            chainage = float(chainages_arr[i])
            gap_exceeded = (
                run_indices
                and prev_chainage is not None
                and (chainage - prev_chainage) > max_measurement_gap_m
            )
            if gap_exceeded:
                close_run(True)
                this_run_gap_split = True
            run_indices.append(i)
            prev_chainage = chainage
        else:
            close_run(this_run_gap_split)
            this_run_gap_split = False
            prev_chainage = None
    close_run(this_run_gap_split)

    if not intervals:
        return pd.DataFrame(columns=list(FREE_SPAN_INTERVAL_COLUMNS))
    return pd.DataFrame(intervals, columns=list(FREE_SPAN_INTERVAL_COLUMNS))

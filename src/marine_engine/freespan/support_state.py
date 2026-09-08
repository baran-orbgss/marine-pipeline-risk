"""Measured support state classification and free-span interval extraction
(MAR-025 Sections 8-10).

Zero dependency on any specific project or dataset. Source-interpreted free-span evidence is
kept structurally separate from geometric (clearance-based) inference throughout (Section 8):
`SOURCE_INTERPRETED_FREE_SPAN` is reachable ONLY via an explicit caller-supplied flag, never
inferred from `clearance_m`. Interval extraction (Section 9) groups ONLY contiguous
`MEASURED_UNSUPPORTED` samples -- never source-interpreted evidence -- and never bridges an
along-route measurement gap wider than an explicit, caller-supplied maximum (Section 10).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# --- Section 8: measured support state vocabulary -- never SAFE/UNSAFE/HIGH RISK/LOW RISK ----

MEASURED_SUPPORTED = "MEASURED_SUPPORTED"
MEASURED_UNSUPPORTED = "MEASURED_UNSUPPORTED"
MEASURED_SUPPORT_TRANSITION = "MEASURED_SUPPORT_TRANSITION"
MEASURED_SUPPORT_STATE_UNRESOLVED = "MEASURED_SUPPORT_STATE_UNRESOLVED"
SOURCE_INTERPRETED_FREE_SPAN = "SOURCE_INTERPRETED_FREE_SPAN"

MEASURED_SUPPORT_STATES = frozenset(
    {
        MEASURED_SUPPORTED,
        MEASURED_UNSUPPORTED,
        MEASURED_SUPPORT_TRANSITION,
        MEASURED_SUPPORT_STATE_UNRESOLVED,
        SOURCE_INTERPRETED_FREE_SPAN,
    }
)


def classify_measured_support_state(
    clearance_m: float | None,
    *,
    is_source_interpreted_free_span: bool,
    classification_threshold_m: float | None,
) -> str:
    """One sample's measured support state (Section 8). `clearance_m` is the ONLY numeric
    quantity permitted to drive the geometric BURIED/TRANSITION/UNSUPPORTED distinction.
    `SOURCE_INTERPRETED_FREE_SPAN` is reachable ONLY via the explicit
    `is_source_interpreted_free_span` flag and takes priority over the geometric state, so the
    two concepts (measured geometry vs. source-interpreted evidence) never collapse into one."""

    if is_source_interpreted_free_span:
        return SOURCE_INTERPRETED_FREE_SPAN
    if classification_threshold_m is None or clearance_m is None or pd.isna(clearance_m):
        return MEASURED_SUPPORT_STATE_UNRESOLVED
    if clearance_m > classification_threshold_m:
        return MEASURED_UNSUPPORTED
    if clearance_m < -classification_threshold_m:
        return MEASURED_SUPPORTED
    return MEASURED_SUPPORT_TRANSITION


# --- Section 9: free-span interval extraction --------------------------------------------------

EVIDENCE_TYPE_MEASURED_FREE_SPAN_INTERVAL = "MEASURED_FREE_SPAN_INTERVAL_FROM_CLASSIFIED_CLEARANCE"

# Section 10: recorded on an interval whose start or end followed an along-route measurement
# gap wider than the caller-supplied maximum -- proof that two observations were never silently
# bridged across an unsurveyed gap and called one free span.
SPAN_SPLIT_BY_MEASUREMENT_GAP = "SPAN_SPLIT_BY_MEASUREMENT_GAP"

FREE_SPAN_INTERVAL_COLUMNS: tuple[str, ...] = (
    "span_id",
    "asset_id",
    "start_chainage_m",
    "end_chainage_m",
    "span_length_m",
    "maximum_clearance_m",
    "median_clearance_m",
    "sample_count",
    "survey_epoch",
    "evidence_type",
    "classification_threshold_m",
    "vertical_reference_semantics",
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
    support_state_column: str = "measured_support_state",
    clearance_column: str = "clearance_m",
    chainage_column: str = "chainage_m",
    span_id_prefix: str | None = None,
) -> pd.DataFrame:
    """Section 9-10: group contiguous `MEASURED_UNSUPPORTED` samples (sorted by chainage) into
    intervals. A sample whose state is anything else (including `MEASURED_SUPPORT_TRANSITION` or
    `SOURCE_INTERPRETED_FREE_SPAN`) ends the current interval. A chainage gap exceeding
    `max_measurement_gap_m` between two consecutive unsupported samples ALSO ends the current
    interval, even though both samples are individually `MEASURED_UNSUPPORTED` -- both resulting
    intervals are flagged `SPAN_SPLIT_BY_MEASUREMENT_GAP` (Section 10)."""

    prefix = span_id_prefix or asset_id
    ordered = profile_df.sort_values(chainage_column).reset_index(drop=True)

    intervals: list[dict[str, Any]] = []
    current_rows: list[pd.Series] = []

    def close_run(flagged: bool) -> None:
        if not current_rows:
            return
        chainages = [float(r[chainage_column]) for r in current_rows]
        clearances = [float(r[clearance_column]) for r in current_rows]
        limitations = [SPAN_SPLIT_BY_MEASUREMENT_GAP] if flagged else []
        intervals.append(
            {
                "span_id": f"{prefix}_SPAN_{len(intervals) + 1:03d}",
                "asset_id": asset_id,
                "start_chainage_m": min(chainages),
                "end_chainage_m": max(chainages),
                "span_length_m": max(chainages) - min(chainages),
                "maximum_clearance_m": max(clearances),
                "median_clearance_m": float(np.median(clearances)),
                "sample_count": len(current_rows),
                "survey_epoch": survey_epoch,
                "evidence_type": EVIDENCE_TYPE_MEASURED_FREE_SPAN_INTERVAL,
                "classification_threshold_m": classification_threshold_m,
                "vertical_reference_semantics": vertical_reference_semantics,
                "limitations": "; ".join(limitations) or None,
            }
        )

    this_run_gap_split = False
    prev_chainage: float | None = None
    for _, row in ordered.iterrows():
        chainage = float(row[chainage_column])
        if row[support_state_column] == MEASURED_UNSUPPORTED:
            gap_exceeded = (
                current_rows
                and prev_chainage is not None
                and (chainage - prev_chainage) > max_measurement_gap_m
            )
            if gap_exceeded:
                close_run(True)
                current_rows = []
                this_run_gap_split = True
            current_rows.append(row)
            prev_chainage = chainage
        else:
            close_run(this_run_gap_split)
            current_rows = []
            this_run_gap_split = False
            prev_chainage = None
    close_run(this_run_gap_split)

    if not intervals:
        return pd.DataFrame(columns=list(FREE_SPAN_INTERVAL_COLUMNS))
    return pd.DataFrame(intervals, columns=list(FREE_SPAN_INTERVAL_COLUMNS))

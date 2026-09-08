"""Generic support-loss susceptibility scenario screening (MAR-025 Sections 11-15).

Kept structurally separate from Section 9's OBSERVED/MEASURED free-span extraction (Section 11):
this module only ever asks "could an operator-supplied seabed-lowering scenario create or
extend an unsupported interval", never "what is currently unsupported" (that is
`support_state.extract_measured_free_span_intervals`). The pipe is never moved vertically here
-- `compute_scenario_pipe_underside_clearance_m` takes only a clearance value and a lowering
magnitude, so there is structurally no channel through which a pipe-response model could be
silently introduced (Section 13's stated limitation: pipe vertical position is fixed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from marine_engine.freespan import support_state
from marine_engine.freespan.clearance import compute_support_clearance_margin_m

# --- Section 12: seabed-lowering scenario evidence -- never invented from an unrelated model --

MEASURED_REPEAT_MBES_LOWERING = "MEASURED_REPEAT_MBES_LOWERING"
ENGINEERING_SCENARIO = "ENGINEERING_SCENARIO"
EXTERNALLY_SUPPLIED_MORPHODYNAMIC_SCENARIO = "EXTERNALLY_SUPPLIED_MORPHODYNAMIC_SCENARIO"

ALLOWED_LOWERING_EVIDENCE_TYPES = frozenset(
    {
        MEASURED_REPEAT_MBES_LOWERING,
        ENGINEERING_SCENARIO,
        EXTERNALLY_SUPPLIED_MORPHODYNAMIC_SCENARIO,
    }
)


@dataclass(frozen=True)
class SeabedLoweringScenarioInput:
    """A defensible, explicitly-typed seabed-lowering magnitude (Section 12). Never accepted as
    a bare float. Never derived from scour-onset screening, sediment mobility, sand-wave
    asymmetry, or a single bathymetry epoch -- those are not in `ALLOWED_LOWERING_EVIDENCE_TYPES`
    and this dataclass rejects anything else outright."""

    seabed_lowering_m: float
    evidence_type: str

    def __post_init__(self) -> None:
        if self.evidence_type not in ALLOWED_LOWERING_EVIDENCE_TYPES:
            raise ValueError(
                f"unknown seabed-lowering evidence_type: {self.evidence_type!r} -- must be one "
                f"of {sorted(ALLOWED_LOWERING_EVIDENCE_TYPES)}"
            )
        if self.seabed_lowering_m < 0:
            raise ValueError(
                f"seabed_lowering_m must be >= 0 (a magnitude); got {self.seabed_lowering_m!r}"
            )


# --- Section 13: scenario clearance -- pipe vertical position is always fixed ------------------


def compute_scenario_pipe_underside_clearance_m(
    current_clearance_m: float | None, seabed_lowering_m: float
) -> float | None:
    """Section 13: `scenario_clearance_m = current_clearance_m + seabed_lowering_m`, exactly
    equivalent to `pipe_bottom_elevation_m - (seabed_support_elevation_m - seabed_lowering_m)`.
    This function has no pipe-elevation parameter at all -- MAR-025 assumes pipe vertical
    position fixed for the screening scenario (Section 13), and that limitation is structurally
    enforced here, not just stated."""

    if current_clearance_m is None:
        return None
    return current_clearance_m + seabed_lowering_m


# --- Section 14: support-loss susceptibility states -- scenario-screening states, never a
# probability -------------------------------------------------------------------------------

REMAINS_SUPPORTED_IN_TESTED_SCENARIO = "REMAINS_SUPPORTED_IN_TESTED_SCENARIO"
NEW_SUPPORT_LOSS_IN_TESTED_SCENARIO = "NEW_SUPPORT_LOSS_IN_TESTED_SCENARIO"
EXISTING_UNSUPPORTED_SPAN_EXTENDS_IN_TESTED_SCENARIO = (
    "EXISTING_UNSUPPORTED_SPAN_EXTENDS_IN_TESTED_SCENARIO"
)
ALREADY_UNSUPPORTED_AT_BASELINE = "ALREADY_UNSUPPORTED_AT_BASELINE"
SUPPORT_LOSS_SCREENING_NOT_AVAILABLE = "SUPPORT_LOSS_SCREENING_NOT_AVAILABLE"

SUPPORT_LOSS_SUSCEPTIBILITY_STATES = frozenset(
    {
        REMAINS_SUPPORTED_IN_TESTED_SCENARIO,
        NEW_SUPPORT_LOSS_IN_TESTED_SCENARIO,
        EXISTING_UNSUPPORTED_SPAN_EXTENDS_IN_TESTED_SCENARIO,
        ALREADY_UNSUPPORTED_AT_BASELINE,
        SUPPORT_LOSS_SCREENING_NOT_AVAILABLE,
    }
)


def screen_support_loss_susceptibility(
    baseline_clearance_m: float | None,
    lowering_input: SeabedLoweringScenarioInput | None,
    classification_threshold_m: float | None,
) -> dict[str, Any]:
    """A single sample's scenario screening result (Sections 12-15), with no interval context.
    Returns `SUPPORT_LOSS_SCREENING_NOT_AVAILABLE` whenever the lowering input, clearance, or
    threshold is missing -- this function never invents any of them (Section 12). A bare single
    sample can only ever report NEW support loss, never an EXTENSION of an existing span, since
    that is an interval-level concept -- see `build_support_loss_scenario_profile` for the
    interval-aware profile screening that can distinguish the two."""

    if lowering_input is None or baseline_clearance_m is None or classification_threshold_m is None:
        return {
            "screening_state": SUPPORT_LOSS_SCREENING_NOT_AVAILABLE,
            "scenario_clearance_m": None,
            "scenario_support_clearance_margin_m": None,
        }

    scenario_clearance_m = compute_scenario_pipe_underside_clearance_m(
        baseline_clearance_m, lowering_input.seabed_lowering_m
    )
    baseline_unsupported = baseline_clearance_m > classification_threshold_m
    scenario_unsupported = scenario_clearance_m > classification_threshold_m

    if baseline_unsupported:
        screening_state = ALREADY_UNSUPPORTED_AT_BASELINE
    elif scenario_unsupported:
        screening_state = NEW_SUPPORT_LOSS_IN_TESTED_SCENARIO
    else:
        screening_state = REMAINS_SUPPORTED_IN_TESTED_SCENARIO

    return {
        "screening_state": screening_state,
        "scenario_clearance_m": scenario_clearance_m,
        "scenario_support_clearance_margin_m": compute_support_clearance_margin_m(
            scenario_clearance_m, classification_threshold_m
        ),
    }


def _distance_to_any_interval(
    chainage_m: float, intervals_df: pd.DataFrame, max_gap_m: float
) -> bool:
    if intervals_df.empty:
        return False
    starts = intervals_df["start_chainage_m"].to_numpy(dtype=float)
    ends = intervals_df["end_chainage_m"].to_numpy(dtype=float)
    distance = np.maximum(0.0, np.maximum(starts - chainage_m, chainage_m - ends))
    return bool(np.any(distance <= max_gap_m))


NEW = "NEW"
EXTENDED = "EXTENDED"
UNCHANGED = "UNCHANGED"


def classify_scenario_interval_vs_measured(
    scenario_start_m: float, scenario_end_m: float, measured_intervals_df: pd.DataFrame
) -> str:
    """One scenario interval's relationship to the baseline measured intervals: `NEW` (no
    overlap with any measured interval), `EXTENDED` (overlaps a measured interval but extends
    beyond its start or end), or `UNCHANGED` (exactly matches a measured interval's extent)."""

    if measured_intervals_df.empty:
        return NEW
    overlapping = measured_intervals_df[
        (measured_intervals_df["start_chainage_m"] <= scenario_end_m)
        & (measured_intervals_df["end_chainage_m"] >= scenario_start_m)
    ]
    if overlapping.empty:
        return NEW
    measured_start = float(overlapping["start_chainage_m"].min())
    measured_end = float(overlapping["end_chainage_m"].max())
    if scenario_start_m < measured_start - 1e-9 or scenario_end_m > measured_end + 1e-9:
        return EXTENDED
    return UNCHANGED


def classify_scenario_intervals_vs_measured(
    scenario_intervals_df: pd.DataFrame, measured_intervals_df: pd.DataFrame
) -> list[str]:
    """`classify_scenario_interval_vs_measured` applied to every row of
    `scenario_intervals_df`, in row order."""

    return [
        classify_scenario_interval_vs_measured(
            float(row["start_chainage_m"]), float(row["end_chainage_m"]), measured_intervals_df
        )
        for _, row in scenario_intervals_df.iterrows()
    ]


def build_support_loss_scenario_profile(
    profile_df: pd.DataFrame,
    *,
    asset_id: str,
    lowering_input: SeabedLoweringScenarioInput | None,
    classification_threshold_m: float | None,
    max_measurement_gap_m: float,
    survey_epoch: str,
    vertical_reference_semantics: str,
    measured_intervals_df: pd.DataFrame,
    interval_boundary_semantics: str = support_state.UNSUPPORTED_SAMPLE_RUN_EXTENT,
    chainage_column: str = "chainage_m",
    clearance_column: str = "clearance_m",
    support_state_column: str = "measured_support_state",
) -> dict[str, Any]:
    """The interval-aware profile screening (Sections 12-15): recomputes scenario clearance and
    scenario measured state per sample (never moving the pipe, and never consuming any source-
    interpretation flag -- MAR-025A Section 7), re-extracts scenario intervals with the SAME
    gap-governance rule as the baseline (Section 10), and classifies each scenario interval
    against the baseline `measured_intervals_df` as NEW, an EXTENSION of an existing span, or
    UNCHANGED -- giving `new_span_count`/`extended_span_count` as real interval counts, not
    per-sample counts. `interval_boundary_semantics` defaults to `UNSUPPORTED_SAMPLE_RUN_EXTENT`,
    same as the baseline extraction (MAR-025A Section 9)."""

    result_df = profile_df.reset_index(drop=True).copy()
    scenario_available = lowering_input is not None and classification_threshold_m is not None

    if not scenario_available:
        result_df["scenario_clearance_m"] = np.nan
        result_df["scenario_measured_support_state"] = (
            support_state.MEASURED_SUPPORT_STATE_UNRESOLVED
        )
        result_df["scenario_support_loss_state"] = SUPPORT_LOSS_SCREENING_NOT_AVAILABLE
        return {
            "screening_state": SUPPORT_LOSS_SCREENING_NOT_AVAILABLE,
            "profile_df": result_df,
            "scenario_intervals_df": pd.DataFrame(
                columns=list(support_state.FREE_SPAN_INTERVAL_COLUMNS)
            ),
            "new_span_count": 0,
            "extended_span_count": 0,
        }

    lowering_m = lowering_input.seabed_lowering_m
    result_df["scenario_clearance_m"] = [
        compute_scenario_pipe_underside_clearance_m(None if pd.isna(v) else float(v), lowering_m)
        for v in result_df[clearance_column]
    ]
    result_df["scenario_measured_support_state"] = [
        support_state.classify_measured_support_state(
            None if pd.isna(v) else float(v),
            classification_threshold_m=classification_threshold_m,
        )
        for v in result_df["scenario_clearance_m"]
    ]

    scenario_intervals_df = support_state.extract_measured_free_span_intervals(
        result_df,
        asset_id=asset_id,
        max_measurement_gap_m=max_measurement_gap_m,
        survey_epoch=survey_epoch,
        classification_threshold_m=classification_threshold_m,
        vertical_reference_semantics=vertical_reference_semantics,
        interval_boundary_semantics=interval_boundary_semantics,
        support_state_column="scenario_measured_support_state",
        clearance_column="scenario_clearance_m",
        chainage_column=chainage_column,
        span_id_prefix=f"{asset_id}_SCENARIO",
    )

    per_sample_states: list[str] = []
    for _, row in result_df.iterrows():
        scenario_clearance = row["scenario_clearance_m"]
        if pd.isna(scenario_clearance):
            per_sample_states.append(SUPPORT_LOSS_SCREENING_NOT_AVAILABLE)
            continue
        baseline_unsupported = row[support_state_column] == support_state.MEASURED_UNSUPPORTED
        scenario_unsupported = (
            row["scenario_measured_support_state"] == support_state.MEASURED_UNSUPPORTED
        )
        if baseline_unsupported:
            per_sample_states.append(ALREADY_UNSUPPORTED_AT_BASELINE)
        elif not scenario_unsupported:
            per_sample_states.append(REMAINS_SUPPORTED_IN_TESTED_SCENARIO)
        elif _distance_to_any_interval(
            float(row[chainage_column]), measured_intervals_df, max_measurement_gap_m
        ):
            per_sample_states.append(EXISTING_UNSUPPORTED_SPAN_EXTENDS_IN_TESTED_SCENARIO)
        else:
            per_sample_states.append(NEW_SUPPORT_LOSS_IN_TESTED_SCENARIO)
    result_df["scenario_support_loss_state"] = per_sample_states

    interval_classifications = classify_scenario_intervals_vs_measured(
        scenario_intervals_df, measured_intervals_df
    )
    scenario_intervals_df = scenario_intervals_df.copy()
    scenario_intervals_df["vs_baseline_classification"] = interval_classifications
    new_span_count = interval_classifications.count(NEW)
    extended_span_count = interval_classifications.count(EXTENDED)

    return {
        "screening_state": (
            NEW_SUPPORT_LOSS_IN_TESTED_SCENARIO
            if new_span_count or extended_span_count
            else REMAINS_SUPPORTED_IN_TESTED_SCENARIO
        ),
        "profile_df": result_df,
        "scenario_intervals_df": scenario_intervals_df,
        "new_span_count": new_span_count,
        "extended_span_count": extended_span_count,
    }

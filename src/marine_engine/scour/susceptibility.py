"""Generic pipeline scour-onset susceptibility screening (MAR-023 Track A).

Reuses MAR-014's Marini et al. (2024) scour-onset engine (`scour_onset.py`)
UNCHANGED -- this module never duplicates or re-derives its equations. What
MAR-014 computed was a TESTED critical/required embedment SCREENING CLASS
(`minimum_tested_embedment_ratio_suppressing_onset`, one of the five fixed
tested ratios, or "onset persists at the maximum tested embedment"). This
module adds the operator's ACTUAL embedment as a genuinely new input and
compares it against that already-computed screening class -- it never
solves the Marini onset equation for an unconstrained continuous critical
embedment (`scour_onset.py`'s own docstring forbids that), so every
`critical_embedment_ratio_*` value produced here is always exactly one of
the five values in `scour_onset.TESTED_EMBEDMENT_RATIOS`, or a right-censored
"greater than the maximum tested embedment" state -- never an interpolated
value that was never actually tested.

Sensitivity dimension, never averaged away
--------------------------------------------
MAR-014 evaluates every timestep across a 3x3 D50 x porosity sensitivity
set (Section 5/6 of `scour_onset.py`). This module keeps that same
discipline: every summary statistic reported here (each critical embedment
percentile, and the exceedance fraction) is computed INDEPENDENTLY per
(D50, porosity) scenario, and each primary single-value field in the
generic output table independently takes its OWN CONSERVATIVE (worst-case,
i.e. largest required embedment / highest exceedance fraction) value
across that 3x3 set -- mirroring `scour_onset.compute_sensitivity_envelope`
's own established "upper class, never averaged" convention. A section's
worst-case p50 and worst-case p95 need not come from the same underlying
(D50, porosity) combination; the full per-scenario breakdown -- from which
any single statistic's own worst-case combination can be recovered -- is
preserved in a separate detail table, never silently collapsed away.

Positive margin is never "safe" (Section 3)
------------------------------------------------
`embedment_protection_margin_p95_e_over_D = actual_embedment_ratio -
critical_embedment_ratio_p95` is an empirical onset-screening margin only.
A positive value is never labelled SAFE, DESIGN ACCEPTABLE, or NO SCOUR --
the categorical `screening_state` vocabulary (Section 4) is the only
vocabulary used for classification, and it deliberately excludes any
risk-tier language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from marine_engine.scour import scour_onset

SCIENTIFIC_ROLE = "PIPELINE_SCOUR_ONSET_SUSCEPTIBILITY_SCREENING"

# --- Section 4: screening state vocabulary -- never LOW/MEDIUM/HIGH RISK ------------------

BELOW_CRITICAL_EMBEDMENT_SCREENING = "BELOW_CRITICAL_EMBEDMENT_SCREENING"
AT_OR_ABOVE_CRITICAL_EMBEDMENT_SCREENING = "AT_OR_ABOVE_CRITICAL_EMBEDMENT_SCREENING"
OUTSIDE_METHOD_EXPERIMENTAL_ENVELOPE = "OUTSIDE_METHOD_EXPERIMENTAL_ENVELOPE"
INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT"
SCREENING_NOT_APPLICABLE = "SCREENING_NOT_APPLICABLE"
# Section 6: a more specific spelling of INSUFFICIENT_INPUT for the one circumstance the
# ticket names explicitly -- no actual route embedment profile was supplied at all.
SITE_SPECIFIC_SCOUR_SUSCEPTIBILITY_NOT_AVAILABLE_NO_EMBEDMENT_PROFILE = (
    "SITE_SPECIFIC_SCOUR_SUSCEPTIBILITY_NOT_AVAILABLE_NO_EMBEDMENT_PROFILE"
)

SCREENING_STATES = frozenset(
    {
        BELOW_CRITICAL_EMBEDMENT_SCREENING,
        AT_OR_ABOVE_CRITICAL_EMBEDMENT_SCREENING,
        OUTSIDE_METHOD_EXPERIMENTAL_ENVELOPE,
        INSUFFICIENT_INPUT,
        SCREENING_NOT_APPLICABLE,
        SITE_SPECIFIC_SCOUR_SUSCEPTIBILITY_NOT_AVAILABLE_NO_EMBEDMENT_PROFILE,
    }
)

WITHIN_SOURCE_DIAMETER_ENVELOPE = "WITHIN_SOURCE_DIAMETER_ENVELOPE"

# Section 5: the exceedance fraction is a descriptive forcing-record statistic only.
EXCEEDANCE_FRACTION_SEMANTICS = "DESCRIPTIVE_FORCING_RECORD_FRACTION_NOT_A_PROBABILITY"
EXCEEDANCE_FRACTION_DISCLAIMERS: tuple[str, ...] = (
    "THIS IS NOT A PROBABILITY OF SCOUR.",
    "THIS IS NOT A FAILURE PROBABILITY.",
    "THIS IS NOT A RETURN-PERIOD METRIC.",
)

# Section 16: the discrete tested embedment ratios, reused unchanged, treated AS a scenario
# "actual" embedment for the PL854 envelope -- never a real observed profile.
TESTED_EMBEDMENT_SCENARIOS: tuple[float, ...] = scour_onset.TESTED_EMBEDMENT_RATIOS

EVIDENCE_TYPE_SITE_SPECIFIC = "SITE_SPECIFIC_ACTUAL_EMBEDMENT_SCREENING"
EVIDENCE_TYPE_SCENARIO = "TESTED_EMBEDMENT_SCENARIO_SCREENING"
EVIDENCE_TYPE_NO_PROFILE = "NO_ACTUAL_EMBEDMENT_PROFILE"

GENERIC_OUTPUT_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "section_id",
    "start_chainage_m",
    "end_chainage_m",
    "pipeline_diameter_m",
    "actual_embedment_m",
    "actual_embedment_ratio",
    "sediment_d50_m",
    "critical_embedment_ratio_p50",
    "critical_embedment_ratio_p95",
    "critical_embedment_ratio_p99",
    "critical_embedment_ratio_max",
    "embedment_protection_margin_p95_e_over_D",
    "scour_onset_screening_exceedance_fraction",
    "valid_forcing_timestep_count",
    "triggered_forcing_timestep_count",
    "screening_state",
    "method_domain_status",
    "scientific_role",
    "evidence_type",
    "limitations",
)

SENSITIVITY_DETAIL_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "section_id",
    "tested_d50_mm",
    "porosity_scenario",
    "actual_embedment_ratio",
    "critical_embedment_ratio_p50",
    "critical_embedment_ratio_p95",
    "critical_embedment_ratio_p99",
    "critical_embedment_ratio_max",
    "valid_timestep_count",
    "triggered_timestep_count",
    "scour_onset_screening_exceedance_fraction",
)


def _nearest_rank_value(sorted_values: np.ndarray, percentile: float) -> float:
    """The smallest actually-observed value at or above `percentile` (nearest-rank, no
    interpolation) -- always one of the discrete tested classes, never an invented value
    between them. `percentile=100` returns the true maximum."""

    n = len(sorted_values)
    rank = int(np.ceil(percentile / 100.0 * n))
    rank = max(1, min(rank, n))
    return float(sorted_values[rank - 1])


@dataclass(frozen=True)
class ScenarioCriticalEmbedmentStats:
    """Per (tested_d50_mm, porosity_scenario) critical-embedment percentiles, computed from
    the per-timestep `minimum_tested_embedment_ratio_suppressing_onset` MAR-014 already
    calculated. A censored (onset-persists-at-max) row is sorted as worse than every tested
    ratio -- never dropped and never treated as a plain missing value."""

    valid_timestep_count: int
    p50_ratio: float | None
    p50_censored: bool
    p95_ratio: float | None
    p95_censored: bool
    p99_ratio: float | None
    p99_censored: bool
    max_ratio: float | None
    max_censored: bool


_CENSORED_SORT_KEY = float("inf")

# A tight numerical tolerance absorbing floating-point noise only -- mirrors
# `scour_onset.compute_embedment_monotonicity_violations`'s own established
# `tolerance=1e-9` convention. Real, concrete need (found by inspecting the actual PL854
# scenario-envelope run): `actual_embedment_m = scenario_ratio * diameter_m` followed by
# `actual_embedment_ratio = actual_embedment_m / diameter_m` does not always round-trip
# exactly (e.g. 0.03 -> 0.029999999999999995 for PL854's real D=0.3048 m), which without
# this tolerance would make a tested scenario spuriously "exceed" its own identical
# critical-embedment class.
_EMBEDMENT_RATIO_TOLERANCE = 1e-9


def compute_scenario_critical_embedment_stats(
    scenario_df: pd.DataFrame,
) -> ScenarioCriticalEmbedmentStats:
    """`scenario_df` is one (tested_d50_mm, porosity_scenario) slice of MAR-014's 3-hourly
    mobility table for a single section/hydro_pair. Rows with a null
    `required_embedment_status` (missing input data, per `determine_required_embedment`) are
    excluded entirely -- they are neither a tested-class value nor a censored one."""

    valid = scenario_df[scenario_df["required_embedment_status"].notna()]
    if valid.empty:
        return ScenarioCriticalEmbedmentStats(0, None, False, None, False, None, False, None, False)

    is_censored = (
        valid["required_embedment_status"] == scour_onset.ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT
    )
    sort_keys = np.where(
        is_censored.to_numpy(),
        _CENSORED_SORT_KEY,
        valid["minimum_tested_embedment_ratio_suppressing_onset"].to_numpy(dtype=float),
    )
    sorted_keys = np.sort(sort_keys)

    def _at(percentile: float) -> tuple[float | None, bool]:
        value = _nearest_rank_value(sorted_keys, percentile)
        if value == _CENSORED_SORT_KEY:
            return None, True
        return value, False

    p50_ratio, p50_censored = _at(50.0)
    p95_ratio, p95_censored = _at(95.0)
    p99_ratio, p99_censored = _at(99.0)
    max_ratio, max_censored = _at(100.0)
    return ScenarioCriticalEmbedmentStats(
        valid_timestep_count=int(len(valid)),
        p50_ratio=p50_ratio,
        p50_censored=p50_censored,
        p95_ratio=p95_ratio,
        p95_censored=p95_censored,
        p99_ratio=p99_ratio,
        p99_censored=p99_censored,
        max_ratio=max_ratio,
        max_censored=max_censored,
    )


def compute_scenario_exceedance(
    scenario_df: pd.DataFrame, actual_embedment_ratio: float
) -> tuple[int, int, float | None]:
    """Section 5: (valid_count, triggered_count, exceedance_fraction) for ONE (D50, porosity)
    scenario at a GIVEN actual embedment ratio. "Triggered" means the empirical tested
    critical embedment for that timestep exceeds the actual embedment -- a censored
    (onset-persists) row always counts as triggered here, because it is only ever evaluated
    when `actual_embedment_ratio` is within the tested envelope (<= the max tested ratio),
    so a critical embedment known only to exceed that maximum necessarily exceeds it too."""

    valid = scenario_df[scenario_df["required_embedment_status"].notna()]
    if valid.empty:
        return 0, 0, None

    is_censored = (
        valid["required_embedment_status"] == scour_onset.ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT
    )
    required_ratio = valid["minimum_tested_embedment_ratio_suppressing_onset"].to_numpy(dtype=float)
    triggered = is_censored.to_numpy() | (
        required_ratio > actual_embedment_ratio + _EMBEDMENT_RATIO_TOLERANCE
    )
    valid_count = int(len(valid))
    triggered_count = int(triggered.sum())
    return valid_count, triggered_count, triggered_count / valid_count


def compute_section_scour_susceptibility(
    *,
    section_mobility_df: pd.DataFrame,
    actual_embedment_m: float | None,
    diameter_m: float,
    evidence_type: str,
    sediment_d50_m: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The Section-15 generic-schema fields for ONE section, for a given (possibly absent)
    actual embedment -- plus a per-(D50,porosity)-scenario detail row list (never silently
    collapsed away). Caller fills in `asset_id`/`section_id`/chainage.

    `evidence_type` distinguishes a genuine operator-supplied actual-embedment profile
    (`EVIDENCE_TYPE_SITE_SPECIFIC`) from a PL854-style tested-embedment scenario treated AS
    an actual value (`EVIDENCE_TYPE_SCENARIO`) from the no-profile-at-all case
    (`EVIDENCE_TYPE_NO_PROFILE`) -- callers choose based on where the embedment value
    actually came from, this function never infers it.
    """

    method_domain_status = (
        WITHIN_SOURCE_DIAMETER_ENVELOPE
        if scour_onset.SOURCE_DIAMETER_RANGE_M[0]
        <= diameter_m
        <= scour_onset.SOURCE_DIAMETER_RANGE_M[1]
        else scour_onset.PIPE_DIAMETER_OUTSIDE_SOURCE_ENVELOPE
    )
    limitations: list[str] = [scour_onset.RESEARCH_SCREENING_EXTRAPOLATION]
    if method_domain_status == scour_onset.PIPE_DIAMETER_OUTSIDE_SOURCE_ENVELOPE:
        limitations.append(method_domain_status)

    base: dict[str, Any] = {
        "pipeline_diameter_m": diameter_m,
        "actual_embedment_m": actual_embedment_m,
        "actual_embedment_ratio": None,
        "sediment_d50_m": sediment_d50_m,
        "critical_embedment_ratio_p50": None,
        "critical_embedment_ratio_p95": None,
        "critical_embedment_ratio_p99": None,
        "critical_embedment_ratio_max": None,
        "embedment_protection_margin_p95_e_over_D": None,
        "scour_onset_screening_exceedance_fraction": None,
        "valid_forcing_timestep_count": 0,
        "triggered_forcing_timestep_count": None,
        "screening_state": SCREENING_NOT_APPLICABLE,
        "method_domain_status": method_domain_status,
        "scientific_role": SCIENTIFIC_ROLE,
        "evidence_type": evidence_type,
        "limitations": "; ".join(limitations),
    }

    if section_mobility_df.empty:
        return base, []

    scenario_groups = list(section_mobility_df.groupby(["tested_d50_mm", "porosity_scenario"]))
    any_scenario = scenario_groups[0][1]
    reference_stats = compute_scenario_critical_embedment_stats(any_scenario)
    base["valid_forcing_timestep_count"] = reference_stats.valid_timestep_count

    # Each percentile (and the exceedance fraction, below) is worst-cased INDEPENDENTLY across
    # the 3x3 D50 x porosity sensitivity set -- never averaged, and never assumed to all come
    # from the same single "worst" scenario (a section can have a different worst combo for
    # its p50 than for its p95). Mirrors `scour_onset.compute_sensitivity_envelope`'s own
    # established "upper class, never averaged" convention, applied per-statistic.
    detail_rows: list[dict[str, Any]] = []
    worst_ratio_by_percentile: dict[str, float] = {}
    worst_censored_by_percentile: dict[str, bool] = {}
    for (d50_mm, porosity), scenario_df in scenario_groups:
        stats = compute_scenario_critical_embedment_stats(scenario_df)
        detail_row = {
            "tested_d50_mm": d50_mm,
            "porosity_scenario": porosity,
            "actual_embedment_ratio": None,
            "critical_embedment_ratio_p50": stats.p50_ratio,
            "critical_embedment_ratio_p95": stats.p95_ratio,
            "critical_embedment_ratio_p99": stats.p99_ratio,
            "critical_embedment_ratio_max": stats.max_ratio,
            "valid_timestep_count": stats.valid_timestep_count,
            "triggered_timestep_count": None,
            "scour_onset_screening_exceedance_fraction": None,
        }
        detail_rows.append(detail_row)

        for key, ratio, censored in (
            ("p50", stats.p50_ratio, stats.p50_censored),
            ("p95", stats.p95_ratio, stats.p95_censored),
            ("p99", stats.p99_ratio, stats.p99_censored),
            ("max", stats.max_ratio, stats.max_censored),
        ):
            sort_key = _CENSORED_SORT_KEY if censored else ratio
            if key not in worst_ratio_by_percentile or sort_key > worst_ratio_by_percentile[key]:
                worst_ratio_by_percentile[key] = sort_key
                worst_censored_by_percentile[key] = censored

    for key, column in (
        ("p50", "critical_embedment_ratio_p50"),
        ("p95", "critical_embedment_ratio_p95"),
        ("p99", "critical_embedment_ratio_p99"),
        ("max", "critical_embedment_ratio_max"),
    ):
        if key in worst_ratio_by_percentile and not worst_censored_by_percentile[key]:
            base[column] = worst_ratio_by_percentile[key]

    p95_censored = worst_censored_by_percentile.get("p95", False)

    if actual_embedment_m is not None:
        actual_ratio = actual_embedment_m / diameter_m
        base["actual_embedment_ratio"] = actual_ratio
        for detail_row in detail_rows:
            detail_row["actual_embedment_ratio"] = actual_ratio

        if actual_ratio > max(TESTED_EMBEDMENT_SCENARIOS) + _EMBEDMENT_RATIO_TOLERANCE:
            base["screening_state"] = OUTSIDE_METHOD_EXPERIMENTAL_ENVELOPE
        else:
            worst_fraction: tuple[int, int, float] | None = None
            for detail_row, (_, scenario_df) in zip(detail_rows, scenario_groups, strict=True):
                valid_count, triggered_count, fraction = compute_scenario_exceedance(
                    scenario_df, actual_ratio
                )
                detail_row["triggered_timestep_count"] = triggered_count
                detail_row["scour_onset_screening_exceedance_fraction"] = fraction
                if fraction is not None and (
                    worst_fraction is None or fraction > worst_fraction[2]
                ):
                    worst_fraction = (valid_count, triggered_count, fraction)
    else:
        base["screening_state"] = (
            SITE_SPECIFIC_SCOUR_SUSCEPTIBILITY_NOT_AVAILABLE_NO_EMBEDMENT_PROFILE
        )
        worst_fraction = None

    if (
        actual_embedment_m is not None
        and base["screening_state"] != OUTSIDE_METHOD_EXPERIMENTAL_ENVELOPE
    ):
        if p95_censored:
            # Even the conservative scenario's p95 required embedment is unknown beyond the
            # tested envelope -- a numeric margin would overclaim precision this method does
            # not have.
            base["screening_state"] = OUTSIDE_METHOD_EXPERIMENTAL_ENVELOPE
        elif base["critical_embedment_ratio_p95"] is not None:
            margin = base["actual_embedment_ratio"] - base["critical_embedment_ratio_p95"]
            base["embedment_protection_margin_p95_e_over_D"] = margin
            base["screening_state"] = (
                AT_OR_ABOVE_CRITICAL_EMBEDMENT_SCREENING
                if margin >= -_EMBEDMENT_RATIO_TOLERANCE
                else BELOW_CRITICAL_EMBEDMENT_SCREENING
            )
        if worst_fraction is not None:
            base["scour_onset_screening_exceedance_fraction"] = worst_fraction[2]
            base["triggered_forcing_timestep_count"] = worst_fraction[1]

    return base, detail_rows


def build_susceptibility_tables(
    *,
    pipeline_id: str,
    sections: list[dict[str, Any]],
    mobility_df: pd.DataFrame,
    diameter_m: float,
    actual_embedment_m_by_section_id: dict[Any, float] | None,
    evidence_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per section in the Section-15 generic schema, plus the sensitivity detail
    table. `sections` is a list of dicts each carrying at least `section_id`,
    `hydro_pair_id`, `start_chainage_m`, `end_chainage_m` (the same contiguous hydro-pair
    route sections MAR-014 already built). `actual_embedment_m_by_section_id=None` means no
    profile exists anywhere on the route (Section 6) -- every row becomes
    `SITE_SPECIFIC_SCOUR_SUSCEPTIBILITY_NOT_AVAILABLE_NO_EMBEDMENT_PROFILE`.
    """

    actual_by_id = actual_embedment_m_by_section_id or {}
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for section in sections:
        section_id = section["section_id"]
        pair_id = section.get("hydro_pair_id")
        section_mobility_df = (
            mobility_df[mobility_df["hydro_pair_id"] == pair_id]
            if pair_id is not None and pd.notna(pair_id)
            else mobility_df.iloc[0:0]
        )
        actual_embedment_m = actual_by_id.get(section_id)
        row, details = compute_section_scour_susceptibility(
            section_mobility_df=section_mobility_df,
            actual_embedment_m=actual_embedment_m,
            diameter_m=diameter_m,
            evidence_type=evidence_type,
        )
        row = {
            "asset_id": pipeline_id,
            "section_id": section_id,
            "start_chainage_m": section.get("start_chainage_m"),
            "end_chainage_m": section.get("end_chainage_m"),
            **row,
        }
        summary_rows.append(row)
        for detail in details:
            detail_rows.append({"asset_id": pipeline_id, "section_id": section_id, **detail})

    summary_df = pd.DataFrame(summary_rows, columns=list(GENERIC_OUTPUT_COLUMNS))
    detail_df = pd.DataFrame(detail_rows, columns=list(SENSITIVITY_DETAIL_COLUMNS))
    return summary_df, detail_df


def build_scenario_envelope_table(
    *,
    pipeline_id: str,
    sections: list[dict[str, Any]],
    mobility_df: pd.DataFrame,
    diameter_m: float,
    embedment_scenarios: tuple[float, ...] = TESTED_EMBEDMENT_SCENARIOS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 16: PL854 scenario envelope -- one row per section x tested embedment
    scenario, treating each of the five already-tested embedment ratios AS the "actual"
    embedment in turn. Never collapses the five scenarios to one best estimate."""

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for scenario_ratio in embedment_scenarios:
        scenario_actual_m = {
            section["section_id"]: scenario_ratio * diameter_m for section in sections
        }
        summary_df, detail_df = build_susceptibility_tables(
            pipeline_id=pipeline_id,
            sections=sections,
            mobility_df=mobility_df,
            diameter_m=diameter_m,
            actual_embedment_m_by_section_id=scenario_actual_m,
            evidence_type=EVIDENCE_TYPE_SCENARIO,
        )
        summary_df.insert(2, "tested_embedment_scenario_ratio", scenario_ratio)
        detail_df.insert(2, "tested_embedment_scenario_ratio", scenario_ratio)
        summary_rows.append(summary_df)
        detail_rows.append(detail_df)

    combined_summary = (
        pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    )
    combined_detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return combined_summary, combined_detail

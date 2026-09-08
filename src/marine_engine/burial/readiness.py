"""Generic, project-agnostic burial-profile data readiness (MAR-024 Section 5).

Mirrors `marine_engine.terrain.readiness`'s established pattern exactly:
`ReadinessCheck`/`ReadinessResult` (genuinely dataset-and-domain-agnostic
pure data structures, imported directly rather than duplicated) plus a
burial-specific `BurialProfileFacts` input and `assess_burial_profile_
readiness` check function. Every check is an explicit pass/fail + severity
+ human-readable reason -- never a numeric readiness score.
"""

from __future__ import annotations

from dataclasses import dataclass

from marine_engine.terrain.readiness import (
    BLOCKING,
    LIMITATION,
    NOT_READY,
    READY,
    READY_WITH_LIMITATIONS,
    ReadinessCheck,
    ReadinessResult,
)

__all__ = [
    "READY",
    "READY_WITH_LIMITATIONS",
    "NOT_READY",
    "BLOCKING",
    "LIMITATION",
    "ReadinessCheck",
    "ReadinessResult",
    "BurialProfileFacts",
    "assess_burial_profile_readiness",
]

# --- Named thresholds -- never inline magic numbers ------------------------------------------

MAX_MISSING_VALUE_FRACTION_FOR_READY = 0.02
MAX_MISSING_VALUE_FRACTION_FOR_LIMITATION = 0.25
MIN_COVERAGE_FRACTION_FOR_LIMITATION = 0.98


@dataclass(frozen=True)
class BurialProfileFacts:
    """Plain facts about an already-loaded burial profile -- the caller computes these with
    its own I/O/pandas logic, so this module never opens a file itself and stays trivially
    unit-testable with synthetic values."""

    source_file_readable: bool
    record_count: int
    route_identifier_available: bool
    kp_available: bool
    kp_is_monotonic: bool
    duplicate_kp_count: int
    coordinate_support: bool
    crs_available: bool
    units_available: bool
    units_consistent_across_sources: bool
    burial_reference_known: bool
    missing_value_fraction: float | None
    coverage_fraction: float | None
    suspicious_spike_count: int
    negative_value_count: int
    zero_value_count: int
    source_uncertainty_available: bool
    survey_epoch_known: bool


def assess_burial_profile_readiness(facts: BurialProfileFacts) -> ReadinessResult:
    """Run every operator-style readiness check against `facts` and derive an explicit status.
    Never a numeric score (Section 5)."""

    checks: list[ReadinessCheck] = []

    checks.append(
        ReadinessCheck(
            "source_file_readable",
            facts.source_file_readable,
            BLOCKING,
            "source file could not be read"
            if not facts.source_file_readable
            else f"source file read successfully ({facts.record_count} record(s))",
        )
    )
    checks.append(
        ReadinessCheck(
            "route_identifier_available",
            facts.route_identifier_available,
            LIMITATION,
            "no explicit per-record route/asset identifier in the source"
            if not facts.route_identifier_available
            else "route/asset identifier available",
        )
    )
    checks.append(
        ReadinessCheck(
            "kp_available",
            facts.kp_available,
            BLOCKING,
            "no KP/chainage column in the source"
            if not facts.kp_available
            else "KP/chainage column present",
        )
    )
    checks.append(
        ReadinessCheck(
            "kp_monotonic",
            facts.kp_is_monotonic,
            LIMITATION,
            "source KP sequence is not monotonically non-decreasing"
            if not facts.kp_is_monotonic
            else "source KP sequence is monotonically non-decreasing",
        )
    )
    checks.append(
        ReadinessCheck(
            "no_duplicate_kp",
            facts.duplicate_kp_count == 0,
            LIMITATION,
            f"{facts.duplicate_kp_count} duplicate KP value(s) in the source"
            if facts.duplicate_kp_count
            else "no duplicate KP values",
        )
    )
    checks.append(
        ReadinessCheck(
            "coordinate_support",
            facts.coordinate_support,
            BLOCKING,
            "no real (x, y) coordinate support in the source"
            if not facts.coordinate_support
            else "real (x, y) coordinate support present",
        )
    )
    checks.append(
        ReadinessCheck(
            "crs_available",
            facts.crs_available,
            BLOCKING,
            "no defined CRS for the source coordinates"
            if not facts.crs_available
            else "CRS available and defined",
        )
    )
    checks.append(
        ReadinessCheck(
            "units_available",
            facts.units_available,
            LIMITATION,
            "measurement units not stated by the source"
            if not facts.units_available
            else "measurement units stated",
        )
    )
    checks.append(
        ReadinessCheck(
            "units_consistent_across_sources",
            facts.units_consistent_across_sources,
            LIMITATION,
            "KP/chainage units differ between the burial listing and the route source -- "
            "converted explicitly, never assumed equal"
            if not facts.units_consistent_across_sources
            else "KP/chainage units consistent across sources",
        )
    )
    checks.append(
        ReadinessCheck(
            "burial_reference_convention_known",
            facts.burial_reference_known,
            LIMITATION,
            "the source's own burial/depth measurement reference could not be established -- "
            "current burial state is reported as MEASURED_REFERENCE_REQUIRES_REVIEW, never "
            "silently assumed"
            if not facts.burial_reference_known
            else "burial measurement reference resolved",
        )
    )

    if facts.missing_value_fraction is None:
        checks.append(
            ReadinessCheck(
                "missing_value_fraction",
                False,
                LIMITATION,
                "missing-value fraction could not be computed",
            )
        )
    else:
        passed = facts.missing_value_fraction <= MAX_MISSING_VALUE_FRACTION_FOR_READY
        limitation_ok = facts.missing_value_fraction <= MAX_MISSING_VALUE_FRACTION_FOR_LIMITATION
        checks.append(
            ReadinessCheck(
                "missing_value_fraction",
                passed,
                LIMITATION if limitation_ok else BLOCKING,
                f"missing-value fraction {facts.missing_value_fraction:.1%}"
                + (
                    ""
                    if passed
                    else " exceeds the tolerance for a usable profile"
                    if not limitation_ok
                    else " is nonzero"
                ),
            )
        )

    if facts.coverage_fraction is None:
        checks.append(
            ReadinessCheck(
                "coverage_fraction", False, LIMITATION, "route coverage fraction unknown"
            )
        )
    else:
        passed = facts.coverage_fraction >= 1.0
        checks.append(
            ReadinessCheck(
                "coverage_fraction",
                passed,
                LIMITATION,
                f"measured coverage is {facts.coverage_fraction:.1%} of the authoritative "
                "route length -- unsurveyed gaps exist and are never interpolated over"
                if not passed
                else "measured coverage spans the full authoritative route length",
            )
        )

    checks.append(
        ReadinessCheck(
            "no_suspicious_spikes",
            facts.suspicious_spike_count == 0,
            LIMITATION,
            f"{facts.suspicious_spike_count} suspicious spike(s) flagged in the measured profile"
            if facts.suspicious_spike_count
            else "no suspicious spikes flagged",
        )
    )
    checks.append(
        ReadinessCheck(
            "negative_or_zero_values_recorded",
            True,
            LIMITATION,
            f"{facts.negative_value_count} negative and {facts.zero_value_count} zero record(s) "
            "present -- recorded descriptively, never itself interpreted as exposure or an "
            "invalid record",
        )
    )
    checks.append(
        ReadinessCheck(
            "source_uncertainty_available",
            facts.source_uncertainty_available,
            LIMITATION,
            "no source-stated measurement uncertainty"
            if not facts.source_uncertainty_available
            else "source-stated measurement uncertainty available",
        )
    )
    checks.append(
        ReadinessCheck(
            "survey_epoch_known",
            facts.survey_epoch_known,
            LIMITATION,
            "survey epoch not established"
            if not facts.survey_epoch_known
            else "survey epoch known",
        )
    )

    checks_t = tuple(checks)
    if any(not c.passed and c.severity == BLOCKING for c in checks_t):
        status = NOT_READY
    elif any(not c.passed and c.severity == LIMITATION for c in checks_t):
        status = READY_WITH_LIMITATIONS
    else:
        status = READY
    return ReadinessResult(status=status, checks=checks_t)

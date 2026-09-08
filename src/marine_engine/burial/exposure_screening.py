"""Generic exposure-susceptibility screening contract (MAR-024 Sections 14-17).

This is the ONLY place "future exposure" is ever discussed in this
package, and it is deliberately gated: a screening result can only be
produced when the caller supplies a defensible, explicitly-typed
seabed-lowering input. No hidden constant, default magnitude, or
project-specific assumption is accepted -- `screen_exposure_susceptibility`
requires `SeabedLoweringInput` to be constructed with one of exactly two
named evidence types, and rejects a negative lowering magnitude outright.

These are SCREENING states only (Section 15): never "likely to expose",
"exposure probability", or "failure probability". No aggregate score is
ever produced anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# --- Section 14: allowed lowering-evidence types -- never an arbitrary hidden constant -------

OBSERVED_MULTI_EPOCH_SEABED_LOWERING = "OBSERVED_MULTI_EPOCH_SEABED_LOWERING"
OPERATOR_DEFINED_LOWERING_SCENARIO = "OPERATOR_DEFINED_LOWERING_SCENARIO"

ALLOWED_LOWERING_EVIDENCE_TYPES = frozenset(
    {OBSERVED_MULTI_EPOCH_SEABED_LOWERING, OPERATOR_DEFINED_LOWERING_SCENARIO}
)

# --- Section 15: screening states -- never LIKELY_TO_EXPOSE / an exposure or failure probability

POSITIVE_COVER_REMAINS_IN_SCREENING = "POSITIVE_COVER_REMAINS_IN_SCREENING"
ZERO_OR_NEGATIVE_COVER_IN_SCREENING = "ZERO_OR_NEGATIVE_COVER_IN_SCREENING"
NO_DEFENSIBLE_SEABED_LOWERING_INPUT = "NO_DEFENSIBLE_SEABED_LOWERING_INPUT"
INSUFFICIENT_BURIAL_INPUT = "INSUFFICIENT_BURIAL_INPUT"

EXPOSURE_SCREENING_STATES = frozenset(
    {
        POSITIVE_COVER_REMAINS_IN_SCREENING,
        ZERO_OR_NEGATIVE_COVER_IN_SCREENING,
        NO_DEFENSIBLE_SEABED_LOWERING_INPUT,
        INSUFFICIENT_BURIAL_INPUT,
    }
)


@dataclass(frozen=True)
class SeabedLoweringInput:
    """A defensible, explicitly-typed seabed-lowering magnitude (Section 14). Never accepted
    as a bare float -- always paired with `evidence_type` naming WHERE it came from."""

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
                f"seabed_lowering_m must be >= 0 (a magnitude, Section 14); got "
                f"{self.seabed_lowering_m!r}"
            )


def screen_exposure_susceptibility(
    cover_above_asset_m: float | None, lowering_input: SeabedLoweringInput | None
) -> dict[str, Any]:
    """Section 14-15 (MAR-024A Section 9): `remaining_cover_after_lowering_m =
    cover_above_asset_m - seabed_lowering_m`. `cover_above_asset_m` must already be the
    canonical, sign-and-reference-normalized cover from `burial.cover` -- never a raw source
    burial value (MAR-024A Section 9). Returns `NO_DEFENSIBLE_SEABED_LOWERING_INPUT` whenever
    no `SeabedLoweringInput` is supplied -- this function never invents one (Section 16's
    real-run rule is enforced simply by never calling this with a fabricated lowering).

    The lowering-input gate is checked before the cover gate: with no lowering scenario at
    all, "no defensible lowering input" is the single actionable blocker regardless of
    whether cover happens to be resolved (MAR-024A Section 1: a dataset with an unresolved
    burial reference AND no lowering input reports `NO_DEFENSIBLE_SEABED_LOWERING_INPUT`, not
    `INSUFFICIENT_BURIAL_INPUT`). `INSUFFICIENT_BURIAL_INPUT` is reserved for a real lowering
    scenario with no cover value to apply it to."""

    if lowering_input is None:
        return {
            "screening_state": NO_DEFENSIBLE_SEABED_LOWERING_INPUT,
            "remaining_cover_after_lowering_m": None,
            "seabed_lowering_m": None,
            "lowering_evidence_type": None,
        }
    if cover_above_asset_m is None:
        return {
            "screening_state": INSUFFICIENT_BURIAL_INPUT,
            "remaining_cover_after_lowering_m": None,
            "seabed_lowering_m": None,
            "lowering_evidence_type": None,
        }

    remaining = cover_above_asset_m - lowering_input.seabed_lowering_m
    screening_state = (
        POSITIVE_COVER_REMAINS_IN_SCREENING
        if remaining > 0
        else ZERO_OR_NEGATIVE_COVER_IN_SCREENING
    )
    return {
        "screening_state": screening_state,
        "remaining_cover_after_lowering_m": remaining,
        "seabed_lowering_m": lowering_input.seabed_lowering_m,
        "lowering_evidence_type": lowering_input.evidence_type,
    }

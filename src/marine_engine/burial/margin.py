"""Target burial and burial margin -- kept distinct from actual/measured burial
(MAR-024 Sections 12-13).

ACTUAL/MEASURED BURIAL, TARGET BURIAL, and MINIMUM ENGINEERING REQUIREMENT
are three different things and are never interchanged here. If a source
does not state a target/reference burial requirement, this module never
invents one -- `compute_burial_margin_m` requires the caller to have
already decided a reference value exists; there is no default.
"""

from __future__ import annotations

# Section 12: a source-stated design/target burial depth is recorded under this label -- it is
# explicitly NOT a universal engineering requirement.
SOURCE_STATED_TARGET_BURIAL = "SOURCE_STATED_TARGET_BURIAL"


def compute_burial_margin_m(
    measured_burial_m: float, reference_required_burial_m: float | None
) -> float | None:
    """Section 13: `measured - reference`. `None` (never a fabricated reference) when no
    source-stated or operator-supplied reference burial requirement exists. A positive value
    is an empirical margin only -- never labelled SAFE by any caller of this function."""

    if reference_required_burial_m is None:
        return None
    return measured_burial_m - reference_required_burial_m

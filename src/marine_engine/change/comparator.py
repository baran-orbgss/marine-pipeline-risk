"""Generic comparison against an independent source-produced difference
product (MAR-021 Section 15).

Zero dependency on any specific project or dataset. The source product is
NEVER used to build this engine's own DoD (Section 3) -- only compared
against it, descriptively, on their common valid support. When the
source's own sign convention cannot be confirmed from real documentation,
this module NEVER picks whichever sign correlates better; it reports BOTH
interpretations side by side and states the ambiguity explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SIGN_CONFIRMED_SAME_AS_OURS = "SIGN_CONFIRMED_SAME_AS_OURS"
SIGN_CONFIRMED_OPPOSITE_OF_OURS = "SIGN_CONFIRMED_OPPOSITE_OF_OURS"
SIGN_UNRESOLVED_FROM_DOCUMENTATION = "SIGN_UNRESOLVED_FROM_DOCUMENTATION"


def _residual_stats(residual: np.ndarray) -> dict[str, float]:
    finite = residual[np.isfinite(residual)]
    median = float(np.median(finite))
    nmad = float(1.4826 * np.median(np.abs(finite - median)))
    p95_abs = float(np.percentile(np.abs(finite), 95))
    return {
        "median_residual_m": median,
        "nmad_residual_m": nmad,
        "p95_absolute_residual_m": p95_abs,
        "comparison_cell_count": int(finite.size),
    }


@dataclass(frozen=True)
class ComparatorResult:
    sign_status: str
    sign_evidence: str
    interpretations: dict[str, dict[str, Any]]
    common_comparison_cell_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sign_status": self.sign_status,
            "sign_evidence": self.sign_evidence,
            "interpretations": self.interpretations,
            "common_comparison_cell_count": self.common_comparison_cell_count,
        }


def compare_dod_to_source_product(
    my_delta_bed_elevation_m: np.ndarray,
    source_product: np.ndarray,
    common_mask: np.ndarray,
    *,
    sign_status: str = SIGN_UNRESOLVED_FROM_DOCUMENTATION,
    sign_evidence: str,
) -> ComparatorResult:
    """Compares this engine's own DoD to an independent source-produced
    product on their shared common valid support. `sign_status` defaults
    to unresolved -- callers must pass real evidence (a citation) to claim
    otherwise, never infer it from which interpretation correlates best.
    """

    both_valid = common_mask & np.isfinite(my_delta_bed_elevation_m) & np.isfinite(source_product)
    n_common = int(both_valid.sum())
    if n_common == 0:
        empty = {
            "median_residual_m": None,
            "nmad_residual_m": None,
            "p95_absolute_residual_m": None,
            "comparison_cell_count": 0,
        }
        return ComparatorResult(
            sign_status, sign_evidence, {"as_is": empty, "sign_flipped": empty}, 0
        )

    mine = my_delta_bed_elevation_m[both_valid]
    theirs = source_product[both_valid]

    interpretations: dict[str, dict[str, Any]] = {}
    if sign_status == SIGN_CONFIRMED_SAME_AS_OURS:
        interpretations["confirmed"] = _residual_stats(mine - theirs)
    elif sign_status == SIGN_CONFIRMED_OPPOSITE_OF_OURS:
        interpretations["confirmed"] = _residual_stats(mine - (-theirs))
    else:
        # Unresolved: report BOTH interpretations, never choose based on which fits better.
        interpretations["as_is"] = _residual_stats(mine - theirs)
        interpretations["sign_flipped"] = _residual_stats(mine - (-theirs))

    return ComparatorResult(sign_status, sign_evidence, interpretations, n_common)


@dataclass(frozen=True)
class VerticalBiasQA:
    median_dod_m: float
    nmad_dod_m: float
    p05_m: float
    p95_m: float
    note: str = (
        "A median offset may represent true regional seabed change, vertical datum bias, survey "
        "systematic offset, or coverage/sampling effects. It is reported descriptively and is "
        "NEVER automatically subtracted from the DoD."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "median_dod_m": self.median_dod_m,
            "nmad_dod_m": self.nmad_dod_m,
            "p05_m": self.p05_m,
            "p95_m": self.p95_m,
            "note": self.note,
        }


def assess_vertical_bias_qa(delta_bed_elevation_m: np.ndarray) -> VerticalBiasQA:
    finite = delta_bed_elevation_m[np.isfinite(delta_bed_elevation_m)]
    median = float(np.median(finite))
    nmad = float(1.4826 * np.median(np.abs(finite - median)))
    p05, p95 = (float(v) for v in np.percentile(finite, [5, 95]))
    return VerticalBiasQA(median_dod_m=median, nmad_dod_m=nmad, p05_m=p05, p95_m=p95)

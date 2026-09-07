"""Generic common-valid-support masking (MAR-021 Section 10).

Zero dependency on any specific project or dataset. Only cells valid in
BOTH epochs (on the already-aligned common grid) may enter the canonical
DoD -- a single-epoch-only cell is never treated as zero change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommonSupportResult:
    common_valid_mask: np.ndarray
    common_valid_cell_count: int
    epoch1_valid_cell_count: int
    epoch2_valid_cell_count: int
    fraction_of_epoch1_covered: float
    fraction_of_epoch2_covered: float

    def to_dict(self) -> dict:
        return {
            "common_valid_cell_count": self.common_valid_cell_count,
            "epoch1_valid_cell_count": self.epoch1_valid_cell_count,
            "epoch2_valid_cell_count": self.epoch2_valid_cell_count,
            "fraction_of_epoch1_covered": self.fraction_of_epoch1_covered,
            "fraction_of_epoch2_covered": self.fraction_of_epoch2_covered,
        }


def build_common_valid_support(valid1: np.ndarray, valid2: np.ndarray) -> CommonSupportResult:
    if valid1.shape != valid2.shape:
        raise ValueError(
            f"epoch validity masks must already be on a common grid (shapes {valid1.shape} vs "
            f"{valid2.shape}) -- call change.alignment.align_to_common_grid first"
        )

    common = valid1 & valid2
    n1 = int(valid1.sum())
    n2 = int(valid2.sum())
    n_common = int(common.sum())

    return CommonSupportResult(
        common_valid_mask=common,
        common_valid_cell_count=n_common,
        epoch1_valid_cell_count=n1,
        epoch2_valid_cell_count=n2,
        fraction_of_epoch1_covered=(n_common / n1) if n1 > 0 else 0.0,
        fraction_of_epoch2_covered=(n_common / n2) if n2 > 0 else 0.0,
    )

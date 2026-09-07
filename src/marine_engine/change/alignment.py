"""Generic common-grid alignment + horizontal misregistration QA (MAR-021
Sections 8-9).

Zero dependency on any specific project or dataset. Prefers direct integer-
pixel cropping (no interpolation) whenever the grid-compatibility
classification allows it; only resamples when genuinely required, and
always records that fact plus the interpolation method used -- resampled
values are never described as original measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marine_engine.change.epoch_compatibility import (
    EXACT_GRID_ALIGNMENT,
    INTEGER_PIXEL_OFFSET_ALIGNMENT,
    RESAMPLING_REQUIRED,
    GridCompatibilityResult,
)


@dataclass(frozen=True)
class AlignedEpoch:
    elevation: np.ndarray
    valid_mask: np.ndarray
    transform: object
    was_resampled: bool
    resampling_method: str | None


def align_to_common_grid(
    *,
    classification: GridCompatibilityResult,
    elevation1: np.ndarray,
    valid1: np.ndarray,
    transform1,
    elevation2: np.ndarray,
    valid2: np.ndarray,
    transform2,
    crs: str,
) -> tuple[AlignedEpoch, AlignedEpoch]:
    """Brings both epochs onto ONE common pixel grid. Epoch 2 (the more
    recent survey) is always the TARGET grid when resampling is required
    (Section 8: "use the 2020 grid as the target grid"); epoch 1 is never
    the resampling target."""

    if classification.status == EXACT_GRID_ALIGNMENT:
        return (
            AlignedEpoch(elevation1, valid1, transform1, False, None),
            AlignedEpoch(elevation2, valid2, transform2, False, None),
        )

    if classification.status == INTEGER_PIXEL_OFFSET_ALIGNMENT:
        return _crop_to_common_window(
            elevation1,
            valid1,
            transform1,
            elevation2,
            valid2,
            transform2,
            row_offset=classification.row_offset,
            col_offset=classification.col_offset,
        )

    if classification.status == RESAMPLING_REQUIRED:
        return _resample_epoch1_onto_epoch2_grid(
            elevation1,
            valid1,
            transform1,
            elevation2,
            valid2,
            transform2,
            crs=crs,
        )

    raise ValueError(
        f"cannot align grids with status {classification.status!r} -- caller must check "
        "classification before calling align_to_common_grid"
    )


def crop_array_to_aligned_window(
    array: np.ndarray, *, original_transform, aligned_transform, aligned_shape: tuple[int, int]
) -> np.ndarray:
    """Crops a THIRD array that shares `original_transform`'s grid (e.g. an
    independent comparator product on epoch2's own native grid) down to the
    SAME window that `align_to_common_grid` already cropped epoch2 to --
    never a fresh/independent crop computation that could silently
    disagree with the one actually used for the DoD. No-ops (returns the
    array unchanged) when the aligned window IS the original window (the
    EXACT_GRID_ALIGNMENT case)."""

    cell_size = abs(original_transform.a)
    col_start = round((aligned_transform.c - original_transform.c) / cell_size)
    row_start = round((original_transform.f - aligned_transform.f) / cell_size)
    height, width = aligned_shape
    if (row_start, col_start, array.shape) == (0, 0, aligned_shape):
        return array
    return array[row_start : row_start + height, col_start : col_start + width]


def _crop_to_common_window(
    elevation1,
    valid1,
    transform1,
    elevation2,
    valid2,
    transform2,
    *,
    row_offset: int,
    col_offset: int,
):
    """Integer-pixel-offset case: epoch2's pixel (0,0) corresponds to
    epoch1's pixel (row_offset, col_offset). Crops both arrays to their
    overlapping window -- pure slicing, zero interpolation."""

    h1, w1 = elevation1.shape
    h2, w2 = elevation2.shape

    # Overlap in epoch1's index space:
    r1_start, r1_end = max(0, row_offset), min(h1, row_offset + h2)
    c1_start, c1_end = max(0, col_offset), min(w1, col_offset + w2)
    # Same overlap in epoch2's index space:
    r2_start, r2_end = r1_start - row_offset, r1_end - row_offset
    c2_start, c2_end = c1_start - col_offset, c1_end - col_offset

    cropped1 = elevation1[r1_start:r1_end, c1_start:c1_end]
    cropped_valid1 = valid1[r1_start:r1_end, c1_start:c1_end]
    cropped2 = elevation2[r2_start:r2_end, c2_start:c2_end]
    cropped_valid2 = valid2[r2_start:r2_end, c2_start:c2_end]

    cropped_transform2 = transform2 * transform2.translation(c2_start, r2_start)

    return (
        AlignedEpoch(cropped1, cropped_valid1, cropped_transform2, False, None),
        AlignedEpoch(cropped2, cropped_valid2, cropped_transform2, False, None),
    )


def _resample_epoch1_onto_epoch2_grid(
    elevation1, valid1, transform1, elevation2, valid2, transform2, *, crs: str
):
    """Genuine resampling path -- epoch2's grid is the target (never
    epoch1's), method is explicit and recorded, and native resolution is
    never inflated (target pixel size = epoch2's own, never finer than
    either source)."""

    from rasterio.warp import Resampling, reproject

    method = Resampling.bilinear
    method_name = "bilinear"

    dst_elevation = np.full(elevation2.shape, np.nan, dtype=np.float64)
    reproject(
        source=np.where(valid1, elevation1, np.nan).astype(np.float64),
        destination=dst_elevation,
        src_transform=transform1,
        src_crs=crs,
        dst_transform=transform2,
        dst_crs=crs,
        resampling=method,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    dst_valid_fraction = np.zeros(elevation2.shape, dtype=np.float64)
    reproject(
        source=valid1.astype(np.float64),
        destination=dst_valid_fraction,
        src_transform=transform1,
        src_crs=crs,
        dst_transform=transform2,
        dst_crs=crs,
        resampling=Resampling.average,
    )
    # A resampled cell is only trusted if it draws mostly from originally-valid source cells.
    resampled_valid = dst_valid_fraction >= 0.5
    dst_elevation = np.where(resampled_valid, dst_elevation, np.nan)

    return (
        AlignedEpoch(dst_elevation, resampled_valid, transform2, True, method_name),
        AlignedEpoch(elevation2, valid2, transform2, False, None),
    )


@dataclass(frozen=True)
class MisregistrationQA:
    slope_dependent_correlation: float | None
    residual_spatial_autocorrelation_proxy: float | None
    shift_estimate: None  # never estimated by this module -- see docstring below.
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "slope_dependent_delta_z_correlation": self.slope_dependent_correlation,
            "residual_spatial_autocorrelation_proxy": self.residual_spatial_autocorrelation_proxy,
            "estimated_horizontal_shift": self.shift_estimate,
            "notes": self.notes,
        }


def assess_misregistration_qa(
    delta_z: np.ndarray, common_valid_mask: np.ndarray, slope_deg: np.ndarray
) -> MisregistrationQA:
    """Descriptive horizontal-misregistration QA (Section 9) -- NOT a
    robust image-registration shift estimator (that is explicitly
    OPTIONAL per the ticket and is not implemented here; `shift_estimate`
    is always None, and this is recorded transparently in `notes` rather
    than silently omitted). Provides two lightweight, real diagnostics
    instead:

    1. Correlation between |delta_z| and local slope: a horizontal
       misregistration produces a systematic Δz pattern concentrated on
       steep terrain (edge-doubling), so a strong positive correlation is
       a real (if not conclusive) misregistration warning sign.
    2. A spatial-autocorrelation proxy on the Δz residual field (lag-1
       Moran's-I-style neighbour correlation): coherent, spatially
       clustered residuals (rather than pixel-to-pixel noise) are also
       consistent with a systematic registration issue rather than random
       measurement noise.
    """

    notes = [
        "No robust image-registration (pixel-shift) estimator is implemented in this generic "
        "engine -- Section 9 marks this optional. The diagnostics below are descriptive QA only, "
        "not a coordinate correction, and no automatic horizontal shift is ever applied.",
    ]

    valid = common_valid_mask & np.isfinite(delta_z) & np.isfinite(slope_deg)
    if valid.sum() < 30:
        return MisregistrationQA(
            None, None, None, notes + ["insufficient common valid cells for QA"]
        )

    abs_dz = np.abs(delta_z[valid])
    slope_values = slope_deg[valid]
    if abs_dz.std() > 0 and slope_values.std() > 0:
        slope_corr = float(np.corrcoef(abs_dz, slope_values)[0, 1])
    else:
        slope_corr = None

    dz_field = np.where(valid, delta_z, np.nan)
    autocorr = _lag1_spatial_autocorrelation(dz_field, valid)

    return MisregistrationQA(slope_corr, autocorr, None, notes)


def _lag1_spatial_autocorrelation(field: np.ndarray, valid_mask: np.ndarray) -> float | None:
    """Correlation between each valid cell's value and its immediate
    east neighbour's value (a simple, cheap lag-1 spatial autocorrelation
    proxy -- not a full Moran's I, but the same underlying idea)."""

    left = field[:, :-1]
    right = field[:, 1:]
    both_valid = valid_mask[:, :-1] & valid_mask[:, 1:] & np.isfinite(left) & np.isfinite(right)
    if both_valid.sum() < 30:
        return None
    a, b = left[both_valid], right[both_valid]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])

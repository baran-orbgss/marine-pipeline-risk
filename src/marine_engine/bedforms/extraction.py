"""Generic per-tile, per-epoch bedform extraction from an in-memory
elevation array (MAR-022 Sections 10-11).

Zero dependency on any specific project or dataset. The array-based
counterpart of the zip/rasterio-file-based transect helpers already
established in `marine_engine.analogs` -- same sampling convention
(nearest-cell along-transect sampling, >=90% valid-fraction acceptance,
small-gap linear fill only), generalized to operate directly on a numpy
array + affine transform rather than requiring a file on disk, since the
canonical multi-epoch arrays here already live in memory after alignment.

Every function reuses `marine_engine.morphology.sandwave_morphometry`
(MAR-017) for the actual scientific core (filtering, detection,
morphometrics) -- this module only adds coordinate bookkeeping (world
x/y <-> along-transect distance) and per-tile/per-transect orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio

from marine_engine.morphology import sandwave_morphometry as swm

MIN_TRANSECT_VALID_FRACTION = 0.90

SANDBED_SAND_WAVE_SCALE = "SANDBED_SAND_WAVE_SCALE"
SANDBED_SMALL_BEDFORM = "SANDBED_SMALL_BEDFORM"

# MAR-022A Section 8: every row this module produces is ONE profile crossing of ONE transect --
# never an independent, uniquely-identified physical crest line, and never implied to be one by
# its own naming. A single real 2D crest can (and typically does) generate up to 3 separate
# `TRANSECT_DERIVED_BEDFORM_OBSERVATION` rows (one per offset transect), and a bedform count is
# therefore an OBSERVATION count, not a unique-feature count, until true 2D crest-line
# reconstruction exists (explicitly out of scope here).
TRANSECT_DERIVED_BEDFORM_OBSERVATION = "TRANSECT_DERIVED_BEDFORM_OBSERVATION"
TRANSECT_DERIVED_CREST_OBSERVATION = "TRANSECT_DERIVED_CREST_OBSERVATION"


def sample_transect_from_array(
    elevation: np.ndarray,
    valid_mask: np.ndarray,
    transform: rasterio.Affine,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    min_valid_fraction: float = MIN_TRANSECT_VALID_FRACTION,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Samples `elevation` along the straight line p1->p2 at native pixel
    spacing, nearest-cell (never bilinear -- matches this project's
    existing `_sample_transect` convention). Returns (distances_m, values)
    or (empty, None) when the sampled length does not meet `min_valid_
    fraction`; the sparse remaining gaps (if any) are linearly filled,
    never a large hole."""

    pixel_size_m = abs(transform.a)
    length_m = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    n_samples = max(int(length_m / pixel_size_m), 2)
    fractions = np.linspace(0.0, 1.0, n_samples)
    xs = p1[0] + fractions * (p2[0] - p1[0])
    ys = p1[1] + fractions * (p2[1] - p1[1])

    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    height, width = elevation.shape
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

    values = np.full(n_samples, np.nan, dtype=np.float64)
    valid = np.zeros(n_samples, dtype=bool)
    values[in_bounds] = elevation[rows[in_bounds], cols[in_bounds]]
    valid[in_bounds] = valid_mask[rows[in_bounds], cols[in_bounds]]

    if valid.mean() < min_valid_fraction:
        return np.array([]), None

    distances = np.linspace(0.0, length_m, n_samples)
    if not valid.all():
        values = np.interp(distances, distances[valid], values[valid])
    return distances, values


def distance_to_xy(
    p1: tuple[float, float], p2: tuple[float, float], distance_m: float, total_length_m: float
) -> tuple[float, float]:
    frac = distance_m / total_length_m if total_length_m > 0 else 0.0
    return p1[0] + frac * (p2[0] - p1[0]), p1[1] + frac * (p2[1] - p1[1])


def classify_bedform_scale(
    wavelength_m: float, *, cutoff_m: float = swm.CANONICAL_SHORT_WAVELENGTH_CUTOFF_M
) -> str:
    """Section 9: 'Do NOT call every seabed oscillation a sand wave.' Both
    scales are PRESERVED (never dropped) -- only `SANDBED_SAND_WAVE_SCALE`
    enters canonical sand-wave statistics."""

    return SANDBED_SAND_WAVE_SCALE if wavelength_m >= cutoff_m else SANDBED_SMALL_BEDFORM


@dataclass(frozen=True)
class TileBedformExtraction:
    tile_id: str
    epoch: str
    transect_rows: list[dict[str, Any]]
    bedform_rows: list[dict[str, Any]]
    crest_points: list[dict[str, Any]]  # {"point_id", "bedform_id", "x", "y", "scale_class"}
    trough_points: list[dict[str, Any]]


def extract_tile_bedforms(
    elevation: np.ndarray,
    valid_mask: np.ndarray,
    transform: rasterio.Affine,
    *,
    tile_id: str,
    epoch: str,
    center_x_m: float,
    center_y_m: float,
    tile_size_m: float,
    crest_azimuth_deg: float,
) -> TileBedformExtraction:
    """Section 10-11 for ONE tile, ONE epoch: three cross-crest transects
    (MAR-017 Section 14, reused unchanged), full bedform detection at
    every detected wavelength (never filtered to >=30 m before this
    point -- see `classify_bedform_scale` for the separate, non-destructive
    scale tag applied afterward), and transect-projected point geometry
    for every crest/trough (Section 11: 'Do NOT fabricate continuous 2D
    crest lines from sparse profile detections')."""

    endpoints = swm.generate_cross_crest_transects(
        center_x_m, center_y_m, tile_size_m, crest_azimuth_deg
    )
    transect_rows: list[dict[str, Any]] = []
    bedform_rows: list[dict[str, Any]] = []
    crest_points: list[dict[str, Any]] = []
    trough_points: list[dict[str, Any]] = []

    for offset_fraction, (p1, p2) in zip((-0.25, 0.0, 0.25), endpoints, strict=True):
        transect_id = f"{tile_id}_{epoch}_offset_{offset_fraction:+.2f}"
        distances_m, values = sample_transect_from_array(elevation, valid_mask, transform, p1, p2)
        if values is None or len(values) < 10:
            transect_rows.append(
                {
                    "transect_id": transect_id,
                    "tile_id": tile_id,
                    "epoch": epoch,
                    "offset_fraction_along_crest": offset_fraction,
                    "start_x_m": p1[0],
                    "start_y_m": p1[1],
                    "end_x_m": p2[0],
                    "end_y_m": p2[1],
                    "crest_azimuth_deg": crest_azimuth_deg,
                    "sample_count": 0 if values is None else len(values),
                    "bedform_count": 0,
                    "status": "EXCLUDED_INSUFFICIENT_VALID_DATA",
                }
            )
            continue

        detrended = swm.remove_linear_trend_1d(values, distances_m)
        pixel_size_m = float(distances_m[1] - distances_m[0]) if len(distances_m) > 1 else 1.0
        filtered = swm.low_pass_filter_profile(
            detrended, pixel_size_m, swm.CANONICAL_SHORT_WAVELENGTH_CUTOFF_M
        )
        detected = swm.compute_bedform_morphometrics(detrended, filtered, distances_m)
        crest_spacing = swm.compute_crest_to_crest_spacing(filtered, distances_m)

        for i, bedform in enumerate(detected):
            bedform_id = f"{transect_id}_bedform_{i}"
            scale_class = classify_bedform_scale(bedform["wavelength_m"])
            bedform_rows.append(
                {
                    "bedform_id": bedform_id,
                    "record_type": TRANSECT_DERIVED_BEDFORM_OBSERVATION,
                    "transect_id": transect_id,
                    "tile_id": tile_id,
                    "epoch": epoch,
                    "scale_classification": scale_class,
                    **bedform,
                    "median_crest_to_crest_spacing_m": (
                        float(np.median(crest_spacing)) if crest_spacing else None
                    ),
                }
            )
            crest_x, crest_y = distance_to_xy(p1, p2, bedform["crest_position_m"], distances_m[-1])
            crest_points.append(
                {
                    "point_id": f"{bedform_id}_crest",
                    "record_type": TRANSECT_DERIVED_CREST_OBSERVATION,
                    "bedform_id": bedform_id,
                    "tile_id": tile_id,
                    "epoch": epoch,
                    "transect_id": transect_id,
                    "offset_fraction_along_crest": offset_fraction,
                    "crest_azimuth_deg": crest_azimuth_deg,
                    "wavelength_m": bedform["wavelength_m"],
                    "wave_height_m": bedform["wave_height_m"],
                    "scale_classification": scale_class,
                    "x": crest_x,
                    "y": crest_y,
                }
            )
            for side, dist_m in (
                ("left", bedform["left_trough_position_m"]),
                ("right", bedform["right_trough_position_m"]),
            ):
                tx, ty = distance_to_xy(p1, p2, dist_m, distances_m[-1])
                trough_points.append(
                    {
                        "point_id": f"{bedform_id}_trough_{side}",
                        "bedform_id": bedform_id,
                        "tile_id": tile_id,
                        "epoch": epoch,
                        "transect_id": transect_id,
                        "x": tx,
                        "y": ty,
                    }
                )

        transect_rows.append(
            {
                "transect_id": transect_id,
                "tile_id": tile_id,
                "epoch": epoch,
                "offset_fraction_along_crest": offset_fraction,
                "start_x_m": p1[0],
                "start_y_m": p1[1],
                "end_x_m": p2[0],
                "end_y_m": p2[1],
                "crest_azimuth_deg": crest_azimuth_deg,
                "sample_count": len(distances_m),
                "bedform_count": len(detected),
                "status": "OK",
            }
        )

    return TileBedformExtraction(
        tile_id=tile_id,
        epoch=epoch,
        transect_rows=transect_rows,
        bedform_rows=bedform_rows,
        crest_points=crest_points,
        trough_points=trough_points,
    )

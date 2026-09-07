"""Generic, project-agnostic high-resolution terrain derivatives (MAR-020).

Zero dependency on any specific project, dataset, or PL854/Sheringham-
specific code -- every function takes a plain elevation array + validity
mask + physical-metre parameters and returns plain arrays. The windowed
moment-sum/local-plane-fit convention mirrors (but never imports)
`morphology.regional` (MAR-007): a local polynomial fit over a physical-
radius neighborhood window.

Square windows, not circular -- a real, measured engineering trade-off
------------------------------------------------------------------------
An earlier version of this module used a CIRCULAR footprint evaluated via
FFT convolution (`scipy.signal.fftconvolve`), matching `regional.py`'s own
circular-footprint convention. On the real Sheringham Shoal benchmark
raster (25,610 x 9,855 = ~252 million pixels), that raised a genuine
`MemoryError: std::bad_alloc` inside SciPy's FFT backend -- FFT convolution
pads both operands to a common transform size and holds multiple complex-
valued frequency-domain arrays, which does not fit in memory at this pixel
count regardless of the requested physical radius.

This module instead uses a SQUARE window. Every windowed sum needed here
(the footprint itself, and the footprint-weighted offset grids dx, dy,
dx^2, dy^2, dx*dy) factors exactly into a product of a row-only function
and a column-only function once the footprint is a square rather than a
circle -- e.g. `dx(j)*footprint(i,j) = 1(i) * dx(j)` -- so every one of
them is computed as two fast, memory-bounded 1D passes
(`scipy.ndimage.correlate1d`) rather than one 2D FFT. This is an EXACT
computation of a square-window statistic (not an approximation of the
circular one), stays tractable at any physical radius on a full native-
resolution raster, and remains a standard, common neighborhood shape in
terrain analysis (many GIS neighborhood-statistics tools default to a
square/rectangular window).

Curvature uses the classic Zevenbergen & Thorne (1987) finite-difference
formulas over an explicit physical step size (never a fixed 3x3 pixel
window unless that happens to equal the requested physical step) -- the
same formulas used by common GIS curvature tools, so results are directly
comparable to third-party software. This is unaffected by the square-vs-
circular window question (it is a fixed 8-neighbour stencil, not a
windowed sum).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

MIN_VALID_NEIGHBORHOOD_FRACTION = 0.90


# --- Shared windowed-moment machinery (square window, separable 1D passes) ------------------


def _radius_px(radius_m: float, cell_size_m: float) -> int:
    return max(1, round(radius_m / cell_size_m))


def _separable_correlate(
    array: np.ndarray, row_weights: np.ndarray, col_weights: np.ndarray
) -> np.ndarray:
    """2D correlation with an implicit kernel `k(i, j) = row_weights[i] *
    col_weights[j]` (i.e. a separable, square-window kernel), computed as
    two fast 1D passes -- never a 2D FFT, so memory use stays bounded
    regardless of the physical radius requested. `scipy.ndimage.correlate1d`
    is a TRUE correlation (no implicit kernel flip), so both odd
    (anti-symmetric, e.g. the plain dx/dy offset weights) and even
    (symmetric, e.g. dx^2/dy^2/dx*dy) weight vectors are handled correctly
    with no sign subtlety to reason about."""

    intermediate = ndimage.correlate1d(array, col_weights, axis=1, mode="constant", cval=0.0)
    return ndimage.correlate1d(intermediate, row_weights, axis=0, mode="constant", cval=0.0)


@dataclass(frozen=True)
class NeighborhoodMoments:
    n: np.ndarray
    z: np.ndarray
    z2: np.ndarray
    valid_fraction: np.ndarray
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    xx: np.ndarray | None = None
    yy: np.ndarray | None = None
    xy: np.ndarray | None = None
    xz: np.ndarray | None = None
    yz: np.ndarray | None = None


def compute_neighborhood_moments(
    elevation: np.ndarray,
    valid_mask: np.ndarray,
    radius_m: float,
    cell_size_m: float,
    *,
    need_plane: bool = False,
) -> NeighborhoodMoments:
    """Per-pixel windowed sums over a SQUARE physical-half-width
    neighborhood, for the WHOLE array at once (vectorized). `need_plane=
    True` additionally computes the cross-moments needed for a local
    `z = ax + by + c` fit. Row index increases southward (standard
    north-up raster convention), so the northing offset is the NEGATED
    row offset."""

    radius_px = _radius_px(radius_m, cell_size_m)
    window_size = 2 * radius_px + 1
    uniform_1d = np.ones(window_size)
    offsets_m = np.arange(-radius_px, radius_px + 1, dtype=np.float64) * cell_size_m

    valid_f = valid_mask.astype(np.float64)
    elevation_filled = np.where(valid_mask, elevation, 0.0).astype(np.float64)

    n = _separable_correlate(valid_f, uniform_1d, uniform_1d)
    z = _separable_correlate(elevation_filled, uniform_1d, uniform_1d)
    z2 = _separable_correlate(elevation_filled * elevation_filled, uniform_1d, uniform_1d)
    valid_fraction = n / float(window_size * window_size)

    x = y = xx = yy = xy = xz = yz = None
    if need_plane:
        dx_1d = offsets_m
        dy_1d = -offsets_m
        x = _separable_correlate(valid_f, uniform_1d, dx_1d)
        y = _separable_correlate(valid_f, dy_1d, uniform_1d)
        xx = _separable_correlate(valid_f, uniform_1d, dx_1d * dx_1d)
        yy = _separable_correlate(valid_f, dy_1d * dy_1d, uniform_1d)
        xy = _separable_correlate(valid_f, dy_1d, dx_1d)
        xz = _separable_correlate(elevation_filled, uniform_1d, dx_1d)
        yz = _separable_correlate(elevation_filled, dy_1d, uniform_1d)

    return NeighborhoodMoments(
        n=n, z=z, z2=z2, valid_fraction=valid_fraction, x=x, y=y, xx=xx, yy=yy, xy=xy, xz=xz, yz=yz
    )


def _apply_validity_threshold(array: np.ndarray, valid_fraction: np.ndarray) -> np.ndarray:
    return np.where(valid_fraction >= MIN_VALID_NEIGHBORHOOD_FRACTION, array, np.nan)


# --- A/B: slope + aspect (local planar fit) --------------------------------------------------


def compute_slope_aspect_deg(
    elevation: np.ndarray, valid_mask: np.ndarray, radius_m: float, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit `z = ax + by + c` inside a square neighborhood.

    slope_deg = atan(sqrt(a^2+b^2)) -- magnitude only, always >= 0.
    aspect_deg = compass bearing (0=N, 90=E, clockwise) of the steepest-
    DESCENT direction (-a, -b); undefined (NaN) on perfectly flat cells
    (a=b=0), never assigned an arbitrary bearing.
    """

    m = compute_neighborhood_moments(elevation, valid_mask, radius_m, cell_size_m, need_plane=True)
    solvable = (m.n >= 3) & (m.valid_fraction >= MIN_VALID_NEIGHBORHOOD_FRACTION)

    a_coef = np.full(elevation.shape, np.nan)
    b_coef = np.full(elevation.shape, np.nan)

    rows, cols = np.where(solvable)
    if len(rows):
        A = np.empty((len(rows), 3, 3))
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = m.xx[rows, cols], m.xy[rows, cols], m.x[rows, cols]
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = m.xy[rows, cols], m.yy[rows, cols], m.y[rows, cols]
        A[:, 2, 0], A[:, 2, 1], A[:, 2, 2] = m.x[rows, cols], m.y[rows, cols], m.n[rows, cols]
        B = np.stack([m.xz[rows, cols], m.yz[rows, cols], m.z[rows, cols]], axis=-1)

        det = np.linalg.det(A)
        ok = np.abs(det) > 1e-6
        coeffs = np.full((len(rows), 3), np.nan)
        # Explicit trailing axis (B as (k, 3, 1), not (k, 3)) so the batched-solve shape is
        # unambiguous -- (k,3,3) vs (k,3) alone should broadcast as batch+vector per the
        # documented gufunc signature, but errors on this numpy version; (k,3,1) always works.
        coeffs[ok] = np.linalg.solve(A[ok], B[ok][..., None])[..., 0]

        a_coef[rows, cols] = coeffs[:, 0]
        b_coef[rows, cols] = coeffs[:, 1]

    slope_deg = np.degrees(np.arctan(np.sqrt(a_coef**2 + b_coef**2)))
    with np.errstate(invalid="ignore"):
        flat = (np.abs(a_coef) < 1e-12) & (np.abs(b_coef) < 1e-12)
        aspect_deg = np.degrees(np.arctan2(-a_coef, -b_coef)) % 360.0
    aspect_deg = np.where(flat, np.nan, aspect_deg)

    return slope_deg, aspect_deg, m.valid_fraction


# --- C/D: profile + plan curvature (Zevenbergen & Thorne 1987) ------------------------------


def _neighbor_offset(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """`result[r, c] = array[r + dy, c + dx]`, NaN outside the array bounds."""

    height, width = array.shape
    result = np.full_like(array, np.nan, dtype=np.float64)
    src_y0, src_y1 = max(0, dy), height + min(0, dy)
    src_x0, src_x1 = max(0, dx), width + min(0, dx)
    dst_y0, dst_y1 = max(0, -dy), height + min(0, -dy)
    dst_x0, dst_x1 = max(0, -dx), width + min(0, -dx)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return result


def compute_profile_plan_curvature(
    elevation: np.ndarray, valid_mask: np.ndarray, step_m: float, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zevenbergen & Thorne (1987) 3x3-stencil curvature, generalized to an
    explicit physical step `step_m` (never a hidden pixel count): each of
    the 8 neighbours is sampled `round(step_m / cell_size_m)` PIXELS away
    (nearest-pixel, no sub-pixel interpolation).

    profile_curvature: curvature along the steepest-slope direction
    (ESRI/Zevenbergen-Thorne convention: positive = upwardly convex, e.g.
    a hill crest; negative = upwardly concave, e.g. a valley).
    plan_curvature: curvature perpendicular to the steepest-slope
    direction (contour curvature).
    Both are 0.0 on perfectly flat/symmetric-critical cells (G=H=0, the
    gradient-relative decomposition is undefined but the value is
    genuinely isotropic there), NaN wherever any of the 9 required cells
    is invalid or out of bounds.
    """

    step_px = max(1, round(step_m / cell_size_m))
    elevation_f = np.where(valid_mask, elevation, np.nan).astype(np.float64)

    z1 = _neighbor_offset(elevation_f, -step_px, -step_px)
    z2 = _neighbor_offset(elevation_f, -step_px, 0)
    z3 = _neighbor_offset(elevation_f, -step_px, step_px)
    z4 = _neighbor_offset(elevation_f, 0, -step_px)
    z5 = elevation_f
    z6 = _neighbor_offset(elevation_f, 0, step_px)
    z7 = _neighbor_offset(elevation_f, step_px, -step_px)
    z8 = _neighbor_offset(elevation_f, step_px, 0)
    z9 = _neighbor_offset(elevation_f, step_px, step_px)

    step_length_m = step_px * cell_size_m
    d_coef = ((z4 + z6) / 2.0 - z5) / step_length_m**2
    e_coef = ((z2 + z8) / 2.0 - z5) / step_length_m**2
    f_coef = (-z1 + z3 + z7 - z9) / (4.0 * step_length_m**2)
    g_coef = (z6 - z4) / (2.0 * step_length_m)
    h_coef = (z2 - z8) / (2.0 * step_length_m)

    denom = g_coef**2 + h_coef**2
    with np.errstate(invalid="ignore", divide="ignore"):
        profile_curvature = np.where(
            denom > 1e-12,
            -2.0 * (d_coef * g_coef**2 + e_coef * h_coef**2 + f_coef * g_coef * h_coef) / denom,
            0.0,
        )
        plan_curvature = np.where(
            denom > 1e-12,
            2.0 * (d_coef * h_coef**2 + e_coef * g_coef**2 - f_coef * g_coef * h_coef) / denom,
            0.0,
        )

    all_nine_valid = ~(
        np.isnan(z1)
        | np.isnan(z2)
        | np.isnan(z3)
        | np.isnan(z4)
        | np.isnan(z5)
        | np.isnan(z6)
        | np.isnan(z7)
        | np.isnan(z8)
        | np.isnan(z9)
    )
    profile_curvature = np.where(all_nine_valid, profile_curvature, np.nan)
    plan_curvature = np.where(all_nine_valid, plan_curvature, np.nan)
    valid_fraction = all_nine_valid.astype(np.float64)

    return profile_curvature, plan_curvature, valid_fraction


# --- F: local relief (max - min) -------------------------------------------------------------


def compute_local_relief(
    elevation: np.ndarray, valid_mask: np.ndarray, radius_m: float, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """local_relief = max(elevation) - min(elevation) inside the square
    neighborhood. Always >= 0. Uses `scipy.ndimage.maximum_filter`/
    `minimum_filter` with a plain rectangular `size=` (not an explicit
    boolean footprint array), which lets SciPy use its fast, separable
    (van Herk/Gil-Werman) rectangular-window algorithm -- O(N) regardless
    of window size, and the same square-window convention as every other
    derivative in this module."""

    radius_px = _radius_px(radius_m, cell_size_m)
    window_size = 2 * radius_px + 1

    filled_for_max = np.where(valid_mask, elevation, -np.inf)
    filled_for_min = np.where(valid_mask, elevation, np.inf)
    local_max = ndimage.maximum_filter(
        filled_for_max, size=window_size, mode="constant", cval=-np.inf
    )
    local_min = ndimage.minimum_filter(
        filled_for_min, size=window_size, mode="constant", cval=np.inf
    )
    relief = local_max - local_min

    moments = compute_neighborhood_moments(elevation, valid_mask, radius_m, cell_size_m)
    relief = _apply_validity_threshold(relief, moments.valid_fraction)
    return relief, moments.valid_fraction


# --- G: terrain standard deviation (broad variability, never rugosity) ----------------------


def compute_terrain_std(
    elevation: np.ndarray, valid_mask: np.ndarray, radius_m: float, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviation of elevation within the neighborhood -- broad
    terrain variability ONLY, never rugosity/TRI (see `compute_ruggedness`
    for that distinct metric) and never measurement uncertainty."""

    m = compute_neighborhood_moments(elevation, valid_mask, radius_m, cell_size_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(m.n > 0, m.z / m.n, np.nan)
        variance = np.where(m.n > 0, m.z2 / m.n - mean**2, np.nan)
    variance = np.clip(variance, 0.0, None)
    std = np.sqrt(variance)
    std = _apply_validity_threshold(std, m.valid_fraction)
    return std, m.valid_fraction


# --- H: terrain ruggedness index (Riley et al. 1999) -----------------------------------------


def compute_ruggedness(
    elevation: np.ndarray, valid_mask: np.ndarray, radius_m: float, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Terrain Ruggedness Index: sqrt(mean((z_center - z_neighbor)^2)) over
    the neighborhood (Riley, DeGloria & Elliot 1999) -- a genuinely
    different formula from `compute_terrain_std` (which is the std of
    elevation itself, not of cell-to-neighbor differences), derived here
    from the SAME moment sums for efficiency:
    mean((z_c - z_n)^2) = z2/n - 2*z_c*(z/n) + z_c^2.
    """

    m = compute_neighborhood_moments(elevation, valid_mask, radius_m, cell_size_m)
    z_center = np.where(valid_mask, elevation, np.nan).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_sq_diff = np.where(
            m.n > 0, m.z2 / m.n - 2.0 * z_center * (m.z / m.n) + z_center**2, np.nan
        )
    mean_sq_diff = np.clip(mean_sq_diff, 0.0, None)
    tri = np.sqrt(mean_sq_diff)
    tri = _apply_validity_threshold(tri, m.valid_fraction)
    tri = np.where(valid_mask, tri, np.nan)
    return tri, m.valid_fraction

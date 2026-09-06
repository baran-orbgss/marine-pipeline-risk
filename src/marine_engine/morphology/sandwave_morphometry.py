"""Reusable high-resolution sand-wave morphometry engine (MAR-017).

A generic engine, never dataset-specific (Section 25)
------------------------------------------------------------
Every function here operates on an arbitrary bathymetry raster (or a plain
numpy array + pixel size), an arbitrary tile/transect geometry, and scale
parameters expressed in METRES -- never a hard-coded HHW coordinate, grid
name, or CRS. The intent (Section 1) is that this SAME module later
accepts a real PL854 high-resolution survey without any change to its
scientific core; only the caller-supplied raster/geometry changes.

Positive-down vs. elevation sign convention (Section 16)
----------------------------------------------------------------
All morphometric functions here operate on `bed_elevation_relative_m`,
where a HIGHER value means SHALLOWER (crestward). If a source dataset
uses positive-down depth, the caller must negate it (`positive_down=True`
on `remove_planar_trend`/profile helpers below) BEFORE any crest/trough
detection -- never confuse a positive-down depth minimum with a crest.

Knaapen-style crest/trough principle (Section 16)
---------------------------------------------------------
Extrema positions are always detected from the FILTERED profile (removing
ripple/megaripple-scale noise that would otherwise fragment a single
crest into several spurious ones); the ELEVATION reported for a detected
crest/trough is always read back from the UNFILTERED, detrended profile
at that position, never the filtered value itself (which is deliberately
smoothed/attenuated).

Filter semantics: attenuation, not a hard zero (MAR-017A Section 7)
--------------------------------------------------------------------------
The 30 m cutoff used throughout is a spectral ATTENUATION scale (a
zero-phase Butterworth low-pass) -- it does not mathematically guarantee
every sub-cutoff wavelength vanishes from the detected extrema. A real
run demonstrated a detected 21.1 m trough-to-trough feature surviving a
nominal 30 m filter. The CANONICAL bedform output therefore always
applies an explicit, separate hard gate afterward
(`apply_canonical_wavelength_gate`): any detected bedform with
`wavelength_m < 30` is excluded from the canonical output and counted,
never silently dropped.

Canonical tile-size floor is never lowered to force a result
--------------------------------------------------------------------
`find_valid_tiles` (the CANONICAL 2D validation path) tries only 2000 m
then 1000 m -- if neither meets the required valid-data fraction
anywhere, it returns an EMPTY candidate list. That is a correct
scientific result for a dataset with insufficient continuous spatial
support, never a bug to paper over by shrinking the floor. A SEPARATE,
explicitly-labelled `find_exploratory_small_support_tiles` cascades below
that floor for diagnostic purposes only -- its results must never be
presented as canonical validation.

References (Section 27)
---------------------------
Knaapen, M.A.F. (2005). Sandwave migration predictor based on shape
information. J. Geophys. Res. Earth Surface. DOI: 10.1029/2004JF000195
Barnard, Erikson & Kvitek (2011). Small-scale sediment transport patterns
and bedform morphodynamics. Geo-Marine Letters. DOI: 10.1007/s00367-011-0227-1
Damen et al. (2018). Spatially Varying Environmental Properties
Controlling Observed Sand Wave Morphology. J. Geophys. Res. Earth Surface.
DOI: 10.1002/2017JF004322
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

CANONICAL_SHORT_WAVELENGTH_CUTOFF_M = 30.0
SENSITIVITY_CUTOFFS_M = (20.0, 30.0, 40.0)
CANONICAL_TILE_SIZE_M = 2000.0
MIN_TILE_SIZE_M = 1000.0
MIN_VALID_FRACTION = 0.90
MIN_WAVELENGTHS_ACROSS_TILE = 3

FILTER_SCALE_SENSITIVE = "FILTER_SCALE_SENSITIVE"

# --- Section 10/6: vertical datum status vocabulary ---------------------------------------
VERTICAL_DATUM_SOURCE_STATED = "VERTICAL_DATUM_SOURCE_STATED"
VERTICAL_DATUM_METADATA_INFERRED = "VERTICAL_DATUM_METADATA_INFERRED"
VERTICAL_DATUM_UNKNOWN = "VERTICAL_DATUM_UNKNOWN"


# --- Section 10: first-order planar-trend removal (Section 10, applied before any spectral --
# --- morphometry so a broad regional depth gradient is never mistaken for a sand wave) -----


def remove_planar_trend(
    z: np.ndarray, valid: np.ndarray, pixel_size_m: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fits z ~= a + b*x + c*y over VALID cells only (least squares) and
    returns (residual, trend_surface, coeffs). Never a higher-order
    polynomial, which could fit away the bedforms themselves."""

    rows, cols = np.indices(z.shape)
    x_m = cols.astype(np.float64) * pixel_size_m
    y_m = rows.astype(np.float64) * pixel_size_m

    x_valid, y_valid, z_valid = x_m[valid], y_m[valid], z[valid].astype(np.float64)
    design = np.column_stack([np.ones_like(x_valid), x_valid, y_valid])
    coeffs, *_ = np.linalg.lstsq(design, z_valid, rcond=None)

    trend = coeffs[0] + coeffs[1] * x_m + coeffs[2] * y_m
    residual = z.astype(np.float64) - trend
    return residual, trend, coeffs


def fill_small_gaps(array: np.ndarray, valid: np.ndarray, max_iterations: int = 200) -> np.ndarray:
    """A conservative nearest-neighbour fill for the SMALL residual gaps
    remaining inside an already-selected (>=90%-or-best-achieved valid)
    tile -- never used across large holes (Section 11); the caller is
    expected to have already rejected tiles whose gaps are not small.

    Critical: `array` at invalid positions may still hold the source's
    raw nodata sentinel (e.g. float32's most-negative value) carried
    through the earlier trend-removal subtraction -- starting the fill
    from that raw value and never actually reaching some far cell within
    `max_iterations` would leak an astronomically large number into every
    downstream FFT (a real, previously-observed corruption: a colourbar
    scaled to 1e37). Missing cells are therefore seeded at the VALID
    mean, never at their own raw value, and any cell still unreached
    after the iterative neighbour-fill is hard-clamped to that same mean
    as a last-resort safety net -- it can never leak an extreme value."""

    valid_mean = float(array[valid].mean()) if valid.any() else 0.0
    filled = np.where(valid, array, valid_mean)
    mask = valid.copy()
    for _ in range(max_iterations):
        if mask.all():
            break
        missing = ~mask
        # Average of the 4-connected neighbours that ARE valid, one pass at a time.
        neighbour_sum = np.zeros_like(filled)
        neighbour_count = np.zeros_like(filled)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_val = np.roll(filled, (dy, dx), axis=(0, 1))
            shifted_mask = np.roll(mask, (dy, dx), axis=(0, 1))
            neighbour_sum += np.where(shifted_mask, shifted_val, 0.0)
            neighbour_count += shifted_mask
        can_fill = missing & (neighbour_count > 0)
        filled[can_fill] = neighbour_sum[can_fill] / neighbour_count[can_fill]
        mask = mask | can_fill
    return filled


# --- Section 12: 2D spectral bedform diagnostic --------------------------------------------


def apply_hann_window_2d(array: np.ndarray) -> np.ndarray:
    h, w = array.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    return array * window


def compute_2d_power_spectrum(
    tile: np.ndarray, pixel_size_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D FFT power spectral density. Returns (power, freq_x, freq_y) in
    cycles/metre, DC-centred."""

    spectrum = np.fft.fftshift(np.fft.fft2(tile))
    power = np.abs(spectrum) ** 2
    freq_y = np.fft.fftshift(np.fft.fftfreq(tile.shape[0], d=pixel_size_m))
    freq_x = np.fft.fftshift(np.fft.fftfreq(tile.shape[1], d=pixel_size_m))
    return power, freq_x, freq_y


def compute_spectral_diagnostics(
    power: np.ndarray,
    freq_x: np.ndarray,
    freq_y: np.ndarray,
    cutoff_wavelength_m: float = CANONICAL_SHORT_WAVELENGTH_CUTOFF_M,
    band_half_width_octaves: float = 0.5,
) -> dict[str, Any] | None:
    """Section 12's descriptive spectral diagnostics -- never a fixed
    "sand wave present = true" threshold. Excludes the zero-frequency
    component and every wavelength shorter than `cutoff_wavelength_m`.
    Azimuth convention throughout: degrees clockwise from north (0-360
    for the wavevector, 0-180 for the (undirected) crest orientation --
    the crest is always perpendicular to the wavevector, Section 12)."""

    fx, fy = np.meshgrid(freq_x, freq_y)
    freq_mag = np.sqrt(fx**2 + fy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = np.where(freq_mag > 0, 1.0 / freq_mag, np.inf)
    band_mask = (freq_mag > 0) & (wavelength >= cutoff_wavelength_m)
    if not band_mask.any():
        return None

    band_power = np.where(band_mask, power, 0.0)
    total_band_power = float(band_power.sum())
    if total_band_power <= 0:
        return None

    peak_idx = np.unravel_index(np.argmax(band_power), band_power.shape)
    peak_power = float(band_power[peak_idx])
    peak_fx, peak_fy = float(fx[peak_idx]), float(fy[peak_idx])
    peak_freq_mag = float(np.hypot(peak_fx, peak_fy))
    dominant_wavelength_m = 1.0 / peak_freq_mag

    # Azimuth: 0=north (+fy), 90=east (+fx) -- standard geographic convention.
    dominant_wavevector_azimuth_deg = float(np.degrees(np.arctan2(peak_fx, peak_fy)) % 360.0)
    dominant_crest_azimuth_deg = float((dominant_wavevector_azimuth_deg + 90.0) % 180.0)

    dominant_peak_power_fraction = peak_power / total_band_power

    band_values = band_power[band_mask]
    median_power = float(np.median(band_values))
    spectral_peak_to_median_power_ratio = (
        peak_power / median_power if median_power > 0 else float("inf")
    )

    # Directional concentration: circular concentration (Rayleigh R) of in-band power around
    # the crest orientation (doubled-angle trick since a crest orientation is undirected, 0-180).
    crest_azimuth_field = np.degrees(np.arctan2(fx, fy)) % 180.0
    theta = np.radians(crest_azimuth_field * 2.0)
    cos_sum = float(np.sum(band_values * np.cos(theta[band_mask])))
    sin_sum = float(np.sum(band_values * np.sin(theta[band_mask])))
    directional_concentration = float(np.hypot(cos_sum, sin_sum) / total_band_power)

    # Wavelength-band power fraction: in-band power within +/- band_half_width_octaves
    # octaves of the dominant wavelength, as a fraction of the total in-band power.
    low = dominant_wavelength_m / (2.0**band_half_width_octaves)
    high = dominant_wavelength_m * (2.0**band_half_width_octaves)
    near_dominant_mask = band_mask & (wavelength >= low) & (wavelength <= high)
    wavelength_band_power_fraction = float(power[near_dominant_mask].sum() / total_band_power)

    return {
        "dominant_wavelength_m": dominant_wavelength_m,
        "dominant_wavevector_azimuth_deg": dominant_wavevector_azimuth_deg,
        "dominant_crest_azimuth_deg": dominant_crest_azimuth_deg,
        "dominant_peak_power_fraction": dominant_peak_power_fraction,
        "directional_concentration": directional_concentration,
        "wavelength_band_power_fraction": wavelength_band_power_fraction,
        "spectral_peak_to_median_power_ratio": spectral_peak_to_median_power_ratio,
    }


def analyze_tile(
    elevation: np.ndarray,
    valid: np.ndarray,
    pixel_size_m: float,
    *,
    cutoff_wavelength_m: float = CANONICAL_SHORT_WAVELENGTH_CUTOFF_M,
) -> dict[str, Any] | None:
    """Section 9-12 end-to-end for one tile: detrend -> gap-fill (small
    gaps only) -> Hann window -> 2D FFT -> descriptive diagnostics."""

    if valid.mean() <= 0:
        return None
    residual, _trend, _coeffs = remove_planar_trend(elevation, valid, pixel_size_m)
    residual_filled = fill_small_gaps(residual, valid)
    windowed = apply_hann_window_2d(residual_filled)
    power, freq_x, freq_y = compute_2d_power_spectrum(windowed, pixel_size_m)
    return compute_spectral_diagnostics(power, freq_x, freq_y, cutoff_wavelength_m)


# --- Section 11: tile design (fixed physical size, cascading down when the real data ------
# --- cannot support it -- never silently forcing a fake validity fraction) -----------------


@dataclass(frozen=True)
class TileCandidate:
    tile_id: str
    row_origin_px: int
    col_origin_px: int
    tile_size_px: int
    tile_size_m: float
    valid_fraction: float
    center_x_m: float
    center_y_m: float


def compute_valid_fraction_grid(valid: np.ndarray, tile_size_px: int) -> np.ndarray:
    """An integral-image (summed-area table) box-sum of `valid`, returning
    the valid FRACTION for every possible tile origin at `tile_size_px` --
    an efficient, exact (never coarse-decimated) exhaustive search."""

    integral = np.zeros((valid.shape[0] + 1, valid.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = valid.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    h, w = valid.shape
    if h < tile_size_px or w < tile_size_px:
        return np.zeros((0, 0))
    a = integral[0 : h - tile_size_px + 1, 0 : w - tile_size_px + 1]
    b = integral[0 : h - tile_size_px + 1, tile_size_px : w + 1]
    c = integral[tile_size_px : h + 1, 0 : w - tile_size_px + 1]
    d = integral[tile_size_px : h + 1, tile_size_px : w + 1]
    box_sum = d - b - c + a
    return box_sum.astype(np.float64) / (tile_size_px * tile_size_px)


def find_valid_tiles(
    valid: np.ndarray,
    pixel_size_m: float,
    *,
    transform,
    tile_size_m: float = CANONICAL_TILE_SIZE_M,
    min_tile_size_m: float = MIN_TILE_SIZE_M,
    min_valid_fraction: float = MIN_VALID_FRACTION,
    step_fraction: float = 0.5,
) -> tuple[list[TileCandidate], dict[str, Any]]:
    """CANONICAL 2D tile search (MAR-017A Section 3): tries `tile_size_m`
    (2000 m) then cascades DOWN only as far as `min_tile_size_m` (1000 m)
    -- NEVER further. If no tile at >=1000 m meets >=`min_valid_fraction`,
    returns an EMPTY candidate list. That is a correct, expected
    scientific result for a dataset with insufficient continuous spatial
    support, never a bug to work around by shrinking the floor (this
    module never names a specific caller dataset -- Section 25). See
    `find_exploratory_small_support_tiles` for the separate, clearly-
    labelled below-floor diagnostic path."""

    sizes_to_try = []
    size = tile_size_m
    while size >= min_tile_size_m:
        sizes_to_try.append(size)
        size /= 2.0

    cascade_log: list[dict[str, Any]] = []
    for candidate_size_m in sizes_to_try:
        tile_size_px = max(int(round(candidate_size_m / pixel_size_m)), 1)
        if tile_size_px > valid.shape[0] or tile_size_px > valid.shape[1]:
            cascade_log.append(
                {"tile_size_m": candidate_size_m, "outcome": "TOO_LARGE_FOR_RASTER_EXTENT"}
            )
            continue
        frac_grid = compute_valid_fraction_grid(valid, tile_size_px)
        if frac_grid.size == 0:
            cascade_log.append(
                {"tile_size_m": candidate_size_m, "outcome": "TOO_LARGE_FOR_RASTER_EXTENT"}
            )
            continue
        best_achieved = float(frac_grid.max())
        passing = np.argwhere(frac_grid >= min_valid_fraction)
        cascade_log.append(
            {
                "tile_size_m": candidate_size_m,
                "best_achieved_valid_fraction": best_achieved,
                "passing_tile_count": int(len(passing)),
                "outcome": "OK" if len(passing) > 0 else "NO_TILE_MET_THRESHOLD",
            }
        )
        if len(passing) == 0:
            continue

        step_px = max(int(tile_size_px * step_fraction), 1)
        seen_origins: set[tuple[int, int]] = set()
        candidates: list[TileCandidate] = []
        for row_px, col_px in passing:
            snapped = (int(row_px) // step_px * step_px, int(col_px) // step_px * step_px)
            if snapped in seen_origins:
                continue
            seen_origins.add(snapped)
            row0, col0 = int(row_px), int(col_px)
            center_col = col0 + tile_size_px / 2.0
            center_row = row0 + tile_size_px / 2.0
            center_x, center_y = transform * (center_col, center_row)
            candidates.append(
                TileCandidate(
                    tile_id=f"tile_{row0}_{col0}_{tile_size_px}",
                    row_origin_px=row0,
                    col_origin_px=col0,
                    tile_size_px=tile_size_px,
                    tile_size_m=candidate_size_m,
                    valid_fraction=float(frac_grid[row_px, col_px]),
                    center_x_m=float(center_x),
                    center_y_m=float(center_y),
                )
            )
        return candidates, {
            "cascade_log": cascade_log,
            "tile_size_used_m": candidate_size_m,
            "below_ticket_floor": candidate_size_m < min_tile_size_m,
        }

    return [], {"cascade_log": cascade_log, "tile_size_used_m": None, "below_ticket_floor": None}


def find_exploratory_small_support_tiles(
    valid: np.ndarray,
    pixel_size_m: float,
    *,
    transform,
    starting_size_m: float = 500.0,
    min_valid_fraction: float = MIN_VALID_FRACTION,
    step_fraction: float = 0.5,
    absolute_floor_m: float = 100.0,
) -> tuple[list[TileCandidate], dict[str, Any]]:
    """EXPLORATORY_SMALL_SUPPORT_DIAGNOSTIC (MAR-017A Section 4): the
    below-canonical-floor cascade (`starting_size_m` down to
    `absolute_floor_m`) that the original MAR-017 implementation
    mistakenly used for CANONICAL validation. Kept only as a clearly
    separate, explicitly-labelled diagnostic path -- callers must never
    feed these tiles into a canonical validation output, and must stamp
    every one with `canonical_validation_eligible=false` and
    `reason=BELOW_MINIMUM_SPATIAL_SUPPORT`."""

    sizes_to_try = []
    size = starting_size_m
    while size >= absolute_floor_m:
        sizes_to_try.append(size)
        size /= 2.0

    cascade_log: list[dict[str, Any]] = []
    for candidate_size_m in sizes_to_try:
        tile_size_px = max(int(round(candidate_size_m / pixel_size_m)), 1)
        if tile_size_px > valid.shape[0] or tile_size_px > valid.shape[1]:
            cascade_log.append(
                {"tile_size_m": candidate_size_m, "outcome": "TOO_LARGE_FOR_RASTER_EXTENT"}
            )
            continue
        frac_grid = compute_valid_fraction_grid(valid, tile_size_px)
        if frac_grid.size == 0:
            cascade_log.append(
                {"tile_size_m": candidate_size_m, "outcome": "TOO_LARGE_FOR_RASTER_EXTENT"}
            )
            continue
        best_achieved = float(frac_grid.max())
        passing = np.argwhere(frac_grid >= min_valid_fraction)
        cascade_log.append(
            {
                "tile_size_m": candidate_size_m,
                "best_achieved_valid_fraction": best_achieved,
                "passing_tile_count": int(len(passing)),
                "outcome": "OK" if len(passing) > 0 else "NO_TILE_MET_THRESHOLD",
            }
        )
        if len(passing) == 0:
            continue

        step_px = max(int(tile_size_px * step_fraction), 1)
        seen_origins: set[tuple[int, int]] = set()
        candidates: list[TileCandidate] = []
        for row_px, col_px in passing:
            snapped = (int(row_px) // step_px * step_px, int(col_px) // step_px * step_px)
            if snapped in seen_origins:
                continue
            seen_origins.add(snapped)
            row0, col0 = int(row_px), int(col_px)
            center_col = col0 + tile_size_px / 2.0
            center_row = row0 + tile_size_px / 2.0
            center_x, center_y = transform * (center_col, center_row)
            candidates.append(
                TileCandidate(
                    tile_id=f"tile_{row0}_{col0}_{tile_size_px}",
                    row_origin_px=row0,
                    col_origin_px=col0,
                    tile_size_px=tile_size_px,
                    tile_size_m=candidate_size_m,
                    valid_fraction=float(frac_grid[row_px, col_px]),
                    center_x_m=float(center_x),
                    center_y_m=float(center_y),
                )
            )
        return candidates, {"cascade_log": cascade_log, "tile_size_used_m": candidate_size_m}

    return [], {"cascade_log": cascade_log, "tile_size_used_m": None}


# --- Section 13: transparent, non-scoring tile ranking (a method-development convenience) --


def rank_and_select_top_tiles(
    tile_diagnostics: list[dict[str, Any]], *, top_n: int = 3
) -> list[dict[str, Any]]:
    """Sort by highest `directional_concentration`, then highest
    `spectral_peak_to_median_power_ratio` -- a transparent, descriptive
    ranking convenience, NEVER a scientific probability or bedform score
    (Section 13)."""

    ranked = sorted(
        tile_diagnostics,
        key=lambda d: (
            d["diagnostics"]["directional_concentration"],
            d["diagnostics"]["spectral_peak_to_median_power_ratio"],
        ),
        reverse=True,
    )
    return ranked[:top_n]


def meets_wavelengths_across_tile(
    dominant_wavelength_m: float,
    tile_size_m: float,
    min_wavelengths: int = MIN_WAVELENGTHS_ACROSS_TILE,
) -> bool:
    return tile_size_m >= min_wavelengths * dominant_wavelength_m


def select_canonical_eligible_tiles(
    tile_diagnostics: list[dict[str, Any]],
    *,
    top_n: int = 3,
    min_wavelengths: int = MIN_WAVELENGTHS_ACROSS_TILE,
) -> list[dict[str, Any]]:
    """MAR-017A Section 5: STRICT `tile_size_m / dominant_wavelength_m >=
    3.0` eligibility for CANONICAL tile selection -- NO fallback to an
    ineligible pool. Each `tile_diagnostics` entry must provide
    `tile_size_m` and `diagnostics["dominant_wavelength_m"]`. Returns only
    the tiles that qualify (ranked by `rank_and_select_top_tiles`), up to
    `top_n` -- if fewer than `top_n` qualify, returns only that many; if
    none qualify, returns an empty list. Never forces `top_n` by including
    an ineligible tile."""

    eligible = [
        d
        for d in tile_diagnostics
        if meets_wavelengths_across_tile(
            d["diagnostics"]["dominant_wavelength_m"], d["tile_size_m"], min_wavelengths
        )
    ]
    return rank_and_select_top_tiles(eligible, top_n=top_n)


def _square_tiles_overlap(
    center_x_1: float,
    center_y_1: float,
    size_1: float,
    center_x_2: float,
    center_y_2: float,
    size_2: float,
) -> bool:
    half_sum = (size_1 + size_2) / 2.0
    return abs(center_x_1 - center_x_2) < half_sum and abs(center_y_1 - center_y_2) < half_sum


def select_spatially_independent_eligible_tiles(
    tile_diagnostics: list[dict[str, Any]],
    *,
    max_tiles: int = 5,
    min_wavelengths: int = MIN_WAVELENGTHS_ACROSS_TILE,
) -> list[dict[str, Any]]:
    """MAR-017B Section 13: detailed canonical validation must never
    report many heavily-overlapping tiles as independent validation
    samples. Applies the SAME strict `>=3`-wavelengths eligibility as
    `select_canonical_eligible_tiles` (no fallback), ranks the eligible
    pool by the same descriptive (directional_concentration, then
    spectral peak-to-median power ratio) ordering, then walks the ranked
    list greedily: a candidate is accepted only if its square footprint
    does not overlap any tile already accepted. Stops at `max_tiles`
    (default 5) or when the ranked list is exhausted, whichever comes
    first -- never forces `max_tiles` by accepting an overlapping or
    ineligible tile. Each `tile_diagnostics` entry must additionally
    provide `center_x_m`/`center_y_m` alongside `tile_size_m` and
    `diagnostics["dominant_wavelength_m"]`."""

    eligible = [
        d
        for d in tile_diagnostics
        if meets_wavelengths_across_tile(
            d["diagnostics"]["dominant_wavelength_m"], d["tile_size_m"], min_wavelengths
        )
    ]
    ranked = sorted(
        eligible,
        key=lambda d: (
            d["diagnostics"]["directional_concentration"],
            d["diagnostics"]["spectral_peak_to_median_power_ratio"],
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    for candidate in ranked:
        if len(selected) >= max_tiles:
            break
        overlaps_existing = any(
            _square_tiles_overlap(
                candidate["center_x_m"],
                candidate["center_y_m"],
                candidate["tile_size_m"],
                s["center_x_m"],
                s["center_y_m"],
                s["tile_size_m"],
            )
            for s in selected
        )
        if not overlaps_existing:
            selected.append(candidate)
    return selected


# --- Section 14: cross-crest transects ------------------------------------------------------


def generate_cross_crest_transects(
    center_x_m: float, center_y_m: float, tile_size_m: float, crest_azimuth_deg: float
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Exactly three transects (Section 14): through the tile centre, and
    offset +/-25% of the tile width ALONG the crest direction -- each
    perpendicular to the inferred crest orientation, never an arbitrary
    N-S/E-W profile. Returns endpoint pairs; callers build their own
    LineString to avoid a hard shapely dependency in this pure-numeric
    module boundary."""

    crest_rad = np.radians(crest_azimuth_deg)
    crest_dir = np.array([np.sin(crest_rad), np.cos(crest_rad)])
    perp_rad = np.radians(crest_azimuth_deg + 90.0)
    perp_dir = np.array([np.sin(perp_rad), np.cos(perp_rad)])

    # Kept within the tile's own footprint (never extending into surrounding, likely far
    # sparser territory) -- a real, demonstrated concern for a patchily-covered analog
    # dataset, not merely a hypothetical one (Section 14: "sample at native raster support",
    # which requires the sampled length to still be densely supported).
    half_length_m = tile_size_m * 0.5
    transects = []
    for offset_fraction in (-0.25, 0.0, 0.25):
        center = np.array([center_x_m, center_y_m]) + offset_fraction * tile_size_m * crest_dir
        p1 = center - half_length_m * perp_dir
        p2 = center + half_length_m * perp_dir
        transects.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
    return transects


# --- Section 15: profile filtering (zero-phase, never a fixed-pixel-count cutoff) ----------


def remove_linear_trend_1d(values: np.ndarray, distances_m: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(values)
    coeffs = np.polyfit(distances_m[valid], values[valid], 1)
    trend = np.polyval(coeffs, distances_m)
    return values - trend


def low_pass_filter_profile(
    values: np.ndarray,
    pixel_size_m: float,
    cutoff_wavelength_m: float = CANONICAL_SHORT_WAVELENGTH_CUTOFF_M,
    order: int = 4,
) -> np.ndarray:
    """A zero-phase (`filtfilt`) Butterworth low-pass in wavelength space
    (Section 15/16: never phase-shifts crest positions). `cutoff_
    wavelength_m` is converted to the actual raster sampling scale, never
    a hard-coded pixel count (Section 9)."""

    nyquist_freq = 1.0 / (2.0 * pixel_size_m)
    cutoff_freq = 1.0 / cutoff_wavelength_m
    normal_cutoff = min(cutoff_freq / nyquist_freq, 0.999)
    b, a = butter(order, normal_cutoff, btype="low")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(values) <= padlen:
        return values.copy()
    return filtfilt(b, a, values)


# --- Section 16: crest / trough detection (from the FILTERED profile; elevation always ------
# --- read back from the UNFILTERED detrended profile) --------------------------------------


def detect_extrema(filtered_profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alternating crest (local maxima) / trough (local minima) INDICES,
    detected from the FILTERED profile only (Section 16)."""

    gradient_sign = np.sign(np.diff(filtered_profile))
    gradient_sign[gradient_sign == 0] = 1  # treat flat runs as continuing the prior direction
    sign_changes = np.diff(gradient_sign)
    # A local max: gradient goes + -> - (sign_changes == -2); a local min: - -> + (== +2).
    crest_idx = np.where(sign_changes == -2)[0] + 1
    trough_idx = np.where(sign_changes == 2)[0] + 1
    return crest_idx, trough_idx


def _alternate_extrema(crest_idx: np.ndarray, trough_idx: np.ndarray) -> list[tuple[str, int]]:
    combined = sorted(
        [("crest", int(i)) for i in crest_idx] + [("trough", int(i)) for i in trough_idx],
        key=lambda t: t[1],
    )
    alternating: list[tuple[str, int]] = []
    for kind, idx in combined:
        if alternating and alternating[-1][0] == kind:
            continue  # keep the FIRST of any same-type run (a conservative, documented choice)
        alternating.append((kind, idx))
    return alternating


# --- Section 17-18: morphometric definitions -------------------------------------------------


def compute_bedform_morphometrics(
    raw_detrended_profile: np.ndarray,
    filtered_profile: np.ndarray,
    distances_m: np.ndarray,
) -> list[dict[str, Any]]:
    """Section 17's exact per-bedform definitions -- one record per
    COMPLETE trough-crest-trough triple. Extrema positions come from the
    filtered profile; every reported ELEVATION comes from the raw
    (unfiltered) detrended profile at that same position (Section 16)."""

    crest_idx, trough_idx = detect_extrema(filtered_profile)
    sequence = _alternate_extrema(crest_idx, trough_idx)

    bedforms = []
    for i in range(1, len(sequence) - 1):
        kind_prev, idx_l = sequence[i - 1]
        kind_mid, idx_c = sequence[i]
        kind_next, idx_r = sequence[i + 1]
        if not (kind_prev == "trough" and kind_mid == "crest" and kind_next == "trough"):
            continue

        x_l, x_c, x_r = distances_m[idx_l], distances_m[idx_c], distances_m[idx_r]
        wavelength_m = x_r - x_l
        if wavelength_m <= 0:
            continue

        crest_elev = raw_detrended_profile[idx_c]
        left_trough_elev = raw_detrended_profile[idx_l]
        right_trough_elev = raw_detrended_profile[idx_r]

        frac = (x_c - x_l) / (x_r - x_l)
        trough_baseline_at_crest = left_trough_elev + frac * (right_trough_elev - left_trough_elev)
        wave_height_m = crest_elev - trough_baseline_at_crest
        if wave_height_m <= 0:
            continue

        left_half_wavelength_m = x_c - x_l
        right_half_wavelength_m = x_r - x_c
        asymmetry_index = (right_half_wavelength_m - left_half_wavelength_m) / wavelength_m

        left_slope = (crest_elev - left_trough_elev) / left_half_wavelength_m
        right_slope = (right_trough_elev - crest_elev) / right_half_wavelength_m
        max_abs_slope = max(abs(left_slope), abs(right_slope))

        bedforms.append(
            {
                "left_trough_position_m": float(x_l),
                "crest_position_m": float(x_c),
                "right_trough_position_m": float(x_r),
                "wavelength_m": float(wavelength_m),
                "wave_height_m": float(wave_height_m),
                "left_half_wavelength_m": float(left_half_wavelength_m),
                "right_half_wavelength_m": float(right_half_wavelength_m),
                "asymmetry_index": float(asymmetry_index),
                "left_mean_slope": float(left_slope),
                "right_mean_slope": float(right_slope),
                "max_abs_slope": float(max_abs_slope),
                "crest_elevation_m": float(crest_elev),
                "left_trough_elevation_m": float(left_trough_elev),
                "right_trough_elevation_m": float(right_trough_elev),
            }
        )
    return bedforms


def apply_canonical_wavelength_gate(
    bedforms: list[dict[str, Any]], cutoff_m: float = CANONICAL_SHORT_WAVELENGTH_CUTOFF_M
) -> tuple[list[dict[str, Any]], int]:
    """MAR-017A Section 6/7: the zero-phase Butterworth low-pass is a
    spectral ATTENUATION scale, not a mathematical guarantee that every
    sub-cutoff extremum vanishes -- a real run demonstrated a detected
    21.1 m trough-to-trough feature surviving a nominal 30 m filter. This
    explicit post-detection gate enforces `wavelength_m >= cutoff_m` for
    the CANONICAL sand-wave-scale bedform output. Rejected bedforms are
    counted, never silently dropped -- track the returned count as
    `SUB_CUTOFF_EXTREMUM_REJECTED` QA."""

    canonical = [b for b in bedforms if b["wavelength_m"] >= cutoff_m]
    rejected_count = len(bedforms) - len(canonical)
    return canonical, rejected_count


def compute_crest_to_crest_spacing(
    filtered_profile: np.ndarray, distances_m: np.ndarray
) -> list[float]:
    """Section 18: adjacent CREST-to-CREST spacing -- a separate
    diagnostic, never conflated with the canonical trough-to-trough
    `wavelength_m` (Section 18's explicit warning)."""

    crest_idx, _trough_idx = detect_extrema(filtered_profile)
    crest_positions = np.sort(distances_m[crest_idx])
    return [float(b - a) for a, b in zip(crest_positions[:-1], crest_positions[1:], strict=False)]


# --- Section 19: filter-sensitivity QA -------------------------------------------------------


def run_filter_sensitivity_qa(
    raw_detrended_profile: np.ndarray,
    distances_m: np.ndarray,
    pixel_size_m: float,
    cutoffs_m: tuple[float, ...] = SENSITIVITY_CUTOFFS_M,
    displacement_flag_threshold_m: float = 10.0,
    count_flag_relative_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compares detected crests/wavelengths/heights across the 20/30/40 m
    filters (Section 19) -- never hides instability: flags
    `FILTER_SCALE_SENSITIVE` if results change drastically."""

    per_cutoff: dict[float, dict[str, Any]] = {}
    for cutoff_m in cutoffs_m:
        filtered = low_pass_filter_profile(raw_detrended_profile, pixel_size_m, cutoff_m)
        bedforms = compute_bedform_morphometrics(raw_detrended_profile, filtered, distances_m)
        crest_positions = [b["crest_position_m"] for b in bedforms]
        per_cutoff[cutoff_m] = {
            "crest_count": len(bedforms),
            "median_wavelength_m": float(np.median([b["wavelength_m"] for b in bedforms]))
            if bedforms
            else None,
            "median_wave_height_m": float(np.median([b["wave_height_m"] for b in bedforms]))
            if bedforms
            else None,
            "crest_positions_m": crest_positions,
        }

    baseline = per_cutoff.get(CANONICAL_SHORT_WAVELENGTH_CUTOFF_M)
    flags: list[str] = []
    displacement_by_cutoff: dict[float, float | None] = {}
    if baseline is not None and baseline["crest_positions_m"]:
        baseline_positions = np.array(baseline["crest_positions_m"])
        for cutoff_m, result in per_cutoff.items():
            if cutoff_m == CANONICAL_SHORT_WAVELENGTH_CUTOFF_M or not result["crest_positions_m"]:
                displacement_by_cutoff[cutoff_m] = None
                continue
            other_positions = np.array(result["crest_positions_m"])
            displacements = [float(np.min(np.abs(other_positions - p))) for p in baseline_positions]
            mean_displacement = float(np.mean(displacements))
            displacement_by_cutoff[cutoff_m] = mean_displacement
            if mean_displacement > displacement_flag_threshold_m:
                flags.append(FILTER_SCALE_SENSITIVE)
            baseline_count = baseline["crest_count"]
            if baseline_count > 0 and (
                abs(result["crest_count"] - baseline_count) / baseline_count
                > count_flag_relative_threshold
            ):
                flags.append(FILTER_SCALE_SENSITIVE)

    return {
        "per_cutoff": per_cutoff,
        "crest_position_displacement_vs_baseline_m": displacement_by_cutoff,
        "flags": sorted(set(flags)),
    }

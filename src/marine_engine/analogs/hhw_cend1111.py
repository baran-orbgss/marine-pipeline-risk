"""HHW CEND 11/11 sand-wave morphometry analog orchestration (MAR-017).

Analog-only semantics (Section 3) -- never PL854 evidence
-------------------------------------------------------------
Every canonical output produced here carries
`scientific_role=HIGH_RESOLUTION_SANDBED_MORPHOMETRY_METHOD_DEVELOPMENT_
ANALOG` and `pl854_evidence=false`/`pl854_feature_input=false`/
`pl854_validation_input=false`/`method_development_analog_only=true`.
This module is the ONLY place that knows about HHW's specific grid names
and file layout; `morphology/sandwave_morphometry.py` (the reusable
engine) never sees an HHW-specific coordinate or identifier (Section 25).

Grid selection is data-driven, never hard-coded (Section 2/25)
--------------------------------------------------------------------
Section 2 states the `asciito_hhw_*` family is the "official processed
bathymetry product" and should be preferred. Which ONE of those four real
grids is actually used is decided here at runtime by evaluating each
one's own best-achievable tile validity (never assumed/hard-coded), using
the SAME cascading tile search the engine itself uses.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import LineString, Point

from marine_engine.morphology import sandwave_morphometry as swm
from marine_engine.providers.bathymetry import hhw_cend1111 as hhw_provider

SCIENTIFIC_ROLE = "HIGH_RESOLUTION_SANDBED_MORPHOMETRY_METHOD_DEVELOPMENT_ANALOG"
ANALOG_ONLY_FLAGS: dict[str, bool] = {
    "pl854_evidence": False,
    "pl854_feature_input": False,
    "pl854_validation_input": False,
    "method_development_analog_only": True,
}

PRIMARY_GRID_FAMILY_PREFIX = "asciito_hhw_"
SELECTION_DECIMATION = 4  # effective ~4 m/px scan for grid SELECTION only, never for final analysis
ANALYSIS_DECIMATION = (
    2  # effective ~2 m/px scan for tile-candidate SEARCH; final tiles read at native res
)

TILE_INVENTORY_COLUMNS = (
    "tile_id",
    "grid_directory",
    "tile_size_m",
    "valid_fraction",
    "center_x_m",
    "center_y_m",
    "below_ticket_floor_1000m",
)

TILE_SPECTRAL_COLUMNS = (
    "tile_id",
    "grid_directory",
    "tile_size_m",
    "valid_fraction",
    "center_x_m",
    "center_y_m",
    "dominant_wavelength_m",
    "dominant_wavevector_azimuth_deg",
    "dominant_crest_azimuth_deg",
    "dominant_peak_power_fraction",
    "directional_concentration",
    "wavelength_band_power_fraction",
    "spectral_peak_to_median_power_ratio",
    "meets_3_wavelengths_across_tile",
    "rank_selected_top3",
    *ANALOG_ONLY_FLAGS.keys(),
    "scientific_role",
)

TRANSECT_COLUMNS = (
    "transect_id",
    "tile_id",
    "offset_fraction_along_crest",
    "start_x_m",
    "start_y_m",
    "end_x_m",
    "end_y_m",
    "crest_azimuth_deg",
    "sample_count",
    "pixel_size_m",
    "canonical_cutoff_m",
    "bedform_count_canonical",
    "median_wavelength_m_canonical",
    "median_wave_height_m_canonical",
    "filter_sensitivity_flags",
    "crest_position_displacement_20m_vs_30m_m",
    "crest_position_displacement_40m_vs_30m_m",
    "crest_count_20m",
    "crest_count_30m",
    "crest_count_40m",
    "median_wavelength_20m_m",
    "median_wavelength_40m_m",
    "median_wave_height_20m_m",
    "median_wave_height_40m_m",
    *ANALOG_ONLY_FLAGS.keys(),
    "scientific_role",
)

INDIVIDUAL_BEDFORM_COLUMNS = (
    "bedform_id",
    "transect_id",
    "tile_id",
    "left_trough_position_m",
    "crest_position_m",
    "right_trough_position_m",
    "wavelength_m",
    "wave_height_m",
    "left_half_wavelength_m",
    "right_half_wavelength_m",
    "asymmetry_index",
    "left_mean_slope",
    "right_mean_slope",
    "max_abs_slope",
    "crest_elevation_m",
    "left_trough_elevation_m",
    "right_trough_elevation_m",
    *ANALOG_ONLY_FLAGS.keys(),
    "scientific_role",
)


def _decimated_read(
    zip_path: Path, grid_dir: str, decimation: int
) -> tuple[np.ndarray, np.ndarray, Affine, float]:
    vsi_path = f"/vsizip/{zip_path}/{grid_dir}"
    with rasterio.open(vsi_path) as src:
        out_h, out_w = max(src.height // decimation, 1), max(src.width // decimation, 1)
        data = src.read(1, out_shape=(out_h, out_w))
        nodata = src.nodata
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        pixel_size_m = abs(transform.a)
    valid = data != nodata
    return data, valid, transform, pixel_size_m


def select_primary_grid(zip_path: Path) -> dict[str, Any]:
    """Section 2/25: evaluate every real `asciito_hhw_*` grid's own best-
    achievable canonical-tile validity and select the best one -- never a
    hard-coded grid name."""

    candidates = [
        g
        for g in hhw_provider.discover_esri_grids(zip_path)
        if g.startswith(PRIMARY_GRID_FAMILY_PREFIX)
    ]
    evaluations = []
    for grid_dir in candidates:
        _data, valid, transform, pixel_size_m = _decimated_read(
            zip_path, grid_dir, SELECTION_DECIMATION
        )
        _tiles, meta = swm.find_valid_tiles(valid, pixel_size_m, transform=transform)
        best_size = meta["tile_size_used_m"]
        best_at_2000 = next(
            (
                c["best_achieved_valid_fraction"]
                for c in meta["cascade_log"]
                if c["tile_size_m"] == swm.CANONICAL_TILE_SIZE_M
            ),
            0.0,
        )
        evaluations.append(
            {
                "grid_directory": grid_dir,
                "tile_size_used_m": best_size,
                "best_achieved_valid_fraction_at_2000m": best_at_2000,
                "cascade_log": meta["cascade_log"],
            }
        )

    # Prefer the grid that found valid tiles at the LARGEST size; break ties on the best
    # achieved 2000 m density as a secondary, transparent criterion.
    def _score(ev: dict[str, Any]) -> tuple[float, float]:
        size = ev["tile_size_used_m"] or 0.0
        return (size, ev["best_achieved_valid_fraction_at_2000m"])

    best = max(evaluations, key=_score)
    return {"selected_grid_directory": best["grid_directory"], "evaluations": evaluations}


def read_native_window_by_center(
    zip_path: Path, grid_dir: str, *, center_x_m: float, center_y_m: float, tile_size_m: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """A NATIVE-resolution window read centred on a real-world (x, y),
    sized in real metres (never a full-grid load, and never mixing a
    decimated-search pixel offset with a native-resolution read -- the
    tile candidate search runs on a decimated valid-mask for speed, so
    only the real-world centre/size are trustworthy across that
    resolution change; this recomputes native pixel indices itself)."""

    vsi_path = f"/vsizip/{zip_path}/{grid_dir}"
    with rasterio.open(vsi_path) as src:
        pixel_size_m = abs(src.transform.a)
        size_px = max(int(round(tile_size_m / pixel_size_m)), 1)
        inverse_transform = ~src.transform
        center_col, center_row = inverse_transform * (center_x_m, center_y_m)
        col_off = int(round(center_col - size_px / 2.0))
        row_off = int(round(center_row - size_px / 2.0))
        window = Window(col_off, row_off, size_px, size_px)
        data = src.read(1, window=window, boundless=True, fill_value=src.nodata)
        nodata = src.nodata
    valid = data != nodata
    return data.astype(np.float64), valid, pixel_size_m


def build_tile_candidates(
    zip_path: Path, grid_dir: str
) -> tuple[list[swm.TileCandidate], dict[str, Any]]:
    """Section 11: the real, cascading tile search on the selected grid, at
    a modest decimation for search-phase speed -- final analysis always
    re-reads each candidate at NATIVE resolution (`read_native_window_by_center`)."""

    _data, valid, transform, pixel_size_m = _decimated_read(zip_path, grid_dir, ANALYSIS_DECIMATION)
    tiles, meta = swm.find_valid_tiles(valid, pixel_size_m, transform=transform)
    return tiles, meta


def build_tile_inventory_table(tiles: list[swm.TileCandidate], grid_dir: str) -> pd.DataFrame:
    if not tiles:
        return pd.DataFrame(columns=list(TILE_INVENTORY_COLUMNS))
    rows = [
        {
            "tile_id": t.tile_id,
            "grid_directory": grid_dir,
            "tile_size_m": t.tile_size_m,
            "valid_fraction": t.valid_fraction,
            "center_x_m": t.center_x_m,
            "center_y_m": t.center_y_m,
            "below_ticket_floor_1000m": t.tile_size_m < swm.MIN_TILE_SIZE_M,
        }
        for t in tiles
    ]
    return pd.DataFrame(rows)[list(TILE_INVENTORY_COLUMNS)]


def build_tile_spectral_table(
    zip_path: Path, grid_dir: str, tiles: list[swm.TileCandidate]
) -> pd.DataFrame:
    """Section 12/20: `tile_spectral_morphometry.parquet`, one row per
    valid tile -- always re-reads each tile at NATIVE resolution before
    any spectral processing."""

    rows = []
    for tile in tiles:
        elevation, valid, pixel_size_m = read_native_window_by_center(
            zip_path,
            grid_dir,
            center_x_m=tile.center_x_m,
            center_y_m=tile.center_y_m,
            tile_size_m=tile.tile_size_m,
        )
        diagnostics = swm.analyze_tile(elevation, valid, pixel_size_m)
        if diagnostics is None:
            continue
        meets_3wl = swm.meets_wavelengths_across_tile(
            diagnostics["dominant_wavelength_m"], tile.tile_size_m
        )
        rows.append(
            {
                "tile_id": tile.tile_id,
                "grid_directory": grid_dir,
                "tile_size_m": tile.tile_size_m,
                "valid_fraction": tile.valid_fraction,
                "center_x_m": tile.center_x_m,
                "center_y_m": tile.center_y_m,
                **diagnostics,
                "meets_3_wavelengths_across_tile": meets_3wl,
                "rank_selected_top3": False,
                **ANALOG_ONLY_FLAGS,
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(TILE_SPECTRAL_COLUMNS))
    df = pd.DataFrame(rows)
    return df[list(TILE_SPECTRAL_COLUMNS)]


def select_top_tiles(tile_spectral_df: pd.DataFrame, *, top_n: int = 3) -> pd.DataFrame:
    """Section 13: transparent ranking convenience only -- never a
    scientific probability or bedform score. Prefers tiles that ALSO
    satisfy the >=3-wavelengths-across-tile requirement where any exist,
    but never silently drops every real candidate if none do (a real,
    reportable limitation of this specific analog dataset's coverage)."""

    if tile_spectral_df.empty:
        return tile_spectral_df

    eligible = tile_spectral_df[tile_spectral_df["meets_3_wavelengths_across_tile"]]
    pool = eligible if not eligible.empty else tile_spectral_df
    ranked = pool.sort_values(
        ["directional_concentration", "spectral_peak_to_median_power_ratio"],
        ascending=[False, False],
    )
    top_ids = set(ranked.head(top_n)["tile_id"])
    result = tile_spectral_df.copy()
    result["rank_selected_top3"] = result["tile_id"].isin(top_ids)
    return result


def build_transect_and_bedform_tables(
    zip_path: Path, grid_dir: str, top_tiles_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Sections 14-21: three cross-crest transects per selected tile, the
    canonical (30 m) bedform detection plus 20/40 m filter-sensitivity QA,
    and the transect-based crest/trough point layers."""

    transect_rows: list[dict[str, Any]] = []
    bedform_rows: list[dict[str, Any]] = []
    crest_points: list[dict[str, Any]] = []
    trough_points: list[dict[str, Any]] = []

    selected = top_tiles_df[top_tiles_df["rank_selected_top3"]]
    for _, tile_row in selected.iterrows():
        crest_azimuth = tile_row["dominant_crest_azimuth_deg"]
        endpoints = swm.generate_cross_crest_transects(
            tile_row["center_x_m"], tile_row["center_y_m"], tile_row["tile_size_m"], crest_azimuth
        )
        for offset_fraction, (p1, p2) in zip((-0.25, 0.0, 0.25), endpoints, strict=True):
            transect_id = f"{tile_row['tile_id']}_offset_{offset_fraction:+.2f}"
            distances_m, elevations = _sample_transect(zip_path, grid_dir, p1, p2)
            if elevations is None or len(elevations) < 10:
                # Recorded (never silently dropped, Section 19's "do not hide instability"
                # ethos applied here too) -- this transect's own sampled length did not meet
                # the 90% validity bar, a real, reportable consequence of a small tile size.
                transect_rows.append(
                    {
                        "transect_id": transect_id,
                        "tile_id": tile_row["tile_id"],
                        "offset_fraction_along_crest": offset_fraction,
                        "start_x_m": p1[0],
                        "start_y_m": p1[1],
                        "end_x_m": p2[0],
                        "end_y_m": p2[1],
                        "crest_azimuth_deg": crest_azimuth,
                        "sample_count": 0 if elevations is None else len(elevations),
                        "pixel_size_m": None,
                        "canonical_cutoff_m": swm.CANONICAL_SHORT_WAVELENGTH_CUTOFF_M,
                        "bedform_count_canonical": None,
                        "median_wavelength_m_canonical": None,
                        "median_wave_height_m_canonical": None,
                        "filter_sensitivity_flags": "EXCLUDED_INSUFFICIENT_VALID_DATA",
                        "crest_position_displacement_20m_vs_30m_m": None,
                        "crest_position_displacement_40m_vs_30m_m": None,
                        "crest_count_20m": None,
                        "crest_count_30m": None,
                        "crest_count_40m": None,
                        "median_wavelength_20m_m": None,
                        "median_wavelength_40m_m": None,
                        "median_wave_height_20m_m": None,
                        "median_wave_height_40m_m": None,
                        **ANALOG_ONLY_FLAGS,
                        "scientific_role": SCIENTIFIC_ROLE,
                    }
                )
                continue
            detrended = swm.remove_linear_trend_1d(elevations, distances_m)
            pixel_size_m = float(distances_m[1] - distances_m[0]) if len(distances_m) > 1 else 1.0

            filtered_canonical = swm.low_pass_filter_profile(
                detrended, pixel_size_m, swm.CANONICAL_SHORT_WAVELENGTH_CUTOFF_M
            )
            bedforms = swm.compute_bedform_morphometrics(detrended, filtered_canonical, distances_m)
            sensitivity = swm.run_filter_sensitivity_qa(detrended, distances_m, pixel_size_m)

            for i, bedform in enumerate(bedforms):
                bedform_id = f"{transect_id}_bedform_{i}"
                bedform_rows.append(
                    {
                        "bedform_id": bedform_id,
                        "transect_id": transect_id,
                        "tile_id": tile_row["tile_id"],
                        **bedform,
                        **ANALOG_ONLY_FLAGS,
                        "scientific_role": SCIENTIFIC_ROLE,
                    }
                )
                crest_x, crest_y = _distance_to_xy(
                    p1, p2, bedform["crest_position_m"], distances_m[-1]
                )
                crest_points.append(
                    {
                        "point_id": f"{bedform_id}_crest",
                        "transect_id": transect_id,
                        "geometry": Point(crest_x, crest_y),
                    }
                )
                for side, dist_m in (
                    ("left", bedform["left_trough_position_m"]),
                    ("right", bedform["right_trough_position_m"]),
                ):
                    tx, ty = _distance_to_xy(p1, p2, dist_m, distances_m[-1])
                    trough_points.append(
                        {
                            "point_id": f"{bedform_id}_trough_{side}",
                            "transect_id": transect_id,
                            "geometry": Point(tx, ty),
                        }
                    )

            per_cutoff = sensitivity["per_cutoff"]
            displacement = sensitivity["crest_position_displacement_vs_baseline_m"]
            transect_rows.append(
                {
                    "transect_id": transect_id,
                    "tile_id": tile_row["tile_id"],
                    "offset_fraction_along_crest": offset_fraction,
                    "start_x_m": p1[0],
                    "start_y_m": p1[1],
                    "end_x_m": p2[0],
                    "end_y_m": p2[1],
                    "crest_azimuth_deg": crest_azimuth,
                    "sample_count": len(distances_m),
                    "pixel_size_m": pixel_size_m,
                    "canonical_cutoff_m": swm.CANONICAL_SHORT_WAVELENGTH_CUTOFF_M,
                    "bedform_count_canonical": len(bedforms),
                    "median_wavelength_m_canonical": (
                        float(np.median([b["wavelength_m"] for b in bedforms]))
                        if bedforms
                        else None
                    ),
                    "median_wave_height_m_canonical": (
                        float(np.median([b["wave_height_m"] for b in bedforms]))
                        if bedforms
                        else None
                    ),
                    "filter_sensitivity_flags": ",".join(sensitivity["flags"]) or None,
                    "crest_position_displacement_20m_vs_30m_m": displacement.get(20.0),
                    "crest_position_displacement_40m_vs_30m_m": displacement.get(40.0),
                    "crest_count_20m": per_cutoff[20.0]["crest_count"],
                    "crest_count_30m": per_cutoff[30.0]["crest_count"],
                    "crest_count_40m": per_cutoff[40.0]["crest_count"],
                    "median_wavelength_20m_m": per_cutoff[20.0]["median_wavelength_m"],
                    "median_wavelength_40m_m": per_cutoff[40.0]["median_wavelength_m"],
                    "median_wave_height_20m_m": per_cutoff[20.0]["median_wave_height_m"],
                    "median_wave_height_40m_m": per_cutoff[40.0]["median_wave_height_m"],
                    **ANALOG_ONLY_FLAGS,
                    "scientific_role": SCIENTIFIC_ROLE,
                }
            )

    transect_df = (
        pd.DataFrame(transect_rows)[list(TRANSECT_COLUMNS)]
        if transect_rows
        else pd.DataFrame(columns=list(TRANSECT_COLUMNS))
    )
    bedform_df = (
        pd.DataFrame(bedform_rows)[list(INDIVIDUAL_BEDFORM_COLUMNS)]
        if bedform_rows
        else pd.DataFrame(columns=list(INDIVIDUAL_BEDFORM_COLUMNS))
    )
    crests_gdf = (
        gpd.GeoDataFrame(crest_points, geometry="geometry", crs="EPSG:32631")
        if crest_points
        else gpd.GeoDataFrame(
            columns=["point_id", "transect_id", "geometry"], geometry="geometry", crs="EPSG:32631"
        )
    )
    troughs_gdf = (
        gpd.GeoDataFrame(trough_points, geometry="geometry", crs="EPSG:32631")
        if trough_points
        else gpd.GeoDataFrame(
            columns=["point_id", "transect_id", "geometry"], geometry="geometry", crs="EPSG:32631"
        )
    )
    return transect_df, bedform_df, crests_gdf, troughs_gdf


def _sample_transect(
    zip_path: Path, grid_dir: str, p1: tuple[float, float], p2: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray | None]:
    vsi_path = f"/vsizip/{zip_path}/{grid_dir}"
    line = LineString([p1, p2])
    with rasterio.open(vsi_path) as src:
        pixel_size_m = abs(src.transform.a)
        n_samples = max(int(line.length / pixel_size_m), 2)
        points = [line.interpolate(i / (n_samples - 1), normalized=True) for i in range(n_samples)]
        coords = [(pt.x, pt.y) for pt in points]
        sampled = list(src.sample(coords))
        nodata = src.nodata
    values = np.array([s[0] for s in sampled], dtype=np.float64)
    valid = values != nodata
    if valid.mean() < 0.9:
        return np.array([]), None
    distances = np.linspace(0.0, line.length, n_samples)
    # Small-gap fill only (never large holes) for the sparse remaining invalid samples.
    if not valid.all():
        values = np.interp(distances, distances[valid], values[valid])
    return distances, values


def _distance_to_xy(
    p1: tuple[float, float], p2: tuple[float, float], distance_m: float, total_length_m: float
) -> tuple[float, float]:
    frac = distance_m / total_length_m if total_length_m > 0 else 0.0
    x = p1[0] + frac * (p2[0] - p1[0])
    y = p1[1] + frac * (p2[1] - p1[1])
    return x, y


# --- Map input helpers (Sections 22-24) ------------------------------------------------------


def get_background_for_map(
    zip_path: Path, grid_dir: str, *, decimation: int = SELECTION_DECIMATION
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """A decimated full-grid read for map-background context only (never
    used for morphometric analysis itself, which always reads native
    resolution)."""

    data, valid, transform, _pixel_size_m = _decimated_read(zip_path, grid_dir, decimation)
    height, width = data.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return data, valid, (left, right, bottom, top)


def get_method_figure_inputs(
    zip_path: Path,
    grid_dir: str,
    top_tile_row: pd.Series,
    transect_df: pd.DataFrame,
    bedform_df: pd.DataFrame,
) -> dict[str, Any]:
    """Section 22: native elevation/valid + filtered/detrended surface for
    the single highest-ranked tile, plus one representative transect's
    profile and its detected bedforms, for the primary method figure."""

    elevation, valid, pixel_size_m = read_native_window_by_center(
        zip_path,
        grid_dir,
        center_x_m=top_tile_row["center_x_m"],
        center_y_m=top_tile_row["center_y_m"],
        tile_size_m=top_tile_row["tile_size_m"],
    )
    residual, _trend, _coeffs = swm.remove_planar_trend(elevation, valid, pixel_size_m)
    residual_filled = swm.fill_small_gaps(residual, valid)
    # For visual display, show the spatial-domain filtered surface directly (not run
    # through the Hann window the FFT diagnostic itself uses, which would darken the tile
    # edges) -- the same 2D short-wavelength suppression, just without the window taper.
    filtered_surface = _spatial_short_wavelength_filter(residual_filled, pixel_size_m)

    half = top_tile_row["tile_size_m"] / 2.0
    extent = (
        top_tile_row["center_x_m"] - half,
        top_tile_row["center_x_m"] + half,
        top_tile_row["center_y_m"] - half,
        top_tile_row["center_y_m"] + half,
    )

    tile_transects = transect_df[transect_df["tile_id"] == top_tile_row["tile_id"]]
    transect_endpoints = [
        ((row["start_x_m"], row["start_y_m"]), (row["end_x_m"], row["end_y_m"]))
        for _, row in tile_transects.iterrows()
    ]

    representative = tile_transects[tile_transects["bedform_count_canonical"] > 0]
    profile_data: dict[str, Any] = {
        "profile_distances_m": np.array([]),
        "profile_raw_detrended": np.array([]),
        "profile_filtered": np.array([]),
        "bedforms": [],
    }
    if not representative.empty:
        chosen = representative.iloc[representative["bedform_count_canonical"].argmax()]
        p1, p2 = (chosen["start_x_m"], chosen["start_y_m"]), (chosen["end_x_m"], chosen["end_y_m"])
        distances_m, elevations = _sample_transect(zip_path, grid_dir, p1, p2)
        if elevations is not None:
            detrended = swm.remove_linear_trend_1d(elevations, distances_m)
            filtered = swm.low_pass_filter_profile(
                detrended, 1.0, swm.CANONICAL_SHORT_WAVELENGTH_CUTOFF_M
            )
            bedforms = swm.compute_bedform_morphometrics(detrended, filtered, distances_m)
            profile_data = {
                "profile_distances_m": distances_m,
                "profile_raw_detrended": detrended,
                "profile_filtered": filtered,
                "bedforms": bedforms,
            }

    return {
        "native_elevation": elevation,
        "native_valid": valid,
        "filtered_detrended": filtered_surface,
        "pixel_size_m": pixel_size_m,
        "tile_extent_m": extent,
        "crest_azimuth_deg": top_tile_row["dominant_crest_azimuth_deg"],
        "transect_endpoints": transect_endpoints,
        **profile_data,
    }


def _spatial_short_wavelength_filter(residual: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """The 2D analogue of `low_pass_filter_profile`, for VISUAL display of
    the canonical sand-wave-scale surface (Section 22 panel B) -- an FFT
    hard cutoff in frequency space, matching `compute_2d_power_spectrum`'s
    own frequency convention."""

    spectrum = np.fft.fft2(residual)
    freq_y = np.fft.fftfreq(residual.shape[0], d=pixel_size_m)
    freq_x = np.fft.fftfreq(residual.shape[1], d=pixel_size_m)
    fx, fy = np.meshgrid(freq_x, freq_y)
    freq_mag = np.hypot(fx, fy)
    cutoff_freq = 1.0 / swm.CANONICAL_SHORT_WAVELENGTH_CUTOFF_M
    spectrum[freq_mag > cutoff_freq] = 0.0
    return np.real(np.fft.ifft2(spectrum))


# --- Section 25: pipeline-transfer contract -------------------------------------------------


def build_pipeline_transfer_contract() -> dict[str, Any]:
    return {
        "scientific_role": SCIENTIFIC_ROLE,
        **ANALOG_ONLY_FLAGS,
        "required_inputs_for_a_future_pl854_survey": [
            "gridded bathymetry, or a point cloud convertible to a regular grid",
            "known horizontal CRS",
            "grid resolution fine enough to resolve the target bedform scale "
            "(the canonical 30 m short-wavelength cutoff assumes a native pixel size well "
            "under 30 m -- e.g. <=2-5 m, as demonstrated on HHW's real 1 m grid)",
            "route geometry",
            "acquisition epoch",
        ],
        "strongly_preferred_inputs": [
            "vertical datum",
            "raw or processed MBES QA metadata",
            "full corridor coverage",
        ],
        "reusable_function_contract": {
            "module": "marine_engine.morphology.sandwave_morphometry",
            "accepts": [
                "an arbitrary bathymetry raster (numpy array + valid mask + pixel size in metres)",
                "arbitrary route/transect geometry (plain (x, y) endpoint tuples, no CRS assumed)",
                "scale parameters expressed in metres (short-wavelength cutoff, tile size, "
                "minimum valid fraction) -- never a hard-coded pixel count or HHW-specific "
                "coordinate",
            ],
            "no_hardcoded_hhw_specifics": True,
        },
        "limitations": [
            "Single-epoch morphology cannot provide a migration rate -- MAR-017 computes no "
            "migration rate anywhere.",
            "Asymmetry is descriptive geometry only, never automatically migration direction.",
            "Crest/trough detection depends on spatial resolution and the chosen filter "
            "cutoff -- see the filter-sensitivity QA (20/30/40 m) on every transect.",
            "Analog validation on HHW CEND 11/11 does not validate PL854 morphology in any way.",
            "A constant vertical datum offset does not change local relative relief/wavelength "
            "if spatially consistent, but the source vertical datum must still be preserved "
            "(HHW's is recorded as unknown -- this archive states no vertical datum field).",
            "No freespan relation is tested anywhere in MAR-017.",
        ],
        "references": [
            {
                "citation": "Knaapen, M.A.F. (2005). Sandwave migration predictor based on "
                "shape information. Journal of Geophysical Research: Earth Surface.",
                "doi": "10.1029/2004JF000195",
            },
            {
                "citation": "Barnard, P.L., Erikson, L.H., & Kvitek, R.G. (2011). Small-scale "
                "sediment transport patterns and bedform morphodynamics: New insights from "
                "high resolution multibeam bathymetry. Geo-Marine Letters.",
                "doi": "10.1007/s00367-011-0227-1",
            },
            {
                "citation": "Damen et al. (2018). Spatially Varying Environmental Properties "
                "Controlling Observed Sand Wave Morphology. Journal of Geophysical Research: "
                "Earth Surface.",
                "doi": "10.1002/2017JF004322",
            },
            {
                "citation": "JNCC / Cefas CEND 11/11 official dataset documentation.",
                "url": hhw_provider.HHW_SOURCE_CATALOGUE_URL,
            },
        ],
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

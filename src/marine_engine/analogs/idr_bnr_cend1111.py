"""IDRBNR CEND 11/11 canonical real-data sand-wave morphometry validation
orchestration (MAR-017B).

Second open analog, CANONICAL-ONLY (Section 1/9) -- never PL854 evidence
-------------------------------------------------------------------------------
MAR-017A established that HHW CEND 11/11's real processed bathymetry
cannot support the canonical (>=1000 m, >=90%-valid) real-data validation
protocol. This module tests the SAME reusable engine
(`marine_engine.morphology.sandwave_morphometry`) against a second,
independent official open dataset -- Inner Dowsing, Race Bank and North
Ridge cSAC, same CEND 11/11 survey programme, a distinct geographic
product. Unlike MAR-017A's HHW module, there is NO exploratory/below-
1000-m pipeline here at all (ticket Section 9: "Do not create exploratory
250/500 m results in this ticket") -- either the canonical protocol is
satisfied on real data, or the run stops early with an explicit,
honestly-reported reason. Every canonical row/output still carries
`scientific_role=HIGH_RESOLUTION_SANDBED_MORPHOMETRY_METHOD_DEVELOPMENT_
ANALOG` and `pl854_evidence=false`/`pl854_feature_input=false`/
`pl854_validation_input=false`/`method_development_analog_only=true`.

Canonical support preflight BEFORE any morphometry (Section 7/9)
-----------------------------------------------------------------------
This archive has 21 real raster candidates (18 ESRI grid directories + 3
standalone ESRI ASCII Grid files, `providers.bathymetry.idr_bnr_cend1111`)
-- every one is run through the SAME `swm.find_valid_tiles` (2000 m then
1000 m only, never lower) the engine already uses for canonical
validation, BEFORE any tile is selected or any spectral/morphometric
processing begins. If NOT ONE candidate has a single qualifying tile,
`derive_early_stop_status` returns `INSUFFICIENT_CONTINUOUS_SPATIAL_
SUPPORT` and the CLI orchestration stops there -- a valid, honestly-
reported scientific result, never a bug to route around.

Tile independence for detailed validation (Section 13)
-----------------------------------------------------------------
`swm.select_spatially_independent_eligible_tiles` (not HHW/MAR-017A's
plain `select_canonical_eligible_tiles`) both enforces the strict
`>=3`-wavelengths eligibility AND greedily rejects any ranked candidate
tile that spatially overlaps one already selected -- up to
`MAX_CANONICAL_TILES` (5), never more, never fewer than actually qualify.
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
from marine_engine.providers.bathymetry import idr_bnr_cend1111 as idrbnr_provider

SCIENTIFIC_ROLE = "HIGH_RESOLUTION_SANDBED_MORPHOMETRY_METHOD_DEVELOPMENT_ANALOG"
ANALOG_ONLY_FLAGS: dict[str, bool] = {
    "pl854_evidence": False,
    "pl854_feature_input": False,
    "pl854_validation_input": False,
    "method_development_analog_only": True,
}

ANALYSIS_DECIMATION = (
    2  # effective ~2 m/px scan for tile-candidate SEARCH; final tiles read at native res
)
MAX_CANONICAL_TILES = 5  # MAR-017B Section 13

# --- MAR-017B Section 24: the shared validation-status vocabulary (never a numeric score) -
CANONICAL_REAL_DATA_VALIDATED = "CANONICAL_REAL_DATA_VALIDATED"
INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT = "INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT"
INSUFFICIENT_WAVELENGTH_SUPPORT = "INSUFFICIENT_WAVELENGTH_SUPPORT"
INSUFFICIENT_VALID_TRANSECTS = "INSUFFICIENT_VALID_TRANSECTS"
INSUFFICIENT_CANONICAL_BEDFORMS = "INSUFFICIENT_CANONICAL_BEDFORMS"
DATA_INTEGRITY_FAILURE = "DATA_INTEGRITY_FAILURE"

# --- MAR-017B Section 17: two-state filter-stability classification, derived from the -----
# --- engine's own `swm.FILTER_SCALE_SENSITIVE` flag, never a new threshold -----------------
FILTER_STABLE_AT_TESTED_SCALES = "FILTER_STABLE_AT_TESTED_SCALES"
FILTER_SCALE_SENSITIVE = swm.FILTER_SCALE_SENSITIVE

CANONICAL_SUPPORT_PREFLIGHT_COLUMNS = (
    "raster_candidate_id",
    "storage_format",
    "crs",
    "crs_is_geographic",
    "pixel_scale_plausible",
    "width",
    "height",
    "native_pixel_size_m",
    "best_valid_fraction_2000m",
    "qualifying_tile_count_2000m",
    "best_valid_fraction_1000m",
    "qualifying_tile_count_1000m",
    "excluded_from_preflight",
    "exclusion_reason",
)

TILE_SPECTRAL_COLUMNS = (
    "tile_id",
    "raster_candidate_id",
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
    "selected_for_detailed_validation",
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
    "sub_cutoff_extrema_rejected_count",
    "median_wavelength_m_canonical",
    "median_wave_height_m_canonical",
    "filter_sensitivity_flags",
    "filter_stability_classification",
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
    zip_path: Path, candidate_id: str, decimation: int
) -> tuple[np.ndarray, np.ndarray, Affine, float]:
    vsi_path = f"/vsizip/{zip_path}/{candidate_id}"
    with rasterio.open(vsi_path) as src:
        out_h, out_w = max(src.height // decimation, 1), max(src.width // decimation, 1)
        data = src.read(1, out_shape=(out_h, out_w))
        nodata = src.nodata
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        pixel_size_m = abs(transform.a)
    valid = data != nodata if nodata is not None else np.ones_like(data, dtype=bool)
    return data, valid, transform, pixel_size_m


# --- Section 7: canonical support preflight -- MUST occur before any morphometry ------------


def run_canonical_support_preflight(zip_path: Path) -> pd.DataFrame:
    """Section 7: every real raster candidate in the archive, tested ONLY
    at 2000 m then 1000 m (`swm.find_valid_tiles`'s own built-in cascade)
    -- never lower. A candidate whose CRS is geographic while claiming a
    ~1 m native pixel step (`idrbnr_provider.is_pixel_scale_plausible`) is
    excluded from candidacy with an explicit reason, never silently
    evaluated on a nonsensical degrees-as-metres pixel size."""

    candidates = idrbnr_provider.list_raster_candidates(zip_path)
    rows = []
    for candidate in candidates:
        raster_candidate_id = candidate["raster_candidate_id"]
        crs_is_geographic = candidate.get("crs_is_geographic")
        plausible = crs_is_geographic is not True

        if not candidate.get("readable_by_rasterio"):
            rows.append(
                {
                    "raster_candidate_id": raster_candidate_id,
                    "storage_format": candidate.get("storage_format"),
                    "crs": candidate.get("crs"),
                    "crs_is_geographic": crs_is_geographic,
                    "pixel_scale_plausible": plausible,
                    "width": candidate.get("width"),
                    "height": candidate.get("height"),
                    "native_pixel_size_m": candidate.get("pixel_size_x_m"),
                    "best_valid_fraction_2000m": None,
                    "qualifying_tile_count_2000m": None,
                    "best_valid_fraction_1000m": None,
                    "qualifying_tile_count_1000m": None,
                    "excluded_from_preflight": True,
                    "exclusion_reason": "NOT_READABLE_BY_RASTERIO",
                }
            )
            continue

        if not plausible:
            rows.append(
                {
                    "raster_candidate_id": raster_candidate_id,
                    "storage_format": candidate.get("storage_format"),
                    "crs": candidate.get("crs"),
                    "crs_is_geographic": crs_is_geographic,
                    "pixel_scale_plausible": plausible,
                    "width": candidate.get("width"),
                    "height": candidate.get("height"),
                    "native_pixel_size_m": candidate.get("pixel_size_x_m"),
                    "best_valid_fraction_2000m": None,
                    "qualifying_tile_count_2000m": None,
                    "best_valid_fraction_1000m": None,
                    "qualifying_tile_count_1000m": None,
                    "excluded_from_preflight": True,
                    "exclusion_reason": "CRS_INCONSISTENT_WITH_NATIVE_PIXEL_SCALE",
                }
            )
            continue

        _data, valid, transform, pixel_size_m = _decimated_read(
            zip_path, raster_candidate_id, ANALYSIS_DECIMATION
        )
        _tiles, meta = swm.find_valid_tiles(valid, pixel_size_m, transform=transform)
        cascade_by_size = {c["tile_size_m"]: c for c in meta["cascade_log"]}
        entry_2000 = cascade_by_size.get(swm.CANONICAL_TILE_SIZE_M, {})
        entry_1000 = cascade_by_size.get(swm.MIN_TILE_SIZE_M, {})

        rows.append(
            {
                "raster_candidate_id": raster_candidate_id,
                "storage_format": candidate.get("storage_format"),
                "crs": candidate.get("crs"),
                "crs_is_geographic": crs_is_geographic,
                "pixel_scale_plausible": plausible,
                "width": candidate.get("width"),
                "height": candidate.get("height"),
                "native_pixel_size_m": candidate.get("pixel_size_x_m"),
                "best_valid_fraction_2000m": entry_2000.get("best_achieved_valid_fraction"),
                "qualifying_tile_count_2000m": entry_2000.get("passing_tile_count", 0),
                "best_valid_fraction_1000m": entry_1000.get("best_achieved_valid_fraction"),
                "qualifying_tile_count_1000m": entry_1000.get("passing_tile_count", 0),
                "excluded_from_preflight": False,
                "exclusion_reason": None,
            }
        )

    return pd.DataFrame(rows)[list(CANONICAL_SUPPORT_PREFLIGHT_COLUMNS)]


def derive_early_stop_status(preflight_df: pd.DataFrame) -> str | None:
    """Section 9's hard early-stop rule: if NOT ONE candidate has a single
    qualifying (>=90%-valid) tile at 1000 m or 2000 m, morphometry
    processing must not begin. Returns the explicit status string in that
    case, else None (processing may proceed)."""

    eligible = preflight_df[~preflight_df["excluded_from_preflight"]]
    if eligible.empty:
        return INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT
    any_qualifying = (
        (eligible["qualifying_tile_count_2000m"].fillna(0) > 0)
        | (eligible["qualifying_tile_count_1000m"].fillna(0) > 0)
    ).any()
    return None if any_qualifying else INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT


def select_primary_grid_from_preflight(preflight_df: pd.DataFrame) -> dict[str, Any]:
    """Section 8: select ONLY from candidates with >=1 qualifying >=1000 m
    tile, preferring (1) the largest supported tile size, (2) the most
    qualifying tiles, (3) the highest valid fraction -- never the
    candidate with merely the most total cells. Returns
    `selected_raster_candidate_id=None` if no candidate qualifies (the
    caller is expected to have already checked `derive_early_stop_status`
    first)."""

    qualifying = preflight_df[
        (~preflight_df["excluded_from_preflight"])
        & (
            (preflight_df["qualifying_tile_count_2000m"].fillna(0) > 0)
            | (preflight_df["qualifying_tile_count_1000m"].fillna(0) > 0)
        )
    ]
    if qualifying.empty:
        return {"selected_raster_candidate_id": None, "qualifying_candidates": []}

    def _score(row: pd.Series) -> tuple[float, float, float]:
        if row["qualifying_tile_count_2000m"] and row["qualifying_tile_count_2000m"] > 0:
            return (
                2000.0,
                row["qualifying_tile_count_2000m"],
                row["best_valid_fraction_2000m"] or 0.0,
            )
        return (1000.0, row["qualifying_tile_count_1000m"], row["best_valid_fraction_1000m"] or 0.0)

    scored = qualifying.assign(_score=qualifying.apply(_score, axis=1))
    best = scored.loc[scored["_score"].idxmax()]
    return {
        "selected_raster_candidate_id": best["raster_candidate_id"],
        "qualifying_candidates": qualifying["raster_candidate_id"].tolist(),
    }


# --- Sections 10-14: canonical tile search / spectral diagnostics / independent selection --
# --- / transects -- all direct calls into the already-accepted reusable engine, no new -----
# --- scientific equations (Section 11) ------------------------------------------------------


def build_tile_candidates(
    zip_path: Path, raster_candidate_id: str
) -> tuple[list[swm.TileCandidate], dict[str, Any]]:
    """CANONICAL tile search (Section 10): tries only 2000 m then 1000 m,
    at a modest decimation for search-phase speed -- final analysis
    always re-reads each candidate at NATIVE resolution
    (`read_native_window_by_center`). Legitimately returns an empty list
    if no >=1000 m tile meets the required valid fraction anywhere."""

    _data, valid, transform, pixel_size_m = _decimated_read(
        zip_path, raster_candidate_id, ANALYSIS_DECIMATION
    )
    tiles, meta = swm.find_valid_tiles(valid, pixel_size_m, transform=transform)
    return tiles, meta


def read_native_window_by_center(
    zip_path: Path,
    raster_candidate_id: str,
    *,
    center_x_m: float,
    center_y_m: float,
    tile_size_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """A NATIVE-resolution window read centred on a real-world (x, y),
    sized in real metres -- the tile candidate search runs on a decimated
    valid-mask for speed, so only the real-world centre/size are
    trustworthy across that resolution change; this recomputes native
    pixel indices itself."""

    vsi_path = f"/vsizip/{zip_path}/{raster_candidate_id}"
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
    valid = data != nodata if nodata is not None else np.ones_like(data, dtype=bool)
    return data.astype(np.float64), valid, pixel_size_m


def build_tile_spectral_table(
    zip_path: Path, raster_candidate_id: str, tiles: list[swm.TileCandidate]
) -> pd.DataFrame:
    """Section 11/12: one row per valid tile -- always re-reads each tile
    at NATIVE resolution before any spectral processing. Every row here IS
    canonical (this ticket has no exploratory pipeline, Section 9)."""

    rows = []
    for tile in tiles:
        elevation, valid, pixel_size_m = read_native_window_by_center(
            zip_path,
            raster_candidate_id,
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
                "raster_candidate_id": raster_candidate_id,
                "tile_size_m": tile.tile_size_m,
                "valid_fraction": tile.valid_fraction,
                "center_x_m": tile.center_x_m,
                "center_y_m": tile.center_y_m,
                **diagnostics,
                "meets_3_wavelengths_across_tile": meets_3wl,
                "selected_for_detailed_validation": False,
                **ANALOG_ONLY_FLAGS,
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(TILE_SPECTRAL_COLUMNS))
    df = pd.DataFrame(rows)
    return df[list(TILE_SPECTRAL_COLUMNS)]


def select_detailed_validation_tiles(
    tile_spectral_df: pd.DataFrame, *, max_tiles: int = MAX_CANONICAL_TILES
) -> pd.DataFrame:
    """Section 12/13: STRICT `>=3`-wavelengths eligibility (no fallback)
    AND spatial independence -- at most `max_tiles` (5) tiles, never
    overlapping, never forced if fewer qualify."""

    if tile_spectral_df.empty:
        return tile_spectral_df

    diagnostics_pool = [
        {
            "tile_id": row["tile_id"],
            "tile_size_m": row["tile_size_m"],
            "center_x_m": row["center_x_m"],
            "center_y_m": row["center_y_m"],
            "diagnostics": row.to_dict(),
        }
        for _, row in tile_spectral_df.iterrows()
    ]
    selected = swm.select_spatially_independent_eligible_tiles(
        diagnostics_pool, max_tiles=max_tiles
    )
    selected_ids = {d["tile_id"] for d in selected}
    result = tile_spectral_df.copy()
    result["selected_for_detailed_validation"] = result["tile_id"].isin(selected_ids)
    return result


def build_transect_and_bedform_tables(
    zip_path: Path, raster_candidate_id: str, top_tiles_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Sections 14-17: three cross-crest transects per selected tile
    (`top_tiles_df` rows with `selected_for_detailed_validation=True`),
    bedform detection through the explicit 30 m wavelength gate plus
    20/40 m filter-sensitivity QA (classified as `FILTER_STABLE_AT_
    TESTED_SCALES`/`FILTER_SCALE_SENSITIVE`, Section 17), and the
    transect-based crest/trough point layers."""

    transect_rows: list[dict[str, Any]] = []
    bedform_rows: list[dict[str, Any]] = []
    crest_points: list[dict[str, Any]] = []
    trough_points: list[dict[str, Any]] = []

    selected = top_tiles_df[top_tiles_df["selected_for_detailed_validation"]]
    for _, tile_row in selected.iterrows():
        crest_azimuth = tile_row["dominant_crest_azimuth_deg"]
        endpoints = swm.generate_cross_crest_transects(
            tile_row["center_x_m"], tile_row["center_y_m"], tile_row["tile_size_m"], crest_azimuth
        )
        for offset_fraction, (p1, p2) in zip((-0.25, 0.0, 0.25), endpoints, strict=True):
            transect_id = f"{tile_row['tile_id']}_offset_{offset_fraction:+.2f}"
            distances_m, elevations = _sample_transect(zip_path, raster_candidate_id, p1, p2)
            if elevations is None or len(elevations) < 10:
                # Recorded (never silently dropped, Section 17's "do not hide instability"
                # ethos applied here too) -- this transect's own sampled length did not meet
                # the 90% validity bar.
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
                        "sub_cutoff_extrema_rejected_count": None,
                        "median_wavelength_m_canonical": None,
                        "median_wave_height_m_canonical": None,
                        "filter_sensitivity_flags": "EXCLUDED_INSUFFICIENT_VALID_DATA",
                        "filter_stability_classification": None,
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
            detected_bedforms = swm.compute_bedform_morphometrics(
                detrended, filtered_canonical, distances_m
            )
            # Section 15: the Butterworth filter attenuates rather than mathematically
            # zeroes sub-cutoff content -- the explicit hard gate enforces the canonical
            # minimum wavelength; anything rejected is counted, never silently dropped.
            bedforms, sub_cutoff_rejected_count = swm.apply_canonical_wavelength_gate(
                detected_bedforms
            )
            sensitivity = swm.run_filter_sensitivity_qa(detrended, distances_m, pixel_size_m)
            filter_stability_classification = (
                FILTER_SCALE_SENSITIVE if sensitivity["flags"] else FILTER_STABLE_AT_TESTED_SCALES
            )

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
                    "sub_cutoff_extrema_rejected_count": sub_cutoff_rejected_count,
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
                    "filter_stability_classification": filter_stability_classification,
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
    zip_path: Path, raster_candidate_id: str, p1: tuple[float, float], p2: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray | None]:
    vsi_path = f"/vsizip/{zip_path}/{raster_candidate_id}"
    line = LineString([p1, p2])
    with rasterio.open(vsi_path) as src:
        pixel_size_m = abs(src.transform.a)
        n_samples = max(int(line.length / pixel_size_m), 2)
        points = [line.interpolate(i / (n_samples - 1), normalized=True) for i in range(n_samples)]
        coords = [(pt.x, pt.y) for pt in points]
        sampled = list(src.sample(coords))
        nodata = src.nodata
    values = np.array([s[0] for s in sampled], dtype=np.float64)
    valid = values != nodata if nodata is not None else np.ones_like(values, dtype=bool)
    if valid.mean() < 0.9:
        return np.array([]), None
    distances = np.linspace(0.0, line.length, n_samples)
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


# --- Map input helpers ------------------------------------------------------------------------


def get_background_for_map(
    zip_path: Path, raster_candidate_id: str, *, decimation: int = 8
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """A decimated full-grid read for map-background context only (never
    used for morphometric analysis itself, which always reads native
    resolution)."""

    data, valid, transform, _pixel_size_m = _decimated_read(
        zip_path, raster_candidate_id, decimation
    )
    height, width = data.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return data, valid, (left, right, bottom, top)


def get_method_figure_inputs(
    zip_path: Path,
    raster_candidate_id: str,
    top_tile_row: pd.Series,
    transect_df: pd.DataFrame,
    bedform_df: pd.DataFrame,
) -> dict[str, Any]:
    """Native elevation/valid + filtered/detrended surface for the single
    highest-ranked selected tile, plus one representative transect's
    profile and its detected bedforms, for the primary method/validation
    figure (Section 20)."""

    elevation, valid, pixel_size_m = read_native_window_by_center(
        zip_path,
        raster_candidate_id,
        center_x_m=top_tile_row["center_x_m"],
        center_y_m=top_tile_row["center_y_m"],
        tile_size_m=top_tile_row["tile_size_m"],
    )
    residual, _trend, _coeffs = swm.remove_planar_trend(elevation, valid, pixel_size_m)
    residual_filled = swm.fill_small_gaps(residual, valid)
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
        distances_m, elevations = _sample_transect(zip_path, raster_candidate_id, p1, p2)
        if elevations is not None:
            detrended = swm.remove_linear_trend_1d(elevations, distances_m)
            filtered = swm.low_pass_filter_profile(
                detrended, 1.0, swm.CANONICAL_SHORT_WAVELENGTH_CUTOFF_M
            )
            detected_bedforms = swm.compute_bedform_morphometrics(detrended, filtered, distances_m)
            bedforms, _rejected = swm.apply_canonical_wavelength_gate(detected_bedforms)
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
    spectrum = np.fft.fft2(residual)
    freq_y = np.fft.fftfreq(residual.shape[0], d=pixel_size_m)
    freq_x = np.fft.fftfreq(residual.shape[1], d=pixel_size_m)
    fx, fy = np.meshgrid(freq_x, freq_y)
    freq_mag = np.hypot(fx, fy)
    cutoff_freq = 1.0 / swm.CANONICAL_SHORT_WAVELENGTH_CUTOFF_M
    spectrum[freq_mag > cutoff_freq] = 0.0
    return np.real(np.fft.ifft2(spectrum))


# --- Section 18: canonical real-validation acceptance -- an explicit, ordered waterfall ----
# --- over criteria A-D; criteria E (no nodata corruption) and F (no PL854 identifier in ----
# --- the reusable engine) are static code-quality invariants verified once by tests, never -
# --- re-derived per run --------------------------------------------------------------------


def derive_canonical_real_validation_status(
    *,
    canonical_tile_count: int,
    any_meets_3_wavelengths: bool,
    successful_transect_count: int,
    canonical_bedform_count: int,
) -> tuple[str, str]:
    """MAR-017B Section 18: returns (status, reason). Never manufactures
    `CANONICAL_REAL_DATA_VALIDATED` by relaxing a criterion -- each gate
    is checked in order and the FIRST one that fails determines the
    status."""

    if canonical_tile_count < 1:
        return (
            INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT,
            "Criterion A failed: zero canonical (>=1000 m, >=90%-valid) tiles were found.",
        )
    if not any_meets_3_wavelengths:
        return (
            INSUFFICIENT_WAVELENGTH_SUPPORT,
            "Criterion B failed: no canonical tile satisfies tile_size_m/dominant_wavelength_m "
            ">= 3.0.",
        )
    if successful_transect_count < 1:
        return (
            INSUFFICIENT_VALID_TRANSECTS,
            "Criterion C failed: zero cross-crest transects met the >=90%-valid-samples bar.",
        )
    if canonical_bedform_count < 3:
        return (
            INSUFFICIENT_CANONICAL_BEDFORMS,
            f"Criterion D failed: only {canonical_bedform_count} canonical (>=30 m) bedform(s) "
            "were retained; at least 3 are required.",
        )
    return (
        CANONICAL_REAL_DATA_VALIDATED,
        "Criteria A-D all satisfied: >=1 canonical tile, >=1 tile meeting the >=3-wavelengths "
        "condition, >=1 successful transect, and >=3 canonical bedforms.",
    )


# --- Section 25: pipeline-transfer contract -------------------------------------------------


def build_pipeline_transfer_contract(
    *,
    canonical_real_validation_status: str,
    canonical_real_validation_reason: str,
) -> dict[str, Any]:
    """MAR-017B Section 25: if canonical validation passes, the contract
    states `canonical_real_data_validation_passed=true` and names IDRBNR
    as `validated_on_analog_dataset`; if it fails (on this analog too),
    `canonical_real_data_validation_passed=false` with the exact reason
    preserved. `PL854_real_data_validation_passed` is always false here --
    this is still an analog, never PL854 evidence."""

    passed = canonical_real_validation_status == CANONICAL_REAL_DATA_VALIDATED
    return {
        "scientific_role": SCIENTIFIC_ROLE,
        **ANALOG_ONLY_FLAGS,
        "generic_scientific_core_implemented": True,
        "synthetic_validation_passed": True,
        "canonical_real_data_validation_passed": passed,
        "canonical_real_data_validation_status": canonical_real_validation_status,
        "canonical_real_data_validation_reason": canonical_real_validation_reason,
        "validated_on_analog_dataset": "IDRBNR_CEND1111" if passed else None,
        "PL854_real_data_validation_passed": False,
        "reusable_function_contract": {
            "module": "marine_engine.morphology.sandwave_morphometry",
            "accepts": [
                "an arbitrary bathymetry raster (numpy array + valid mask + pixel size in metres)",
                "arbitrary route/transect geometry (plain (x, y) endpoint tuples, no CRS assumed)",
                "scale parameters expressed in metres (short-wavelength cutoff, tile size, "
                "minimum valid fraction) -- never a hard-coded pixel count or dataset-specific "
                "coordinate",
            ],
            "no_hardcoded_dataset_specifics": True,
        },
        "limitations": [
            "Single-epoch morphology cannot provide a migration rate -- no migration rate is "
            "computed anywhere in this module.",
            "Asymmetry is descriptive geometry only, never automatically migration direction.",
            "Analog validation on IDRBNR CEND 11/11 does not validate PL854 morphology in any "
            "way, regardless of this analog's own validation outcome.",
            "No freespan relation, scour prediction, or risk score is tested anywhere here.",
        ],
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

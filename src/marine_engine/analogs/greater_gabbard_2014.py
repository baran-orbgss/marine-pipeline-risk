"""Greater Gabbard 2014 (ADUS DeepOcean) FINAL canonical real-data
sand-wave morphometry validation orchestration (MAR-017C).

Third and FINAL open analog (Section 1) -- never PL854 evidence
-------------------------------------------------------------------------
HHW (MAR-017A) and IDRBNR (MAR-017B) both failed the canonical (>=1000 m,
>=90%-valid) real-data validation protocol because their real processed
bathymetry is genuinely swath/patch-like. This module tests the SAME
reusable engine (`marine_engine.morphology.sandwave_morphometry`) one
final time against Greater Gabbard 2014 -- a structurally DIFFERENT kind
of real-data limitation: the survey was scoped entirely around individual
engineering features (one small area per turbine foundation, one narrow
corridor per inter-array cable route), never a continuous regional survey
at all (`providers.bathymetry.greater_gabbard_2014`'s real archive
inspection: largest confirmed candidate is 777 m x 509 m). Every canonical
output still carries `scientific_role=HIGH_RESOLUTION_SANDBED_
MORPHOMETRY_METHOD_DEVELOPMENT_ANALOG` and
`pl854_evidence=false`/`pl854_feature_input=false`/
`pl854_validation_input=false`/`method_development_analog_only=true`.

Anthropogenic-disturbance problem (Sections 9-12) -- unique to this analog
-----------------------------------------------------------------------------
Unlike HHW/IDRBNR (open-seabed characterisation surveys), Greater Gabbard
is an OPERATING WIND FARM construction survey -- any qualifying canonical
tile could plausibly be dominated by monopile scour, cable disturbance, or
rock/concrete protection rather than natural bedforms. This module derives
an infrastructure-context inventory DIRECTLY from the bathymetry package's
own file structure (no second large GIS-package download, Section 10):

- Turbine/monopile positions: the centroid of each of the 144
  `Foundations/GRIDDED 0.25x0.25/*.txt` grids IS that turbine's location
  (one grid per foundation, by construction) -- the foundation named with
  "SUB" (`GASUB`/`IGSUB`) is the substation, not a turbine.
- Inter-array cable routes: `Corridors/ALL ASCII/<A>-<B>.txt` entry NAMES
  encode the two foundation IDs a corridor connects -- approximated as the
  straight line between those two foundations' own centroids (Section 10:
  "do not require a second large download unless necessary" -- the
  corridor's OWN point-cloud data is never downloaded for this purpose).
- Rock/concrete protection: the bounding box of each of the 32 `Concrete
  mattressing/*.txt` files.

`assess_natural_seabed_eligibility` never invents a generic hazard buffer
(Section 11) -- it only rejects a tile that directly CONTAINS a turbine/
substation foundation point or overlaps a mattressing bounding box, and
separately reports (never conflates) the nearest-cable distance as
context.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from marine_engine.morphology import sandwave_morphometry as swm
from marine_engine.providers.bathymetry import greater_gabbard_2014 as gg_provider

SCIENTIFIC_ROLE = "HIGH_RESOLUTION_SANDBED_MORPHOMETRY_METHOD_DEVELOPMENT_ANALOG"
ANALOG_ONLY_FLAGS: dict[str, bool] = {
    "pl854_evidence": False,
    "pl854_feature_input": False,
    "pl854_validation_input": False,
    "method_development_analog_only": True,
}

MAX_CANONICAL_TILES = 5  # Section 12

# --- MAR-017C Section 15: the shared validation-status vocabulary (never a numeric score) --
CANONICAL_REAL_DATA_VALIDATED = "CANONICAL_REAL_DATA_VALIDATED"
INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT = "INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT"
INSUFFICIENT_WAVELENGTH_SUPPORT = "INSUFFICIENT_WAVELENGTH_SUPPORT"
NO_NATURAL_SEABED_VALIDATION_TILE = "NO_NATURAL_SEABED_VALIDATION_TILE"
INFRASTRUCTURE_CONTEXT_INSUFFICIENT = "INFRASTRUCTURE_CONTEXT_INSUFFICIENT"
INSUFFICIENT_VALID_TRANSECTS = "INSUFFICIENT_VALID_TRANSECTS"
INSUFFICIENT_CANONICAL_BEDFORMS = "INSUFFICIENT_CANONICAL_BEDFORMS"
DATA_INTEGRITY_FAILURE = "DATA_INTEGRITY_FAILURE"

# --- Section 11: natural-seabed eligibility vocabulary --------------------------------------
NATURAL_SEABED_VALIDATION_ELIGIBLE = "NATURAL_SEABED_VALIDATION_ELIGIBLE"
ANTHROPOGENIC_DISTURBANCE_PRESENT = "ANTHROPOGENIC_DISTURBANCE_PRESENT"

# --- Section 17: two-state filter-stability classification, matching MAR-017B --------------
FILTER_STABLE_AT_TESTED_SCALES = "FILTER_STABLE_AT_TESTED_SCALES"
FILTER_SCALE_SENSITIVE = swm.FILTER_SCALE_SENSITIVE

SUBSTATION_FOUNDATION_MARKER = (
    "SUB"  # a data-driven substring check (GASUB/IGSUB), never a hard-coded name
)

CANONICAL_SUPPORT_PREFLIGHT_COLUMNS = (
    "raster_candidate_id",
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

INFRASTRUCTURE_CONTEXT_COLUMNS = (
    "infrastructure_id",
    "infrastructure_type",
    "center_x_m",
    "center_y_m",
    "min_x_m",
    "min_y_m",
    "max_x_m",
    "max_y_m",
    "is_substation",
    *ANALOG_ONLY_FLAGS.keys(),
    "scientific_role",
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
    "natural_seabed_eligibility_status",
    "nearest_turbine_distance_m",
    "nearest_cable_distance_m",
    "overlaps_rock_protection",
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


# --- Section 7: canonical support preflight -- MUST occur before any morphometry ------------


def run_canonical_support_preflight(raw_dir: Path) -> pd.DataFrame:
    """Section 7: every real properly-gridded (square-tile-capable)
    candidate in the archive -- BOTH `Foundations/GRIDDED 0.25x0.25`
    (144) and `Corridors/GRIDDED 0.5x0.5` (152) grids (Section 5) --
    tested ONLY at 2000 m then 1000 m (`swm.find_valid_tiles`'s own
    built-in cascade), never lower."""

    candidates = gg_provider.list_all_canonical_candidates(raw_dir)
    rows = []
    for path, pixel_size_m_hint in candidates:
        elevation, valid, transform, pixel_size_m = gg_provider.load_xyz_regular_grid(
            path, pixel_size_m=pixel_size_m_hint
        )
        if valid.size == 0:
            rows.append(
                {
                    "raster_candidate_id": path.stem,
                    "width": 0,
                    "height": 0,
                    "native_pixel_size_m": pixel_size_m,
                    "best_valid_fraction_2000m": None,
                    "qualifying_tile_count_2000m": 0,
                    "best_valid_fraction_1000m": None,
                    "qualifying_tile_count_1000m": 0,
                    "excluded_from_preflight": True,
                    "exclusion_reason": "EMPTY_GRID",
                }
            )
            continue

        _tiles, meta = swm.find_valid_tiles(valid, pixel_size_m, transform=transform)
        cascade_by_size = {c["tile_size_m"]: c for c in meta["cascade_log"]}
        entry_2000 = cascade_by_size.get(swm.CANONICAL_TILE_SIZE_M, {})
        entry_1000 = cascade_by_size.get(swm.MIN_TILE_SIZE_M, {})
        rows.append(
            {
                "raster_candidate_id": path.stem,
                "width": elevation.shape[1],
                "height": elevation.shape[0],
                "native_pixel_size_m": pixel_size_m,
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
    """Section 8's hard early-stop rule: if NOT ONE candidate has a single
    qualifying (>=90%-valid) tile at 1000 m or 2000 m, morphometry
    processing must not begin."""

    eligible = preflight_df[~preflight_df["excluded_from_preflight"]]
    if eligible.empty:
        return INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT
    any_qualifying = (
        (eligible["qualifying_tile_count_2000m"].fillna(0) > 0)
        | (eligible["qualifying_tile_count_1000m"].fillna(0) > 0)
    ).any()
    return None if any_qualifying else INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT


def select_primary_grid_from_preflight(preflight_df: pd.DataFrame) -> dict[str, Any]:
    """Selects ONLY from candidates with >=1 qualifying tile, preferring
    (1) the largest supported tile size, (2) the most qualifying tiles,
    (3) the highest valid fraction."""

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


# --- Sections 9-10: infrastructure-context inventory, derived from the bathymetry package's -
# --- own file structure -- never a second large GIS-package download (Section 10) ----------


def build_infrastructure_context_inventory(raw_dir: Path) -> pd.DataFrame:
    """Section 10: turbine/monopile positions (one per `Foundations/
    GRIDDED 0.25x0.25/*.txt` grid's own centroid -- a foundation named
    with "SUB" is the substation, not a turbine), plus rock/concrete
    protection footprints (one per `Concrete mattressing/*.txt` file's
    bounding box). Cable corridors are NOT enumerated here as their own
    rows (Section 10: no second download of their point-cloud content is
    needed) -- `build_cable_corridor_segments` derives them separately,
    directly from the remote archive's entry NAMES plus this inventory's
    own turbine positions."""

    rows = []
    for path in gg_provider.list_gridded_ascii_candidates(raw_dir):
        elevation, valid, transform, _pixel_size_m = gg_provider.load_xyz_regular_grid(path)
        if valid.size == 0 or not valid.any():
            continue
        rows_idx, cols_idx = np.where(valid)
        xs, ys = transform * (cols_idx, rows_idx)
        foundation_id = path.stem
        rows.append(
            {
                "infrastructure_id": foundation_id,
                "infrastructure_type": "turbine_or_substation_foundation",
                "center_x_m": float(np.mean(xs)),
                "center_y_m": float(np.mean(ys)),
                "min_x_m": float(np.min(xs)),
                "min_y_m": float(np.min(ys)),
                "max_x_m": float(np.max(xs)),
                "max_y_m": float(np.max(ys)),
                "is_substation": SUBSTATION_FOUNDATION_MARKER in foundation_id.upper(),
                **ANALOG_ONLY_FLAGS,
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )

    for path in gg_provider.list_mattressing_files(raw_dir):
        elevation, valid, transform, _pixel_size_m = gg_provider.load_xyz_regular_grid(path)
        if valid.size == 0 or not valid.any():
            continue
        rows_idx, cols_idx = np.where(valid)
        xs, ys = transform * (cols_idx, rows_idx)
        rows.append(
            {
                "infrastructure_id": path.stem,
                "infrastructure_type": "rock_concrete_protection",
                "center_x_m": float(np.mean(xs)),
                "center_y_m": float(np.mean(ys)),
                "min_x_m": float(np.min(xs)),
                "min_y_m": float(np.min(ys)),
                "max_x_m": float(np.max(xs)),
                "max_y_m": float(np.max(ys)),
                "is_substation": False,
                **ANALOG_ONLY_FLAGS,
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(INFRASTRUCTURE_CONTEXT_COLUMNS))
    return pd.DataFrame(rows)[list(INFRASTRUCTURE_CONTEXT_COLUMNS)]


def build_cable_corridor_segments(
    remote_entries: list[str], infrastructure_df: pd.DataFrame
) -> list[dict[str, Any]]:
    """Section 10: a cable corridor's real point-cloud data is never
    downloaded (Section 10: no second download); each `Corridors/ALL
    ASCII/<A>-<B>.txt` entry NAME already encodes the two foundation IDs
    it connects, so the corridor is approximated as the straight line
    between those two foundations' own already-known centroids. A
    corridor naming an unrecognised foundation ID is skipped, never
    guessed."""

    centroid_by_id = {
        row["infrastructure_id"]: (row["center_x_m"], row["center_y_m"])
        for _, row in infrastructure_df.iterrows()
    }
    segments = []
    prefix = gg_provider.CORRIDOR_ALL_ASCII_PREFIX
    for entry in remote_entries:
        if not entry.startswith(prefix) or not entry.lower().endswith(".txt"):
            continue
        stem = entry[len(prefix) : -len(".txt")]
        if "-" not in stem:
            continue
        id_a, _, id_b = stem.partition("-")
        if id_a not in centroid_by_id or id_b not in centroid_by_id:
            continue
        x1, y1 = centroid_by_id[id_a]
        x2, y2 = centroid_by_id[id_b]
        segments.append({"corridor_id": stem, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return segments


def _point_to_segment_distance_m(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    return Point(px, py).distance(LineString([(x1, y1), (x2, y2)]))


# --- Section 11: natural-seabed eligibility --------------------------------------------------


def assess_natural_seabed_eligibility(
    *,
    tile_center_x_m: float,
    tile_center_y_m: float,
    tile_size_m: float,
    infrastructure_df: pd.DataFrame,
    cable_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Section 11: a canonical tile may proceed to sand-wave validation
    only if it does not directly CONTAIN a turbine/substation foundation
    point or overlap a mapped rock-protection bounding box -- never a
    generic hazard buffer (Section 11: "do not invent"). Cable proximity
    is preserved as descriptive context (`nearest_cable_distance_m`),
    never itself a rejection criterion (Section 11 lists cables under
    "preserve distance/intersection context", not the hard-rejection
    list). Returns `INFRASTRUCTURE_CONTEXT_INSUFFICIENT` if no
    infrastructure context exists at all to assess against -- never
    silently defaults to eligible in that case."""

    if infrastructure_df.empty and not cable_segments:
        return {
            "status": INFRASTRUCTURE_CONTEXT_INSUFFICIENT,
            "nearest_turbine_distance_m": None,
            "nearest_cable_distance_m": None,
            "overlaps_rock_protection": None,
        }

    half = tile_size_m / 2.0
    tile_min_x, tile_max_x = tile_center_x_m - half, tile_center_x_m + half
    tile_min_y, tile_max_y = tile_center_y_m - half, tile_center_y_m + half

    foundations = infrastructure_df[
        infrastructure_df["infrastructure_type"] == "turbine_or_substation_foundation"
    ]
    rock_protection = infrastructure_df[
        infrastructure_df["infrastructure_type"] == "rock_concrete_protection"
    ]

    contains_foundation = bool(
        (
            (foundations["center_x_m"] >= tile_min_x)
            & (foundations["center_x_m"] <= tile_max_x)
            & (foundations["center_y_m"] >= tile_min_y)
            & (foundations["center_y_m"] <= tile_max_y)
        ).any()
    )
    overlaps_rock_protection = bool(
        (
            (rock_protection["min_x_m"] <= tile_max_x)
            & (rock_protection["max_x_m"] >= tile_min_x)
            & (rock_protection["min_y_m"] <= tile_max_y)
            & (rock_protection["max_y_m"] >= tile_min_y)
        ).any()
    )

    nearest_turbine_distance_m = (
        float(
            np.hypot(
                foundations["center_x_m"] - tile_center_x_m,
                foundations["center_y_m"] - tile_center_y_m,
            ).min()
        )
        if not foundations.empty
        else None
    )
    nearest_cable_distance_m = (
        float(
            min(
                _point_to_segment_distance_m(
                    tile_center_x_m, tile_center_y_m, seg["x1"], seg["y1"], seg["x2"], seg["y2"]
                )
                for seg in cable_segments
            )
        )
        if cable_segments
        else None
    )

    status = (
        ANTHROPOGENIC_DISTURBANCE_PRESENT
        if (contains_foundation or overlaps_rock_protection)
        else NATURAL_SEABED_VALIDATION_ELIGIBLE
    )
    return {
        "status": status,
        "nearest_turbine_distance_m": nearest_turbine_distance_m,
        "nearest_cable_distance_m": nearest_cable_distance_m,
        "overlaps_rock_protection": overlaps_rock_protection,
    }


def _resolve_candidate(raw_dir: Path, raster_candidate_id: str) -> tuple[Path, float]:
    """Maps a `raster_candidate_id` (a bare filename stem, e.g. `IGSUB` or
    `GAA01-GAB02`) back to its real local path and correct native pixel
    size -- foundation and corridor candidates live in different
    directories at different resolutions (Section 5/6: never guess a
    scale). Foundation and corridor IDs never collide (a foundation ID
    never contains a hyphen; a corridor ID always does, encoding its two
    endpoint foundation IDs)."""

    foundation_path = raw_dir / gg_provider.GRIDDED_ASCII_PREFIX / f"{raster_candidate_id}.txt"
    if foundation_path.exists():
        return foundation_path, gg_provider.NATIVE_PIXEL_SIZE_M
    corridor_path = raw_dir / gg_provider.CORRIDOR_GRIDDED_PREFIX / f"{raster_candidate_id}.txt"
    if corridor_path.exists():
        return corridor_path, gg_provider.CORRIDOR_NATIVE_PIXEL_SIZE_M
    raise FileNotFoundError(
        f"raster_candidate_id={raster_candidate_id!r} not found under either the foundation or "
        "corridor GRIDDED directories."
    )


# --- Sections 13-14: spectral diagnostics / transects -- direct calls into the already- ----
# --- accepted reusable engine, no new scientific equations (Section 13) ---------------------


def build_tile_candidates(raw_dir: Path, raster_candidate_id: str) -> tuple[list, dict[str, Any]]:
    path, pixel_size_m_hint = _resolve_candidate(raw_dir, raster_candidate_id)
    _elevation, valid, transform, pixel_size_m = gg_provider.load_xyz_regular_grid(
        path, pixel_size_m=pixel_size_m_hint
    )
    return swm.find_valid_tiles(valid, pixel_size_m, transform=transform)


def build_tile_spectral_table(
    raw_dir: Path,
    raster_candidate_id: str,
    tiles: list,
    *,
    infrastructure_df: pd.DataFrame,
    cable_segments: list[dict[str, Any]],
) -> pd.DataFrame:
    """One row per valid tile, ALWAYS carrying the natural-seabed
    eligibility assessment (Section 11) alongside the spectral
    diagnostics -- eligibility is assessed independently of ranking
    (Section 22 test D), never inferred from spectral quality."""

    path, pixel_size_m_hint = _resolve_candidate(raw_dir, raster_candidate_id)
    full_elevation, full_valid, full_transform, native_pixel_size_m = (
        gg_provider.load_xyz_regular_grid(path, pixel_size_m=pixel_size_m_hint)
    )
    inverse_transform = ~full_transform

    rows = []
    for tile in tiles:
        col_center, row_center = inverse_transform * (tile.center_x_m, tile.center_y_m)
        size_px = tile.tile_size_px
        col_off = int(round(col_center - size_px / 2.0))
        row_off = int(round(row_center - size_px / 2.0))
        elevation = full_elevation[row_off : row_off + size_px, col_off : col_off + size_px]
        valid = full_valid[row_off : row_off + size_px, col_off : col_off + size_px]

        diagnostics = swm.analyze_tile(elevation, valid, native_pixel_size_m)
        if diagnostics is None:
            continue
        meets_3wl = swm.meets_wavelengths_across_tile(
            diagnostics["dominant_wavelength_m"], tile.tile_size_m
        )
        eligibility = assess_natural_seabed_eligibility(
            tile_center_x_m=tile.center_x_m,
            tile_center_y_m=tile.center_y_m,
            tile_size_m=tile.tile_size_m,
            infrastructure_df=infrastructure_df,
            cable_segments=cable_segments,
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
                "natural_seabed_eligibility_status": eligibility["status"],
                "nearest_turbine_distance_m": eligibility["nearest_turbine_distance_m"],
                "nearest_cable_distance_m": eligibility["nearest_cable_distance_m"],
                "overlaps_rock_protection": eligibility["overlaps_rock_protection"],
                "selected_for_detailed_validation": False,
                **ANALOG_ONLY_FLAGS,
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(TILE_SPECTRAL_COLUMNS))
    return pd.DataFrame(rows)[list(TILE_SPECTRAL_COLUMNS)]


def select_detailed_validation_tiles(
    tile_spectral_df: pd.DataFrame, *, max_tiles: int = MAX_CANONICAL_TILES
) -> pd.DataFrame:
    """Section 12 A-F: strict >=3-wavelengths eligibility AND natural-
    seabed eligibility (Section 12E) AND spatial independence (12F) --
    only tiles satisfying ALL of these may be ranked, at most `max_tiles`
    (5)."""

    if tile_spectral_df.empty:
        return tile_spectral_df

    natural_only = tile_spectral_df[
        tile_spectral_df["natural_seabed_eligibility_status"] == NATURAL_SEABED_VALIDATION_ELIGIBLE
    ]
    diagnostics_pool = [
        {
            "tile_id": row["tile_id"],
            "tile_size_m": row["tile_size_m"],
            "center_x_m": row["center_x_m"],
            "center_y_m": row["center_y_m"],
            "diagnostics": row.to_dict(),
        }
        for _, row in natural_only.iterrows()
    ]
    selected = swm.select_spatially_independent_eligible_tiles(
        diagnostics_pool, max_tiles=max_tiles
    )
    selected_ids = {d["tile_id"] for d in selected}
    result = tile_spectral_df.copy()
    result["selected_for_detailed_validation"] = result["tile_id"].isin(selected_ids)
    return result


# --- Sections 14/17: cross-crest transects + bedform detection -- these grids are already ---
# --- small enough to load fully into memory ONCE per candidate (never rasterio/vsizip; the ---
# --- custom XYZ parser produces a plain in-memory array), so transect sampling is direct -----
# --- array indexing rather than a native-window re-read per transect ------------------------


def _sample_transect_from_array(
    elevation: np.ndarray,
    valid: np.ndarray,
    transform,
    p1: tuple[float, float],
    p2: tuple[float, float],
    pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    line_length_m = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    n_samples = max(int(line_length_m / pixel_size_m), 2)
    distances_m = np.linspace(0.0, line_length_m, n_samples)
    xs = np.linspace(p1[0], p2[0], n_samples)
    ys = np.linspace(p1[1], p2[1], n_samples)
    inverse_transform = ~transform
    cols, rows = inverse_transform * (xs, ys)
    cols = np.round(cols).astype(np.int64)
    rows = np.round(rows).astype(np.int64)
    in_bounds = (
        (rows >= 0) & (rows < elevation.shape[0]) & (cols >= 0) & (cols < elevation.shape[1])
    )

    values = np.full(n_samples, np.nan, dtype=np.float64)
    sample_valid = np.zeros(n_samples, dtype=bool)
    values[in_bounds] = elevation[rows[in_bounds], cols[in_bounds]]
    sample_valid[in_bounds] = valid[rows[in_bounds], cols[in_bounds]]

    if sample_valid.mean() < 0.9:
        return np.array([]), None
    if not sample_valid.all():
        values = np.interp(distances_m, distances_m[sample_valid], values[sample_valid])
    return distances_m, values


def _distance_to_xy(
    p1: tuple[float, float], p2: tuple[float, float], distance_m: float, total_length_m: float
) -> tuple[float, float]:
    frac = distance_m / total_length_m if total_length_m > 0 else 0.0
    x = p1[0] + frac * (p2[0] - p1[0])
    y = p1[1] + frac * (p2[1] - p1[1])
    return x, y


def build_transect_and_bedform_tables(
    raw_dir: Path, raster_candidate_id: str, top_tiles_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Sections 14/16-17: three cross-crest transects per selected tile
    (`top_tiles_df` rows with `selected_for_detailed_validation=True`),
    bedform detection through the explicit 30 m wavelength gate plus
    20/40 m filter-sensitivity QA, and the transect-based crest/trough
    point layers -- identical scientific definitions to HHW/IDRBNR
    (Section 13: no scientific-core change)."""

    path, pixel_size_m_hint = _resolve_candidate(raw_dir, raster_candidate_id)
    full_elevation, full_valid, full_transform, native_pixel_size_m = (
        gg_provider.load_xyz_regular_grid(path, pixel_size_m=pixel_size_m_hint)
    )

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
            distances_m, elevations = _sample_transect_from_array(
                full_elevation, full_valid, full_transform, p1, p2, native_pixel_size_m
            )
            if elevations is None or len(elevations) < 10:
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
        gpd.GeoDataFrame(crest_points, geometry="geometry", crs=None)
        if crest_points
        else gpd.GeoDataFrame(columns=["point_id", "transect_id", "geometry"], geometry="geometry")
    )
    troughs_gdf = (
        gpd.GeoDataFrame(trough_points, geometry="geometry", crs=None)
        if trough_points
        else gpd.GeoDataFrame(columns=["point_id", "transect_id", "geometry"], geometry="geometry")
    )
    return transect_df, bedform_df, crests_gdf, troughs_gdf


# --- Map input helpers -----------------------------------------------------------------------


def get_background_for_map(
    raw_dir: Path, raster_candidate_id: str
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """The full native-resolution array for map-background context -- these
    grids are already small (largest confirmed: 777 m x 509 m), so no
    decimation is needed, unlike HHW/IDRBNR's much larger real grids."""

    path, pixel_size_m_hint = _resolve_candidate(raw_dir, raster_candidate_id)
    data, valid, transform, _pixel_size_m = gg_provider.load_xyz_regular_grid(
        path, pixel_size_m=pixel_size_m_hint
    )
    height, width = data.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return data, valid, (left, right, bottom, top)


def get_method_figure_inputs(
    raw_dir: Path,
    raster_candidate_id: str,
    top_tile_row: pd.Series,
    transect_df: pd.DataFrame,
    bedform_df: pd.DataFrame,
) -> dict[str, Any]:
    """Native elevation/valid + filtered/detrended surface for the single
    highest-ranked selected tile, plus one representative transect's
    profile and its detected bedforms, for the primary validation figure
    (Section 18)."""

    path, pixel_size_m_hint = _resolve_candidate(raw_dir, raster_candidate_id)
    full_elevation, full_valid, full_transform, pixel_size_m = gg_provider.load_xyz_regular_grid(
        path, pixel_size_m=pixel_size_m_hint
    )
    inverse_transform = ~full_transform
    half = top_tile_row["tile_size_m"] / 2.0
    col_center, row_center = inverse_transform * (
        top_tile_row["center_x_m"],
        top_tile_row["center_y_m"],
    )
    size_px = int(round(top_tile_row["tile_size_m"] / pixel_size_m))
    col_off = int(round(col_center - size_px / 2.0))
    row_off = int(round(row_center - size_px / 2.0))
    elevation = full_elevation[row_off : row_off + size_px, col_off : col_off + size_px]
    valid = full_valid[row_off : row_off + size_px, col_off : col_off + size_px]

    residual, _trend, _coeffs = swm.remove_planar_trend(elevation, valid, pixel_size_m)
    residual_filled = swm.fill_small_gaps(residual, valid)
    filtered_surface = _spatial_short_wavelength_filter(residual_filled, pixel_size_m)

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
        distances_m, elevations = _sample_transect_from_array(
            full_elevation, full_valid, full_transform, p1, p2, pixel_size_m
        )
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
# --- over criteria A-E; criteria F (no unresolved data-integrity failure) and G (no PL854 --
# --- leakage) are static invariants verified once by tests, never re-derived per run --------


def derive_canonical_real_validation_status(
    *,
    canonical_tile_count: int,
    any_meets_3_wavelengths: bool,
    any_natural_seabed_eligible: bool,
    successful_transect_count: int,
    canonical_bedform_count: int,
) -> tuple[str, str]:
    """MAR-017C Section 15: returns (status, reason). Each gate is checked
    in order and the FIRST one that fails determines the status --
    natural-seabed eligibility (criterion C) is checked independently of,
    and after, the spectral eligibility gate (criterion B), never folded
    into it."""

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
    if not any_natural_seabed_eligible:
        return (
            NO_NATURAL_SEABED_VALIDATION_TILE,
            "Criterion C failed: no tile meeting the spectral eligibility gate is also "
            "NATURAL_SEABED_VALIDATION_ELIGIBLE (Section 11).",
        )
    if successful_transect_count < 1:
        return (
            INSUFFICIENT_VALID_TRANSECTS,
            "Criterion D failed: zero cross-crest transects met the >=90%-valid-samples bar.",
        )
    if canonical_bedform_count < 3:
        return (
            INSUFFICIENT_CANONICAL_BEDFORMS,
            f"Criterion E failed: only {canonical_bedform_count} canonical (>=30 m) bedform(s) "
            "were retained; at least 3 are required.",
        )
    return (
        CANONICAL_REAL_DATA_VALIDATED,
        "Criteria A-E all satisfied: >=1 canonical tile, >=1 tile meeting the >=3-wavelengths "
        "condition, >=1 natural-seabed-eligible tile, >=1 successful transect, and >=3 canonical "
        "bedforms.",
    )


# --- Section 16: pipeline-transfer / validation metadata ------------------------------------


def build_pipeline_transfer_contract(
    *,
    canonical_real_validation_status: str,
    canonical_real_validation_reason: str,
) -> dict[str, Any]:
    passed = canonical_real_validation_status == CANONICAL_REAL_DATA_VALIDATED
    return {
        "scientific_role": SCIENTIFIC_ROLE,
        **ANALOG_ONLY_FLAGS,
        "generic_scientific_core_implemented": True,
        "synthetic_validation_passed": True,
        "canonical_real_data_validation_passed": passed,
        "canonical_real_data_validation_status": canonical_real_validation_status,
        "canonical_real_data_validation_reason": canonical_real_validation_reason,
        "validated_on_analog_dataset": "GREATER_GABBARD_2014" if passed else None,
        "PL854_real_data_validation_passed": False,
        "reusable_function_contract": {
            "module": "marine_engine.morphology.sandwave_morphometry",
            "accepts": [
                "an arbitrary bathymetry raster (numpy array + valid mask + pixel size in metres)",
                "arbitrary route/transect geometry (plain (x, y) endpoint tuples, no CRS assumed)",
                "scale parameters expressed in metres -- never a hard-coded pixel count or "
                "dataset-specific coordinate",
            ],
            "no_hardcoded_dataset_specifics": True,
        },
        "limitations": [
            "Single-epoch morphology cannot provide a migration rate -- no migration rate is "
            "computed anywhere in this module.",
            "Asymmetry is descriptive geometry only, never automatically migration direction.",
            "Analog validation on Greater Gabbard 2014 does not validate PL854 morphology in "
            "any way, regardless of this analog's own validation outcome.",
            "Infrastructure context (turbines/cables/rock-protection) is derived from the "
            "bathymetry package's own file structure, never a full engineering as-built dataset "
            "-- jack-up vessel footprints specifically are not separately identifiable from the "
            "accessible package and are never checked (Section 11: 'where mapped').",
            "No freespan relation, scour prediction, or risk score is tested anywhere here.",
        ],
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def build_final_mar017_family_decision(
    *,
    hhw_status: str,
    idrbnr_status: str,
    greater_gabbard_status: str,
) -> dict[str, Any]:
    """Section 21: the cumulative, final MAR-017-family decision across
    all three real-analog attempts. `NO_FURTHER_OPEN_ANALOG_SEARCH_
    PLANNED` is recorded whenever all three failed -- this closes analog
    hunting, it never triggers a fourth dataset search (Section 1: "Do
    not silently choose a fourth dataset")."""

    passed = greater_gabbard_status == CANONICAL_REAL_DATA_VALIDATED
    all_three_failed = not passed  # HHW and IDRBNR are already known-failed, closed tickets
    return {
        "scientific_role": SCIENTIFIC_ROLE,
        **ANALOG_ONLY_FLAGS,
        "generic_scientific_core_implemented": True,
        "synthetic_validation_passed": True,
        "real_analog_validation_attempts": 3,
        "attempt_1_hhw_cend1111_status": hhw_status,
        "attempt_2_idrbnr_cend1111_status": idrbnr_status,
        "attempt_3_greater_gabbard_2014_status": greater_gabbard_status,
        "canonical_real_data_validation_passed": passed,
        "validated_dataset": "GREATER_GABBARD_2014" if passed else None,
        "PL854_real_data_validation_passed": False,
        "analog_search_status": "NO_FURTHER_OPEN_ANALOG_SEARCH_PLANNED"
        if all_three_failed
        else "CLOSED_VALIDATED",
        "conclusion": (
            "The reusable morphometry engine remains implemented and synthetically verified, "
            "but is NOT canonically real-data validated -- three independent official open "
            "analogs (HHW CEND 11/11, IDRBNR CEND 11/11, Greater Gabbard 2014) all failed the "
            "canonical support/eligibility protocol for structurally different, honestly-"
            "documented reasons. This closes analog hunting for the MAR-017 family; it does not "
            "delay or block the main PL854 project."
            if all_three_failed
            else f"The reusable morphometry engine is canonically real-data validated on "
            f"{greater_gabbard_status}."
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

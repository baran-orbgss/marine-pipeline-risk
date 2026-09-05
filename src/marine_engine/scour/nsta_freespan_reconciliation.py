"""NSTA line-specific freespan registry reconciliation (MAR-014C).

A SECOND, independent official source for freespan evidence -- NSTA's own
Pipeline Freespans registry (`providers/nsta_freespan.py`) -- reconciled
against Ithaca Energy's Table B.1 evidence (MAR-014A/B). This is evidence
reconciliation only: it never computes a freespan susceptibility score,
probability, predictive model, or accuracy/AUC/ROC metric (Section 20 --
our current hydrodynamic model is 2024-2026 forcing, not contemporaneous
with the 2018 survey).

Never snap before measuring (Section 6-7)
------------------------------------------------
Every NSTA feature's ORIGINAL geometry (both WGS84 and working-CRS) is
preserved and its real distance to the canonical route is measured BEFORE
any route projection -- never silently snapped onto the route first. A
route-conforming substring is only ever built afterwards, for map overlay,
and stored separately from the canonical geometry.

Cross-source matching is NOT historical lineage inference (Section 9-10)
-------------------------------------------------------------------------------
Matching an NSTA record to a Table B.1 event answers "does this look like
the same physically reported freespan", using MULTIPLE diagnostics
(interval overlap, midpoint separation, length/height agreement, date/
survey compatibility) -- never spatial proximity alone, and never a
probabilistic score. A `STRONG_CROSS_SOURCE_MATCH` requires clear spatial
correspondence AND at least one independent attribute agreement.

Piggyback coincidence is a first-class outcome, not a nuisance to resolve
(Section 12)
----------------------------------------------------------------------------------
Because PL854/PL855 are piggybacked, NSTA may report the same physical
corridor event against BOTH pipeline numbers. This is detected and labelled
explicitly (`PIGGYBACK_COINCIDENT_FREESPAN_RECORDS`) -- never resolved by
arbitrarily keeping only the PL854 copy.
"""

import sys
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring

from marine_engine.preprocessing.chainage import format_kp_label, project_point_to_route
from marine_engine.providers.nsta_freespan import (
    CURRENT_REGISTRY_LAYER,
    REMOVED_REGISTRY_LAYER,
)

__all__ = [
    "CURRENT_REGISTRY_LAYER",
    "REMOVED_REGISTRY_LAYER",
]

SCIENTIFIC_ROLE = "NSTA_LINE_SPECIFIC_FREESPAN_REGISTRY_EVIDENCE"
NSTA_CROSS_SOURCE_ATTRIBUTION_SOURCE = "NSTA_PIPELINE_FREESPAN_REGISTRY"

# --- Section 8: PL854 vs PL855 attribution evidence statuses ---------------------------
PL854_SUPPORTED_BY_NSTA = "PL854_SUPPORTED_BY_NSTA"
PL855_SUPPORTED_BY_NSTA = "PL855_SUPPORTED_BY_NSTA"
BOTH_PIGGYBACK_COINCIDENT = "BOTH_PL854_AND_PL855_COINCIDENT_NSTA_RECORDS"
LINE_ATTRIBUTION_AMBIGUOUS = "NSTA_MATCH_PRESENT_BUT_LINE_ATTRIBUTION_AMBIGUOUS"
NO_NSTA_CROSS_SOURCE_MATCH = "NO_NSTA_CROSS_SOURCE_MATCH"

# --- Section 10: cross-source match statuses --------------------------------------------
STRONG_CROSS_SOURCE_MATCH = "STRONG_CROSS_SOURCE_MATCH"
POSSIBLE_CROSS_SOURCE_MATCH = "POSSIBLE_CROSS_SOURCE_MATCH"
AMBIGUOUS_MULTIPLE_NSTA_CANDIDATES = "AMBIGUOUS_MULTIPLE_NSTA_CANDIDATES"
NO_NSTA_MATCH = "NO_NSTA_MATCH"
INSUFFICIENT_ATTRIBUTES_FOR_MATCH = "INSUFFICIENT_ATTRIBUTES_FOR_MATCH"

# --- Section 12: piggyback coincidence among NSTA's own PL854/PL855 records -------------
PIGGYBACK_COINCIDENT_FREESPAN_RECORDS = "PIGGYBACK_COINCIDENT_FREESPAN_RECORDS"

# QA heuristics (project-chosen thresholds, never physical constants) -------------------
CANDIDATE_SEARCH_MIDPOINT_THRESHOLD_M = 100.0
CLEAR_SPATIAL_MIDPOINT_THRESHOLD_M = 20.0
CLEAR_SPATIAL_OVERLAP_FRACTION_THRESHOLD = 0.5
LENGTH_AGREEMENT_RELATIVE_THRESHOLD_PCT = 25.0
HEIGHT_AGREEMENT_ABSOLUTE_THRESHOLD_M = 0.15
GROSS_SPATIAL_INCONSISTENCY_THRESHOLD_M = 5000.0
PIGGYBACK_COINCIDENCE_MIDPOINT_THRESHOLD_M = 20.0


# --- Section 6-7: parse raw NSTA features + route reconciliation -----------------------

NSTA_FREESPAN_REGISTRY_COLUMNS = (
    "registry_layer",
    "feature_id",
    "nsta_pipeline_number",
    "label",
    "pipe_name",
    "rep_group",
    "freespanno",
    "length_m",
    "mxheight_m",
    "comments",
    "crs_code",
    "crs_name",
    "start_date",
    "end_date",
    "end_reas",
    "upd_date",
    "upd_type",
    "upd_reas",
    "survey_id",
    "source_geometry_wgs84_wkt",
    "source_to_route_distance_m",
    "gross_spatial_inconsistency_flag",
    "endpoint_a_route_distance_m",
    "endpoint_b_route_distance_m",
    "canonical_chainage_a_m",
    "canonical_chainage_b_m",
    "canonical_chainage_min_m",
    "canonical_chainage_max_m",
    "canonical_kp_min",
    "canonical_kp_max",
    "canonical_mid_chainage_m",
    "canonical_mid_kp",
    "scientific_role",
)


def parse_raw_freespan_features(
    geojson_payload: dict[str, Any], registry_layer: str
) -> list[dict[str, Any]]:
    """Every NSTA source attribute preserved verbatim, plus the parsed WGS84 geometry
    and `registry_layer` provenance tag -- never `CURRENT_...` interpreted as "current
    survey condition" or `REMOVED_...` as "freespan physically disappeared" (Section 5)."""

    records = []
    for feature in geojson_payload.get("features", []):
        props = feature.get("properties", {})
        geometry = shape(feature["geometry"]) if feature.get("geometry") else None
        records.append(
            {
                "registry_layer": registry_layer,
                "feature_id": props.get("FEATURE_ID"),
                "nsta_pipeline_number": props.get("NSTAPIPNO"),
                "label": props.get("LABEL"),
                "pipe_name": props.get("PIPE_NAME"),
                "rep_group": props.get("REP_GROUP"),
                "freespanno": props.get("FREESPANNO"),
                "length_m": props.get("LENGTH_M"),
                "mxheight_m": props.get("MXHEIGHT_M"),
                "comments": props.get("COMMENTS"),
                "crs_code": props.get("CRS_CODE"),
                "crs_name": props.get("CRS_NAME"),
                "start_date": props.get("START_DATE"),
                "end_date": props.get("END_DATE"),
                "end_reas": props.get("END_REAS"),
                "upd_date": props.get("UPD_DATE"),
                "upd_type": props.get("UPD_TYPE"),
                "upd_reas": props.get("UPD_REAS"),
                "survey_id": props.get("SURVEY_ID"),
                "_geometry_wgs84": geometry,
            }
        )
    return records


def _representative_points(geometry) -> tuple[Point, Point]:
    """The two endpoints to project for a line, or the same point twice for a
    point geometry -- so downstream code never special-cases geometry type."""

    if geometry.geom_type == "Point":
        return geometry, geometry
    if geometry.geom_type in ("LineString", "MultiLineString"):
        coords = list(geometry.coords) if geometry.geom_type == "LineString" else None
        if coords is None:
            # MultiLineString: use the first and last part's own endpoints.
            parts = list(geometry.geoms)
            return Point(parts[0].coords[0]), Point(parts[-1].coords[-1])
        return Point(coords[0]), Point(coords[-1])
    # Fallback for any other geometry type (e.g. a malformed Polygon): use the
    # centroid at both ends rather than crashing.
    centroid = geometry.centroid
    return centroid, centroid


def build_nsta_freespan_registry_gdf(
    records: list[dict[str, Any]], route: LineString, working_crs: str
) -> gpd.GeoDataFrame:
    """Route-reconcile every parsed NSTA feature (Section 7): reproject, measure
    the REAL (never snapped) geometry-to-route separation, project representative
    points onto the route, derive canonical chainage. Returns an empty
    (correctly-shaped) GeoDataFrame if `records` is empty -- a real, valid outcome
    (Section 3), never an error.
    """

    if not records:
        return gpd.GeoDataFrame(
            columns=list(NSTA_FREESPAN_REGISTRY_COLUMNS), geometry=[], crs=working_crs
        )

    wgs84_geometries = [r["_geometry_wgs84"] for r in records]
    working_geometries = list(gpd.GeoSeries(wgs84_geometries, crs="EPSG:4326").to_crs(working_crs))

    out_records = []
    for record, geometry_wgs84, geometry_working in zip(
        records, wgs84_geometries, working_geometries, strict=True
    ):
        # The REAL, un-snapped separation between the source geometry and the route.
        source_to_route_distance_m = float(geometry_working.distance(route))
        gross_flag = source_to_route_distance_m > GROSS_SPATIAL_INCONSISTENCY_THRESHOLD_M

        point_a, point_b = _representative_points(geometry_working)
        projection_a = project_point_to_route(route, point_a)
        projection_b = project_point_to_route(route, point_b)
        chainage_a, chainage_b = projection_a.chainage_m, projection_b.chainage_m
        chainage_min, chainage_max = min(chainage_a, chainage_b), max(chainage_a, chainage_b)
        mid_chainage = (chainage_min + chainage_max) / 2.0

        out_records.append(
            {
                **{k: v for k, v in record.items() if k != "_geometry_wgs84"},
                "source_geometry_wgs84_wkt": geometry_wgs84.wkt,
                "source_to_route_distance_m": source_to_route_distance_m,
                "gross_spatial_inconsistency_flag": gross_flag,
                "endpoint_a_route_distance_m": projection_a.distance_m,
                "endpoint_b_route_distance_m": projection_b.distance_m,
                "canonical_chainage_a_m": chainage_a,
                "canonical_chainage_b_m": chainage_b,
                "canonical_chainage_min_m": chainage_min,
                "canonical_chainage_max_m": chainage_max,
                "canonical_kp_min": format_kp_label(chainage_min),
                "canonical_kp_max": format_kp_label(chainage_max),
                "canonical_mid_chainage_m": mid_chainage,
                "canonical_mid_kp": format_kp_label(mid_chainage),
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )

    return gpd.GeoDataFrame(
        out_records,
        geometry=working_geometries,
        crs=working_crs,
        columns=list(NSTA_FREESPAN_REGISTRY_COLUMNS),
    )


def write_nsta_freespan_registry(gdf: gpd.GeoDataFrame, parquet_path, gpkg_path) -> tuple[Any, Any]:
    """Always writes both files, even when `gdf` is empty (a real, valid outcome,
    Section 3) -- a correctly-shaped zero-row GPKG is written rather than silently
    skipped, so a caller never has to guess whether "file missing" means "empty
    registry" or "this step never ran"."""

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.drop(columns="geometry").to_parquet(parquet_path, index=False)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(gpkg_path, driver="GPKG", layer="nsta_pl854_pl855_freespan_registry")
    return parquet_path, gpkg_path


def build_route_conforming_substrings(gdf: gpd.GeoDataFrame, route: LineString) -> list[Any]:
    """A route-conforming substring per feature, for MAP OVERLAY ONLY (Section 6's
    "after QA, a route-conforming substring may be stored separately") -- never
    the canonical stored geometry, which stays the real, un-snapped source geometry."""

    return [
        substring(
            route,
            row["canonical_chainage_min_m"],
            row["canonical_chainage_max_m"],
            normalized=False,
        )
        for _, row in gdf.iterrows()
    ]


# --- Section 9-11: cross-source matching against Table B.1 -----------------------------

MATCH_DIAGNOSTICS_COLUMNS = (
    "table_b1_event_id",
    "table_b1_survey_year",
    "nsta_feature_id",
    "nsta_pipeline_number",
    "nsta_freespanno",
    "nsta_survey_id",
    "registry_layer",
    "midpoint_separation_m",
    "interval_overlap_m",
    "interval_overlap_fraction",
    "length_difference_m",
    "length_difference_pct",
    "height_difference_m",
    "cross_source_match_status",
)


def _interval_overlap_m(b1_min: float, b1_max: float, nsta_min: float, nsta_max: float) -> float:
    return max(0.0, min(b1_max, nsta_max) - max(b1_min, nsta_min))


def compute_candidate_diagnostics(b1_event: pd.Series, nsta_row: pd.Series) -> dict[str, Any]:
    """Every diagnostic Section 11 requires for one (Table B.1 event, NSTA feature)
    candidate pair -- no opaque aggregate score, only named, inspectable numbers."""

    overlap_m = _interval_overlap_m(
        b1_event["canonical_chainage_min_m"],
        b1_event["canonical_chainage_max_m"],
        nsta_row["canonical_chainage_min_m"],
        nsta_row["canonical_chainage_max_m"],
    )
    b1_length = b1_event["source_length_m"]
    overlap_fraction = overlap_m / b1_length if b1_length and b1_length > 0 else 0.0

    nsta_length = nsta_row["length_m"]
    length_difference_m = (
        abs(b1_length - nsta_length) if pd.notna(b1_length) and pd.notna(nsta_length) else None
    )
    length_difference_pct = (
        100.0 * length_difference_m / b1_length
        if length_difference_m is not None and b1_length and b1_length > 0
        else None
    )

    b1_height = b1_event.get("source_height_m")
    nsta_height = nsta_row.get("mxheight_m")
    height_difference_m = (
        abs(b1_height - nsta_height) if pd.notna(b1_height) and pd.notna(nsta_height) else None
    )

    return {
        "table_b1_event_id": b1_event["event_id"],
        "table_b1_survey_year": b1_event["survey_year"],
        "nsta_feature_id": nsta_row["feature_id"],
        "nsta_pipeline_number": nsta_row["nsta_pipeline_number"],
        "nsta_freespanno": nsta_row["freespanno"],
        "nsta_survey_id": nsta_row["survey_id"],
        "registry_layer": nsta_row["registry_layer"],
        "midpoint_separation_m": abs(
            b1_event["canonical_mid_chainage_m"] - nsta_row["canonical_mid_chainage_m"]
        ),
        "interval_overlap_m": overlap_m,
        "interval_overlap_fraction": overlap_fraction,
        "length_difference_m": length_difference_m,
        "length_difference_pct": length_difference_pct,
        "height_difference_m": height_difference_m,
    }


def _has_independent_attribute_agreement(diagnostics: dict[str, Any]) -> bool:
    length_pct = diagnostics["length_difference_pct"]
    if length_pct is not None and length_pct <= LENGTH_AGREEMENT_RELATIVE_THRESHOLD_PCT:
        return True
    height_diff = diagnostics["height_difference_m"]
    return height_diff is not None and height_diff <= HEIGHT_AGREEMENT_ABSOLUTE_THRESHOLD_M


def _has_any_independent_attribute(diagnostics: dict[str, Any]) -> bool:
    return (
        diagnostics["length_difference_m"] is not None
        or diagnostics["height_difference_m"] is not None
    )


def classify_candidate_match_status(diagnostics: dict[str, Any]) -> str:
    """One candidate pair's status (Section 10) -- a STRONG match requires clear
    spatial correspondence AND one independent attribute agreement; never spatial
    proximity alone, never a probabilistic score."""

    clear_spatial = (
        diagnostics["midpoint_separation_m"] <= CLEAR_SPATIAL_MIDPOINT_THRESHOLD_M
        or diagnostics["interval_overlap_fraction"] >= CLEAR_SPATIAL_OVERLAP_FRACTION_THRESHOLD
    )
    if clear_spatial and _has_independent_attribute_agreement(diagnostics):
        return STRONG_CROSS_SOURCE_MATCH
    if clear_spatial and not _has_any_independent_attribute(diagnostics):
        return INSUFFICIENT_ATTRIBUTES_FOR_MATCH
    return POSSIBLE_CROSS_SOURCE_MATCH


def match_table_b1_events_to_nsta(
    table_b1_df: pd.DataFrame, nsta_registry_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Every Table B.1 event (all years) against every spatially-candidate NSTA
    feature -- a candidate is any NSTA feature within
    `CANDIDATE_SEARCH_MIDPOINT_THRESHOLD_M` or with nonzero interval overlap.
    One row per (event, candidate) pair; `summarize_event_match_status` below
    reduces this to one final status per event.
    """

    if nsta_registry_gdf.empty:
        return pd.DataFrame(columns=list(MATCH_DIAGNOSTICS_COLUMNS))

    rows = []
    for _, b1_event in table_b1_df.iterrows():
        for _, nsta_row in nsta_registry_gdf.iterrows():
            overlap_m = _interval_overlap_m(
                b1_event["canonical_chainage_min_m"],
                b1_event["canonical_chainage_max_m"],
                nsta_row["canonical_chainage_min_m"],
                nsta_row["canonical_chainage_max_m"],
            )
            midpoint_sep = abs(
                b1_event["canonical_mid_chainage_m"] - nsta_row["canonical_mid_chainage_m"]
            )
            if overlap_m <= 0 and midpoint_sep > CANDIDATE_SEARCH_MIDPOINT_THRESHOLD_M:
                continue  # not a spatial candidate at all

            diagnostics = compute_candidate_diagnostics(b1_event, nsta_row)
            diagnostics["cross_source_match_status"] = classify_candidate_match_status(diagnostics)
            rows.append(diagnostics)

    return pd.DataFrame(rows, columns=list(MATCH_DIAGNOSTICS_COLUMNS))


def summarize_event_match_status(candidate_rows: pd.DataFrame) -> str:
    """Reduce one event's candidate rows to a single final status (Section 10) --
    multiple STRONG/POSSIBLE candidates from DIFFERENT NSTA features become
    AMBIGUOUS rather than an arbitrarily "best" pick."""

    if candidate_rows.empty:
        return NO_NSTA_MATCH

    distinct_features = candidate_rows["nsta_feature_id"].nunique()
    strong = candidate_rows[
        candidate_rows["cross_source_match_status"] == STRONG_CROSS_SOURCE_MATCH
    ]
    possible = candidate_rows[
        candidate_rows["cross_source_match_status"] == POSSIBLE_CROSS_SOURCE_MATCH
    ]
    insufficient = candidate_rows[
        candidate_rows["cross_source_match_status"] == INSUFFICIENT_ATTRIBUTES_FOR_MATCH
    ]

    if distinct_features > 1 and (len(strong) + len(possible) + len(insufficient)) > 1:
        return AMBIGUOUS_MULTIPLE_NSTA_CANDIDATES
    if len(strong) >= 1:
        return STRONG_CROSS_SOURCE_MATCH
    if len(possible) >= 1:
        return POSSIBLE_CROSS_SOURCE_MATCH
    if len(insufficient) >= 1:
        return INSUFFICIENT_ATTRIBUTES_FOR_MATCH
    return NO_NSTA_MATCH


# --- Section 12: piggyback PL854/PL855 coincidence among NSTA's own records ------------


def detect_piggyback_coincident_records(nsta_registry_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """PL854 vs PL855 NSTA record pairs that are spatially/attributively coincident
    (Section 12) -- reported explicitly, never resolved by keeping only one copy."""

    columns = (
        "pl854_feature_id",
        "pl855_feature_id",
        "midpoint_separation_m",
        "length_agreement",
        "height_agreement",
        "survey_id_agreement",
        "classification",
    )
    if nsta_registry_gdf.empty:
        return pd.DataFrame(columns=list(columns))

    pl854_rows = nsta_registry_gdf[nsta_registry_gdf["nsta_pipeline_number"] == "PL854"]
    pl855_rows = nsta_registry_gdf[nsta_registry_gdf["nsta_pipeline_number"] == "PL855"]

    records = []
    for _, row_854 in pl854_rows.iterrows():
        for _, row_855 in pl855_rows.iterrows():
            midpoint_sep = abs(
                row_854["canonical_mid_chainage_m"] - row_855["canonical_mid_chainage_m"]
            )
            if midpoint_sep > PIGGYBACK_COINCIDENCE_MIDPOINT_THRESHOLD_M:
                continue
            length_agree = (
                pd.notna(row_854["length_m"])
                and pd.notna(row_855["length_m"])
                and abs(row_854["length_m"] - row_855["length_m"]) <= 1.0
            )
            height_agree = (
                pd.notna(row_854["mxheight_m"])
                and pd.notna(row_855["mxheight_m"])
                and abs(row_854["mxheight_m"] - row_855["mxheight_m"])
                <= HEIGHT_AGREEMENT_ABSOLUTE_THRESHOLD_M
            )
            survey_agree = (
                pd.notna(row_854["survey_id"]) and row_854["survey_id"] == row_855["survey_id"]
            )
            records.append(
                {
                    "pl854_feature_id": row_854["feature_id"],
                    "pl855_feature_id": row_855["feature_id"],
                    "midpoint_separation_m": midpoint_sep,
                    "length_agreement": bool(length_agree),
                    "height_agreement": bool(height_agree),
                    "survey_id_agreement": bool(survey_agree),
                    "classification": PIGGYBACK_COINCIDENT_FREESPAN_RECORDS,
                }
            )
    return pd.DataFrame(records, columns=list(columns))


# --- Section 8, 16: attribution evidence derivation -------------------------------------


def derive_attribution_evidence_status(
    matched_candidates: pd.DataFrame, piggyback_df: pd.DataFrame
) -> str:
    """The Section 8 downstream attribution status for one event, given its matched
    NSTA candidates -- never PL854-only merely because PL854 is the study route."""

    strong_or_possible = matched_candidates[
        matched_candidates["cross_source_match_status"].isin(
            (STRONG_CROSS_SOURCE_MATCH, POSSIBLE_CROSS_SOURCE_MATCH)
        )
    ]
    if strong_or_possible.empty:
        return NO_NSTA_CROSS_SOURCE_MATCH

    matched_feature_ids = set(strong_or_possible["nsta_feature_id"])
    if not piggyback_df.empty:
        coincident = piggyback_df[
            piggyback_df["pl854_feature_id"].isin(matched_feature_ids)
            | piggyback_df["pl855_feature_id"].isin(matched_feature_ids)
        ]
        if not coincident.empty:
            return BOTH_PIGGYBACK_COINCIDENT

    pipeline_numbers = set(strong_or_possible["nsta_pipeline_number"])
    if pipeline_numbers == {"PL854"}:
        return PL854_SUPPORTED_BY_NSTA
    if pipeline_numbers == {"PL855"}:
        return PL855_SUPPORTED_BY_NSTA
    return LINE_ATTRIBUTION_AMBIGUOUS


ATTRIBUTION_EVIDENCE_COLUMNS = (
    "event_id",
    "survey_year",
    "canonical_mid_kp",
    "source_length_m",
    "source_height_m",
    "original_asset_scope",
    "original_individual_line_attribution",
    "cross_source_match_status",
    "matched_nsta_pipeline_numbers",
    "matched_nsta_feature_ids",
    "attribution_evidence_status",
    "attribution_evidence_source",
)


def build_2018_attribution_evidence(
    table_b1_2018_df: pd.DataFrame, match_diagnostics_df: pd.DataFrame, piggyback_df: pd.DataFrame
) -> pd.DataFrame:
    """The Section 16 derived 2018 attribution-evidence file -- NEVER mutates the
    original Table B.1 canonical evidence; `original_individual_line_attribution`
    is always carried through unchanged (UNRESOLVED)."""

    records = []
    for _, event in table_b1_2018_df.iterrows():
        event_candidates = (
            match_diagnostics_df[match_diagnostics_df["table_b1_event_id"] == event["event_id"]]
            if not match_diagnostics_df.empty
            else match_diagnostics_df
        )
        match_status = summarize_event_match_status(event_candidates)
        attribution_status = derive_attribution_evidence_status(event_candidates, piggyback_df)

        matched = (
            event_candidates[
                event_candidates["cross_source_match_status"].isin(
                    (STRONG_CROSS_SOURCE_MATCH, POSSIBLE_CROSS_SOURCE_MATCH)
                )
            ]
            if not event_candidates.empty
            else event_candidates
        )
        matched_pipeline_numbers = (
            sorted(set(matched["nsta_pipeline_number"])) if not matched.empty else []
        )
        matched_feature_ids = sorted(set(matched["nsta_feature_id"])) if not matched.empty else []

        records.append(
            {
                "event_id": event["event_id"],
                "survey_year": event["survey_year"],
                "canonical_mid_kp": event["canonical_mid_kp"],
                "source_length_m": event["source_length_m"],
                "source_height_m": event["source_height_m"],
                "original_asset_scope": event["asset_scope"],
                "original_individual_line_attribution": event["individual_line_attribution"],
                "cross_source_match_status": match_status,
                "matched_nsta_pipeline_numbers": matched_pipeline_numbers,
                "matched_nsta_feature_ids": matched_feature_ids,
                "attribution_evidence_status": attribution_status,
                "attribution_evidence_source": NSTA_CROSS_SOURCE_ATTRIBUTION_SOURCE,
            }
        )
    return pd.DataFrame(records, columns=list(ATTRIBUTION_EVIDENCE_COLUMNS))


# --- Section 13, 15: temporal semantics + count/total comparisons ----------------------


def summarize_registry_temporal_context(nsta_registry_gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Distribution of START_DATE/END_DATE/UPD_DATE/SURVEY_ID -- descriptive only,
    never assumed to mean "2018" (Section 13)."""

    if nsta_registry_gdf.empty:
        return {
            "feature_count": 0,
            "start_date_range": None,
            "end_date_range": None,
            "upd_date_range": None,
            "survey_id_values": [],
            "missingness": {},
        }

    def _range(column: str) -> list[Any] | None:
        values = nsta_registry_gdf[column].dropna()
        if values.empty:
            return None
        return [values.min(), values.max()]

    return {
        "feature_count": int(len(nsta_registry_gdf)),
        "start_date_range": _range("start_date"),
        "end_date_range": _range("end_date"),
        "upd_date_range": _range("upd_date"),
        "survey_id_values": sorted(set(nsta_registry_gdf["survey_id"].dropna())),
        "missingness": {
            column: int(nsta_registry_gdf[column].isna().sum())
            for column in ("start_date", "end_date", "upd_date", "survey_id")
        },
    }


def compare_registry_totals(
    nsta_registry_gdf: gpd.GeoDataFrame, table_b1_df: pd.DataFrame
) -> dict[str, Any]:
    """NSTA vs Table B.1 feature count/total length/max length/max height --
    reported as differences, never forced to equality (Section 15)."""

    nsta_lengths = (
        nsta_registry_gdf["length_m"].dropna()
        if not nsta_registry_gdf.empty
        else pd.Series(dtype=float)
    )
    nsta_heights = (
        nsta_registry_gdf["mxheight_m"].dropna()
        if not nsta_registry_gdf.empty
        else pd.Series(dtype=float)
    )
    return {
        "nsta_feature_count": int(len(nsta_registry_gdf)),
        "table_b1_feature_count": int(len(table_b1_df)),
        "nsta_total_length_m": float(nsta_lengths.sum()) if not nsta_lengths.empty else 0.0,
        "table_b1_total_length_m": float(table_b1_df["source_length_m"].sum()),
        "nsta_max_length_m": float(nsta_lengths.max()) if not nsta_lengths.empty else None,
        "table_b1_max_length_m": float(table_b1_df["source_length_m"].max()),
        "nsta_max_height_m": float(nsta_heights.max()) if not nsta_heights.empty else None,
        "table_b1_max_height_m": float(table_b1_df["source_height_m"].max()),
        "counts_forced_to_match": False,
    }


# --- Report printer ----------------------------------------------------------------------


def print_reconciliation_report(
    *,
    nsta_registry_gdf: gpd.GeoDataFrame,
    temporal_context: dict[str, Any],
    attribution_evidence_df: pd.DataFrame,
    totals_comparison: dict[str, Any],
    piggyback_df: pd.DataFrame,
    outputs: dict[str, Any],
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== NSTA Line-Specific Freespan Registry Reconciliation (MAR-014C) ===", ""]

    lines.append("## NSTA registry")
    for layer in ("CURRENT_PIPELINE_FREESPANS", "REMOVED_PIPELINE_FREESPANS"):
        count = (
            int((nsta_registry_gdf["registry_layer"] == layer).sum())
            if not nsta_registry_gdf.empty
            else 0
        )
        for pipeline_number in ("PL854", "PL855"):
            pipeline_count = (
                int(
                    (
                        (nsta_registry_gdf["registry_layer"] == layer)
                        & (nsta_registry_gdf["nsta_pipeline_number"] == pipeline_number)
                    ).sum()
                )
                if not nsta_registry_gdf.empty
                else 0
            )
            lines.append(f"  {layer} / {pipeline_number}: {pipeline_count}")
        lines.append(f"  {layer} total: {count}")
    lines.append("")

    lines.append("## Registry temporal context")
    for key, value in temporal_context.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    lines.append("## Counts/totals comparison (never forced to match)")
    for key, value in totals_comparison.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    lines.append("## Piggyback coincidence")
    lines.append(f"  Coincident PL854/PL855 record pairs: {len(piggyback_df)}")
    lines.append("")

    lines.append("## 2018 attribution evidence")
    if not attribution_evidence_df.empty:
        for status, count in (
            attribution_evidence_df["attribution_evidence_status"].value_counts().items()
        ):
            lines.append(f"  {status}: {count}")
    lines.append("")

    lines.append("## Outputs")
    for key, value in outputs.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    lines.append(
        "NSTA LINE-SPECIFIC ATTRIBUTION EVIDENCE IS PRESERVED AS A SECOND SOURCE; THE "
        "ORIGINAL ITHACA PL854/PL855 CORRIDOR ATTRIBUTION IS NOT OVERWRITTEN."
    )
    lines.append("NO MODEL VALIDATION OR FREESPAN SUSCEPTIBILITY SCORE HAS BEEN CREATED.")

    print("\n".join(lines), file=file)

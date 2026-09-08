"""Real NSTA UKCS-wide pipeline free-span registry: cached acquisition and audit
(MAR-025 Sections 18-20).

Reuses the accepted NSTA freespan provider architecture (`marine_engine.providers.nsta_freespan`)
at the level of its query/parse primitives -- `query_freespan_layer`,
`marine_engine.scour.nsta_freespan_reconciliation.parse_raw_freespan_features` -- rather than its
PL854/PL855-filtered `ingest_freespan_registry`/`build_nsta_freespan_registry_gdf` high-level
functions. Section 20's registry-wide audit (total current/removed records, unique pipeline IDs)
is only meaningful across the FULL UKCS-wide registry, not the two-pipeline filtered subset
(which is real, already investigated, and correctly zero for PL854/PL855 -- see
`providers.nsta_freespan`'s own module docstring; that zero result is a genuine finding, not a
gap this module tries to work around).

Unlike `ingest_freespan_registry` (which always re-fetches from the network), this module
acquires from the network only when the current accepted cache is absent (Section 18).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from marine_engine.providers import nsta_freespan
from marine_engine.scour import nsta_freespan_reconciliation

FULL_REGISTRY_WHERE_CLAUSE = "1=1"

_TARGET_LAYERS: tuple[tuple[str, str], ...] = (
    (nsta_freespan.CURRENT_REGISTRY_LAYER, nsta_freespan.CURRENT_FREESPAN_SERVICE_URL),
    (nsta_freespan.REMOVED_REGISTRY_LAYER, nsta_freespan.REMOVED_FREESPAN_SERVICE_URL),
)

RAW_CACHE_FILENAMES: dict[str, str] = {
    nsta_freespan.CURRENT_REGISTRY_LAYER: "current_pipeline_freespans_full_registry.geojson",
    nsta_freespan.REMOVED_REGISTRY_LAYER: "removed_pipeline_freespans_full_registry.geojson",
}


def acquire_full_registry_cached(cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Section 18: acquire/cache only if the current accepted cache is absent -- a real network
    call happens only for a layer whose cache file does not already exist. Returns a dict keyed
    by registry_layer -> {"payload", "already_cached", "cache_path"}."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for registry_layer, service_url in _TARGET_LAYERS:
        cache_path = cache_dir / RAW_CACHE_FILENAMES[registry_layer]
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            already_cached = True
        else:
            payload = nsta_freespan.query_freespan_layer(service_url, FULL_REGISTRY_WHERE_CLAUSE)
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            already_cached = False
        results[registry_layer] = {
            "payload": payload,
            "already_cached": already_cached,
            "cache_path": cache_path,
        }
    return results


# --- Section 20: registry-wide audit -- descriptive only, no quality score --------------------

NSTA_REGISTRY_AUDIT_COLUMNS: tuple[str, ...] = (
    "registry_layer",
    "feature_id",
    "nsta_pipeline_number",
    "pipe_name",
    "freespanno",
    "length_m",
    "mxheight_m",
    "comments",
    "start_date",
    "end_date",
    "upd_date",
    "survey_id",
    "has_valid_length_m",
    "has_valid_mxheight_m",
    "has_survey_or_date_field",
    "has_geometry",
    "has_pipeline_number",
    "longitude",
    "latitude",
)


def _is_valid_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def build_nsta_registry_audit_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Section 20: one row per real NSTA feature (both layers combined), preserving every field
    Section 18 requires plus derived completeness/validity flags. Never a quality score."""

    if not records:
        return pd.DataFrame(columns=list(NSTA_REGISTRY_AUDIT_COLUMNS))

    rows = []
    for r in records:
        geometry = r.get("_geometry_wgs84")
        rows.append(
            {
                "registry_layer": r.get("registry_layer"),
                "feature_id": r.get("feature_id"),
                "nsta_pipeline_number": r.get("nsta_pipeline_number"),
                "pipe_name": r.get("pipe_name"),
                "freespanno": r.get("freespanno"),
                "length_m": r.get("length_m"),
                "mxheight_m": r.get("mxheight_m"),
                "comments": r.get("comments"),
                "start_date": r.get("start_date"),
                "end_date": r.get("end_date"),
                "upd_date": r.get("upd_date"),
                "survey_id": r.get("survey_id"),
                "has_valid_length_m": _is_valid_positive_number(r.get("length_m")),
                "has_valid_mxheight_m": _is_valid_positive_number(r.get("mxheight_m")),
                "has_survey_or_date_field": any(
                    r.get(f) is not None
                    for f in ("start_date", "end_date", "upd_date", "survey_id")
                ),
                "has_geometry": geometry is not None,
                "has_pipeline_number": r.get("nsta_pipeline_number") is not None,
                "longitude": geometry.centroid.x if geometry is not None else None,
                "latitude": geometry.centroid.y if geometry is not None else None,
            }
        )
    return pd.DataFrame(rows, columns=list(NSTA_REGISTRY_AUDIT_COLUMNS))


def summarize_registry_audit(audit_df: pd.DataFrame) -> dict[str, Any]:
    """Section 20's required report facts. No aggregate quality score anywhere."""

    if audit_df.empty:
        return {
            "total_record_count": 0,
            "total_current_records": 0,
            "total_removed_records": 0,
            "unique_pipeline_id_count": 0,
            "valid_length_m_count": 0,
            "valid_mxheight_m_count": 0,
            "records_with_survey_or_date_field_count": 0,
            "records_with_geometry_count": 0,
            "duplicated_feature_id_count": 0,
            "records_missing_pipeline_id_count": 0,
        }

    current_df = audit_df[audit_df["registry_layer"] == nsta_freespan.CURRENT_REGISTRY_LAYER]
    removed_df = audit_df[audit_df["registry_layer"] == nsta_freespan.REMOVED_REGISTRY_LAYER]
    return {
        "total_record_count": int(len(audit_df)),
        "total_current_records": int(len(current_df)),
        "total_removed_records": int(len(removed_df)),
        "unique_pipeline_id_count": int(audit_df["nsta_pipeline_number"].nunique(dropna=True)),
        "valid_length_m_count": int(audit_df["has_valid_length_m"].sum()),
        "valid_mxheight_m_count": int(audit_df["has_valid_mxheight_m"].sum()),
        "records_with_survey_or_date_field_count": int(audit_df["has_survey_or_date_field"].sum()),
        "records_with_geometry_count": int(audit_df["has_geometry"].sum()),
        "duplicated_feature_id_count": int(audit_df["feature_id"].duplicated(keep=False).sum()),
        "records_missing_pipeline_id_count": int((~audit_df["has_pipeline_number"]).sum()),
    }


def build_nsta_registry_gdf(records: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    """Section 28's real-evidence GIS layer: every real parsed NSTA feature, WGS84, no route
    projection -- there is no single route spanning 222 different real pipelines."""

    if not records:
        return gpd.GeoDataFrame(
            pd.DataFrame(
                columns=[
                    c for c in NSTA_REGISTRY_AUDIT_COLUMNS if c not in ("longitude", "latitude")
                ]
            ),
            geometry=[],
            crs="EPSG:4326",
        )
    rows = [{k: v for k, v in r.items() if k != "_geometry_wgs84"} for r in records]
    geometries = [r.get("_geometry_wgs84") for r in records]
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geometries, crs="EPSG:4326")


def select_example_pipeline_by_record_count(audit_df: pd.DataFrame) -> tuple[str | None, int]:
    """Section 26's documented example-selection rule: the pipeline with the MOST registry
    records -- a data-driven pick for map readability, never an indication of highest risk."""

    if audit_df.empty:
        return None, 0
    counts = audit_df["nsta_pipeline_number"].dropna().value_counts()
    if counts.empty:
        return None, 0
    top_pipeline = str(counts.idxmax())
    return top_pipeline, int(counts.loc[top_pipeline])


def build_real_nsta_evidence(cache_dir: Path) -> dict[str, Any]:
    """Full Section 18-20 pipeline: cached acquisition -> field-preserving parse -> audit."""

    acquisitions = acquire_full_registry_cached(cache_dir)
    all_records: list[dict[str, Any]] = []
    for registry_layer, info in acquisitions.items():
        all_records.extend(
            nsta_freespan_reconciliation.parse_raw_freespan_features(
                info["payload"], registry_layer
            )
        )
    audit_df = build_nsta_registry_audit_df(all_records)
    return {
        "acquisitions": acquisitions,
        "records": all_records,
        "audit_df": audit_df,
        "audit_summary": summarize_registry_audit(audit_df),
        "registry_gdf": build_nsta_registry_gdf(all_records),
    }

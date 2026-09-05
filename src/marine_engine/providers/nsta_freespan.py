"""NSTA (North Sea Transition Authority) Pipeline Freespans provider (MAR-014C).

A separate, dedicated official source from `nsta.py`'s pipeline-geometry
services and from Ithaca Energy's own Table B.1: NSTA's own line-specific
freespan registry, published as public ArcGIS Online Feature Services under
the same `NSTA_GIS` organisation account.

Source discovery -- verified, never guessed (Section 1)
--------------------------------------------------------------
The ticket-supplied URL (`data.nstauthority.co.uk`) does not resolve (a
genuine NXDOMAIN against a public resolver, confirmed 2026-09-06) -- not a
formatting issue, a different real host. The two services below were
instead found the same way `nsta.py`'s own services were: the public
ArcGIS Online item-search API
(`https://www.arcgis.com/sharing/rest/search?q=freespan AND owner:NSTA_GIS`),
confirmed reachable/queryable and `contentStatus=public_authoritative`:

- "UKCS offshore infrastructure pipeline freespans (WGS84)"
  (item cb4a29cf7368469082eef555a4ffac97) -- the current/active freespan feed.
- "UKCS offshore infrastructure pipeline freespans removed (WGS84)"
  (item fcdf8d516fd8464e951e73e313217019) -- the removed/decommissioned feed.

Both services expose layer id 1 (matching `nsta.py`'s own pipeline-linear
services), field schema confirmed via `<service>/1?f=json`: `FEATURE_ID`,
`NSTAPIPNO`, `LABEL`, `PIPE_NAME`, `REP_GROUP`, `FREESPANNO`, `LENGTH_M`,
`MXHEIGHT_M`, `COMMENTS`, `CRS_CODE`, `CRS_NAME`, `START_DATE`, `END_DATE`,
`END_REAS`, `UPD_DATE`, `UPD_TYPE`, `UPD_REAS`, `SURVEY_ID`, plus
`LEGACY_ID`/`LEG_P_ID` (an older identifier scheme) not mentioned in the
ticket's own field list.

PL854/PL855 genuinely absent -- inspected, not assumed (Section 3)
-------------------------------------------------------------------------
`NSTAPIPNO IN ('PL854','PL855')` returns zero features in BOTH layers (953
+ 25 = 978 total freespan records; 222 distinct NSTAPIPNO values across
both). Per Section 3's explicit instruction, this was investigated rather
than silently broadened to fuzzy `PIPE_NAME` matching: a substring check
of every distinct `NSTAPIPNO` value for "854"/"855" found none, a check of
`LEGACY_ID`/`LEG_P_ID` (still formal identifier fields, never fuzzy name
matching) for the same substrings found none, and a `PIPE_NAME` search for
"ANGLIA"/"LOGGS" (diagnostic only -- never used as an actual match key)
found only unrelated pipelines sharing the LOGGS terminal (PL2643, PL454)
with no "ANGLIA" match. This module therefore treats a zero-count result
as a real, reportable acquisition outcome, never an error to hide or a
signal to fall back to name-based matching.
"""

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

CURRENT_FREESPAN_SERVICE_URL = (
    "https://services-eu1.arcgis.com/OZMfUznmLTnWccBc/arcgis/rest/services/"
    "UKCS offshore infrastructure pipeline freespans WGS84/FeatureServer/1"
)
CURRENT_FREESPAN_SOURCE_TITLE = "NSTA UKCS offshore infrastructure pipeline freespans (WGS84)"
CURRENT_REGISTRY_LAYER = "CURRENT_PIPELINE_FREESPANS"

REMOVED_FREESPAN_SERVICE_URL = (
    "https://services-eu1.arcgis.com/OZMfUznmLTnWccBc/arcgis/rest/services/"
    "UKCS_offshore_infrastructure_pipeline_freespans_removed_WGS84/FeatureServer/1"
)
REMOVED_FREESPAN_SOURCE_TITLE = (
    "NSTA UKCS offshore infrastructure pipeline freespans removed (WGS84)"
)
REMOVED_REGISTRY_LAYER = "REMOVED_PIPELINE_FREESPANS"

_LAYERS = (
    (CURRENT_REGISTRY_LAYER, CURRENT_FREESPAN_SERVICE_URL, CURRENT_FREESPAN_SOURCE_TITLE),
    (REMOVED_REGISTRY_LAYER, REMOVED_FREESPAN_SERVICE_URL, REMOVED_FREESPAN_SOURCE_TITLE),
)

PIPELINE_NUMBER_FIELD = "NSTAPIPNO"  # confirmed via <service>/1?f=json field list
TARGET_PIPELINE_NUMBERS = ("PL854", "PL855")
SOURCE_CRS = "EPSG:4326"  # both services are published as "(WGS84)"
REQUEST_TIMEOUT_S = 30.0


class NstaFreespanServiceError(RuntimeError):
    """A freespan FeatureServer layer returned a server-side error payload."""


@dataclass(frozen=True)
class FreespanLayerAcquisition:
    """One registry layer's (current or removed) raw acquisition result."""

    registry_layer: str
    source_title: str
    service_url: str
    query_where_clause: str
    feature_count: int
    raw_cache_path: Path
    response_crs: str | None
    schema_field_names: tuple[str, ...]
    retrieved_at: datetime


def build_pipeline_number_where_clause(pipeline_numbers: tuple[str, ...]) -> str:
    """`NSTAPIPNO IN ('PL854','PL855')` -- the exact identifier filter (Section 3),
    never a fuzzy `PIPE_NAME LIKE` fallback."""

    quoted = ",".join(f"'{number}'" for number in pipeline_numbers)
    return f"{PIPELINE_NUMBER_FIELD} IN ({quoted})"


def fetch_layer_schema(service_url: str, timeout: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    response = requests.get(service_url, params={"f": "json"}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def query_freespan_layer(
    service_url: str, where_clause: str, timeout: float = REQUEST_TIMEOUT_S
) -> dict[str, Any]:
    """Query one freespan FeatureServer layer's documented `/query` REST operation
    (JSON in, GeoJSON out) -- not HTML scraping. `outFields=*`, full geometry."""

    response = requests.get(
        f"{service_url}/query",
        params={
            "where": where_clause,
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if "error" in payload:
        raise NstaFreespanServiceError(
            f"NSTA freespan service error querying {service_url}: {payload['error']}"
        )
    return payload


def query_distinct_pipeline_numbers(
    service_url: str, timeout: float = REQUEST_TIMEOUT_S
) -> list[str]:
    """Every distinct `NSTAPIPNO` value in a layer -- the Section 3 "inspect distinct
    NSTAPIPNO semantics" diagnostic, run automatically whenever the exact-identifier
    query returns zero records, never a substitute for the exact-identifier query
    itself."""

    response = requests.get(
        f"{service_url}/query",
        params={
            "where": "1=1",
            "outFields": PIPELINE_NUMBER_FIELD,
            "returnDistinctValues": "true",
            "returnGeometry": "false",
            "f": "json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise NstaFreespanServiceError(
            f"NSTA freespan service error querying distinct values at {service_url}: "
            f"{payload['error']}"
        )
    return [
        feature["attributes"][PIPELINE_NUMBER_FIELD]
        for feature in payload.get("features", [])
        if feature.get("attributes", {}).get(PIPELINE_NUMBER_FIELD) is not None
    ]


def _cache_raw_response(cache_dir: Path, registry_layer: str, payload: dict[str, Any]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{registry_layer.lower()}.geojson"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def acquire_freespan_layer(
    registry_layer: str,
    service_url: str,
    source_title: str,
    *,
    pipeline_numbers: tuple[str, ...],
    cache_dir: Path,
    timeout: float = REQUEST_TIMEOUT_S,
) -> tuple[FreespanLayerAcquisition, dict[str, Any]]:
    """Acquire one registry layer's PL854/PL855 freespan records (the ONE live
    request this ticket performs per layer). Returns the acquisition record plus
    the raw, unmodified GeoJSON payload for the caller to write/parse further."""

    where_clause = build_pipeline_number_where_clause(pipeline_numbers)
    payload = query_freespan_layer(service_url, where_clause, timeout=timeout)
    retrieved_at = datetime.now(UTC)
    raw_cache_path = _cache_raw_response(cache_dir, registry_layer, payload)

    schema = fetch_layer_schema(service_url, timeout=timeout)
    schema_field_names = tuple(f["name"] for f in schema.get("fields", []))

    features = payload.get("features", [])
    response_crs = None
    if isinstance(payload.get("crs"), dict):
        response_crs = payload["crs"].get("properties", {}).get("name")

    acquisition = FreespanLayerAcquisition(
        registry_layer=registry_layer,
        source_title=source_title,
        service_url=service_url,
        query_where_clause=where_clause,
        feature_count=len(features),
        raw_cache_path=raw_cache_path,
        response_crs=response_crs,
        schema_field_names=schema_field_names,
        retrieved_at=retrieved_at,
    )
    return acquisition, payload


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_acquisition_manifest_entry(acquisition: FreespanLayerAcquisition) -> dict[str, Any]:
    """Every field Section 4 requires: source agency, service URL, layer ID/label,
    query, retrieval UTC timestamp, returned feature count, raw file path, SHA256,
    response CRS, schema field names."""

    return {
        "source_agency": "North Sea Transition Authority (NSTA)",
        "registry_layer": acquisition.registry_layer,
        "source_title": acquisition.source_title,
        "service_url": acquisition.service_url,
        "query_where_clause": acquisition.query_where_clause,
        "retrieved_at_utc": acquisition.retrieved_at.isoformat(),
        "returned_feature_count": acquisition.feature_count,
        "raw_file_path": str(acquisition.raw_cache_path),
        "sha256": compute_sha256(acquisition.raw_cache_path),
        "response_crs": acquisition.response_crs,
        "schema_field_names": list(acquisition.schema_field_names),
        "raw_unmodified": True,
    }


def write_acquisition_manifest(manifest_path: Path, entries: list[dict[str, Any]]) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
    return manifest_path


@dataclass(frozen=True)
class FreespanRegistryIngestionReport:
    """Everything needed to summarize a completed ingestion run."""

    acquisitions: list[FreespanLayerAcquisition]
    manifest_path: Path
    zero_result_diagnostics: dict[str, list[str]]


def ingest_freespan_registry(
    *,
    cache_dir: Path,
    manifest_path: Path,
    pipeline_numbers: tuple[str, ...] = TARGET_PIPELINE_NUMBERS,
    timeout: float = REQUEST_TIMEOUT_S,
) -> FreespanRegistryIngestionReport:
    """End-to-end live acquisition: query both registry layers, cache raw
    responses, write the acquisition manifest. If a layer's exact-identifier
    query returns zero records, automatically runs (and reports) the Section 3
    distinct-NSTAPIPNO diagnostic instead of silently broadening the query.
    """

    manifest_entries: list[dict[str, Any]] = []
    acquisitions: list[FreespanLayerAcquisition] = []
    zero_result_diagnostics: dict[str, list[str]] = {}

    for registry_layer, service_url, source_title in _LAYERS:
        acquisition, _payload = acquire_freespan_layer(
            registry_layer,
            service_url,
            source_title,
            pipeline_numbers=pipeline_numbers,
            cache_dir=cache_dir,
            timeout=timeout,
        )
        acquisitions.append(acquisition)
        manifest_entries.append(build_acquisition_manifest_entry(acquisition))

        if acquisition.feature_count == 0:
            # Section 3: never silently broaden to fuzzy PIPE_NAME matching --
            # only ever inspect the SAME identifier field's own distinct values.
            base_service_url = service_url.rsplit("/query", 1)[0]
            zero_result_diagnostics[registry_layer] = query_distinct_pipeline_numbers(
                base_service_url, timeout=timeout
            )

    written_manifest_path = write_acquisition_manifest(manifest_path, manifest_entries)
    return FreespanRegistryIngestionReport(
        acquisitions=acquisitions,
        manifest_path=written_manifest_path,
        zero_result_diagnostics=zero_result_diagnostics,
    )


def print_ingestion_report(report: FreespanRegistryIngestionReport, *, file: Any = None) -> None:
    file = file or sys.stdout
    lines = ["=== NSTA Pipeline Freespan Registry Acquisition (MAR-014C) ===", ""]
    for acquisition in report.acquisitions:
        lines.append(f"## {acquisition.registry_layer}")
        lines.append(f"  Source:          {acquisition.source_title}")
        lines.append(f"  Service URL:     {acquisition.service_url}")
        lines.append(f"  Query:           {acquisition.query_where_clause}")
        lines.append(f"  Retrieved (UTC): {acquisition.retrieved_at.isoformat()}")
        lines.append(f"  Feature count:   {acquisition.feature_count}")
        lines.append(f"  Response CRS:    {acquisition.response_crs}")
        lines.append(f"  Raw cache:       {acquisition.raw_cache_path}")
        lines.append("")
        if acquisition.feature_count == 0:
            distinct = report.zero_result_diagnostics.get(acquisition.registry_layer, [])
            lines.append(
                f"  ZERO records for {TARGET_PIPELINE_NUMBERS} -- inspected "
                f"{len(distinct)} distinct {PIPELINE_NUMBER_FIELD} values in this layer; "
                "none match (never broadened to fuzzy PIPE_NAME matching)."
            )
            lines.append("")
    lines.append(f"Manifest: {report.manifest_path}")
    print("\n".join(lines), file=file)

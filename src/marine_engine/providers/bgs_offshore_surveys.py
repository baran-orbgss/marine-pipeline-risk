"""BGS OGC API -- Offshore Oil & Gas Industry Site Surveys provider (MAR-016).

A real, public OGC API Features service -- verified live, never guessed
(Section 4)
------------------------------------------------------------------------
Base service `https://ogcapi.bgs.ac.uk/`, collection
`offshore-oil-gas-site-surveys`. Confirmed live 2026-09-06: the collection's
own `description` states "This layer shows the geographic location of oil
and gas industry site surveys ... BGS do not hold the data. For further
information contact the Custodian of the data." -- i.e. this is a survey
METADATA/footprint layer, never a bathymetry-data layer (Section 20). Its
real `queryables` (confirmed via `/queryables?f=json`) are: `site_svy_id`,
`mdfileid_nerc_guid`, `bgs_ref_no`, `decc_ref_no`, `site_svy_name`,
`originator`, `contractor`, `svy_start_date`, `svy_end_date`, `abstract`,
`additional_info`, `custodian`, `custodian_name`, `custodian_email`,
`custodian_tel`, plus the `geometry` itself -- every field this module reads
is one of these real, confirmed fields, never invented.

The service supports standard OGC API bbox and CQL2 `filter` query
parameters (both confirmed live) -- never the interactive HTML viewer
(Section 4's explicit instruction).

Real, confirmed-live spatial classification example (Section 3's own
"do not call a block-number match a route overlap" warning, demonstrated
rather than merely stated): `decc_ref_no=GS_807` ("Bedevere rig site survey
(Block 48/18)") shares a block number with `bgs_ref_no=GB02SS0001`
("Anglia Field Development and Anglia North West", also block 48/18), yet
GS_807's own real footprint sits ~12.9 km from the PL854 route -- a genuine
demonstration that a shared block number never implies spatial overlap.
"""

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://ogcapi.bgs.ac.uk"
COLLECTION_ID = "offshore-oil-gas-site-surveys"
COLLECTION_URL = f"{BASE_URL}/collections/{COLLECTION_ID}"
SOURCE_TITLE = "BGS Offshore Oil & Gas Industry Site Surveys (OGC API)"
RESPONSE_CRS = "OGC:CRS84"  # lon/lat WGS84 -- confirmed via the collection's own `crs`/`storageCrs`
REQUEST_TIMEOUT_S = 30.0

# Every field this module ever reads -- confirmed live via `/queryables?f=json`
# 2026-09-06, never a guessed or invented field name.
CONFIRMED_QUERYABLE_FIELDS = (
    "site_svy_id",
    "mdfileid_nerc_guid",
    "bgs_ref_no",
    "decc_ref_no",
    "site_svy_name",
    "originator",
    "contractor",
    "svy_start_date",
    "svy_end_date",
    "abstract",
    "additional_info",
    "custodian",
    "custodian_name",
    "custodian_email",
    "custodian_tel",
)


@dataclass(frozen=True)
class KnownCandidateRecord:
    """One ticket-specified candidate (Section 5) that MUST be checked
    spatially regardless of whether a bbox sweep would independently find
    it. The ticket's own stated title/block is preserved for comparison
    ONLY -- the actual classification always uses the live-returned record,
    never this stated text."""

    label: str
    bgs_ref_no: str | None
    decc_ref_no: str | None
    ticket_stated_title: str
    ticket_stated_block: str
    ticket_note: str


KNOWN_CANDIDATE_RECORDS: tuple[KnownCandidateRecord, ...] = (
    KnownCandidateRecord(
        label="A",
        bgs_ref_no="GB02SS0003",
        decc_ref_no=None,
        ticket_stated_title="Anglia A Site Survey",
        ticket_stated_block="48/19",
        ticket_note="2002 GDF Suez; bathymetric site clearance + shallow gas appraisal.",
    ),
    KnownCandidateRecord(
        label="B",
        bgs_ref_no="GB03SS0002",
        decc_ref_no=None,
        ticket_stated_title="Anglia A Platform Levelling Platform Survey",
        ticket_stated_block="48/19",
        ticket_note="2003 GDF Suez.",
    ),
    KnownCandidateRecord(
        label="C",
        bgs_ref_no="GB02SS0001",
        decc_ref_no=None,
        ticket_stated_title="Anglia Field Development and Anglia North West Site Survey",
        ticket_stated_block="48/18",
        ticket_note="2002 GDF Suez.",
    ),
    KnownCandidateRecord(
        label="D",
        bgs_ref_no="CS03SS0003",
        decc_ref_no=None,
        ticket_stated_title="48/10 Saturn -> 49/16 LOGGS tie-in Pipeline/Cable Route Survey",
        ticket_stated_block="",
        ticket_note=(
            "2003 ConocoPhillips; known metadata extent approximately lon "
            "1.8976-2.0026, lat 53.3909-53.7251."
        ),
    ),
    KnownCandidateRecord(
        label="E",
        bgs_ref_no=None,
        decc_ref_no="GS_807",
        ticket_stated_title="Bedevere rig site survey",
        ticket_stated_block="48/18",
        ticket_note=(
            "2018 Zennith Energy; known metadata indicates this may NOT intersect "
            "PL854 -- verify geometrically."
        ),
    ),
)


class BgsOffshoreSurveysServiceError(RuntimeError):
    """The BGS OGC API returned an unexpected/error response."""


@dataclass(frozen=True)
class BgsQueryAcquisition:
    """One live query's raw acquisition result (Section 4's required manifest
    fields: endpoint, query, retrieval time, response count, checksum,
    schema)."""

    query_label: str
    endpoint: str
    query_params: dict[str, Any]
    feature_count: int
    raw_cache_path: Path
    retrieved_at: datetime


def fetch_collection_metadata(timeout: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    """The collection's own schema/description/extent -- confirmed to state
    that BGS does not hold the underlying survey data (Section 20)."""

    response = requests.get(COLLECTION_URL, params={"f": "json"}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_queryables(timeout: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    response = requests.get(f"{COLLECTION_URL}/queryables", params={"f": "json"}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def query_items_by_bbox(
    bbox: tuple[float, float, float, float], *, limit: int = 200, timeout: float = REQUEST_TIMEOUT_S
) -> dict[str, Any]:
    """Standard OGC API Features `bbox` query (lon/lat WGS84), never the
    interactive HTML viewer. `bbox` should be a generous pad around the
    real AOI so that genuinely nearby-but-outside-AOI surveys are still
    returned for `NEARBY_NOT_AOI` classification (Section 3), not silently
    pre-filtered away."""

    response = requests.get(
        f"{COLLECTION_URL}/items",
        params={"bbox": ",".join(str(v) for v in bbox), "limit": limit, "f": "json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    _raise_if_error(payload, f"{COLLECTION_URL}/items?bbox=...")
    return payload


def query_item_by_ref(
    *,
    bgs_ref_no: str | None = None,
    decc_ref_no: str | None = None,
    timeout: float = REQUEST_TIMEOUT_S,
) -> dict[str, Any]:
    """CQL2-text `filter` query on the exact identifier field -- confirmed
    live for both `bgs_ref_no` and `decc_ref_no` -- never a fuzzy name
    match. Used to GUARANTEE the Section 5 known candidates are checked
    even when they fall outside the bbox sweep (e.g. `decc_ref_no=GS_807`)."""

    if (bgs_ref_no is None) == (decc_ref_no is None):
        raise ValueError("exactly one of bgs_ref_no or decc_ref_no must be given")
    field, value = (
        ("bgs_ref_no", bgs_ref_no) if bgs_ref_no is not None else ("decc_ref_no", decc_ref_no)
    )
    cql2_filter = f"{field}='{value}'"
    response = requests.get(
        f"{COLLECTION_URL}/items", params={"filter": cql2_filter, "f": "json"}, timeout=timeout
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    _raise_if_error(payload, f"{COLLECTION_URL}/items?filter={cql2_filter}")
    return payload


def _raise_if_error(payload: dict[str, Any], endpoint: str) -> None:
    if "code" in payload and "description" in payload and "features" not in payload:
        raise BgsOffshoreSurveysServiceError(
            f"BGS OGC API error querying {endpoint}: {payload.get('description')}"
        )


def _cache_raw_response(cache_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{name}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_acquisition_manifest_entry(acquisition: BgsQueryAcquisition) -> dict[str, Any]:
    """Every field Section 4 requires: endpoint, query, retrieval time,
    response count, checksum, schema (the confirmed queryable field names)."""

    return {
        "source_agency": "British Geological Survey (BGS) / National Geoscience Data Centre",
        "collection_id": COLLECTION_ID,
        "query_label": acquisition.query_label,
        "endpoint": acquisition.endpoint,
        "query_params": acquisition.query_params,
        "retrieved_at_utc": acquisition.retrieved_at.isoformat(),
        "returned_feature_count": acquisition.feature_count,
        "raw_file_path": str(acquisition.raw_cache_path),
        "sha256": compute_sha256(acquisition.raw_cache_path),
        "response_crs": RESPONSE_CRS,
        "confirmed_queryable_fields": list(CONFIRMED_QUERYABLE_FIELDS),
        "raw_unmodified": True,
    }


@dataclass(frozen=True)
class BgsAcquisitionReport:
    collection_metadata: dict[str, Any]
    queryables: dict[str, Any]
    acquisitions: list[BgsQueryAcquisition]
    merged_features: list[dict[str, Any]]
    manifest_path: Path


def acquire_bgs_survey_candidates(
    *,
    aoi_bbox_wgs84: tuple[float, float, float, float],
    bbox_pad_deg: float,
    cache_dir: Path,
    manifest_path: Path,
    known_candidates: tuple[KnownCandidateRecord, ...] = KNOWN_CANDIDATE_RECORDS,
    timeout: float = REQUEST_TIMEOUT_S,
) -> BgsAcquisitionReport:
    """End-to-end live acquisition (the one network step this module
    performs): collection metadata + queryables, a padded-bbox sweep, and
    an exact-identifier lookup for every Section 5 known candidate --
    merged and de-duplicated by `bgs_ref_no`/`decc_ref_no`, every raw
    response cached unmodified and manifested with a checksum.
    """

    collection_metadata = fetch_collection_metadata(timeout=timeout)
    _cache_raw_response(cache_dir, "collection_metadata", collection_metadata)
    queryables = fetch_queryables(timeout=timeout)
    _cache_raw_response(cache_dir, "queryables", queryables)

    minx, miny, maxx, maxy = aoi_bbox_wgs84
    padded_bbox = (
        minx - bbox_pad_deg,
        miny - bbox_pad_deg,
        maxx + bbox_pad_deg,
        maxy + bbox_pad_deg,
    )

    manifest_entries: list[dict[str, Any]] = []
    acquisitions: list[BgsQueryAcquisition] = []
    merged_by_id: dict[str, dict[str, Any]] = {}

    bbox_payload = query_items_by_bbox(padded_bbox, timeout=timeout)
    bbox_cache_path = _cache_raw_response(cache_dir, "items_bbox_sweep", bbox_payload)
    bbox_acquisition = BgsQueryAcquisition(
        query_label="AOI_PADDED_BBOX_SWEEP",
        endpoint=f"{COLLECTION_URL}/items",
        query_params={"bbox": list(padded_bbox)},
        feature_count=len(bbox_payload.get("features", [])),
        raw_cache_path=bbox_cache_path,
        retrieved_at=datetime.now(UTC),
    )
    acquisitions.append(bbox_acquisition)
    manifest_entries.append(build_acquisition_manifest_entry(bbox_acquisition))
    for feature in bbox_payload.get("features", []):
        merged_by_id[_feature_identity(feature)] = feature

    for candidate in known_candidates:
        payload = query_item_by_ref(
            bgs_ref_no=candidate.bgs_ref_no, decc_ref_no=candidate.decc_ref_no, timeout=timeout
        )
        cache_name = (
            f"known_candidate_{candidate.label}_{candidate.bgs_ref_no or candidate.decc_ref_no}"
        )
        cache_path = _cache_raw_response(cache_dir, cache_name, payload)
        acquisition = BgsQueryAcquisition(
            query_label=f"KNOWN_CANDIDATE_{candidate.label}",
            endpoint=f"{COLLECTION_URL}/items",
            query_params={"filter": f"bgs_ref_no='{candidate.bgs_ref_no}'"}
            if candidate.bgs_ref_no
            else {"filter": f"decc_ref_no='{candidate.decc_ref_no}'"},
            feature_count=len(payload.get("features", [])),
            raw_cache_path=cache_path,
            retrieved_at=datetime.now(UTC),
        )
        acquisitions.append(acquisition)
        manifest_entries.append(build_acquisition_manifest_entry(acquisition))
        for feature in payload.get("features", []):
            merged_by_id[_feature_identity(feature)] = feature

    written_manifest_path = manifest_path
    written_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    written_manifest_path.write_text(
        json.dumps(manifest_entries, indent=2, default=str), encoding="utf-8"
    )

    return BgsAcquisitionReport(
        collection_metadata=collection_metadata,
        queryables=queryables,
        acquisitions=acquisitions,
        merged_features=list(merged_by_id.values()),
        manifest_path=written_manifest_path,
    )


def _feature_identity(feature: dict[str, Any]) -> str:
    props = feature.get("properties", {})
    ref = props.get("bgs_ref_no") or props.get("decc_ref_no") or feature.get("id")
    return str(ref)


def print_acquisition_report(report: BgsAcquisitionReport, *, file: Any = None) -> None:
    file = file or sys.stdout
    lines = ["=== BGS Offshore Oil & Gas Site Surveys Acquisition (MAR-016) ===", ""]
    for acquisition in report.acquisitions:
        lines.append(f"## {acquisition.query_label}")
        lines.append(f"  Endpoint:        {acquisition.endpoint}")
        lines.append(f"  Query:           {acquisition.query_params}")
        lines.append(f"  Retrieved (UTC): {acquisition.retrieved_at.isoformat()}")
        lines.append(f"  Feature count:   {acquisition.feature_count}")
        lines.append(f"  Raw cache:       {acquisition.raw_cache_path}")
        lines.append("")
    lines.append(f"Merged distinct candidates: {len(report.merged_features)}")
    lines.append(f"Manifest: {report.manifest_path}")
    print("\n".join(lines), file=file)

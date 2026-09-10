"""Sheringham Shoal 2008 (GEO) CPT/CPTU geotechnical investigation -- real MDE source (MAR-032).

Real, verified-live source (never guessed)
---------------------------------------------
The Crown Estate's Marine Data Exchange (MDE) series `TCE-1964`: "2008, GEO, Sheringham Shoal,
Cone Penetration Test Geotechnical Investigation". Canonical source page
`https://www.marinedataexchange.co.uk/details/1964/summary` (confirmed live 2026-09-10; the site
canonicalizes it to `/details/TCE-1964/2008-geo-sheringham-shoal-cone-penetration-test-
geotechnical-investigation`). Published by The Crown Estate; contractor GEO (Kgs. Lyngby, DK) for
Scira Offshore Energy Ltd; collection 29 September - 16 October 2008 (source page: "October 2008
- October 2008"). Source-page facts recorded as SOURCE-DECLARED collection metadata: "101
continuous CPTs (CPTU) in 86 locations", datasets: 1, reports: 1, dataset label
"Cone PenetrationTests Logs". They are recorded, never enforced as expected parsed counts.

Acquisition mechanics (confirmed live, same shape as MAR-020/021/022)
----------------------------------------------------------------------
The MDE site lists the package's files individually through its own API
(`.../api/Collection/7017/storageItems?directory=/&tenant=TCE`, collection id 7017 = the
"Cone PenetrationTests Logs" dataset package) but serves ONE combined ZIP per package. The two
package URLs below were resolved by intercepting the page's own "Download Dataset" button anchor-
click targets in a real browser session -- never constructed from a guessed pattern:

    https://www.marinedataexchange.co.uk/pub/TCE/122-1964-Cone%20PenetrationTests%20Logs.zip
        HEAD: 200, Content-Length 2,658,522, application/octet-stream, Accept-Ranges: bytes
    https://www.marinedataexchange.co.uk/pub/TCE/122-1964-Part%20B%20(Geotechnical%20Data)
        %20and%20Part%20C%20(Field%20Operations)%20Reports.zip
        HEAD: 200, Content-Length 34,642,187, application/octet-stream, Accept-Ranges: bytes

Both are small, so each is a single ordinary whole-file GET saved UNALTERED, then extracted
beside it; a JSON sidecar makes every rerun offline (cache reuse, SHA-256 re-verified).

What the CPT package actually contains (inspected bytes, not assumed)
-----------------------------------------------------------------------
100 files `CPT-<id>.csv` (ISO-8859 text, CRLF, semicolon-delimited, decimal point), one
`_metadata/medin_metadata.xml` (MEDIN discovery metadata), one `dataname.txt` ("CPT"). Every CSV
has the identical structure -- seven header lines, a column-name row, a unit row, then numeric
rows:

    Job ;31800;Sheringham Shoal Wind Farm
    CPT name;;CPT-A2
    Date & Time ;2008-10-14 5:47:16 PM
    Depth ;15.70
    Clients ref. :;
    Position:; 371996.3; 5892425.5;WGS 84;UTM;31
    Dist. Tip to Sleeve center;10.50
    Scan#;Depth;Tip;Sleeve;Pore;Incl;Time;
    #;m.;MPa;MPa;MPa;<degree sign>;

So the source provides DEFENSIBLY MACHINE-READABLE NUMERIC CPT/CPTU PROFILES. The report package
holds two PDFs (Part B Geotechnical Data, Part C Field Operations) -- DOCUMENTARY; never digitized.

Source-established semantics (quoted; nothing below is inferred from column names alone)
---------------------------------------------------------------------------------------------
See the `SOURCE_*` constants: depth zero definition, qc / fs / u definitions, pore-filter
location (u2), cone area, the source's OWN area ratio a = 0.75 (used by GEO for its qt plots --
recorded, NOT applied here: qt_mpa stays null), and the coordinate system definition. The
header line `Depth ;15.70` is a separate header field whose meaning the CSV does not state; it
is preserved raw and NOT interpreted (Part B's log table lists a "Seabed Level (m)" of equal
magnitude and opposite sign relative to Chart Datum at Cromer, while Part C's geodetic block
names Mean Sea Level -- a source-internal inconsistency that is reported, not resolved).
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pandas as pd
import requests
from pyproj import CRS
from pyproj.exceptions import CRSError

from marine_engine.geotechnical import cpt_contract as contract
from marine_engine.geotechnical import cpt_profile

__all__ = [
    "PROVIDER_ID",
    "PackageAcquisition",
    "ProviderCanonical",
    "CrsAssessment",
    "GeoCptFormatError",
    "source_declaration",
    "acquire",
    "inventory_probes",
    "probe_geo_cpt_csv",
    "parse_geo_cpt_csv",
    "channel_declarations",
    "depth_declaration",
    "build_canonical",
    "assess_crs",
    "source_reference_crs",
    "horizontal_axis_unit",
    "SOURCE_REFERENCE_CRS",
    "SOURCE_HORIZONTAL_UNIT",
    "acquisition_to_dict",
]

PROVIDER_ID = "sheringham_shoal_2008_cptu"

MDE_SERIES_ID = "TCE-1964"
MDE_SOURCE_PAGE_URL = "https://www.marinedataexchange.co.uk/details/1964/summary"
MDE_CANONICAL_PAGE_URL = (
    "https://www.marinedataexchange.co.uk/details/TCE-1964/"
    "2008-geo-sheringham-shoal-cone-penetration-test-geotechnical-investigation"
)
MDE_CPT_PACKAGE_COLLECTION_ID = 7017
CPT_LOGS_PACKAGE_URL = (
    "https://www.marinedataexchange.co.uk/pub/TCE/122-1964-Cone%20PenetrationTests%20Logs.zip"
)
REPORTS_PACKAGE_URL = (
    "https://www.marinedataexchange.co.uk/pub/TCE/122-1964-Part%20B%20(Geotechnical%20Data)"
    "%20and%20Part%20C%20(Field%20Operations)%20Reports.zip"
)
URL_RESOLUTION_METHOD = (
    "resolved live (2026-09-10) by intercepting the MDE series page's own 'Download Dataset' "
    "button anchor-click targets in a browser session; not constructed from a guessed pattern"
)

CPT_LOGS_PACKAGE_ID = "cpt_logs"
REPORTS_PACKAGE_ID = "reports"

DATASET_TITLE = "2008, GEO, Sheringham Shoal, Cone Penetration Test Geotechnical Investigation"
DATASET_PACKAGE_LABEL = "Cone PenetrationTests Logs"
REPORTS_PACKAGE_LABEL = "Part B (Geotechnical Data) and Part C (Field Operations) Reports"
PUBLISHER = "The Crown Estate"
CONTRACTOR = "GEO (Maglebjergvej 1, DK-2800 Kgs. Lyngby) for Scira Offshore Energy Ltd"
COLLECTION_PERIOD = "2008-09-29/2008-10-16 (source page: October 2008 - October 2008)"
SOURCE_DECLARED_TEST_COUNT = 101
SOURCE_DECLARED_LOCATION_COUNT = 86
SOURCE_DECLARED_DATASET_COUNT = 1
SOURCE_DECLARED_REPORT_COUNT = 1
GEO_PROJECT_NUMBER = "31800"
CLIENT_DOCUMENT_NUMBER = "SC-00-NN-G15-00008"
LICENCE = (
    "The Crown Estate Marine Data Exchange Terms of Use: a non-exclusive, non-transferable "
    "licence (without the right to sublicense) to copy and use the information -- the same "
    "sitewide policy confirmed for other TCE-prefixed series in this project."
)
REQUEST_TIMEOUT_S = 300.0

# --- Source-established semantics (verbatim quotes from the acquired Part B / Part C PDFs) --------

SOURCE_DEPTH_ZERO_DEFINITION = (
    'Part B, B.11.2: "Start of test (depth 0 m) is defined as the interface between the rig '
    'bottom plate and the seabed, hence, when the cone is exiting out of the rig." Legend B.11.0: '
    '"Depth: Depth refers to the penetration depth below start of test level". Part C, C.4.4: '
    '"The offset of each cone sensor was eliminated (zeroed) after the rig had settled on the '
    "seabed just before commencement of the test, at which time the cone tip was positioned at "
    'the reference level."'
)
SOURCE_DEPTH_REFERENCE_LIMITATION = (
    "Depth zero is the seabed/rig-bottom-plate interface after rig settlement; any plate "
    "embedment into soft seabed is not quantified by the source. Negative depth rows are the cone "
    "still inside the rig before exiting; they are preserved, never removed."
)
SOURCE_QC_DEFINITION = (
    'Part B, B.11.2: "qc is the measured cone resistance." Legend: "qc: Tip resistance". The CSV '
    "column 'Tip' (MPa) is therefore the MEASURED cone resistance qc."
)
SOURCE_FS_DEFINITION = (
    'Part B, B.11.2: "fs is the measured sleeve friction." CSV column: Sleeve (MPa).'
)
SOURCE_U_DEFINITION = (
    'Part B, B.11.2: "u is the pore water pressure (relative to seabed level)." Part C, C.4.3: '
    '"Tip resistance, sleeve friction, pore water pressure (single filter located just behind '
    'the cone tip) and inclination of the cone were recorded during each test." A single filter '
    "just behind the cone tip is the u2 position; CSV column: Pore (MPa)."
)
SOURCE_QT_STATEMENT = (
    'Part B, B.11.2: "qt is the corrected cone resistance ... defined by qt = qc + (1 - a) u, '
    "a = 0.75\". qt appears only in the source's PDF plots; it is NOT a CSV column. MAR-032 does "
    "not apply this correction: qt_mpa = null."
)
SOURCE_CONE_AREA_RATIO = 0.75  # source-stated for GEO's own qt plots; recorded, never applied
SOURCE_CONE_AREA_CM2 = 10.0
SOURCE_CONE_DEFINITION = (
    'Part C, C.4.2: "The cones used were standard analog Van den Berg 60-degree type with cross '
    'sectional areas of 10 cm2." Legend: "Cone area: 10.0 cm2 in agreement with the ISOPT1 '
    'recommendations". Penetration 20 mm/s, data recorded every second (Part C, C.4.3).'
)
SOURCE_INCLINATION_DEFINITION = "Cone inclination, CSV column 'Incl', unit row: degree sign."
SOURCE_TIME_STATEMENT = (
    "CSV column 'Time' carries no unit in the unit row; preserved raw only (numeric magnitude is "
    "consistent with a spreadsheet serial day, but that is not stated by the source and is not "
    "assumed)."
)
SOURCE_COORDINATE_STATEMENT = (
    'Part B, B.11.3: "All coordinates are UTM 31/WGS 84." Part C, 1.4 Geodetic Parameters: '
    "Geodetic datum WGS 84; ellipsoid semi-major axis 6378137 m, inverse flattening "
    "298.2572235630; projection UTM zone 31, latitude of natural origin 0 degrees north, longitude "
    "of natural origin 3 degrees east, scale factor 0.9996, false easting 500 000.00, false "
    "northing 0.00, metre. Each CSV header carries 'WGS 84;UTM;31' beside its easting/northing."
)
SOURCE_VERTICAL_STATEMENT = (
    'Part B, B.11.3: "The levels are Chart Datum (CD) with reference at Cromer." Part C, 1.4.3: '
    '"Definition of vertical datum: Mean Sea Level (MSL)". Source-internal inconsistency for the '
    "seabed LEVEL only; it does not affect penetration depth below seabed."
)
SOURCE_UTM_ZONE = "31N"
SOURCE_DATUM_NAME_FRAGMENT = "1984"  # matches pyproj datum name "World Geodetic System 1984"
# MAR-032A: the source-defined system (Part C 1.4: WGS 84; UTM zone 31; natural origin 0N / 3E;
# scale 0.9996; false easting 500 000; false northing 0; metre) IS EPSG:32631. A declared CRS is
# accepted only when pyproj establishes SEMANTIC equivalence with this reference CRS and its
# horizontal axes are in metres -- a WGS 84 / UTM 31N system in feet is a material conflict.
SOURCE_REFERENCE_CRS_EPSG = 32631
SOURCE_REFERENCE_CRS = f"EPSG:{SOURCE_REFERENCE_CRS_EPSG}"
SOURCE_HORIZONTAL_UNIT = "metre"
SOURCE_HORIZONTAL_UNIT_TO_M_FACTOR = 1.0

# Verbatim tokens as they appear in the CSV (latin-1). The unit row is exactly this.
GEO_COLUMN_HEADER = "Scan#;Depth;Tip;Sleeve;Pore;Incl;Time;"
GEO_UNIT_ROW = "#;m.;MPa;MPa;MPa;\xb0;"
GEO_COLUMNS = ("Scan#", "Depth", "Tip", "Sleeve", "Pore", "Incl", "Time")
GEO_HEADER_LINE_COUNT = 7
# Explicit, deterministic alias of the SOURCE'S verbatim unit tokens -> canonical unit tokens.
GEO_UNIT_TOKEN_ALIASES = {"m.": "m", "MPa": "MPa", "\xb0": "deg", "#": None, "": None}


class GeoCptFormatError(ValueError):
    """The file is not a GEO-format CPT CSV of the structure inspected in this real package."""


@dataclass(frozen=True)
class PackageAcquisition:
    package_id: str
    package_label: str
    source_page_url: str
    package_url: str
    url_resolution_method: str
    local_archive_path: Path
    extracted_dir: Path
    package_bytes: int
    package_sha256: str
    archive_type: str
    http_evidence: dict[str, Any]
    retrieved_at_utc: datetime
    already_cached: bool
    licence_note: str


def acquisition_to_dict(acq: PackageAcquisition) -> dict[str, Any]:
    d = asdict(acq)
    d["local_archive_path"] = str(acq.local_archive_path)
    d["extracted_dir"] = str(acq.extracted_dir)
    d["retrieved_at_utc"] = acq.retrieved_at_utc.isoformat()
    return d


def source_declaration() -> dict[str, Any]:
    """Section 7/28: the publisher's own collection metadata, recorded as a DECLARATION."""

    return {
        "role": contract.SOURCE_DECLARED_COLLECTION_METADATA,
        "source_name": DATASET_TITLE,
        "source_uri": MDE_SOURCE_PAGE_URL,
        "source_canonical_uri": MDE_CANONICAL_PAGE_URL,
        "source_series_id": MDE_SERIES_ID,
        "source_publisher": PUBLISHER,
        "source_contractor": CONTRACTOR,
        "source_collection_period": COLLECTION_PERIOD,
        "source_declared_test_count": SOURCE_DECLARED_TEST_COUNT,
        "source_declared_location_count": SOURCE_DECLARED_LOCATION_COUNT,
        "source_declared_dataset_count": SOURCE_DECLARED_DATASET_COUNT,
        "source_declared_report_count": SOURCE_DECLARED_REPORT_COUNT,
        "source_dataset_package_label": DATASET_PACKAGE_LABEL,
        "source_reports_package_label": REPORTS_PACKAGE_LABEL,
        "geo_project_number": GEO_PROJECT_NUMBER,
        "client_document_number": CLIENT_DOCUMENT_NUMBER,
        "licence_note": LICENCE,
        "declared_counts_are_enforced_as_parsed_counts": False,
    }


def _archive_name_from_url(url: str) -> str:
    return unquote(urlsplit(url).path.rsplit("/", 1)[-1])


def _sidecar_path(raw_dir: Path, package_id: str) -> Path:
    return raw_dir / f"{package_id}.acquisition.json"


def _extract(archive_path: Path, extracted_dir: Path) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extracted_dir)


def _acquire_package(
    raw_dir: Path,
    *,
    package_id: str,
    package_label: str,
    url: str,
    log=None,
) -> PackageAcquisition:
    """One whole-file GET (both packages are small), saved unaltered, extracted beside it, and
    cached via a JSON sidecar. On a cache hit no network request is made at all; the archive's
    SHA-256 is recomputed and must equal the recorded one (integrity, never silent)."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / _archive_name_from_url(url)
    extracted_dir = raw_dir / package_id
    sidecar = _sidecar_path(raw_dir, package_id)

    if sidecar.exists() and archive_path.exists():
        cached = json.loads(sidecar.read_text(encoding="utf-8"))
        if archive_path.stat().st_size == cached["package_bytes"]:
            sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            if sha256 != cached["package_sha256"]:
                raise ValueError(
                    f"cached archive {archive_path} SHA-256 {sha256} does not match recorded "
                    f"{cached['package_sha256']}; refusing to reuse silently"
                )
            if not extracted_dir.is_dir() or not any(extracted_dir.iterdir()):
                _extract(archive_path, extracted_dir)
            if log:
                log(f"[{package_id}] cache hit: {archive_path.name} ({cached['package_bytes']} B)")
            return PackageAcquisition(
                package_id=package_id,
                package_label=package_label,
                source_page_url=MDE_SOURCE_PAGE_URL,
                package_url=url,
                url_resolution_method=URL_RESOLUTION_METHOD,
                local_archive_path=archive_path,
                extracted_dir=extracted_dir,
                package_bytes=cached["package_bytes"],
                package_sha256=sha256,
                archive_type="application/zip",
                http_evidence=cached.get("http_evidence", {}),
                retrieved_at_utc=datetime.fromisoformat(cached["retrieved_at_utc"]),
                already_cached=True,
                licence_note=LICENCE,
            )

    if log:
        log(f"[{package_id}] acquiring {url}")
    response = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"PK\x03\x04"):
        raise ValueError(f"{url}: response is not a ZIP archive (magic {content[:4]!r})")
    archive_path.write_bytes(content)
    sha256 = hashlib.sha256(content).hexdigest()
    retrieved_at = datetime.now(UTC)
    http_evidence = {
        "status_code": response.status_code,
        "content_length_header": response.headers.get("Content-Length"),
        "content_type": response.headers.get("Content-Type"),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "accept_ranges": response.headers.get("Accept-Ranges"),
        "final_url": response.url,
    }
    _extract(archive_path, extracted_dir)
    sidecar.write_text(
        json.dumps(
            {
                "package_url": url,
                "package_bytes": len(content),
                "package_sha256": sha256,
                "retrieved_at_utc": retrieved_at.isoformat(),
                "http_evidence": http_evidence,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PackageAcquisition(
        package_id=package_id,
        package_label=package_label,
        source_page_url=MDE_SOURCE_PAGE_URL,
        package_url=url,
        url_resolution_method=URL_RESOLUTION_METHOD,
        local_archive_path=archive_path,
        extracted_dir=extracted_dir,
        package_bytes=len(content),
        package_sha256=sha256,
        archive_type="application/zip",
        http_evidence=http_evidence,
        retrieved_at_utc=retrieved_at,
        already_cached=False,
        licence_note=LICENCE,
    )


def acquire(raw_dir: Path, *, include_reports: bool, log=None) -> list[PackageAcquisition]:
    """Section 8: acquire the CPT dataset package and, when requested, the associated report
    package (needed here to establish depth reference, channel semantics and the CRS)."""

    acquisitions = [
        _acquire_package(
            raw_dir,
            package_id=CPT_LOGS_PACKAGE_ID,
            package_label=DATASET_PACKAGE_LABEL,
            url=CPT_LOGS_PACKAGE_URL,
            log=log,
        )
    ]
    if include_reports:
        acquisitions.append(
            _acquire_package(
                raw_dir,
                package_id=REPORTS_PACKAGE_ID,
                package_label=REPORTS_PACKAGE_LABEL,
                url=REPORTS_PACKAGE_URL,
                log=log,
            )
        )
    return acquisitions


# --- Inventory probes -----------------------------------------------------------------------------


def probe_geo_cpt_csv(path: Path, head: bytes) -> bool:
    """Strict content probe: the file's own leading bytes must carry the GEO header block AND the
    exact column row. A '.csv' extension alone is never sufficient; a PDF can never pass."""

    text = head.decode("latin-1")
    return "CPT name;" in text and GEO_COLUMN_HEADER in text and GEO_UNIT_ROW in text


def inventory_probes(package_id: str) -> dict[str, Any]:
    if package_id == CPT_LOGS_PACKAGE_ID:
        return {
            "profile_probe": probe_geo_cpt_csv,
            "location_probe": None,
            "documentary_role": None,
        }
    if package_id == REPORTS_PACKAGE_ID:
        # The publisher itself labels this package "... Reports" (package title and its own
        # `_metadata/reports_metadata.xml`), so its documentary files are factual reports.
        return {
            "profile_probe": probe_geo_cpt_csv,
            "location_probe": None,
            "documentary_role": contract.ROLE_FACTUAL_REPORT,
        }
    return {"profile_probe": None, "location_probe": None, "documentary_role": None}


# --- Strict GEO CSV reader ------------------------------------------------------------------------


@dataclass(frozen=True)
class GeoCptCsv:
    header: dict[str, list[str]]
    columns: tuple[str, ...]
    unit_tokens: tuple[str, ...]
    observations: pd.DataFrame


def parse_geo_cpt_csv(path: Path) -> GeoCptCsv:
    text = path.read_bytes().decode("latin-1")
    lines = text.splitlines()
    if len(lines) < GEO_HEADER_LINE_COUNT + 2:
        raise GeoCptFormatError(f"{path.name}: too short for a GEO CPT CSV")
    if lines[GEO_HEADER_LINE_COUNT] != GEO_COLUMN_HEADER:
        raise GeoCptFormatError(
            f"{path.name}: column row {lines[GEO_HEADER_LINE_COUNT]!r} != {GEO_COLUMN_HEADER!r}"
        )
    if lines[GEO_HEADER_LINE_COUNT + 1] != GEO_UNIT_ROW:
        raise GeoCptFormatError(
            f"{path.name}: unit row {lines[GEO_HEADER_LINE_COUNT + 1]!r} != {GEO_UNIT_ROW!r}"
        )
    header: dict[str, list[str]] = {}
    for line in lines[:GEO_HEADER_LINE_COUNT]:
        parts = [p.strip() for p in line.split(";")]
        header[parts[0]] = parts[1:]
    required = (
        "Job",
        "CPT name",
        "Date & Time",
        "Depth",
        "Position:",
        "Dist. Tip to Sleeve center",
    )
    missing = [k for k in required if k not in header]
    if missing:
        raise GeoCptFormatError(f"{path.name}: header keys missing: {missing}")

    rows: list[list[str]] = []
    for line in lines[GEO_HEADER_LINE_COUNT + 2 :]:
        if not line.strip():
            continue
        parts = line.split(";")
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if len(parts) != len(GEO_COLUMNS):
            raise GeoCptFormatError(
                f"{path.name}: data row has {len(parts)} fields, expected {len(GEO_COLUMNS)}: "
                f"{line[:60]!r}"
            )
        rows.append(parts)
    observations = pd.DataFrame(rows, columns=list(GEO_COLUMNS))
    unit_tokens = tuple(GEO_UNIT_ROW.split(";")[: len(GEO_COLUMNS)])
    return GeoCptCsv(
        header=header, columns=GEO_COLUMNS, unit_tokens=unit_tokens, observations=observations
    )


def depth_declaration() -> cpt_profile.DepthDeclaration:
    return cpt_profile.DepthDeclaration(
        source_column="Depth",
        source_unit=GEO_UNIT_TOKEN_ALIASES["m."],
        source_unit_token="m.",
        depth_reference=contract.DEPTH_BELOW_SEABED,
        reference_basis=SOURCE_DEPTH_ZERO_DEFINITION,
    )


def channel_declarations() -> tuple[cpt_profile.ChannelDeclaration, ...]:
    """Explicit column -> semantics mapping with the source quote that justifies each one. No
    channel is declared as qt: the source's qt exists only in its PDF plots."""

    return (
        cpt_profile.ChannelDeclaration(
            source_column="Tip",
            source_unit="MPa",
            source_unit_token="MPa",
            canonical_field=contract.QC_MPA,
            semantics_basis=SOURCE_QC_DEFINITION,
        ),
        cpt_profile.ChannelDeclaration(
            source_column="Sleeve",
            source_unit="MPa",
            source_unit_token="MPa",
            canonical_field=contract.FS_KPA,
            semantics_basis=SOURCE_FS_DEFINITION,
        ),
        cpt_profile.ChannelDeclaration(
            source_column="Pore",
            source_unit="MPa",
            source_unit_token="MPa",
            canonical_field=contract.U2_KPA,
            semantics_basis=SOURCE_U_DEFINITION,
        ),
        cpt_profile.ChannelDeclaration(
            source_column="Incl",
            source_unit=GEO_UNIT_TOKEN_ALIASES["\xb0"],
            source_unit_token="\xb0",
            canonical_field=None,
            semantics_basis=SOURCE_INCLINATION_DEFINITION,
        ),
        cpt_profile.ChannelDeclaration(
            source_column="Time",
            source_unit=None,
            source_unit_token="",
            canonical_field=None,
            semantics_basis=SOURCE_TIME_STATEMENT,
        ),
    )


@dataclass(frozen=True)
class ProviderCanonical:
    measurements: pd.DataFrame
    tests: pd.DataFrame
    field_provenance: tuple[dict[str, Any], ...]
    unresolved: tuple[str, ...]
    parse_failures: tuple[str, ...]
    source_semantics: dict[str, Any]


def _header_value(header: dict[str, list[str]], key: str, index: int) -> str | None:
    values = header.get(key, [])
    if index < len(values) and values[index] != "":
        return values[index]
    return None


def build_canonical(extracted_dir: Path, *, source_id: str) -> ProviderCanonical:
    """Parse every GEO CPT CSV that passes the strict content probe, in sorted order, and
    normalize each with the explicit declarations above. Files failing the strict format check are
    reported as parse failures, never partially guessed."""

    depth = depth_declaration()
    channels = channel_declarations()
    frames: list[pd.DataFrame] = []
    test_rows: list[dict[str, Any]] = []
    unresolved: set[str] = set()
    failures: list[str] = []
    provenance: tuple[dict[str, Any], ...] = ()

    candidates = sorted(
        (p for p in extracted_dir.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(extracted_dir).as_posix(),
    )
    for path in candidates:
        data = path.read_bytes()
        if not probe_geo_cpt_csv(path, data[:4096]):
            continue
        try:
            parsed = parse_geo_cpt_csv(path)
        except GeoCptFormatError as exc:
            failures.append(str(exc))
            continue
        test_id = _header_value(parsed.header, "CPT name", 1) or _header_value(
            parsed.header, "CPT name", 0
        )
        if test_id is None:
            failures.append(f"{path.name}: header carries no CPT name")
            continue
        build = cpt_profile.build_canonical_measurements(
            parsed.observations,
            source_id=source_id,
            test_id=test_id,
            depth=depth,
            channels=channels,
            location_id=None,  # Section 17/18: 86 locations declared, none identified per test
            observation_index_column="Scan#",
        )
        frames.append(build.measurements)
        unresolved.update(build.unresolved)
        provenance = build.field_provenance
        depth_values = pd.to_numeric(parsed.observations["Depth"], errors="coerce")
        position = parsed.header.get("Position:", [])
        test_rows.append(
            {
                contract.SOURCE_ID: source_id,
                contract.TEST_ID: test_id,
                contract.LOCATION_ID: None,
                "location_id_basis": "NOT_STATED_IN_SOURCE (never inferred from test name)",
                "source_file": path.relative_to(extracted_dir).as_posix(),
                "source_file_sha256": hashlib.sha256(data).hexdigest(),
                "job_number_raw": _header_value(parsed.header, "Job", 0),
                "job_site_raw": _header_value(parsed.header, "Job", 1),
                "date_time_raw": _header_value(parsed.header, "Date & Time", 0),
                "date_time_timezone": "NOT_STATED_IN_CSV (Part C 1.4.4: all times UTC)",
                "header_depth_raw": _header_value(parsed.header, "Depth", 0),
                "header_depth_semantics": (
                    "SOURCE_HEADER_FIELD_UNRESOLVED: not interpreted (see provider docstring)"
                ),
                "clients_ref_raw": _header_value(parsed.header, "Clients ref. :", 0),
                "position_x_raw": pd.to_numeric(position[0], errors="coerce")
                if len(position) > 0
                else None,
                "position_y_raw": pd.to_numeric(position[1], errors="coerce")
                if len(position) > 1
                else None,
                "position_datum_raw": position[2] if len(position) > 2 else None,
                "position_projection_raw": position[3] if len(position) > 3 else None,
                "position_zone_raw": position[4] if len(position) > 4 else None,
                "tip_to_sleeve_center_raw": _header_value(
                    parsed.header, "Dist. Tip to Sleeve center", 0
                ),
                "tip_to_sleeve_center_unit": "NOT_STATED_IN_SOURCE",
                "row_count": int(len(parsed.observations)),
                "depth_source_min": float(depth_values.min()) if len(depth_values) else None,
                "depth_source_max": float(depth_values.max()) if len(depth_values) else None,
            }
        )

    measurements = cpt_profile.concat_canonical(frames)
    tests = pd.DataFrame(test_rows)
    semantics = {
        "depth_reference": contract.DEPTH_BELOW_SEABED,
        "depth_reference_source_definition": SOURCE_DEPTH_ZERO_DEFINITION,
        "depth_reference_limitation": SOURCE_DEPTH_REFERENCE_LIMITATION,
        "qc_definition": SOURCE_QC_DEFINITION,
        "fs_definition": SOURCE_FS_DEFINITION,
        "u2_definition": SOURCE_U_DEFINITION,
        "qt_statement": SOURCE_QT_STATEMENT,
        "cone_area_ratio_source_stated": SOURCE_CONE_AREA_RATIO,
        "cone_area_ratio_applied": False,
        "cone_area_cm2_source_stated": SOURCE_CONE_AREA_CM2,
        "cone_definition": SOURCE_CONE_DEFINITION,
        "inclination_statement": SOURCE_INCLINATION_DEFINITION,
        "time_statement": SOURCE_TIME_STATEMENT,
        "coordinate_statement": SOURCE_COORDINATE_STATEMENT,
        "vertical_statement": SOURCE_VERTICAL_STATEMENT,
        "source_unit_tokens": dict(zip(GEO_COLUMNS, GEO_UNIT_ROW.split(";"), strict=False)),
        "source_unit_token_aliases": GEO_UNIT_TOKEN_ALIASES,
    }
    return ProviderCanonical(
        measurements=measurements,
        tests=tests,
        field_provenance=provenance,
        unresolved=tuple(sorted(unresolved)),
        parse_failures=tuple(failures),
        source_semantics=semantics,
    )


# --- CRS ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CrsAssessment:
    coordinates_available: bool
    observed: dict[str, Any]
    declared_crs: str | None
    resolved_crs: str | None
    crs_resolved: bool
    conflict: str | None
    note: str
    # MAR-032A source-CRS facts (defaults are the honest 'not established' state).
    source_reference_crs: str = SOURCE_REFERENCE_CRS
    source_horizontal_unit: str = SOURCE_HORIZONTAL_UNIT
    declared_crs_semantically_matches_source: bool = False
    declared_crs_horizontal_unit: str | None = None
    declared_crs_horizontal_unit_to_m_factor: float | None = None
    reprojection_performed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared_crs": self.declared_crs,
            "resolved_crs": self.resolved_crs,
            "crs_resolved": self.crs_resolved,
            "conflict": self.conflict,
            "observed": dict(self.observed),
            "note": self.note,
            "source_reference_crs": self.source_reference_crs,
            "source_horizontal_unit": self.source_horizontal_unit,
            "declared_crs_semantically_matches_source": (
                self.declared_crs_semantically_matches_source
            ),
            "declared_crs_horizontal_unit": self.declared_crs_horizontal_unit,
            "declared_crs_horizontal_unit_to_m_factor": (
                self.declared_crs_horizontal_unit_to_m_factor
            ),
            "reprojection_performed": self.reprojection_performed,
        }


def source_reference_crs() -> CRS:
    """The source-defined coordinate system as a pyproj CRS: EPSG:32631 (WGS 84 / UTM zone 31N,
    metre). This is the ONLY comparison target; nothing is inferred from coordinates or site."""

    return CRS.from_epsg(SOURCE_REFERENCE_CRS_EPSG)


def horizontal_axis_unit(crs: CRS) -> tuple[str | None, float | None]:
    """(unit name, unit-to-metre factor) shared by the CRS's horizontal axes. The factor is
    reported only for a projected CRS (a degree has no length factor); None when the axes are
    absent or carry different units."""

    axes = list(crs.axis_info)
    if not axes:
        return None, None
    names = {a.unit_name for a in axes}
    factors = {float(a.unit_conversion_factor) for a in axes}
    if len(names) != 1 or len(factors) != 1:
        return None, None
    return names.pop(), (factors.pop() if crs.is_projected else None)


def _short(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def assess_crs(declared_crs: str | None, tests: pd.DataFrame) -> CrsAssessment:
    """Section 17: source coordinates are preserved first. The source itself states 'UTM 31 /
    WGS 84' with false northing 0.00 (northern hemisphere), i.e. EPSG:32631. A USER-declared CRS
    is accepted as the resolved CRS ONLY if pyproj establishes it is SEMANTICALLY that same system
    (`CRS.from_epsg(32631) == declared`; an equivalent WKT without an EPSG label passes, a string
    match is neither required nor sufficient) AND its horizontal axes are in metres. Nothing is
    guessed from coordinate magnitude or site location; nothing is reprojected; with no
    declaration the CRS stays unresolved and no geospatial layer is written."""

    if tests.empty or "position_x_raw" not in tests.columns:
        return CrsAssessment(
            False, {}, declared_crs, None, False, None, contract.SPATIAL_LOCATION_UNRESOLVED
        )
    x = pd.to_numeric(tests["position_x_raw"], errors="coerce")
    y = pd.to_numeric(tests["position_y_raw"], errors="coerce")
    coords_available = bool(x.notna().any() and y.notna().any())
    observed = {
        "position_datum_raw_values": sorted(tests["position_datum_raw"].dropna().unique().tolist()),
        "position_projection_raw_values": sorted(
            tests["position_projection_raw"].dropna().unique().tolist()
        ),
        "position_zone_raw_values": sorted(tests["position_zone_raw"].dropna().unique().tolist()),
        "tests_with_coordinates": int((x.notna() & y.notna()).sum()),
        "source_report_statement": SOURCE_COORDINATE_STATEMENT,
    }
    if not coords_available:
        return CrsAssessment(
            False, observed, declared_crs, None, False, None, contract.SPATIAL_LOCATION_UNRESOLVED
        )
    csv_consistent = (
        observed["position_datum_raw_values"] == ["WGS 84"]
        and observed["position_projection_raw_values"] == ["UTM"]
        and observed["position_zone_raw_values"] == ["31"]
    )
    if not csv_consistent:
        return CrsAssessment(
            True,
            observed,
            declared_crs,
            None,
            False,
            "CSV header coordinate tokens are not uniformly 'WGS 84;UTM;31'",
            contract.CRS_UNRESOLVED,
        )
    if declared_crs is None:
        return CrsAssessment(
            True,
            observed,
            None,
            None,
            False,
            None,
            f"{contract.CRS_UNRESOLVED}: no declared CRS to compare against the source statement",
        )
    try:
        crs = CRS.from_user_input(declared_crs)
    except CRSError:
        return CrsAssessment(
            True,
            observed,
            declared_crs,
            None,
            False,
            f"declared CRS {declared_crs!r} is not a valid CRS identifier",
            contract.CRS_UNRESOLVED,
        )
    reference = source_reference_crs()
    unit_name, unit_factor = horizontal_axis_unit(crs)
    semantically_equal = bool(reference == crs)  # pyproj semantic comparison, axis order kept
    unit_is_metre = unit_factor is not None and math.isclose(
        unit_factor, SOURCE_HORIZONTAL_UNIT_TO_M_FACTOR, rel_tol=0.0, abs_tol=1e-12
    )
    # Diagnostics only -- none of these establishes equivalence; the verdict is the pyproj
    # comparison plus the explicit horizontal-unit check.
    datum_name = (crs.datum.name if crs.datum is not None else "") or ""
    problems: list[str] = []
    if not crs.is_projected:
        problems.append("not projected")
    if crs.utm_zone != SOURCE_UTM_ZONE:
        problems.append(f"UTM zone {crs.utm_zone!r} != source-stated {SOURCE_UTM_ZONE!r}")
    if SOURCE_DATUM_NAME_FRAGMENT not in datum_name:
        problems.append(f"datum {datum_name!r} is not WGS 84")
    if crs.is_projected and not unit_is_metre:
        problems.append(
            f"horizontal axis unit {unit_name!r} (1 unit = {unit_factor!r} m) is not "
            f"{SOURCE_HORIZONTAL_UNIT}"
        )
    if not semantically_equal and not problems:
        problems.append(
            f"not semantically equivalent to {SOURCE_REFERENCE_CRS} under pyproj CRS comparison "
            "(projection parameters or datum definition differ)"
        )
    if not semantically_equal or not unit_is_metre:
        return CrsAssessment(
            coordinates_available=True,
            observed=observed,
            declared_crs=declared_crs,
            resolved_crs=None,
            crs_resolved=False,
            conflict=(
                f"declared CRS {_short(declared_crs)!r} is not semantically the source-defined "
                f"{SOURCE_REFERENCE_CRS} (WGS 84 / UTM zone 31N, {SOURCE_HORIZONTAL_UNIT}): "
                + "; ".join(problems)
            ),
            note="material declared-vs-source CRS conflict; no layer written, nothing reprojected",
            declared_crs_semantically_matches_source=False,
            declared_crs_horizontal_unit=unit_name,
            declared_crs_horizontal_unit_to_m_factor=unit_factor,
        )
    return CrsAssessment(
        coordinates_available=True,
        observed=observed,
        declared_crs=declared_crs,
        # The declared CRS verified semantically identical to the source-defined system, so the
        # resolved CRS is that system; the raw source coordinates are labelled, never converted.
        resolved_crs=SOURCE_REFERENCE_CRS,
        crs_resolved=True,
        conflict=None,
        note=(
            f"declared CRS is semantically the source-defined {SOURCE_REFERENCE_CRS} (WGS 84 / "
            f"UTM zone 31N, false northing 0.00, {SOURCE_HORIZONTAL_UNIT}) under pyproj CRS "
            "comparison; horizontal unit metre verified; nothing reprojected"
        ),
        declared_crs_semantically_matches_source=True,
        declared_crs_horizontal_unit=unit_name,
        declared_crs_horizontal_unit_to_m_factor=unit_factor,
    )

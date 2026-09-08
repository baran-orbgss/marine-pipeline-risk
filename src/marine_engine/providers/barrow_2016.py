"""2016 Deep BV Barrow Offshore Wind Farm Export Cable Depth of Burial Survey (MAR-024).

Crown Estate Marine Data Exchange, TCE-48: "2016, Deep BV, Barrow, Export
Cable Geophysical Depth of Burial Survey" (collection Sep-Oct 2016,
published May 2021). Real objectives, quoted directly from the source's
own MEDIN lineage statement (never paraphrased into a different claim):
"Acquire depth of burial (DoB or 'z') of the cable ... using the cable
tracking system"; "Verify DoB cable tracker data with sub-bottom profiler
data"; "Classify targets on the seabed, including exposed cables, free
spans and rock dumping areas." The series covers three export cables
(Barrow, Walney I, Walney II) but every real file in the packages used
here is filename-scoped to Barrow only (`P3111_BOW_...`, BOW = Barrow
Offshore Wind), confirmed by direct inspection.

Minimal acquisition only (Section 3)
-------------------------------------
Only two packages are downloaded: the Depth of Burial Listing (required)
and the Route Position List Files (required, to spatialize the DoB
values against a real, authoritative route -- never an invented
centreline). Both real download URLs were confirmed live via the same
button-click-intercept technique already established in this project.
Neither Multibeam Bathymetry, Side-scan Sonar, Sub-bottom Profiler,
Trackplots, Charts, Cable Tracker Survey, nor Seabed Features packages are
acquired: the DoB Listing itself already carries a real, explicit
source-interpreted exposure flag (see below), so no supplementary
"seabed-feature interpretation" package is needed to satisfy Section 9.

Real inspection of the extracted packages, before any classification
logic was written (Section 2's "do NOT assume file formats from package
labels")
-------------------------------------------------------------------------------
**Depth of Burial Listing** (`P3111_BOW_DoB_listing_correlated_with_acoustics_R03.xlsx`,
2,756,040 bytes) is an Excel workbook, NOT a plain-text listing despite
the package's plain "Listing" folder name -- confirmed by inspection, not
assumed. Sheet1 holds 20,946 real records; the true header row is Excel
row 2 (row 1 is blank), giving real columns: `KP`, `E projected`,
`N projected`, `offset`, `Z`, `Uncertainty`, `Data Quality`,
`DepthSD Value`, `Decibel`, `Libra PS  MRU Pitch`, `Libra PS  MRU Roll`,
`Storage Db`, `Frequency Value`, `Current Value`. `KP` is in METRES here
(range 1399-26688). `Storage Db` holds either a raw processing database
filename (e.g. `"0100 - BOW - 0007.db"`, `"Innomar"`) OR, for 1,288 real
rows, the literal string `"Exposure"` -- this is the source's own
explicit exposure flag, independent of any numeric column (see
`marine_engine.burial.exposure_evidence`).

**Route Position List Files** (216,571 bytes) contains a real single-
LineString route shapefile (`P3111_BOW_RPL_KPEN_Roo_line.shp`, 26,233.4 m
length) and a real 5,252-point per-station shapefile
(`P3111_BOW_RPL_KPEN_Roo_points.shp`, columns `Easting`/`Northing`/`KP`)
plus a plain-text `.pts` equivalent of the points. `KP` here is in
KILOMETRES (range 0.659-26.893) -- a REAL, confirmed unit mismatch against
the DoB listing's own metres-based `KP` column (never silently assumed
equal; see `marine_engine.burial.readiness`). Both shapefiles' own `.prj`
state `ETRS89_UTM_zone_30N`, matching the DoB listing's own MEDIN
metadata CRS code (`EPSG::25830`) exactly -- confirmed independently
consistent across both real packages, not merely assumed.

Burial-reference semantics are NOT resolved by this provider module
------------------------------------------------------------------------
This module only ingests and passes through the real source values
unmodified. Whether the source's "Z" column can defensibly be read as a
literal depth-of-burial-below-seabed is a scientific-semantics question
answered in `marine_engine.burial.semantics`, not here (Section 4).
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import requests

MDE_SOURCE_PAGE_URL = "https://www.marinedataexchange.co.uk/details/TCE-48"
DOB_LISTING_URL = (
    "https://www.marinedataexchange.co.uk/pub/TCE/5-48-Depth%20of%20Burial%20Listing.zip"
)
ROUTE_POSITION_LIST_URL = (
    "https://www.marinedataexchange.co.uk/pub/TCE/5-48-Route%20Position%20List%20Files.zip"
)
REQUEST_TIMEOUT_S = 300.0

SURVEY_CONTRACTOR = "Deep BV"
SURVEY_EPOCH = "2016-09-05/2016-09-30"
DOB_LISTING_PUBLICATION_DATE = "2016-10-25"
SOURCE_CRS = "EPSG:25830"

DOB_LISTING_FILENAME = "P3111_BOW_DoB_listing_correlated_with_acoustics_R03.xlsx"
DOB_KP_COLUMN = "KP"
DOB_EASTING_COLUMN = "E projected"
DOB_NORTHING_COLUMN = "N projected"
DOB_OFFSET_COLUMN = "offset"
DOB_Z_COLUMN = "Z"
DOB_UNCERTAINTY_COLUMN = "Uncertainty"
DOB_DATA_QUALITY_COLUMN = "Data Quality"
DOB_STORAGE_DB_COLUMN = "Storage Db"
DOB_KP_UNITS = "m"

RPL_LINE_FILENAME = "P3111_BOW_RPL_KPEN_Roo_line.shp"
RPL_POINTS_FILENAME = "P3111_BOW_RPL_KPEN_Roo_points.shp"
RPL_KP_COLUMN = "KP"
RPL_KP_UNITS = "km"

# The source's own explicit exposure flag (Section 9) -- a categorical value in
# `DOB_STORAGE_DB_COLUMN`, never a numeric threshold on `Z`.
SOURCE_EXPOSURE_FLAG_VALUE = "Exposure"


@dataclass(frozen=True)
class BarrowPackageAcquisition:
    source_page_url: str
    package_url: str
    package_name: str
    package_bytes: int
    package_sha256: str
    extracted_dir: Path
    retrieved_at_utc: datetime
    already_cached: bool


def _sidecar_path(extracted_dir: Path, package_name: str) -> Path:
    return extracted_dir / f"{package_name}.acquisition.json"


def _download_package(
    raw_dir: Path,
    *,
    package_name: str,
    url: str,
    expected_files: list[str],
) -> BarrowPackageAcquisition:
    """Shared minimal-download-and-cache logic (Section 3): a single whole-file GET, cached
    thereafter via a JSON sidecar -- the same offline-rerun-safe convention as every other
    acquisition in this project."""

    extracted_dir = raw_dir / package_name
    sidecar_path = _sidecar_path(extracted_dir, package_name)

    if sidecar_path.exists() and all((extracted_dir / f).exists() for f in expected_files):
        cached = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return BarrowPackageAcquisition(
            source_page_url=MDE_SOURCE_PAGE_URL,
            package_url=url,
            package_name=package_name,
            package_bytes=cached["package_bytes"],
            package_sha256=cached["package_sha256"],
            extracted_dir=extracted_dir,
            retrieved_at_utc=datetime.fromisoformat(cached["retrieved_at_utc"]),
            already_cached=True,
        )

    extracted_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    content = response.content
    with ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(extracted_dir)

    package_sha256 = hashlib.sha256(content).hexdigest()
    retrieved_at_utc = datetime.now(UTC)
    sidecar_path.write_text(
        json.dumps(
            {
                "package_bytes": len(content),
                "package_sha256": package_sha256,
                "retrieved_at_utc": retrieved_at_utc.isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return BarrowPackageAcquisition(
        source_page_url=MDE_SOURCE_PAGE_URL,
        package_url=url,
        package_name=package_name,
        package_bytes=len(content),
        package_sha256=package_sha256,
        extracted_dir=extracted_dir,
        retrieved_at_utc=retrieved_at_utc,
        already_cached=False,
    )


def download_depth_of_burial_listing(
    raw_dir: Path, *, url: str = DOB_LISTING_URL
) -> BarrowPackageAcquisition:
    return _download_package(
        raw_dir,
        package_name="depth_of_burial_listing",
        url=url,
        expected_files=[f"Listing/{DOB_LISTING_FILENAME}"],
    )


def download_route_position_list(
    raw_dir: Path, *, url: str = ROUTE_POSITION_LIST_URL
) -> BarrowPackageAcquisition:
    return _download_package(
        raw_dir,
        package_name="route_position_list",
        url=url,
        expected_files=[
            f"RPL/{RPL_LINE_FILENAME}",
            f"RPL/{RPL_POINTS_FILENAME}",
        ],
    )


def load_dob_listing_raw(extracted_dir: Path) -> pd.DataFrame:
    """Reads the real DoB listing xlsx exactly as the source provides it -- the true header is
    Excel row 2 (row 1 is blank), and a spurious fully-blank leading column is dropped. No
    column is renamed, reinterpreted, or dropped based on assumed meaning here."""

    path = extracted_dir / "Listing" / DOB_LISTING_FILENAME
    df = pd.read_excel(path, sheet_name=0, header=1)
    blank_columns = [c for c in df.columns if str(c).startswith("Unnamed: 0")]
    return df.drop(columns=blank_columns)


def load_route_position_list(extracted_dir: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Returns `(line_gdf, points_gdf)`: the real single-LineString route and the real
    per-station KP/Easting/Northing point list, both exactly as the source provides them."""

    rpl_dir = extracted_dir / "RPL"
    line_gdf = gpd.read_file(rpl_dir / RPL_LINE_FILENAME)
    points_gdf = gpd.read_file(rpl_dir / RPL_POINTS_FILENAME)
    return line_gdf, points_gdf

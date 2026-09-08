"""2024 XOCEAN Sheringham Shoal Seabed Monitoring Survey (MAR-023 Section 9).

Crown Estate Marine Data Exchange, TCE-3974: "2024, XOCEAN, Sheringham
Shoal, Seabed Monitoring Survey" (collection Jan-Feb 2024, published June
2025). A multibeam echosounder + Pseudo SSS survey over the IAC corridors,
88 WTG boxes, and two export cable routes, explicitly targeting
integrity-relevant conditions "including scour, debris, condition of WTG
base and cable".

Minimal acquisition only (Section 9)
-------------------------------------
Only the "Interpretation Data" package is downloaded here -- a single real
shapefile of 746 point "targets", confirmed live via the same
button-click-intercept technique already established in this project
(`https://www.marinedataexchange.co.uk/pub/TCE/122_3974_Interpretation%20Data.zip`,
~138 KB). The series also offers a separate "Bathymetry data" package, but
it is NOT downloaded: the 746 real target points fall ENTIRELY within the
bounding box of the already-cached MAR-020/021 2020 canonical bed elevation
raster (`data/processed/sheringham_shoal_2020/terrain/
canonical_bed_elevation.tif`, confirmed by direct spatial check -- same
EPSG:32631 CRS, same windfarm site), so that already-downloaded raster is
reused as spatial background context for the 2024 evidence map instead of
re-acquiring a second, comparably large MBES surface. This is a real,
verified overlap, not an assumption -- and every use of it is labelled with
its own 2020 epoch, never presented as 2024 bathymetry.

Real inspection of the extracted package, before any classification logic
was written (Section 10's "do NOT assume any class exists before
inspection")
--------------------------------------------------------------------------------
One real shapefile,
`00965-REA-ENG-BATH_SheringhamShoal_Targets_Rev01` (746 Point features,
EPSG:32631). Columns: `Contact ID`, `Easting`, `Northing`, `Length`,
`Width`, `Height`, `Water Dept`, `Descriptio`, `Comment` (plus two always-
empty `field_10`/`field_11` columns). `Descriptio` distinct values, with
real counts: "Concrete block" (161), "Exposure" (152), "Cable" (151),
"Jack-up footprint" (99), "Boulder" (96), "Debris" (62), "Rock-Bag" (20),
"Jack-up footprint with sediment build up" (4), "Infrastructure" (1).

NO literal "scour" feature class exists anywhere in this real package,
despite the survey's own stated integrity-relevant purpose -- confirmed by
inspection, not assumed (see `marine_engine.scour.observed_evidence` for
how this genuinely-empty scour-class finding is classified and reported,
never silently substituted with the closest-sounding class). `Comment`
carries a real, source-stated asset/location association ("IAC/WTG OSS A",
"IAC/WTG OSS B", "ECR" = Export Cable Route, plus one
"IAC/WTG OSS B - not associated with IAC" -- preserved verbatim, never
reinterpreted).
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
import requests

MDE_SOURCE_PAGE_URL = "https://www.marinedataexchange.co.uk/details/TCE-3974"
INTERPRETATION_DATA_URL = (
    "https://www.marinedataexchange.co.uk/pub/TCE/122_3974_Interpretation%20Data.zip"
)
TARGETS_LAYER_NAME = "00965-REA-ENG-BATH_SheringhamShoal_Targets_Rev01"
REQUEST_TIMEOUT_S = 300.0

DESCRIPTION_COLUMN = "Descriptio"
SOURCE_ID_COLUMN = "Contact ID"
ASSET_ASSOCIATION_COLUMN = "Comment"
LENGTH_COLUMN = "Length"
WIDTH_COLUMN = "Width"
HEIGHT_COLUMN = "Height"
WATER_DEPTH_COLUMN = "Water Dept"

SURVEY_EPOCH = "2024-01-01/2024-02-29"


@dataclass(frozen=True)
class InterpretationDataAcquisition:
    source_page_url: str
    package_url: str
    package_bytes: int
    package_sha256: str
    extracted_dir: Path
    layer_name: str
    retrieved_at_utc: datetime
    already_cached: bool


def _sidecar_path(extracted_dir: Path) -> Path:
    return extracted_dir / "interpretation_data.acquisition.json"


def download_sheringham_shoal_2024_interpretation_data(
    raw_dir: Path, *, url: str = INTERPRETATION_DATA_URL
) -> InterpretationDataAcquisition:
    """ONE minimal live acquisition (Section 9): a single, tiny ~138 KB whole-file download,
    cached thereafter via a JSON sidecar -- the same offline-rerun-safe convention as every
    other acquisition in this project. Extracts whole into
    `raw_dir / "interpretation_data"` so the shapefile's full fileset is available to
    `geopandas` on local disk."""

    extracted_dir = raw_dir / "interpretation_data"
    sidecar_path = _sidecar_path(extracted_dir)
    expected_files = [f"{TARGETS_LAYER_NAME}{ext}" for ext in (".shp", ".dbf", ".shx")]

    if sidecar_path.exists() and all((extracted_dir / f).exists() for f in expected_files):
        cached = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return InterpretationDataAcquisition(
            source_page_url=MDE_SOURCE_PAGE_URL,
            package_url=url,
            package_bytes=cached["package_bytes"],
            package_sha256=cached["package_sha256"],
            extracted_dir=extracted_dir,
            layer_name=TARGETS_LAYER_NAME,
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
    return InterpretationDataAcquisition(
        source_page_url=MDE_SOURCE_PAGE_URL,
        package_url=url,
        package_bytes=len(content),
        package_sha256=package_sha256,
        extracted_dir=extracted_dir,
        layer_name=TARGETS_LAYER_NAME,
        retrieved_at_utc=retrieved_at_utc,
        already_cached=False,
    )


def load_targets_layer(extracted_dir: Path) -> gpd.GeoDataFrame:
    """Reads the single real `Targets` shapefile, unmodified -- every column and value is
    passed through exactly as the source provides it."""

    return gpd.read_file(extracted_dir / f"{TARGETS_LAYER_NAME}.shp")

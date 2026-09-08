"""Sheringham Shoal 2020 (Fugro GB Marine) bathymetry acquisition (MAR-020).

Real, verified-live source (never guessed)
---------------------------------------------
The Crown Estate's Marine Data Exchange (MDE), series `TCE-1986`: "2020,
Fugro, Sheringham Shoal, Seabed Monitoring Survey"
(`https://www.marinedataexchange.co.uk/details/TCE-1986/summary`, confirmed
live). Published by The Crown Estate; survey organisation Fugro GB Marine,
contracted by Equinor UK Ltd. Collection date (source-stated on the MDE
page): November 2020 - December 2020. Development: Sheringham Shoal
(offshore wind, Round 2, post-construction phase). Licence: The Crown
Estate Marine Data Exchange Terms of Use -- the same sitewide policy
already confirmed for other TCE-prefixed series in this project (MAR-017B/
017C); not re-fetched per-series since MDE's terms are a platform-wide
policy, not a per-dataset one.

Acquisition mechanics (confirmed live, same shape as MAR-017C)
-----------------------------------------------------------------
The MDE site's per-file browser lists every file individually (with real
sizes), but provides no per-file download -- only ONE combined ZIP per
package, resolved via direct browser interaction (intercepting the
"Download Dataset" button's own anchor-click targets) to
`https://www.marinedataexchange.co.uk/pub/TCE/122_1986_Bathymetry%20Data.zip`,
confirmed via HEAD request at 1,007,697,908 bytes (~1.01 GB) with
`Accept-Ranges: bytes`. That bundle also contains a 2.56 GB raw XYZ point
cloud, two ~135-156 MB MBES-difference rasters, and a 38.6 MB bathymetry-
contour shapefile -- none of which this ticket needs (Section 2: "use ONLY
the minimum processed bathymetry package required"). As in MAR-017C,
`RemoteZipReader` reads ONLY the remote ZIP's real central directory (a
few KB) to enumerate every entry, then `fetch_entry_bytes` extracts ONLY
the one target entry, `G201193_20210127_SS_MBES_1m_LAT.tif` (confirmed
real: compress_size=153,567,376, file_size=162,924,933 bytes), via a
single targeted range request -- never the other ~850 MB.

Format verified, never trusted from the filename (Section 2)
-----------------------------------------------------------------
The extracted GeoTIFF was independently inspected with `rasterio`: single
band, dtype float32, `ColorInterp.gray` (never RGB/uint8 -- confirmed
genuinely analytical, not a rendered preview, unlike MAR-017C's
`GEOTIFF.zip`). CRS `EPSG:32631` (matches this project's own PL854 working
CRS). Native pixel size confirmed exactly 1.0 x 1.0 m from the real affine
transform (the filename's "_1m_" is corroborated, not merely trusted).
Raw values are all NEGATIVE (min -24.397, max -3.189 relative to the
raster's own nodata-masked extent) -- i.e. the source is ALREADY in an
elevation-style convention (0 = the LAT vertical datum, negative = below
it), not a positive-down depth needing a sign flip; see
`SOURCE_SIGN_CONVENTION` and `terrain.canonical` for how this is carried
through explicitly rather than assumed from the "LAT" filename token
alone.
"""

import hashlib
import io
import json
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import geopandas as gpd
import requests

MDE_SERIES_ID = "TCE-1986"
MDE_SOURCE_PAGE_URL = "https://www.marinedataexchange.co.uk/details/TCE-1986/summary"
BATHYMETRY_BUNDLE_URL = (
    "https://www.marinedataexchange.co.uk/pub/TCE/122_1986_Bathymetry%20Data.zip"
)
TARGET_ENTRY_NAME = "G201193_20210127_SS_MBES_1m_LAT.tif"

DATASET_TITLE = "2020, Fugro, Sheringham Shoal, Seabed Monitoring Survey"
PUBLISHER = "The Crown Estate"
SURVEY_ORGANISATION = "Fugro GB Marine (for Equinor UK Ltd)"
SURVEY_YEAR = 2020
SURVEY_PERIOD = "November 2020 - December 2020"
LICENCE = (
    "The Crown Estate Marine Data Exchange Terms of Use: a non-exclusive, non-transferable "
    "licence (without the right to sublicense) to copy and use the information -- the same "
    "sitewide policy confirmed for other TCE-prefixed series in this project."
)

REQUEST_TIMEOUT_S = 300.0
NATIVE_PIXEL_SIZE_M = 1.0  # confirmed by direct rasterio inspection of the real GeoTIFF transform
EXPECTED_CRS = "EPSG:32631"  # confirmed by direct rasterio inspection

# Verified real; the raw band is ALREADY elevation-style (higher = shallower, 0 near LAT),
# never a positive-down depth -- see `terrain.canonical.ALREADY_ELEVATION_STYLE`.
SOURCE_SIGN_CONVENTION = "ALREADY_ELEVATION_STYLE"
SOURCE_VERTICAL_DATUM = "LAT (Lowest Astronomical Tide)"

# Real, source-documented evidence of 2018/2020 vertical-datum compatibility (MAR-021 Section 6's
# hard gate) -- quoted verbatim from the 2020 Comparison Report
# (`03_Comparison/Comparison Report/201193-R004(02) Sheringham Shoal 2020 Comparison Report.docx`,
# inside the `122_1986_Reports.zip` bundle, fetched and searched directly by extracting
# word/document.xml). NOT an assumption from the shared "LAT" filename token between the two
# surveys' own bathymetry file names.
VERTICAL_DATUM_COMPATIBILITY_EVIDENCE = (
    'Source: "201193-R004(02) Sheringham Shoal 2020 Comparison Report" (Fugro, 15 June 2021), '
    'Section 1.4 "Vertical Datum" and Section 3.1 "Data Acquisition and Processing": '
    '"All soundings shall be reduced to Lowest Astronomical Tide (LAT)." and "The 2013, 2014, '
    "2015, 2018 and 2020 surveys also utilised the United Kingdom hydrographic office (UKHO) "
    "vertical offshore reference frame (VORF) geoid model; this allowed all heights to be more "
    'accurately referenced to LAT." This explicitly states the 2018 and 2020 surveys share the '
    "same VORF-to-LAT vertical referencing methodology."
)

# Real, source-documented precision evidence (same report) -- a genuine "TVU/precision statement"
# (MAR-021 Section 12), not invented: nominal per-epoch MBES vertical accuracy, and empirically
# observed repeatability at a stable reference area spanning multiple epochs including 2018/2020.
REPORTED_MBES_VERTICAL_ACCURACY_M = 0.2
REPORTED_DATUM_SQUARE_REPEATABILITY_M = 0.15
REPORTED_ANALYST_SIGNIFICANCE_THRESHOLD_M = 0.3
PRECISION_EVIDENCE_SOURCE = (
    'Source: "201193-R004(02) Sheringham Shoal 2020 Comparison Report" (Fugro, 15 June 2021): '
    '"The final accuracy of MBES soundings collected during all four surveys was typically less '
    'than +/-0.5 m horizontally and less than +/-0.2 m vertically." "When depths were compared '
    "between the three winter surveys conducted in 2013, 2014, 2015, 2018 and 2020 differences "
    'equal to, or less than 0.15 m were observed" (at a 100 m^2 stable datum square). "changes of '
    "less than 0.3 m were not considered significant, when assessing areas of erosion or "
    "accretion\" (Fugro's own analyst-applied significance threshold, not this project's)."
)


class RemoteZipReader:
    """A minimal HTTP-Range-backed seek/read file-like object, just enough
    for `zipfile.ZipFile` to read a remote ZIP's central directory and
    extract individual entries WITHOUT downloading the whole file. Every
    `read()` call issues exactly one HTTP Range request for exactly the
    requested span -- callers that want large entries in ONE request
    (rather than `zipfile`'s own small internal read chunks) should use
    `fetch_entry_bytes` below instead of `ZipFile.read()`."""

    def __init__(self, url: str, *, timeout: float = REQUEST_TIMEOUT_S):
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()
        head = self.session.head(url, timeout=timeout)
        head.raise_for_status()
        self.length = int(head.headers["Content-Length"])
        self._pos = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.length + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        end = (
            self.length - 1
            if size is None or size < 0
            else min(self._pos + size - 1, self.length - 1)
        )
        if self._pos > end:
            return b""
        resp = self.session.get(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.content
        self._pos += len(data)
        return data


def list_remote_entries(url: str = BATHYMETRY_BUNDLE_URL) -> list[ZipInfo]:
    """Reads ONLY the remote ZIP's central directory (a handful of HTTP
    Range requests, never the bulk of the file) and returns every real
    entry's metadata."""

    reader = RemoteZipReader(url)
    with ZipFile(reader) as zf:
        return zf.infolist()


def fetch_entry_bytes(info: ZipInfo, *, url: str = BATHYMETRY_BUNDLE_URL) -> bytes:
    """Extracts ONE entry's real decompressed bytes via a SINGLE HTTP Range
    request covering exactly its compressed data span (never `zipfile`'s
    own small-chunk internal reads, and never the rest of the archive).
    `info` must come from `list_remote_entries` against the SAME `url`."""

    session = requests.Session()
    header_start = info.header_offset
    local_header = session.get(
        url,
        headers={"Range": f"bytes={header_start}-{header_start + 29}"},
        timeout=REQUEST_TIMEOUT_S,
    ).content
    if local_header[:4] != b"PK\x03\x04":
        raise ValueError(
            f"{info.filename}: unexpected local file header magic {local_header[:4]!r}"
        )
    name_len = int.from_bytes(local_header[26:28], "little")
    extra_len = int.from_bytes(local_header[28:30], "little")
    data_start = header_start + 30 + name_len + extra_len
    data_end = data_start + info.compress_size - 1

    resp = session.get(
        url, headers={"Range": f"bytes={data_start}-{data_end}"}, timeout=REQUEST_TIMEOUT_S
    )
    resp.raise_for_status()
    raw = resp.content
    if info.compress_type == ZIP_STORED:
        return raw
    return zlib.decompress(raw, -zlib.MAX_WBITS)


@dataclass(frozen=True)
class SheringhamShoalAcquisition:
    source_page_url: str
    bundle_url: str
    bundle_total_bytes: int
    target_entry_name: str
    target_entry_compressed_bytes: int
    target_entry_bytes: int
    target_entry_sha256: str
    retrieved_at_utc: datetime
    licence: str
    dataset_title: str
    survey_organisation: str
    survey_period: str
    survey_year: int
    already_cached: bool


def _acquisition_sidecar_path(local_path: Path) -> Path:
    return local_path.with_suffix(local_path.suffix + ".acquisition.json")


def download_sheringham_shoal_bathymetry(
    raw_dir: Path, *, url: str = BATHYMETRY_BUNDLE_URL
) -> SheringhamShoalAcquisition:
    """Extracts ONLY the single target entry (`TARGET_ENTRY_NAME`) from the
    real ~1.01 GB remote bundle, via range requests, never a full
    download. Persists the extracted entry's ORIGINAL bytes unmodified at
    `raw_dir / TARGET_ENTRY_NAME`, plus a small JSON sidecar recording the
    bundle/entry metadata from that one real acquisition. Skips re-download
    AND re-acquires zero network state on a cache hit -- the sidecar is
    read back from disk instead of re-issuing even a HEAD request (Section
    4: "subsequent runs must work offline", verified literally, not just
    for the large-file re-download)."""

    import hashlib

    raw_dir.mkdir(parents=True, exist_ok=True)
    local_path = raw_dir / TARGET_ENTRY_NAME
    sidecar_path = _acquisition_sidecar_path(local_path)

    if local_path.exists() and sidecar_path.exists():
        cached = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if local_path.stat().st_size == cached["target_entry_bytes"]:
            sha256 = hashlib.sha256(local_path.read_bytes()).hexdigest()
            return SheringhamShoalAcquisition(
                source_page_url=MDE_SOURCE_PAGE_URL,
                bundle_url=url,
                bundle_total_bytes=cached["bundle_total_bytes"],
                target_entry_name=TARGET_ENTRY_NAME,
                target_entry_compressed_bytes=cached["target_entry_compressed_bytes"],
                target_entry_bytes=cached["target_entry_bytes"],
                target_entry_sha256=sha256,
                retrieved_at_utc=datetime.fromisoformat(cached["retrieved_at_utc"]),
                licence=LICENCE,
                dataset_title=DATASET_TITLE,
                survey_organisation=SURVEY_ORGANISATION,
                survey_period=SURVEY_PERIOD,
                survey_year=SURVEY_YEAR,
                already_cached=True,
            )

    reader = RemoteZipReader(url)
    with ZipFile(reader) as zf:
        entries = zf.infolist()
    matches = [e for e in entries if e.filename == TARGET_ENTRY_NAME]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one entry named {TARGET_ENTRY_NAME!r} in the remote bundle, "
            f"found {len(matches)}"
        )
    target = matches[0]

    content = fetch_entry_bytes(target, url=url)
    if len(content) != target.file_size:
        raise ValueError(
            f"extracted {len(content)} bytes for {TARGET_ENTRY_NAME}, expected {target.file_size}"
        )
    local_path.write_bytes(content)
    sha256 = hashlib.sha256(local_path.read_bytes()).hexdigest()
    retrieved_at_utc = datetime.now(UTC)

    sidecar_path.write_text(
        json.dumps(
            {
                "bundle_total_bytes": reader.length,
                "target_entry_compressed_bytes": target.compress_size,
                "target_entry_bytes": target.file_size,
                "retrieved_at_utc": retrieved_at_utc.isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return SheringhamShoalAcquisition(
        source_page_url=MDE_SOURCE_PAGE_URL,
        bundle_url=url,
        bundle_total_bytes=reader.length,
        target_entry_name=TARGET_ENTRY_NAME,
        target_entry_compressed_bytes=target.compress_size,
        target_entry_bytes=target.file_size,
        target_entry_sha256=sha256,
        retrieved_at_utc=retrieved_at_utc,
        licence=LICENCE,
        dataset_title=DATASET_TITLE,
        survey_organisation=SURVEY_ORGANISATION,
        survey_period=SURVEY_PERIOD,
        survey_year=SURVEY_YEAR,
        already_cached=False,
    )


# --- MAR-021: two additional real entries from the SAME already-known 2020 bundle -------------
#
# `G201193_MBESDIFF_20v18.tif` (163,918,539 bytes) -- an independent, SOURCE-PRODUCED bathymetry
# comparison product (2020 vs 2018), confirmed real via the bundle's own central directory.
# MAR-021 acquires it as a `SOURCE_PRODUCED_COMPARISON_PRODUCT` -- it must NEVER be used to
# calculate this project's own DoD, only as an independent comparator (see
# `marine_engine.change.comparator`). Its numeric sign convention is NOT stated anywhere in the
# bundle's own sidecars (confirmed by direct inspection of its `.tif.xml`/`.tif.aux.xml`), and the
# 2020 Comparison Report's prose (fetched and searched directly) never gives an explicit pixel-
# sign statement for this specific file either -- so callers must treat its sign as UNRESOLVED,
# never assumed from the "20v18" filename token.
#
# `G201193_20210127_SS_MBESHSD_1m_LAT.tif` (34,531,969 bytes) -- a candidate uncertainty/
# precision product ("HSD"). Its own `.tif.aux.xml` sidecar (inspected directly) shows
# `RepresentationType=THEMATIC` and a classified 0-254 byte value range, NOT a direct floating-
# point metres-scale standard deviation -- real evidence that this is likely a classified/scaled
# confidence grid, not a total-vertical-uncertainty grid. Acquired anyway (cheap, ~34.5 MB) so its
# real scale/units can be verified directly once opened, per Section 12/13 -- never assumed usable
# without that verification.

MBESDIFF_20V18_ENTRY_NAME = "G201193_MBESDIFF_20v18.tif"
MBESHSD_ENTRY_NAME = "G201193_20210127_SS_MBESHSD_1m_LAT.tif"


def _download_single_bundle_entry(raw_dir: Path, entry_name: str, *, url: str) -> tuple[Path, bool]:
    """Generic single-entry range-fetch-and-cache helper, reused for both
    additional 2020 entries below (and usable for any future one) --
    returns (local_path, already_cached)."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    local_path = raw_dir / entry_name
    sidecar_path = _acquisition_sidecar_path(local_path)

    if local_path.exists() and sidecar_path.exists():
        cached = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if local_path.stat().st_size == cached["target_entry_bytes"]:
            return local_path, True

    entries = list_remote_entries(url)
    target = next(e for e in entries if e.filename == entry_name)
    content = fetch_entry_bytes(target, url=url)
    if len(content) != target.file_size:
        raise ValueError(
            f"extracted {len(content)} bytes for {entry_name}, expected {target.file_size}"
        )
    local_path.write_bytes(content)
    sidecar_path.write_text(
        json.dumps(
            {
                "target_entry_bytes": target.file_size,
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return local_path, False


def download_sheringham_shoal_2020_comparison_product(
    raw_dir: Path, *, url: str = BATHYMETRY_BUNDLE_URL
) -> tuple[Path, bool]:
    """Range-fetches ONLY `MBESDIFF_20v18.tif` from the already-known 2020
    bundle -- the independent source-produced comparison product (MAR-021
    Section 3). Returns (local_path, already_cached)."""

    return _download_single_bundle_entry(raw_dir, MBESDIFF_20V18_ENTRY_NAME, url=url)


def download_sheringham_shoal_2020_hsd_grid(
    raw_dir: Path, *, url: str = BATHYMETRY_BUNDLE_URL
) -> tuple[Path, bool]:
    """Range-fetches ONLY the MBESHSD candidate-uncertainty grid from the
    already-known 2020 bundle (MAR-021 Section 4/12). Returns (local_path,
    already_cached)."""

    return _download_single_bundle_entry(raw_dir, MBESHSD_ENTRY_NAME, url=url)


# --- MAR-022: the separate "Interpretation Shapefiles" package (confirmed real, distinct from -
# --- the "Bathymetry Data" bundle above -- the TCE-1986 series page's own "Datasets" list shows -
# --- 7 separate top-level packages, one of which is literally titled "Interpretation
# --- Shapefiles") ------------------------------------------------------------------------------
#
# Confirmed live via direct browser interaction (the same "Download Dataset" button-click
# intercept technique already established for the other bundles in this project):
# `https://www.marinedataexchange.co.uk/pub/TCE/122_1986_Interpretation%20Shapefiles.zip`,
# confirmed via HEAD request at 121,996 bytes (~119 KB) -- genuinely tiny, so this acquisition is
# a single ordinary full download (never range-fetched; there is no bandwidth reason to), then
# extracted whole so `geopandas` can read each `.shp`'s sibling `.dbf`/`.shx`/`.prj` files from
# local disk (a shapefile is a fileset, not resolvable via a single in-memory byte string).
#
# Real inspection of the extracted package (direct `geopandas.read_file` on every layer, never
# assumed from filenames alone -- MAR-022 Section 6: "Do NOT assume any class exists before
# inspection") found exactly 5 real shapefiles plus a `_metadata/medin_metadata.xml`:
#
#   G201193_20210127_SheringhamShoal_SSS_PointFeatures (146 Point features): `Descriptio` is
#     always "Jackup location" -- anthropogenic (Section 6's own named "jack-up footprints").
#   G201193_20210127_SheringhamShoal_SSS_SandWaveCrests_PollardBank (10 LineString features):
#     `Descriptio` always "Sandwave crest" -- a natural-bedform interpretation, geographically
#     concentrated in a small (~1.2 km x 0.35 km) area near the southern part of the site.
#   G201193_20210205_SheringhamShoal_SSS_Exposures (24 Point features): named cable exposures
#     (e.g. "Exposure_Cable_Route_B"), with `Freespan` Yes/No and `Max height` (m) fields --
#     anthropogenic (cable infrastructure) context.
#   G201193_20210205_SheringhamShoal_SSS_LinearFeatures (359 LineString/MultiLineString
#     features): a MIXED layer -- `Descriptio` in {"Fishing gear", "Rock dump", "Rope",
#     "Sandwave crest", "Sandwave Crest" (both capitalisations occur in the real data -- a real,
#     unnormalized source inconsistency, not a transcription error introduced here), "Exposure",
#     "Unknown_Linear_Feature"}. This is actually the much larger and more spatially extensive
#     (~6.6 km x 12.2 km) source of natural sand-wave-crest interpretation (270 rows), spanning
#     the northern/central part of the site -- the dedicated PollardBank file above covers only
#     one smaller area within the wider site.
#   G201193_20210205_SheringhamShoal_SSS_PolygonFeatures (46 Polygon features): `Descriptio` in
#     {"Wreck", "Trenching", "Possible Debris"} -- anthropogenic/uncertain context (Section 6's
#     own named "cable trenches" among others).
#
# No literal "megaripple" or "scour" feature class exists anywhere in this real package -- Section
# 7 lists these as classes to preserve IF PRESENT, and inspection (not assumption) is what
# establishes they are not part of this real dataset.
#
# All 5 layers share `EPSG:32631` (matches the bathymetry rasters, no CRS reconciliation needed).

INTERPRETATION_SHAPEFILES_URL = (
    "https://www.marinedataexchange.co.uk/pub/TCE/122_1986_Interpretation%20Shapefiles.zip"
)
INTERPRETATION_LAYER_NAMES = (
    "G201193_20210127_SheringhamShoal_SSS_PointFeatures",
    "G201193_20210127_SheringhamShoal_SSS_SandWaveCrests_PollardBank",
    "G201193_20210205_SheringhamShoal_SSS_Exposures",
    "G201193_20210205_SheringhamShoal_SSS_LinearFeatures",
    "G201193_20210205_SheringhamShoal_SSS_PolygonFeatures",
)


@dataclass(frozen=True)
class InterpretationShapefilesAcquisition:
    source_page_url: str
    package_url: str
    package_bytes: int
    package_sha256: str
    extracted_dir: Path
    layer_names: tuple[str, ...]
    retrieved_at_utc: datetime
    already_cached: bool


def _interpretation_sidecar_path(extracted_dir: Path) -> Path:
    return extracted_dir / "interpretation_shapefiles.acquisition.json"


def download_sheringham_shoal_2020_interpretation_shapefiles(
    raw_dir: Path, *, url: str = INTERPRETATION_SHAPEFILES_URL
) -> InterpretationShapefilesAcquisition:
    """MAR-022 Section 6/30: ONE minimal live acquisition (a single, tiny
    ~120 KB whole-file download, cached thereafter via a JSON sidecar --
    the same offline-rerun-safe convention as every other acquisition in
    this project) of the real "Interpretation Shapefiles" package,
    extracted whole into `raw_dir / "interpretation_shapefiles"` so every
    layer's full shapefile fileset is available to `geopandas` on local
    disk."""

    extracted_dir = raw_dir / "interpretation_shapefiles"
    sidecar_path = _interpretation_sidecar_path(extracted_dir)
    expected_files = [
        f"{name}{ext}" for name in INTERPRETATION_LAYER_NAMES for ext in (".shp", ".dbf", ".shx")
    ]

    if sidecar_path.exists() and all((extracted_dir / f).exists() for f in expected_files):
        cached = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return InterpretationShapefilesAcquisition(
            source_page_url=MDE_SOURCE_PAGE_URL,
            package_url=url,
            package_bytes=cached["package_bytes"],
            package_sha256=cached["package_sha256"],
            extracted_dir=extracted_dir,
            layer_names=INTERPRETATION_LAYER_NAMES,
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
    return InterpretationShapefilesAcquisition(
        source_page_url=MDE_SOURCE_PAGE_URL,
        package_url=url,
        package_bytes=len(content),
        package_sha256=package_sha256,
        extracted_dir=extracted_dir,
        layer_names=INTERPRETATION_LAYER_NAMES,
        retrieved_at_utc=retrieved_at_utc,
        already_cached=False,
    )


def load_interpretation_layers(extracted_dir: Path) -> dict[str, gpd.GeoDataFrame]:
    """Reads all 5 real interpretation layers. The `Exposures` layer (see
    the acquisition docstring above) has no free-text `Descriptio` column
    of its own -- every one of its rows is, by construction, a cable
    exposure, so a synthetic `Descriptio="Exposure"` column is added here
    (matching the same literal tag `LinearFeatures` uses for its own
    exposure-related rows) purely so every layer can be classified
    uniformly downstream. This is the ONLY layer-specific adjustment made
    -- every value is otherwise passed through unmodified from the real
    shapefile attribute tables."""

    layers: dict[str, gpd.GeoDataFrame] = {}
    for name in INTERPRETATION_LAYER_NAMES:
        gdf = gpd.read_file(extracted_dir / f"{name}.shp")
        if "Descriptio" not in gdf.columns:
            gdf = gdf.copy()
            gdf["Descriptio"] = "Exposure"
        layers[name] = gdf
    return layers

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

import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

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
    import json

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

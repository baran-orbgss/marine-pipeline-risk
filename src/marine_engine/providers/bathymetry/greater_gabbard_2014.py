"""Greater Gabbard 2014 (ADUS DeepOcean) bathymetry acquisition (MAR-017C).

Real, verified-live source (never guessed)
---------------------------------------------
The Crown Estate's Marine Data Exchange (MDE), series `TCE-439`: "2014,
ADUS DeepOcean, Greater Gabbard, Bathymetry Survey (Inter Array Cables,
Monopiles and Jack Up Vessel Zones)"
(`https://www.marinedataexchange.co.uk/details/TCE-439/summary`, confirmed
live 2026-09-06 -- deliberately verified as a real, distinct record by
also confirming a fabricated `TCE-999999` id returns HTTP 404). Publisher:
The Crown Estate. Organisation/contractor: ADUS DeepOcean (for SSE).
Survey period: July-August 2014 (source-stated on the MDE page). Coverage:
~146 km2, 140 turbines and inter-array cables across two sites. Licence:
The Crown Estate's own "Terms of Use" (a non-exclusive, non-transferable
licence to copy and use the information) -- NOT an Open Government
Licence, unlike HHW/IDRBNR; recorded explicitly, never assumed OGL.
Vertical/horizontal datum: not stated anywhere accessible from the source
-- recorded as unknown, never assumed.

The site's ONLY download mechanism is a single combined ZIP per data
category (never a per-file download) -- confirmed live via direct browser
interaction: selecting "Bathymetry Data" and clicking "Download Dataset"
resolves to ONE Azure Blob Storage URL,
`https://www.marinedataexchange.co.uk/pub/TCE/45_439_Bathymetry%20Data.zip`,
confirmed via HEAD request at 8,474,296,318 bytes (~8.47 GB) -- dramatically
larger than the ticket's ~770 MB two-file estimate, because it bundles far
more than `ASCBathy.zip`/`GEOTIFF.zip`: dozens of individual per-feature
ASCII/point-cloud products the on-page file browser does not fully
enumerate. Downloading the whole 8.47 GB bundle would violate Section 4's
"download only the minimum bathymetry package needed" -- instead,
`RemoteZipReader` reads ONLY the remote ZIP's central directory (a few KB,
via HTTP range requests, confirmed the server supports `Accept-Ranges:
bytes`) to enumerate every real entry, then extracts ONLY the specific
entries actually needed (see `download_greater_gabbard_ascii_subset`) via
one targeted range request per entry -- never the intervening bytes.

Archive inspection changed the acquisition plan (Section 2's own escape
hatch, exercised for real)
-----------------------------------------------------------------------------
Section 2 states "use the GeoTIFF package as primary candidate unless
archive inspection demonstrates that the ASCII package provides a
materially more complete canonical surface." Real inspection (2026-09-06)
found `GEOTIFF.zip`'s 1,192 entries are 3-band uint8 RGB colour renders
(`ColorInterp.red/green/blue`, verified via `rasterio`) -- rendered preview
images, not analytical elevation data; a uint8 image cannot carry
metre-scale depth values at any usable precision. This is a data-format
problem, not a coverage one, but the practical consequence is the same:
GeoTIFF cannot be used at all, so this module uses the ASCII package
instead -- specifically its TWO properly-gridded sub-products,
`Foundations/GRIDDED 0.25x0.25/*.txt` (144 files, confirmed on a real,
exact 0.25 m regular spacing) and `Corridors/GRIDDED 0.5x0.5/*.txt` (152
files, confirmed at 0.5 m spacing -- initially missed in this module's
OWN first inspection pass, a real reminder that "never assume" applies to
this module's own reconnaissance too, not only to the archive itself).
`Foundations/ALL ASCII` and `Corridors/ALL ASCII` remain irregular XYZ
point soundings (never used here -- gridding/interpolating them would
introduce new logic outside this ticket's scope, Section 13). Every one
of these text files uses a plain "easting northing depth" format with NO
header of any kind (not a standard ESRI ASCII Grid, despite the folders'
names) -- `load_xyz_regular_grid` below parses it directly into a proper
raster, given the CORRECT native pixel size for whichever sub-product it
is (`_native_pixel_size_for`) -- the two classes are NOT the same
resolution.

Real archive scale (recorded honestly, not assumed a priori)
-----------------------------------------------------------------
The 144 `Foundations/GRIDDED 0.25x0.25` grids total real spatial extents
of roughly 200-800 m per side (largest confirmed: `IGSUB.txt` at 777 m x
509 m) -- too small in at least one dimension to even attempt a 1000 m
tile. The 152 `Corridors/GRIDDED 0.5x0.5` grids are a different real
failure mode: several (the long export-cable routes to the substation,
e.g. `IGB04-IGSUB` at 5036 m x 2283 m) have a bounding box large enough
in BOTH dimensions to attempt a 1000 m tile, but the real valid-data
density within that box is still far short of 90% (best achieved across
all 296 real candidates, foundations and corridors combined: 53.7% at
1000 m, by `IGSUB-IGH07`) -- a corridor survey follows a narrow cable
route, not a full-width swath, so most of its own bounding rectangle is
genuinely unsurveyed. The survey was scoped entirely around individual
engineering features either way, never a continuous regional survey, a
real, structural difference from HHW/IDRBNR's swath-line-sparse-but-
nominally-contiguous coverage.
"""

import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np
import pandas as pd
import requests
from affine import Affine

MDE_SERIES_ID = "TCE-439"
MDE_SOURCE_PAGE_URL = "https://www.marinedataexchange.co.uk/details/TCE-439/summary"
BATHYMETRY_BUNDLE_URL = "https://www.marinedataexchange.co.uk/pub/TCE/45_439_Bathymetry%20Data.zip"
DATASET_TITLE = (
    "2014, ADUS DeepOcean, Greater Gabbard, Bathymetry Survey "
    "(Inter Array Cables, Monopiles and Jack Up Vessel Zones)"
)
PUBLISHER = "The Crown Estate"
SURVEY_ORGANISATION = "ADUS DeepOcean (for SSE)"
SURVEY_YEAR = 2014
SURVEY_PERIOD = "July 2014 - August 2014"
LICENCE = (
    "The Crown Estate Marine Data Exchange Terms of Use: a non-exclusive, "
    "non-transferable licence (without the right to sublicense) to copy and use the "
    "information -- NOT the UK Open Government Licence."
)

GRIDDED_ASCII_PREFIX = "ASCII/Foundations/GRIDDED 0.25x0.25/"
CORRIDOR_GRIDDED_PREFIX = "ASCII/Corridors/GRIDDED 0.5x0.5/"
MATTRESSING_PREFIX = "ASCII/Concrete mattressing/"
FOUNDATION_ALL_ASCII_PREFIX = "ASCII/Foundations/ALL ASCII/"
CORRIDOR_ALL_ASCII_PREFIX = "ASCII/Corridors/ALL ASCII/"
GEOTIFF_ENTRY = "GEOTIFF.zip"

REQUEST_TIMEOUT_S = 300.0
NATIVE_PIXEL_SIZE_M = 0.25  # confirmed by direct inspection of every GRIDDED 0.25x0.25 file
CORRIDOR_NATIVE_PIXEL_SIZE_M = (
    0.5  # confirmed by direct inspection of every Corridors/GRIDDED 0.5x0.5 file
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
class GreaterGabbardAcquisition:
    source_page_url: str
    bundle_url: str
    bundle_total_bytes: int
    retrieved_at_utc: datetime
    extracted_entry_count: int
    extracted_total_bytes: int
    licence: str
    dataset_title: str
    survey_organisation: str
    survey_period: str
    survey_year: int
    already_cached: bool


def download_greater_gabbard_ascii_subset(
    raw_dir: Path, *, url: str = BATHYMETRY_BUNDLE_URL
) -> GreaterGabbardAcquisition:
    """Section 4: extracts ONLY the three ASCII sub-products this module
    actually uses -- the 144 `Foundations/GRIDDED 0.25x0.25/*.txt` regular
    grids, the 152 `Corridors/GRIDDED 0.5x0.5/*.txt` regular grids (both
    canonical bathymetry candidates), and the 32 `Concrete mattressing/
    *.txt` files (rock/concrete protection context, Section 10) -- from
    the real 8.47 GB remote bundle, via range requests, never a full
    download. Persists each extracted entry's ORIGINAL bytes unmodified at
    its own relative path under `raw_dir`. Skips re-download entirely if
    every expected file is already present with its expected byte size
    (Section 4: "subsequent reruns must use cache")."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    entries = list_remote_entries(url)
    to_extract = [
        e
        for e in entries
        if e.filename.startswith(GRIDDED_ASCII_PREFIX)
        or e.filename.startswith(CORRIDOR_GRIDDED_PREFIX)
        or e.filename.startswith(MATTRESSING_PREFIX)
    ]

    already_cached = True
    for entry in to_extract:
        local_path = raw_dir / entry.filename
        if not local_path.exists() or local_path.stat().st_size != entry.file_size:
            already_cached = False
            break

    if not already_cached:
        for entry in to_extract:
            local_path = raw_dir / entry.filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            content = fetch_entry_bytes(entry, url=url)
            local_path.write_bytes(content)

    reader = RemoteZipReader(url)
    bundle_total_bytes = reader.length
    extracted_total_bytes = sum(e.file_size for e in to_extract)

    return GreaterGabbardAcquisition(
        source_page_url=MDE_SOURCE_PAGE_URL,
        bundle_url=url,
        bundle_total_bytes=bundle_total_bytes,
        retrieved_at_utc=datetime.now(UTC),
        extracted_entry_count=len(to_extract),
        extracted_total_bytes=extracted_total_bytes,
        licence=LICENCE,
        dataset_title=DATASET_TITLE,
        survey_organisation=SURVEY_ORGANISATION,
        survey_period=SURVEY_PERIOD,
        survey_year=SURVEY_YEAR,
        already_cached=already_cached,
    )


# --- Section 5: real-format inventory -- a non-standard, header-less XYZ text grid --------


def list_gridded_ascii_candidates(raw_dir: Path) -> list[Path]:
    """Every locally-cached `Foundations/GRIDDED 0.25x0.25/*.txt` file --
    one of the two properly-gridded (square-tile-capable) canonical
    bathymetry candidate classes in this archive (Section 5: never assume
    a single mosaic exists -- there isn't one; there are 144 small
    individual foundation grids). See also `list_gridded_corridor_
    candidates` for the second class."""

    grid_dir = raw_dir / GRIDDED_ASCII_PREFIX
    if not grid_dir.exists():
        return []
    return sorted(grid_dir.glob("*.txt"))


def list_gridded_corridor_candidates(raw_dir: Path) -> list[Path]:
    """Every locally-cached `Corridors/GRIDDED 0.5x0.5/*.txt` file -- the
    SECOND properly-gridded canonical bathymetry candidate class (152
    inter-array cable corridor grids, initially missed in this module's
    own archive inspection -- Section 5's "never assume" applies to this
    module's own reconnaissance too, not only to the source archive's
    structure). Elongated by construction (a corridor survey follows a
    cable route), so most are expected to fail a SQUARE canonical tile
    even where their along-route length exceeds 1000 m -- but this must
    be verified per-candidate by the real preflight, never assumed."""

    corridor_dir = raw_dir / CORRIDOR_GRIDDED_PREFIX
    if not corridor_dir.exists():
        return []
    return sorted(corridor_dir.glob("*.txt"))


def list_all_canonical_candidates(raw_dir: Path) -> list[tuple[Path, float]]:
    """Every real, properly-gridded canonical bathymetry candidate in the
    archive -- foundations (0.25 m native) and corridors (0.5 m native)
    combined, each paired with its own correct native pixel size (never a
    single assumed resolution across both classes)."""

    foundations = [(p, NATIVE_PIXEL_SIZE_M) for p in list_gridded_ascii_candidates(raw_dir)]
    corridors = [
        (p, CORRIDOR_NATIVE_PIXEL_SIZE_M) for p in list_gridded_corridor_candidates(raw_dir)
    ]
    return foundations + corridors


def list_mattressing_files(raw_dir: Path) -> list[Path]:
    """Every locally-cached `Concrete mattressing/*.txt` file -- rock/
    concrete protection context (Section 10), never a canonical bathymetry
    candidate (these cover already-armoured seabed by definition)."""

    mattressing_dir = raw_dir / MATTRESSING_PREFIX
    if not mattressing_dir.exists():
        return []
    return sorted(mattressing_dir.glob("*.txt"))


def load_xyz_regular_grid(
    path: Path, *, pixel_size_m: float = NATIVE_PIXEL_SIZE_M
) -> tuple[np.ndarray, np.ndarray, Affine, float]:
    """Parses this archive's real, non-standard "easting northing depth"
    text format (NO header of any kind, despite the source folder being
    named "GRIDDED 0.25x0.25") into a proper raster: an elevation array
    (float64, NaN outside the point set), a boolean valid mask, an affine
    transform, and the confirmed native pixel size in metres. A real
    coverage GAP within the file's own bounding rectangle (this archive's
    real per-tile valid fraction is well under 100%, confirmed by direct
    inspection) becomes `valid=False` at that cell -- never interpolated
    or filled here; gap-handling remains the reusable engine's own job
    (`swm.fill_small_gaps`), exactly as for every other analog."""

    text = path.read_text(encoding="utf-8", errors="replace")
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows:
        empty = np.zeros((0, 0), dtype=np.float64)
        return empty, empty.astype(bool), Affine.identity(), pixel_size_m
    arr = np.array(rows, dtype=np.float64)
    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]

    x_min, y_max = x.min(), y.max()
    col = np.round((x - x_min) / pixel_size_m).astype(np.int64)
    row = np.round((y_max - y) / pixel_size_m).astype(np.int64)
    n_cols = int(col.max()) + 1
    n_rows = int(row.max()) + 1

    elevation = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    elevation[row, col] = z
    valid = ~np.isnan(elevation)
    transform = Affine.translation(x_min, y_max) * Affine.scale(pixel_size_m, -pixel_size_m)
    return elevation, valid, transform, pixel_size_m


def _classify_ascii_format(filename: str) -> str:
    if filename.startswith(GRIDDED_ASCII_PREFIX):
        return "XYZ_TEXT_REGULAR_GRID"
    if filename.startswith(CORRIDOR_GRIDDED_PREFIX):
        return "XYZ_TEXT_REGULAR_GRID"
    if filename.startswith(MATTRESSING_PREFIX):
        return "XYZ_TEXT_REGULAR_GRID"
    if filename.startswith(FOUNDATION_ALL_ASCII_PREFIX) or filename.startswith(
        CORRIDOR_ALL_ASCII_PREFIX
    ):
        return "XYZ_TEXT_IRREGULAR_POINT_CLOUD"
    if filename.endswith(".pts"):
        return "XYZ_TEXT_IRREGULAR_POINT_CLOUD"
    return "UNKNOWN"


def _native_pixel_size_for(filename: str) -> float:
    """The correct native pixel size for a REGULAR-grid entry -- corridors
    (0.5 m) and foundations/mattressing (0.25 m) are NOT the same
    resolution; using the wrong one would silently misplace every parsed
    point (Section 6: never guess a scale)."""

    return (
        CORRIDOR_NATIVE_PIXEL_SIZE_M
        if filename.startswith(CORRIDOR_GRIDDED_PREFIX)
        else NATIVE_PIXEL_SIZE_M
    )


def build_source_file_inventory(raw_dir: Path) -> pd.DataFrame:
    """Section 5: every credible bathymetric raster/point entry in the
    real remote archive (enumerated via `list_remote_entries` -- cheap,
    central-directory-only, never a full download), classified by real
    format. Full technical metadata (width/height/native pixel size/valid
    fraction/bounds) is populated ONLY for the 328 entries this module
    actually extracts and can open (144 `Foundations/GRIDDED 0.25x0.25` +
    152 `Corridors/GRIDDED 0.5x0.5` grids + 32 `Concrete mattressing`
    files) -- never fabricated for an entry never opened.
    `GEOTIFF.zip` is recorded with its real, confirmed format
    (`RGB_PREVIEW_IMAGE_NOT_ELEVATION_DATA`) rather than silently omitted,
    so the reason it is unused is itself part of the permanent record."""

    entries = list_remote_entries()
    rows = []
    for entry in entries:
        filename = entry.filename
        if filename == GEOTIFF_ENTRY:
            rows.append(
                {
                    "filename": filename,
                    "format": "RGB_PREVIEW_IMAGE_NOT_ELEVATION_DATA",
                    "byte_size": entry.file_size,
                    "width": None,
                    "height": None,
                    "native_pixel_size_m": None,
                    "nodata": None,
                    "dtype": "uint8 (3-band RGB)",
                    "bounds": None,
                    "valid_fraction": None,
                    "vertical_datum": None,
                    "acquisition_year": SURVEY_YEAR,
                    "extracted_locally": False,
                }
            )
            continue
        if not filename.startswith("ASCII/"):
            continue

        local_path = raw_dir / filename
        is_extracted = local_path.exists()
        width = height = native_pixel_size_m = valid_fraction = bounds = None
        if (
            is_extracted
            and filename.endswith(".txt")
            and _classify_ascii_format(filename) == "XYZ_TEXT_REGULAR_GRID"
        ):
            elevation, valid, transform, native_pixel_size_m = load_xyz_regular_grid(
                local_path, pixel_size_m=_native_pixel_size_for(filename)
            )
            if valid.size > 0:
                height, width = elevation.shape
                valid_fraction = float(valid.mean())
                left, top = transform * (0, 0)
                right, bottom = transform * (width, height)
                bounds = (left, bottom, right, top)

        rows.append(
            {
                "filename": filename,
                "format": _classify_ascii_format(filename),
                "byte_size": entry.file_size,
                "width": width,
                "height": height,
                "native_pixel_size_m": native_pixel_size_m,
                "nodata": None,  # missing (x,y) combinations, never a literal sentinel value
                "dtype": "float64 (parsed from text)" if is_extracted else None,
                "bounds": bounds,
                "valid_fraction": valid_fraction,
                # Never discoverable from this archive -- explicitly recorded as such.
                "vertical_datum": None,
                "acquisition_year": SURVEY_YEAR,
                "extracted_locally": is_extracted,
            }
        )
    return pd.DataFrame(rows)

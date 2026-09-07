"""Sheringham Shoal 2018 (Fugro GB Marine) bathymetry acquisition (MAR-021).

Real, verified-live source (never guessed)
---------------------------------------------
The Crown Estate's Marine Data Exchange (MDE), series `TCE-1975`: "2018,
Fugro, Sheringham Shoal, Seabed Monitoring Survey"
(`https://www.marinedataexchange.co.uk/details/TCE-1975/summary`, confirmed
live). Collection date (source-stated): October 2018 - November 2018.
Same publisher/survey organisation/licence family as the 2020 survey
(MAR-020) -- Fugro GB Marine, for Equinor UK Ltd, The Crown Estate MDE
Terms of Use.

Acquisition mechanics (confirmed live) -- genuinely different from 2020
-------------------------------------------------------------------------
Unlike the 2020 package (a single ready-made analytical GeoTIFF), the 2018
"Bathymetry Data" package's own file browser shows only FOLDERS: `Applanix
POS MV` (raw positioning binaries), `Qinsy DB` (raw daily survey-project
databases), `XYZ` (processed bathymetry export), `_metadata`. The
`Applanix POS MV`/`Qinsy DB` folders are raw acquisition data, never
analytical bathymetry -- excluded entirely. The combined "Download
Dataset" bundle for this package
(`https://www.marinedataexchange.co.uk/pub/TCE/122_1975_Bathymetry%20Data.zip`,
confirmed via its real ZIP central directory) is genuinely enormous
(357.74 GB, 4131 entries -- almost entirely raw Applanix logs), so this
module NEVER opens that bundle; `RemoteZipReader`/`fetch_entry_bytes` are
reused directly from `sheringham_shoal_2020` (a purely mechanical HTTP-
Range-over-ZIP utility with zero Sheringham-specific content -- reusing it
avoids a third copy of the same class, following `sheringham_shoal_2020`'s
own precedent of reusing `greater_gabbard_2014`'s pattern) to range-fetch
ONLY the 3 real XYZ export parts
(`XYZ/G181231_20190125_SheringhamShoal_MBE_Total_WGS84_UTM31N_LAT_1m_{1,2,3}of3.xyz`,
confirmed real sizes ~1.01 GB + ~1.02 GB + ~187 MB uncompressed, ~137 MB +
~129 MB + ~22 MB compressed) plus the tiny `_metadata/medin_metadata.xml`
-- never the ~357 GB whole bundle, never the Applanix/Qinsy folders.

Format verified, never trusted from the filename (never assumed gridded
just because "_1m_" appears in the name)
-------------------------------------------------------------------------
A real partial-DEFLATE-stream peek (decompressing only the first 32 KB of
part 1's compressed span, not the whole ~137 MB entry) confirmed: plain
ASCII CSV rows `easting,northing,elevation` (e.g.
`375640.50,5868402.50,-3.84`), each easting/northing landing exactly on a
half-integer coordinate -- i.e. already the CENTRE of a native 1 m grid
cell, so building the canonical raster from these rows is a direct
placement (`round(x-0.5)`, `round(y-0.5)`), never a scattered-point
interpolation. Values are all negative, matching the SAME already-
elevation-style convention independently found for 2020 (never assumed
from the shared "_LAT_" filename token alone): the 2018 Results Report
bundle's own `Range and Interpolation.txt` sidecar states the real
depth range as "-3.69m and -24.50m" and explicitly "No interpolation was
undertaken" -- both corroborated directly, not assumed.

Vertical datum -- real source evidence for epoch compatibility (MAR-021
Section 6's hard gate)
-------------------------------------------------------------------------
The 2020 Comparison Report (`201193-R004(02) Sheringham Shoal 2020
Comparison Report.docx`, fetched and searched directly -- see
`marine_engine.change.epoch_compatibility` for how this is used)
explicitly states: "All soundings shall be reduced to Lowest Astronomical
Tide (LAT)" and "The 2013, 2014, 2015, 2018 and 2020 surveys also utilised
the United Kingdom hydrographic office (UKHO) vertical offshore reference
frame (VORF) geoid model; this allowed all heights to be more accurately
referenced to LAT" -- i.e. the SAME source explicitly states 2018 and 2020
share the same VORF-to-LAT methodology. This is a genuine, source-stated
transformation/equivalence, not an assumption from the shared "LAT"
filename token (the 2018 series' own `medin_metadata.xml` does NOT
separately restate the vertical datum in a way a naive keyword search
would catch -- its only "LAT" occurrences are geographic-latitude bounding-
box tags, a real false-positive trap this module's caller must not fall
into either).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from marine_engine.providers.bathymetry.sheringham_shoal_2020 import (
    fetch_entry_bytes,
    list_remote_entries,
)

MDE_SERIES_ID = "TCE-1975"
MDE_SOURCE_PAGE_URL = "https://www.marinedataexchange.co.uk/details/TCE-1975/summary"
BATHYMETRY_BUNDLE_URL = (
    "https://www.marinedataexchange.co.uk/pub/TCE/122_1975_Bathymetry%20Data.zip"
)

XYZ_ENTRY_NAMES = (
    "XYZ/G181231_20190125_SheringhamShoal_MBE_Total_WGS84_UTM31N_LAT_1m_1of3.xyz",
    "XYZ/G181231_20190125_SheringhamShoal_MBE_Total_WGS84_UTM31N_LAT_1m_2of3.xyz",
    "XYZ/G181231_20190125_SheringhamShoal_MBE_Total_WGS84_UTM31N_LAT_1m_3of3.xyz",
)
METADATA_ENTRY_NAME = "_metadata/medin_metadata.xml"

DATASET_TITLE = "2018, Fugro, Sheringham Shoal, Seabed Monitoring Survey"
PUBLISHER = "The Crown Estate"
SURVEY_ORGANISATION = "Fugro GB Marine (for Equinor UK Ltd)"
SURVEY_YEAR = 2018
SURVEY_PERIOD = "October 2018 - November 2018"
LICENCE = (
    "The Crown Estate Marine Data Exchange Terms of Use: a non-exclusive, non-transferable "
    "licence (without the right to sublicense) to copy and use the information -- the same "
    "sitewide policy confirmed for other TCE-prefixed series in this project."
)

NATIVE_PIXEL_SIZE_M = 1.0  # confirmed: XYZ rows fall exactly on half-integer (cell-centre) coords
EXPECTED_CRS = "EPSG:32631"  # WGS84 / UTM 31N -- same zone as the 2018 XYZ filename states and 2020

# Verified real (see module docstring): raw values are already elevation-style, negative,
# LAT-referenced -- never a positive-down depth needing a sign flip.
SOURCE_SIGN_CONVENTION = "ALREADY_ELEVATION_STYLE"
SOURCE_VERTICAL_DATUM = "LAT (Lowest Astronomical Tide)"

# The real, source-documented vertical-datum-compatibility and precision evidence (quoted from
# the 2020 Comparison Report -- see the module docstring above) lives as constants on
# `sheringham_shoal_2020` (`VERTICAL_DATUM_COMPATIBILITY_EVIDENCE`, `PRECISION_EVIDENCE_SOURCE`,
# etc.), not duplicated here, since that report is physically part of the 2020 series' own
# Reports package.


@dataclass(frozen=True)
class SheringhamShoal2018Acquisition:
    source_page_url: str
    bundle_url: str
    xyz_entry_names: tuple[str, ...]
    xyz_entry_bytes: tuple[int, ...]
    xyz_entry_sha256: tuple[str, ...]
    metadata_entry_name: str
    metadata_entry_sha256: str
    retrieved_at_utc: datetime
    licence: str
    dataset_title: str
    survey_organisation: str
    survey_period: str
    survey_year: int
    already_cached: bool


def _acquisition_sidecar_path(raw_dir: Path) -> Path:
    return raw_dir / "sheringham_shoal_2018_xyz.acquisition.json"


def download_sheringham_shoal_2018_bathymetry(
    raw_dir: Path, *, url: str = BATHYMETRY_BUNDLE_URL
) -> SheringhamShoal2018Acquisition:
    """Extracts ONLY the 3 real XYZ export parts and the metadata sidecar
    from the real ~357.74 GB remote bundle, via targeted range requests --
    never the whole bundle, never the raw Applanix/Qinsy folders. Persists
    each extracted entry's ORIGINAL bytes unmodified under `raw_dir`, plus
    a JSON sidecar recording acquisition metadata so a cache hit performs
    zero network I/O (mirrors the MAR-020 offline-rerun fix)."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    local_paths = [raw_dir / Path(name).name for name in XYZ_ENTRY_NAMES]
    metadata_local_path = raw_dir / "sheringham_shoal_2018_medin_metadata.xml"
    sidecar_path = _acquisition_sidecar_path(raw_dir)

    if (
        sidecar_path.exists()
        and all(p.exists() for p in local_paths)
        and metadata_local_path.exists()
    ):
        cached = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sizes_match = all(
            local_paths[i].stat().st_size == cached["xyz_entry_bytes"][i] for i in range(3)
        )
        if sizes_match:
            return SheringhamShoal2018Acquisition(
                source_page_url=MDE_SOURCE_PAGE_URL,
                bundle_url=url,
                xyz_entry_names=XYZ_ENTRY_NAMES,
                xyz_entry_bytes=tuple(cached["xyz_entry_bytes"]),
                xyz_entry_sha256=tuple(
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in local_paths
                ),
                metadata_entry_name=METADATA_ENTRY_NAME,
                metadata_entry_sha256=hashlib.sha256(metadata_local_path.read_bytes()).hexdigest(),
                retrieved_at_utc=datetime.fromisoformat(cached["retrieved_at_utc"]),
                licence=LICENCE,
                dataset_title=DATASET_TITLE,
                survey_organisation=SURVEY_ORGANISATION,
                survey_period=SURVEY_PERIOD,
                survey_year=SURVEY_YEAR,
                already_cached=True,
            )

    entries = list_remote_entries(url)
    entries_by_name = {e.filename: e for e in entries}

    xyz_entry_bytes: list[int] = []
    for name, local_path in zip(XYZ_ENTRY_NAMES, local_paths, strict=True):
        target = entries_by_name[name]
        content = fetch_entry_bytes(target, url=url)
        if len(content) != target.file_size:
            raise ValueError(
                f"extracted {len(content)} bytes for {name}, expected {target.file_size}"
            )
        local_path.write_bytes(content)
        xyz_entry_bytes.append(target.file_size)

    meta_target = entries_by_name[METADATA_ENTRY_NAME]
    metadata_local_path.write_bytes(fetch_entry_bytes(meta_target, url=url))

    retrieved_at_utc = datetime.now(UTC)
    sidecar_path.write_text(
        json.dumps(
            {"xyz_entry_bytes": xyz_entry_bytes, "retrieved_at_utc": retrieved_at_utc.isoformat()},
            indent=2,
        ),
        encoding="utf-8",
    )

    return SheringhamShoal2018Acquisition(
        source_page_url=MDE_SOURCE_PAGE_URL,
        bundle_url=url,
        xyz_entry_names=XYZ_ENTRY_NAMES,
        xyz_entry_bytes=tuple(xyz_entry_bytes),
        xyz_entry_sha256=tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in local_paths),
        metadata_entry_name=METADATA_ENTRY_NAME,
        metadata_entry_sha256=hashlib.sha256(metadata_local_path.read_bytes()).hexdigest(),
        retrieved_at_utc=retrieved_at_utc,
        licence=LICENCE,
        dataset_title=DATASET_TITLE,
        survey_organisation=SURVEY_ORGANISATION,
        survey_period=SURVEY_PERIOD,
        survey_year=SURVEY_YEAR,
        already_cached=False,
    )


def rasterize_xyz_points(
    xyz_paths: list[Path], *, cell_size_m: float = NATIVE_PIXEL_SIZE_M
) -> tuple[np.ndarray, np.ndarray, object]:
    """Builds a canonical 1 m raster directly from the real XYZ rows --
    DIRECT grid placement (rows already fall on half-integer cell-centre
    coordinates; the source's own documentation states no interpolation
    was performed, so this function performs none either), never a
    scattered-point interpolation/kriging.

    Returns (elevation, valid_mask, transform) where `elevation` is the
    ALREADY elevation-style (unconverted) raw value grid -- callers must
    still route it through `terrain.canonical.build_canonical_bed_elevation`
    for the canonical `bed_elevation_m` conversion, exactly once, exactly
    like MAR-020.
    """

    import rasterio

    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for path in xyz_paths:
        data = np.loadtxt(path, delimiter=",", dtype=np.float64)
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        rows.append((x, y, z))
        min_x, max_x = min(min_x, x.min()), max(max_x, x.max())
        min_y, max_y = min(min_y, y.min()), max(max_y, y.max())

    # Cell centres are at half-integer coordinates -> cell (col, row) covers
    # [floor(x), floor(x)+1) with centre floor(x)+0.5; the grid origin (top-left corner) is the
    # floor of the minimum coordinate, and the raster's north-up convention means row 0 is the
    # MAXIMUM northing.
    origin_x = np.floor(min_x)
    origin_y = np.ceil(max_y)
    width = int(np.floor(max_x) - origin_x) + 1
    height = int(origin_y - np.floor(min_y)) + 1

    elevation = np.full((height, width), np.nan, dtype=np.float64)
    for x, y, z in rows:
        col = np.round(x - origin_x - 0.5).astype(np.int64)
        row = np.round(origin_y - y - 0.5).astype(np.int64)
        in_bounds = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        elevation[row[in_bounds], col[in_bounds]] = z[in_bounds]

    valid_mask = np.isfinite(elevation)
    transform = rasterio.Affine(cell_size_m, 0.0, origin_x, 0.0, -cell_size_m, origin_y)
    return elevation, valid_mask, transform

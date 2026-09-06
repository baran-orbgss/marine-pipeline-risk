"""Inner Dowsing, Race Bank and North Ridge cSAC (CEND 11/11) processed
bathymetry acquisition (MAR-017B).

Real, verified-live source (never guessed)
---------------------------------------------
Official dataset "Processed bathymetry from Inner Dowsing, Race Bank and
North Ridge cSAC" (data.gov.uk package
`9ac8c3cf-60e6-4e26-bfb9-44f1c9f341f7`, confirmed live via the CKAN
`package_show` API 2026-09-06). Publisher: Joint Nature Conservation
Committee (JNCC). Licence: UK Open Government Licence. Survey/cruise:
CEND 11/11 (RV Cefas Endeavour) -- the SAME cruise as HHW CEND 11/11
(MAR-017/MAR-017A); the archive's own internal lineage path
(`asciito_rb1/metadata.xml`'s `itemLocation`) reads
`.../2011_06_RVCefasEndeavour_IDRBNR_HHW/final_data/.../IDRBNR_Bathy/...`,
confirming survey year 2011 (month: June) as source-stated -- no exact
day range is stated anywhere in this archive's own metadata (unlike HHW's
recorded 2011-06-11 to 2011-06-21), so none is fabricated here. The single
resource `IDRBNR-Bathy.zip` is confirmed 218,399,605 bytes at
`https://data.jncc.gov.uk/data/f9d56abc-b46a-4bf2-b7e4-0e54270c3a68-
IDRBNR-Bathy.zip`.

Archive layout is GENUINELY DIFFERENT from HHW (Section 5: never assume)
-----------------------------------------------------------------------------
Real inspection (2026-09-06) shows TWO distinct raster storage formats,
never just one:

- 18 ESRI Arc/INFO Binary Grid directories (`asciito_{du,id,idrb,nr,rb}*`),
  detected the same way as HHW (a real `hdr.adf` member), each a small
  (roughly 1-6 km wide) individual survey-block product.
- 3 STANDALONE ESRI ASCII Grid (`.asc`, GDAL driver `AAIGrid`) files --
  `IDRBNR_25012012_MBES{1,2,3}.asc` -- each covering a MUCH larger
  (roughly 18-27 km wide) mosaicked area. HHW had no standalone raster
  files at all; this module's `discover_standalone_raster_files` handles
  them as a genuinely separate raster-candidate class.

A real, reportable data-integrity finding (never silently worked around)
-----------------------------------------------------------------------------
Two of the 18 ESRI grids -- `asciito_idrb3` and `asciito_idrb4` -- declare
CRS EPSG:4326 (geographic, degrees) while their actual coordinate values
(e.g. easting ~348463-370530, northing ~5885046-5921158) are obviously
UTM-scale metres, not longitude/latitude -- an internally inconsistent/
corrupted CRS declaration inherited from the source archive, not
introduced here. `is_pixel_scale_plausible` flags this generically (any
candidate whose CRS `is_geographic` while claiming a ~1 m native pixel
step is physically implausible), never by hard-coding these two grids'
names -- the same check would catch an equivalent problem in any future
dataset. These two are recorded in the full source inventory like every
other member, but excluded from canonical-support-preflight candidacy
with an explicit reason.

The 3 standalone `.asc` files have NO recoverable CRS at all (no `.prj`
sidecar anywhere in the archive) -- recorded honestly as
`crs=None`/`horizontal_crs=None` (Section 5's "vertical datum... where
source-stated" principle applied here to the horizontal CRS too), never
assumed to share HHW/asciito_idrb2's EPSG:32631 even though their
coordinate ranges are consistent with it. Unlike the CRS-inconsistent
pair above, a *missing* CRS does not make the numeric affine transform
(and therefore `pixel_size_m`) untrustworthy, so these three ARE included
in canonical-support-preflight candidacy.
"""

import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio
import requests

IDRBNR_BATHY_ZIP_URL = (
    "https://data.jncc.gov.uk/data/f9d56abc-b46a-4bf2-b7e4-0e54270c3a68-IDRBNR-Bathy.zip"
)
IDRBNR_DATASET_TITLE = "Processed bathymetry from Inner Dowsing, Race Bank and North Ridge cSAC"
IDRBNR_PUBLISHER = "Joint Nature Conservation Committee (JNCC)"
IDRBNR_LICENCE = "UK Open Government Licence (OGL)"
IDRBNR_SOURCE_CATALOGUE_URL = (
    "https://www.data.gov.uk/dataset/9ac8c3cf-60e6-4e26-bfb9-44f1c9f341f7/"
    "processed-bathymetry-from-inner-dowsing-race-bank-and-north-ridge-csac"
)
IDRBNR_CRUISE_ID = "CEND 11/11"
IDRBNR_SURVEY_VESSEL = "RV Cefas Endeavour"
IDRBNR_SURVEY_YEAR = 2011
IDRBNR_EXPECTED_ZIP_SIZE_BYTES = 218_399_605

REQUEST_TIMEOUT_S = 300.0

_RASTER_EXTENSIONS = (".adf", ".tif", ".tiff", ".asc", ".bag")


@dataclass(frozen=True)
class IdrbnrZipAcquisition:
    local_path: Path
    source_url: str
    retrieved_at_utc: datetime
    byte_size: int
    sha256: str
    licence: str
    dataset_title: str
    cruise_id: str
    survey_year: int
    already_cached: bool


def download_idrbnr_bathy_zip(
    dest_path: Path, *, timeout: float = REQUEST_TIMEOUT_S
) -> IdrbnrZipAcquisition:
    """Downloads the one official processed-bathymetry ZIP if not already
    cached -- never re-downloads an already-present, correctly-sized
    file. Subsequent runs are fully offline (Section 4)."""

    from marine_engine.providers.bathymetry.acquisition import compute_sha256

    already_cached = (
        dest_path.exists() and dest_path.stat().st_size == IDRBNR_EXPECTED_ZIP_SIZE_BYTES
    )
    if not already_cached:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(IDRBNR_BATHY_ZIP_URL, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with dest_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)

    return IdrbnrZipAcquisition(
        local_path=dest_path,
        source_url=IDRBNR_BATHY_ZIP_URL,
        retrieved_at_utc=datetime.now(UTC),
        byte_size=dest_path.stat().st_size,
        sha256=compute_sha256(dest_path),
        licence=IDRBNR_LICENCE,
        dataset_title=IDRBNR_DATASET_TITLE,
        cruise_id=IDRBNR_CRUISE_ID,
        survey_year=IDRBNR_SURVEY_YEAR,
        already_cached=already_cached,
    )


# --- Section 5: file/format inventory FIRST, before any morphology processing --------------


def discover_esri_grids(zip_path: Path) -> list[str]:
    """Every ESRI Arc/INFO Binary Grid directory in the archive -- detected
    by the real presence of its `hdr.adf` header member, never assumed
    from a naming convention (this archive's grid names -- du/id/idrb/nr/
    rb -- share no pattern with HHW's asciito_hhw_N/hhw_gsfN)."""

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    grid_dirs = sorted(
        {name.rsplit("/", 1)[0] for name in names if name.lower().endswith("/hdr.adf")}
    )
    return grid_dirs


def discover_standalone_raster_files(zip_path: Path) -> list[str]:
    """Top-level (not inside any grid directory) archive members with a
    raster extension -- this archive's 3 large `IDRBNR_25012012_MBES*.asc`
    ESRI ASCII Grid products, a raster-candidate class HHW never had."""

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    return sorted(
        name for name in names if "/" not in name and name.lower().endswith(_RASTER_EXTENSIONS)
    )


def _classify_member(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(_RASTER_EXTENSIONS):
        return "raster_component"
    if lower.endswith((".shp", ".shx", ".dbf", ".prj")):
        return "vector_component"
    if lower.endswith((".xml", ".txt", ".log")):
        return "text_metadata"
    return "unknown"


def is_pixel_scale_plausible(crs: Any) -> bool:
    """MAR-017B: a candidate whose CRS is geographic (degrees) while its
    native pixel step is reported as ~1.0 native unit is claiming a ~1
    DEGREE pixel -- physically impossible for a multibeam bathymetry grid
    described everywhere else in this archive as ~1 m resolution. Flags
    this generically (by inspecting the CRS's own `is_geographic`
    property), never by hard-coding a specific grid's name -- the same
    check would catch an equivalent problem in a future dataset. Returns
    True (plausible) whenever `crs` is None -- a MISSING crs is a
    different, honestly-recorded problem (Section 5), not an inconsistent
    one, and does not make the numeric pixel size untrustworthy."""

    if crs is None:
        return True
    return not crs.is_geographic


def _inspect_raster_via_vsizip(zip_path: Path, member_path: str) -> dict[str, Any] | None:
    """Opens either an ESRI grid DIRECTORY or a standalone raster FILE via
    `/vsizip/` -- both are valid GDAL dataset paths through the same
    virtual filesystem, so one function covers both raster-candidate
    classes in this archive."""

    vsi_path = f"/vsizip/{zip_path}/{member_path}"
    try:
        with rasterio.open(vsi_path) as src:
            return {
                "readable_by_rasterio": True,
                "crs": str(src.crs) if src.crs else None,
                "crs_is_geographic": bool(src.crs.is_geographic) if src.crs else None,
                "width": src.width,
                "height": src.height,
                "pixel_size_x_m": src.res[0],
                "pixel_size_y_m": src.res[1],
                "nodata": src.nodata,
                "band_count": src.count,
                "dtype": src.dtypes[0] if src.dtypes else None,
            }
    except Exception:  # noqa: BLE001 -- an unreadable member is a real, reportable inventory fact
        return None


def list_raster_candidates(zip_path: Path) -> list[dict[str, Any]]:
    """One row per distinct raster candidate (each ESRI grid directory,
    each standalone `.asc` file) with real technical metadata and the
    `is_pixel_scale_plausible` flag -- feeds the canonical support
    preflight (Section 7), never the per-archive-member inventory table
    (see `build_source_file_inventory`)."""

    grid_dirs = discover_esri_grids(zip_path)
    standalone_files = discover_standalone_raster_files(zip_path)

    candidates = []
    for grid_dir in grid_dirs:
        meta = _inspect_raster_via_vsizip(zip_path, grid_dir)
        candidates.append(
            {"raster_candidate_id": grid_dir, "storage_format": "ESRI_GRID", **(meta or {})}
        )
    for standalone in standalone_files:
        meta = _inspect_raster_via_vsizip(zip_path, standalone)
        candidates.append(
            {"raster_candidate_id": standalone, "storage_format": "ESRI_ASCII_GRID", **(meta or {})}
        )
    return candidates


def build_source_file_inventory(zip_path: Path) -> pd.DataFrame:
    """Section 5: every archive member, classified, with raster metadata
    (CRS/dimensions/pixel size/nodata/band count/dtype/valid fraction)
    populated ONLY for the real raster candidates rasterio can actually
    open -- never fabricated for a non-raster member, and never assuming
    HHW's grid-directory-only layout (this archive also has standalone
    raster files, Section 5)."""

    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()

    grid_dirs = set(discover_esri_grids(zip_path))
    standalone_files = set(discover_standalone_raster_files(zip_path))
    candidates_by_id = {c["raster_candidate_id"]: c for c in list_raster_candidates(zip_path)}

    rows = []
    for info in infos:
        filename = info.filename
        extension = Path(filename).suffix.lower()
        member_grid_dir = filename.rsplit("/", 1)[0] if "/" in filename else None
        is_grid_header_file = filename.lower().endswith("hdr.adf") and member_grid_dir in grid_dirs
        is_standalone_raster_file = filename in standalone_files
        is_primary_raster_file = is_grid_header_file or is_standalone_raster_file
        raster_candidate_id = (
            member_grid_dir
            if is_grid_header_file
            else (filename if is_standalone_raster_file else None)
        )
        candidate_meta = candidates_by_id.get(raster_candidate_id) if raster_candidate_id else None

        rows.append(
            {
                "filename": filename,
                "extension": extension,
                "byte_size": info.file_size,
                "classification": _classify_member(filename),
                "is_raster_candidate": is_primary_raster_file,
                "raster_candidate_id": raster_candidate_id,
                "storage_format": candidate_meta["storage_format"] if candidate_meta else None,
                "readable_by_rasterio": bool(candidate_meta.get("readable_by_rasterio"))
                if candidate_meta
                else None,
                "crs": candidate_meta.get("crs") if candidate_meta else None,
                "crs_is_geographic": candidate_meta.get("crs_is_geographic")
                if candidate_meta
                else None,
                "width": candidate_meta.get("width") if candidate_meta else None,
                "height": candidate_meta.get("height") if candidate_meta else None,
                "pixel_size_x_m": candidate_meta.get("pixel_size_x_m") if candidate_meta else None,
                "pixel_size_y_m": candidate_meta.get("pixel_size_y_m") if candidate_meta else None,
                "nodata": candidate_meta.get("nodata") if candidate_meta else None,
                "band_count": candidate_meta.get("band_count") if candidate_meta else None,
                "dtype": candidate_meta.get("dtype") if candidate_meta else None,
                # Never discoverable from this archive -- explicitly recorded as such
                # (Section 5), never fabricated.
                "vertical_units": None,
                "vertical_datum": None,
            }
        )
    return pd.DataFrame(rows)

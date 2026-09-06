"""Haisborough, Hammond and Winterton cSAC (CEND 11/11) processed bathymetry
acquisition (MAR-017).

Real, verified-live source (never guessed)
---------------------------------------------
Official dataset "Processed bathymetry from Haisborough, Hammond and
Winterton cSAC" (data.gov.uk package
`processed-bathymetry-from-haisborough-hammond-and-winterton-csac`,
confirmed live via the CKAN `package_search` API 2026-09-06). Publisher:
Joint Nature Conservation Committee (JNCC). Licence: UK Open Government
Licence. Survey/cruise: CEND 11/11 (RV Cefas Endeavour), field dates
2011-06-11 to 2011-06-21. The single resource `HHW-Bathy.zip` is confirmed
127,960,429 bytes at `https://data.jncc.gov.uk/data/ca5231e7-d5e8-43f0-ae43-
f73f22113b67-HHW-Bathy.zip`.

Format: real ESRI Arc/INFO Binary Grid, never assumed GeoTIFF (Section 4)
-----------------------------------------------------------------------------
Opening the real archive shows 13 separate ESRI GRID directories (each a
`hdr.adf`/`dblbnd.adf`/`sta.adf`/`prj.adf`/`w001001.adf`(+`x`) set, magic
bytes `GRID1.2`) -- 4 larger `asciito_hhw_N` grids (the "official processed
bathymetry product", Section 2's stated preference) and 9 smaller
`hhw_gsfN` grids, plus `.aux`/`.rrd` pyramid sidecars. GDAL's `/vsizip/`
virtual filesystem reads these directly from the ZIP (confirmed live) --
this module never extracts the ~7.1 GB uncompressed archive to disk for
mere inventory/inspection, only for actual pixel access when a specific
grid is selected for processing.
"""

import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio
import requests

HHW_BATHY_ZIP_URL = (
    "https://data.jncc.gov.uk/data/ca5231e7-d5e8-43f0-ae43-f73f22113b67-HHW-Bathy.zip"
)
HHW_DATASET_TITLE = "Processed bathymetry from Haisborough, Hammond and Winterton cSAC"
HHW_PUBLISHER = "Joint Nature Conservation Committee (JNCC)"
HHW_LICENCE = "UK Open Government Licence (OGL)"
HHW_SOURCE_CATALOGUE_URL = (
    "https://www.data.gov.uk/dataset/"
    "processed-bathymetry-from-haisborough-hammond-and-winterton-csac"
)
HHW_CRUISE_ID = "CEND 11/11"
HHW_SURVEY_VESSEL = "RV Cefas Endeavour"
HHW_SURVEY_START_DATE = "2011-06-11"
HHW_SURVEY_END_DATE = "2011-06-21"
HHW_SURVEY_YEAR = 2011
HHW_EXPECTED_ZIP_SIZE_BYTES = 127_960_429

REQUEST_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class HhwZipAcquisition:
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


def download_hhw_bathy_zip(
    dest_path: Path, *, timeout: float = REQUEST_TIMEOUT_S
) -> HhwZipAcquisition:
    """Downloads the one official processed-bathymetry ZIP if not already
    cached (Section 28: subsequent reruns must be cacheable/offline) --
    never re-downloads an already-present, correctly-sized file."""

    from marine_engine.providers.bathymetry.acquisition import compute_sha256

    already_cached = dest_path.exists() and dest_path.stat().st_size == HHW_EXPECTED_ZIP_SIZE_BYTES
    if not already_cached:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(HHW_BATHY_ZIP_URL, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with dest_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)

    return HhwZipAcquisition(
        local_path=dest_path,
        source_url=HHW_BATHY_ZIP_URL,
        retrieved_at_utc=datetime.now(UTC),
        byte_size=dest_path.stat().st_size,
        sha256=compute_sha256(dest_path),
        licence=HHW_LICENCE,
        dataset_title=HHW_DATASET_TITLE,
        cruise_id=HHW_CRUISE_ID,
        survey_year=HHW_SURVEY_YEAR,
        already_cached=already_cached,
    )


# --- Section 5: file/format inventory FIRST, before any morphology processing --------------

_RASTER_EXTENSIONS = (".adf", ".tif", ".tiff", ".asc", ".bag")


def discover_esri_grids(zip_path: Path) -> list[str]:
    """Every ESRI Arc/INFO Binary Grid directory in the archive -- detected
    by the real presence of its `hdr.adf` header member, never assumed
    from a naming convention alone."""

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    grid_dirs = sorted(
        {name.rsplit("/", 1)[0] for name in names if name.lower().endswith("/hdr.adf")}
    )
    return grid_dirs


def _classify_member(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(_RASTER_EXTENSIONS):
        return "raster_component"
    if lower.endswith((".shp", ".shx", ".dbf", ".prj")):
        return "vector_component"
    if lower.endswith((".xml", ".txt", ".log")):
        return "text_metadata"
    return "unknown"


def _inspect_grid_via_vsizip(zip_path: Path, grid_dir: str) -> dict[str, Any] | None:
    vsi_path = f"/vsizip/{zip_path}/{grid_dir}"
    try:
        with rasterio.open(vsi_path) as src:
            return {
                "readable_by_rasterio": True,
                "crs": str(src.crs) if src.crs else None,
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


def build_source_file_inventory(zip_path: Path) -> pd.DataFrame:
    """Section 5: every archive member, classified, with raster metadata
    (CRS/dimensions/pixel size/nodata/band count/dtype) populated ONLY for
    the real ESRI GRID directories that rasterio can actually open --
    never fabricated for a non-raster member. No automatic mosaicking."""

    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()

    grid_dirs = set(discover_esri_grids(zip_path))
    grid_metadata_by_dir = {d: _inspect_grid_via_vsizip(zip_path, d) for d in grid_dirs}

    rows = []
    for info in infos:
        filename = info.filename
        extension = Path(filename).suffix.lower()
        member_grid_dir = filename.rsplit("/", 1)[0] if "/" in filename else None
        grid_meta = grid_metadata_by_dir.get(member_grid_dir) if member_grid_dir else None
        is_primary_grid_file = filename.lower().endswith("hdr.adf")

        rows.append(
            {
                "filename": filename,
                "extension": extension,
                "byte_size": info.file_size,
                "classification": _classify_member(filename),
                "grid_directory": member_grid_dir if member_grid_dir in grid_dirs else None,
                "readable_by_rasterio": bool(grid_meta) if is_primary_grid_file else None,
                "crs": grid_meta["crs"] if (grid_meta and is_primary_grid_file) else None,
                "width": grid_meta["width"] if (grid_meta and is_primary_grid_file) else None,
                "height": grid_meta["height"] if (grid_meta and is_primary_grid_file) else None,
                "pixel_size_x_m": (
                    grid_meta["pixel_size_x_m"] if (grid_meta and is_primary_grid_file) else None
                ),
                "pixel_size_y_m": (
                    grid_meta["pixel_size_y_m"] if (grid_meta and is_primary_grid_file) else None
                ),
                "nodata": grid_meta["nodata"] if (grid_meta and is_primary_grid_file) else None,
                "band_count": grid_meta["band_count"]
                if (grid_meta and is_primary_grid_file)
                else None,
                "dtype": grid_meta["dtype"] if (grid_meta and is_primary_grid_file) else None,
                # Never discoverable from this archive -- explicitly recorded as such
                # (Section 6), never fabricated.
                "vertical_units": None,
                "vertical_datum": None,
            }
        )
    return pd.DataFrame(rows)

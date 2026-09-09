"""Generic project-supplied bathymetry raster ingestion (MAR-026 Section 10).

Does NOT reimplement MAR-020's readiness checks -- `inspect_bathymetry_raster` only opens the
real raster (via rasterio) and assembles the existing `marine_engine.terrain.readiness.
RasterFacts`; the caller passes those facts to the existing, unmodified
`assess_bathymetry_readiness` (imported directly, never duplicated). Declared survey epoch and
vertical datum come from manifest provenance (Section 10: "survey epoch and vertical datum
often come from provenance rather than GeoTIFF metadata") -- this module never claims they were
embedded in the raster when they were, in fact, declared.

MAR-026A: `RasterFacts` itself is never modified (it only exposes the coarser
`crs_is_present`/`crs_is_geographic`/`crs_linear_units`) -- `inspect_bathymetry_raster` also
returns the raster's exact embedded CRS as a separate observed fact, mirroring the existing
`preprocessing.bathymetry` convention of `crs.to_string()`, so the project layer can perform an
exact declared-vs-observed CRS comparison instead of only a geographic/projected-kind check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from marine_engine.terrain import readiness as terrain_readiness

__all__ = ["inspect_bathymetry_raster", "RasterOpenError"]


class RasterOpenError(RuntimeError):
    """The raster file could not be opened/read by rasterio -- a registration-level failure,
    distinct from a scientific readiness failure (there are no facts to assess)."""


def inspect_bathymetry_raster(
    path: Path,
    *,
    declared_vertical_datum: str | None,
    declared_survey_epoch: str | None,
) -> tuple[terrain_readiness.RasterFacts, str | None]:
    """Section 10 steps 1-2: inspect the operator raster and build the existing `RasterFacts`.
    Raises `RasterOpenError` if the file cannot be opened at all -- callers must treat that as a
    registration failure (see `project.registry`), since `assess_bathymetry_readiness` itself
    always assumes a successfully-opened raster (its own `file_readable` check is unconditional).

    Returns `(facts, observed_crs)`. `observed_crs` (MAR-026A) is the raster's exact embedded
    CRS, e.g. `"EPSG:32631"`, or `None` if the raster carries no CRS at all -- that absence is
    still handled by `assess_bathymetry_readiness`'s own `crs_present` check, unchanged here."""

    if not path.is_file():
        raise RasterOpenError(f"bathymetry raster not found: {path}")

    try:
        with rasterio.open(path) as src:
            band = src.read(1)
            crs = src.crs
            transform = src.transform
            nodata = src.nodata
            width, height = src.width, src.height
            bounds = tuple(src.bounds)
            band_count = src.count
            dtype = src.dtypes[0]
            color_interp = tuple(str(c) for c in src.colorinterp)
    except rasterio.errors.RasterioIOError as exc:
        raise RasterOpenError(f"bathymetry raster could not be opened: {path} ({exc})") from exc

    valid_mask = (
        np.isfinite(band)
        if nodata is None
        else (np.isfinite(band) & ~np.isclose(band, nodata, rtol=0, atol=abs(nodata) * 1e-6 + 1e-6))
    )
    valid_values = band[valid_mask]

    facts = terrain_readiness.RasterFacts(
        band_count=band_count,
        dtype=str(dtype),
        color_interpretations=color_interp,
        crs_is_present=crs is not None,
        crs_is_geographic=crs.is_geographic if crs is not None else None,
        crs_linear_units=crs.linear_units if crs is not None else None,
        width=width,
        height=height,
        pixel_size_x_m=transform.a,
        pixel_size_y_m=-transform.e,
        bounds=bounds,
        nodata_value=nodata,
        vertical_datum=declared_vertical_datum,
        survey_epoch=declared_survey_epoch,
        data_min=float(valid_values.min()) if valid_values.size else None,
        data_max=float(valid_values.max()) if valid_values.size else None,
        data_std=float(valid_values.std()) if valid_values.size else None,
        valid_cell_fraction=float(valid_mask.mean()) if valid_mask.size else None,
    )
    observed_crs = crs.to_string() if crs is not None else None
    return facts, observed_crs

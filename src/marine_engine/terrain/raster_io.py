"""Generic canonical GeoTIFF writing for terrain rasters (MAR-020 Section 13).

Zero dependency on any specific project or dataset. Mirrors the
float32/NaN-nodata/GTiff writing convention already established in
`preprocessing.bathymetry.write_canonical_raster` (MAR-006/007) -- CRS and
transform are always caller-supplied (never re-derived or guessed), never
imported directly to keep this package independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def write_terrain_raster(
    array: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    output_path: Path,
    tags: dict[str, Any],
) -> Path:
    """Write a single-band terrain raster as a tagged GeoTIFF with explicit
    NaN nodata, float32 dtype."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
    }
    with rasterio.open(output_path, "w", **profile) as dataset:
        dataset.write(array.astype("float32"), 1)
        dataset.update_tags(**{k: str(v) for k, v in tags.items()})
    return output_path

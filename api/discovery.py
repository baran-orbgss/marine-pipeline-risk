"""Spatial-metadata auto-discovery: raster / vector / CSV inspection.

Reports only what a file itself declares -- embedded CRS, bounds, geometry type, pixel size,
fields. Never infers scientific semantic role (that stays "unclassified" until an accepted config/
manifest establishes it, see `api/layers.py`), and never guesses CSV coordinate columns from
column name or magnitude -- those require an explicit, separate declaration.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
from pyproj import CRS, Transformer

from api.models import BoundingBox, SpatialFieldInfo, SpatialMetadataInspection

RASTER_EXTENSIONS = {".tif", ".tiff"}
VECTOR_EXTENSIONS = {".gpkg", ".geojson", ".json"}
TABULAR_EXTENSIONS = {".csv"}
PARQUET_EXTENSIONS = {".parquet"}

SUPPORTED_EXTENSIONS = (
    RASTER_EXTENSIONS | VECTOR_EXTENSIONS | TABULAR_EXTENSIONS | PARQUET_EXTENSIONS
)


def bounds_to_wgs84(
    minx: float, miny: float, maxx: float, maxy: float, crs_str: str | None
) -> BoundingBox | None:
    if crs_str is None:
        return None
    try:
        src_crs = CRS.from_user_input(crs_str)
    except Exception:
        return None
    if src_crs.to_epsg() == 4326:
        return BoundingBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    try:
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
        xs, ys = transformer.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
        return BoundingBox(minx=min(xs), miny=min(ys), maxx=max(xs), maxy=max(ys))
    except Exception:
        return None


def inspect_raster(path: Path) -> SpatialMetadataInspection:
    warnings: list[str] = []
    with rasterio.open(path) as src:
        crs_str = src.crs.to_string() if src.crs else None
        if crs_str is None:
            warnings.append("no CRS embedded in raster")
        bounds = src.bounds
        native_bbox = BoundingBox(
            minx=bounds.left,
            miny=bounds.bottom,
            maxx=bounds.right,
            maxy=bounds.top,
            crs=crs_str or "unknown",
        )
        wgs84_bbox = bounds_to_wgs84(bounds.left, bounds.bottom, bounds.right, bounds.top, crs_str)
        if wgs84_bbox is None:
            warnings.append("could not reproject bounds to EPSG:4326 for display")
        return SpatialMetadataInspection(
            kind="raster",
            observed_crs=crs_str,
            bounds_native=native_bbox,
            bounds_wgs84=wgs84_bbox,
            pixel_size_x=abs(src.transform.a),
            pixel_size_y=abs(src.transform.e),
            width=src.width,
            height=src.height,
            band_count=src.count,
            nodata=src.nodata,
            warnings=warnings,
        )


def inspect_vector(path: Path, layer: str | None = None) -> SpatialMetadataInspection:
    warnings: list[str] = []
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    crs_str = gdf.crs.to_string() if gdf.crs is not None else None
    if crs_str is None:
        warnings.append("no CRS embedded in vector source")
    if gdf.empty:
        bounds_native = None
        wgs84_bbox = None
    else:
        minx, miny, maxx, maxy = gdf.total_bounds
        bounds_native = BoundingBox(
            minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=crs_str or "unknown"
        )
        wgs84_bbox = bounds_to_wgs84(minx, miny, maxx, maxy, crs_str)
    geometry_types = set(gdf.geom_type.dropna().unique()) if not gdf.empty else set()
    fields = [
        SpatialFieldInfo(name=str(col), dtype=str(dtype))
        for col, dtype in gdf.dtypes.items()
        if col != gdf.geometry.name
    ]
    return SpatialMetadataInspection(
        kind="vector",
        observed_crs=crs_str,
        bounds_native=bounds_native,
        bounds_wgs84=wgs84_bbox,
        geometry_type=", ".join(sorted(geometry_types)) if geometry_types else None,
        feature_count=len(gdf),
        fields=fields,
        warnings=warnings,
    )


def inspect_csv_columns(path: Path) -> SpatialMetadataInspection:
    import pandas as pd

    df = pd.read_csv(path, nrows=200)
    fields = [SpatialFieldInfo(name=str(col), dtype=str(dtype)) for col, dtype in df.dtypes.items()]
    return SpatialMetadataInspection(
        kind="tabular_unresolved",
        fields=fields,
        coordinate_columns_declared=False,
        warnings=[
            "coordinate columns not yet declared; geometry cannot be built until an explicit "
            "x/y column + CRS is provided"
        ],
    )


def inspect_parquet(path: Path) -> SpatialMetadataInspection:
    """Geoparquet (has a geometry column) inspects as a vector; a plain tabular parquet falls back
    to the same "columns only, no geometry yet" shape as an unresolved CSV -- never guessed."""
    try:
        gdf = gpd.read_parquet(path)
    except Exception:
        import pandas as pd

        df = pd.read_parquet(path)
        fields = [
            SpatialFieldInfo(name=str(col), dtype=str(dtype)) for col, dtype in df.dtypes.items()
        ]
        return SpatialMetadataInspection(
            kind="tabular_unresolved",
            fields=fields,
            warnings=["parquet file has no recognised geometry column"],
        )
    crs_str = gdf.crs.to_string() if gdf.crs is not None else None
    if gdf.empty:
        bounds_native = None
        wgs84_bbox = None
    else:
        minx, miny, maxx, maxy = gdf.total_bounds
        bounds_native = BoundingBox(
            minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=crs_str or "unknown"
        )
        wgs84_bbox = bounds_to_wgs84(minx, miny, maxx, maxy, crs_str)
    geometry_types = set(gdf.geom_type.dropna().unique()) if not gdf.empty else set()
    fields = [
        SpatialFieldInfo(name=str(col), dtype=str(dtype))
        for col, dtype in gdf.dtypes.items()
        if col != gdf.geometry.name
    ]
    return SpatialMetadataInspection(
        kind="vector",
        observed_crs=crs_str,
        bounds_native=bounds_native,
        bounds_wgs84=wgs84_bbox,
        geometry_type=", ".join(sorted(geometry_types)) if geometry_types else None,
        feature_count=len(gdf),
        fields=fields,
        warnings=[],
    )


def inspect_file(path: Path, layer: str | None = None) -> SpatialMetadataInspection:
    suffix = path.suffix.lower()
    if suffix in RASTER_EXTENSIONS:
        return inspect_raster(path)
    if suffix in VECTOR_EXTENSIONS:
        return inspect_vector(path, layer=layer)
    if suffix in PARQUET_EXTENSIONS:
        return inspect_parquet(path)
    if suffix in TABULAR_EXTENSIONS:
        return inspect_csv_columns(path)
    raise ValueError(f"unsupported file extension for spatial inspection: {suffix}")


# A small, explicit table of KNOWN non-geoparquet engine outputs whose coordinate columns are a
# verified property of the capability that produced them -- e.g. `build-bedform-morphodynamics-
# poc`'s crest-match table always carries `epoch1_x_m`/`epoch1_y_m` projected coordinates in the
# study's working CRS. This is capability-specific knowledge about a fixed, known output shape,
# never a guess about an arbitrary file's columns -- contrast with an uploaded CSV, whose
# coordinate columns always require an explicit, separate declaration (see
# `build_csv_point_geometry`).
_KNOWN_TABULAR_COORDINATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("epoch1_x_m", "epoch1_y_m"),
    ("easting_m", "northing_m"),
)


def reconstruct_known_tabular_geometry(path: Path, crs: str | None) -> gpd.GeoDataFrame:
    import pandas as pd

    df = pd.read_parquet(path)
    if crs:
        for x_col, y_col in _KNOWN_TABULAR_COORDINATE_COLUMNS:
            if x_col in df.columns and y_col in df.columns:
                return gpd.GeoDataFrame(
                    df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=crs
                )
    raise ValueError(f"no known geometry reconstruction for {path}")


def read_vector_any(
    path: Path, layer: str | None = None, *, crs_hint: str | None = None
) -> gpd.GeoDataFrame:
    """Read a vector source regardless of container format (GeoPackage/GeoJSON/geoparquet), with a
    narrow fallback for known plain-tabular parquet outputs (see
    `reconstruct_known_tabular_geometry`)."""
    if path.suffix.lower() in PARQUET_EXTENSIONS:
        try:
            return gpd.read_parquet(path)
        except Exception:
            return reconstruct_known_tabular_geometry(path, crs_hint)
    return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)


def build_csv_point_geometry(
    path: Path, *, x_column: str, y_column: str, crs: str
) -> gpd.GeoDataFrame:
    """Build point geometry from explicitly-declared coordinate columns. Never called with
    columns inferred from name or magnitude -- the caller must supply an explicit declaration."""
    import pandas as pd

    df = pd.read_csv(path)
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(f"declared coordinate columns not found: {x_column!r}, {y_column!r}")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_column], df[y_column]), crs=crs)

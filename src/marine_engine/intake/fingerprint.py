"""Generic structural data fingerprinting (MAR-033 Section 26).

`fingerprint_path` reports OBSERVED structural facts only -- container type (from real magic
bytes, never a filename alone, where a reliable signature exists), CRS, bounds, geometry type,
raster dimensions/pixel size/band count, column names/dtypes, row/feature count, any embedded
Parquet schema metadata, and layer names. It assigns NO scientific semantic role: a GeoTIFF with
a CRS and elevation-like values is observed as a georeferenced raster, never as "bathymetry" --
that judgement belongs to `recognition.py`, and even there only when evidence is sufficiently
specific (Section 28).

A file that cannot be opened by any reader still returns a fingerprint
(`container_type = UNREADABLE`, with the failure recorded in `read_problems`) rather than
raising -- recognition then honestly reports `INVALID`, never crashing the planner.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CONTAINER_PARQUET",
    "CONTAINER_GEOTIFF",
    "CONTAINER_GEOPACKAGE",
    "CONTAINER_VECTOR_OTHER",
    "CONTAINER_CSV",
    "CONTAINER_UNKNOWN",
    "CONTAINER_UNREADABLE",
    "DataFingerprint",
    "fingerprint_path",
]

CONTAINER_PARQUET = "PARQUET"
CONTAINER_GEOTIFF = "GEOTIFF"
CONTAINER_GEOPACKAGE = "GEOPACKAGE"
CONTAINER_VECTOR_OTHER = "VECTOR_OTHER"
CONTAINER_CSV = "CSV"
CONTAINER_UNKNOWN = "UNKNOWN"
CONTAINER_UNREADABLE = "UNREADABLE"

_PARQUET_MAGIC = b"PAR1"
_SQLITE_MAGIC = b"SQLite format 3\x00"
_TIFF_MAGIC_LE = b"II*\x00"
_TIFF_MAGIC_BE = b"MM\x00*"


@dataclass(frozen=True)
class DataFingerprint:
    """Observed facts only (Section 26) -- no field here is an interpretation."""

    path: str
    byte_size: int
    sha256: str
    file_extension: str
    container_type: str
    geometry_type: str | None = None
    crs: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    raster_width: int | None = None
    raster_height: int | None = None
    pixel_size: tuple[float, float] | None = None
    band_count: int | None = None
    column_names: tuple[str, ...] = ()
    column_dtypes: dict[str, str] = field(default_factory=dict)
    row_or_feature_count: int | None = None
    layer_names: tuple[str, ...] = ()
    parquet_schema_metadata: dict[str, str] = field(default_factory=dict)
    read_problems: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "file_extension": self.file_extension,
            "container_type": self.container_type,
            "geometry_type": self.geometry_type,
            "crs": self.crs,
            "bounds": list(self.bounds) if self.bounds else None,
            "raster_width": self.raster_width,
            "raster_height": self.raster_height,
            "pixel_size": list(self.pixel_size) if self.pixel_size else None,
            "band_count": self.band_count,
            "column_names": list(self.column_names),
            "column_dtypes": dict(self.column_dtypes),
            "row_or_feature_count": self.row_or_feature_count,
            "layer_names": list(self.layer_names),
            "parquet_schema_metadata": dict(self.parquet_schema_metadata),
            "read_problems": list(self.read_problems),
        }


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_parquet(path: Path, base: dict[str, Any]) -> DataFingerprint:
    import pyarrow.parquet as pq

    problems: list[str] = []
    schema = pq.read_schema(path)
    metadata: dict[str, str] = {}
    for raw_key, raw_value in (schema.metadata or {}).items():
        try:
            metadata[raw_key.decode("utf-8")] = raw_value.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            continue
    column_names = tuple(schema.names)
    column_dtypes = {name: str(schema.field(name).type) for name in schema.names}
    try:
        row_count: int | None = pq.ParquetFile(path).metadata.num_rows
    except Exception as exc:  # noqa: BLE001 -- degrade to "unknown row count", never fail the fingerprint
        row_count = None
        problems.append(f"row count unavailable: {exc}")

    geometry_type = crs = None
    bounds: tuple[float, float, float, float] | None = None
    if "geo" in metadata or any(name.lower() == "geometry" for name in column_names):
        try:
            import geopandas as gpd

            gdf = gpd.read_parquet(path)
            crs = gdf.crs.to_string() if gdf.crs is not None else None
            if not gdf.empty:
                minx, miny, maxx, maxy = gdf.total_bounds
                bounds = (float(minx), float(miny), float(maxx), float(maxy))
                geometry_type = ", ".join(sorted(set(gdf.geom_type.dropna().unique())))
        except Exception as exc:  # noqa: BLE001 -- a plain tabular parquet is simply not geoparquet
            problems.append(f"not read as geoparquet: {exc}")

    return DataFingerprint(
        **base,
        container_type=CONTAINER_PARQUET,
        column_names=column_names,
        column_dtypes=column_dtypes,
        row_or_feature_count=row_count,
        parquet_schema_metadata=metadata,
        geometry_type=geometry_type,
        crs=crs,
        bounds=bounds,
        read_problems=tuple(problems),
    )


def _fingerprint_raster(path: Path, base: dict[str, Any]) -> DataFingerprint:
    import rasterio

    with rasterio.open(path) as src:
        crs = src.crs.to_string() if src.crs else None
        bounds = (
            float(src.bounds.left),
            float(src.bounds.bottom),
            float(src.bounds.right),
            float(src.bounds.top),
        )
        return DataFingerprint(
            **base,
            container_type=CONTAINER_GEOTIFF,
            crs=crs,
            bounds=bounds,
            raster_width=src.width,
            raster_height=src.height,
            pixel_size=(abs(float(src.transform.a)), abs(float(src.transform.e))),
            band_count=src.count,
        )


def _list_layers(path: Path) -> tuple[str, ...]:
    """The repository's actual vector I/O engine is `pyogrio` (geopandas' default backend; the
    optional `fiona` dependency is not installed) -- try it first and fall back to `fiona` only
    if some future environment has it instead. A single-layer format (e.g. GeoJSON) that
    supports neither still degrades to an empty tuple rather than raising."""

    try:
        import pyogrio

        return tuple(str(name) for name, _geometry_type in pyogrio.list_layers(path))
    except Exception:  # noqa: BLE001 -- fall back, never fail the fingerprint over layer listing
        pass
    try:
        import fiona

        return tuple(fiona.listlayers(path))
    except Exception:  # noqa: BLE001 -- a single-layer format may support neither backend
        return ()


def _fingerprint_vector(
    path: Path, base: dict[str, Any], *, container_type: str
) -> DataFingerprint:
    import geopandas as gpd

    layers = _list_layers(path)
    layer = layers[0] if layers else None
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    crs = gdf.crs.to_string() if gdf.crs is not None else None
    bounds = geometry_type = None
    if not gdf.empty:
        minx, miny, maxx, maxy = gdf.total_bounds
        bounds = (float(minx), float(miny), float(maxx), float(maxy))
        geometry_type = ", ".join(sorted(set(gdf.geom_type.dropna().unique())))
    return DataFingerprint(
        **base,
        container_type=container_type,
        crs=crs,
        bounds=bounds,
        geometry_type=geometry_type,
        column_names=tuple(c for c in gdf.columns if c != gdf.geometry.name),
        column_dtypes={c: str(dt) for c, dt in gdf.dtypes.items() if c != gdf.geometry.name},
        row_or_feature_count=int(len(gdf)),
        layer_names=layers,
    )


def _fingerprint_csv(path: Path, base: dict[str, Any]) -> DataFingerprint:
    import pandas as pd

    df = pd.read_csv(path, nrows=200)
    return DataFingerprint(
        **base,
        container_type=CONTAINER_CSV,
        column_names=tuple(str(c) for c in df.columns),
        column_dtypes={str(c): str(dt) for c, dt in df.dtypes.items()},
    )


def fingerprint_path(path: str | Path) -> DataFingerprint:
    """Section 26. Container type is decided by real magic bytes wherever a reliable signature
    exists (Parquet, SQLite/GeoPackage, TIFF); a file extension only decides among the remaining
    formats that have no fixed-offset signature (CSV, plain GeoJSON)."""

    path = Path(path)
    byte_size = path.stat().st_size
    sha256 = _sha256_of(path)
    base = {
        "path": str(path),
        "byte_size": byte_size,
        "sha256": sha256,
        "file_extension": path.suffix.lower(),
    }
    with path.open("rb") as fh:
        head = fh.read(16)

    try:
        if head.startswith(_PARQUET_MAGIC):
            return _fingerprint_parquet(path, base)
        if head.startswith(_SQLITE_MAGIC):
            return _fingerprint_vector(path, base, container_type=CONTAINER_GEOPACKAGE)
        if head.startswith(_TIFF_MAGIC_LE) or head.startswith(_TIFF_MAGIC_BE):
            return _fingerprint_raster(path, base)
        if path.suffix.lower() in (".csv", ".tsv", ".txt"):
            return _fingerprint_csv(path, base)
        if path.suffix.lower() in (".geojson", ".json"):
            return _fingerprint_vector(path, base, container_type=CONTAINER_VECTOR_OTHER)
    except Exception as exc:  # noqa: BLE001 -- an unreadable/corrupt file is reported, never raised
        return DataFingerprint(
            **base, container_type=CONTAINER_UNREADABLE, read_problems=(str(exc),)
        )

    return DataFingerprint(**base, container_type=CONTAINER_UNKNOWN)

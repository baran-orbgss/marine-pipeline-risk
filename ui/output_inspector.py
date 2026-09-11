"""Metadata-first inspection of LOCAL engine outputs (UI-001 Sections 17-20, 23).

Every function here reads; nothing writes, rewrites, reprojects, resamples canonical data or
downloads anything. Missing files are reported as `NOT PRESENT LOCALLY`, never fabricated. On page
load only metadata is read; row / pixel previews are bounded and requested explicitly. A raster
preview is a decimated DISPLAY PREVIEW -- NOT CANONICAL DATA -- and is labelled as such.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ui import REPO_ROOT
from ui import capability_registry as registry

__all__ = [
    "OutputInspectorError",
    "OutputEntry",
    "DATA_ROOT",
    "NOT_PRESENT_LOCALLY",
    "PREVIEW_LABEL",
    "LARGE_FILE_BYTES",
    "contain_path",
    "classify_kind",
    "discover_outputs",
    "list_directory_outputs",
    "evidence_kind_for_path",
    "with_registry_evidence_kind",
    "inspect_json",
    "inspect_parquet",
    "parquet_head",
    "inspect_gpkg",
    "inspect_geotiff",
    "geotiff_preview",
    "summarize_readiness",
]

DATA_ROOT = (REPO_ROOT / "data").resolve()
_ALLOWED_ROOTS = (DATA_ROOT / "processed", DATA_ROOT / "interim")
NOT_PRESENT_LOCALLY = "NOT PRESENT LOCALLY"
PREVIEW_LABEL = "DISPLAY PREVIEW -- NOT CANONICAL DATA"
LARGE_FILE_BYTES = 50 * 1024 * 1024  # above this, the UI must ask before reading rows
MAX_JSON_BYTES = 8 * 1024 * 1024

_KIND_BY_SUFFIX = {
    ".json": "json",
    ".parquet": "parquet",
    ".gpkg": "gpkg",
    ".tif": "geotiff",
    ".tiff": "geotiff",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".html": "html",
    ".md": "text",
    ".txt": "text",
}


class OutputInspectorError(ValueError):
    """Path outside the local data roots, unsupported kind, or a read guard tripped."""


@dataclass(frozen=True)
class OutputEntry:
    label: str
    relative_path: str  # relative to data/, posix
    exists: bool
    kind: str
    byte_size: int | None
    evidence_kind: str
    project_id: str | None = None
    pattern: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        return "PRESENT" if self.exists else NOT_PRESENT_LOCALLY

    @property
    def absolute_path(self) -> Path:
        return DATA_ROOT / self.relative_path

    @property
    def is_large(self) -> bool:
        return bool(self.byte_size and self.byte_size > LARGE_FILE_BYTES)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status, "is_large": self.is_large}


def contain_path(path: str | Path) -> Path:
    """Resolve `path` (absolute, or relative to data/) and require it to live under
    data/processed or data/interim. Traversal, symlink escapes and other roots are rejected."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = DATA_ROOT / candidate
    resolved = candidate.resolve()
    for root in _ALLOWED_ROOTS:
        if resolved == root or root in resolved.parents:
            return resolved
    raise OutputInspectorError(f"path is outside the local output roots: {path}")


def classify_kind(path: str | Path) -> str:
    return _KIND_BY_SUFFIX.get(Path(path).suffix.lower(), "other")


def _entry_for_file(
    path: Path, *, label: str, evidence_kind: str, project_id: str | None, pattern: str | None
) -> OutputEntry:
    resolved = contain_path(path)
    exists = resolved.is_file()
    return OutputEntry(
        label=label,
        relative_path=resolved.relative_to(DATA_ROOT).as_posix(),
        exists=exists,
        kind=classify_kind(resolved),
        byte_size=resolved.stat().st_size if exists else None,
        evidence_kind=evidence_kind if exists else registry.NOT_AVAILABLE,
        project_id=project_id,
        pattern=pattern,
    )


def discover_outputs(patterns: Sequence[registry.OutputPattern]) -> list[OutputEntry]:
    """Expand each registered pattern under data/. A pattern with no local match yields ONE entry
    with `exists=False` (rendered as NOT PRESENT LOCALLY) -- never a fabricated file."""

    entries: list[OutputEntry] = []
    for pattern in patterns:
        matches = sorted(p for p in DATA_ROOT.glob(pattern.glob) if p.is_file())
        matches = [p for p in matches if _within_allowed(p)]
        if not matches:
            entries.append(
                OutputEntry(
                    label=pattern.label,
                    relative_path=pattern.glob,
                    exists=False,
                    kind=classify_kind(pattern.glob),
                    byte_size=None,
                    evidence_kind=registry.NOT_AVAILABLE,
                    project_id=pattern.project_id,
                    pattern=pattern.glob,
                    notes=(NOT_PRESENT_LOCALLY,),
                )
            )
            continue
        for match in matches:
            entries.append(
                _entry_for_file(
                    match,
                    label=pattern.label,
                    evidence_kind=pattern.evidence_kind,
                    project_id=pattern.project_id,
                    pattern=pattern.glob,
                )
            )
    return entries


def _within_allowed(path: Path) -> bool:
    try:
        contain_path(path)
    except OutputInspectorError:
        return False
    return True


def evidence_kind_for_path(relative_path: str) -> str:
    """The registry-declared evidence kind for a local file: SYNTHETIC_TEST_FIXTURE when any
    registered synthetic pattern matches, otherwise CACHED_REAL_OUTPUT. Never inferred from the
    filename alone -- only from explicit registry patterns."""

    for capability in registry.HAZARDS:
        for pattern in capability.output_patterns:
            if pattern.evidence_kind == registry.SYNTHETIC_TEST_FIXTURE and fnmatch.fnmatch(
                relative_path, pattern.glob
            ):
                return registry.SYNTHETIC_TEST_FIXTURE
    return registry.CACHED_REAL_OUTPUT


def with_registry_evidence_kind(entry: OutputEntry) -> OutputEntry:
    if not entry.exists:
        return entry
    return replace(entry, evidence_kind=evidence_kind_for_path(entry.relative_path))


def list_directory_outputs(
    relative_dir: str,
    *,
    evidence_kind: str = registry.CACHED_REAL_OUTPUT,
    project_id: str | None = None,
) -> list[OutputEntry]:
    """Every file under one data/<processed|interim>/<dir> tree (metadata only). A directory that
    is absent locally yields an empty list."""

    try:
        root = contain_path(relative_dir)
    except OutputInspectorError:
        raise
    if not root.is_dir():
        return []
    return [
        _entry_for_file(
            p,
            label=p.relative_to(root).as_posix(),
            evidence_kind=evidence_kind,
            project_id=project_id,
            pattern=None,
        )
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


# --- JSON --------------------------------------------------------------------------------------


def inspect_json(path: str | Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    resolved = contain_path(path)
    if not resolved.is_file():
        return {"status": NOT_PRESENT_LOCALLY, "path": str(path)}
    size = resolved.stat().st_size
    if size > max_bytes:
        raise OutputInspectorError(f"JSON larger than {max_bytes} bytes; not loaded ({size} bytes)")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return {
        "status": "PRESENT",
        "byte_size": size,
        "top_level_type": type(data).__name__,
        "top_level_keys": sorted(data.keys()) if isinstance(data, dict) else None,
        "data": data,
    }


_STATUS_KEYS = (
    "status",
    "readiness_status",
    "readiness_status_effective",
    "readiness_status_intrinsic",
)


def summarize_readiness(payload: Any) -> dict[str, Any]:
    """Pull explicit status-like fields out of an engine readiness JSON WITHOUT reinterpreting them.
    Only keys the engine itself wrote are echoed; nothing is derived."""

    found: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in _STATUS_KEYS:
            if key in payload and isinstance(payload[key], str | int | float | bool):
                found[key] = payload[key]
        for key, value in payload.items():
            if isinstance(value, dict):
                for sub in _STATUS_KEYS:
                    if sub in value and isinstance(value[sub], str):
                        found[f"{key}.{sub}"] = value[sub]
        for key in ("blocking_reasons", "limitation_reasons", "limitations", "findings"):
            if isinstance(payload.get(key), list):
                found[key] = payload[key]
    return found


# --- Parquet -----------------------------------------------------------------------------------


def inspect_parquet(path: str | Path) -> dict[str, Any]:
    """Footer metadata first: rows, columns, schema, row groups, marine_engine_* file metadata and
    null counts from row-group statistics. Columns whose statistics are absent (pyarrow writes none
    for an all-null column) are counted by a column-restricted scan only when the file is not large;
    otherwise the count stays None (unknown). No full table is ever materialized."""

    import pyarrow.parquet as pq

    resolved = contain_path(path)
    if not resolved.is_file():
        return {"status": NOT_PRESENT_LOCALLY, "path": str(path)}
    pf = pq.ParquetFile(resolved)
    meta = pf.metadata
    schema = pf.schema_arrow
    null_counts: dict[str, int | None] = dict.fromkeys(schema.names, 0)
    for rg in range(meta.num_row_groups):
        group = meta.row_group(rg)
        for col in range(group.num_columns):
            column = group.column(col)
            name = column.path_in_schema
            if name not in null_counts:
                continue
            stats = column.statistics
            if stats is None or not stats.has_null_count or null_counts[name] is None:
                null_counts[name] = None
            else:
                null_counts[name] = (null_counts[name] or 0) + int(stats.null_count)
    basis = dict.fromkeys(schema.names, "row_group_statistics")
    unknown = [name for name, count in null_counts.items() if count is None]
    byte_size = resolved.stat().st_size
    if unknown and byte_size <= LARGE_FILE_BYTES:
        scanned = pq.read_table(resolved, columns=unknown)
        for name in unknown:
            null_counts[name] = int(scanned.column(name).null_count)
            basis[name] = "column_scan"
    else:
        for name in unknown:
            basis[name] = "unknown"
    file_meta = {
        k.decode("utf-8", "replace"): v.decode("utf-8", "replace")
        for k, v in (schema.metadata or {}).items()
        if k.startswith(b"marine_engine_")
    }
    return {
        "status": "PRESENT",
        "byte_size": byte_size,
        "rows": int(meta.num_rows),
        "row_groups": int(meta.num_row_groups),
        "columns": list(schema.names),
        "schema": [f"{f.name}: {f.type}" for f in schema],
        "null_counts": null_counts,
        "null_count_basis": basis,
        "marine_engine_metadata": file_meta,
        "created_by": meta.created_by,
    }


def parquet_head(path: str | Path, rows: int = 20):
    """Read ONLY the first `rows` rows (first batch) as a pandas DataFrame."""

    import pyarrow.parquet as pq

    resolved = contain_path(path)
    if not resolved.is_file():
        raise OutputInspectorError(NOT_PRESENT_LOCALLY)
    rows = max(1, min(int(rows), 500))
    pf = pq.ParquetFile(resolved)
    for batch in pf.iter_batches(batch_size=rows):
        return batch.to_pandas()
    import pandas as pd

    return pd.DataFrame(columns=pf.schema_arrow.names)


# --- GeoPackage --------------------------------------------------------------------------------


def inspect_gpkg(path: str | Path) -> dict[str, Any]:
    """Layer names, feature counts, CRS, geometry types and bounds from OGR metadata; no feature
    data is loaded and nothing is rewritten."""

    import pyogrio

    resolved = contain_path(path)
    if not resolved.is_file():
        return {"status": NOT_PRESENT_LOCALLY, "path": str(path)}
    layers = []
    for name, geometry_type in pyogrio.list_layers(str(resolved)):
        info = pyogrio.read_info(str(resolved), layer=name, force_total_bounds=True)
        bounds = info.get("total_bounds")
        layers.append(
            {
                "layer": str(name),
                "geometry_type": str(geometry_type) if geometry_type is not None else None,
                "feature_count": int(info["features"])
                if info.get("features") is not None
                else None,
                "crs": info.get("crs"),
                "bounds": [float(b) for b in bounds] if bounds is not None else None,
                "fields": [str(f) for f in info.get("fields", [])],
            }
        )
    return {"status": "PRESENT", "byte_size": resolved.stat().st_size, "layers": layers}


# --- GeoTIFF -----------------------------------------------------------------------------------


def inspect_geotiff(path: str | Path) -> dict[str, Any]:
    """Dataset header only (CRS, shape, resolution, transform, nodata, bounds, band count, dtypes).
    No pixel is read."""

    import rasterio

    resolved = contain_path(path)
    if not resolved.is_file():
        return {"status": NOT_PRESENT_LOCALLY, "path": str(path)}
    with rasterio.open(resolved) as src:
        transform = src.transform
        return {
            "status": "PRESENT",
            "byte_size": resolved.stat().st_size,
            "crs": src.crs.to_string() if src.crs else None,
            "width": int(src.width),
            "height": int(src.height),
            "cell_count": int(src.width) * int(src.height),
            "band_count": int(src.count),
            "dtypes": list(src.dtypes),
            "resolution": [float(r) for r in src.res],
            "transform": [
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            ],
            "nodata": src.nodata,
            "bounds": [float(b) for b in src.bounds],
            "overviews": [src.overviews(i) for i in range(1, src.count + 1)],
        }


def geotiff_preview(path: str | Path, *, max_pixels: int = 512, band: int = 1) -> dict[str, Any]:
    """Bounded, decimated read for DISPLAY ONLY. The output grid is at most `max_pixels` on its
    longer side; nodata is masked. The canonical raster on disk is untouched, and the result is
    labelled `PREVIEW_LABEL` so it can never be mistaken for canonical data."""

    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    resolved = contain_path(path)
    if not resolved.is_file():
        raise OutputInspectorError(NOT_PRESENT_LOCALLY)
    max_pixels = max(16, min(int(max_pixels), 2048))
    with rasterio.open(resolved) as src:
        scale = max(src.width, src.height) / max_pixels
        out_h = max(1, int(round(src.height / scale))) if scale > 1 else src.height
        out_w = max(1, int(round(src.width / scale))) if scale > 1 else src.width
        data = src.read(band, out_shape=(out_h, out_w), resampling=Resampling.nearest, masked=True)
        array = np.ma.filled(data.astype("float64"), np.nan)
        finite = array[np.isfinite(array)]
        return {
            "label": PREVIEW_LABEL,
            "array": array,
            "source_shape": [int(src.height), int(src.width)],
            "preview_shape": [int(out_h), int(out_w)],
            "decimation_factor": float(max(scale, 1.0)),
            "vmin": float(finite.min()) if finite.size else None,
            "vmax": float(finite.max()) if finite.size else None,
            "crs": src.crs.to_string() if src.crs else None,
            "bounds": [float(b) for b in src.bounds],
        }

"""Per-layer property allowlisting and CPT profile extraction.

`/features` never returns raw GeoDataFrame columns -- only the keys declared in the layer's own
`display.tooltip_fields` -- so there is exactly one code path that can ever emit vector properties,
and it structurally cannot regress into a raw-attribute dump.

CPT channels are a closed set (qc/fs/u2/qt): `qt` is reported explicitly as unavailable when the
source has no finite values for it, never silently omitted and never used to derive CRR/CSR/a
factor of safety/soil classification -- none of that is computed anywhere in this module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from api.discovery import read_vector_any
from api.models import CptChannel, CptProfile, LayerDescriptor

_CPT_CHANNEL_COLUMNS: dict[str, tuple[str, str]] = {
    "qc": ("qc_mpa", "MPa"),
    "fs": ("fs_kpa", "kPa"),
    "u2": ("u2_kpa", "kPa"),
    "qt": ("qt_mpa", "MPa"),
}


def features_geojson(layer: LayerDescriptor, absolute_path: Path) -> dict:
    gdf = read_vector_any(absolute_path, layer=layer.gpkg_layer, crs_hint=layer.crs_observed)
    if gdf.crs is not None:
        gdf = gdf.to_crs(4326)
    allowlist = [f.key for f in layer.display.tooltip_fields]
    keep_columns = [c for c in allowlist if c in gdf.columns]
    filtered = gdf[[*keep_columns, gdf.geometry.name]].copy()
    return filtered.__geo_interface__


def _read_measurements(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def cpt_profile(measurements_path: Path, test_id: str) -> CptProfile | None:
    df = _read_measurements(measurements_path)
    if "test_id" not in df.columns or "depth_bsf_m" not in df.columns:
        return None
    subset = df[df["test_id"] == test_id].sort_values("depth_bsf_m")
    if subset.empty:
        return None

    depths = [float(v) for v in subset["depth_bsf_m"].tolist()]
    channels: dict[str, CptChannel] = {}
    for key, (column, unit) in _CPT_CHANNEL_COLUMNS.items():
        if column not in subset.columns or subset[column].notna().sum() == 0:
            channels[key] = CptChannel(unit=unit, available=False, values=None)
            continue
        values = [float(v) if pd.notna(v) else None for v in subset[column].tolist()]
        channels[key] = CptChannel(unit=unit, available=True, values=values)
    return CptProfile(test_id=test_id, depth_bsf_m=depths, channels=channels)

"""Bounded, display-only map/chart rendering for the Guided View (UI-002).

Every function here READS existing local files under `data/processed`/`data/interim` (via
`output_inspector.contain_path`) and returns an in-memory `matplotlib.figure.Figure` plus a
metadata dict -- nothing is written, reprojected in place, or fabricated when a file is absent.
Vector reads may reproject a COPY to a shared display CRS; the source file on disk is never
touched. Raster previews delegate to `output_inspector.geotiff_preview` (bounded/decimated).

DISPLAY VIEW -- CANONICAL DATA UNCHANGED.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering only -- this module never opens a GUI window

from ui import output_inspector as oi  # noqa: E402

__all__ = [
    "DISPLAY_VIEW_LABEL",
    "resolve_first_existing",
    "render_vector_scene",
    "render_raster_preview",
    "cpt_channel_profile",
    "render_cpt_profile",
    "render_burial_profile",
    "cpt_locations_frame",
    "resolve_layer_state",
]

DISPLAY_VIEW_LABEL = "DISPLAY VIEW -- CANONICAL DATA UNCHANGED"

_PALETTE = ("#3fa7d6", "#ee6352", "#59cd90", "#fac05e", "#9d4edd", "#48cae4", "#f4a259")

_CPT_MEASUREMENTS_GLOB = "processed/sheringham_shoal_2008_cptu/cpt_measurements.parquet"
_CPT_CHANNELS = (("qc_mpa", "MPa"), ("fs_kpa", "kPa"), ("u2_kpa", "kPa"), ("qt_mpa", "MPa"))
_CPT_CHANNEL_LABELS = {"qc_mpa": "qc", "fs_kpa": "fs", "u2_kpa": "u2", "qt_mpa": "qt"}

_CPT_LOCATIONS_GLOB = "processed/sheringham_shoal_2008_cptu/cpt_locations.gpkg"

_BURIAL_PROFILE_GLOB = "processed/barrow_2016/burial/canonical_burial_profile.parquet"
_BURIAL_SEMANTICS_GLOB = "processed/barrow_2016/burial/source_burial_semantics.json"


def _within_allowed(path: Path) -> bool:
    try:
        oi.contain_path(path)
    except oi.OutputInspectorError:
        return False
    return True


def resolve_first_existing(globs: Sequence[str]) -> Path | None:
    """First real local match across `globs` (each relative to `data/`), or None if none of
    them resolve to a file -- callers must render a "not present locally" state, never fabricate
    one."""

    for pattern in globs:
        matches = sorted(p for p in oi.DATA_ROOT.glob(pattern) if p.is_file())
        matches = [p for p in matches if _within_allowed(p)]
        if matches:
            return matches[0]
    return None


def _plot_layer(ax: Any, gdf: Any, color: str, label: str) -> None:
    kinds = set(gdf.geom_type.dropna().unique())
    kwargs: dict[str, Any] = {"color": color, "label": label}
    if kinds and kinds <= {"Point", "MultiPoint"}:
        kwargs["markersize"] = 8
    elif kinds and kinds <= {"LineString", "MultiLineString"}:
        kwargs["linewidth"] = 2.2
    elif kinds and kinds <= {"Polygon", "MultiPolygon"}:
        kwargs["alpha"] = 0.35
        kwargs["edgecolor"] = color
    gdf.plot(ax=ax, **kwargs)


def _draw_priority(gdf: Any) -> int:
    """Polygons draw first (background), then points, then lines last -- so a route line stays
    visible on top of an AOI fill or KP-marker points sitting directly on it."""

    kinds = set(gdf.geom_type.dropna().unique())
    if kinds and kinds <= {"Polygon", "MultiPolygon"}:
        return 0
    if kinds and kinds <= {"LineString", "MultiLineString"}:
        return 2
    return 1


def render_vector_scene(
    layers: Sequence[tuple[str, str | None, str]],
    *,
    title: str = "",
    max_points_per_layer: int = 150,
) -> tuple[Any, dict]:
    """Compose the existing layers named in `layers` (glob, layer_name_or_None, display_label)
    into one matplotlib scene. A layer whose file is absent locally is skipped and reported in
    `meta["missing_labels"]` -- never fabricated. Point layers larger than
    `max_points_per_layer` are thinned to real, regularly-spaced existing points (not
    resampled/interpolated) for legibility. Each gpkg is read once via `geopandas.read_file`
    (read-only); a layer in a different CRS is reprojected on an in-memory COPY to match the
    first-loaded layer's CRS -- the source file is never rewritten."""

    import geopandas as gpd

    present_labels: list[str] = []
    missing_labels: list[str] = []
    loaded: list[tuple[Any, str]] = []
    display_crs = None
    for glob, layer_name, label in layers:
        path = resolve_first_existing([glob])
        if path is None:
            missing_labels.append(label)
            continue
        gdf = gpd.read_file(path, layer=layer_name) if layer_name else gpd.read_file(path)
        if display_crs is None:
            display_crs = gdf.crs
        elif gdf.crs is not None and display_crs is not None and gdf.crs != display_crs:
            gdf = gdf.to_crs(display_crs)
        if len(gdf) > max_points_per_layer and set(gdf.geom_type.dropna().unique()) <= {
            "Point",
            "MultiPoint",
        }:
            step = max(1, len(gdf) // max_points_per_layer)
            gdf = gdf.iloc[::step]
        loaded.append((gdf, label))
        present_labels.append(label)

    meta: dict[str, Any] = {
        "present_labels": present_labels,
        "missing_labels": missing_labels,
        "crs": display_crs.to_string() if display_crs is not None else None,
        "label": DISPLAY_VIEW_LABEL,
    }
    if not loaded:
        return None, meta

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    colored = [(gdf, label, _PALETTE[i % len(_PALETTE)]) for i, (gdf, label) in enumerate(loaded)]
    for gdf, label, color in sorted(colored, key=lambda triple: _draw_priority(triple[0])):
        _plot_layer(ax, gdf, color, label)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig, meta


def render_raster_preview(glob: str, *, title: str = "") -> tuple[Any, dict]:
    """Bounded/decimated raster preview for one existing GeoTIFF matching `glob`. Delegates
    entirely to `output_inspector.geotiff_preview` -- never a raw full-resolution read."""

    path = resolve_first_existing([glob])
    if path is None:
        return None, {"present": False, "label": DISPLAY_VIEW_LABEL}

    preview = oi.geotiff_preview(path)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(preview["array"], cmap="viridis")
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.7)
    fig.tight_layout()
    meta = {
        "present": True,
        "label": preview["label"],
        "vmin": preview["vmin"],
        "vmax": preview["vmax"],
        "crs": preview["crs"],
        "decimation_factor": preview["decimation_factor"],
        "source_shape": preview["source_shape"],
        "preview_shape": preview["preview_shape"],
    }
    return fig, meta


def cpt_channel_profile(test_id: str) -> dict:
    """Bounded, column- and row-filtered read of the canonical CPT measurements for ONE
    `test_id` (never the full 138,514-row table). Channel availability is decided by an actual
    non-null check on the filtered slice, never hardcoded."""

    path = resolve_first_existing([_CPT_MEASUREMENTS_GLOB])
    if path is None:
        return {"present": False, "test_id": test_id, "depth_bsf_m": None, "channels": {}}

    import pyarrow.parquet as pq

    columns = ["depth_bsf_m", *[c for c, _unit in _CPT_CHANNELS]]
    table = pq.read_table(path, columns=columns, filters=[("test_id", "=", test_id)])
    if table.num_rows == 0:
        return {"present": True, "test_id": test_id, "depth_bsf_m": None, "channels": {}}

    df = table.to_pandas().sort_values("depth_bsf_m")
    channels: dict[str, dict[str, Any]] = {}
    for col, unit in _CPT_CHANNELS:
        series = df[col]
        available = bool(series.notna().any())
        channels[col] = {
            "unit": unit,
            "available": available,
            "values": series.to_numpy() if available else None,
        }
    return {
        "present": True,
        "test_id": test_id,
        "depth_bsf_m": df["depth_bsf_m"].to_numpy(),
        "channels": channels,
    }


def render_cpt_profile(test_id: str) -> tuple[Any, dict]:
    """One subplot per AVAILABLE measured channel only (qt is never plotted for this source),
    sharing a single depth axis inverted so depth increases downward; each subplot keeps its
    own channel's unit on its own x-axis (qc in MPa is never mixed with fs/u2 in kPa)."""

    profile = cpt_channel_profile(test_id)
    unavailable = [col for col, chan in profile["channels"].items() if not chan["available"]]
    if not profile["present"] or profile["depth_bsf_m"] is None:
        return None, {
            "present": profile["present"],
            "test_id": test_id,
            "unavailable_channels": unavailable,
        }

    available = [(col, chan) for col, chan in profile["channels"].items() if chan["available"]]
    if not available:
        return None, {"present": True, "test_id": test_id, "unavailable_channels": unavailable}

    import matplotlib.pyplot as plt

    depth = profile["depth_bsf_m"]
    fig, axes = plt.subplots(1, len(available), sharey=True, figsize=(3.2 * len(available), 6))
    if len(available) == 1:
        axes = [axes]
    for ax, (col, chan) in zip(axes, available, strict=True):
        ax.plot(chan["values"], depth, linewidth=0.9, color="#3fa7d6")
        ax.set_xlabel(f"{_CPT_CHANNEL_LABELS.get(col, col)} ({chan['unit']})")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel("Depth below seabed (m)")
    axes[0].invert_yaxis()
    fig.suptitle(f"CPT {test_id}", fontsize=11)
    fig.tight_layout()
    meta = {"present": True, "test_id": test_id, "unavailable_channels": unavailable}
    return fig, meta


def cpt_locations_frame() -> Any | None:
    """CPT test locations with `easting_m`/`northing_m` derived FROM THE CANONICAL GEOMETRY
    (never from the source's declared `position_x_raw`/`position_y_raw` fields). None if the
    layer is not present locally."""

    path = resolve_first_existing([_CPT_LOCATIONS_GLOB])
    if path is None:
        return None

    import geopandas as gpd

    gdf = gpd.read_file(path, layer="cpt_locations")
    return gdf.assign(easting_m=gdf.geometry.x, northing_m=gdf.geometry.y)


def render_burial_profile(project_id: str = "barrow_2016") -> tuple[Any, dict]:  # noqa: ARG001
    """The source's raw measured burial value ('Z') vs chainage. The y-axis is deliberately
    never labelled 'burial depth': `source_burial_semantics.json` records the reference point
    and sign convention as UNRESOLVED, so the value is shown exactly as measured. Colours points
    by `measured_burial_state` -- an existing engine-declared classification, not an invented
    one."""

    path = resolve_first_existing([_BURIAL_PROFILE_GLOB])
    if path is None:
        return None, {"present": False}

    import pyarrow.parquet as pq

    table = pq.read_table(
        path, columns=["chainage_m", "measured_burial_value_m", "measured_burial_state"]
    )
    df = table.to_pandas().sort_values("chainage_m")

    semantics = oi.inspect_json(_BURIAL_SEMANTICS_GLOB)
    caveat = None
    if semantics.get("status") == "PRESENT":
        caveat = semantics["data"].get("source_sign_convention_text")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (state, group) in enumerate(df.groupby("measured_burial_state")):
        ax.scatter(
            group["chainage_m"],
            group["measured_burial_value_m"],
            s=4,
            color=_PALETTE[i % len(_PALETTE)],
            label=str(state),
        )
    ax.set_xlabel("Chainage (m)")
    ax.set_ylabel("Z (m) -- reference/sign unresolved")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig, {"present": True, "caveat": caveat}


def resolve_layer_state(available: bool, default_on: bool, requested: bool | None) -> bool:
    """Pure logic backing the layer-control checkboxes: an unavailable layer can never end up
    checked, regardless of what was requested or defaulted."""

    if not available:
        return False
    if requested is None:
        return default_on
    return requested

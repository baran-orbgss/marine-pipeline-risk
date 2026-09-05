"""NSTA/Table B.1 freespan registry reconciliation maps (MAR-014C).

Kept self-contained (never importing MAR-012/013/014/MAR-014A's own map
modules), matching this project's established convention so a future change
here can never destabilize an already-shipped map.

No model prediction layer, ever (Section 17)
-------------------------------------------------
These maps show only officially tabulated/registered evidence -- Table B.1
2018 freespan intervals and the NSTA PL854/PL855 freespan registry. No
susceptibility score, probability, or model-output layer is ever drawn here.
"""

import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NO_MODEL_PREDICTION_STATEMENT = (
    "NO MODEL PREDICTION LAYER, SUSCEPTIBILITY SCORE, OR PROBABILITY IS SHOWN ON THIS MAP -- "
    "only officially tabulated/registered freespan evidence."
)

_PIPELINE_COLORS = {"PL854": "tab:blue", "PL855": "tab:orange"}


# --- Shared self-contained plotting helpers (duplicated by design) ---------------------


def _content_bounds(
    route: LineString, background_raster_path: Path | None = None
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = route.bounds
    if background_raster_path is not None and Path(background_raster_path).exists():
        try:
            import rasterio

            with rasterio.open(background_raster_path) as src:
                bounds = src.bounds
            minx, miny = min(minx, bounds.left), min(miny, bounds.bottom)
            maxx, maxy = max(maxx, bounds.right), max(maxy, bounds.top)
        except Exception:  # noqa: BLE001 -- background context is optional, never fatal
            pass
    return minx, miny, maxx, maxy


def _plot_background_raster(ax, raster_path: Path) -> None:
    try:
        import rasterio

        with rasterio.open(raster_path) as src:
            band = src.read(1, masked=True)
            bounds = src.bounds
        ax.imshow(
            band,
            extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
            cmap="gray",
            alpha=0.35,
            zorder=0,
        )
    except Exception as exc:  # noqa: BLE001 -- background context is optional, never fatal
        print(f"note: could not render background bathymetry context: {exc}", file=sys.stderr)


def _kp_tick_chainages_m(total_length_m: float, interval_km: float = 5.0) -> list[float]:
    interval_m = interval_km * 1000.0
    ticks = list(np.arange(0.0, total_length_m, interval_m))
    if not ticks or not np.isclose(ticks[-1], total_length_m):
        ticks.append(total_length_m)
    return ticks


def _add_kp_labels(ax, route: LineString) -> None:
    from marine_engine.preprocessing.chainage import format_kp_label

    for chainage_m in _kp_tick_chainages_m(route.length):
        point = route.interpolate(chainage_m)
        ax.plot(point.x, point.y, marker="o", markersize=3, color="black", zorder=5)
        ax.annotate(
            format_kp_label(chainage_m),
            (point.x, point.y),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=7,
        )


def _add_endpoint_labels(ax, route: LineString) -> None:
    start, end = Point(route.coords[0]), Point(route.coords[-1])
    for point, label in ((start, "Source geometry start"), (end, "Source geometry terminus")):
        ax.plot(point.x, point.y, marker="s", markersize=5, color="black", zorder=5)
        ax.annotate(
            label,
            (point.x, point.y),
            textcoords="offset points",
            xytext=(8, -26),
            fontsize=8,
            fontweight="bold",
        )


def _add_scale_bar(ax) -> None:
    xlim = ax.get_xlim()
    view_width_m = abs(xlim[1] - xlim[0])
    bar_km = 0.5
    for candidate_km in (0.5, 1, 2, 5, 10, 20, 50):
        if candidate_km * 1000.0 <= view_width_m * 0.35:
            bar_km = candidate_km
        else:
            break
    bar_m = bar_km * 1000.0

    x0 = xlim[0] + view_width_m * 0.05
    y0 = ax.get_ylim()[0] + abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
    ax.add_line(Line2D([x0, x0 + bar_m], [y0, y0], color="black", linewidth=2))
    ax.annotate(
        f"{bar_km:g} km",
        (x0 + bar_m / 2.0, y0),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=8,
    )


def _add_north_arrow(ax) -> None:
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = xlim[1] - abs(xlim[1] - xlim[0]) * 0.08
    y0 = ylim[0] + abs(ylim[1] - ylim[0]) * 0.08
    y1 = y0 + abs(ylim[1] - ylim[0]) * 0.08
    ax.annotate(
        "N",
        xy=(x, y1),
        xytext=(x, y0),
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.5},
        ha="center",
        fontsize=9,
        fontweight="bold",
    )


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px


def _setup_route_axes(
    route: LineString, background_raster_path: Path | None = None
) -> tuple[Any, Any]:
    minx, miny, maxx, maxy = _content_bounds(route, background_raster_path)
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    fig_height = 7.5
    fig_width = max(8.0, min(18.0, fig_height * (span_x / span_y)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pad_x, pad_y = span_x * 0.06, span_y * 0.10
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if background_raster_path is not None and Path(background_raster_path).exists():
        _plot_background_raster(ax, Path(background_raster_path))

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


# --- Map (Section 17) -------------------------------------------------------------------


def render_nsta_table_b1_reconciliation_map(
    *,
    events_2018_gdf: gpd.GeoDataFrame,
    nsta_registry_gdf: gpd.GeoDataFrame,
    piggyback_df: pd.DataFrame,
    route: LineString,
    output_path: Path,
    background_raster_path: Path | None = None,
    title: str = "PL854/PL855 — Official Freespan Registry Reconciliation",
    dpi: int = 150,
) -> Path:
    """Table B.1 2018 intervals + NSTA PL854/PL855 freespan geometries, current vs
    removed distinguished by marker style, piggyback coincidence shown as an
    explicit offset pair rather than one geometry hidden under the other."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _setup_route_axes(route, background_raster_path)

    route_gdf = gpd.GeoDataFrame(geometry=[route], crs=events_2018_gdf.crs)
    route_gdf.plot(color="0.75", linewidth=1.5, ax=ax, zorder=1)

    for geom in events_2018_gdf.geometry:
        if geom.geom_type == "Point":
            ax.plot(geom.x, geom.y, marker="o", markersize=7, color="black", zorder=4)
        else:
            xs, ys = geom.xy
            ax.plot(xs, ys, color="black", linewidth=5, solid_capstyle="round", zorder=4)

    piggyback_feature_ids = set()
    if not piggyback_df.empty:
        piggyback_feature_ids = set(piggyback_df["pl854_feature_id"]) | set(
            piggyback_df["pl855_feature_id"]
        )

    for _, row in nsta_registry_gdf.iterrows():
        colour = _PIPELINE_COLORS.get(row["nsta_pipeline_number"], "tab:gray")
        marker = "^" if row["registry_layer"] == "REMOVED_PIPELINE_FREESPANS" else "o"
        is_coincident = row["feature_id"] in piggyback_feature_ids
        # Piggyback-coincident records get a visible ring + a small perpendicular
        # offset so one is never hidden directly under the other.
        point = route.interpolate(row["canonical_mid_chainage_m"])
        offset = 0.0
        if is_coincident:
            offset = 15.0 if row["nsta_pipeline_number"] == "PL854" else -15.0
        ax.plot(
            point.x + offset,
            point.y,
            marker=marker,
            markersize=10 if is_coincident else 8,
            markeredgecolor="black" if is_coincident else colour,
            markeredgewidth=1.5 if is_coincident else 0.0,
            color=colour,
            zorder=6,
        )

    _add_kp_labels(ax, route)
    _add_endpoint_labels(ax, route)
    _add_scale_bar(ax)
    _add_north_arrow(ax)

    # Maximum 3 legend labels (Section 17).
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=4, label="Table B.1 2018 freespan interval"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_PIPELINE_COLORS["PL854"],
            markersize=8,
            label="NSTA PL854 freespan (circle=current, triangle=removed)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_PIPELINE_COLORS["PL855"],
            markersize=8,
            label="NSTA PL855 freespan (circle=current, triangle=removed)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    nsta_count = len(nsta_registry_gdf)
    ax.set_title(
        f"Table B.1: {len(events_2018_gdf)} 2018 event(s) | NSTA PL854/PL855 registry: "
        f"{nsta_count} feature(s)",
        fontsize=9,
        style="italic",
        pad=14,
    )

    footer_lines = [
        "Source scope: Table B.1 is the PIGGYBACKED PL854/PL855 corridor; NSTA "
        "records carry their own NSTAPIPNO attribution independently.",
        NO_MODEL_PREDICTION_STATEMENT,
        "Registry layer (current vs removed) is lifecycle provenance only -- never "
        "read as current survey condition or as proof a freespan physically ended.",
    ]
    ax.text(
        0.0,
        -0.14,
        "\n".join(footer_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.2",
        wrap=True,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Attribution crosswalk figure (Section 18) ------------------------------------------


def render_2018_attribution_crosswalk_figure(
    *,
    attribution_evidence_df: pd.DataFrame,
    match_diagnostics_df: pd.DataFrame,
    output_path: Path,
    title: str = "PL854/PL855 — 2018 Freespan Attribution Crosswalk",
    dpi: int = 150,
) -> Path:
    """One compact, report-ready row per 2018 Table B.1 event (Section 18):
    canonical KP, source length/height, NSTA match status, NSTAPIPNO match,
    registry current/removed, survey/date context."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = attribution_evidence_df.sort_values("canonical_mid_kp").reset_index(drop=True)
    column_labels = [
        "Event",
        "Canonical KP",
        "Length (m)",
        "Height (m)",
        "NSTA match status",
        "NSTAPIPNO match",
        "Registry layer(s)",
    ]
    cell_rows = []
    for _, row in ordered.iterrows():
        candidates = (
            match_diagnostics_df[match_diagnostics_df["table_b1_event_id"] == row["event_id"]]
            if not match_diagnostics_df.empty
            else match_diagnostics_df
        )
        registry_layers = (
            ", ".join(sorted(set(candidates["registry_layer"]))) if not candidates.empty else "n/a"
        )
        cell_rows.append(
            [
                row["event_id"],
                row["canonical_mid_kp"],
                f"{row['source_length_m']:.2f}",
                f"{row['source_height_m']:.2f}",
                row["cross_source_match_status"],
                ", ".join(row["matched_nsta_pipeline_numbers"]) or "none",
                registry_layers,
            ]
        )

    fig_height = 1.2 + 0.45 * len(cell_rows)
    fig, ax = plt.subplots(figsize=(13.0, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_rows,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    table.auto_set_column_width(col=list(range(len(column_labels))))

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.text(
        0.0,
        0.02,
        f"{NO_MODEL_PREDICTION_STATEMENT}\nOriginal Table B.1 individual_line_attribution "
        "field (UNRESOLVED) is preserved unchanged; this table is second-source evidence only.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.2",
        wrap=True,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

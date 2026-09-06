"""Freespan context-audit maps (MAR-015).

Kept self-contained (never importing MAR-012/013/014/MAR-014A/B/C's own map
modules), matching this project's established convention.

Never colour the route by a score (Section 17)
-------------------------------------------------
The primary map uses subtle alternation ONLY to make the 14 hydrodynamic
support-section boundaries visible -- never a susceptibility-score colour
ramp. Observed freespan intervals are always drawn at their TRUE length,
never widened for visibility (Section 12 of MAR-014B's own discipline,
continued here).
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

from marine_engine.validation.freespan_context_audit import (  # noqa: E402
    OBSERVED_SPAN_LENGTH_RANGE_M,
)

NO_SCORE_STATEMENT = (
    "NO FREESPAN SUSCEPTIBILITY SCORE, CLASSIFIER, PROBABILITY, OR MODEL ACCURACY METRIC IS "
    "SHOWN. This is a descriptive evidence audit, not model validation."
)
_SECTION_ALTERNATION_COLORS = ("0.88", "0.94")


# --- Shared self-contained plotting helpers (duplicated by design) ---------------------


def _add_kp_labels(ax, route: LineString) -> None:
    from marine_engine.preprocessing.chainage import format_kp_label

    total_length_m = route.length
    interval_m = 5000.0
    ticks = list(np.arange(0.0, total_length_m, interval_m))
    if not ticks or not np.isclose(ticks[-1], total_length_m):
        ticks.append(total_length_m)
    for chainage_m in ticks:
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


# --- Section 17: primary resolution-audit map -------------------------------------------


def render_freespan_evidence_resolution_audit_map(
    *,
    events_2018_gdf: gpd.GeoDataFrame,
    section_df: pd.DataFrame,
    route: LineString,
    independent_section_count: int,
    output_path: Path,
    background_raster_path: Path | None = None,
    title: str = "PL854/PL855 — Observed Freespans and Model-Support Resolution",
    dpi: int = 150,
) -> Path:
    """Neutral route, TRUE-length 2018 freespan intervals, the 14 hydrodynamic
    support-section boundaries (subtle alternation only, never a score
    colour ramp), event count per section where nonzero, and an explicit
    scale-comparison box (Section 17)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = _content_bounds(route, background_raster_path)
    span_x, span_y = max(maxx - minx, 1.0), max(maxy - miny, 1.0)
    fig_height = 7.5
    fig_width = max(8.0, min(18.0, fig_height * (span_x / span_y)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pad_x, pad_y = span_x * 0.06, span_y * 0.10
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if background_raster_path is not None and Path(background_raster_path).exists():
        _plot_background_raster(ax, Path(background_raster_path))

    from shapely.ops import substring

    for i, (_, section) in enumerate(section_df.iterrows()):
        section_geom = substring(
            route, section["start_chainage_m"], section["end_chainage_m"], normalized=False
        )
        colour = _SECTION_ALTERNATION_COLORS[i % 2]
        ax.plot(*section_geom.xy, color=colour, linewidth=6, solid_capstyle="butt", zorder=1)

    ax.plot(*route.xy, color="0.4", linewidth=1.2, zorder=2)

    for _, row in events_2018_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "Point":
            ax.plot(geom.x, geom.y, marker="o", markersize=7, color="tab:red", zorder=5)
        else:
            xs, ys = geom.xy
            ax.plot(xs, ys, color="tab:red", linewidth=5, solid_capstyle="round", zorder=5)

    for _, section in section_df[section_df["observed_2018_event_count"] > 0].iterrows():
        mid_chainage = (section["start_chainage_m"] + section["end_chainage_m"]) / 2.0
        point = route.interpolate(mid_chainage)
        ax.annotate(
            f"{int(section['observed_2018_event_count'])} event(s)",
            (point.x, point.y),
            textcoords="offset points",
            xytext=(0, 14),
            fontsize=7,
            ha="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "tab:red", "alpha": 0.9},
        )

    _add_kp_labels(ax, route)
    _add_endpoint_labels(ax, route)
    _add_scale_bar(ax)
    _add_north_arrow(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="tab:red",
            linewidth=4,
            label="Official 2018 freespan interval (true length)",
        ),
        Line2D(
            [0],
            [0],
            color="0.4",
            linewidth=1.2,
            label="Canonical route (section shading: support boundaries)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Placed under the legend (upper-left) rather than upper-right, which on this
    # route collides with the "Source geometry terminus" label and the nearby
    # event-count annotation.
    scale_lines = [
        f"Observed span length: ~{OBSERVED_SPAN_LENGTH_RANGE_M[0]:g}-"
        f"{OBSERVED_SPAN_LENGTH_RANGE_M[1]:g} m",
        "Current/wave support: ~1.5-2 km",
        "Morphology source class: ~115 m",
        "Pipeline length: ~23.5 km",
    ]
    ax.text(
        0.0,
        0.80,
        "\n".join(scale_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.4", "alpha": 0.92},
    )

    footer_lines = [
        f"{len(events_2018_gdf)} corridor-level events occupy only {independent_section_count} "
        "independent hydrodynamic support sections.",
        NO_SCORE_STATEMENT,
    ]
    ax.text(
        0.0,
        -0.18,
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


# --- Section 18: model-context small-multiple profile ------------------------------------


def render_feature_context_small_multiple(
    *,
    section_df: pd.DataFrame,
    events_2018_df: pd.DataFrame,
    total_length_m: float,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Panels A-D exactly (Section 18): MAR-012 combined-shear p95 envelope,
    MAR-013 p95 mobility capacity, MAR-007 local relief, and the 2018 observed
    freespan intervals at TRUE width -- no correlation line, no fitted trend."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 10.0), sharex=True)
    ax_shear, ax_mobility, ax_relief, ax_events = axes

    for _, row in section_df.iterrows():
        x = [row["start_chainage_m"] / 1000.0, row["end_chainage_m"] / 1000.0]
        ax_shear.fill_between(
            x,
            [row["tau_max_p95_sensitivity_min_pa"]] * 2,
            [row["tau_max_p95_sensitivity_max_pa"]] * 2,
            color="tab:blue",
            alpha=0.35,
        )
        ax_mobility.plot(
            x,
            [row["largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"]] * 2,
            color="tab:purple",
            linewidth=3,
            solid_capstyle="butt",
        )
        ax_relief.plot(
            x,
            [row["local_relief_1000m_median_m"]] * 2,
            color="tab:brown",
            linewidth=3,
            solid_capstyle="butt",
        )

    ax_shear.set_ylabel("MAR-012 tau_max\np95 sensitivity (Pa)", fontsize=8)
    ax_mobility.set_ylabel("MAR-013 largest\npassing D50 (p95, mm)", fontsize=8)
    ax_relief.set_ylabel("MAR-007 local\nrelief 1000 m (m)", fontsize=8)
    ax_events.set_ylabel("2018 observed\nfreespans", fontsize=8)
    ax_events.set_yticks([])
    ax_events.set_xlabel("Chainage (km)")

    for _, event in events_2018_df.iterrows():
        chainage_min_km = event["canonical_chainage_min_m"] / 1000.0
        chainage_max_km = event["canonical_chainage_max_m"] / 1000.0
        width_km = max(chainage_max_km - chainage_min_km, 0.0)
        ax_events.barh(
            0,
            width=max(width_km, 0.0001),
            left=chainage_min_km,
            height=0.6,
            color="tab:red",
        )
        for ax in (ax_shear, ax_mobility, ax_relief):
            ax.axvline(
                (chainage_min_km + chainage_max_km) / 2.0,
                color="tab:red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
            )

    for ax in axes:
        ax.set_xlim(0, total_length_m / 1000.0)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "PL854 — Model/Context Feature Audit vs. Observed 2018 Freespans (no fused score)",
        fontsize=12,
        fontweight="bold",
    )
    ax_shear.set_title(
        "Panels: (A) combined bed shear, (B) mobility capacity, (C) local relief, (D) observed "
        "freespans (true width). No correlation line, no fitted trend.",
        fontsize=8,
        style="italic",
    )

    fig.text(
        0.0, -0.02, NO_SCORE_STATEMENT, ha="left", va="top", fontsize=8, color="0.2", wrap=True
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 19: descriptive audit-table figure ------------------------------------------


def render_feature_audit_table(
    *,
    contrasts_by_key: dict[str, dict[str, Any]],
    categorical_audits_by_key: dict[str, dict[str, Any]],
    readiness: dict[str, Any],
    feature_keys: tuple[str, ...],
    output_path: Path,
    title: str = "PL854/PL855 — Freespan Feature Evidence Audit",
    dpi: int = 150,
) -> Path:
    """One row per audited feature family (Section 19) -- spatial support,
    temporal relationship to 2018, route distinct values, event-section
    context, key limitation. Neutral text formatting only, never a
    green/red traffic-light score."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    column_labels = [
        "Feature",
        "Spatial support",
        "Temporal relationship to 2018",
        "Route distinct values",
        "Event-section context",
        "Key limitation",
    ]
    cell_rows = []
    for key in feature_keys:
        readiness_entry = readiness["features"][key]
        if key in categorical_audits_by_key:
            audit = categorical_audits_by_key[key]
            distinct_values_joined = ", ".join(audit["route_distinct_values"])
            distinct_text = f"{audit['route_distinct_value_count']} ({distinct_values_joined})"
            event_context = str(audit["event_containing_sections_per_value"])
        else:
            contrast = contrasts_by_key[key]
            distinct_text = "continuous"
            event_context = (
                f"median={contrast['event_section_median']:.4g}"
                if contrast["event_section_median"] is not None
                else "n/a"
            )
        support_text = (
            readiness_entry["spatial_support_note"]
            if readiness_entry["spatial_support_m"] is None
            else (
                f"~{readiness_entry['spatial_support_m']:.0f} m "
                f"({readiness_entry['spatial_support_note']})"
            )
        )
        cell_rows.append(
            [
                key,
                support_text,
                readiness_entry["temporal_status"],
                distinct_text,
                event_context,
                readiness_entry["known_major_limitation"],
            ]
        )

    fig_height = 1.4 + 0.55 * len(cell_rows)
    fig, ax = plt.subplots(figsize=(15.0, fig_height))
    ax.axis("off")

    table = ax.table(cellText=cell_rows, colLabels=column_labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.8)
    table.auto_set_column_width(col=list(range(len(column_labels))))

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.text(
        0.0, 0.01, NO_SCORE_STATEMENT, ha="left", va="bottom", fontsize=7, color="0.2", wrap=True
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

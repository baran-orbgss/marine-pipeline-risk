"""Generic free-span support map, KP/route view, and the real NSTA UKCS-wide overview map
(MAR-025 Sections 23-24, 26).

Self-contained, zero dependency on any other map module (this project's established
map-module-isolation convention). Primary map variable is always the real physical clearance or
a support/support-loss state -- never a risk colour or score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import substring as shapely_substring

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px


def _add_scale_bar(ax) -> None:
    xlim = ax.get_xlim()
    view_width_m = abs(xlim[1] - xlim[0])
    bar_km = 0.05
    for candidate_km in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50):
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


# --- Section 23: generic free-span support map ------------------------------------------------


def render_free_span_support_map(
    *,
    route: LineString,
    profile_df: pd.DataFrame,
    measured_intervals_df: pd.DataFrame,
    scenario_intervals_df: pd.DataFrame | None,
    output_path: Path,
    value_label: str = "Pipe underside clearance (m)",
    title: str = "Generic Pipeline Free-Span Support Map",
    subtitle: str = "Measured clearance shown exactly as computed; scenario-created / "
    "scenario-extended intervals shown independently -- no risk colouring",
    dpi: int = 150,
) -> Path:
    """Section 23: route, measured support state (coloured by clearance_m), observed
    unsupported intervals, and scenario-created/-extended support-loss intervals, all shown
    independently. Primary variable is always the real physical clearance."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = route.bounds
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, span_x * 0.15)
    fig_height = 6.0
    fig_width = max(8.0, min(16.0, fig_height * (span_x / span_y)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pad_x, pad_y = span_x * 0.06, span_y * 0.4
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.plot(*route.xy, color="0.6", linewidth=2.0, zorder=1, label="Route")

    if not profile_df.empty:
        sc = ax.scatter(
            profile_df["x_m"],
            profile_df["y_m"],
            c=profile_df["clearance_m"],
            cmap="RdYlGn_r",
            vmin=-abs(profile_df["clearance_m"]).max() if len(profile_df) else -1.0,
            vmax=abs(profile_df["clearance_m"]).max() if len(profile_df) else 1.0,
            s=18,
            zorder=3,
            label="Measured sample",
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(value_label, fontsize=8)

    for _, interval in measured_intervals_df.iterrows():
        seg = shapely_substring(route, interval["start_chainage_m"], interval["end_chainage_m"])
        xs, ys = seg.xy
        ax.plot(xs, ys, color="#8b0000", linewidth=5, alpha=0.5, zorder=2, solid_capstyle="butt")

    if scenario_intervals_df is not None and not scenario_intervals_df.empty:
        for _, interval in scenario_intervals_df.iterrows():
            seg = shapely_substring(route, interval["start_chainage_m"], interval["end_chainage_m"])
            xs, ys = seg.xy
            ax.plot(
                xs,
                ys,
                color="#ff8c00",
                linewidth=2.5,
                linestyle=(0, (3, 2)),
                zorder=4,
                solid_capstyle="butt",
            )

    legend_handles = [
        Line2D([0], [0], color="0.6", linewidth=2.0, label="Route"),
        Line2D(
            [0],
            [0],
            color="#8b0000",
            linewidth=5,
            alpha=0.5,
            label="Observed/measured unsupported interval",
        ),
        Line2D(
            [0],
            [0],
            color="#ff8c00",
            linewidth=2.5,
            linestyle=(0, (3, 2)),
            label="Scenario support-loss interval",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.9)
    _add_scale_bar(ax)
    _add_north_arrow(ax)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=8, style="italic", pad=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 24: KP / route view -----------------------------------------------------------------


def render_free_span_kp_view(
    *,
    profile_df: pd.DataFrame,
    measured_intervals_df: pd.DataFrame,
    scenario_intervals_df: pd.DataFrame | None,
    classification_threshold_m: float | None,
    seabed_lowering_m: float | None,
    output_path: Path,
    title: str = "Generic Pipeline Free-Span KP View",
    dpi: int = 150,
) -> Path:
    """Section 24: six bands aligned on one chainage axis -- pipe bottom elevation, seabed
    support elevation, measured underside clearance, observed unsupported intervals, scenario
    seabed lowering, scenario support-loss intervals."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 1, figsize=(12.0, 14.0), sharex=True)
    kp_km = profile_df["chainage_m"] / 1000.0 if not profile_df.empty else pd.Series(dtype=float)

    ax = axes[0]
    ax.plot(
        kp_km, profile_df.get("pipe_bottom_elevation_m"), color="tab:blue", marker=".", markersize=3
    )
    ax.set_ylabel("Pipe bottom\nelevation (m)", fontsize=8)
    ax.set_title("1. Pipe bottom elevation", fontsize=9, loc="left")

    ax = axes[1]
    ax.plot(
        kp_km,
        profile_df.get("seabed_support_elevation_m"),
        color="tab:brown",
        marker=".",
        markersize=3,
    )
    ax.set_ylabel("Seabed support\nelevation (m)", fontsize=8)
    ax.set_title("2. Seabed support elevation", fontsize=9, loc="left")

    ax = axes[2]
    ax.plot(kp_km, profile_df.get("clearance_m"), color="tab:purple", marker=".", markersize=3)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    if classification_threshold_m is not None:
        ax.axhspan(-classification_threshold_m, classification_threshold_m, color="0.85", zorder=0)
    ax.set_ylabel("Underside\nclearance (m)", fontsize=8)
    ax.set_title(
        "3. Measured underside clearance (shaded band = classification threshold)",
        fontsize=9,
        loc="left",
    )

    ax = axes[3]
    for _, interval in measured_intervals_df.iterrows():
        ax.axvspan(
            interval["start_chainage_m"] / 1000.0,
            interval["end_chainage_m"] / 1000.0,
            color="#8b0000",
            alpha=0.6,
        )
    ax.set_ylabel("Observed\nunsupported", fontsize=8)
    ax.set_yticks([])
    ax.set_title("4. Observed/measured unsupported intervals", fontsize=9, loc="left")

    ax = axes[4]
    if seabed_lowering_m is not None:
        ax.axhline(seabed_lowering_m, color="tab:orange", linewidth=1.5)
        ax.text(
            0.01,
            0.7,
            f"scenario seabed_lowering_m = {seabed_lowering_m:g} m",
            transform=ax.transAxes,
            fontsize=8,
        )
    else:
        ax.text(
            0.01, 0.5, "NO_DEFENSIBLE_SEABED_LOWERING_INPUT", transform=ax.transAxes, fontsize=8
        )
    ax.set_ylabel("Scenario\nlowering (m)", fontsize=8)
    ax.set_title("5. Scenario seabed lowering", fontsize=9, loc="left")

    ax = axes[5]
    if scenario_intervals_df is not None:
        for _, interval in scenario_intervals_df.iterrows():
            ax.axvspan(
                interval["start_chainage_m"] / 1000.0,
                interval["end_chainage_m"] / 1000.0,
                color="#ff8c00",
                alpha=0.6,
            )
    ax.set_ylabel("Scenario\nsupport-loss", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("KP (km)")
    ax.set_title("6. Scenario support-loss intervals", fontsize=9, loc="left")

    for a in axes:
        a.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 26: real NSTA UKCS-wide observed evidence map --------------------------------------

# A simple equal-area-ish metric CRS spanning the whole UK/North Sea shelf, used only to compute
# a representative centroid for each (possibly LineString) record before plotting in lon/lat --
# never used for the plot's own axes, so the exact projection choice is not scientifically load-
# bearing here, only the accuracy of the centroid.
_UK_METRIC_CRS_FOR_CENTROIDS = "EPSG:27700"


def _lonlat_centroids(gdf) -> tuple[Any, Any]:
    """Representative (longitude, latitude) for every geometry in `gdf`, computed by
    reprojecting to a metric CRS first -- avoids the (accuracy, not correctness) warning that
    computing `.centroid` directly in a geographic CRS would raise."""

    projected = gdf.geometry.to_crs(_UK_METRIC_CRS_FOR_CENTROIDS).centroid
    lonlat = projected.to_crs("EPSG:4326")
    return lonlat.x, lonlat.y


def render_nsta_registry_overview_map(
    *,
    registry_gdf,
    example_pipeline_number: str | None,
    example_record_count: int,
    output_path: Path,
    title: str = "UKCS Observed Pipeline Free-Span Evidence (NSTA Registry)",
    dpi: int = 150,
) -> Path:
    """Section 26: an observed-evidence-only map (no risk colouring). Because the full UKCS
    registry spans 222+ distinct pipelines and is unreadable as one detailed map, this renders
    an overview (every real record, by longitude/latitude) plus one data-driven high-record-
    count pipeline example -- the example is selected purely by registry record count and is
    explicitly NOT an indication of highest risk (Section 26)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_overview, ax_example) = plt.subplots(1, 2, figsize=(14.0, 6.5))

    if len(registry_gdf) and registry_gdf.geometry.notna().any():
        lon, lat = _lonlat_centroids(registry_gdf)
        ax_overview.scatter(lon, lat, s=4, color="tab:blue", alpha=0.5)
    ax_overview.set_title(
        f"Overview -- {len(registry_gdf)} real record(s), both registries", fontsize=9
    )
    ax_overview.set_xlabel("Longitude")
    ax_overview.set_ylabel("Latitude")
    ax_overview.grid(True, alpha=0.3)

    if example_pipeline_number is not None:
        example_gdf = registry_gdf[registry_gdf["nsta_pipeline_number"] == example_pipeline_number]
        if len(example_gdf) and example_gdf.geometry.notna().any():
            example_lon, example_lat = _lonlat_centroids(example_gdf)
            ax_example.scatter(example_lon, example_lat, s=18, color="#8b0000")
        ax_example.set_title(
            f"Example: {example_pipeline_number} ({example_record_count} record(s) -- "
            "selected by record count only, not a risk indication)",
            fontsize=8,
        )
    else:
        ax_example.set_title("No example pipeline available (empty registry)", fontsize=9)
    ax_example.set_xlabel("Longitude")
    ax_example.set_ylabel("Latitude")
    ax_example.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    footer = (
        "Source: NSTA UKCS offshore infrastructure pipeline freespans (current + removed "
        "registries). Source-reported observed free-span evidence only -- not a susceptibility "
        "or structural failure indication."
    )
    fig.text(0.01, 0.01, footer, fontsize=7, color="0.2")

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

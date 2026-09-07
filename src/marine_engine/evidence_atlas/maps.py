"""PL854 engineering evidence atlas figures (MAR-018): map first, report second.

Self-contained by design (mirrors `combined_bed_shear_map.py`'s own stated
convention): small cartographic helpers (scale bar, north arrow, background
raster) are re-implemented locally rather than imported from
`noncohesive_mobility_map.py`/`scour_onset_map.py`, so those already-shipped
map modules are never put at risk by a change made for this ticket. The two
small classification ladders below mirror (never import) the canonical
`TESTED_D50_SCENARIOS_MM` (MAR-013) / `TESTED_EMBEDMENT_RATIOS` (MAR-014)
constants for the same reason.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import pandas as pd
from shapely.geometry import LineString

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede pyplot.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Mirrors MAR-013's TESTED_D50_SCENARIOS_MM / MAR-014's TESTED_EMBEDMENT_RATIOS -- never
# imported, per this project's map-module-isolation convention (see module docstring).
_MOBILITY_D50_LADDER_MM = (0.063, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
_EMBEDMENT_RATIO_LADDER = (0.0, 0.03, 0.06, 0.10, 0.15)

_KP_REFERENCE_INTERVAL_LABEL = "KP references: 0 / 5 / 10 / 15 / 20 / terminus"

# Non-directional corridor naming (MAR-019 Section 5): canonical geometry direction is not
# independently verified as physical installation direction (Anglia A vs. LOGGS), so this
# must never read as an asserted flow/installation direction like "Anglia A -> LOGGS".
_ROUTE_CORRIDOR_LABEL = "Anglia A-LOGGS corridor"

# --- Shared cartographic helpers (mirrors combined_bed_shear_map.py's style) --------------


def _plot_background_raster(ax, raster_path: Path) -> None:
    """Muted grayscale EMODnet bathymetry context, best-effort only."""

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


def _content_bounds(
    route: LineString, background_raster_path: Path | None
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


def _setup_axes(ax, minx: float, miny: float, maxx: float, maxy: float) -> None:
    span_x, span_y = max(maxx - minx, 1.0), max(maxy - miny, 1.0)
    pad_x, pad_y = span_x * 0.06, span_y * 0.10
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


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
    ax.add_line(Line2D([x0, x0 + bar_m], [y0, y0], color="black", linewidth=2.4))
    ax.annotate(
        f"{bar_km:g} km",
        (x0 + bar_m / 2.0, y0),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=9,
    )


def _add_north_arrow(ax) -> None:
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = xlim[1] - abs(xlim[1] - xlim[0]) * 0.06
    y0 = ylim[0] + abs(ylim[1] - ylim[0]) * 0.08
    y1 = y0 + abs(ylim[1] - ylim[0]) * 0.07
    ax.annotate(
        "N",
        xy=(x, y1),
        xytext=(x, y0),
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.6},
        ha="center",
        fontsize=10,
        fontweight="bold",
    )


def _add_kp_reference_labels(ax, chainage_reference_gdf: gpd.GeoDataFrame) -> None:
    """Dot + label at exactly the 6 canonical KP reference points (Section 10) --
    never the full 941-station chainage grid, which would clutter the atlas."""

    for _, row in chainage_reference_gdf.iterrows():
        point = row.geometry
        ax.plot(point.x, point.y, marker="o", markersize=4, color="black", zorder=6)
        ax.annotate(
            row["reference_label"],
            (point.x, point.y),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=7.5,
        )


def _add_panel_caption(ax, text: str) -> None:
    """Type-descriptor caption (OBSERVED/MODELLED/...) placed INSIDE the axes,
    top-left -- never above the axes box, which collides with `ax.set_title`
    once panels are short."""

    ax.text(
        0.01,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        color="0.15",
        va="top",
        ha="left",
        zorder=8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.82},
    )


def _discrete_cmap(ladder: tuple[float, ...], name: str = "viridis"):
    values = sorted(ladder)
    boundaries = [values[0] - (values[1] - values[0]) / 2.0]
    boundaries += [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
    boundaries.append(values[-1] + (values[-1] - values[-2]) / 2.0)
    cmap = plt.get_cmap(name, len(values))
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, values


def _plot_freespan_reference_marks(ax, freespans_2018_gdf: gpd.GeoDataFrame, *, halo: bool) -> None:
    """The TRUE route-substring geometry, plus (if `halo`) a visibility halo --
    the true geometry is always drawn; the halo never replaces it (Section 8)."""

    if freespans_2018_gdf is None or freespans_2018_gdf.empty:
        return
    freespans_2018_gdf.plot(ax=ax, color="#B4131A", linewidth=3.0, zorder=7)
    if halo:
        for _, row in freespans_2018_gdf.iterrows():
            centroid = row.geometry.centroid
            ax.plot(
                centroid.x,
                centroid.y,
                marker="o",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor="#B4131A",
                markeredgewidth=1.2,
                zorder=6,
            )


def _annotate_event_sections(ax, sections_gdf: gpd.GeoDataFrame) -> None:
    """One readable annotation per event-containing section (never per tiny
    individual event), stacked so nearby event sections do not collide."""

    event_sections = sections_gdf[
        sections_gdf["observed_2018_freespan_count"].fillna(0) > 0
    ].sort_values("segment_id")
    for rank, (_, row) in enumerate(event_sections.iterrows()):
        centroid = row.geometry.centroid
        label = (
            f"{int(row['observed_2018_freespan_count'])} event(s)\n"
            f"{row['observed_2018_freespan_total_length_m']:.1f} m total"
        )
        ax.annotate(
            label,
            (centroid.x, centroid.y),
            textcoords="offset points",
            xytext=(0, 22 + rank * 28),
            fontsize=8,
            ha="center",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#B4131A", "alpha": 0.9},
        )


# --- Primary deliverable: 4-panel engineering evidence atlas (Section 9-12) ----------------


def _panel_a_observed_condition(
    ax,
    *,
    route: LineString,
    sections_gdf: gpd.GeoDataFrame,
    freespans_2018_gdf: gpd.GeoDataFrame,
    historical_freespans_gdf: gpd.GeoDataFrame,
    chainage_reference_gdf: gpd.GeoDataFrame,
) -> None:
    ax.plot(*route.xy, color="0.25", linewidth=1.6, zorder=1)
    non_2018 = historical_freespans_gdf[historical_freespans_gdf["survey_year"] != 2018]
    if not non_2018.empty:
        non_2018.plot(ax=ax, color="0.6", linewidth=1.5, zorder=2)
    _plot_freespan_reference_marks(ax, freespans_2018_gdf, halo=True)
    _annotate_event_sections(ax, sections_gdf)
    _add_kp_reference_labels(ax, chainage_reference_gdf)
    ax.set_title("A. Observed Condition", fontsize=14, fontweight="bold")
    _add_panel_caption(
        ax, "OBSERVED -- official 2018 corridor freespans (bold red); 2012/2014 context (grey)"
    )


def _panel_b_hydrodynamic_forcing(
    ax,
    *,
    sections_gdf: gpd.GeoDataFrame,
    freespans_2018_gdf: gpd.GeoDataFrame,
    chainage_reference_gdf: gpd.GeoDataFrame,
) -> None:
    has_value = sections_gdf["combined_tau_max_p95_upper_pa"].notna().any()
    if has_value:
        sections_gdf.plot(
            column="combined_tau_max_p95_upper_pa",
            cmap="inferno",
            linewidth=5,
            legend=True,
            legend_kwds={"label": "Combined bed shear p95 upper bound (Pa)", "shrink": 0.6},
            ax=ax,
        )
        cbar = ax.get_figure().axes[-1]
        cbar.tick_params(labelsize=9)
        cbar.yaxis.label.set_fontsize(9.5)
    else:
        sections_gdf.plot(color="0.4", linewidth=5, ax=ax)
    _plot_freespan_reference_marks(ax, freespans_2018_gdf, halo=False)
    _add_kp_reference_labels(ax, chainage_reference_gdf)
    ax.set_title("B. Hydrodynamic Bed Forcing", fontsize=14, fontweight="bold")
    _add_panel_caption(
        ax,
        "MODELLED -- MAR-012 p95 sensitivity upper bound; 2018 events are reference marks only, "
        "not validation of 2024-2026 forcing",
    )


def _panel_c_mobility_capacity(
    ax,
    *,
    sections_gdf: gpd.GeoDataFrame,
    freespans_2018_gdf: gpd.GeoDataFrame,
    chainage_reference_gdf: gpd.GeoDataFrame,
) -> None:
    cmap, norm, ladder_values = _discrete_cmap(_MOBILITY_D50_LADDER_MM)
    has_value = sections_gdf["mobility_capacity_p95_d50_mm"].notna().any()
    if has_value:
        colours = [cmap(norm(v)) for v in sections_gdf["mobility_capacity_p95_d50_mm"]]
        sections_gdf.plot(ax=ax, color=colours, linewidth=5)
        present = sorted(sections_gdf["mobility_capacity_p95_d50_mm"].dropna().unique())
        legend_handles = [
            Patch(facecolor=cmap(norm(v)), edgecolor=cmap(norm(v)), label=f"{v:g} mm")
            for v in present
        ]
        ax.legend(
            handles=legend_handles,
            title="p95 mobility capacity (largest passing D50)",
            fontsize=8,
            title_fontsize=8.5,
            loc="upper right",
        )
    else:
        sections_gdf.plot(color="0.4", linewidth=5, ax=ax)
    _plot_freespan_reference_marks(ax, freespans_2018_gdf, halo=False)
    _add_kp_reference_labels(ax, chainage_reference_gdf)
    ax.set_title("C. Noncohesive Sediment Mobility Capacity", fontsize=14, fontweight="bold")
    _add_panel_caption(
        ax,
        f"MODELLED -- discrete scale, ladder {ladder_values[0]:g}-{ladder_values[-1]:g} mm "
        "(MAR-013)",
    )


def _panel_d_evidence_support(
    ax,
    *,
    sections_gdf: gpd.GeoDataFrame,
    highres_survey_gdf: gpd.GeoDataFrame,
    psa_points_gdf: gpd.GeoDataFrame,
    chainage_reference_gdf: gpd.GeoDataFrame,
) -> None:
    # Route/support-section structure is the visually PRIMARY layer in this panel (Section 7):
    # drawn thicker/darker than the old style, and AFTER the survey hatch (never before), so it
    # always reads on top rather than being visually competed with by the data-availability
    # overlay.
    if highres_survey_gdf is not None and not highres_survey_gdf.empty:
        highres_survey_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#8B7BC7",
            hatch="/",
            linewidth=0.5,
            alpha=0.3,
            zorder=1,
        )
    sections_gdf.plot(ax=ax, color="0.55", linewidth=4, zorder=3)
    for _, row in sections_gdf.iterrows():
        boundary_point = row.geometry.interpolate(0.0)
        ax.plot(
            boundary_point.x, boundary_point.y, marker="|", color="0.25", markersize=10, zorder=4
        )
    if psa_points_gdf is not None and not psa_points_gdf.empty:
        psa_points_gdf.plot(ax=ax, color="#1C7C54", marker="^", markersize=34, zorder=5)

    _add_kp_reference_labels(ax, chainage_reference_gdf)
    ax.set_title("D. Evidence / Data Support", fontsize=14, fontweight="bold")
    survey_note = (
        f"{len(highres_survey_gdf)} route-area survey record(s), ALL metadata-only "
        "(no verified open high-resolution grid)"
        if highres_survey_gdf is not None and not highres_survey_gdf.empty
        else "KNOWN DATA GAP -- no verified open high-resolution route-specific grid"
    )
    _add_panel_caption(
        ax, f"DATA AVAILABILITY -- {survey_note}; triangles are observed PSA D50 points"
    )
    legend_handles = [
        Line2D([0], [0], color="0.55", linewidth=4, label="Support-section structure"),
        Patch(
            facecolor="none",
            edgecolor="#8B7BC7",
            hatch="/",
            alpha=0.5,
            label="High-res survey (metadata only)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#1C7C54",
            markersize=9,
            label="Observed PSA D50",
        ),
        Line2D([0], [0], marker="|", color="0.25", markersize=10, label="Support-section boundary"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")


def render_engineering_evidence_atlas(
    *,
    route: LineString,
    sections_gdf: gpd.GeoDataFrame,
    freespans_2018_gdf: gpd.GeoDataFrame,
    historical_freespans_gdf: gpd.GeoDataFrame,
    psa_points_gdf: gpd.GeoDataFrame,
    highres_survey_gdf: gpd.GeoDataFrame,
    chainage_reference_gdf: gpd.GeoDataFrame,
    observed_condition_summary: dict[str, Any],
    key_limitations: tuple[str, ...],
    output_path: Path,
    background_raster_path: Path | None = None,
    dpi: int = 150,
) -> Path:
    """The PRIMARY MAR-018 deliverable: 4 aligned geographic panels sharing one
    extent/legend key, plus an observed-condition summary box (Section 11) and
    a key-limitations box (Section 12). No fused score anywhere."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = _content_bounds(route, background_raster_path)

    # A 2x2 landscape grid, but sized from the REAL content aspect ratio (route +
    # background raster, padded) rather than a guessed height -- `ax.set_aspect
    # ("equal")` always shrinks a mismatched box to the data's own aspect, so an
    # arbitrarily-chosen row height just moves the wasted space around (verified
    # empirically: a too-tall row letterboxes the panel horizontally instead of
    # eliminating the gap). Deriving panel height from panel width and the true
    # content aspect makes the box matplotlib is given already the right shape.
    span_x, span_y = max(maxx - minx, 1.0), max(maxy - miny, 1.0)
    pad_x, pad_y = span_x * 0.06, span_y * 0.10
    content_aspect = (span_x + 2 * pad_x) / (span_y + 2 * pad_y)

    fig_width = 21.0
    col_width_in = fig_width * 0.96 / 2.0
    panel_height_in = col_width_in / content_aspect
    text_row_height_in = 3.3
    title_budget_in = 1.4
    fig_height = title_budget_in + 2 * panel_height_in + 0.5 + text_row_height_in

    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[panel_height_in, panel_height_in, text_row_height_in],
        hspace=0.4,
        wspace=0.08,
        top=1.0 - title_budget_in / fig_height,
        bottom=0.02,
        left=0.02,
        right=0.98,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    axes += [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    for ax in axes:
        if background_raster_path is not None and Path(background_raster_path).exists():
            _plot_background_raster(ax, Path(background_raster_path))
        _setup_axes(ax, minx, miny, maxx, maxy)
        _add_scale_bar(ax)
        _add_north_arrow(ax)

    _panel_a_observed_condition(
        axes[0],
        route=route,
        sections_gdf=sections_gdf,
        freespans_2018_gdf=freespans_2018_gdf,
        historical_freespans_gdf=historical_freespans_gdf,
        chainage_reference_gdf=chainage_reference_gdf,
    )
    _panel_b_hydrodynamic_forcing(
        axes[1],
        sections_gdf=sections_gdf,
        freespans_2018_gdf=freespans_2018_gdf,
        chainage_reference_gdf=chainage_reference_gdf,
    )
    _panel_c_mobility_capacity(
        axes[2],
        sections_gdf=sections_gdf,
        freespans_2018_gdf=freespans_2018_gdf,
        chainage_reference_gdf=chainage_reference_gdf,
    )
    _panel_d_evidence_support(
        axes[3],
        sections_gdf=sections_gdf,
        highres_survey_gdf=highres_survey_gdf,
        psa_points_gdf=psa_points_gdf,
        chainage_reference_gdf=chainage_reference_gdf,
    )

    summary_ax = fig.add_subplot(grid[2, 0])
    summary_ax.axis("off")
    summary_lines = [
        "Official 2018 PL854/PL855 corridor condition",
        f"  {observed_condition_summary['event_count']} freespan events, "
        f"{observed_condition_summary['total_length_m']:.2f} m exact tabulated total",
        f"  Maximum tabulated span length: {observed_condition_summary['max_length_m']:.2f} m",
        f"  Maximum tabulated span height: {observed_condition_summary['max_height_m']:.2f} m",
        f"  {observed_condition_summary['exposed_section_count']} exposed sections, "
        f"{observed_condition_summary['total_exposed_length_m']:.0f} m total exposed pipeline",
        "",
        "Exposure: aggregate corridor evidence only; spatial locations unavailable.",
        "Freespans: spatially tabulated; PL854 vs PL855 attribution unresolved.",
    ]
    summary_ax.text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        transform=summary_ax.transAxes,
        fontsize=11.5,
        va="top",
        ha="left",
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.6", "fc": "#F5F2E9", "ec": "0.4"},
    )

    limitations_ax = fig.add_subplot(grid[2, 1])
    limitations_ax.axis("off")
    limitations_text = "Key evidence limits\n" + "\n".join(
        f"  - {item}" for item in key_limitations
    )
    limitations_ax.text(
        0.02,
        0.95,
        limitations_text,
        transform=limitations_ax.transAxes,
        fontsize=10.5,
        va="top",
        ha="left",
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.6", "fc": "#FBEEEE", "ec": "0.4"},
    )

    fig.suptitle(
        "PL854 Engineering Evidence Atlas",
        fontsize=21,
        fontweight="bold",
        y=1.0 - 0.28 * title_budget_in / fig_height,
    )
    fig.text(
        0.5,
        1.0 - 0.66 * title_budget_in / fig_height,
        f"{_ROUTE_CORRIDOR_LABEL}, Southern North Sea | {_KP_REFERENCE_INTERVAL_LABEL} | "
        "MAP FIRST, REPORT SECOND -- no fused risk/susceptibility score anywhere in this atlas",
        ha="center",
        fontsize=11.5,
        style="italic",
        color="0.25",
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Evidence strip (Section 13) -----------------------------------------------------------


def _kp_primary_ticks(total_length_m: float) -> tuple[list[float], list[str]]:
    """KP-formatted ticks at 0/5/10/15/20 km plus the exact route terminus
    (Section 9) -- the terminus is always derived from the real route length,
    never hard-coded to "23.48"."""

    round_ticks_km = [0, 5, 10, 15, 20]
    ticks_m = [km * 1000.0 for km in round_ticks_km if km * 1000.0 < total_length_m]
    labels = [f"KP {km:g}" for km in round_ticks_km if km * 1000.0 < total_length_m]
    ticks_m.append(total_length_m)
    labels.append(f"KP {total_length_m / 1000.0:.2f}")
    return ticks_m, labels


def render_evidence_strip(
    *,
    section_df: pd.DataFrame,
    freespans_2018_gdf: gpd.GeoDataFrame,
    output_path: Path,
    folk_class_descriptions: dict[str, str] | None = None,
    dpi: int = 150,
) -> Path:
    """6 aligned chainage bands for same-route evidence COMPARISON -- never a
    fused score, correlation, or fitted trend (Section 13). KP is the primary
    horizontal reference (Section 9); chainage metres remain a secondary,
    supporting reference on a light top axis."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    folk_class_descriptions = folk_class_descriptions or {}
    ordered = section_df.sort_values("start_chainage_m").reset_index(drop=True)
    total_length_m = float(ordered["end_chainage_m"].max())

    embedment_classes = sorted(ordered["p95_required_embedment_upper_class"].dropna().unique())
    embedment_uniform = len(embedment_classes) == 1

    height_ratios = [0.9, 1.3, 1.3, 0.45 if embedment_uniform else 1.1, 1.3, 1.15]
    fig = plt.figure(figsize=(15, 11.5))
    grid = fig.add_gridspec(6, 1, height_ratios=height_ratios, hspace=0.22, top=0.92, bottom=0.08)
    axes = [fig.add_subplot(grid[0, 0])]
    axes += [fig.add_subplot(grid[i, 0], sharex=axes[0]) for i in range(1, 6)]

    # Band 1: official 2018 observed freespan intervals, TRUE width, plus an
    # independent visibility marker for screen readability (Section 10) -- the marker
    # is placed via the axes' x-data/y-axes-fraction transform so it never implies a
    # y-value, and never substitutes for the true-width span itself.
    ax = axes[0]
    if freespans_2018_gdf is not None and not freespans_2018_gdf.empty:
        for _, row in freespans_2018_gdf.iterrows():
            ax.axvspan(
                row["canonical_chainage_min_m"], row["canonical_chainage_max_m"], color="#B4131A"
            )
            mid_chainage_m = (
                row["canonical_chainage_min_m"] + row["canonical_chainage_max_m"]
            ) / 2.0
            ax.plot(
                mid_chainage_m,
                0.78,
                marker="v",
                color="#B4131A",
                markersize=7,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                zorder=5,
            )
    ax.set_ylabel("2018\nfreespans\n(true width)", fontsize=8, rotation=0, ha="right", va="center")
    ax.set_yticks([])
    ax.set_title(
        "PL854 Engineering Evidence Strip -- same-route comparison, not prediction",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.995,
        0.05,
        "bar width = true tabulated spatial interval; marker (▽) = visibility aid only",
        transform=ax.transAxes,
        fontsize=6.5,
        style="italic",
        color="0.3",
        ha="right",
        va="bottom",
    )
    secondary_ax = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    secondary_ax.set_xlabel("Chainage (m) -- secondary reference", fontsize=7.5, color="0.4")
    secondary_ax.tick_params(labelsize=7, colors="0.4")

    # Band 2: combined bed-shear p95 sensitivity envelope.
    ax = axes[1]
    for _, row in ordered.iterrows():
        ax.fill_between(
            [row["start_chainage_m"], row["end_chainage_m"]],
            row["combined_tau_max_p95_lower_pa"],
            row["combined_tau_max_p95_upper_pa"],
            step="post",
            color="#C1440E",
            alpha=0.6,
        )
    ax.set_ylabel("Combined\nshear p95\n(Pa)", fontsize=8, rotation=0, ha="right", va="center")

    # Band 3: MAR-013 mobility capacity.
    ax = axes[2]
    ax.step(
        ordered["start_chainage_m"],
        ordered["mobility_capacity_p95_d50_mm"],
        where="post",
        color="#2E6F40",
    )
    ax.set_ylabel("Mobility\ncapacity\n(mm)", fontsize=8, rotation=0, ha="right", va="center")

    # Band 4: MAR-014 scour-onset screening class -- a slim categorical band when the
    # real value is spatially uniform (Section 11), never labelled "required"/"safe"/
    # "design" burial; falls back to the full line-plot rendering if a future dataset
    # ever shows real spatial variation, so this stays correct rather than hard-coded to
    # today's uniform finding.
    ax = axes[3]
    if embedment_uniform:
        value = embedment_classes[0]
        ax.set_facecolor("#EDE7F6")
        ax.text(
            0.5,
            0.5,
            f"Scour-onset screening: {value}D throughout current support",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#3D2B7A",
        )
        ax.set_yticks([])
    else:
        embedment_values = pd.to_numeric(
            ordered["p95_required_embedment_upper_class"], errors="coerce"
        )
        ax.step(ordered["start_chainage_m"], embedment_values, where="post", color="#5A3EBD")
    ax.set_ylabel("Scour-onset\nscreening\n(xD)", fontsize=8, rotation=0, ha="right", va="center")

    # Band 5: local relief (1000 m radius) morphology context.
    ax = axes[4]
    ax.step(
        ordered["start_chainage_m"],
        ordered["local_relief_1000m_median_m"],
        where="post",
        color="0.35",
    )
    ax.set_ylabel("Local\nrelief\n(m)", fontsize=8, rotation=0, ha="right", va="center")

    # Band 6: mapped 1:250k Folk class -- legend descriptions come ONLY from the
    # already-existing BGS free-text label (never an invented meaning, Section 12).
    ax = axes[5]
    folk_classes = sorted(ordered["mapped_250k_folk_class"].dropna().unique())
    folk_colours = {cls: plt.get_cmap("tab10")(i) for i, cls in enumerate(folk_classes)}
    for _, row in ordered.iterrows():
        ax.axvspan(
            row["start_chainage_m"],
            row["end_chainage_m"],
            color=folk_colours.get(row["mapped_250k_folk_class"], "0.8"),
        )
    ax.set_yticks([])
    ax.set_ylabel(
        "Folk class\n(BGS 1:250k\nregional)", fontsize=8, rotation=0, ha="right", va="center"
    )
    handles = [
        Patch(
            facecolor=colour,
            label=(
                f"{cls} — {folk_class_descriptions[cls].lower()}"
                if cls in folk_class_descriptions
                else cls
            ),
        )
        for cls, colour in folk_colours.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.75),
        ncol=len(handles),
        fontsize=8,
    )

    kp_ticks_m, kp_labels = _kp_primary_ticks(total_length_m)
    axes[-1].set_xticks(kp_ticks_m)
    axes[-1].set_xticklabels(kp_labels)
    axes[-1].tick_params(axis="x", labelsize=9.5)
    axes[-1].set_xlabel("KP (primary reference)", fontsize=9)
    axes[-1].set_xlim(0.0, total_length_m)

    fig.text(
        0.5,
        0.01,
        "Bands are ALIGNED, not fused: observed / modelled (screening) / legacy-regional-context "
        "evidence remain visually and semantically distinct. No correlation or fitted trend is "
        "shown.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="0.25",
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Engineering section summary table (Section 14) ----------------------------------------


def render_section_summary_table(
    *,
    section_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Human-readable 14-row table -- neutral typography, no traffic-light colouring."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = section_df.sort_values("segment_id").reset_index(drop=True)

    columns = [
        "KP range",
        "2018\nevents",
        "Shear p95\n(Pa)",
        "Mobility\n(mm)",
        "Embedment\nclass (xD)",
        "Folk\nclass",
        "Local\nrelief (m)",
        "Primary limitation",
    ]
    col_widths = [0.20, 0.07, 0.12, 0.09, 0.10, 0.07, 0.09, 0.26]
    rows = []
    for _, row in ordered.iterrows():
        limitation = (
            "Observed 2018 event(s)"
            if row["observed_2018_freespan_count"] and row["observed_2018_freespan_count"] > 0
            else "No 2018 event in section"
        )
        rows.append(
            [
                f"{row['kp_start']} - {row['kp_end']}",
                f"{int(row['observed_2018_freespan_count'])}",
                f"{row['combined_tau_max_p95_lower_pa']:.2f}-{row['combined_tau_max_p95_upper_pa']:.2f}",
                f"{row['mobility_capacity_p95_d50_mm']:.3g}",
                f"{row['p95_required_embedment_upper_class']}",
                f"{row['mapped_250k_folk_class']}",
                f"{row['local_relief_1000m_median_m']:.2f}",
                limitation,
            ]
        )

    fig_height = 1.3 + 0.5 * len(rows)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("0.7")
        if row_idx == 0:
            cell.set_facecolor("0.85")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
        if col_idx in (0, len(columns) - 1) and row_idx > 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02

    ax.set_title(
        "PL854 Engineering Section Summary -- 14 hydro-pair support sections",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.0,
        "Neutral summary only -- no traffic-light colouring, no fused score or priority ranking.",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="0.25",
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Evidence provenance timeline (Section 15) ----------------------------------------------


def render_provenance_timeline(
    *,
    epochs: list[dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Each `epoch` dict: {label, start_year, end_year, kind}. `kind` is either
    'OBSERVATION_PERIOD' (drawn as a bar) or 'PUBLICATION_EVENT' (drawn as a
    marker) -- publication is never confused with observation (Section 15).
    `label` must already carry any date-range text the caller wants shown --
    it is rendered VERBATIM, never reconstructed from rounded `start_year`/
    `end_year` floats (rounding a fractional year, e.g. 2026.67, to `.0f`
    silently produces a misleading label like "2027")."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))

    periods = [e for e in epochs if e["kind"] == "OBSERVATION_PERIOD"]
    events = [e for e in epochs if e["kind"] == "PUBLICATION_EVENT"]

    for i, epoch in enumerate(periods):
        ax.barh(
            i,
            epoch["end_year"] - epoch["start_year"],
            left=epoch["start_year"],
            height=0.5,
            color=epoch.get("color", "#4C6EF5"),
            edgecolor="0.3",
        )
        ax.text(
            epoch["start_year"],
            i + 0.35,
            epoch["label"],
            fontsize=8,
            va="bottom",
        )

    for epoch in events:
        ax.plot(epoch["start_year"], -1, marker="D", markersize=9, color="#B4131A")
        ax.annotate(
            epoch["label"],
            (epoch["start_year"], -1),
            textcoords="offset points",
            xytext=(6, -4),
            fontsize=8,
            fontweight="bold",
        )

    all_years = [e["start_year"] for e in epochs] + [e["end_year"] for e in epochs]
    ax.set_xlim(min(all_years) - 2, max(all_years) + 2)
    ax.set_yticks([])
    ax.set_ylim(-1.8, len(periods))
    ax.set_xlabel("Year")
    ax.set_title("PL854 Evidence Provenance Timeline", fontsize=13, fontweight="bold")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.text(
        0.5,
        -0.02,
        "Bars are acquisition/observation periods; diamonds are PUBLICATION dates of a source "
        "document -- never the observation itself.",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="0.25",
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px

"""Official freespan spatial evidence maps + model-context profile (MAR-014A).

Kept self-contained (Sections 21-23)
-------------------------------------------
Duplicates its own small plotting helpers (`_add_kp_labels`, `_add_scale_bar`,
`_add_north_arrow`, `read_png_dimensions`, `_plot_background_raster`) rather
than importing them from MAR-012/013/014's map modules, so a future change
made here can never destabilize an already-shipped map (the same
convention every prior map module in this project follows).

Corridor scope + no score, on every map (Sections 2, 17, 21-23)
-------------------------------------------------------------------------
Every rendered map/profile states, verbatim: the source scope is the
PIGGYBACKED PL854/PL855 corridor (never PL854 alone, individual line
attribution UNRESOLVED); the source CRS was empirically inferred, never
source-stated; and NO EXPOSURE OR FREESPAN SUSCEPTIBILITY SCORE has been
computed anywhere in this module.

2014 partial coverage (Section 22)
-----------------------------------------
The historical multi-year map computes (never hard-codes) each survey
year's own reported-event chainage zones and flags any zone reported by
another year but entirely absent from 2014 -- a statement about which
zones Table B.1 happens to report events in for each year, never a claim
that a specific freespan persisted, migrated, disappeared, or was
newly-formed between surveys.
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

CORRIDOR_SCOPE_STATEMENT = (
    "Source scope: PIGGYBACKED PL854/PL855 corridor (Ithaca Energy Comparative Assessment, "
    "April 2020, Appendix B Table B.1) -- NOT PL854 alone; individual line attribution "
    "UNRESOLVED for every event."
)
CRS_INFERENCE_STATEMENT = (
    "Source CRS was NOT stated; EPSG:23031 (ED50 / UTM 31N) was empirically inferred from "
    "point-to-route fit, never assumed."
)
NO_SCORE_STATEMENT = (
    "NO EXPOSURE OR FREESPAN SUSCEPTIBILITY SCORE HAS BEEN CREATED. Symbols/colours below show "
    "only the officially tabulated survey events themselves."
)

_YEAR_COLORS = {2012: "tab:orange", 2014: "tab:green", 2018: "tab:red"}


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


def _plot_route_and_events(
    ax, route: LineString, events_gdf: gpd.GeoDataFrame, colours: list
) -> None:
    route_gdf = gpd.GeoDataFrame(geometry=[route], crs=events_gdf.crs)
    route_gdf.plot(color="0.75", linewidth=1.5, ax=ax, zorder=1)

    for geom, colour in zip(events_gdf.geometry, colours, strict=True):
        if geom.geom_type == "Point":
            ax.plot(geom.x, geom.y, marker="o", markersize=7, color=colour, zorder=4)
        else:
            xs, ys = geom.xy
            ax.plot(xs, ys, color=colour, linewidth=5, solid_capstyle="round", zorder=4)


# --- Map 1: 2018 official freespan evidence (Section 21) -------------------------------


def _stagger_offsets_by_proximity(
    chainages_m: list[float],
    total_length_m: float,
    base: int = 12,
    step: int = 16,
    proximity_fraction: float = 0.02,
) -> list[int]:
    """An increasing vertical offset for each point in a RUN of mutually close
    points (within `proximity_fraction` of the route length of its predecessor,
    values assumed pre-sorted), resetting to `base` whenever a point is far
    from its predecessor -- so isolated events stay close to their marker
    while a tight cluster fans out and never stacks illegibly."""

    threshold_m = total_length_m * proximity_fraction
    offsets = []
    current = base
    for i, chainage in enumerate(chainages_m):
        if i > 0 and (chainage - chainages_m[i - 1]) <= threshold_m:
            current += step
        else:
            current = base
        offsets.append(current)
    return offsets


def _stagger_label_x_offsets_by_proximity(
    chainages_m: list[float],
    total_length_m: float,
    step: int = 12,
    proximity_fraction: float = 0.02,
) -> list[int]:
    """Horizontal analogue of `_stagger_offsets_by_proximity` -- a RUN of mutually
    close points (values assumed pre-sorted) fans out horizontally instead of
    stacking, always AWAY from whichever route edge the run sits nearer to.

    Fanning in a fixed alternating +/- pattern (as MAR-014A originally did)
    pushes labels near chainage 0 partly into negative offsets, i.e. past the
    y-axis -- crowding the axis tick labels instead of the plot area
    (MAR-014B Section 11). Biasing every run's whole fan away from its
    nearer edge avoids that regardless of where the run sits.
    """

    threshold_m = total_length_m * proximity_fraction
    offsets = [0] * len(chainages_m)
    run_start = 0
    for i in range(1, len(chainages_m) + 1):
        run_broke = i == len(chainages_m) or (chainages_m[i] - chainages_m[i - 1]) > threshold_m
        if not run_broke:
            continue
        run = list(range(run_start, i))
        if len(run) > 1:
            run_mean_chainage = sum(chainages_m[j] for j in run) / len(run)
            direction = 1 if run_mean_chainage < total_length_m / 2.0 else -1
            for rank, j in enumerate(run):
                offsets[j] = direction * rank * step
        run_start = i
    return offsets


def render_2018_freespan_evidence_map(
    *,
    events_2018_gdf: gpd.GeoDataFrame,
    route: LineString,
    output_path: Path,
    background_raster_path: Path | None = None,
    title: str = "PL854/PL855 Corridor — 2018 Official Freespan Survey Evidence",
    dpi: int = 150,
) -> Path:
    """The 8-event 2018 freespan corridor evidence map (primary validation dataset)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _setup_route_axes(route, background_raster_path)

    colours = ["tab:red"] * len(events_2018_gdf)
    _plot_route_and_events(ax, route, events_2018_gdf, colours)

    _add_kp_labels(ax, route)
    _add_endpoint_labels(ax, route)
    _add_scale_bar(ax)
    _add_north_arrow(ax)

    ordered = events_2018_gdf.sort_values("canonical_mid_chainage_m").reset_index(drop=True)
    offsets = _stagger_offsets_by_proximity(
        ordered["canonical_mid_chainage_m"].tolist(), route.length
    )
    for (_, row), y_offset in zip(ordered.iterrows(), offsets, strict=True):
        centroid = row.geometry.centroid
        ax.annotate(
            f"{row['event_id']} ({row['source_length_m']:.1f} m)",
            (centroid.x, centroid.y),
            textcoords="offset points",
            xytext=(0, y_offset),
            fontsize=7,
            ha="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.5", "alpha": 0.85},
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="tab:red",
            linewidth=5,
            label=f"2018 freespan event (n={len(events_2018_gdf)})",
        )
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(
        "Ithaca Energy Comparative Assessment (April 2020), Appendix B Table B.1",
        fontsize=9,
        style="italic",
        pad=14,
    )

    footer_lines = [CORRIDOR_SCOPE_STATEMENT, CRS_INFERENCE_STATEMENT, NO_SCORE_STATEMENT]
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


# --- Map 2: historical multi-year freespan evidence (Section 22) -----------------------


def compute_zone_coverage_by_year(
    events_df: pd.DataFrame, merge_gap_m: float = 300.0
) -> dict[int, list[tuple[float, float]]]:
    """Per survey year: merged reported-event chainage zones (adjacent events within
    `merge_gap_m` combined into one zone). Purely descriptive of which chainage
    zones Table B.1 happens to report events in for that year -- never a claim
    about any single freespan's identity/persistence across years."""

    zones_by_year: dict[int, list[tuple[float, float]]] = {}
    for year, group in events_df.groupby("survey_year"):
        intervals = sorted(
            zip(group["canonical_chainage_min_m"], group["canonical_chainage_max_m"], strict=True)
        )
        merged: list[list[float]] = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1] + merge_gap_m:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        zones_by_year[int(year)] = [(lo, hi) for lo, hi in merged]
    return zones_by_year


def compute_2014_coverage_gap_zones(
    events_df: pd.DataFrame, merge_gap_m: float = 300.0, proximity_tolerance_m: float = 100.0
) -> list[dict[str, Any]]:
    """Chainage zones reported by 2012 or 2018 with NO overlapping/nearby 2014 zone.

    A statement about reported-event coverage only: it never asserts that a
    specific freespan is absent, migrated, or newly-formed -- only that
    Table B.1 lists no 2014 event near a zone another year's survey does.
    """

    zones_by_year = compute_zone_coverage_by_year(events_df, merge_gap_m=merge_gap_m)
    zones_2014 = zones_by_year.get(2014, [])

    gaps = []
    for other_year in (2012, 2018):
        for lo, hi in zones_by_year.get(other_year, []):
            covered = any(
                (lo - proximity_tolerance_m) <= z_hi and (hi + proximity_tolerance_m) >= z_lo
                for z_lo, z_hi in zones_2014
            )
            if not covered:
                gaps.append(
                    {
                        "reported_by_year": other_year,
                        "zone_start_chainage_m": lo,
                        "zone_end_chainage_m": hi,
                    }
                )
    return gaps


def _format_2014_gap_warning(gaps: list[dict[str, Any]]) -> str:
    from marine_engine.preprocessing.chainage import format_kp_label

    if not gaps:
        return (
            "No chainage zone reported by 2012 or 2018 is entirely absent from the 2014 "
            "event listing."
        )
    zone_descriptions = ", ".join(
        f"{format_kp_label(g['zone_start_chainage_m'])}-"
        f"{format_kp_label(g['zone_end_chainage_m'])} "
        f"(reported by {g['reported_by_year']})"
        for g in gaps
    )
    return (
        f"2014 PARTIAL COVERAGE: Table B.1 lists no 2014 event near {zone_descriptions}. This "
        "may reflect incomplete 2014 survey coverage of the corridor rather than the absence "
        "of a freespan -- absence of a 2014 event must never be read as evidence of no "
        "freespan at that location."
    )


def render_historical_freespan_evidence_map(
    *,
    events_all_gdf: gpd.GeoDataFrame,
    route: LineString,
    output_path: Path,
    background_raster_path: Path | None = None,
    title: str = "PL854/PL855 Corridor — Historical Freespan Survey Evidence (2012, 2014, 2018)",
    dpi: int = 150,
) -> Path:
    """All 17 recovered freespan events across 2012/2014/2018, with an explicit,
    data-derived 2014-partial-coverage warning (Section 22).

    Deliberately unlabelled per-event (unlike the 8-event 2018 map): 17 events
    across three overlapping survey years would render as illegible stacked
    text at this route length -- the legend (colour-by-year) and the
    zone-level 2014-coverage-gap warning below carry the required information
    instead.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _setup_route_axes(route, background_raster_path)

    colours = [_YEAR_COLORS[int(year)] for year in events_all_gdf["survey_year"]]
    _plot_route_and_events(ax, route, events_all_gdf, colours)

    _add_kp_labels(ax, route)
    _add_endpoint_labels(ax, route)
    _add_scale_bar(ax)
    _add_north_arrow(ax)

    counts_by_year = events_all_gdf["survey_year"].value_counts().to_dict()
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=_YEAR_COLORS[year],
            linewidth=5,
            label=f"{year} freespan event (n={counts_by_year.get(year, 0)})",
        )
        for year in (2012, 2014, 2018)
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(
        "Ithaca Energy Comparative Assessment (April 2020), Appendix B Table B.1",
        fontsize=9,
        style="italic",
        pad=14,
    )

    gaps = compute_2014_coverage_gap_zones(events_all_gdf)
    footer_lines = [
        CORRIDOR_SCOPE_STATEMENT,
        CRS_INFERENCE_STATEMENT,
        _format_2014_gap_warning(gaps),
        NO_SCORE_STATEMENT,
        "No automatic cross-survey event matching (persistent/migrated/disappeared/"
        "newly-formed) has been performed.",
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


# --- Map 3: 2018 freespan vs. model-output context profile (Section 23) ----------------


def render_freespan_model_context_profile(
    *,
    context_df: pd.DataFrame,
    events_2018_span_df: pd.DataFrame,
    combined_bed_shear_segments_df: pd.DataFrame,
    noncohesive_mobility_segments_df: pd.DataFrame,
    scour_onset_segments_df: pd.DataFrame,
    total_length_m: float,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Three independent, separately-scaled panels (never fused into one score)
    juxtaposing MAR-012/013/014's own already-computed route-wide outputs against
    the 8 2018 event chainage positions, honestly showing MAR-014's field even
    though it is spatially uniform along the whole route (Section 23).

    `events_2018_span_df` (`event_id`, `canonical_chainage_min_m`,
    `canonical_chainage_max_m`) additionally draws each event's own REAL
    observed-span width as a short horizontal marker (MAR-014B Section 11) --
    true spans are ~0.2-23 m on a 23.5 km route, so this is never widened to
    stay visible; an honest, near-invisible mark is the correct one.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True)
    ax_shear, ax_mobility, ax_scour = axes

    for _, row in combined_bed_shear_segments_df.iterrows():
        x = [row["start_chainage_m"] / 1000.0, row["end_chainage_m"] / 1000.0]
        ax_shear.fill_between(
            x,
            [row["tau_max_p95_sensitivity_min_pa"]] * 2,
            [row["tau_max_p95_sensitivity_max_pa"]] * 2,
            color="tab:blue",
            alpha=0.35,
        )
    ax_shear.set_ylabel("MAR-012 tau_max\np95 sensitivity (Pa)", fontsize=8)

    for _, row in noncohesive_mobility_segments_df.iterrows():
        x = [row["start_chainage_m"] / 1000.0, row["end_chainage_m"] / 1000.0]
        value = row["largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"]
        if pd.notna(value):
            ax_mobility.plot(x, [value] * 2, color="tab:purple", linewidth=3, solid_capstyle="butt")
    ax_mobility.set_ylabel("MAR-013 largest passing\nD50 (p95, mm)", fontsize=8)

    for _, row in scour_onset_segments_df.iterrows():
        x = [row["start_chainage_m"] / 1000.0, row["end_chainage_m"] / 1000.0]
        value = row["p95_required_embedment_upper_ratio"]
        if pd.notna(value):
            ax_scour.plot(x, [value] * 2, color="tab:brown", linewidth=3, solid_capstyle="butt")
    ax_scour.set_ylabel("MAR-014 p95 required\nembedment upper (e/D)", fontsize=8)
    ax_scour.set_xlabel("Chainage (km)")

    span_by_event_id = {
        row["event_id"]: (row["canonical_chainage_min_m"], row["canonical_chainage_max_m"])
        for _, row in events_2018_span_df.iterrows()
    }

    for ax in axes:
        for _, event in context_df.iterrows():
            chainage_km = event["canonical_mid_chainage_m"] / 1000.0
            ax.axvline(chainage_km, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8)

            span = span_by_event_id.get(event["event_id"])
            if span is not None:
                y0 = ax.get_ylim()[0]
                ax.plot(
                    [span[0] / 1000.0, span[1] / 1000.0],
                    [y0, y0],
                    color="tab:red",
                    linewidth=4,
                    solid_capstyle="butt",
                    alpha=0.9,
                    zorder=6,
                    clip_on=False,
                )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, total_length_m / 1000.0)

    ordered_events = context_df.sort_values("canonical_mid_chainage_m").reset_index(drop=True)
    label_offsets = _stagger_label_x_offsets_by_proximity(
        ordered_events["canonical_mid_chainage_m"].tolist(), total_length_m
    )
    for (_, event), x_offset in zip(ordered_events.iterrows(), label_offsets, strict=True):
        chainage_km = event["canonical_mid_chainage_m"] / 1000.0
        ax_shear.annotate(
            event["event_id"],
            (chainage_km, ax_shear.get_ylim()[1]),
            textcoords="offset points",
            xytext=(x_offset, 2),
            fontsize=6,
            ha="left" if x_offset >= 0 else "right",
            rotation=90,
            va="bottom",
            color="tab:red",
        )

    fig.suptitle(
        "PL854 — 2018 Freespan Events vs. Independent Model Outputs (context only)",
        fontsize=12,
        fontweight="bold",
    )
    ax_shear.set_title(
        "Dashed lines: 8 official 2018 freespan event positions. Solid base marks: "
        "REAL observed span width.",
        fontsize=9,
        style="italic",
        pad=32,
    )

    footer_lines = [
        NO_SCORE_STATEMENT,
        "Each panel is an independent, already-published model output (MAR-012/013/014) "
        "shown for side-by-side human review only -- values are never fused, differenced, "
        "or scored against event presence, and event presence is not a validated outcome "
        "this model was fitted against.",
        "Observed-span marks use each event's true length (~0.2-23 m) at true scale on this "
        "23.5 km route -- never widened to stay visible.",
    ]
    fig.text(
        0.0, -0.02, "\n".join(footer_lines), ha="left", va="top", fontsize=8, color="0.2", wrap=True
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Map 4: source-stated freespan evolution evidence (MAR-014B Section 10) ------------


def render_freespan_temporal_evolution_figure(
    *,
    temporal_evidence_df: pd.DataFrame,
    route: LineString,
    output_path: Path,
    background_raster_path: Path | None = None,
    title: str = "PL854/PL855 Corridor — Source-Stated Freespan Evolution Evidence",
    dpi: int = 150,
) -> Path:
    """Draws a link ONLY where the source states a relationship AND both event
    IDs are unambiguous (Section 10) -- never an inferred/nearest-neighbour
    connection, and never using canonical chainage to invent a pairing the
    source itself does not state. A single-event statement (e.g. "not
    surveyed in 2014") gets a marker at that one event's own position. A
    pure group/narrative statement (both event IDs null) is never given a
    fabricated geometry -- it is listed in a text box instead.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _setup_route_axes(route, background_raster_path)

    ax.plot(*route.xy, color="0.75", linewidth=1.5, zorder=1)
    _add_kp_labels(ax, route)
    _add_endpoint_labels(ax, route)
    _add_scale_bar(ax)
    _add_north_arrow(ax)

    has_a = temporal_evidence_df["event_id_a"].notna()
    has_b = temporal_evidence_df["event_id_b"].notna()
    paired_df = temporal_evidence_df[has_a & has_b]
    single_df = temporal_evidence_df[has_a & ~has_b]
    narrative_df = temporal_evidence_df[~has_a & ~has_b]

    # Collect every annotation task first (never rendered yet) so its ANCHOR
    # chainage can be sorted and vertically staggered as one combined set --
    # events from different relationship pairs can sit within a few hundred
    # metres of each other on this 23.5 km route (Section 11's crowding fix
    # applies here too, not just the model-context profile).
    annotation_tasks = []

    for (event_id_a, event_id_b), group in paired_df.groupby(["event_id_a", "event_id_b"]):
        first = group.iloc[0]
        point_a = route.interpolate(first["canonical_mid_chainage_a_m"])
        point_b = route.interpolate(first["canonical_mid_chainage_b_m"])
        ax.plot(
            [point_a.x, point_b.x],
            [point_a.y, point_b.y],
            color="tab:purple",
            linewidth=2.0,
            zorder=3,
            alpha=0.85,
        )
        for point in (point_a, point_b):
            ax.plot(point.x, point.y, marker="o", markersize=7, color="tab:purple", zorder=4)
        statements = "; ".join(group["source_statement"].tolist())
        annotation_tasks.append(
            {
                "chainage_m": (
                    first["canonical_mid_chainage_a_m"] + first["canonical_mid_chainage_b_m"]
                )
                / 2.0,
                "anchor_x": (point_a.x + point_b.x) / 2.0,
                "anchor_y": (point_a.y + point_b.y) / 2.0,
                "text": f"{event_id_a} <-> {event_id_b}\n{statements}",
                "facecolor": "lavender",
            }
        )

    for event_id_a, group in single_df.groupby("event_id_a"):
        first = group.iloc[0]
        point = route.interpolate(first["canonical_mid_chainage_a_m"])
        ax.plot(point.x, point.y, marker="^", markersize=8, color="tab:blue", zorder=4)
        statements = "; ".join(group["source_statement"].tolist())
        annotation_tasks.append(
            {
                "chainage_m": first["canonical_mid_chainage_a_m"],
                "anchor_x": point.x,
                "anchor_y": point.y,
                "text": f"{event_id_a}\n{statements}",
                "facecolor": "lightyellow",
            }
        )

    annotation_tasks.sort(key=lambda t: t["chainage_m"])
    y_offsets = _stagger_offsets_by_proximity(
        [t["chainage_m"] for t in annotation_tasks], route.length, base=16, step=42
    )
    for task, y_offset in zip(annotation_tasks, y_offsets, strict=True):
        ax.annotate(
            task["text"],
            (task["anchor_x"], task["anchor_y"]),
            textcoords="offset points",
            xytext=(0, y_offset),
            fontsize=6,
            ha="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": task["facecolor"], "ec": "0.5", "alpha": 0.9},
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="tab:purple",
            linewidth=2,
            marker="o",
            label="Source-stated same-span / length-change relationship",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="tab:blue",
            markersize=8,
            label="Single-event source statement",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(
        "Links drawn ONLY where the source states a relationship AND both events are "
        "unambiguous; group/narrative-only statements are listed below, never as geometry.",
        fontsize=9,
        style="italic",
        pad=14,
    )

    if not narrative_df.empty:
        narrative_lines = ["Group/narrative source statements (no specific event geometry):"]
        for _, row in narrative_df.iterrows():
            narrative_lines.append(f"- {row['source_statement']}")
        ax.text(
            0.99,
            0.99,
            "\n".join(narrative_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6,
            wrap=True,
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.4", "alpha": 0.92},
        )

    footer_lines = [
        CORRIDOR_SCOPE_STATEMENT,
        "NO CROSS-SURVEY RELATIONSHIP WAS CREATED BY SPATIAL PROXIMITY ALONE.",
        NO_SCORE_STATEMENT,
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


# --- Final report (Section 27) ----------------------------------------------------------


def print_freespan_evidence_report(
    *,
    events_all_df: pd.DataFrame,
    events_2018_df: pd.DataFrame,
    accepted_epsg: int,
    crs_checks: dict[str, bool],
    survey_direction: str,
    segment_counts_df: pd.DataFrame,
    evidence_path: Path,
    evidence_2018_path: Path,
    model_context_path: Path,
    reconciliation_metadata_path: Path,
    map_2018_path: Path,
    map_2018_dimensions: tuple[int, int],
    map_historical_path: Path,
    map_historical_dimensions: tuple[int, int],
    profile_path: Path,
    profile_dimensions: tuple[int, int],
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854/PL855 Official Freespan Spatial Evidence (MAR-014A) ===", ""]

    lines.append("## Source recovery")
    lines.append(
        "  Ithaca Energy (UK) Limited (2020). Pipelines and Umbilical Comparative "
        "Assessment. April 2020. Appendix B, Table B.1."
    )
    lines.append(f"  Total events recovered: {len(events_all_df)} (2012/2014/2018)")
    for year, group in events_all_df.groupby("survey_year"):
        lines.append(
            f"    {year}: {len(group)} event(s), sum length {group['source_length_m'].sum():.2f} m"
        )
    lines.append(CORRIDOR_SCOPE_STATEMENT)
    lines.append("")

    lines.append("## CRS reconciliation")
    lines.append(f"  Accepted CRS: EPSG:{accepted_epsg} (source CRS was NOT stated)")
    for check, passed in crs_checks.items():
        lines.append(f"    {check}: {passed}")
    lines.append(f"  Survey KP direction relative to canonical route: {survey_direction}")
    lines.append("")

    lines.append("## Chainage reconciliation")
    lines.append(f"  2018 primary validation subset: {len(events_2018_df)} event(s)")
    max_endpoint_distance_m = max(
        events_all_df["endpoint_a_route_distance_m"].max(),
        events_all_df["endpoint_b_route_distance_m"].max(),
    )
    lines.append(f"  Max endpoint-to-route distance: {max_endpoint_distance_m:.2f} m")
    lines.append("")

    lines.append("## Model context (Section 17)")
    lines.append(f"  {model_context_path}")
    lines.append(
        "NO SCORE, PROBABILITY, RANK, OR ACCURACY METRIC HAS BEEN COMPUTED -- model outputs "
        "are shown alongside observed events for human review only."
    )
    lines.append("")

    lines.append("## Segment event counts (2018)")
    any_count = int(segment_counts_df["any_2018_freespan"].sum())
    lines.append(f"  Segments with >=1 2018 event: {any_count} / {len(segment_counts_df)}")
    lines.append("")

    lines.append("## Maps")
    lines.append(
        f"  2018 evidence map: {map_2018_path} "
        f"({map_2018_dimensions[0]}x{map_2018_dimensions[1]} px)"
    )
    lines.append(
        f"  Historical evidence map: {map_historical_path} "
        f"({map_historical_dimensions[0]}x{map_historical_dimensions[1]} px)"
    )
    lines.append(
        f"  Model-context profile: {profile_path} "
        f"({profile_dimensions[0]}x{profile_dimensions[1]} px)"
    )
    lines.append("")

    lines.append("## Outputs")
    lines.append(f"  {evidence_path}")
    lines.append(f"  {evidence_2018_path}")
    lines.append(f"  {reconciliation_metadata_path}")
    lines.append("")

    lines.append(CORRIDOR_SCOPE_STATEMENT)
    lines.append("Individual PL854-vs-PL855 freespan attribution remains UNRESOLVED.")
    lines.append(NO_SCORE_STATEMENT)

    print("\n".join(lines), file=file)

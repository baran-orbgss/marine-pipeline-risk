"""Route sections, tangent bearings, capacity map, and profile (MAR-014).

Map-ready spatial support reuses MAR-012/013 hydro pairs (Sections 12, 31)
-------------------------------------------------------------------------------
Route sections are the SAME contiguous hydro-pair runs MAR-012/013 already
built and verified. `build_route_sections` additionally derives each
section's own LOCAL route-tangent bearing (a numerical tangent from the
TRUE curved geometry at the section's chainage midpoint -- never a
whole-route chord) so the physics layer can project current/wave forcing
onto the pipeline-normal direction BEFORE the final capacity results
exist; `build_scour_onset_embedment_segments` then attaches those results
back onto the same sections. Kept as an independent, self-contained module
so already-shipped maps are never put at risk by a change made for this
ticket.

Map honesty: discrete embedment classes, extrapolation always visible
(Sections 32-35)
-------------------------------------------------------------------------------
The primary map's route colour is `p95_required_embedment_upper_class`,
one of exactly six discrete classes (`0`/`0.03D`/`0.06D`/`0.10D`/`0.15D`/
`>0.15D`) -- never a continuously interpolated value. The map, its footer,
and its metadata always state that PL854's D=0.305 m exceeds the source
experimental D=0.05-0.10 m envelope and that oblique-flow projection is a
screening extension, not a validated model. No risk/safety language is
ever used for labels.
"""

import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring

from marine_engine.preprocessing.chainage import format_kp_label
from marine_engine.scour.scour_onset import (
    MORPHOLOGY_ROLE,
    PIPE_DIAMETER_OUTSIDE_SOURCE_ENVELOPE,
    SCIENTIFIC_ROLE,
    SOURCE_DIAMETER_RANGE_M,
    TESTED_EMBEDMENT_RATIOS,
    compute_local_tangent_bearing_deg,
)

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_EMBEDMENT_CLASS_LABELS: tuple[str, ...] = tuple(
    f"{e:g}" for e in sorted(TESTED_EMBEDMENT_RATIOS)
) + (f">{max(TESTED_EMBEDMENT_RATIOS):g}",)

SCOUR_ONSET_EMBEDMENT_SEGMENTS_COLUMNS = (
    "pipeline_id",
    "segment_id",
    "start_chainage_m",
    "end_chainage_m",
    "kp_start",
    "kp_end",
    "hydro_pair_id",
    "route_tangent_bearing_deg",
    "p95_required_embedment_lower_class",
    "p95_required_embedment_upper_class",
    "p95_required_embedment_lower_ratio",
    "p95_required_embedment_upper_ratio",
    "p95_required_embedment_lower_m",
    "p95_required_embedment_upper_m",
    "pipeline_diameter_m",
    "pipe_diameter_source_envelope_status",
    "projected_uc_source_range_fraction",
    "projected_uw_source_range_fraction",
    "kc_source_range_fraction",
    "slope_500m_median_deg",
    "slope_500m_p95_deg",
    "slope_1000m_median_deg",
    "tpi_1000m_median_m",
    "local_relief_1000m_median_m",
    "terrain_std_1000m_median_m",
    "morphology_role",
    "scientific_role",
)


def _contiguous_runs(ids: pd.Series) -> pd.Series:
    """A run-id per row: increments only where the id actually changes.

    NaN-safe -- a run of consecutive missing assignments is still treated
    as one contiguous run rather than splitting on every row.
    """

    previous = ids.shift()
    changed = ~((ids == previous) | (ids.isna() & previous.isna()))
    changed.iloc[0] = True
    return changed.cumsum()


def build_route_sections(
    route: LineString, chainage_hydro_df: pd.DataFrame
) -> list[dict[str, Any]]:
    """Contiguous hydro-pair chainage runs, each with its own LOCAL tangent bearing.

    `chainage_hydro_df` is one row per real chainage station: `chainage_m`,
    `hydro_pair_id`, sorted by chainage. Returns one dict per contiguous
    run: `hydro_pair_id`, `start_chainage_m`, `end_chainage_m`,
    `route_tangent_bearing_deg` (a numerical tangent at the run's own
    chainage MIDPOINT -- never a whole-route chord, Section 12/test P).
    Used BOTH to build the `{hydro_pair_id: bearing}` map the physics layer
    needs before capacity results exist, and later to attach those results
    back onto the same sections.
    """

    if chainage_hydro_df.empty:
        return []

    ordered = chainage_hydro_df.sort_values("chainage_m").reset_index(drop=True)
    run_id = _contiguous_runs(ordered["hydro_pair_id"])
    total_length_m = route.length

    raw_runs = []
    for _, group in ordered.groupby(run_id, sort=True):
        raw_runs.append(
            {
                "hydro_pair_id": group["hydro_pair_id"].iloc[0],
                "first_chainage_m": float(group["chainage_m"].iloc[0]),
                "last_chainage_m": float(group["chainage_m"].iloc[-1]),
            }
        )

    sections = []
    for i, run in enumerate(raw_runs):
        start_chainage_m = (
            0.0 if i == 0 else (raw_runs[i - 1]["last_chainage_m"] + run["first_chainage_m"]) / 2.0
        )
        end_chainage_m = (
            total_length_m
            if i == len(raw_runs) - 1
            else (run["last_chainage_m"] + raw_runs[i + 1]["first_chainage_m"]) / 2.0
        )
        midpoint_chainage_m = (start_chainage_m + end_chainage_m) / 2.0
        sections.append(
            {
                "segment_id": i,
                "hydro_pair_id": run["hydro_pair_id"],
                "start_chainage_m": start_chainage_m,
                "end_chainage_m": end_chainage_m,
                "route_tangent_bearing_deg": compute_local_tangent_bearing_deg(
                    route, midpoint_chainage_m
                ),
            }
        )
    return sections


def build_tangent_bearing_by_pair_id(sections: list[dict[str, Any]]) -> dict[str, float]:
    """`{hydro_pair_id: route_tangent_bearing_deg}`, one entry per contiguous run.

    A hydro pair that (unexpectedly) spans more than one disjoint run uses
    the FIRST run's bearing -- real PL854 pairs are each one contiguous
    run, so this never actually arises in practice.
    """

    result: dict[str, float] = {}
    for section in sections:
        pair_id = section["hydro_pair_id"]
        if pd.notna(pair_id) and pair_id not in result:
            result[pair_id] = section["route_tangent_bearing_deg"]
    return result


def build_scour_onset_embedment_segments(
    *,
    pipeline_id: str,
    route: LineString,
    sections: list[dict[str, Any]],
    envelope_by_pair_id: dict[str, dict[str, Any]],
    applicability: dict[str, Any],
    morphology_by_pair_id: dict[str, dict[str, Any]],
    diameter_m: float,
    working_crs: str,
) -> gpd.GeoDataFrame:
    """Attach capacity/QA/applicability/morphology context onto the bare route sections.

    Morphology fields are LEGACY REGIONAL CONTEXT ONLY -- they never enter
    the onset calculation itself (Section 27); this function only carries
    them alongside the already-computed capacity results for map/report
    convenience.
    """

    if not sections:
        return gpd.GeoDataFrame(columns=list(SCOUR_ONSET_EMBEDMENT_SEGMENTS_COLUMNS), geometry=[])

    total_length_m = route.length
    within_diameter_envelope = (
        SOURCE_DIAMETER_RANGE_M[0] <= diameter_m <= SOURCE_DIAMETER_RANGE_M[1]
    )
    diameter_status = (
        "WITHIN_SOURCE_DIAMETER_ENVELOPE"
        if within_diameter_envelope
        else PIPE_DIAMETER_OUTSIDE_SOURCE_ENVELOPE
    )

    records = []
    geometries = []
    for section in sections:
        start_chainage_m = section["start_chainage_m"]
        end_chainage_m = min(section["end_chainage_m"], total_length_m)
        segment_geom = substring(route, start_chainage_m, end_chainage_m, normalized=False)

        pair_id = section["hydro_pair_id"]
        envelope = envelope_by_pair_id.get(pair_id, {}) if pd.notna(pair_id) else {}
        morphology = morphology_by_pair_id.get(pair_id, {}) if pd.notna(pair_id) else {}

        lower_ratio = envelope.get("p95_required_embedment_lower_ratio")
        upper_ratio = envelope.get("p95_required_embedment_upper_ratio")

        records.append(
            {
                "pipeline_id": pipeline_id,
                "segment_id": section["segment_id"],
                "start_chainage_m": start_chainage_m,
                "end_chainage_m": end_chainage_m,
                "kp_start": format_kp_label(start_chainage_m),
                "kp_end": format_kp_label(end_chainage_m),
                "hydro_pair_id": pair_id if pd.notna(pair_id) else None,
                "route_tangent_bearing_deg": section["route_tangent_bearing_deg"],
                "p95_required_embedment_lower_class": envelope.get(
                    "p95_required_embedment_lower_class"
                ),
                "p95_required_embedment_upper_class": envelope.get(
                    "p95_required_embedment_upper_class"
                ),
                "p95_required_embedment_lower_ratio": lower_ratio,
                "p95_required_embedment_upper_ratio": upper_ratio,
                "p95_required_embedment_lower_m": (
                    lower_ratio * diameter_m if lower_ratio is not None else None
                ),
                "p95_required_embedment_upper_m": (
                    upper_ratio * diameter_m if upper_ratio is not None else None
                ),
                "pipeline_diameter_m": diameter_m,
                "pipe_diameter_source_envelope_status": diameter_status,
                "projected_uc_source_range_fraction": applicability.get(
                    "projected_uc_source_range_fraction"
                ),
                "projected_uw_source_range_fraction": applicability.get(
                    "projected_uw_source_range_fraction"
                ),
                "kc_source_range_fraction": applicability.get("kc_source_range_fraction"),
                "slope_500m_median_deg": morphology.get("slope_500m_median_deg"),
                "slope_500m_p95_deg": morphology.get("slope_500m_p95_deg"),
                "slope_1000m_median_deg": morphology.get("slope_1000m_median_deg"),
                "tpi_1000m_median_m": morphology.get("tpi_1000m_median_m"),
                "local_relief_1000m_median_m": morphology.get("local_relief_1000m_median_m"),
                "terrain_std_1000m_median_m": morphology.get("terrain_std_1000m_median_m"),
                "morphology_role": MORPHOLOGY_ROLE,
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )
        geometries.append(segment_geom)

    return gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs=working_crs,
        columns=list(SCOUR_ONSET_EMBEDMENT_SEGMENTS_COLUMNS),
    )


def write_scour_onset_embedment_segments_gpkg(
    gdf: gpd.GeoDataFrame, output_path: Path, layer: str = "scour_onset_embedment_segments"
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer)
    return output_path


# --- Discrete embedment-class colour scale (Section 32) --------------------------------


def _discrete_embedment_class_colors() -> dict[str, tuple]:
    cmap = plt.get_cmap("viridis", len(_EMBEDMENT_CLASS_LABELS))
    return {label: cmap(i) for i, label in enumerate(_EMBEDMENT_CLASS_LABELS)}


def _colour_for_class(colors_by_class: dict[str, tuple], class_label: Any) -> tuple:
    if class_label is None or (isinstance(class_label, float) and pd.isna(class_label)):
        return (0.6, 0.6, 0.6, 1.0)
    return colors_by_class.get(str(class_label), (0.6, 0.6, 0.6, 1.0))


# --- Static PNG map rendering (Sections 32-35) ------------------------------------------


def _kp_tick_chainages_m(total_length_m: float, interval_km: float = 5.0) -> list[float]:
    interval_m = interval_km * 1000.0
    ticks = list(np.arange(0.0, total_length_m, interval_m))
    if not ticks or not np.isclose(ticks[-1], total_length_m):
        ticks.append(total_length_m)
    return ticks


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


def _format_embedment_hotspot_label(row: pd.Series) -> str:
    kp_start_km = row["start_chainage_m"] / 1000.0
    kp_end_km = row["end_chainage_m"] / 1000.0
    lower = row.get("p95_required_embedment_lower_class") or "n/a"
    upper = row.get("p95_required_embedment_upper_class") or "n/a"
    return f"KP {kp_start_km:.1f}-{kp_end_km:.1f} | p95 required e/D: {lower}-{upper}"


def render_scour_onset_embedment_map(
    *,
    segments_gdf: gpd.GeoDataFrame,
    route: LineString,
    working_crs: str,
    output_path: Path,
    diameter_m: float,
    background_raster_path: Path | None = None,
    route_start_label: str = "Source geometry start",
    route_end_label: str = "Source geometry terminus",
    title: str = "PL854 — Pipeline Scour-Onset Embedment Screening",
    subtitle: str = (
        "Marini et al. (2024) combined wave-current onset criterion; pipeline-normal 2D "
        "screening projection"
    ),
    dpi: int = 150,
) -> Path:
    """Render the required static PL854 scour-onset-embedment-screening PNG.

    Colours the route by `p95_required_embedment_upper_class` -- one of
    six DISCRETE classes, never a continuously interpolated value
    (Section 32). Always shows PL854 millimetre equivalents in the legend
    and states the diameter-extrapolation limitation in the footer.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    value_column = "p95_required_embedment_upper_class"
    has_value = value_column in segments_gdf.columns and segments_gdf[value_column].notna().any()
    if has_value:
        colors_by_class = _discrete_embedment_class_colors()
        colours = [_colour_for_class(colors_by_class, v) for v in segments_gdf[value_column]]
        segments_gdf.plot(color=colours, linewidth=4, ax=ax)

        legend_handles = [
            Patch(
                facecolor=colors_by_class[label],
                edgecolor="0.3",
                label=f"{label}D" if label != _EMBEDMENT_CLASS_LABELS[-1] else label,
            )
            for label in _EMBEDMENT_CLASS_LABELS
        ]
        mm_by_class = {
            label: (
                f"{float(label) * diameter_m * 1000.0:.0f} mm"
                if not label.startswith(">")
                else f">{float(label[1:]) * diameter_m * 1000.0:.0f} mm"
            )
            for label in _EMBEDMENT_CLASS_LABELS
        }
        for handle, label in zip(legend_handles, _EMBEDMENT_CLASS_LABELS, strict=True):
            handle.set_label(f"{handle.get_label()} ({mm_by_class[label]})")
        ax.legend(
            handles=legend_handles,
            title="p95 required embedment (upper class)",
            loc="upper left",
            fontsize=7,
            title_fontsize=7,
            framealpha=0.9,
        )
    else:
        segments_gdf.plot(color="0.4", linewidth=4, ax=ax)

    _add_kp_labels(ax, route)
    _add_endpoint_labels(ax, route, route_start_label, route_end_label)
    _add_scale_bar(ax)
    _add_north_arrow(ax)
    _label_top_sections(ax, segments_gdf)
    _add_2018_benchmark_box(ax)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=9, style="italic", pad=14)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    footer_lines = [
        "Route: NSTA | Hydrodynamics: Copernicus Marine | Morphology/background: EMODnet | "
        "Sediment context: BGS",
        "Colours show the upper p95 tested embedment class required to suppress the "
        "empirical onset criterion; actual route embedment is unknown.",
        f"Research screening extrapolation: PL854 D={diameter_m:.3f} m exceeds the source "
        f"experimental D={SOURCE_DIAMETER_RANGE_M[0]:g}-{SOURCE_DIAMETER_RANGE_M[1]:g} m "
        "envelope.",
        "Oblique forcing is projected normal to the pipe; this extension is not directly "
        "validated by the source experiments.",
    ]
    ax.text(
        0.0,
        -0.10,
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


def _add_kp_labels(ax, route: LineString) -> None:
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


def _add_endpoint_labels(ax, route: LineString, start_label: str, end_label: str) -> None:
    start, end = Point(route.coords[0]), Point(route.coords[-1])
    for point, label, offset in ((start, start_label, (8, -26)), (end, end_label, (8, -26))):
        ax.plot(point.x, point.y, marker="s", markersize=5, color="black", zorder=5)
        ax.annotate(
            label,
            (point.x, point.y),
            textcoords="offset points",
            xytext=offset,
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


def _label_top_sections(ax, segments_gdf: gpd.GeoDataFrame, top_n: int = 3) -> None:
    """Max 3 annotations, FULL sensitivity range -- never risk/safety language (Section 33)."""

    value_column = "p95_required_embedment_upper_ratio"
    if value_column not in segments_gdf.columns:
        return
    ranked = segments_gdf.copy()
    ranked["_sort_key"] = ranked[value_column].apply(lambda v: v if pd.notna(v) else float("inf"))
    has_any = ranked["p95_required_embedment_upper_class"].notna()
    ranked = ranked[has_any].nlargest(top_n, "_sort_key")
    for rank, (_, row) in enumerate(ranked.iterrows()):
        centroid = row.geometry.centroid
        ax.annotate(
            _format_embedment_hotspot_label(row),
            (centroid.x, centroid.y),
            textcoords="offset points",
            xytext=(0, 16 + rank * 14),
            fontsize=7,
            ha="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.5", "alpha": 0.85},
        )


def _add_2018_benchmark_box(ax) -> None:
    """Compact, clearly separated HISTORICAL CONTEXT ONLY box (Section 34) -- never
    drawn as fake event symbols on the route itself."""

    lines = [
        "2018 official survey context — PL854/PL855 corridor",
        "Majority buried ≥ 0.6 m",
        "19 exposed sections / 519 m",
        "8 free spans / 97 m",
        "Locations not spatially assigned in this model",
    ]
    ax.text(
        0.99,
        0.99,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.4", "alpha": 0.92},
    )


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px


# --- Secondary chainage profile PNG (Section 36) ----------------------------------------


def render_scour_onset_embedment_profile(
    *, segments_gdf: gpd.GeoDataFrame, diameter_m: float, output_path: Path, dpi: int = 150
) -> Path:
    """Stepped chainage profile of the p95 required embedment sensitivity BAND
    (lower-to-upper class), never a continuous interpolation (Section 36)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_ratios = sorted(TESTED_EMBEDMENT_RATIOS)
    class_to_index = {f"{e:g}": i for i, e in enumerate(sorted_ratios)}
    class_to_index[f">{max(sorted_ratios):g}"] = len(sorted_ratios)

    fig, ax = plt.subplots(figsize=(12.0, 5.0))

    for _, row in segments_gdf.iterrows():
        lower = row.get("p95_required_embedment_lower_class")
        upper = row.get("p95_required_embedment_upper_class")
        if pd.isna(lower) or pd.isna(upper):
            continue
        lower_idx = class_to_index.get(str(lower))
        upper_idx = class_to_index.get(str(upper))
        if lower_idx is None or upper_idx is None:
            continue
        x = [row["start_chainage_m"] / 1000.0, row["end_chainage_m"] / 1000.0]
        ax.fill_between(
            x, [lower_idx] * 2, [upper_idx] * 2, color="tab:blue", alpha=0.35, step=None
        )
        ax.plot(x, [upper_idx] * 2, color="tab:blue", linewidth=2.5, solid_capstyle="butt")
        ax.plot(x, [lower_idx] * 2, color="tab:blue", linewidth=1.0, solid_capstyle="butt")

    class_labels = list(class_to_index.keys())
    ax.set_yticks(list(class_to_index.values()))
    ax.set_yticklabels(class_labels)
    ax.set_ylim(-0.5, len(class_labels) - 0.5)

    # Identity-mapped secondary axis: same tick POSITIONS as the primary
    # categorical axis, only the tick LABELS differ (mm equivalents) -- not
    # a real unit conversion, so both transform functions are the identity.
    secondary_ax = ax.secondary_yaxis("right", functions=(lambda idx: idx, lambda idx: idx))
    secondary_ax.set_yticks(list(class_to_index.values()))
    mm_labels = [f"{e * diameter_m * 1000.0:.0f}" for e in sorted_ratios] + [
        f">{max(sorted_ratios) * diameter_m * 1000.0:.0f}"
    ]
    secondary_ax.set_yticklabels(mm_labels)
    secondary_ax.set_ylabel(f"Required embedment (mm, D={diameter_m * 1000.0:.1f} mm)")

    ax.set_xlabel("Chainage (km)")
    ax.set_ylabel("p95 required tested embedment class (e/D)")
    ax.set_title(
        "PL854 — Scour-Onset Embedment Screening Profile\n"
        "p95 required embedment sensitivity band (D50 x porosity), by route section",
        fontsize=11,
    )
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.text(
        0.0,
        -0.18,
        "No observed route embedment profile is available.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.2",
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Final scientific report (Section 42) -----------------------------------------------


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def print_scour_onset_report(
    *,
    diameter_m: float,
    applicability: dict[str, Any],
    projection_qa: dict[str, Any],
    stats_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
    monotonicity_violation_count: int,
    morphology_summary: dict[str, Any],
    benchmark: dict[str, Any],
    segments_gdf: gpd.GeoDataFrame,
    segments_path: Path,
    png_path: Path,
    png_dimensions: tuple[int, int],
    profile_path: Path,
    profile_dimensions: tuple[int, int],
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854 Pipeline Scour-Onset Embedment Screening (MAR-014) ===", ""]

    lines.append("## Applicability")
    lines.append(
        f"  PL854 D = {diameter_m:.4f} m | source D range = "
        f"{SOURCE_DIAMETER_RANGE_M[0]:g}-{SOURCE_DIAMETER_RANGE_M[1]:g} m"
    )
    lines.append(
        f"  within_source_pipe_diameter_envelope: "
        f"{applicability.get('within_source_pipe_diameter_envelope')}"
    )
    lines.append(
        "  Projected combined-flow states within source range: "
        f"Uc={_fmt(applicability.get('projected_uc_source_range_fraction'), '.1%')} "
        f"Uw={_fmt(applicability.get('projected_uw_source_range_fraction'), '.1%')} "
        f"KC={_fmt(applicability.get('kc_source_range_fraction'), '.1%')}"
    )
    lines.append("")
    lines.append(
        "PL854 PIPE DIAMETER IS OUTSIDE THE SOURCE EXPERIMENTAL ENVELOPE; MAR-014 IS A "
        "RESEARCH SCREENING EXTRAPOLATION."
    )
    lines.append("")

    lines.append("## Projection")
    lines.append(
        "  Current projection ratio p05/median/p95: "
        f"{_fmt(projection_qa.get('current_projection_ratio_p05'), '.3f')}/"
        f"{_fmt(projection_qa.get('current_projection_ratio_median'), '.3f')}/"
        f"{_fmt(projection_qa.get('current_projection_ratio_p95'), '.3f')}"
    )
    lines.append(
        "  Wave projection ratio p05/median/p95: "
        f"{_fmt(projection_qa.get('wave_projection_ratio_p05'), '.3f')}/"
        f"{_fmt(projection_qa.get('wave_projection_ratio_median'), '.3f')}/"
        f"{_fmt(projection_qa.get('wave_projection_ratio_p95'), '.3f')}"
    )
    lines.append("")

    lines.append("## Scour-onset screening")
    if not stats_df.empty:
        for (d50_mm, porosity), group in stats_df.groupby(
            ["tested_d50_mm", "porosity_scenario"], sort=True
        ):
            lines.append(f"  D50={d50_mm:g} mm, n={porosity:g}:")
            for suffix, ratio in zip(
                ("000", "003", "006", "010", "015"), (0.00, 0.03, 0.06, 0.10, 0.15), strict=True
            ):
                onset_pct = group[f"onset_pct_eD_{suffix}"].mean()
                lines.append(
                    f"    e/D={ratio:g}: route-wide onset fraction={_fmt(onset_pct, '.1f')}%"
                )
    lines.append("")
    if not segments_gdf.empty:
        counts = segments_gdf["p95_required_embedment_upper_class"].value_counts(dropna=False)
        for label, count in counts.items():
            display = label if pd.notna(label) else "n/a"
            lines.append(f"  sections at p95 upper class {display}: {count}")
    lines.append(f"  Embedment monotonicity violation count: {monotonicity_violation_count}")
    lines.append("")

    lines.append("## Morphology context")
    for key, value in morphology_summary.items():
        lines.append(f"  {key}: {_fmt(value)}")
    lines.append(
        "MORPHOLOGY IS LEGACY REGIONAL CONTEXT ONLY AND DOES NOT ENTER THE SCOUR-ONSET EQUATION."
    )
    lines.append("")

    lines.append("## 2018 benchmark")
    lines.append(
        f"  PL854/PL855 corridor: majority buried >= {benchmark['majority_buried_at_least_m']:g} m"
    )
    lines.append(
        f"  {benchmark['exposed_section_count']} exposed sections / "
        f"{benchmark['total_exposed_length_m']} m"
    )
    lines.append(
        f"  {benchmark['free_span_count']} free spans / {benchmark['total_free_span_length_m']} m"
    )
    lines.append(f"  Maximum tabled free-span height: {benchmark['max_free_span_height_m']} m")
    lines.append(f"  Maximum tabled free-span length: {benchmark['max_free_span_length_m']} m")
    lines.append(
        "THE 2018 CONDITION BENCHMARK IS AGGREGATE CORRIDOR EVIDENCE AND WAS NOT SPATIALLY "
        "ASSIGNED TO PL854 SEGMENTS."
    )
    lines.append(f"  Source consistency note: {benchmark['source_internal_consistency_note']}")
    lines.append("")

    lines.append("## Map")
    lines.append(f"  Segment count: {len(segments_gdf)}")
    lines.append(f"  Map path: {png_path}")
    lines.append(f"  Map dimensions: {png_dimensions[0]} x {png_dimensions[1]} px")
    lines.append(f"  Profile path: {profile_path}")
    lines.append(f"  Profile dimensions: {profile_dimensions[0]} x {profile_dimensions[1]} px")
    lines.append("")

    lines.append(
        "MAP COLOURS SHOW THE UPPER P95 TESTED EMBEDMENT CLASS REQUIRED TO SUPPRESS THE "
        "EMPIRICAL SCOUR-ONSET CONDITION ACROSS THE D50/POROSITY SENSITIVITY SET."
    )
    lines.append(
        "THIS IS NOT AN OBSERVED BURIAL MAP, SCOUR-DEPTH PREDICTION, FREE-SPAN PREDICTION "
        "OR RISK MAP."
    )

    print("\n".join(lines), file=file)

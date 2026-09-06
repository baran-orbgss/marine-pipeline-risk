"""High-resolution seabed survey inventory maps (MAR-016).

Kept self-contained (never importing another ticket's own map modules),
matching this project's established convention.

Never colour by risk or morphology (Section 17)
-------------------------------------------------
Footprints are coloured ONLY by `route_overlap_class` (a factual spatial
category), never by a susceptibility/risk/morphology gradient. A hatch
pattern marks `METADATA_ONLY_*` access -- in this real run EVERY
route-relevant candidate is metadata-only, which is itself an honest,
reportable finding, not a design assumption.
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import LineString

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from marine_engine.validation.highres_seabed_survey_inventory import (  # noqa: E402
    AOI_INTERSECTING_NOT_ROUTE,
    CURRENT_HYDRODYNAMIC_FORCING_YEARS,
    FUGRO_2018_SURVEY_YEAR,
    MAR007_MORPHOLOGY_SOURCE_YEARS,
    METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED,
    NEARBY_NOT_AOI,
    ROUTE_INTERSECTING,
)

TARGET_GAP_STATEMENT = "Target evidence gap: pipeline-scale / current-era seabed morphology"
_OVERLAP_CLASS_COLORS = {
    ROUTE_INTERSECTING: "tab:blue",
    AOI_INTERSECTING_NOT_ROUTE: "tab:orange",
    NEARBY_NOT_AOI: "0.55",
}
_OVERLAP_CLASS_LABELS = {
    ROUTE_INTERSECTING: "Route-intersecting",
    AOI_INTERSECTING_NOT_ROUTE: "AOI-intersecting, not route",
    NEARBY_NOT_AOI: "Nearby, outside AOI",
}
_ACCESS_CLASS_ABBREVIATIONS = {
    METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED: "METADATA-ONLY",
}


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
    from matplotlib.lines import Line2D

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
    route: LineString, aoi_geometry, background_raster_path: Path | None = None
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = route.bounds
    aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = aoi_geometry.bounds
    minx, miny = min(minx, aoi_minx), min(miny, aoi_miny)
    maxx, maxy = max(maxx, aoi_maxx), max(maxy, aoi_maxy)
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


# --- Section 17: coverage map ------------------------------------------------------------


def render_survey_inventory_coverage_map(
    *,
    survey_inventory_df: pd.DataFrame,
    footprints_gdf: gpd.GeoDataFrame,
    route: LineString,
    aoi_geometry,
    output_path: Path,
    background_raster_path: Path | None = None,
    title: str = "PL854 -- High-Resolution Seabed Survey Data Inventory",
    dpi: int = 150,
) -> Path:
    """Route, 5 km AOI, real survey footprints coloured ONLY by
    `route_overlap_class` (never risk/morphology), each labelled with
    acquisition year, and a hatch pattern for metadata-only access
    (Section 17)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = _content_bounds(route, aoi_geometry, background_raster_path)
    span_x, span_y = max(maxx - minx, 1.0), max(maxy - miny, 1.0)
    fig_height = 8.0
    fig_width = max(8.0, min(18.0, fig_height * (span_x / span_y)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pad_x, pad_y = span_x * 0.06, span_y * 0.10
    view_minx, view_maxx = minx - pad_x, maxx + pad_x
    view_miny, view_maxy = miny - pad_y, maxy + pad_y
    ax.set_xlim(view_minx, view_maxx)
    ax.set_ylim(view_miny, view_maxy)

    from shapely.geometry import box as shapely_box

    view_box = shapely_box(view_minx, view_miny, view_maxx, view_maxy)

    if background_raster_path is not None and Path(background_raster_path).exists():
        _plot_background_raster(ax, Path(background_raster_path))

    aoi_gs = gpd.GeoSeries([aoi_geometry])
    aoi_gs.plot(ax=ax, facecolor="none", edgecolor="0.4", linewidth=1.0, linestyle="--", zorder=1)

    merged = footprints_gdf.merge(survey_inventory_df, on="survey_id", how="inner")
    # Several real legacy site surveys share an IDENTICAL licensed-block footprint (e.g. the
    # same block 49/16 polygon reused across 6 distinct ConocoPhillips surveys) -- drawing each
    # as a separate overlapping shape/label makes the map illegible without changing any fact,
    # so shapes are grouped by their real (exact) geometry before drawing; every individual
    # survey_id remains its own row in the parquet table regardless of this grouping.
    merged["_geom_key"] = merged["geometry"].apply(lambda g: g.wkt)
    for _, group in merged.groupby("_geom_key", sort=False):
        first = group.iloc[0]
        colour = _OVERLAP_CLASS_COLORS.get(first["route_overlap_class"], "0.5")
        hatch = "//" if first["access_class"] == METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED else None
        gpd.GeoSeries([first["geometry"]]).plot(
            ax=ax,
            facecolor=colour,
            edgecolor=colour,
            alpha=0.25,
            hatch=hatch,
            linewidth=1.5,
            zorder=2,
        )
        survey_ids = sorted(group["survey_id"].tolist())
        years = sorted({y for y in group["acquisition_year"].tolist() if pd.notna(y)})
        year_text = f"{years[0]}" if len(years) <= 1 else f"{years[0]}-{years[-1]}"
        id_text = (
            survey_ids[0] if len(survey_ids) == 1 else f"{survey_ids[0]} +{len(survey_ids) - 1}"
        )
        # Anchor the label on the portion of the footprint actually visible in this view --
        # some real licensed-block footprints are much larger than the AOI itself, so the
        # true polygon centroid can fall well outside the plotted extent.
        visible_portion = first["geometry"].intersection(view_box)
        anchor = (
            visible_portion.centroid if not visible_portion.is_empty else first["geometry"].centroid
        )
        ax.annotate(
            f"{id_text}\n{year_text}",
            (anchor.x, anchor.y),
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
        )

    ax.plot(*route.xy, color="0.15", linewidth=1.6, zorder=4)

    _add_kp_labels(ax, route)
    _add_scale_bar(ax)
    _add_north_arrow(ax)

    legend_handles = [
        Patch(facecolor=colour, edgecolor=colour, alpha=0.4, label=_OVERLAP_CLASS_LABELS[cls])
        for cls, colour in _OVERLAP_CLASS_COLORS.items()
    ]
    legend_handles.append(
        Patch(facecolor="white", edgecolor="0.3", hatch="//", label="Metadata-only access")
    )
    legend_handles.append(
        Patch(facecolor="none", edgecolor="0.4", linestyle="--", label="5 km AOI boundary")
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Placed as a figure-level (never axes-level) annotation, outside the data area entirely --
    # the data-filled block footprints can occupy any part of the axes depending on the real
    # AOI, so an in-axes box risks colliding with a survey label; the space below the title and
    # above the plot is the one region guaranteed clear of both the legend and the data.
    fig.text(
        0.5,
        0.925,
        TARGET_GAP_STATEMENT,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.4", "alpha": 0.95},
    )

    footer = (
        "MAR-016 INVENTORIES EVIDENCE AND ACCESS ONLY. NO NEW MORPHOLOGY, FREESPAN PREDICTION "
        "OR SUSCEPTIBILITY SCORE HAS BEEN CREATED."
    )
    fig.text(0.01, 0.01, footer, ha="left", va="bottom", fontsize=8, color="0.2", wrap=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 18: acquisition-epoch timeline ----------------------------------------------


def render_seabed_data_timeline(
    *,
    survey_inventory_df: pd.DataFrame,
    output_path: Path,
    title: str = "PL854 -- Seabed Survey Data Timeline",
    dpi: int = 150,
) -> Path:
    """One row per survey candidate, x=acquisition year, with fixed reference
    markers for MAR-007's legacy morphology source (1991-1992), the 2018
    Fugro/freespan survey epoch, and the current 2024-2026 hydrodynamic
    forcing interval -- making the temporal mismatch obvious (Section 18)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = survey_inventory_df.sort_values("acquisition_year", na_position="last").reset_index(
        drop=True
    )

    fig_height = max(3.0, 0.5 * max(len(ordered), 1) + 1.5)
    fig, ax = plt.subplots(figsize=(11.0, fig_height))

    for i, row in ordered.iterrows():
        colour = _OVERLAP_CLASS_COLORS.get(row["route_overlap_class"], "0.5")
        start_year = row["acquisition_start"].year if pd.notna(row["acquisition_start"]) else None
        end_year = row["acquisition_end"].year if pd.notna(row["acquisition_end"]) else start_year
        if start_year is not None:
            ax.plot(
                [start_year, max(end_year, start_year + 0.2)],
                [i, i],
                color=colour,
                linewidth=6,
                solid_capstyle="butt",
            )
        equipment = row["equipment"]
        equipment_text = (
            ", ".join(equipment[:3])
            + (f" +{len(equipment) - 3} more" if len(equipment) > 3 else "")
            if equipment
            else "n/a"
        )
        access_abbrev = _ACCESS_CLASS_ABBREVIATIONS.get(row["access_class"], row["access_class"])
        label = f"{row['survey_id']} | {access_abbrev} | {equipment_text}"
        ax.annotate(label, (max(end_year or 0, start_year or 0) + 0.3, i), va="center", fontsize=7)

    for years, ref_label in (
        (MAR007_MORPHOLOGY_SOURCE_YEARS, "MAR-007 morphology source"),
        ((FUGRO_2018_SURVEY_YEAR, FUGRO_2018_SURVEY_YEAR), "2018 Fugro / freespan survey epoch"),
        (CURRENT_HYDRODYNAMIC_FORCING_YEARS, "Current hydrodynamic forcing"),
    ):
        mid = sum(years) / 2.0
        ax.axvspan(years[0] - 0.4, years[-1] + 0.4, color="tab:red", alpha=0.08, zorder=0)
        ax.annotate(
            ref_label,
            (mid, len(ordered) + 0.3 if len(ordered) else 0.3),
            ha="center",
            fontsize=7,
            color="tab:red",
            fontweight="bold",
            rotation=90 if years[0] != years[-1] else 0,
        )

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered["survey_id"].tolist(), fontsize=7)
    ax.set_xlabel("Acquisition year")
    ax.set_xlim(1985, 2030)
    ax.set_ylim(-1, len(ordered) + 1)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(
        0.01,
        0.01,
        "MAR-016 INVENTORIES EVIDENCE AND ACCESS ONLY. NO NEW MORPHOLOGY, FREESPAN PREDICTION "
        "OR SUSCEPTIBILITY SCORE HAS BEEN CREATED.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.2",
        wrap=True,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

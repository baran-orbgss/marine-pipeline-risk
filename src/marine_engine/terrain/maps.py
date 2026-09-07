"""Generic high-resolution terrain figures (MAR-020 Sections 10-11).

Zero dependency on any specific project or dataset: every title, label,
and context string is an explicit parameter, never hard-coded to
Sheringham Shoal or any other benchmark. Self-contained (small
cartographic helpers re-implemented locally rather than imported from
`evidence_atlas.maps`), matching this project's established map-module-
isolation convention.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede pyplot.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def _add_scale_bar(ax, *, extent: tuple[float, float, float, float]) -> None:
    minx, maxx, miny, maxy = extent
    view_width_m = abs(maxx - minx)
    bar_m = 100.0
    for candidate_m in (10, 25, 50, 100, 250, 500, 1000, 2500, 5000):
        if candidate_m <= view_width_m * 0.3:
            bar_m = candidate_m
        else:
            break
    x0 = minx + view_width_m * 0.04
    y0 = miny + abs(maxy - miny) * 0.05
    ax.add_line(Line2D([x0, x0 + bar_m], [y0, y0], color="black", linewidth=2.2))
    label = f"{bar_m / 1000:g} km" if bar_m >= 1000 else f"{bar_m:g} m"
    ax.annotate(
        label,
        (x0 + bar_m / 2.0, y0),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=8,
    )


def _add_north_arrow(ax, *, extent: tuple[float, float, float, float]) -> None:
    minx, maxx, miny, maxy = extent
    x = maxx - abs(maxx - minx) * 0.06
    y0 = miny + abs(maxy - miny) * 0.08
    y1 = y0 + abs(maxy - miny) * 0.07
    ax.annotate(
        "N",
        xy=(x, y1),
        xytext=(x, y0),
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.5},
        ha="center",
        fontsize=9,
        fontweight="bold",
    )


def _raster_extent(transform, width: int, height: int) -> tuple[float, float, float, float]:
    minx = transform.c
    maxy = transform.f
    maxx = minx + transform.a * width
    miny = maxy + transform.e * height
    return minx, maxx, miny, maxy


# --- Section 10: bathymetry QA map (data readiness, never a hazard map) ---------------------


def render_bathymetry_qa_map(
    *,
    elevation: np.ndarray,
    valid_mask: np.ndarray,
    transform,
    crs_label: str,
    native_pixel_size_m: float,
    vertical_datum: str | None,
    survey_epoch: str | None,
    readiness_status: str,
    output_path: Path,
    title: str,
    dpi: int = 150,
) -> Path:
    """A DATA READINESS figure -- valid coverage, nodata, elevation range,
    extent, CRS, resolution, vertical datum, survey epoch. Never a hazard
    map, never risk-coloured."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = elevation.shape
    extent = _raster_extent(transform, width, height)
    minx, maxx, miny, maxy = extent

    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    coverage = np.where(valid_mask, 1.0, 0.0)
    ax_map.imshow(
        coverage, extent=(minx, maxx, miny, maxy), cmap="Greens", vmin=0, vmax=1, origin="upper"
    )
    ax_map.set_title("Valid-cell coverage (green) vs. nodata (white)", fontsize=10)
    ax_map.set_xlabel("Easting (m)")
    ax_map.set_ylabel("Northing (m)")
    _add_scale_bar(ax_map, extent=(minx, maxx, miny, maxy))
    _add_north_arrow(ax_map, extent=(minx, maxx, miny, maxy))
    ax_map.set_aspect("equal")

    valid_values = elevation[valid_mask]
    if valid_values.size:
        ax_hist.hist(valid_values, bins=60, color="#2E6F40", alpha=0.8)
    ax_hist.set_title("bed_elevation_m distribution\n(valid cells only)", fontsize=10)
    ax_hist.set_xlabel("bed_elevation_m (higher = shallower)")
    ax_hist.set_ylabel("cell count")

    valid_fraction = float(valid_mask.mean()) if valid_mask.size else 0.0
    info_lines = [
        f"Readiness: {readiness_status}",
        f"CRS: {crs_label}",
        f"Native pixel size: {native_pixel_size_m:g} m",
        f"Vertical datum: {vertical_datum or 'unknown'}",
        f"Survey epoch: {survey_epoch or 'unknown'}",
        f"Extent: {width} x {height} px "
        f"({(maxx - minx) / 1000:.2f} x {(maxy - miny) / 1000:.2f} km)",
        f"Valid cells: {valid_fraction:.1%}",
    ]
    if valid_values.size:
        info_lines.append(
            f"bed_elevation_m range: {float(valid_values.min()):.2f} to "
            f"{float(valid_values.max()):.2f} m"
        )
    fig.text(
        0.5,
        -0.02,
        "  |  ".join(info_lines),
        ha="center",
        fontsize=8.5,
        wrap=True,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 11: terrain atlas ----------------------------------------------------------------


def render_terrain_atlas(
    *,
    layers: dict[str, tuple[np.ndarray, str, str]],
    transform,
    output_path: Path,
    title: str,
    subtitle: str,
    footer_note: str,
    dpi: int = 150,
) -> Path:
    """`layers`: ordered dict of panel_label -> (array, cmap, colorbar_label).
    Minimum 4 panels (A. Bathymetry, B. Slope, C. Local relief, D.
    Ruggedness), professional layout, no risk colours or risk language.
    Panel geometry is derived from the real content's own aspect ratio,
    never a fixed roughly-square grid -- an elongated survey swath (this
    benchmark's own real shape: a narrow ~1:2.7 width:height corridor)
    would otherwise render as a thin strip inside a mostly-blank square
    cell, the same class of layout defect already root-caused (via direct
    `ax.get_position()` introspection) and fixed in the MAR-018/019
    evidence atlas."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = next(iter(layers.values()))[0].shape
    minx, maxx, miny, maxy = _raster_extent(transform, width, height)
    content_aspect = abs(maxx - minx) / abs(maxy - miny)  # width_m / height_m; <1 = portrait

    n_panels = len(layers)
    # Portrait content is best spread across COLUMNS (each column narrow-but-tall); landscape
    # content is best spread across ROWS (each row short-but-wide) -- a fixed 2-column grid, as
    # if content were always roughly square, is what produced the mostly-blank cells.
    n_cols = min(n_panels, 4) if content_aspect < 0.75 else min(n_panels, 2)
    n_rows = -(-n_panels // n_cols)  # ceil division

    target_long_side_in = 9.0
    if content_aspect >= 1.0:
        panel_map_width_in = target_long_side_in
        panel_map_height_in = target_long_side_in / content_aspect
    else:
        panel_map_height_in = target_long_side_in
        panel_map_width_in = target_long_side_in * content_aspect
    colorbar_and_margin_in = 1.7
    title_budget_in = 1.3
    footer_budget_in = 0.4
    fig_width_in = (panel_map_width_in + colorbar_and_margin_in) * n_cols
    fig_height_in = panel_map_height_in * n_rows + title_budget_in + footer_budget_in

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width_in, fig_height_in))
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, (array, cmap, cbar_label)) in zip(axes, layers.items(), strict=False):
        finite = np.isfinite(array)
        if finite.any():
            vmin, vmax = np.nanpercentile(array[finite], [2, 98])
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(
            array, extent=(minx, maxx, miny, maxy), cmap=cmap, vmin=vmin, vmax=vmax, origin="upper"
        )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        _add_scale_bar(ax, extent=(minx, maxx, miny, maxy))
        cbar = fig.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    for ax in axes[len(layers) :]:
        ax.axis("off")
    _add_north_arrow(axes[0], extent=(minx, maxx, miny, maxy))

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=10, style="italic", color="0.25")
    fig.text(0.5, 0.005, footer_note, ha="center", fontsize=8.5, style="italic", color="0.25")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px

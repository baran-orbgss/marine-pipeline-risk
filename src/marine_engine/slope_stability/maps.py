"""Slope-instability screening figure (MAR-031 Section 27).

Self-contained (small cartographic helpers implemented locally, matching the repository's
map-module-isolation convention). Sequential perceptual colormaps only -- never traffic-light
hazard colours, never LOW/MODERATE/HIGH classes. Every title/label is an explicit parameter.

Display decimation: the figure may render a strided VIEW of a very large raster (hundreds of
millions of cells) purely to bound plotting memory. This affects the PNG only -- the canonical
GeoTIFF products are always written at full native resolution with no resampling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede pyplot.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Perceptually-uniform sequential maps only; no diverging/traffic-light map is permitted here.
ALLOWED_COLORMAPS: frozenset[str] = frozenset({"viridis", "cividis", "magma", "inferno", "plasma"})


def _raster_extent(transform, width: int, height: int) -> tuple[float, float, float, float]:
    minx = transform.c
    maxy = transform.f
    maxx = minx + transform.a * width
    miny = maxy + transform.e * height
    return minx, maxx, miny, maxy


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


def display_stride_for(shape: tuple[int, int], *, max_display_side_px: int = 4000) -> int:
    """Stride so the longest side of the DISPLAYED array is <= `max_display_side_px`. 1 = no
    decimation. Visual-only; never applied to GIS products."""

    longest = max(shape)
    return max(1, -(-longest // max_display_side_px))


def decimate_for_display(array: np.ndarray, stride: int) -> np.ndarray:
    """Plain strided subsample (every `stride`-th row/col). Not an average, not an interpolation
    -- a subsample of real cell values for plotting only."""

    if stride <= 1:
        return array
    return array[::stride, ::stride]


def render_slope_instability_screening_figure(
    *,
    layers: dict[str, tuple[np.ndarray, str, str]],
    transform,
    full_shape: tuple[int, int],
    output_path: Path,
    title: str,
    subtitle: str,
    footer_note: str,
    dpi: int = 150,
) -> Path:
    """`layers`: ordered dict of panel_label -> (array, cmap, colorbar_label). Arrays may already
    be display-decimated; `full_shape` + `transform` define the true georeferenced extent so a
    decimated view still plots at the correct real-world coordinates. Layout follows the real
    content aspect ratio (portrait swaths spread across columns)."""

    for label, (_array, cmap, _cbar) in layers.items():
        if cmap not in ALLOWED_COLORMAPS:
            raise ValueError(
                f"panel {label!r} uses colormap {cmap!r}; only sequential perceptual maps "
                f"{sorted(ALLOWED_COLORMAPS)} are permitted (no traffic-light hazard colours)"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = full_shape
    minx, maxx, miny, maxy = _raster_extent(transform, width, height)
    content_aspect = abs(maxx - minx) / abs(maxy - miny)

    n_panels = len(layers)
    n_cols = min(n_panels, 4) if content_aspect < 0.75 else min(n_panels, 2)
    n_rows = -(-n_panels // n_cols)

    target_long_side_in = 9.0
    if content_aspect >= 1.0:
        panel_w_in = target_long_side_in
        panel_h_in = target_long_side_in / content_aspect
    else:
        panel_h_in = target_long_side_in
        panel_w_in = target_long_side_in * content_aspect
    fig_w_in = (panel_w_in + 1.7) * n_cols
    fig_h_in = panel_h_in * n_rows + 1.3 + 0.4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w_in, fig_h_in))
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, (array, cmap, cbar_label)) in zip(axes, layers.items(), strict=False):
        finite = np.isfinite(array)
        if finite.any():
            vmin, vmax = np.nanpercentile(array[finite], [2, 98])
            if vmin == vmax:
                vmin, vmax = float(np.nanmin(array)), float(np.nanmax(array))
            if vmin == vmax:
                vmax = vmin + 1.0
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
    return image.shape[1], image.shape[0]

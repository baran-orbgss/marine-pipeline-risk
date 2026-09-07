"""Generic multi-epoch seabed-change figures (MAR-021 Sections 19-20).

Zero dependency on any specific project or dataset -- every title, label,
and context string is an explicit parameter. Self-contained cartographic
helpers (matching this project's established map-module-isolation
convention: each package's maps.py owns its own small helpers rather than
importing another package's).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

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


def render_seabed_change_map(
    *,
    bed_elevation_epoch1_m: np.ndarray,
    bed_elevation_epoch2_m: np.ndarray,
    delta_bed_elevation_m: np.ndarray,
    qa_panel_array: np.ndarray,
    qa_panel_label: str,
    qa_panel_cmap: str,
    transform,
    output_path: Path,
    title: str,
    epoch1_label: str,
    epoch2_label: str,
    dpi: int = 150,
) -> Path:
    """4-panel primary change map: A. epoch1 elevation, B. epoch2
    elevation, C. DoD (diverging, centred exactly at 0), D. uncertainty/
    comparison QA. No hazard score, no risk colours or language.

    Panel geometry is derived from the REAL content aspect ratio (this
    benchmark's own survey footprint is tall/narrow -- a naive fixed 2x2
    grid of roughly-square panels leaves most of each panel blank once
    `set_aspect("equal")` shrinks the image down to fit, the same class of
    layout defect already root-caused and fixed for the MAR-018/019 atlas
    and the MAR-020 terrain atlas). A single ROW of 4 panels, each sized to
    the content's own aspect ratio, uses the space efficiently for both
    portrait and landscape content -- never a hardcoded assumption about
    which shape a future benchmark will have.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = delta_bed_elevation_m.shape
    minx, maxx, miny, maxy = _raster_extent(transform, width, height)

    content_aspect = height / width  # >1 = taller than wide
    panel_width_in = 4.6
    panel_height_in = panel_width_in * content_aspect
    title_budget_in = 1.3
    fig_height_in = panel_height_in + title_budget_in
    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(
        1,
        4,
        figsize=(4 * panel_width_in * 1.55, fig_height_in),
    )
    fig.subplots_adjust(top=1.0 - (title_budget_in / fig_height_in), wspace=0.55)

    finite1 = np.isfinite(bed_elevation_epoch1_m)
    finite2 = np.isfinite(bed_elevation_epoch2_m)
    if finite1.any() and finite2.any():
        vmin = min(
            np.nanpercentile(bed_elevation_epoch1_m[finite1], 2),
            np.nanpercentile(bed_elevation_epoch2_m[finite2], 2),
        )
        vmax = max(
            np.nanpercentile(bed_elevation_epoch1_m[finite1], 98),
            np.nanpercentile(bed_elevation_epoch2_m[finite2], 98),
        )
    else:
        vmin, vmax = -1.0, 1.0

    for ax, array, label in (
        (ax_a, bed_elevation_epoch1_m, f"A. {epoch1_label} bed_elevation_m"),
        (ax_b, bed_elevation_epoch2_m, f"B. {epoch2_label} bed_elevation_m"),
    ):
        im = ax.imshow(
            array,
            extent=(minx, maxx, miny, maxy),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
        )
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label("m (higher = shallower)", fontsize=8)

    finite_delta = np.isfinite(delta_bed_elevation_m)
    if finite_delta.any():
        abs_bound = float(np.nanpercentile(np.abs(delta_bed_elevation_m[finite_delta]), 98)) or 1.0
    else:
        abs_bound = 1.0
    im_c = ax_c.imshow(
        delta_bed_elevation_m,
        extent=(minx, maxx, miny, maxy),
        cmap="RdBu",
        vmin=-abs_bound,
        vmax=abs_bound,
        origin="upper",
    )
    ax_c.set_title("C. Change (epoch2 - epoch1)", fontsize=10, fontweight="bold")
    ax_c.set_aspect("equal")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    _add_scale_bar(ax_c, extent=(minx, maxx, miny, maxy))
    _add_north_arrow(ax_c, extent=(minx, maxx, miny, maxy))
    cbar_c = fig.colorbar(im_c, ax=ax_c, shrink=0.75)
    cbar_c.set_label("m (negative = lowering, positive = raising)", fontsize=8)

    finite_qa = np.isfinite(qa_panel_array)
    if finite_qa.any():
        qa_vmin, qa_vmax = np.nanpercentile(qa_panel_array[finite_qa], [2, 98])
    else:
        qa_vmin, qa_vmax = 0.0, 1.0
    im_d = ax_d.imshow(
        qa_panel_array,
        extent=(minx, maxx, miny, maxy),
        cmap=qa_panel_cmap,
        vmin=qa_vmin,
        vmax=qa_vmax,
        origin="upper",
    )
    ax_d.set_title("D. |Change| (QA context)", fontsize=10, fontweight="bold")
    ax_d.set_aspect("equal")
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    cbar_d = fig.colorbar(im_d, ax=ax_d, shrink=0.75)
    cbar_d.set_label(qa_panel_label, fontsize=8)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.965,
        "Observed multi-epoch seabed elevation change -- no hazard/risk score, no future "
        "prediction",
        ha="center",
        fontsize=10,
        style="italic",
        color="0.25",
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_change_distribution_qa(
    *,
    delta_bed_elevation_m: np.ndarray,
    source_residuals_m: np.ndarray | None,
    common_support_summary: dict[str, Any],
    output_path: Path,
    title: str,
    dpi: int = 150,
) -> Path:
    """QA/distribution figure: DoD histogram, robust distribution summary,
    source-comparator residual distribution (if available), overlap/
    support summary. No predictive distribution is fit."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_panels = 3 if source_residuals_m is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5))
    if n_panels == 2:
        ax_hist, ax_support = axes
    else:
        ax_hist, ax_resid, ax_support = axes

    finite = delta_bed_elevation_m[np.isfinite(delta_bed_elevation_m)]
    ax_hist.hist(finite, bins=80, color="#4C6EF5", alpha=0.85)
    median = float(np.median(finite)) if finite.size else 0.0
    ax_hist.axvline(
        median, color="black", linestyle="--", linewidth=1.2, label=f"median={median:.3f} m"
    )
    ax_hist.set_title("Observed DoD distribution", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("delta_bed_elevation_m")
    ax_hist.set_ylabel("cell count")
    ax_hist.legend(fontsize=8)

    if source_residuals_m is not None:
        finite_resid = source_residuals_m[np.isfinite(source_residuals_m)]
        ax_resid.hist(finite_resid, bins=80, color="#F08C00", alpha=0.85)
        ax_resid.set_title(
            "Residual vs. source-produced comparator", fontsize=11, fontweight="bold"
        )
        ax_resid.set_xlabel("residual (m)")
        ax_resid.set_ylabel("cell count")

    labels = list(common_support_summary.keys())
    values = [common_support_summary[k] for k in labels]
    ax_support.barh(labels, values, color="#2E6F40")
    ax_support.set_title("Coverage / support summary", fontsize=11, fontweight="bold")
    ax_support.set_xlabel("fraction or count")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px

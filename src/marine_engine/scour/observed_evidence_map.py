"""Sheringham Shoal 2024 observed-scour-evidence map (MAR-023 Section 18).

Self-contained, zero dependency on any other map module (see
`susceptibility_map.py`/`freespan_evidence_map.py` docstrings for this
project's established rationale). Shows ONLY observed/source-interpreted
evidence plus subdued background context -- never a pipeline scour
susceptibility claim, since this survey covers windfarm infrastructure
(monopiles, cables, protection), not a pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px


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
            alpha=0.30,
            zorder=0,
        )
    except Exception as exc:  # noqa: BLE001 -- background context is optional, never fatal
        print(f"note: could not render background bathymetry context: {exc}", file=sys.stderr)


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


_CATEGORY_COLORS = {
    "SOURCE_INTERPRETED_OBSERVED_SCOUR_EVIDENCE": "#8b0000",
    "SOURCE_INTERPRETED_EXPOSURE_EVIDENCE": "#d1495b",
    "ANTHROPOGENIC_DISTURBANCE_CONTEXT": "#edae49",
    "ASSET_INFRASTRUCTURE_CONTEXT": "#2e6f95",
    "SEABED_OBJECT_CONTEXT": "#66a182",
    "UNCLASSIFIED_INTERPRETATION_FEATURE": "0.5",
}


def render_observed_scour_evidence_map(
    *,
    evidence_gdf: Any,
    output_path: Path,
    background_raster_path: Path | None = None,
    background_raster_label: str | None = None,
    explicit_scour_evidence_present: bool = True,
    title: str = "Sheringham Shoal 2024 — Source-Interpreted Scour / Integrity Context",
    subtitle: str = "Source-interpreted survey targets, classified exactly as source-stated",
    dpi: int = 150,
) -> Path:
    """Shows ONLY the observed/source-interpreted evidence plus subdued background context
    (Section 18) -- title is never "Pipeline Scour Susceptibility": this survey covers
    windfarm infrastructure, not a pipeline.

    MAR-023A Section 6: the title itself is never "Observed Scour Evidence" -- most (usually
    all) of what this map shows is general integrity context (exposure, infrastructure,
    disturbance, seabed objects), not scour. When `explicit_scour_evidence_present` is False
    (the real Sheringham 2024 result), a prominent banner states that plainly rather than
    letting the legend's mere presence of colour categories imply otherwise."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if evidence_gdf is not None and not evidence_gdf.empty:
        minx, miny, maxx, maxy = evidence_gdf.total_bounds
    else:
        minx, miny, maxx, maxy = 0.0, 0.0, 1.0, 1.0
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    fig_height = 8.0
    fig_width = max(7.0, min(16.0, fig_height * (span_x / span_y)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pad_x, pad_y = span_x * 0.08, span_y * 0.08
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if background_raster_path is not None and Path(background_raster_path).exists():
        _plot_background_raster(ax, Path(background_raster_path))

    if evidence_gdf is not None and not evidence_gdf.empty:
        for category, group in evidence_gdf.groupby("interpretation_category"):
            color = _CATEGORY_COLORS.get(str(category), "0.5")
            group.plot(ax=ax, color=color, markersize=14, marker="o", label=str(category), zorder=5)
        legend_handles = [
            Patch(facecolor=color, edgecolor="0.2", label=category.replace("_", " ").title())
            for category, color in _CATEGORY_COLORS.items()
            if category in evidence_gdf["interpretation_category"].unique()
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.9)

    _add_scale_bar(ax)
    _add_north_arrow(ax)

    # Fixed headroom (whether or not the banner below actually renders, so the figure layout
    # is identical either way) for three stacked lines -- suptitle, the explicit-scour-absence
    # banner, and the axes' own subtitle -- avoiding the title/subtitle collision class of bug
    # already found and fixed elsewhere in this ticket (MAR-023's `susceptibility_map.py`).
    fig.subplots_adjust(top=0.83)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    if not explicit_scour_evidence_present:
        fig.text(
            0.5,
            0.935,
            "NO EXPLICIT SOURCE-INTERPRETED SCOUR FEATURES WERE PRESENT",
            transform=fig.transFigure,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#8b0000",
        )
    ax.set_title(subtitle, fontsize=9, style="italic", pad=14)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    footer_lines = [
        "Source: The Crown Estate Marine Data Exchange, 2024 XOCEAN Sheringham Shoal Seabed "
        "Monitoring Survey (TCE-3974) interpretation package.",
        "Evidence classified exactly as source-stated; no pipeline scour, cable scour, or "
        "foundation scour is inferred beyond what the source attribution states.",
    ]
    if background_raster_label:
        footer_lines.insert(0, f"Background: {background_raster_label}.")
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

"""Pipeline scour susceptibility map rendering (MAR-023 Sections 16-17).

Self-contained, zero dependency on any other map module -- matches this
project's established map-module-isolation convention (see
`freespan_evidence_map.py`'s docstring): a future change here can never
destabilize an already-shipped map, and vice versa. Small figure helpers
(`_add_kp_labels`/`_add_scale_bar`/`_add_north_arrow`/`_content_bounds`/
`_plot_background_raster`/`read_png_dimensions`) are duplicated rather than
imported from `scour_onset_map.py` for the same reason.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from marine_engine.preprocessing.chainage import format_kp_label

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px


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


# --- Section 17: generic, future-operator-ready map ------------------------------------------


def render_pipeline_scour_susceptibility_map(
    *,
    segments_gdf: Any,
    route: LineString,
    working_crs: str,
    output_path: Path,
    diameter_m: float,
    value_column: str = "embedment_protection_margin_p95_e_over_D",
    value_label: str = "Embedment protection margin p95, e/D (actual - critical)",
    higher_is_worse: bool = False,
    background_raster_path: Path | None = None,
    title: str = "Pipeline Scour-Onset Susceptibility Screening",
    subtitle: str = (
        "Actual embedment vs. Marini et al. (2024) tested critical embedment screening class"
    ),
    unavailable_message: str | None = None,
    dpi: int = 150,
) -> Path:
    """Colours the route by a genuine PHYSICAL quantity (a signed embedment margin, in e/D,
    or a descriptive exceedance fraction) -- never normalized to 0-1 merely for presentation
    (Section 17). Works for any operator route once `segments_gdf` carries `value_column` and
    geometry; every value in this repository's own real CLI run is null (Section 6: PL854 has
    no actual embedment profile), so `unavailable_message` is shown prominently instead of a
    fabricated route colouring."""

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

    has_value = (
        segments_gdf is not None
        and not segments_gdf.empty
        and value_column in segments_gdf.columns
        and segments_gdf[value_column].notna().any()
    )
    if has_value:
        values = segments_gdf[value_column].astype(float)
        vmax = float(np.nanmax(np.abs(values))) or 1.0
        cmap = plt.get_cmap("RdYlGn_r" if higher_is_worse else "RdYlGn")
        norm = (
            matplotlib.colors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
            if not higher_is_worse
            else matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
        )
        segments_gdf.plot(column=value_column, cmap=cmap, norm=norm, linewidth=4, ax=ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(value_label, fontsize=8)
    elif segments_gdf is not None and not segments_gdf.empty:
        segments_gdf.plot(color="0.6", linewidth=4, ax=ax)

    _add_kp_labels(ax, route)
    _add_scale_bar(ax)
    _add_north_arrow(ax)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=9, style="italic", pad=14)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if unavailable_message:
        fig.text(
            0.5,
            0.5,
            unavailable_message,
            transform=fig.transFigure,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="0.15",
            bbox={"boxstyle": "round,pad=0.6", "fc": "white", "ec": "0.3", "alpha": 1.0},
            zorder=10,
            wrap=True,
        )

    footer_lines = [
        f"Pipeline diameter D={diameter_m:.4f} m. Marini et al. (2024) tested critical "
        "embedment screening class, never an unconstrained continuous critical embedment.",
        "A positive margin is an empirical onset-screening result only -- never SAFE, DESIGN "
        "ACCEPTABLE, or NO SCOUR.",
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


# --- Section 16: PL854 tested-embedment scenario envelope profile ---------------------------

_SCENARIO_COLOR_BY_RATIO = {
    0.00: "#8b0000",
    0.03: "#d1495b",
    0.06: "#edae49",
    0.10: "#66a182",
    0.15: "#2e5339",
}


def render_scour_onset_scenario_envelope_map(
    *,
    envelope_summary_df: pd.DataFrame,
    total_length_m: float,
    diameter_m: float,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Section 16: chainage profile of `scour_onset_screening_exceedance_fraction` for each
    of the five TESTED embedment scenarios (never collapsed to one best estimate), plus the
    (scenario-independent) `critical_embedment_ratio_p95` from the forcing record. Always
    prominently labelled SCENARIO SCREENING ONLY -- these are tested ratios treated as an
    "actual" value in turn, never an observed continuous PL854 embedment profile."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12.0, 8.0), sharex=True, height_ratios=[2.0, 1.0]
    )

    if not envelope_summary_df.empty:
        for scenario_ratio, group in envelope_summary_df.groupby("tested_embedment_scenario_ratio"):
            ordered = group.sort_values("start_chainage_m")
            color = _SCENARIO_COLOR_BY_RATIO.get(float(scenario_ratio), "0.5")
            for _, row in ordered.iterrows():
                fraction = row["scour_onset_screening_exceedance_fraction"]
                if fraction is None or (isinstance(fraction, float) and pd.isna(fraction)):
                    continue
                x = [row["start_chainage_m"] / 1000.0, row["end_chainage_m"] / 1000.0]
                ax_top.plot(
                    x,
                    [fraction, fraction],
                    color=color,
                    linewidth=2.5,
                    solid_capstyle="butt",
                    label=f"{scenario_ratio:g} D",
                )

        # de-duplicate the legend (one entry per scenario, not per section)
        handles, labels = ax_top.get_legend_handles_labels()
        seen: dict[str, Any] = {}
        for handle, label in zip(handles, labels, strict=True):
            seen.setdefault(label, handle)
        ax_top.legend(
            seen.values(),
            seen.keys(),
            title="Tested embedment scenario",
            fontsize=8,
            loc="upper right",
        )

        reference = envelope_summary_df[
            envelope_summary_df["tested_embedment_scenario_ratio"]
            == envelope_summary_df["tested_embedment_scenario_ratio"].min()
        ].sort_values("start_chainage_m")
        for _, row in reference.iterrows():
            p95 = row["critical_embedment_ratio_p95"]
            if p95 is None or (isinstance(p95, float) and pd.isna(p95)):
                continue
            x = [row["start_chainage_m"] / 1000.0, row["end_chainage_m"] / 1000.0]
            ax_bottom.plot(x, [p95, p95], color="tab:blue", linewidth=2.5, solid_capstyle="butt")

    ax_top.set_ylabel("Scour-onset screening\nexceedance fraction")
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.grid(True, alpha=0.3)
    # Both rows use plain axes-fraction `text()` (not a mix of `text()` and `set_title()`,
    # whose position is governed by a separate points-based `pad`) so their vertical
    # separation is directly controllable and does not collide regardless of figure size.
    ax_top.text(
        0.0,
        1.03,
        "NOT a probability of scour. NOT a failure probability. NOT a return-period metric.",
        transform=ax_top.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.2",
        style="italic",
    )

    ax_bottom.set_ylabel("Critical embedment\np95, e/D")
    ax_bottom.set_ylim(
        -0.01,
        max(
            0.16,
            envelope_summary_df["critical_embedment_ratio_p95"].max()
            if not envelope_summary_df.empty
            and envelope_summary_df["critical_embedment_ratio_p95"].notna().any()
            else 0.16,
        ),
    )
    ax_bottom.set_xlabel("Chainage (km)")
    ax_bottom.grid(True, alpha=0.3)

    # Title/subtitle/disclaimer collision fix (matches the established MAR-018/019/020/022
    # pattern): the suptitle is pinned near the very top edge, and BOTH the red disclaimer and
    # the axes' own title are anchored with `va="top"` so each hangs DOWN from its own anchor
    # point rather than being vertically centered on it -- with the `tight_layout` rect leaving
    # enough headroom above the axes for all three stacked lines to render without overlap.
    fig.suptitle(
        "PL854 — Scour-Onset Tested-Embedment Scenario Envelope",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.94,
        "SCENARIO SCREENING ONLY — NO OBSERVED CONTINUOUS PL854 EMBEDMENT PROFILE",
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#8b0000",
    )
    ax_top.text(
        0.5,
        1.12,
        f"D={diameter_m:.4f} m | route length={total_length_m / 1000.0:.2f} km",
        transform=ax_top.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

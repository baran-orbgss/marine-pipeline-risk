"""Observed burial-state map and KP/route profile view (MAR-024 Sections 10-11).

Self-contained, zero dependency on any other map module (this project's
established map-module-isolation convention). No risk colours anywhere --
the primary map variable is always the real measured physical quantity,
labelled with its own resolved-or-unresolved reference status rather than
asserted to be a validated "depth of burial".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from shapely.geometry import LineString

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


# --- Section 10: observed burial-state map ----------------------------------------------------


def render_observed_burial_state_map(
    *,
    route: LineString,
    profile_df: pd.DataFrame,
    output_path: Path,
    value_label: str,
    title: str = "Barrow 2016 — Observed Burial State",
    subtitle: str = "Measured values shown exactly as recorded; source-interpreted exposure "
    "shown independently",
    dpi: int = 150,
) -> Path:
    """Section 10: authoritative route, measured values along route, source-interpreted
    exposures shown independently, coverage/gaps visible from the plotted point density. No
    risk colours -- the colour scale is always the real physical quantity."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = route.bounds
    if not profile_df.empty:
        minx = min(minx, profile_df["x_m"].min())
        maxx = max(maxx, profile_df["x_m"].max())
        miny = min(miny, profile_df["y_m"].min())
        maxy = max(maxy, profile_df["y_m"].max())
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    fig_height = 8.0
    fig_width = max(7.0, min(16.0, fig_height * (span_x / span_y)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pad_x, pad_y = span_x * 0.06, span_y * 0.08
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.plot(*route.xy, color="0.6", linewidth=2.5, zorder=1, label="Authoritative route")

    if not profile_df.empty:
        non_exposed = profile_df[~profile_df["source_interpreted_exposure"]]
        exposed = profile_df[profile_df["source_interpreted_exposure"]]
        if not non_exposed.empty:
            sc = ax.scatter(
                non_exposed["x_m"],
                non_exposed["y_m"],
                c=non_exposed["measured_burial_value_m"],
                cmap="viridis",
                s=10,
                zorder=3,
                label="Measured value",
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label(value_label, fontsize=8)
        if not exposed.empty:
            ax.scatter(
                exposed["x_m"],
                exposed["y_m"],
                facecolor="none",
                edgecolor="#8b0000",
                linewidth=1.2,
                s=28,
                zorder=4,
                label="Source-interpreted exposure",
            )

    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    _add_scale_bar(ax)
    _add_north_arrow(ax)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=9, style="italic", pad=14)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    footer_lines = [
        "Source: The Crown Estate Marine Data Exchange, 2016 Deep BV Barrow Offshore Wind "
        "Farm Export Cable Geophysical Depth of Burial Survey (TCE-48).",
        "Exposure shown here is source-interpreted evidence only, never inferred from the "
        "measured value's numeric sign or magnitude.",
    ]
    ax.text(
        0.0,
        -0.08,
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


# --- Section 11: KP / route profile view -------------------------------------------------------


def render_burial_kp_profile(
    *,
    profile_df: pd.DataFrame,
    output_path: Path,
    value_label: str,
    target_burial_m: float | None = None,
    coverage_gaps_m: list[tuple[float, float]] | None = None,
    title: str = "Barrow 2016 — Burial KP Profile",
    dpi: int = 150,
) -> Path:
    """Section 11: X = canonical KP (km), Y = measured value. Source-interpreted exposure,
    unsurveyed gaps, and QA flags are all overlaid independently. A source-stated target
    burial (if any) is shown labelled `SOURCE_STATED_TARGET_BURIAL`, never as a universal
    engineering requirement."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.0, 5.5))

    if not profile_df.empty:
        kp_km = profile_df["chainage_m"] / 1000.0
        non_exposed = profile_df[~profile_df["source_interpreted_exposure"]]
        exposed = profile_df[profile_df["source_interpreted_exposure"]]

        ax.plot(
            kp_km.loc[non_exposed.index],
            non_exposed["measured_burial_value_m"],
            color="tab:blue",
            linewidth=0.8,
            marker=".",
            markersize=2,
            label="Measured value",
        )
        if not exposed.empty:
            ax.scatter(
                kp_km.loc[exposed.index],
                exposed["measured_burial_value_m"],
                facecolor="none",
                edgecolor="#8b0000",
                linewidth=1.2,
                s=28,
                zorder=4,
                label="Source-interpreted exposure",
            )

        qa_rows = profile_df[profile_df["qa_flags"].notna()]
        if not qa_rows.empty:
            ax.scatter(
                kp_km.loc[qa_rows.index],
                qa_rows["measured_burial_value_m"],
                facecolor="none",
                edgecolor="#edae49",
                marker="s",
                s=40,
                zorder=5,
                label="QA flag",
            )

    if target_burial_m is not None:
        ax.axhline(
            target_burial_m,
            color="tab:green",
            linestyle="--",
            linewidth=1.2,
            label=f"SOURCE_STATED_TARGET_BURIAL ({target_burial_m:g} m)",
        )

    for gap_start_m, gap_end_m in coverage_gaps_m or []:
        ax.axvspan(gap_start_m / 1000.0, gap_end_m / 1000.0, color="0.85", zorder=0)

    ax.set_xlabel("KP (km)")
    ax.set_ylabel(value_label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(
        "Grey bands (if any) mark unsurveyed gaps -- never interpolated over",
        fontsize=9,
        style="italic",
        pad=14,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

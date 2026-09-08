"""Generic bedform-morphodynamics figures (MAR-022 Sections 19-22).

Zero dependency on any specific project or dataset -- every title, label,
and context string is an explicit parameter. Self-contained cartographic
helpers (matching this project's established map-module-isolation
convention). No hazard/risk colouring or language anywhere in this
module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _safe_masked_array(elevation: np.ndarray, valid: np.ndarray) -> np.ma.MaskedArray:
    safe = np.where(valid, elevation, np.nan).astype(np.float64)
    return np.ma.masked_invalid(safe)


def _raster_extent(transform, width: int, height: int) -> tuple[float, float, float, float]:
    minx = transform.c
    maxy = transform.f
    maxx = minx + transform.a * width
    miny = maxy + transform.e * height
    return minx, maxx, miny, maxy


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    return image.shape[1], image.shape[0]


# --- Section 19: primary bedform morphometry map ---------------------------------------------


def render_bedform_morphometry_map(
    *,
    background_elevation: np.ndarray,
    background_valid: np.ndarray,
    transform,
    canonical_tiles_df: pd.DataFrame,
    transect_endpoints: list[tuple[tuple[float, float], tuple[float, float]]],
    output_path: Path,
    title: str,
    unavailable_message: str | None = None,
    dpi: int = 150,
) -> Path:
    """A. 2020 bathymetry. B. natural-bedform context (natural-eligible
    tiles vs anthropogenic-disturbed tiles). C. dominant wavelength per
    tile. D. crest orientation + the representative tile's own
    transects. `canonical_tiles_df` must provide `center_x_m`/
    `center_y_m`/`tile_size_m`/`natural_bedform_validation_status`/
    `dominant_wavelength_m`/`dominant_crest_azimuth_deg` -- may be empty
    (a real, valid outcome when canonical support is zero). MAR-022A
    Section 15: when `canonical_tiles_df` is empty, `unavailable_message`
    must be set and is displayed prominently -- never silently rendering
    an empty-looking figure as if it were a normal result."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = background_elevation.shape
    extent = _raster_extent(transform, width, height)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    masked_bg = _safe_masked_array(background_elevation, background_valid)

    im_a = ax_a.imshow(masked_bg, cmap="viridis", extent=extent, origin="upper")
    ax_a.set_title("A. 2020 bathymetry (bed_elevation_m)", fontsize=11, fontweight="bold")
    fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)

    ax_b.imshow(masked_bg, cmap="gray", extent=extent, origin="upper", alpha=0.6)
    status_style = {
        "NATURAL_SEABED_ELIGIBLE": ("tab:green", "Natural-seabed eligible"),
        "ANTHROPOGENIC_DISTURBANCE_PRESENT": ("tab:red", "Anthropogenic disturbance present"),
        "INFRASTRUCTURE_CONTEXT_INSUFFICIENT": ("tab:gray", "Infrastructure context insufficient"),
    }
    seen_labels: set[str] = set()
    for _, row in canonical_tiles_df.iterrows():
        color, label = status_style.get(
            row["natural_bedform_validation_status"], ("black", "Unknown")
        )
        half = row["tile_size_m"] / 2.0
        ax_b.add_patch(
            plt.Rectangle(
                (row["center_x_m"] - half, row["center_y_m"] - half),
                row["tile_size_m"],
                row["tile_size_m"],
                fill=False,
                edgecolor=color,
                linewidth=2,
                label=label if label not in seen_labels else None,
                zorder=5,
            )
        )
        seen_labels.add(label)
    ax_b.set_title("B. Natural-bedform validation context", fontsize=11, fontweight="bold")
    if seen_labels:
        ax_b.legend(fontsize=7, loc="lower right")

    ax_c.imshow(masked_bg, cmap="gray", extent=extent, origin="upper", alpha=0.5)
    if not canonical_tiles_df.empty and canonical_tiles_df["dominant_wavelength_m"].notna().any():
        scatter = ax_c.scatter(
            canonical_tiles_df["center_x_m"],
            canonical_tiles_df["center_y_m"],
            c=canonical_tiles_df["dominant_wavelength_m"],
            cmap="plasma",
            s=140,
            edgecolor="black",
            zorder=4,
        )
        fig.colorbar(scatter, ax=ax_c, label="Dominant wavelength (m)", fraction=0.046, pad=0.04)
    ax_c.set_title("C. Dominant wavelength per canonical tile", fontsize=11, fontweight="bold")

    ax_d.imshow(masked_bg, cmap="gray", extent=extent, origin="upper", alpha=0.6)
    for p1, p2 in transect_endpoints:
        ax_d.plot([p1[0], p2[0]], [p1[1], p2[1]], color="tab:red", linewidth=2, zorder=5)
    ax_d.set_title(
        "D. Crest orientation -- representative transects", fontsize=11, fontweight="bold"
    )

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.955,
        "Static bedform geometry only -- no future migration prediction, no hazard/risk score",
        ha="center",
        va="top",
        fontsize=9.5,
        style="italic",
        color="0.25",
    )
    if unavailable_message:
        fig.text(
            0.5,
            0.935,
            unavailable_message,
            ha="center",
            va="top",
            fontsize=11,
            color="tab:red",
            fontweight="bold",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.90 if unavailable_message else 0.93))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 20: multi-epoch bedform change map -----------------------------------------------


def render_bedform_change_map(
    *,
    background_delta_bed_elevation_m: np.ndarray,
    transform,
    crests_epoch1_xy: list[tuple[float, float]],
    crests_epoch2_xy: list[tuple[float, float]],
    matched_pairs: list[dict[str, Any]],
    epoch1_label: str,
    epoch2_label: str,
    output_path: Path,
    title: str,
    vector_exaggeration: float = 15.0,
    subdued: bool = False,
    unavailable_message: str | None = None,
    dpi: int = 150,
) -> Path:
    """Section 20: independently-detected crest positions for both
    epochs, canonical matched pairs as displacement vectors along each
    pair's own local cross-crest normal, and the MAR-021 DoD as subdued
    background context. Displacement vectors are drawn at
    `vector_exaggeration`x real scale (stated explicitly in the legend --
    real crest displacements are metres against a kilometre-scale map,
    invisible at true scale) -- never implying every visible crest was
    successfully tracked (Section 20's explicit caution).

    MAR-022A Section 15: `subdued=True` (for a NONCANONICAL diagnostic
    call) renders every marker/arrow in muted gray, visually subordinate
    to a primary canonical figure -- the caller must still give this a
    visibly different, explicitly noncanonical `title`.
    `unavailable_message` (for a canonical call with zero stable matches)
    is displayed prominently instead of a silently near-empty figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = background_delta_bed_elevation_m.shape
    extent = _raster_extent(transform, width, height)
    # This benchmark's own survey footprint is tall/narrow -- a fixed landscape figure size
    # leaves most of the canvas blank once `imshow`'s implicit equal-aspect scaling shrinks the
    # image to fit (the same class of layout defect already root-caused and fixed for the
    # MAR-018/019/020/021 figures; `render_seabed_change_map` in `change/maps.py` documents the
    # identical fix). Deriving the figure size from the real content aspect ratio avoids it here
    # too, never a hardcoded assumption about which shape a future benchmark will have.
    content_aspect = height / width
    panel_width_in = 7.0
    panel_height_in = panel_width_in * content_aspect
    title_budget_in = 1.3
    fig_height_in = panel_height_in + title_budget_in
    fig, ax = plt.subplots(figsize=(panel_width_in + 2.3, fig_height_in))
    ax.set_aspect("equal")

    finite = np.isfinite(background_delta_bed_elevation_m)
    abs_bound = (
        float(np.nanpercentile(np.abs(background_delta_bed_elevation_m[finite]), 98))
        if finite.any()
        else 1.0
    ) or 1.0
    im = ax.imshow(
        background_delta_bed_elevation_m,
        extent=extent,
        cmap="RdBu",
        vmin=-abs_bound,
        vmax=abs_bound,
        origin="upper",
        alpha=0.55,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("MAR-021 DoD (m) -- subdued context only", fontsize=8)

    epoch1_color = "0.6" if subdued else "tab:blue"
    epoch2_color = "0.5" if subdued else "tab:orange"
    arrow_color = "0.65" if subdued else "black"
    arrow_alpha = 0.5 if subdued else 1.0

    if crests_epoch1_xy:
        xs, ys = zip(*crests_epoch1_xy, strict=True)
        ax.scatter(
            xs, ys, s=22, color=epoch1_color, marker="o", label=f"{epoch1_label} crest", zorder=4
        )
    if crests_epoch2_xy:
        xs, ys = zip(*crests_epoch2_xy, strict=True)
        ax.scatter(
            xs, ys, s=22, color=epoch2_color, marker="^", label=f"{epoch2_label} crest", zorder=4
        )

    for pair in matched_pairs:
        x1, y1 = pair["epoch1_x_m"], pair["epoch1_y_m"]
        dx = (pair["epoch2_x_m"] - x1) * vector_exaggeration
        dy = (pair["epoch2_y_m"] - y1) * vector_exaggeration
        color = (
            arrow_color
            if subdued
            else ("black" if pair["match_status"] == "MATCHED_HIGH_SUPPORT" else "0.4")
        )
        ax.annotate(
            "",
            xy=(x1 + dx, y1 + dy),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 1.6,
                "alpha": arrow_alpha,
            },
            zorder=6,
        )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=8)

    if unavailable_message:
        fig.text(
            0.5,
            0.5,
            unavailable_message,
            ha="center",
            va="center",
            fontsize=13,
            color="tab:red",
            fontweight="bold",
            transform=fig.transFigure,
            wrap=True,
        )

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.955,
        f"Displacement vectors exaggerated {vector_exaggeration:g}x for visibility -- not every "
        "visible crest was successfully matched between epochs",
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
        color="0.25",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 21: per-epoch morphometry statistics ----------------------------------------------


def render_bedform_statistics(
    *,
    bedforms_epoch1_df: pd.DataFrame,
    bedforms_epoch2_df: pd.DataFrame,
    epoch1_label: str,
    epoch2_label: str,
    output_path: Path,
    title: str,
    dpi: int = 150,
) -> Path:
    """Section 21: wavelength/height/asymmetry distributions, epoch1 vs
    epoch2, side by side -- canonical (>=30 m) retained bedforms only. No
    predictive distribution is ever fitted."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    fields = (
        ("wavelength_m", "Wavelength (m)", axes[0]),
        ("wave_height_m", "Wave height (m)", axes[1]),
        ("asymmetry_index", "Asymmetry index", axes[2]),
    )
    for column, label, ax in fields:
        v1 = bedforms_epoch1_df[column].dropna() if column in bedforms_epoch1_df else pd.Series([])
        v2 = bedforms_epoch2_df[column].dropna() if column in bedforms_epoch2_df else pd.Series([])
        bins = min(15, max(len(v1), len(v2), 1))
        if len(v1):
            ax.hist(
                v1, bins=bins, alpha=0.55, color="tab:blue", label=f"{epoch1_label} (n={len(v1)})"
            )
        if len(v2):
            ax.hist(
                v2, bins=bins, alpha=0.55, color="tab:orange", label=f"{epoch2_label} (n={len(v2)})"
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(
        0.01,
        0.01,
        "Canonical (>=30 m trough-to-trough) retained bedforms only. No predictive distribution "
        "has been fitted.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.2",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 22: observed-displacement statistics ----------------------------------------------


def render_displacement_statistics(
    *,
    canonical_matches_df: pd.DataFrame,
    output_path: Path,
    title: str,
    dpi: int = 150,
) -> Path:
    """Section 22: only called when >=3 canonical matched crests exist.
    Absolute displacement, signed normal displacement, apparent
    displacement rate, orientation difference -- no extrapolation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.2))
    fields = (
        ("absolute_normal_displacement_m", "Absolute displacement (m)"),
        ("normal_displacement_m", "Signed normal displacement (m)"),
        ("apparent_rate_m_per_year", "Apparent displacement rate (m/yr)"),
        ("orientation_difference_deg", "Orientation difference (deg)"),
    )
    n = len(canonical_matches_df)
    for (column, label), ax in zip(fields, axes, strict=True):
        values = canonical_matches_df[column].dropna()
        ax.hist(values, bins=min(12, max(len(values), 1)), color="tab:purple", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{title} (n={n} canonical matches)", fontsize=13, fontweight="bold")
    fig.text(
        0.01,
        0.01,
        "Observed apparent displacement between two survey epochs only -- NOT a future migration "
        "rate, NOT extrapolated.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.2",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

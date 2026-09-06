"""Sand-wave morphometry maps (MAR-017).

Generic renderers, no HHW-specific coordinates hard-coded (Section 25) --
every function takes real data (arrays, tile/transect records) from the
caller. Never colours by risk/hazard/freespan language (Section 24); never
fits a predictive distribution to the bedform statistics (Section 23).
"""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px


def _safe_masked_array(elevation: np.ndarray, valid: np.ndarray) -> np.ma.MaskedArray:
    """MAR-017B Section 6: a source's raw nodata sentinel (e.g. float32's
    most-negative value, ~-3.4e38) must never reach matplotlib's colour
    normalization -- `np.ma.masked_where` alone marks a cell invalid but
    leaves that extreme value sitting in the masked array's own `.data`,
    which some matplotlib internals (bin-index normalization) still touch
    and overflow on, even though the final rendered pixel is correctly
    masked/transparent. Replacing invalid cells with `nan` before masking
    removes the extreme value from the underlying buffer entirely -- never
    altering a single valid bathymetry value."""

    safe = np.where(valid, elevation, np.nan).astype(np.float64)
    return np.ma.masked_invalid(safe)


# --- Section 8: first QA map (no morphology interpretation yet) ---------------------------


def render_native_bathymetry_overview(
    *,
    elevation: np.ndarray,
    valid: np.ndarray,
    extent_m: tuple[float, float, float, float],
    crs: str,
    native_pixel_size_m: float,
    survey_year: int,
    nodata_value: float,
    output_path: Path,
    title: str = "HHW CEND 11/11 -- Native Bathymetry Overview",
    dpi: int = 150,
) -> Path:
    """Section 8: full available bathymetry at its TRUE source extent, a
    depth/elevation colourbar, and explicit CRS/grid-spacing/survey-year/
    nodata annotations -- no morphology interpretation of any kind yet."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 9.0))

    masked = _safe_masked_array(elevation, valid)
    im = ax.imshow(masked, cmap="viridis", extent=extent_m, origin="upper")
    fig.colorbar(im, ax=ax, label="Elevation (m)", fraction=0.04, pad=0.03)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    info_lines = [
        f"CRS: {crs}",
        f"Native grid spacing: {native_pixel_size_m:g} m",
        f"Survey year: {survey_year}",
        f"Nodata value: {nodata_value:g}",
        f"Valid cells: {100.0 * float(valid.mean()):.1f}% of extent",
    ]
    ax.text(
        0.01,
        0.99,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.4", "alpha": 0.92},
    )
    fig.text(
        0.01,
        0.01,
        "HHW CEND 11/11 IS A METHOD-DEVELOPMENT ANALOG ONLY AND DOES NOT ENTER PL854 "
        "SCIENTIFIC EVIDENCE. No morphology interpretation yet -- source overview only.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="0.2",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 22: primary method figure -----------------------------------------------------


def render_method_figure(
    *,
    native_elevation: np.ndarray,
    native_valid: np.ndarray,
    filtered_detrended: np.ndarray,
    pixel_size_m: float,
    tile_extent_m: tuple[float, float, float, float],
    crest_azimuth_deg: float,
    transect_endpoints: list[tuple[tuple[float, float], tuple[float, float]]],
    profile_distances_m: np.ndarray,
    profile_raw_detrended: np.ndarray,
    profile_filtered: np.ndarray,
    bedforms: list[dict[str, Any]],
    output_path: Path,
    title: str = "HHW CEND 11/11 -- Sand-Wave Morphometry Method",
    subtitle: str | None = None,
    dpi: int = 150,
) -> Path:
    """Section 22: panels A (native bathymetry), B (filtered/detrended
    sand-wave surface), C (crest orientation + 3 transects), D (one
    representative profile with detected crests/troughs/wavelength/
    height) -- for the single highest-ranked candidate tile. `subtitle`
    (MAR-017A Section 12) makes exploratory/below-canonical-floor results
    visually unmistakable -- pass e.g. "Exploratory 250 m support -- below
    canonical >=1000 m validation floor" whenever this figure is NOT
    canonical validation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 12.0))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    masked_native = _safe_masked_array(native_elevation, native_valid)
    im_a = ax_a.imshow(masked_native, cmap="viridis", extent=tile_extent_m, origin="upper")
    ax_a.set_title("A. Native bathymetry (elevation, m)")
    fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)

    masked_filtered = _safe_masked_array(filtered_detrended, native_valid)
    im_b = ax_b.imshow(masked_filtered, cmap="RdBu_r", extent=tile_extent_m, origin="upper")
    ax_b.set_title(f"B. {int(round(30))} m-filtered / detrended sand-wave surface (m)")
    fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)

    ax_c.imshow(masked_native, cmap="gray", extent=tile_extent_m, origin="upper", alpha=0.6)
    for p1, p2 in transect_endpoints:
        ax_c.plot([p1[0], p2[0]], [p1[1], p2[1]], color="tab:red", linewidth=2)
    center_x = (tile_extent_m[0] + tile_extent_m[1]) / 2.0
    center_y = (tile_extent_m[2] + tile_extent_m[3]) / 2.0
    crest_rad = np.radians(crest_azimuth_deg)
    arrow_len = (tile_extent_m[1] - tile_extent_m[0]) * 0.35
    dx, dy = arrow_len * np.sin(crest_rad), arrow_len * np.cos(crest_rad)
    ax_c.plot(
        [center_x - dx, center_x + dx],
        [center_y - dy, center_y + dy],
        color="tab:blue",
        linewidth=2.5,
    )
    ax_c.set_title(f"C. Dominant crest orientation ({crest_azimuth_deg:.0f} deg) + 3 transects")

    ax_d.plot(
        profile_distances_m,
        profile_raw_detrended,
        color="0.6",
        linewidth=1,
        label="Raw (detrended)",
    )
    ax_d.plot(
        profile_distances_m,
        profile_filtered,
        color="tab:blue",
        linewidth=2,
        label="Filtered (30 m)",
    )
    for bedform in bedforms:
        ax_d.plot(
            bedform["crest_position_m"],
            bedform["crest_elevation_m"],
            "^",
            color="tab:red",
            markersize=9,
        )
        ax_d.plot(
            bedform["left_trough_position_m"],
            bedform["left_trough_elevation_m"],
            "v",
            color="tab:green",
            markersize=9,
        )
        ax_d.plot(
            bedform["right_trough_position_m"],
            bedform["right_trough_elevation_m"],
            "v",
            color="tab:green",
            markersize=9,
        )
        mid_x = (bedform["left_trough_position_m"] + bedform["right_trough_position_m"]) / 2.0
        ax_d.annotate(
            f"λ={bedform['wavelength_m']:.0f} m, H={bedform['wave_height_m']:.2f} m",
            (mid_x, bedform["crest_elevation_m"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )
    ax_d.set_title("D. Representative cross-crest profile")
    ax_d.set_xlabel("Distance along transect (m)")
    ax_d.set_ylabel("Detrended elevation (m)")
    ax_d.legend(fontsize=8, loc="best")
    ax_d.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if subtitle:
        fig.text(
            0.5,
            0.955,
            subtitle,
            ha="center",
            va="top",
            fontsize=10.5,
            color="tab:red",
            fontweight="bold",
        )
    fig.text(
        0.01,
        0.01,
        "HHW CEND 11/11 IS A METHOD-DEVELOPMENT ANALOG ONLY AND DOES NOT ENTER PL854 "
        "SCIENTIFIC EVIDENCE.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.2",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 23: bedform distribution figure -------------------------------------------------


def render_bedform_distribution_figure(
    *,
    bedforms_df: pd.DataFrame,
    output_path: Path,
    title: str = "HHW CEND 11/11 -- Detected Bedform Distributions",
    subtitle: str | None = None,
    dpi: int = 150,
) -> Path:
    """Section 23: wavelength/wave-height/asymmetry distributions across
    only the successfully-detected complete bedforms -- no predictive
    distribution is ever fitted. `subtitle` (MAR-017A Section 12) must
    say EXPLORATORY whenever `bedforms_df` did not come from canonical
    (>=1000 m, >=90%-valid) tiles -- an exploratory n must never be
    presented as an accepted site bedform distribution."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    fields = (
        ("wavelength_m", "Wavelength (m)", axes[0]),
        ("wave_height_m", "Wave height (m)", axes[1]),
        ("asymmetry_index", "Asymmetry index", axes[2]),
    )
    for column, label, ax in fields:
        values = bedforms_df[column].dropna()
        ax.hist(values, bins=min(15, max(len(values), 1)), color="tab:blue", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"n={len(values)}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(
            0.5,
            0.90,
            subtitle,
            ha="center",
            va="top",
            fontsize=10,
            color="tab:red",
            fontweight="bold",
        )
    fig.text(
        0.01,
        0.01,
        "HHW CEND 11/11 IS A METHOD-DEVELOPMENT ANALOG ONLY AND DOES NOT ENTER PL854 "
        "SCIENTIFIC EVIDENCE. No predictive distribution has been fitted.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.2",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88 if subtitle else 0.94))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Section 24: spectral map -----------------------------------------------------------------


def render_dominant_bedform_scale_map(
    *,
    background_elevation: np.ndarray,
    background_valid: np.ndarray,
    background_extent_m: tuple[float, float, float, float],
    tile_spectral_df: pd.DataFrame,
    tile_size_m: float,
    output_path: Path,
    title: str = "HHW CEND 11/11 -- Dominant Bedform Scale",
    subtitle: str | None = None,
    canonical_unavailable_message: str | None = None,
    dpi: int = 150,
) -> Path:
    """Section 24: candidate tiles over the background bathymetry, coloured
    by `dominant_wavelength_m`, with an orientation glyph for
    `dominant_crest_azimuth_deg`. No risk/hazard/freespan language.
    `subtitle` labels the plotted set as exploratory when applicable
    (MAR-017A Section 12); `canonical_unavailable_message`, when given,
    is displayed prominently INSTEAD of ever silently plotting exploratory
    points as if they were canonical (Section 12: "if canonical tile set
    is empty: state this visibly rather than plotting exploratory points
    as canonical")."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 9.0))

    masked_bg = _safe_masked_array(background_elevation, background_valid)
    ax.imshow(masked_bg, cmap="gray", extent=background_extent_m, origin="upper", alpha=0.5)

    if not tile_spectral_df.empty:
        scatter = ax.scatter(
            tile_spectral_df["center_x_m"],
            tile_spectral_df["center_y_m"],
            c=tile_spectral_df["dominant_wavelength_m"],
            cmap="plasma",
            s=120,
            edgecolor="black",
            zorder=3,
        )
        fig.colorbar(scatter, ax=ax, label="Dominant wavelength (m)", fraction=0.04, pad=0.03)

        half = tile_size_m / 2.0
        selected_ids = tile_spectral_df.loc[
            tile_spectral_df["rank_selected_top3"], "tile_id"
        ].tolist()
        for _, row in tile_spectral_df.iterrows():
            azimuth_rad = np.radians(row["dominant_crest_azimuth_deg"])
            dx, dy = half * np.sin(azimuth_rad), half * np.cos(azimuth_rad)
            ax.plot(
                [row["center_x_m"] - dx, row["center_x_m"] + dx],
                [row["center_y_m"] - dy, row["center_y_m"] + dy],
                color="black",
                linewidth=1.5,
                zorder=4,
            )
            if row.get("rank_selected_top3"):
                ax.add_patch(
                    plt.Rectangle(
                        (row["center_x_m"] - half, row["center_y_m"] - half),
                        tile_size_m,
                        tile_size_m,
                        fill=False,
                        edgecolor="tab:red",
                        linewidth=2,
                        zorder=5,
                    )
                )
            # Only the top-3 selected tiles are labelled, and with a short rank number
            # rather than the full tile_id -- several real candidate tiles often cluster
            # within one tile-width of each other, so even a short label can overlap; the
            # full tile_id stays available in the parquet table for anyone who needs it.
            if row.get("rank_selected_top3"):
                rank = selected_ids.index(row["tile_id"]) + 1
                ax.annotate(
                    f"#{rank}",
                    (row["center_x_m"], row["center_y_m"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    fontweight="bold",
                    zorder=6,
                    annotation_clip=True,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                )

    if not tile_spectral_df.empty:
        # Zoom to the candidate-tile cluster (with generous padding) -- at the full
        # background grid's own scale (often many km) a few hundred-metre tiles would
        # otherwise render as invisible pinpricks.
        pad = tile_size_m * 4.0
        xmin = tile_spectral_df["center_x_m"].min() - pad
        xmax = tile_spectral_df["center_x_m"].max() + pad
        ymin = tile_spectral_df["center_y_m"].min() - pad
        ymax = tile_spectral_df["center_y_m"].max() + pad
        # Never zoom in tighter than the background extent itself allows.
        ax.set_xlim(max(xmin, background_extent_m[0]), min(xmax, background_extent_m[1]))
        ax.set_ylim(max(ymin, background_extent_m[2]), min(ymax, background_extent_m[3]))

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    # A figure-level suptitle (not ax.set_title) so its position is independent of how
    # tight_layout resizes the axes below -- ax.set_title is pinned to the axes' own edge,
    # which collided with the fig-level message text once the axes were shrunk to make room
    # for it (MAR-017A: this message must be prominent and legible, never overlapping the
    # plot title, and never leaving a large blank gap either).
    fig.suptitle(title, fontsize=13, fontweight="bold")
    message = canonical_unavailable_message or subtitle
    if message:
        fig.text(
            0.5,
            0.94,
            message,
            ha="center",
            va="top",
            fontsize=10.5 if canonical_unavailable_message else 10,
            color="tab:red",
            fontweight="bold",
            transform=fig.transFigure,
        )
    fig.text(
        0.01,
        0.01,
        "HHW CEND 11/11 IS A METHOD-DEVELOPMENT ANALOG ONLY AND DOES NOT ENTER PL854 "
        "SCIENTIFIC EVIDENCE. Colour = descriptive dominant wavelength only, never a risk/"
        "hazard score.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="0.2",
        transform=fig.transFigure,
    )
    top = 0.88 if message else 0.94
    fig.tight_layout(rect=(0, 0.04, 1, top))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- MAR-017B Section 21: canonical support QA map (always rendered, regardless of ---------
# --- whether canonical validation ultimately proceeds) --------------------------------------


def render_canonical_support_audit_map(
    *,
    background_elevation: np.ndarray,
    background_valid: np.ndarray,
    background_extent_m: tuple[float, float, float, float],
    background_candidate_id: str,
    qualifying_2000m_tiles: list[tuple[float, float, float]],
    qualifying_1000m_tiles: list[tuple[float, float, float]],
    no_qualifying_tile_message: str | None,
    output_path: Path,
    title: str = "Canonical Support Audit",
    dpi: int = 150,
) -> Path:
    """MAR-017B Section 21: the raw source footprint/coverage (nodata
    gaps immediately visible) for `background_candidate_id`, with any
    real qualifying 2000 m (red) / 1000 m (orange) tile footprints
    overlaid as `(center_x_m, center_y_m, tile_size_m)` triples. When
    NEITHER list has any entries, `no_qualifying_tile_message` must be
    set and is displayed prominently -- the figure itself must make a
    negative support result unmistakable, never merely blank."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 9.0))

    masked_bg = _safe_masked_array(background_elevation, background_valid)
    ax.imshow(masked_bg, cmap="gray", extent=background_extent_m, origin="upper", alpha=0.7)

    for center_x, center_y, size_m in qualifying_2000m_tiles:
        half = size_m / 2.0
        ax.add_patch(
            plt.Rectangle(
                (center_x - half, center_y - half),
                size_m,
                size_m,
                fill=False,
                edgecolor="tab:red",
                linewidth=2,
                zorder=5,
                label="Qualifying 2000 m tile",
            )
        )
    for center_x, center_y, size_m in qualifying_1000m_tiles:
        half = size_m / 2.0
        ax.add_patch(
            plt.Rectangle(
                (center_x - half, center_y - half),
                size_m,
                size_m,
                fill=False,
                edgecolor="tab:orange",
                linewidth=2,
                zorder=5,
                label="Qualifying 1000 m tile",
            )
        )
    if qualifying_2000m_tiles or qualifying_1000m_tiles:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=False))
        ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=8)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.93,
        f"Background candidate shown: {background_candidate_id}",
        ha="center",
        va="top",
        fontsize=9.5,
        color="0.3",
        transform=fig.transFigure,
    )
    if no_qualifying_tile_message:
        fig.text(
            0.5,
            0.90,
            no_qualifying_tile_message,
            ha="center",
            va="top",
            fontsize=10.5,
            color="tab:red",
            fontweight="bold",
            transform=fig.transFigure,
        )
    fig.text(
        0.01,
        0.01,
        "IDRBNR CEND 11/11 IS A METHOD-DEVELOPMENT ANALOG ONLY AND DOES NOT ENTER PL854 "
        "SCIENTIFIC EVIDENCE.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="0.2",
        transform=fig.transFigure,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

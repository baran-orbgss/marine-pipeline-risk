"""Noncohesive relative excess Shields transport-potential intensity (MAR-030).

Scope -- read before touching this module
--------------------------------------------
A dimensionless transport-POTENTIAL intensity layer built strictly on top
of the accepted MAR-013 mobility product. It consumes MAR-013's canonical
3-hourly mobility table and never recomputes MAR-012 hydrodynamics or
MAR-013 grain-related skin stress / Soulsby-Whitehouse thresholds. The
dependency direction is one-way: `transport_intensity` imports
`noncohesive_mobility`, never the reverse.

Scientific role
---------------
`NONCOHESIVE_RELATIVE_EXCESS_SHIELDS_TRANSPORT_POTENTIAL_INTENSITY`: the
dimensionless magnitude by which the accepted MAR-013 maximum combined
grain-related skin stress exceeds the corresponding incipient-motion
critical stress for a TESTED noncohesive D50 scenario. It is NOT a
sediment transport rate, bedload/suspended/total-load flux, net transport,
transport direction, erosion/deposition rate, morphological change rate,
scour rate, burial-loss rate, probability, or risk score.

Fixed definition (Section 5)
----------------------------
For every MAR-013 row with `M = mobility_ratio = tau_max_grain_skin_pa /
tau_critical_pa`:

    relative_shields_stage             = M - 1          (signed, diagnostic)
    relative_excess_shields_intensity  = max(M - 1, 0)

which is algebraically `max((tau_max - tau_cr) / tau_cr, 0)` for
`tau_cr > 0`. An undefined mobility ratio (e.g. `tau_cr <= 0`) stays null
-- never replaced by zero.

Relation to Van Rijn (Section 6)
--------------------------------
The intensity is algebraically equivalent to the relative-excess structure
of the Van Rijn (1984) transport-stage parameter `T = (u*'^2 - u*cr^2) /
u*cr^2 = tau/tau_cr - 1` when formed from consistent grain-related and
critical stresses. It is used here ONLY as a dimensionless forcing-
exceedance diagnostic; the Van Rijn bed-load transport-rate formula is NOT
applied, and no other transport-rate formulation (Meyer-Peter & Muller,
Van Rijn 2007, Soulsby-Van Rijn, Ribberink, Camenen-Larson) is introduced.

Combined wave-current limitation (Section 7)
--------------------------------------------
MAR-013's `tau_max_grain_skin_pa` is the Soulsby-algebraic MAXIMUM combined
grain-related skin stress during the representative wave cycle, so every
timestamp here is the representative-wave-cycle PEAK relative excess of
the incipient-motion threshold -- never wave-cycle-mean transport,
phase-resolved transport, net wave-cycle flux, onshore/offshore transport,
or along-route direction.

D50 semantics (Section 9)
-------------------------
`TESTED_D50_SCENARIOS_MM` / `GRAIN_SIZE_SCENARIO_SEMANTICS` are reused
unchanged from MAR-013. Each tested scenario remains independent: no
preferred/default/local D50, no averaging or weighting across scenarios,
no BGS Folk -> D50 conversion, no PSA point interpolation. No intensity
classes (LOW/HIGH/...) exist -- intensity is a continuous dimensionless
quantity; zero means only "no positive excess above the MAR-013 threshold
at that timestamp / tested scenario", not "no sediment movement in nature".
"""

import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

from marine_engine.sediment import noncohesive_mobility as ncm
from marine_engine.sediment.noncohesive_mobility import (
    GRAIN_SIZE_SCENARIO_SEMANTICS,
    TESTED_D50_SCENARIOS_MM,
)

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# --- Roles and fixed semantic vocabulary (Sections 4, 7, 12, 16) --------------------

SCIENTIFIC_ROLE = "NONCOHESIVE_RELATIVE_EXCESS_SHIELDS_TRANSPORT_POTENTIAL_INTENSITY"
SOURCE_SCIENTIFIC_ROLE = ncm.SCIENTIFIC_ROLE  # NONCOHESIVE_SEDIMENT_MOBILITY_CAPACITY

TRANSPORT_INTENSITY_SUPPORT_SEMANTICS = (
    "REPRESENTATIVE_WAVE_CYCLE_PEAK_RELATIVE_EXCESS_OF_INCIPIENT_MOTION_THRESHOLD"
)
FRACTION_DENOMINATOR = "VALID_CONTEMPORANEOUS_MATCHED_TIMESTAMPS"
FRACTION_DENOMINATOR_DESCRIPTION = "fraction of valid contemporaneous matched timestamps"
UNITS = "dimensionless"

INTENSITY_DEFINITION = (
    "relative_excess_shields_intensity = max(mobility_ratio - 1, 0) "
    "= max((tau_max_grain_skin_pa - tau_critical_pa) / tau_critical_pa, 0) for "
    "tau_critical_pa > 0; null where the MAR-013 mobility ratio is undefined"
)
STAGE_DEFINITION = "relative_shields_stage = mobility_ratio - 1 (signed diagnostic)"

VAN_RIJN_RELATION = (
    "The MAR-030 intensity is algebraically equivalent to the relative-excess "
    "structure of the Van Rijn transport-stage parameter when formed from "
    "consistent grain-related and critical stresses. It is used here only as a "
    "dimensionless forcing-exceedance diagnostic. The Van Rijn bed-load "
    "transport-rate formula is NOT applied."
)
COMBINED_WAVE_CURRENT_LIMITATION = (
    "tau_max_grain_skin_pa is MAR-013's Soulsby-algebraic maximum combined "
    "grain-related skin stress during the representative wave cycle, so every "
    "timestamp-level intensity is the representative-wave-cycle PEAK relative "
    "excess of the incipient-motion threshold. It does not represent wave-cycle-"
    "mean transport, phase-resolved transport, net wave-cycle sediment flux, "
    "onshore/offshore transport, or along-route transport direction."
)
ZERO_INTENSITY_SEMANTICS = (
    "Zero means only: no positive excess above the MAR-013 incipient-motion "
    "threshold at that timestamp / tested scenario. It does not mean that no "
    "sediment movement exists in nature."
)

REFERENCES: tuple[dict[str, str], ...] = (
    {
        "citation": (
            "van Rijn, L.C. (1984). Sediment Transport, Part I: Bed Load Transport. "
            "Journal of Hydraulic Engineering, 110(10), 1431-1456."
        ),
        "doi": "10.1061/(ASCE)0733-9429(1984)110:10(1431)",
        "use_in_mar_030": (
            "Scientific lineage of the relative-excess / transport-stage structure "
            "T = (u*'^2 - u*cr^2)/u*cr^2 = tau/tau_cr - 1 only. MAR-030 does NOT "
            "implement the Van Rijn 1984 bed-load transport-rate formula."
        ),
    },
    {
        "citation": (
            "van Rijn, L.C. (2007). Unified View of Sediment Transport by Currents and "
            "Waves. I: Initiation of Motion, Bed Roughness, and Bed-Load Transport. "
            "Journal of Hydraulic Engineering, 133(6), 649-667."
        ),
        "doi": "10.1061/(ASCE)0733-9429(2007)133:6(649)",
        "use_in_mar_030": (
            "Context only: full coastal sediment-transport prediction requires more "
            "complete hydrodynamic and sediment characteristics than MAR-030 possesses."
        ),
    },
    {
        "citation": (
            "Ribberink, J.S. (1998). Bed-load transport for steady flows and unsteady "
            "oscillatory flows. Coastal Engineering, 34, 59-82."
        ),
        "doi": "10.1016/S0378-3839(98)00013-1",
        "use_in_mar_030": (
            "Documents why oscillatory / wave-current transport-rate calculations "
            "require more than a representative-cycle maximum stress."
        ),
    },
)

# Strict floating-point tolerance for the independent algebraic QA (Section 14).
MOBILITY_RATIO_CONSISTENCY_RTOL = 1e-9
MOBILITY_RATIO_CONSISTENCY_ATOL = 1e-12


class TransportIntensitySchemaError(Exception):
    """The MAR-013 source table lacks a required scientific column."""


class TransportIntensitySourceRoleError(Exception):
    """The source table does not carry the accepted MAR-013 scientific role."""


class MobilityRatioConsistencyError(Exception):
    """A source row's mobility ratio / intensity disagrees with its own stresses."""


class MobilityStatsCrossCheckError(Exception):
    """MAR-030 counts disagree with the accepted MAR-013 statistics where definitions match."""


REQUIRED_MOBILITY_COLUMNS: tuple[str, ...] = (
    "hydro_pair_id",
    "current_node_id",
    "wave_node_id",
    "time_utc",
    "tested_d50_mm",
    "tested_d50_m",
    "tau_max_grain_skin_pa",
    "tau_critical_pa",
    "critical_shields_parameter",
    "mobility_ratio",
    "incipient_motion_status",
    "scientific_role",
)


# --- Core formula (Sections 5, 13) -----------------------------------------------------


def compute_relative_shields_stage(mobility_ratio: np.ndarray) -> np.ndarray:
    """`relative_shields_stage = M - 1`; null (NaN) wherever `M` is not finite."""

    m = np.asarray(mobility_ratio, dtype=float)
    return np.where(np.isfinite(m), m - 1.0, np.nan)


def compute_relative_excess_shields_intensity(mobility_ratio: np.ndarray) -> np.ndarray:
    """`relative_excess_shields_intensity = max(M - 1, 0)`; null wherever `M` is undefined.

    Never negative, never clipped above, and an undefined ratio is NEVER
    replaced by zero (Section 13).
    """

    stage = compute_relative_shields_stage(mobility_ratio)
    return np.where(np.isfinite(stage), np.maximum(stage, 0.0), np.nan)


# --- Source validation and independent algebraic QA (Sections 11, 14) -----------------


def validate_mobility_source(mobility_df: pd.DataFrame) -> None:
    """Fail cleanly on a missing required scientific column or a foreign scientific role."""

    missing = [c for c in REQUIRED_MOBILITY_COLUMNS if c not in mobility_df.columns]
    if missing:
        raise TransportIntensitySchemaError(
            f"MAR-013 source table is missing required column(s): {missing}"
        )
    if mobility_df.empty:
        return
    roles = set(mobility_df["scientific_role"].dropna().unique().tolist())
    if roles != {SOURCE_SCIENTIFIC_ROLE}:
        raise TransportIntensitySourceRoleError(
            f"expected every source row to carry scientific_role={SOURCE_SCIENTIFIC_ROLE!r}, "
            f"found {sorted(roles)}"
        )


def verify_mobility_ratio_consistency(
    mobility_df: pd.DataFrame,
    *,
    rtol: float = MOBILITY_RATIO_CONSISTENCY_RTOL,
    atol: float = MOBILITY_RATIO_CONSISTENCY_ATOL,
) -> None:
    """Independent algebraic QA (Section 14) -- not a second scientific model.

    For every row checks that (a) `mobility_ratio ~= tau_max / tau_cr` where
    `tau_cr > 0` and both stresses are finite, (b) the ratio is undefined
    (NaN) wherever `tau_cr <= 0` / non-finite or `tau_max` is non-finite,
    and (c) `max(M - 1, 0) ~= max((tau_max - tau_cr)/tau_cr, 0)`. Any
    material inconsistency raises `MobilityRatioConsistencyError`.
    """

    if mobility_df.empty:
        return
    tau_max = mobility_df["tau_max_grain_skin_pa"].to_numpy(dtype=float)
    tau_cr = mobility_df["tau_critical_pa"].to_numpy(dtype=float)
    ratio = mobility_df["mobility_ratio"].to_numpy(dtype=float)

    definable = np.isfinite(tau_max) & np.isfinite(tau_cr) & (tau_cr > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected_ratio = np.where(definable, tau_max / tau_cr, np.nan)
        expected_intensity = np.where(
            definable, np.maximum((tau_max - tau_cr) / tau_cr, 0.0), np.nan
        )

    ratio_defined = np.isfinite(ratio)
    definedness_mismatch = ratio_defined != definable
    value_mismatch = definable & ~np.isclose(ratio, expected_ratio, rtol=rtol, atol=atol)
    intensity = compute_relative_excess_shields_intensity(ratio)
    intensity_mismatch = definable & ~np.isclose(
        intensity, expected_intensity, rtol=rtol, atol=atol
    )

    bad = definedness_mismatch | value_mismatch | intensity_mismatch
    if bad.any():
        offending = np.flatnonzero(bad)
        raise MobilityRatioConsistencyError(
            f"{len(offending)} source row(s) have a mobility_ratio inconsistent with "
            f"tau_max_grain_skin_pa / tau_critical_pa (rtol={rtol}, atol={atol}); first "
            f"offending row positions: {offending[:10].tolist()}"
        )


# --- Timestamp-level output (Section 12) --------------------------------------------

TRANSPORT_INTENSITY_3HOURLY_COLUMNS: tuple[str, ...] = (
    # Identity (preserved from MAR-013)
    "hydro_pair_id",
    "current_node_id",
    "wave_node_id",
    "time_utc",
    # Grain scenario (preserved from MAR-013)
    "tested_d50_mm",
    "tested_d50_m",
    # Accepted MAR-013 stress / threshold facts (preserved, never recomputed)
    "tau_max_grain_skin_pa",
    "tau_critical_pa",
    "critical_shields_parameter",
    "mobility_ratio",
    "incipient_motion_status",
    # MAR-030 derived quantities
    "relative_shields_stage",
    "relative_excess_shields_intensity",
    # Semantics / provenance
    "transport_intensity_support_semantics",
    "scientific_role",
    "source_scientific_role",
)


def build_transport_intensity_3hourly(mobility_df: pd.DataFrame) -> pd.DataFrame:
    """One output row per MAR-013 mobility row, in source order (Section 12).

    Validates the source schema/role, runs the independent algebraic QA,
    then derives the signed stage and the canonical intensity from the
    ACCEPTED `mobility_ratio`. Every preserved column is copied verbatim --
    the MAR-013 `>= 1` threshold status is never reclassified here.
    """

    validate_mobility_source(mobility_df)
    if mobility_df.empty:
        return pd.DataFrame(columns=list(TRANSPORT_INTENSITY_3HOURLY_COLUMNS))
    verify_mobility_ratio_consistency(mobility_df)

    ratio = mobility_df["mobility_ratio"].to_numpy(dtype=float)
    result = pd.DataFrame(
        {
            "hydro_pair_id": mobility_df["hydro_pair_id"].to_numpy(),
            "current_node_id": mobility_df["current_node_id"].to_numpy(),
            "wave_node_id": mobility_df["wave_node_id"].to_numpy(),
            "time_utc": mobility_df["time_utc"].to_numpy(),
            "tested_d50_mm": mobility_df["tested_d50_mm"].to_numpy(dtype=float),
            "tested_d50_m": mobility_df["tested_d50_m"].to_numpy(dtype=float),
            "tau_max_grain_skin_pa": mobility_df["tau_max_grain_skin_pa"].to_numpy(dtype=float),
            "tau_critical_pa": mobility_df["tau_critical_pa"].to_numpy(dtype=float),
            "critical_shields_parameter": mobility_df["critical_shields_parameter"].to_numpy(
                dtype=float
            ),
            "mobility_ratio": ratio,
            "incipient_motion_status": mobility_df["incipient_motion_status"].to_numpy(),
            "relative_shields_stage": compute_relative_shields_stage(ratio),
            "relative_excess_shields_intensity": compute_relative_excess_shields_intensity(ratio),
            "transport_intensity_support_semantics": TRANSPORT_INTENSITY_SUPPORT_SEMANTICS,
            "scientific_role": SCIENTIFIC_ROLE,
            "source_scientific_role": SOURCE_SCIENTIFIC_ROLE,
        }
    )
    return result[list(TRANSPORT_INTENSITY_3HOURLY_COLUMNS)]


# --- Per-pair / per-D50 statistics (Sections 15-16) ------------------------------------

TRANSPORT_INTENSITY_STATS_COLUMNS: tuple[str, ...] = (
    "hydro_pair_id",
    "tested_d50_mm",
    "overlap_start_time_utc",
    "overlap_end_time_utc",
    "tau_critical_pa",
    "valid_intensity_timestamp_count",
    "at_or_above_incipient_motion_count",
    "at_or_above_incipient_motion_fraction",
    "strict_positive_excess_count",
    "strict_positive_excess_fraction",
    "relative_excess_intensity_mean",
    "relative_excess_intensity_p50",
    "relative_excess_intensity_p90",
    "relative_excess_intensity_p95",
    "relative_excess_intensity_p99",
    "relative_excess_intensity_max",
    "relative_excess_intensity_mean_when_positive",
    "fraction_denominator",
    "grain_size_scenario_semantics",
    "scientific_role",
)


def _fraction(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def compute_transport_intensity_stats(intensity_df: pd.DataFrame) -> pd.DataFrame:
    """Per `hydro_pair_id x tested_d50_mm` descriptive statistics (Section 15).

    Every statistic is computed from the TIMESTAMP-LEVEL derived intensity
    series itself -- never by subtracting 1 from MAR-013 percentile columns
    (percentile of max(M-1,0) != max(percentile(M)-1, 0) in general once
    clipping is involved, and the ticket forbids the shortcut regardless).
    Null intensities are dropped, never counted as zero observations; every
    fraction's denominator is the count of VALID contemporaneous matched
    timestamps in that pair x scenario group.
    """

    if intensity_df.empty:
        return pd.DataFrame(columns=list(TRANSPORT_INTENSITY_STATS_COLUMNS))

    records = []
    for (pair_id, d50_mm), group in intensity_df.groupby(["hydro_pair_id", "tested_d50_mm"]):
        valid_mask = group["relative_excess_shields_intensity"].notna()
        valid = group.loc[valid_mask]
        intensity = valid["relative_excess_shields_intensity"].to_numpy(dtype=float)
        ratio = valid["mobility_ratio"].to_numpy(dtype=float)
        valid_count = int(len(intensity))

        # At-or-above uses the SAME >= 1 convention MAR-013 accepted; the
        # preserved status column must agree with it exactly.
        at_or_above_count = int((ratio >= 1.0).sum())
        status_count = int((valid["incipient_motion_status"] == ncm.ABOVE_OR_AT_THRESHOLD).sum())
        if at_or_above_count != status_count:
            raise MobilityStatsCrossCheckError(
                f"{pair_id} / D50={d50_mm} mm: {at_or_above_count} valid rows have "
                f"mobility_ratio >= 1 but {status_count} carry the MAR-013 "
                f"'{ncm.ABOVE_OR_AT_THRESHOLD}' status"
            )
        strict_positive_count = int((intensity > 0.0).sum())
        positive = intensity[intensity > 0.0]

        records.append(
            {
                "hydro_pair_id": pair_id,
                "tested_d50_mm": d50_mm,
                "overlap_start_time_utc": group["time_utc"].min(),
                "overlap_end_time_utc": group["time_utc"].max(),
                "tau_critical_pa": float(group["tau_critical_pa"].iloc[0]),
                "valid_intensity_timestamp_count": valid_count,
                "at_or_above_incipient_motion_count": at_or_above_count,
                "at_or_above_incipient_motion_fraction": _fraction(at_or_above_count, valid_count),
                "strict_positive_excess_count": strict_positive_count,
                "strict_positive_excess_fraction": _fraction(strict_positive_count, valid_count),
                "relative_excess_intensity_mean": float(intensity.mean()) if valid_count else None,
                "relative_excess_intensity_p50": (
                    float(np.percentile(intensity, 50)) if valid_count else None
                ),
                "relative_excess_intensity_p90": (
                    float(np.percentile(intensity, 90)) if valid_count else None
                ),
                "relative_excess_intensity_p95": (
                    float(np.percentile(intensity, 95)) if valid_count else None
                ),
                "relative_excess_intensity_p99": (
                    float(np.percentile(intensity, 99)) if valid_count else None
                ),
                "relative_excess_intensity_max": float(intensity.max()) if valid_count else None,
                "relative_excess_intensity_mean_when_positive": (
                    float(positive.mean()) if len(positive) else None
                ),
                "fraction_denominator": FRACTION_DENOMINATOR,
                "grain_size_scenario_semantics": GRAIN_SIZE_SCENARIO_SEMANTICS,
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )
    return pd.DataFrame(records, columns=list(TRANSPORT_INTENSITY_STATS_COLUMNS))


def cross_check_against_mobility_stats(
    intensity_stats_df: pd.DataFrame, mobility_stats_df: pd.DataFrame
) -> int:
    """Cross-check MAR-030 counts against the accepted MAR-013 statistics (Section 15).

    Where definitions are equivalent -- MAR-013 `valid_count` (finite
    mobility ratios) vs `valid_intensity_timestamp_count`, and MAR-013
    `threshold_exceedance_count` (`mobility_ratio >= 1`) vs
    `at_or_above_incipient_motion_count` -- the two products must agree
    exactly for every `hydro_pair_id x tested_d50_mm` present in both.
    Returns the number of groups compared; raises on any disagreement.
    """

    if intensity_stats_df.empty or mobility_stats_df.empty:
        return 0
    keys = ["hydro_pair_id", "tested_d50_mm"]
    merged = intensity_stats_df.merge(
        mobility_stats_df[[*keys, "valid_count", "threshold_exceedance_count"]],
        on=keys,
        how="inner",
    )
    valid_mismatch = merged["valid_intensity_timestamp_count"] != merged["valid_count"]
    exceed_mismatch = (
        merged["at_or_above_incipient_motion_count"] != merged["threshold_exceedance_count"]
    )
    bad = merged.loc[valid_mismatch | exceed_mismatch, keys]
    if not bad.empty:
        raise MobilityStatsCrossCheckError(
            "MAR-030 counts disagree with accepted MAR-013 statistics for "
            f"{len(bad)} hydro_pair x D50 group(s): {bad.head(5).to_dict('records')}"
        )
    return int(len(merged))


# --- Scenario GIS output (Section 19) --------------------------------------------------

TRANSPORT_INTENSITY_SEGMENTS_COLUMNS: tuple[str, ...] = (
    "pipeline_id",
    "segment_id",
    "start_chainage_m",
    "end_chainage_m",
    "kp_start",
    "kp_end",
    "hydro_pair_id",
    "tested_d50_mm",
    "valid_intensity_timestamp_count",
    "at_or_above_incipient_motion_count",
    "at_or_above_incipient_motion_fraction",
    "strict_positive_excess_count",
    "strict_positive_excess_fraction",
    "relative_excess_intensity_p50",
    "relative_excess_intensity_p95",
    "relative_excess_intensity_max",
    "relative_excess_intensity_mean_when_positive",
    "fraction_denominator",
    "grain_size_scenario_semantics",
    "scientific_role",
)

_SEGMENT_STAT_FIELDS = (
    "valid_intensity_timestamp_count",
    "at_or_above_incipient_motion_count",
    "at_or_above_incipient_motion_fraction",
    "strict_positive_excess_count",
    "strict_positive_excess_fraction",
    "relative_excess_intensity_p50",
    "relative_excess_intensity_p95",
    "relative_excess_intensity_max",
    "relative_excess_intensity_mean_when_positive",
)

_MOBILITY_SEGMENT_REQUIRED = (
    "pipeline_id",
    "segment_id",
    "start_chainage_m",
    "end_chainage_m",
    "kp_start",
    "kp_end",
    "hydro_pair_id",
    "geometry",
)


def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float | np.floating) and np.isnan(value):
        return None
    return value


def build_transport_intensity_segments(
    mobility_segments_gdf: gpd.GeoDataFrame, stats_df: pd.DataFrame
) -> gpd.GeoDataFrame:
    """LONG scenario layer: one feature per accepted MAR-013 route segment x tested D50.

    Geometry, chainage bounds, KP labels, and hydro-pair assignment are
    INHERITED verbatim from the accepted MAR-013 capacity segments -- route
    segmentation is never rebuilt here. The same physical segment therefore
    legitimately appears once per tested scenario (nine times). No field
    implies an actual local D50, and no scenario is singled out.
    """

    missing = [c for c in _MOBILITY_SEGMENT_REQUIRED if c not in mobility_segments_gdf.columns]
    if missing:
        raise TransportIntensitySchemaError(
            f"MAR-013 capacity segments are missing required column(s): {missing}"
        )
    if mobility_segments_gdf.empty:
        return gpd.GeoDataFrame(
            columns=list(TRANSPORT_INTENSITY_SEGMENTS_COLUMNS),
            geometry=[],
            crs=mobility_segments_gdf.crs,
        )

    stats_by_key = (
        stats_df.set_index(["hydro_pair_id", "tested_d50_mm"]) if not stats_df.empty else None
    )

    records = []
    geometries = []
    for _, segment in mobility_segments_gdf.iterrows():
        pair_id = segment["hydro_pair_id"]
        pair_id = pair_id if pd.notna(pair_id) else None
        for d50_mm in sorted(TESTED_D50_SCENARIOS_MM):
            record: dict[str, Any] = {
                "pipeline_id": segment["pipeline_id"],
                "segment_id": int(segment["segment_id"]),
                "start_chainage_m": float(segment["start_chainage_m"]),
                "end_chainage_m": float(segment["end_chainage_m"]),
                "kp_start": segment["kp_start"],
                "kp_end": segment["kp_end"],
                "hydro_pair_id": pair_id,
                "tested_d50_mm": d50_mm,
                "fraction_denominator": FRACTION_DENOMINATOR,
                "grain_size_scenario_semantics": GRAIN_SIZE_SCENARIO_SEMANTICS,
                "scientific_role": SCIENTIFIC_ROLE,
            }
            key = (pair_id, d50_mm)
            if stats_by_key is not None and pair_id is not None and key in stats_by_key.index:
                row = stats_by_key.loc[key]
                for field in _SEGMENT_STAT_FIELDS:
                    record[field] = _none_if_nan(row[field])
            else:
                for field in _SEGMENT_STAT_FIELDS:
                    record[field] = None
            records.append(record)
            geometries.append(segment.geometry)

    return gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs=mobility_segments_gdf.crs,
        columns=list(TRANSPORT_INTENSITY_SEGMENTS_COLUMNS),
    )


def write_transport_intensity_segments_gpkg(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    layer: str = "noncohesive_transport_intensity_segments",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer)
    return output_path


# --- Scenario matrix visualization (Section 20) ----------------------------------------

MATRIX_VISUAL_QUANTITY = "relative_excess_intensity_p95"
MATRIX_MISSING_COLOUR = (0.78, 0.78, 0.78, 1.0)


def render_transport_intensity_scenario_matrix(
    segments_gdf: gpd.GeoDataFrame,
    *,
    output_path: Path,
    title_prefix: str = "",
) -> Path:
    """Route section x tested-D50 matrix of `relative_excess_intensity_p95`.

    Every one of the nine tested scenarios is a row of its own -- no single
    D50 is ever selected as "the" route condition. A single-hue sequential
    colour scale (light -> dark) encodes magnitude from exactly zero; cells
    without a valid statistic are neutral grey and labelled `n/a`.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    d50_values = sorted(TESTED_D50_SCENARIOS_MM)

    if segments_gdf.empty:
        fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
        ax.text(0.5, 0.5, "No route segments available", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    segments = (
        segments_gdf.drop(columns="geometry", errors="ignore")
        .drop_duplicates(subset=["segment_id"])
        .sort_values("start_chainage_m")
        .reset_index(drop=True)
    )
    by_key = segments_gdf.drop(columns="geometry", errors="ignore").set_index(
        ["segment_id", "tested_d50_mm"]
    )

    values = np.full((len(d50_values), len(segments)), np.nan)
    for j, seg in segments.iterrows():
        for i, d50_mm in enumerate(d50_values):
            key = (seg["segment_id"], d50_mm)
            if key in by_key.index:
                v = by_key.loc[key, MATRIX_VISUAL_QUANTITY]
                v = v.iloc[0] if isinstance(v, pd.Series) else v
                if v is not None and pd.notna(v):
                    values[i, j] = float(v)

    finite = values[np.isfinite(values)]
    vmax = float(finite.max()) if len(finite) and finite.max() > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("Blues")

    total_km = float(segments["end_chainage_m"].max()) / 1000.0
    narrow_km = total_km / 40.0

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
    for j, seg in segments.iterrows():
        x0 = float(seg["start_chainage_m"]) / 1000.0
        x1 = float(seg["end_chainage_m"]) / 1000.0
        for i in range(len(d50_values)):
            v = values[i, j]
            colour = cmap(norm(v)) if np.isfinite(v) else MATRIX_MISSING_COLOUR
            ax.add_patch(
                Rectangle((x0, i), x1 - x0, 1.0, facecolor=colour, edgecolor="white", linewidth=0.8)
            )
            label = f"{v:.2f}" if np.isfinite(v) else "n/a"
            luminance = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
            ax.text(
                (x0 + x1) / 2.0,
                i + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                # Narrow sections get a vertical label so neighbouring values never collide.
                rotation=90 if (x1 - x0) < narrow_km else 0,
                color="white" if luminance < 0.5 else "#222222",
            )

    ax.set_xlim(0.0, total_km)
    ax.set_ylim(0, len(d50_values))
    ax.set_yticks([i + 0.5 for i in range(len(d50_values))])
    ax.set_yticklabels([f"{d:g} mm" for d in d50_values], fontsize=8)
    ax.set_ylabel("Tested noncohesive D50 scenario (NOT observed local D50)", fontsize=9)
    ax.set_xlabel("Route chainage (km) -- accepted MAR-013 hydro-pair sections", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("relative_excess_intensity_p95 = p95 of max(M - 1, 0)  [dimensionless]", size=8)
    cbar.ax.tick_params(labelsize=7)

    prefix = f"{title_prefix} " if title_prefix else ""
    ax.set_title(
        f"{prefix}Noncohesive relative excess Shields transport-potential intensity "
        "(MAR-030)\nroute section x tested D50 scenario matrix -- each row is an "
        "independent tested scenario",
        fontsize=10,
    )
    fig.text(
        0.01,
        0.01,
        "Rows are TESTED D50 scenarios, not observed local sediment assignments. Intensity = "
        "max(mobility_ratio - 1, 0) on MAR-013's representative-wave-cycle PEAK combined "
        "grain-skin stress. Dimensionless forcing-exceedance diagnostic only: no transport "
        "rate, no direction, no erosion/deposition, no scour, no risk score, no preferred "
        "D50. Grey = no valid statistic.",
        fontsize=6.5,
        ha="left",
        va="bottom",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# --- Metadata (Sections 21-22) ----------------------------------------------------------


def build_transport_intensity_metadata(
    *,
    outputs: dict[str, str],
    row_count: int,
    hydro_pair_count: int,
    cross_checked_group_count: int,
) -> dict[str, Any]:
    """Deterministic metadata dict; only `outputs` and the counts vary between studies."""

    return {
        "scientific_role": SCIENTIFIC_ROLE,
        "source_scientific_role": SOURCE_SCIENTIFIC_ROLE,
        "source_product": "MAR-013 noncohesive_mobility_3hourly (accepted, consumed unchanged)",
        "definition": INTENSITY_DEFINITION,
        "signed_stage_definition": STAGE_DEFINITION,
        "units": UNITS,
        "stress_basis": "MAR-013 tau_max_grain_skin_pa",
        "threshold_basis": (
            "MAR-013 tau_critical_pa (Soulsby-Whitehouse critical Shields threshold)"
        ),
        "incipient_motion_convention": "MAR-013 mobility_ratio >= 1 (unchanged)",
        "transport_intensity_support_semantics": TRANSPORT_INTENSITY_SUPPORT_SEMANTICS,
        "grain_size_scenario_semantics": GRAIN_SIZE_SCENARIO_SEMANTICS,
        "tested_d50_scenarios_mm": list(TESTED_D50_SCENARIOS_MM),
        "tested_d50_scenario_count": len(TESTED_D50_SCENARIOS_MM),
        "van_rijn_relation": VAN_RIJN_RELATION,
        "combined_wave_current_limitation": COMBINED_WAVE_CURRENT_LIMITATION,
        "zero_intensity_semantics": ZERO_INTENSITY_SEMANTICS,
        "invalid_data_policy": (
            "An undefined MAR-013 mobility ratio (e.g. tau_critical_pa <= 0) yields null "
            "stage and null intensity; nulls are never replaced by zero and never counted "
            "as observations."
        ),
        "fraction_denominator": FRACTION_DENOMINATOR,
        "fraction_denominator_description": FRACTION_DENOMINATOR_DESCRIPTION,
        "fraction_is_not": [
            "annual probability",
            "event probability",
            "return-period probability",
            "fraction of project life",
            "fraction of route",
        ],
        "statistics_basis": (
            "All summary statistics are computed from the timestamp-level derived intensity "
            "series, never by subtracting 1 from MAR-013 percentile columns."
        ),
        "mar_013_cross_check": {
            "valid_count_vs_valid_intensity_timestamp_count": "exact agreement required",
            "threshold_exceedance_count_vs_at_or_above_incipient_motion_count": (
                "exact agreement required"
            ),
            "groups_cross_checked": cross_checked_group_count,
        },
        "algebraic_qa": (
            "Every source row independently verified: mobility_ratio ~= tau_max_grain_skin_pa / "
            "tau_critical_pa and max(M-1,0) ~= max((tau_max-tau_cr)/tau_cr,0) within "
            f"rtol={MOBILITY_RATIO_CONSISTENCY_RTOL}, atol={MOBILITY_RATIO_CONSISTENCY_ATOL}; "
            "a material inconsistency raises MobilityRatioConsistencyError."
        ),
        "matrix_visual_quantity": MATRIX_VISUAL_QUANTITY,
        "matrix_visual_semantics": (
            "Route section x tested-D50 matrix; every tested scenario is its own row. No "
            "single D50 is selected as the route condition. Tested D50 scenarios, not "
            "observed local sediment assignments."
        ),
        "intensity_classes_defined": False,
        "aggregation_across_d50_scenarios": False,
        "preferred_d50_assigned": False,
        "continuous_pipeline_d50_field_created": False,
        "bgs_folk_to_numeric_d50_mapping_applied": False,
        "psa_d50_interpolation_applied": False,
        "transport_rate_computed": False,
        "bedload_flux_computed": False,
        "suspended_load_computed": False,
        "total_load_computed": False,
        "net_transport_direction_computed": False,
        "erosion_deposition_prediction_computed": False,
        "morphological_change_rate_computed": False,
        "scour_computed": False,
        "burial_loss_rate_computed": False,
        "probability_computed": False,
        "risk_score_computed": False,
        "van_rijn_1984_transport_rate_formula_applied": False,
        "mar_012_or_mar_013_formulations_changed": False,
        "site_specific_sediment_transport_intensity_along_route": False,
        "limitations": [
            "Intensity is a dimensionless transport-POTENTIAL diagnostic for nine fixed "
            "tested noncohesive grain-size scenarios under real hydrodynamic forcing; it is "
            "not a site-specific sediment transport intensity because no defensible "
            "continuous route D50 field exists.",
            "No dimensional transport rate is computed: no continuous site D50/D90 field, "
            "grain-size distribution, intrawave phase-resolved transport, suspended-sediment "
            "concentration, accepted settling-velocity model, or transport-direction model "
            "exists for this project, and none is manufactured from BGS Folk class, nearest "
            "PSA point, tested D50 scenarios, or MAR-012 roughness scenarios.",
            "Each timestamp is the representative-wave-cycle PEAK relative excess, not a "
            "wave-cycle mean, phase-resolved, or net quantity.",
            "The analysed time series is not an independent probabilistic sample of future "
            "conditions; fractions are fractions of valid contemporaneous matched "
            "timestamps only.",
            "Zero intensity means no positive excess above the MAR-013 threshold for that "
            "tested scenario at that timestamp, not the absence of sediment movement in "
            "nature.",
        ],
        "references": [dict(r) for r in REFERENCES],
        "source_row_count": row_count,
        "hydro_pair_count": hydro_pair_count,
        "outputs": outputs,
    }


# --- Concise scientific report (Section 23) ------------------------------------------


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def print_transport_intensity_report(
    *,
    intensity_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    segments_gdf: gpd.GeoDataFrame,
    cross_checked_group_count: int,
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = [
        "=== Noncohesive Relative Excess Shields Transport-Potential Intensity (MAR-030) ===",
        "",
        f"  scientific_role        = {SCIENTIFIC_ROLE}",
        f"  source_scientific_role = {SOURCE_SCIENTIFIC_ROLE}",
        f"  definition             = max(mobility_ratio - 1, 0)  [{UNITS}]",
        "  stress basis           = MAR-013 tau_max_grain_skin_pa",
        f"  support semantics      = {TRANSPORT_INTENSITY_SUPPORT_SEMANTICS}",
        "",
        "## Timestamp-level table",
    ]
    if intensity_df.empty:
        lines.append("  (empty)")
    else:
        intensity = intensity_df["relative_excess_shields_intensity"]
        lines.append(
            f"  rows={len(intensity_df)} | hydro pairs={intensity_df['hydro_pair_id'].nunique()} | "
            f"tested D50 scenarios={intensity_df['tested_d50_mm'].nunique()} | "
            f"time range={intensity_df['time_utc'].min()} .. {intensity_df['time_utc'].max()}"
        )
        lines.append(
            f"  valid={int(intensity.notna().sum())} | null={int(intensity.isna().sum())} | "
            f"intensity min/max={_fmt(intensity.min())}/{_fmt(intensity.max())} | "
            f"strictly positive rows={int((intensity > 0).sum())}"
        )
    lines.append("")

    lines.append("## Per tested D50 scenario (across hydro pairs; scenarios are NOT combined)")
    if stats_df.empty:
        lines.append("  (empty)")
    else:
        for d50_mm, group in stats_df.groupby("tested_d50_mm", sort=True):
            p95 = group["relative_excess_intensity_p95"].dropna()
            frac = group["at_or_above_incipient_motion_fraction"].dropna()
            pos = group["strict_positive_excess_fraction"].dropna()
            lines.append(
                f"  D50={d50_mm:g} mm: p95 intensity min/median/max across pairs="
                f"{_fmt(p95.min() if len(p95) else None, '.3f')}/"
                f"{_fmt(p95.median() if len(p95) else None, '.3f')}/"
                f"{_fmt(p95.max() if len(p95) else None, '.3f')} | "
                f"max intensity={_fmt(group['relative_excess_intensity_max'].max(), '.3f')} | "
                f"at-or-above fraction range={_fmt(frac.min() if len(frac) else None, '.3f')}-"
                f"{_fmt(frac.max() if len(frac) else None, '.3f')} | "
                f"strict-positive fraction range={_fmt(pos.min() if len(pos) else None, '.3f')}-"
                f"{_fmt(pos.max() if len(pos) else None, '.3f')}"
            )
        lines.append(f"  fractions are {FRACTION_DENOMINATOR_DESCRIPTION} (not probabilities)")
        lines.append(
            f"  MAR-013 cross-check: {cross_checked_group_count} hydro-pair x D50 group(s) agree "
            "exactly on valid count and >= 1 exceedance count"
        )
    lines.append("")

    lines.append("## Scenario GIS layer")
    if segments_gdf.empty:
        lines.append("  (empty)")
    else:
        lines.append(
            f"  features={len(segments_gdf)} | segments="
            f"{segments_gdf['segment_id'].nunique()} | tested D50 per segment="
            f"{segments_gdf['tested_d50_mm'].nunique()} | crs={segments_gdf.crs}"
        )
    lines.append("")

    lines.extend(
        [
            "## Interpretation guard",
            "  REAL HYDRODYNAMIC FORCING (via MAR-013)                = YES",
            "  GENERIC TESTED-D50 TRANSPORT-POTENTIAL INTENSITY       = YES",
            "  SITE-SPECIFIC SEDIMENT TRANSPORT INTENSITY ALONG ROUTE = NO (no continuous D50)",
            "  SEDIMENT TRANSPORT RATE / BEDLOAD / SUSPENDED LOAD     = NO",
            "  NET TRANSPORT DIRECTION                                = NO",
            "  PREFERRED / ASSIGNED ACTUAL D50                        = NO",
            "  INTENSITY CLASSES OR RISK SCORE                        = NO",
            "  MAR-012 / MAR-013 FORMULATIONS CHANGED                 = NO",
            f"  {VAN_RIJN_RELATION}",
        ]
    )
    print("\n".join(lines), file=file)

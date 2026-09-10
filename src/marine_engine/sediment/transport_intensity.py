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

MAR-013 source contract (MAR-030A)
----------------------------------
MAR-030A is a source/integration integrity repair, not new sediment science
and not a new scientific validation. Before any intensity is derived, the
supplied table must be proven to be structurally the accepted MAR-013
product it claims to be: EVERY row carries
`scientific_role == NONCOHESIVE_SEDIMENT_MOBILITY_CAPACITY` (null, empty,
foreign, or mixed roles fail -- a missing source role is never converted
into an asserted `source_scientific_role`); the observed `tested_d50_mm`
vocabulary equals `noncohesive_mobility.TESTED_D50_SCENARIOS_MM` exactly
(no extra, missing, null, or non-finite scenario; exact floating equality,
never a tolerance that could reinterpret an arbitrary value as a scenario);
`tested_d50_m == tested_d50_mm / 1000` within floating-point serialization
noise only; `hydro_pair_id x time_utc x tested_d50_mm` source keys are
unique (duplicates are never aggregated, deduplicated, or averaged); every
`hydro_pair_id x time_utc` carries exactly the nine canonical scenarios
once each; and the preserved `incipient_motion_status` agrees with MAR-013's
own `classify_incipient_motion_status` re-applied to the supplied ratio
(a mismatch fails, it is never corrected). The MAR-013 statistics
cross-check first proves EXACT `hydro_pair_id x tested_d50_mm` key-set
equality (no missing/extra/duplicate group on either side) before comparing
counts, and the scenario GIS layer refuses to write a nominal nine-scenario
feature set with silent `n/a` values for a hydro-pair-supported segment
whose statistics are incomplete. Source rows are never mutated.
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


# --- MAR-013 source contract constants (MAR-030A) --------------------------------------

# The accepted MAR-013 long table is keyed by hydro pair x timestamp x tested scenario.
SOURCE_KEY_COLUMNS: tuple[str, ...] = ("hydro_pair_id", "time_utc", "tested_d50_mm")
DUPLICATE_MAR013_MOBILITY_SOURCE_KEY = "DUPLICATE_MAR013_MOBILITY_SOURCE_KEY"
# `tested_d50_m` must equal `tested_d50_mm / 1000` to within floating-point
# serialization noise only. Adjacent tested scenarios differ by a factor of two,
# so this tolerance can never reinterpret one scenario as another.
D50_UNIT_CONSISTENCY_RTOL = 1e-9
SOURCE_CONTRACT_SEMANTICS = (
    "Source/integration integrity of the consumed MAR-013 product (structural identity of "
    "the supplied table), not a new scientific validation and not new sediment physics."
)


class TransportIntensityError(Exception):
    """Base class for every controlled MAR-030 / MAR-030A failure."""


class TransportIntensitySchemaError(TransportIntensityError):
    """The MAR-013 source table lacks a required scientific column."""


class TransportIntensitySourceRoleError(TransportIntensityError):
    """Not every source row carries the accepted MAR-013 scientific role (MAR-030A)."""


class TransportIntensityScenarioContractError(TransportIntensityError):
    """The source violates the canonical MAR-013 tested-D50 scenario contract (MAR-030A)."""


class TransportIntensitySourceKeyError(TransportIntensityError):
    """Source `hydro_pair_id x time_utc x tested_d50_mm` keys are null or not unique (MAR-030A)."""


class IncipientMotionStatusConsistencyError(TransportIntensityError):
    """A preserved MAR-013 `incipient_motion_status` disagrees with its own ratio (MAR-030A)."""


class MobilityRatioConsistencyError(TransportIntensityError):
    """A source row's mobility ratio / intensity disagrees with its own stresses."""


class MobilityStatsCrossCheckError(TransportIntensityError):
    """MAR-030 and MAR-013 statistics differ in group identity or counts."""


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


def _as_float_array(series: pd.Series, column: str) -> np.ndarray:
    """A float view of a numeric source column; a non-numeric column is a contract failure."""

    try:
        return series.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TransportIntensityScenarioContractError(
            f"source column {column!r} is not numeric: {exc}"
        ) from exc


def _verify_source_scientific_role(mobility_df: pd.DataFrame) -> None:
    """EVERY row must carry `scientific_role == SOURCE_SCIENTIFIC_ROLE` (MAR-030A Section 5).

    Null / missing roles are counted, never dropped -- a missing source role
    must never be converted into an asserted `source_scientific_role`.
    """

    roles = mobility_df["scientific_role"]
    is_null = roles.isna().to_numpy()
    null_count = int(is_null.sum())
    present = roles[~is_null]
    unexpected = sorted(
        {str(value) for value in present.unique().tolist() if value != SOURCE_SCIENTIFIC_ROLE}
    )
    if null_count or unexpected:
        raise TransportIntensitySourceRoleError(
            f"every source row must carry scientific_role={SOURCE_SCIENTIFIC_ROLE!r}; found "
            f"{null_count} null/missing role(s) and unexpected role value(s) {unexpected} "
            f"across {len(mobility_df)} row(s)"
        )


def _verify_tested_d50_scenario_vocabulary(mobility_df: pd.DataFrame) -> None:
    """`set(tested_d50_mm) == set(TESTED_D50_SCENARIOS_MM)` exactly (MAR-030A Section 6).

    Exact floating equality is deliberate: the tested scenarios are fixed
    canonical constants serialized by MAR-013 as float64, not noisy
    measurements, so no tolerance may reinterpret an arbitrary value as one
    of them. Nothing is dropped, rounded, substituted, or mapped.
    """

    d50_mm = _as_float_array(mobility_df["tested_d50_mm"], "tested_d50_mm")
    non_finite = ~np.isfinite(d50_mm)
    if non_finite.any():
        raise TransportIntensityScenarioContractError(
            f"{int(non_finite.sum())} source row(s) carry a null/non-finite tested_d50_mm; "
            f"first offending row positions: {np.flatnonzero(non_finite)[:10].tolist()}"
        )
    observed = set(np.unique(d50_mm).tolist())
    canonical = set(TESTED_D50_SCENARIOS_MM)
    if observed != canonical:
        raise TransportIntensityScenarioContractError(
            "source tested_d50_mm vocabulary is not exactly the canonical MAR-013 scenario set "
            f"{sorted(canonical)}: missing {sorted(canonical - observed)}, unexpected "
            f"{sorted(observed - canonical)}"
        )


def _verify_d50_unit_consistency(mobility_df: pd.DataFrame) -> None:
    """`tested_d50_m == tested_d50_mm / 1000` per row within serialization noise (Section 7)."""

    d50_mm = _as_float_array(mobility_df["tested_d50_mm"], "tested_d50_mm")
    d50_m = _as_float_array(mobility_df["tested_d50_m"], "tested_d50_m")
    non_finite = ~np.isfinite(d50_m)
    if non_finite.any():
        raise TransportIntensityScenarioContractError(
            f"{int(non_finite.sum())} source row(s) carry a null/non-finite tested_d50_m; "
            f"first offending row positions: {np.flatnonzero(non_finite)[:10].tolist()}"
        )
    inconsistent = ~np.isclose(d50_m, d50_mm / 1000.0, rtol=D50_UNIT_CONSISTENCY_RTOL, atol=0.0)
    if inconsistent.any():
        positions = np.flatnonzero(inconsistent)
        examples = [
            {"row": int(i), "tested_d50_mm": float(d50_mm[i]), "tested_d50_m": float(d50_m[i])}
            for i in positions[:5]
        ]
        raise TransportIntensityScenarioContractError(
            f"{len(positions)} source row(s) have tested_d50_m inconsistent with "
            f"tested_d50_mm / 1000 (rtol={D50_UNIT_CONSISTENCY_RTOL}); the source is never "
            f"recomputed or overwritten; examples: {examples}"
        )


def _verify_source_key_uniqueness(mobility_df: pd.DataFrame) -> None:
    """`hydro_pair_id x time_utc x tested_d50_mm` must be non-null and unique (Section 9).

    Duplicate source rows fail outright -- they are never aggregated,
    deduplicated (first/last), or averaged.
    """

    key_columns = list(SOURCE_KEY_COLUMNS)
    null_key = mobility_df[key_columns].isna().any(axis=1).to_numpy()
    if null_key.any():
        raise TransportIntensitySourceKeyError(
            f"{int(null_key.sum())} source row(s) have a null hydro_pair_id / time_utc / "
            f"tested_d50_mm key component; first offending row positions: "
            f"{np.flatnonzero(null_key)[:10].tolist()}"
        )
    duplicated = mobility_df.duplicated(subset=key_columns, keep=False)
    if duplicated.any():
        duplicated_keys = mobility_df.loc[duplicated, key_columns].drop_duplicates()
        raise TransportIntensitySourceKeyError(
            f"{DUPLICATE_MAR013_MOBILITY_SOURCE_KEY}: {int(duplicated.sum())} source row(s) "
            f"share {len(duplicated_keys)} non-unique hydro_pair_id x time_utc x tested_d50_mm "
            "key(s); duplicates are never aggregated, deduplicated, or averaged; examples: "
            f"{duplicated_keys.head(5).to_dict('records')}"
        )


def _verify_per_timestamp_scenario_completeness(mobility_df: pd.DataFrame) -> int:
    """Every `hydro_pair_id x time_utc` carries exactly the nine canonical scenarios once (S8).

    Run AFTER the global vocabulary and key checks: with every value already
    proven to lie in the canonical set and every key non-null, a group of
    exactly nine rows with nine distinct D50 values is exactly the canonical
    set with no duplicate, missing, extra, or null scenario. Returns the
    number of hydro-pair timestamps verified. Table structure only -- no
    hydrodynamics are inferred.
    """

    expected = len(TESTED_D50_SCENARIOS_MM)
    grouped = mobility_df.groupby(["hydro_pair_id", "time_utc"], sort=False, dropna=False)[
        "tested_d50_mm"
    ]
    sizes = grouped.size()
    distinct = grouped.nunique()
    bad = (sizes != expected) | (distinct != expected)
    if bad.any():
        offending = sizes[bad]
        examples = [
            {"hydro_pair_id": key[0], "time_utc": str(key[1]), "scenario_rows": int(n)}
            for key, n in offending.head(5).items()
        ]
        raise TransportIntensityScenarioContractError(
            f"{len(offending)} hydro_pair_id x time_utc group(s) do not carry exactly the "
            f"{expected} canonical tested D50 scenarios once each (missing, duplicate, or "
            f"extra scenario); examples: {examples}"
        )
    return int(len(sizes))


def verify_incipient_motion_status_consistency(mobility_df: pd.DataFrame) -> None:
    """Preserved MAR-013 status must agree with MAR-013's own classifier (Section 10).

    Re-applies `noncohesive_mobility.classify_incipient_motion_status` (the
    accepted `>= 1` convention, no new threshold) to the supplied
    `mobility_ratio` and requires the supplied `incipient_motion_status` to
    match exactly, including null where the ratio is undefined. A mismatch
    fails; the status is never corrected here.
    """

    if mobility_df.empty:
        return
    ratio = _as_float_array(mobility_df["mobility_ratio"], "mobility_ratio")
    expected = ncm.classify_incipient_motion_status(ratio)
    expected_is_null = np.array([value is None for value in expected], dtype=bool)
    expected_text = np.where(expected_is_null, "", expected).astype(object)

    status = mobility_df["incipient_motion_status"]
    actual_is_null = status.isna().to_numpy()
    actual_text = np.where(actual_is_null, "", status.to_numpy(dtype=object)).astype(object)

    mismatch = (actual_is_null != expected_is_null) | (actual_text != expected_text)
    if mismatch.any():
        positions = np.flatnonzero(mismatch)
        examples = [
            {
                "row": int(i),
                "mobility_ratio": None if not np.isfinite(ratio[i]) else float(ratio[i]),
                "incipient_motion_status": None if actual_is_null[i] else str(actual_text[i]),
                "mar013_classifier": None if expected_is_null[i] else str(expected_text[i]),
            }
            for i in positions[:5]
        ]
        raise IncipientMotionStatusConsistencyError(
            f"{len(positions)} source row(s) carry an incipient_motion_status that disagrees "
            "with MAR-013's own classify_incipient_motion_status applied to the supplied "
            f"mobility_ratio; the status is never corrected here; examples: {examples}"
        )


def _empty_source_contract() -> dict[str, Any]:
    return {
        "source_contract_semantics": SOURCE_CONTRACT_SEMANTICS,
        "source_scientific_role_required": SOURCE_SCIENTIFIC_ROLE,
        "canonical_tested_d50_scenarios_mm": list(TESTED_D50_SCENARIOS_MM),
        "source_key_columns": list(SOURCE_KEY_COLUMNS),
        "d50_unit_consistency_rtol": D50_UNIT_CONSISTENCY_RTOL,
        "source_row_count": 0,
        "source_hydro_pair_count": 0,
        "source_hydro_pair_timestamp_count": 0,
        "source_hydro_pair_d50_group_count": 0,
        # Not applicable on an empty source: nothing was verified, nothing is asserted.
        "scientific_role_all_rows_verified": None,
        "exact_scenario_set_verified": None,
        "d50_unit_consistency_verified": None,
        "source_key_uniqueness_verified": None,
        "per_timestamp_scenario_completeness_verified": None,
        "incipient_motion_status_consistency_verified": None,
    }


def validate_mobility_source(mobility_df: pd.DataFrame) -> dict[str, Any]:
    """Prove the supplied table is structurally the accepted MAR-013 product (MAR-030A).

    Fails with a controlled `TransportIntensityError` subclass on: a missing
    required column; any null/empty/foreign `scientific_role`; a
    `tested_d50_mm` vocabulary other than exactly
    `TESTED_D50_SCENARIOS_MM`; a `tested_d50_m` inconsistent with
    `tested_d50_mm / 1000`; a null or duplicate
    `hydro_pair_id x time_utc x tested_d50_mm` key; a hydro-pair timestamp
    without exactly the nine canonical scenarios; or a preserved
    `incipient_motion_status` inconsistent with MAR-013's own classifier.
    Never mutates the source. Returns the machine-readable source-contract
    record (every `*_verified` flag is True only because the corresponding
    check ran and passed on this table; `None` on an empty source).
    """

    missing = [c for c in REQUIRED_MOBILITY_COLUMNS if c not in mobility_df.columns]
    if missing:
        raise TransportIntensitySchemaError(
            f"MAR-013 source table is missing required column(s): {missing}"
        )
    record = _empty_source_contract()
    if mobility_df.empty:
        return record

    _verify_source_scientific_role(mobility_df)
    record["scientific_role_all_rows_verified"] = True
    _verify_tested_d50_scenario_vocabulary(mobility_df)
    record["exact_scenario_set_verified"] = True
    _verify_d50_unit_consistency(mobility_df)
    record["d50_unit_consistency_verified"] = True
    _verify_source_key_uniqueness(mobility_df)
    record["source_key_uniqueness_verified"] = True
    timestamp_count = _verify_per_timestamp_scenario_completeness(mobility_df)
    record["per_timestamp_scenario_completeness_verified"] = True
    verify_incipient_motion_status_consistency(mobility_df)
    record["incipient_motion_status_consistency_verified"] = True

    record["source_row_count"] = int(len(mobility_df))
    record["source_hydro_pair_count"] = int(mobility_df["hydro_pair_id"].nunique())
    record["source_hydro_pair_timestamp_count"] = timestamp_count
    record["source_hydro_pair_d50_group_count"] = int(
        len(mobility_df[["hydro_pair_id", "tested_d50_mm"]].drop_duplicates())
    )
    return record


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
    """Cross-check MAR-030 counts against the accepted MAR-013 statistics (Section 15 / 30A-11).

    First proves EXACT `hydro_pair_id x tested_d50_mm` group identity: no
    duplicate key in either table, and
    `set(MAR-030 keys) == set(MAR-013 keys)` (a group missing from or extra
    to either side fails -- an inner-merge intersection is never the only
    identity test). Only then, where definitions are equivalent -- MAR-013
    `valid_count` (finite mobility ratios) vs `valid_intensity_timestamp_count`,
    and MAR-013 `threshold_exceedance_count` (`mobility_ratio >= 1`) vs
    `at_or_above_incipient_motion_count` -- the two products must agree
    exactly for every group. Returns the number of groups compared.
    """

    if intensity_stats_df.empty and mobility_stats_df.empty:
        return 0
    keys = ["hydro_pair_id", "tested_d50_mm"]
    required = {
        "MAR-030": (
            intensity_stats_df,
            [*keys, "valid_intensity_timestamp_count", "at_or_above_incipient_motion_count"],
        ),
        "MAR-013": (mobility_stats_df, [*keys, "valid_count", "threshold_exceedance_count"]),
    }
    for label, (frame, columns) in required.items():
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise TransportIntensitySchemaError(
                f"{label} statistics are missing required column(s): {missing}"
            )
        duplicated = frame.duplicated(subset=keys, keep=False)
        if duplicated.any():
            raise MobilityStatsCrossCheckError(
                f"{label} statistics carry duplicated hydro_pair_id x tested_d50_mm key(s): "
                f"{frame.loc[duplicated, keys].drop_duplicates().head(5).to_dict('records')}"
            )

    mar030_keys = set(intensity_stats_df[keys].itertuples(index=False, name=None))
    mar013_keys = set(mobility_stats_df[keys].itertuples(index=False, name=None))
    if mar030_keys != mar013_keys:
        only_030 = sorted(mar030_keys - mar013_keys, key=str)
        only_013 = sorted(mar013_keys - mar030_keys, key=str)
        raise MobilityStatsCrossCheckError(
            "MAR-030 and MAR-013 hydro_pair_id x tested_d50_mm key sets differ: "
            f"{len(only_030)} group(s) only in MAR-030 (e.g. {only_030[:5]}), "
            f"{len(only_013)} group(s) only in MAR-013 (e.g. {only_013[:5]})"
        )

    merged = intensity_stats_df.merge(
        mobility_stats_df[[*keys, "valid_count", "threshold_exceedance_count"]],
        on=keys,
        how="inner",
    )
    if len(merged) != len(mar030_keys):
        raise MobilityStatsCrossCheckError(
            f"merged {len(merged)} group(s) but {len(mar030_keys)} unique key(s) were expected"
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

    stats_by_key = None
    available_keys: set[tuple[Any, float]] = set()
    if not stats_df.empty:
        duplicated = stats_df.duplicated(subset=["hydro_pair_id", "tested_d50_mm"], keep=False)
        if duplicated.any():
            duplicated_stats_keys = stats_df.loc[duplicated, ["hydro_pair_id", "tested_d50_mm"]]
            raise TransportIntensityScenarioContractError(
                "MAR-030 statistics carry duplicated hydro_pair_id x tested_d50_mm key(s): "
                f"{duplicated_stats_keys.head(5).to_dict('records')}"
            )
        stats_by_key = stats_df.set_index(["hydro_pair_id", "tested_d50_mm"])
        available_keys = set(
            stats_df[["hydro_pair_id", "tested_d50_mm"]].itertuples(index=False, name=None)
        )

    # MAR-030A Section 12: every accepted route segment WITH a hydro-pair assignment
    # must have the complete canonical nine-scenario statistics set; a nominal
    # nine-scenario feature set with silent n/a values is never written for it.
    # A segment with no hydro-pair support (accepted MAR-013 semantics: null
    # `hydro_pair_id`) keeps nine features with null statistics, as before.
    supported_pairs = [p for p in mobility_segments_gdf["hydro_pair_id"].unique() if pd.notna(p)]
    incomplete = {
        pair_id: [d for d in sorted(TESTED_D50_SCENARIOS_MM) if (pair_id, d) not in available_keys]
        for pair_id in supported_pairs
    }
    incomplete = {pair_id: missing for pair_id, missing in incomplete.items() if missing}
    if incomplete:
        examples = dict(list(incomplete.items())[:5])
        raise TransportIntensityScenarioContractError(
            f"{len(incomplete)} hydro-pair-supported MAR-013 route segment(s) lack complete "
            f"{len(TESTED_D50_SCENARIOS_MM)}-scenario intensity statistics; the scenario GIS "
            "layer is never written with silent n/a values for a supported segment; missing "
            f"tested_d50_mm by hydro_pair_id: {examples}"
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
    source_contract: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic metadata dict; only `outputs`, the counts, and the contract record vary.

    `source_contract` is the record returned by `validate_mobility_source`
    (MAR-030A). The MAR-013 statistics key-set match is recorded as verified
    only when the cross-check covered exactly every `hydro_pair_id x
    tested_d50_mm` group the validated source carries.
    """

    expected_groups = int(source_contract.get("source_hydro_pair_d50_group_count") or 0)
    if cross_checked_group_count != expected_groups:
        raise MobilityStatsCrossCheckError(
            f"{cross_checked_group_count} hydro_pair_id x tested_d50_mm group(s) were "
            f"cross-checked but the validated source carries {expected_groups}"
        )
    source_contract_record = {
        **source_contract,
        "mar013_stats_key_set_match_verified": True if expected_groups else None,
        "mar013_stats_groups_cross_checked": cross_checked_group_count,
    }

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
        "source_contract": source_contract_record,
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


def _flag_text(value: bool | None) -> str:
    if value is None:
        return "N/A (empty source)"
    return "YES" if value else "NO"


def print_transport_intensity_report(
    *,
    intensity_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    segments_gdf: gpd.GeoDataFrame,
    cross_checked_group_count: int,
    source_contract: dict[str, Any] | None = None,
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

    if source_contract is not None:
        lines.append("## MAR-013 source contract (MAR-030A: source/integration integrity)")
        for flag in (
            "scientific_role_all_rows_verified",
            "exact_scenario_set_verified",
            "d50_unit_consistency_verified",
            "source_key_uniqueness_verified",
            "per_timestamp_scenario_completeness_verified",
            "incipient_motion_status_consistency_verified",
        ):
            lines.append(f"  {flag:<48} = {_flag_text(source_contract.get(flag))}")
        lines.append(
            f"  {'mar013_stats_key_set_match_verified':<48} = "
            f"{_flag_text(True if cross_checked_group_count else None)} "
            f"({cross_checked_group_count} group(s))"
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

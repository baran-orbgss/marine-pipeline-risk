"""Canonical CPT/CPTU measurement normalization and QA (MAR-032 Sections 13-19).

Input: a structured numeric observation table that a SOURCE-SPECIFIC reader has already parsed
from a defensibly machine-readable file, plus EXPLICIT declarations of what each source column
means and which unit it carries. Output: the canonical, SI-friendly measurement table.

Rules enforced here, not left to callers:

* a canonical field is populated only when the source semantics are declared, the source unit is
  declared, and the (source unit -> canonical unit) factor is in the explicit conversion table;
  anything else stays null and is reported as unresolved -- no unit is ever guessed;
* `depth_bsf_m` is populated only when the declared depth reference is `DEPTH_BELOW_SEABED`;
* `qt_mpa` is populated only when a source column is explicitly declared as corrected cone
  resistance -- it is never copied from qc and never computed from qc + (1-a)u (no area-ratio
  or pore-pressure correction exists in MAR-032);
* every source channel is preserved verbatim (`raw__<column>`) beside its normalized form;
* QA reports counts (duplicates, depth-order violations, missing values); it never repairs. There
  is no smoothing, interpolation, gap filling, despiking, resampling or averaging in this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from marine_engine.geotechnical import cpt_contract as contract

__all__ = [
    "ChannelDeclaration",
    "DepthDeclaration",
    "CanonicalBuild",
    "CptProfileError",
    "conversion_factor",
    "build_canonical_measurements",
    "concat_canonical",
    "compute_profile_qa",
    "canonical_channels_present",
]


class CptProfileError(ValueError):
    """A declaration is internally inconsistent (e.g. two channels claim the same canonical
    field). Raised rather than silently resolved."""


@dataclass(frozen=True)
class ChannelDeclaration:
    """What one SOURCE column means. `canonical_field` may be None to preserve the column raw only
    (e.g. inclination, timestamps). `source_unit` is the canonical unit token of the source
    (e.g. "MPa") -- None means the source did not state a unit, in which case the channel is
    preserved raw and its canonical field (if any) stays null. `source_unit_token` is the verbatim
    unit string as written in the source (e.g. "m.") for provenance."""

    source_column: str
    source_unit: str | None
    canonical_field: str | None
    semantics_basis: str
    source_unit_token: str | None = None

    def __post_init__(self) -> None:
        if self.canonical_field is not None and (
            self.canonical_field not in contract.CANONICAL_MEASUREMENT_FIELDS
        ):
            raise CptProfileError(
                f"channel {self.source_column!r}: unknown canonical field "
                f"{self.canonical_field!r} (allowed: {contract.CANONICAL_MEASUREMENT_FIELDS})"
            )
        if not self.semantics_basis.strip():
            raise CptProfileError(
                f"channel {self.source_column!r}: semantics_basis must state the source evidence"
            )


@dataclass(frozen=True)
class DepthDeclaration:
    """What the source depth column means (Section 16). `depth_reference` must be a member of
    `DEPTH_REFERENCES`; only `DEPTH_BELOW_SEABED` allows `depth_bsf_m` to be populated."""

    source_column: str
    source_unit: str | None
    depth_reference: str
    reference_basis: str
    source_unit_token: str | None = None

    def __post_init__(self) -> None:
        if self.depth_reference not in contract.DEPTH_REFERENCES:
            raise CptProfileError(
                f"unknown depth_reference {self.depth_reference!r} "
                f"(allowed: {sorted(contract.DEPTH_REFERENCES)})"
            )


@dataclass(frozen=True)
class CanonicalBuild:
    measurements: pd.DataFrame
    field_provenance: tuple[dict[str, Any], ...]
    unresolved: tuple[str, ...]


def conversion_factor(from_unit: str | None, to_unit: str) -> float | None:
    """Deterministic factor from the explicit table, or None when the pair is not defined (the
    caller must then leave the canonical value null). A None `from_unit` is always None."""

    if from_unit is None:
        return None
    return contract.UNIT_CONVERSION_FACTORS.get((from_unit, to_unit))


def _coerce_numeric(series: pd.Series) -> tuple[pd.Series, int]:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    non_numeric = int((numeric.isna() & series.notna()).sum())
    return numeric, non_numeric


def build_canonical_measurements(
    observations: pd.DataFrame,
    *,
    source_id: str,
    test_id: str,
    depth: DepthDeclaration,
    channels: Sequence[ChannelDeclaration],
    location_id: str | None = None,
    observation_index_column: str | None = None,
) -> CanonicalBuild:
    """Normalize ONE test's observation table. Rows are kept in source order; nothing is dropped,
    merged, sorted, resampled or interpolated. Returns the canonical frame plus a per-field
    provenance record (original column, original unit, normalized field, conversion applied)."""

    canonical_targets = [c.canonical_field for c in channels if c.canonical_field is not None]
    duplicates = {f for f in canonical_targets if canonical_targets.count(f) > 1}
    if duplicates:
        raise CptProfileError(
            f"more than one source channel is declared for canonical field(s) {sorted(duplicates)}"
        )
    for column in [depth.source_column, *(c.source_column for c in channels)]:
        if column not in observations.columns:
            raise CptProfileError(f"declared source column {column!r} is absent from observations")
    if observation_index_column is not None and observation_index_column not in observations:
        raise CptProfileError(
            f"observation_index_column {observation_index_column!r} is absent from observations"
        )

    n = len(observations)
    out: dict[str, Any] = {
        contract.SOURCE_ID: [source_id] * n,
        contract.TEST_ID: [test_id] * n,
        contract.LOCATION_ID: [location_id] * n,
    }
    if observation_index_column is not None:
        idx, _bad = _coerce_numeric(observations[observation_index_column])
        out[contract.OBSERVATION_INDEX] = idx.to_numpy()
    else:
        out[contract.OBSERVATION_INDEX] = np.arange(n, dtype="float64")

    provenance: list[dict[str, Any]] = []
    unresolved: list[str] = []

    depth_values, depth_bad = _coerce_numeric(observations[depth.source_column])
    out[contract.DEPTH_SOURCE_VALUE] = depth_values.to_numpy()
    out[contract.DEPTH_SOURCE_UNIT] = [depth.source_unit] * n
    out[contract.DEPTH_REFERENCE_FIELD] = [depth.depth_reference] * n
    depth_factor = conversion_factor(
        depth.source_unit, contract.CANONICAL_FIELD_UNITS["depth_bsf_m"]
    )
    if depth.depth_reference == contract.DEPTH_BELOW_SEABED and depth_factor is not None:
        out[contract.DEPTH_BSF_M] = (depth_values * depth_factor).to_numpy()
        depth_conversion: str | None = f"x{depth_factor:g} ({depth.source_unit} -> m)"
    else:
        out[contract.DEPTH_BSF_M] = np.full(n, np.nan)
        depth_conversion = None
        if depth.depth_reference != contract.DEPTH_BELOW_SEABED:
            unresolved.append(f"{contract.DEPTH_BSF_M}: {contract.DEPTH_REFERENCE_UNRESOLVED}")
        if depth_factor is None:
            unresolved.append(
                f"{contract.DEPTH_BSF_M}: source depth unit {depth.source_unit!r} has no "
                "deterministic conversion to m"
            )
    provenance.append(
        {
            "original_column": depth.source_column,
            "original_unit": depth.source_unit,
            "original_unit_token": depth.source_unit_token,
            "normalized_field": contract.DEPTH_BSF_M,
            "conversion_applied": depth_conversion,
            "depth_reference": depth.depth_reference,
            "semantics_basis": depth.reference_basis,
            "non_numeric_source_tokens": depth_bad,
        }
    )

    for field in contract.CANONICAL_MEASUREMENT_FIELDS:
        out[field] = np.full(n, np.nan)

    for ch in channels:
        values, bad = _coerce_numeric(observations[ch.source_column])
        out[f"{contract.RAW_CHANNEL_PREFIX}{ch.source_column}"] = values.to_numpy()
        conversion: str | None = None
        if ch.canonical_field is not None:
            target_unit = contract.CANONICAL_FIELD_UNITS[ch.canonical_field]
            factor = conversion_factor(ch.source_unit, target_unit)
            if factor is None:
                unresolved.append(
                    f"{ch.canonical_field}: source column {ch.source_column!r} unit "
                    f"{ch.source_unit!r} is unknown or has no deterministic conversion to "
                    f"{target_unit}; value left null"
                )
            else:
                out[ch.canonical_field] = (values * factor).to_numpy()
                conversion = f"x{factor:g} ({ch.source_unit} -> {target_unit})"
        elif ch.source_unit is None:
            unresolved.append(
                f"raw__{ch.source_column}: source unit not stated; preserved raw only"
            )
        provenance.append(
            {
                "original_column": ch.source_column,
                "original_unit": ch.source_unit,
                "original_unit_token": ch.source_unit_token,
                "normalized_field": ch.canonical_field,
                "conversion_applied": conversion,
                "semantics_basis": ch.semantics_basis,
                "non_numeric_source_tokens": bad,
            }
        )

    frame = pd.DataFrame(out)
    return CanonicalBuild(
        measurements=frame,
        field_provenance=tuple(provenance),
        unresolved=tuple(unresolved),
    )


def concat_canonical(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Row-wise concatenation in the given order; no sorting, no deduplication."""

    if not frames:
        columns = [
            *contract.IDENTITY_FIELDS,
            contract.DEPTH_SOURCE_VALUE,
            contract.DEPTH_SOURCE_UNIT,
            contract.DEPTH_REFERENCE_FIELD,
            contract.DEPTH_BSF_M,
            *contract.CANONICAL_MEASUREMENT_FIELDS,
        ]
        return pd.DataFrame(columns=columns)
    return pd.concat(list(frames), ignore_index=True)


def canonical_channels_present(measurements: pd.DataFrame) -> tuple[str, ...]:
    present = []
    for field in contract.CANONICAL_MEASUREMENT_FIELDS:
        if field in measurements.columns and bool(np.isfinite(measurements[field]).any()):
            present.append(field)
    return tuple(present)


def _finite_fraction(series: pd.Series) -> float | None:
    if len(series) == 0:
        return None
    return float(np.isfinite(series.to_numpy(dtype="float64")).mean())


def compute_profile_qa(measurements: pd.DataFrame) -> dict[str, Any]:
    """Section 19 statistics. Deterministic for identical input. Reports; never repairs."""

    df = measurements
    row_count = int(len(df))
    if row_count == 0:
        return {
            "row_count": 0,
            "test_count": 0,
            "tests": {},
            "channels_present": [],
            "channel_finite_fraction": {},
            "channel_missing_count": {},
            "duplicate_observation_identity_count": 0,
            "duplicate_depth_row_count": 0,
            "contradictory_duplicate_depth_row_count": 0,
            "depth_order_violation_count": 0,
            "negative_depth_row_count": 0,
            "depth_bsf_available": False,
        }

    identity = [contract.SOURCE_ID, contract.TEST_ID, contract.OBSERVATION_INDEX]
    dup_identity = int(df.duplicated(subset=identity, keep=False).sum())

    depth_key = [contract.SOURCE_ID, contract.TEST_ID, contract.DEPTH_SOURCE_VALUE]
    dup_depth_mask = df.duplicated(subset=depth_key, keep=False)
    dup_depth = int(dup_depth_mask.sum())
    contradictory = 0
    if dup_depth:
        value_cols = [
            c
            for c in df.columns
            if c in contract.CANONICAL_MEASUREMENT_FIELDS
            or c.startswith(contract.RAW_CHANNEL_PREFIX)
        ]
        grouped = df.loc[dup_depth_mask].groupby(depth_key, dropna=False, sort=False)
        for _key, group in grouped:
            rounded = group[value_cols].round(12)
            if len(rounded.drop_duplicates()) > 1:
                contradictory += int(len(group))

    tests: dict[str, dict[str, Any]] = {}
    order_violations_total = 0
    negative_total = 0
    for test_id, group in df.groupby(contract.TEST_ID, sort=True, dropna=False):
        depth_values = group[contract.DEPTH_SOURCE_VALUE].to_numpy(dtype="float64")
        finite_depth = depth_values[np.isfinite(depth_values)]
        diffs = np.diff(finite_depth)
        violations = int((diffs < 0).sum())
        negatives = int((finite_depth < 0).sum())
        order_violations_total += violations
        negative_total += negatives
        tests[str(test_id)] = {
            "row_count": int(len(group)),
            "depth_source_min": float(finite_depth.min()) if finite_depth.size else None,
            "depth_source_max": float(finite_depth.max()) if finite_depth.size else None,
            "depth_order_violation_count": violations,
            "negative_depth_row_count": negatives,
        }

    channel_fields = [
        c for c in [contract.DEPTH_BSF_M, *contract.CANONICAL_MEASUREMENT_FIELDS] if c in df.columns
    ] + [c for c in df.columns if c.startswith(contract.RAW_CHANNEL_PREFIX)]
    finite_fraction = {c: _finite_fraction(df[c]) for c in channel_fields}
    missing_count = {
        c: int((~np.isfinite(df[c].to_numpy(dtype="float64"))).sum()) for c in channel_fields
    }

    depth_bsf = (
        df[contract.DEPTH_BSF_M].to_numpy(dtype="float64") if contract.DEPTH_BSF_M in df else []
    )
    return {
        "row_count": row_count,
        "test_count": int(df[contract.TEST_ID].nunique(dropna=False)),
        "tests": tests,
        "channels_present": list(canonical_channels_present(df)),
        "channel_finite_fraction": finite_fraction,
        "channel_missing_count": missing_count,
        "duplicate_observation_identity_count": dup_identity,
        "duplicate_depth_row_count": dup_depth,
        "contradictory_duplicate_depth_row_count": contradictory,
        "depth_order_violation_count": order_violations_total,
        "negative_depth_row_count": negative_total,
        "depth_bsf_available": bool(len(depth_bsf) and np.isfinite(depth_bsf).any()),
    }


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

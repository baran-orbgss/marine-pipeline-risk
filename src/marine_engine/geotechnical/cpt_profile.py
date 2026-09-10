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

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from marine_engine.geotechnical import cpt_contract as contract

__all__ = [
    "ChannelDeclaration",
    "DepthDeclaration",
    "CanonicalBuild",
    "CanonicalProductMarker",
    "CanonicalLineage",
    "CanonicalProductVerification",
    "CptProfileError",
    "CANONICAL_PRODUCT_COLUMNS",
    "canonical_product_columns_missing",
    "structural_columns_missing",
    "verify_canonical_lineage",
    "verify_canonical_cpt_product",
    "conversion_factor",
    "build_canonical_measurements",
    "concat_canonical",
    "compute_profile_qa",
    "canonical_channels_present",
    "canonical_units_contract",
    "canonical_value_sha256",
    "write_canonical_cpt_measurements",
    "read_canonical_cpt_product_marker",
]

# Every column a CPT_CANONICAL_PROFILE_V1 measurements product MUST carry (MAR-032A Section 8;
# MAR-032B Sections 4-5). This tuple is the SINGLE schema authority: the canonical writer refuses
# anything narrower and the reader/adapter verifies canonical product identity against the same
# tuple (`verify_canonical_cpt_product`). It is a strict superset of
# `contract.CANONICAL_STRUCTURAL_COLUMNS`, which only says "structurally resembles CPT data" and
# never establishes V1 product identity. Extra columns (e.g. `raw__*`) remain allowed.
CANONICAL_PRODUCT_COLUMNS = (
    *contract.IDENTITY_FIELDS,
    contract.DEPTH_SOURCE_VALUE,
    contract.DEPTH_SOURCE_UNIT,
    contract.DEPTH_REFERENCE_FIELD,
    contract.DEPTH_BSF_M,
    *contract.CANONICAL_MEASUREMENT_FIELDS,
)


def canonical_product_columns_missing(columns: Sequence[str]) -> list[str]:
    """Required V1 product columns absent from `columns`, in contract order (empty = complete)."""

    present = set(columns)
    return [c for c in CANONICAL_PRODUCT_COLUMNS if c not in present]


def structural_columns_missing(columns: Sequence[str]) -> list[str]:
    """Structural-lookalike columns absent from `columns` (observational only; never identity)."""

    present = set(columns)
    return [c for c in contract.CANONICAL_STRUCTURAL_COLUMNS if c not in present]


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


# --- MAR-032A: canonical CPT product marker (file-level Parquet schema metadata) ------------------


def canonical_units_contract() -> str:
    """Deterministic serialized unit contract: sorted-key, whitespace-free JSON of
    `contract.CANONICAL_FIELD_UNITS`. Stored verbatim in the product marker so a reader can prove
    the file was written against the same unit contract it is about to be read with."""

    return json.dumps(contract.CANONICAL_FIELD_UNITS, sort_keys=True, separators=(",", ":"))


def canonical_value_sha256(measurements: pd.DataFrame) -> str:
    """Content-VALUE identity of a canonical table: SHA-256 of a deterministic CSV serialization
    (full-precision floats, LF line ends). Independent of Parquet encoding and file metadata, so
    two files with different raw bytes but identical rows/values share this hash."""

    payload = measurements.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CanonicalProductMarker:
    """What the file's own schema metadata says about itself (OBSERVED, never declared), plus the
    verification verdict against the current canonical contract. `observed` holds every
    `marine_engine_*` metadata entry exactly as read (decoded), whether or not it verifies."""

    observed: dict[str, str]
    product_role: str | None
    contract_version: str | None
    evidence_id: str | None
    canonical_units: str | None
    verified: bool
    problems: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_metadata": dict(self.observed),
            "product_role": self.product_role,
            "contract_version": self.contract_version,
            "evidence_id": self.evidence_id,
            "canonical_units": self.canonical_units,
            "verified": self.verified,
            "problems": list(self.problems),
            "expected": {
                contract.PRODUCT_ROLE_METADATA_KEY: contract.MEASURED_CPT_CPTU_PROFILE,
                contract.PRODUCT_CONTRACT_METADATA_KEY: contract.CPT_CANONICAL_PROFILE_CONTRACT,
                contract.PRODUCT_CANONICAL_UNITS_METADATA_KEY: canonical_units_contract(),
            },
        }


def write_canonical_cpt_measurements(
    measurements: pd.DataFrame, path: Path, *, evidence_id: str
) -> Path:
    """The ONLY writer of a canonical MAR CPT measurements Parquet product. Writes the frame's
    values unchanged (same pyarrow conversion pandas itself uses) and stamps the explicit,
    versioned product marker into the file-level schema metadata:

        marine_engine_product_role     = MEASURED_CPT_CPTU_PROFILE
        marine_engine_cpt_contract     = CPT_CANONICAL_PROFILE_V1
        marine_engine_evidence_id      = <non-empty evidence id>
        marine_engine_canonical_units  = canonical_units_contract()

    Refuses (raises `CptProfileError`) an empty evidence id, a frame missing any
    `CANONICAL_PRODUCT_COLUMNS` member -- a product can never be narrower than its contract --
    and (MAR-032B Section 7) a frame whose row-level `source_id` lineage is not ONE non-null,
    non-blank value exactly equal to `evidence_id`. Nothing is normalized, guessed or rewritten:
    an inconsistent frame is rejected and no file is written."""

    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise CptProfileError("canonical CPT product requires a non-empty evidence_id")
    missing = canonical_product_columns_missing(list(measurements.columns))
    if missing:
        raise CptProfileError(
            f"canonical CPT product is missing required column(s) {missing}; refusing to write"
        )
    lineage = verify_canonical_lineage(measurements, evidence_id=evidence_id)
    if not lineage.verified:
        raise CptProfileError(
            "canonical CPT product evidence lineage is inconsistent "
            f"({'; '.join(lineage.problems)}); refusing to write"
        )
    table = pa.Table.from_pandas(measurements, preserve_index=False)
    marker = {
        contract.PRODUCT_ROLE_METADATA_KEY: contract.MEASURED_CPT_CPTU_PROFILE,
        contract.PRODUCT_CONTRACT_METADATA_KEY: contract.CPT_CANONICAL_PROFILE_CONTRACT,
        contract.PRODUCT_EVIDENCE_ID_METADATA_KEY: evidence_id,
        contract.PRODUCT_CANONICAL_UNITS_METADATA_KEY: canonical_units_contract(),
    }
    metadata = dict(table.schema.metadata or {})
    metadata.update({k.encode("utf-8"): v.encode("utf-8") for k, v in marker.items()})
    table = table.replace_schema_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def read_canonical_cpt_product_marker(path: Path) -> CanonicalProductMarker:
    """Read the OBSERVED product marker from a Parquet file's schema metadata and verify it against
    the current canonical contract. Reads the schema only (no row data). Raises whatever pyarrow
    raises for a file that is not a readable Parquet file; the caller decides how to classify it.

    Verification requires ALL of: role == MEASURED_CPT_CPTU_PROFILE, contract ==
    CPT_CANONICAL_PROFILE_V1, non-empty evidence id, and a canonical-units entry that parses to
    exactly `contract.CANONICAL_FIELD_UNITS`. Anything else is reported as a named problem."""

    schema = pq.read_schema(path)
    observed: dict[str, str] = {}
    for raw_key, raw_value in (schema.metadata or {}).items():
        try:
            key = raw_key.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if key.startswith("marine_engine_"):
            observed[key] = raw_value.decode("utf-8", errors="replace")

    problems: list[str] = []
    role = observed.get(contract.PRODUCT_ROLE_METADATA_KEY)
    if role is None:
        problems.append(f"no {contract.PRODUCT_ROLE_METADATA_KEY} metadata")
    elif role != contract.MEASURED_CPT_CPTU_PROFILE:
        problems.append(f"product role {role!r} is not {contract.MEASURED_CPT_CPTU_PROFILE!r}")
    version = observed.get(contract.PRODUCT_CONTRACT_METADATA_KEY)
    if version is None:
        problems.append(f"no {contract.PRODUCT_CONTRACT_METADATA_KEY} metadata")
    elif version != contract.CPT_CANONICAL_PROFILE_CONTRACT:
        problems.append(
            f"canonical contract {version!r} is not {contract.CPT_CANONICAL_PROFILE_CONTRACT!r}"
        )
    evidence_id = observed.get(contract.PRODUCT_EVIDENCE_ID_METADATA_KEY)
    if evidence_id is None or not evidence_id.strip():
        problems.append(f"missing or empty {contract.PRODUCT_EVIDENCE_ID_METADATA_KEY} metadata")
    units = observed.get(contract.PRODUCT_CANONICAL_UNITS_METADATA_KEY)
    if units is None:
        problems.append(f"no {contract.PRODUCT_CANONICAL_UNITS_METADATA_KEY} metadata")
    else:
        try:
            parsed = json.loads(units)
        except ValueError:
            parsed = None
        if parsed != contract.CANONICAL_FIELD_UNITS:
            problems.append(
                "canonical unit contract in file differs from the current "
                f"{contract.CPT_CANONICAL_PROFILE_CONTRACT} unit contract"
            )
    return CanonicalProductMarker(
        observed=observed,
        product_role=role,
        contract_version=version,
        evidence_id=evidence_id,
        canonical_units=units,
        verified=not problems,
        problems=tuple(problems),
    )


# --- MAR-032B: schema binding and evidence lineage of a CPT_CANONICAL_PROFILE_V1 product ----------
# Permanent invariants: a valid marker + the small structural-lookalike column subset is NOT a
# verified V1 product, and the marker's evidence id must be the row-level `source_id` identity
# (exactly one non-null, non-blank value, equal to the evidence id, no normalization).

_LINEAGE_VALUES_REPORTED = 20


@dataclass(frozen=True)
class CanonicalLineage:
    """OBSERVED evidence lineage of a canonical measurements frame against a marker evidence id.

    `row_source_id_values` are the DISTINCT row-level `source_id` values exactly as found (a null is
    reported as None; a blank string is reported verbatim), capped for reporting; nothing is
    stripped, cased or otherwise normalized before comparison."""

    marker_evidence_id: str | None
    row_source_id_values: tuple[Any, ...]
    row_source_id_unique_count: int
    null_or_blank_source_id_row_count: int
    evidence_id_matches_row_source_id: bool
    verified: bool
    problems: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker_evidence_id": self.marker_evidence_id,
            "row_source_id_values": list(self.row_source_id_values),
            "row_source_id_unique_count": self.row_source_id_unique_count,
            "null_or_blank_source_id_row_count": self.null_or_blank_source_id_row_count,
            "evidence_id_matches_row_source_id": self.evidence_id_matches_row_source_id,
            "verified": self.verified,
            "problems": list(self.problems),
        }


def _is_null_or_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def verify_canonical_lineage(
    measurements: pd.DataFrame, *, evidence_id: str | None
) -> CanonicalLineage:
    """MAR-032B Sections 7-8. Verified only when the `source_id` column exists, no row is null or
    blank, exactly one distinct value exists and that value == `evidence_id` exactly. Every
    other case is reported as a named problem; nothing is repaired."""

    problems: list[str] = []
    if contract.SOURCE_ID not in measurements.columns:
        problems.append(f"{contract.SOURCE_ID} column absent; row-level lineage cannot be verified")
        return CanonicalLineage(
            marker_evidence_id=evidence_id,
            row_source_id_values=(),
            row_source_id_unique_count=0,
            null_or_blank_source_id_row_count=0,
            evidence_id_matches_row_source_id=False,
            verified=False,
            problems=tuple(problems),
        )
    raw_values = measurements[contract.SOURCE_ID].tolist()
    null_or_blank = sum(1 for v in raw_values if _is_null_or_blank(v))
    distinct: list[Any] = []
    seen: set[Any] = set()
    for value in raw_values:
        key = None if _is_null_or_blank(value) and not isinstance(value, str) else value
        if key not in seen:
            seen.add(key)
            distinct.append(key)
    valid_distinct = [v for v in distinct if not _is_null_or_blank(v)]
    unique_count = len(valid_distinct)
    if len(raw_values) == 0:
        problems.append("no measurement rows; row-level lineage cannot be verified")
    if null_or_blank:
        problems.append(f"{null_or_blank} row(s) carry a null or blank {contract.SOURCE_ID}")
    non_string = [v for v in valid_distinct if not isinstance(v, str)]
    if non_string:
        problems.append(f"{contract.SOURCE_ID} carries non-string value(s): {non_string[:5]}")
    if unique_count > 1:
        problems.append(
            f"{unique_count} distinct {contract.SOURCE_ID} values in one canonical product "
            f"(exactly one is required): {valid_distinct[:_LINEAGE_VALUES_REPORTED]}"
        )
    matches = (
        unique_count == 1
        and null_or_blank == 0
        and isinstance(evidence_id, str)
        and valid_distinct[0] == evidence_id
    )
    if unique_count == 1 and not matches and null_or_blank == 0:
        problems.append(
            f"row {contract.SOURCE_ID} {valid_distinct[0]!r} does not equal marker evidence id "
            f"{evidence_id!r} (exact comparison; nothing normalized)"
        )
    return CanonicalLineage(
        marker_evidence_id=evidence_id,
        row_source_id_values=tuple(distinct[:_LINEAGE_VALUES_REPORTED]),
        row_source_id_unique_count=unique_count,
        null_or_blank_source_id_row_count=int(null_or_blank),
        evidence_id_matches_row_source_id=bool(matches),
        verified=not problems and bool(matches),
        problems=tuple(problems),
    )


@dataclass(frozen=True)
class CanonicalProductVerification:
    """The three independent OBSERVED checks whose conjunction is CPT_CANONICAL_PROFILE_V1 product
    identity: marker metadata valid, full required V1 column contract present, evidence lineage
    consistent. The structural-lookalike observation is reported beside them but establishes
    nothing."""

    marker: CanonicalProductMarker
    structural_columns_missing: tuple[str, ...]
    required_columns_missing: tuple[str, ...]
    lineage: CanonicalLineage

    @property
    def structural_columns_present(self) -> bool:
        return not self.structural_columns_missing

    @property
    def required_columns_present(self) -> bool:
        return not self.required_columns_missing

    @property
    def verified(self) -> bool:
        return bool(
            self.marker.verified and self.required_columns_present and self.lineage.verified
        )

    @property
    def problems(self) -> tuple[str, ...]:
        out = list(self.marker.problems)
        if self.required_columns_missing:
            out.append(
                f"required {contract.CPT_CANONICAL_PROFILE_CONTRACT} product column(s) absent "
                f"from schema: {list(self.required_columns_missing)}"
            )
        out.extend(f"evidence lineage: {p}" for p in self.lineage.problems)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker_verified": self.marker.verified,
            "structural_canonical_columns_present": self.structural_columns_present,
            "structural_canonical_columns_missing": list(self.structural_columns_missing),
            "canonical_product_required_columns_present": self.required_columns_present,
            "canonical_product_columns_missing": list(self.required_columns_missing),
            "lineage": self.lineage.to_dict(),
            "verified": self.verified,
            "problems": list(self.problems),
        }


def verify_canonical_cpt_product(
    measurements: pd.DataFrame, marker: CanonicalProductMarker
) -> CanonicalProductVerification:
    """Verify an already-read frame against an already-read marker. Shared by the project adapter
    (registered bytes) and the provider build (written bytes read back) so there is one reader-side
    authority, consuming the same `CANONICAL_PRODUCT_COLUMNS` the writer enforces."""

    columns = list(measurements.columns)
    return CanonicalProductVerification(
        marker=marker,
        structural_columns_missing=tuple(structural_columns_missing(columns)),
        required_columns_missing=tuple(canonical_product_columns_missing(columns)),
        lineage=verify_canonical_lineage(measurements, evidence_id=marker.evidence_id),
    )

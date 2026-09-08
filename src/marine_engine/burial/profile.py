"""Canonical burial profile, current burial state, and exposure evidence (MAR-024 Sections 7-9).

Zero dependency on any specific project or dataset -- every source-column
name is a caller-supplied parameter, never hardcoded here. Deliberately
does NOT interpolate over unsurveyed gaps merely to produce a continuous-
looking profile (Section 7): the profile has exactly one row per real
source record, never a fabricated in-between one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from marine_engine.burial.route import project_measurements_to_chainage
from marine_engine.burial.semantics import SOURCE_BURIAL_REFERENCE_UNRESOLVED

# --- Section 7: canonical profile ------------------------------------------------------------

EVIDENCE_TYPE_OPERATOR_MEASURED_BURIAL_PROFILE = "OPERATOR_MEASURED_BURIAL_PROFILE"

CANONICAL_BURIAL_PROFILE_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "source_record_id",
    "chainage_m",
    "source_kp",
    "x_m",
    "y_m",
    "measured_burial_value_m",
    "burial_reference_type",
    "burial_sign_convention",
    "survey_epoch",
    "measurement_method",
    "source_uncertainty_m",
    "evidence_type",
    "current_burial_state",
    "source_interpreted_exposure",
    "qa_flags",
)

# --- Section 8: current burial state vocabulary -- never SAFE/UNSAFE/HIGH RISK/LOW RISK -----

MEASURED_BURIED = "MEASURED_BURIED"
MEASURED_AT_SEABED_LEVEL = "MEASURED_AT_SEABED_LEVEL"
SOURCE_INTERPRETED_EXPOSED = "SOURCE_INTERPRETED_EXPOSED"
MEASURED_REFERENCE_REQUIRES_REVIEW = "MEASURED_REFERENCE_REQUIRES_REVIEW"
NO_MEASUREMENT = "NO_MEASUREMENT"

CURRENT_BURIAL_STATES = frozenset(
    {
        MEASURED_BURIED,
        MEASURED_AT_SEABED_LEVEL,
        SOURCE_INTERPRETED_EXPOSED,
        MEASURED_REFERENCE_REQUIRES_REVIEW,
        NO_MEASUREMENT,
    }
)

# A record is only ever "at seabed level" (rather than "buried") within this small tolerance of
# a resolved-reference zero -- never a magic inline number.
AT_SEABED_LEVEL_TOLERANCE_M = 0.05


def classify_current_burial_state(
    measured_value_m: float | None,
    *,
    burial_reference_type: str,
    is_source_interpreted_exposed: bool,
) -> str:
    """One record's current burial state (Section 8). `SOURCE_INTERPRETED_EXPOSED` is
    reachable ONLY via explicit source-stated evidence (`is_source_interpreted_exposed`) --
    never inferred from `measured_value_m`'s sign or magnitude, resolved reference or not
    (Section 8: "Do NOT infer exposure merely from an unclear source convention" -- applied
    here to any numeric convention, not only unclear ones, since a numeric value alone is
    never source-stated exposure evidence)."""

    if measured_value_m is None or (
        isinstance(measured_value_m, float) and pd.isna(measured_value_m)
    ):
        return NO_MEASUREMENT
    if is_source_interpreted_exposed:
        return SOURCE_INTERPRETED_EXPOSED
    if burial_reference_type == SOURCE_BURIAL_REFERENCE_UNRESOLVED:
        return MEASURED_REFERENCE_REQUIRES_REVIEW
    if measured_value_m <= AT_SEABED_LEVEL_TOLERANCE_M:
        return MEASURED_AT_SEABED_LEVEL
    return MEASURED_BURIED


def build_canonical_burial_profile(
    *,
    records_df: pd.DataFrame,
    asset_id: str,
    route: LineString,
    x_column: str,
    y_column: str,
    measured_value_column: str,
    source_kp_column: str,
    record_id_column: str,
    burial_reference_type: str,
    burial_sign_convention: str | None,
    survey_epoch: str,
    measurement_method: str | None,
    source_uncertainty_column: str | None = None,
    exposure_flag_column: str | None = None,
    qa_flags_by_index: dict[Any, list[str]] | None = None,
) -> pd.DataFrame:
    """Section 7: one row per real source record, chainage derived by projecting each
    record's real (x, y) onto the real canonical route (never interpolated, never a
    fabricated in-between record for an unsurveyed gap)."""

    if records_df.empty:
        return pd.DataFrame(columns=list(CANONICAL_BURIAL_PROFILE_COLUMNS))

    x = records_df[x_column].to_numpy(dtype=float)
    y = records_df[y_column].to_numpy(dtype=float)
    chainage_m = project_measurements_to_chainage(route, x, y)

    is_exposed = (
        records_df[exposure_flag_column].to_numpy(dtype=bool)
        if exposure_flag_column is not None
        else np.zeros(len(records_df), dtype=bool)
    )
    measured_value = records_df[measured_value_column].to_numpy(dtype=float)

    current_state = [
        classify_current_burial_state(
            None if pd.isna(v) else float(v),
            burial_reference_type=burial_reference_type,
            is_source_interpreted_exposed=bool(exposed),
        )
        for v, exposed in zip(measured_value, is_exposed, strict=True)
    ]

    qa_flags_by_index = qa_flags_by_index or {}
    qa_flags = ["; ".join(qa_flags_by_index.get(idx, [])) or None for idx in records_df.index]

    result = pd.DataFrame(
        {
            "asset_id": asset_id,
            "source_record_id": records_df[record_id_column].astype(str).to_numpy(),
            "chainage_m": chainage_m,
            "source_kp": records_df[source_kp_column].to_numpy(dtype=float),
            "x_m": x,
            "y_m": y,
            "measured_burial_value_m": measured_value,
            "burial_reference_type": burial_reference_type,
            "burial_sign_convention": burial_sign_convention,
            "survey_epoch": survey_epoch,
            "measurement_method": measurement_method,
            "source_uncertainty_m": (
                records_df[source_uncertainty_column].to_numpy(dtype=float)
                if source_uncertainty_column is not None
                else np.full(len(records_df), np.nan)
            ),
            "evidence_type": EVIDENCE_TYPE_OPERATOR_MEASURED_BURIAL_PROFILE,
            "current_burial_state": current_state,
            "source_interpreted_exposure": is_exposed,
            "qa_flags": qa_flags,
        }
    )
    return result.sort_values("chainage_m").reset_index(drop=True)[
        list(CANONICAL_BURIAL_PROFILE_COLUMNS)
    ]


def extract_source_interpreted_exposure(profile_df: pd.DataFrame) -> pd.DataFrame:
    """Section 9: the subset of the canonical profile the SOURCE itself explicitly flagged as
    exposed -- preserved as its own independent product (GIS layer, map overlay), never
    merged back into a generic "burial measurements" layer that would obscure which rows are
    a plain measurement versus explicit source-stated exposure evidence."""

    if profile_df.empty or "source_interpreted_exposure" not in profile_df.columns:
        return profile_df.iloc[0:0]
    return profile_df[profile_df["source_interpreted_exposure"]].copy()


def compute_profile_statistics(profile_df: pd.DataFrame) -> dict[str, Any]:
    """Section 25's required profile stats -- descriptive only."""

    if profile_df.empty:
        return {
            "measured_record_count": 0,
            "burial_min_m": None,
            "burial_median_m": None,
            "burial_p95_m": None,
            "burial_max_m": None,
            "zero_or_negative_record_count": 0,
            "current_burial_state_counts": {},
            "explicit_exposure_feature_count": 0,
        }

    values = profile_df["measured_burial_value_m"].dropna()
    state_counts = {
        str(k): int(v) for k, v in profile_df["current_burial_state"].value_counts().items()
    }
    return {
        "measured_record_count": int(len(profile_df)),
        "burial_min_m": float(values.min()) if len(values) else None,
        "burial_median_m": float(values.median()) if len(values) else None,
        "burial_p95_m": float(values.quantile(0.95)) if len(values) else None,
        "burial_max_m": float(values.max()) if len(values) else None,
        "zero_or_negative_record_count": int((values <= 0).sum()),
        "current_burial_state_counts": state_counts,
        "explicit_exposure_feature_count": int(profile_df["source_interpreted_exposure"].sum()),
    }

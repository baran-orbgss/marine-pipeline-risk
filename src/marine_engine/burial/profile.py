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

from marine_engine.burial.cover import (
    compute_cover_above_asset_m,
    normalize_canonical_reference_burial_depth_m,
)
from marine_engine.burial.route import project_measurements_to_chainage

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
    "sign_convention",
    "source_sign_convention_text",
    "reference_to_asset_top_offset_m",
    "canonical_reference_burial_depth_m",
    "cover_above_asset_m",
    "survey_epoch",
    "measurement_method",
    "source_uncertainty_m",
    "evidence_type",
    "measured_burial_state",
    "source_interpreted_exposure",
    "qa_flags",
)

# --- Section 8: measured burial state vocabulary -- never SAFE/UNSAFE/HIGH RISK/LOW RISK -----

MEASURED_BURIED = "MEASURED_BURIED"
MEASURED_AT_SEABED_LEVEL = "MEASURED_AT_SEABED_LEVEL"
MEASURED_ABOVE_SEABED = "MEASURED_ABOVE_SEABED"
SOURCE_INTERPRETED_EXPOSED = "SOURCE_INTERPRETED_EXPOSED"
MEASURED_REFERENCE_REQUIRES_REVIEW = "MEASURED_REFERENCE_REQUIRES_REVIEW"
NO_MEASUREMENT = "NO_MEASUREMENT"

MEASURED_BURIAL_STATES = frozenset(
    {
        MEASURED_BURIED,
        MEASURED_AT_SEABED_LEVEL,
        MEASURED_ABOVE_SEABED,
        SOURCE_INTERPRETED_EXPOSED,
        MEASURED_REFERENCE_REQUIRES_REVIEW,
        NO_MEASUREMENT,
    }
)

# A record is only ever "at seabed level" (rather than "buried"/"above seabed") within this
# small tolerance of a resolved-cover zero -- never a magic inline number.
AT_SEABED_LEVEL_TOLERANCE_M = 0.05


def classify_current_burial_state(
    cover_above_asset_m: float | None,
    *,
    has_measurement: bool,
    is_source_interpreted_exposed: bool,
) -> str:
    """One record's measured burial state (Section 8, repaired by MAR-024A Section 7).
    `cover_above_asset_m` -- the canonical, sign-and-reference-normalized quantity from
    `burial.cover` -- is the ONLY numeric input this function accepts; there is structurally
    no way for a raw source measurement to bypass normalization here (MAR-024A Section 6).
    `SOURCE_INTERPRETED_EXPOSED` is reachable ONLY via explicit source-stated evidence
    (`is_source_interpreted_exposed`) -- never inferred from cover's sign or magnitude, and it
    takes priority over a resolved cover so the two concepts never collapse into one (MAR-024A
    Section 8: the source may say "Exposure" even when the numeric reference is unresolved)."""

    if not has_measurement:
        return NO_MEASUREMENT
    if is_source_interpreted_exposed:
        return SOURCE_INTERPRETED_EXPOSED
    if cover_above_asset_m is None or (
        isinstance(cover_above_asset_m, float) and pd.isna(cover_above_asset_m)
    ):
        return MEASURED_REFERENCE_REQUIRES_REVIEW
    if cover_above_asset_m > AT_SEABED_LEVEL_TOLERANCE_M:
        return MEASURED_BURIED
    if cover_above_asset_m < -AT_SEABED_LEVEL_TOLERANCE_M:
        return MEASURED_ABOVE_SEABED
    return MEASURED_AT_SEABED_LEVEL


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
    sign_convention: str,
    survey_epoch: str,
    measurement_method: str | None,
    source_sign_convention_text: str | None = None,
    reference_to_asset_top_offset_m: float | None = None,
    source_uncertainty_column: str | None = None,
    exposure_flag_column: str | None = None,
    qa_flags_by_index: dict[Any, list[str]] | None = None,
) -> pd.DataFrame:
    """Section 7: one row per real source record, chainage derived by projecting each
    record's real (x, y) onto the real canonical route (never interpolated, never a
    fabricated in-between record for an unsurveyed gap).

    `sign_convention` and `burial_reference_type` (MAR-024A) are normalized into
    `canonical_reference_burial_depth_m` and `cover_above_asset_m` exactly once per record via
    `burial.cover`; `measured_burial_state` is classified from `cover_above_asset_m` alone,
    never from the raw `measured_burial_value_m` (MAR-024A Section 6)."""

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

    canonical_depth_list: list[float | None] = []
    cover_list: list[float | None] = []
    for v in measured_value:
        raw = None if pd.isna(v) else float(v)
        canonical_depth = normalize_canonical_reference_burial_depth_m(
            raw, sign_convention=sign_convention
        )
        canonical_depth_list.append(canonical_depth)
        cover_list.append(
            compute_cover_above_asset_m(
                canonical_depth,
                burial_reference_type=burial_reference_type,
                reference_to_asset_top_offset_m=reference_to_asset_top_offset_m,
            )
        )

    current_state = [
        classify_current_burial_state(
            cover_val,
            has_measurement=not pd.isna(v),
            is_source_interpreted_exposed=bool(exposed),
        )
        for v, cover_val, exposed in zip(measured_value, cover_list, is_exposed, strict=True)
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
            "sign_convention": sign_convention,
            "source_sign_convention_text": source_sign_convention_text,
            "reference_to_asset_top_offset_m": reference_to_asset_top_offset_m,
            "canonical_reference_burial_depth_m": canonical_depth_list,
            "cover_above_asset_m": cover_list,
            "survey_epoch": survey_epoch,
            "measurement_method": measurement_method,
            "source_uncertainty_m": (
                records_df[source_uncertainty_column].to_numpy(dtype=float)
                if source_uncertainty_column is not None
                else np.full(len(records_df), np.nan)
            ),
            "evidence_type": EVIDENCE_TYPE_OPERATOR_MEASURED_BURIAL_PROFILE,
            "measured_burial_state": current_state,
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
            "measured_burial_state_counts": {},
            "explicit_exposure_feature_count": 0,
            "canonical_reference_burial_depth_available_count": 0,
            "cover_above_asset_available_count": 0,
        }

    values = profile_df["measured_burial_value_m"].dropna()
    state_counts = {
        str(k): int(v) for k, v in profile_df["measured_burial_state"].value_counts().items()
    }
    return {
        "measured_record_count": int(len(profile_df)),
        "burial_min_m": float(values.min()) if len(values) else None,
        "burial_median_m": float(values.median()) if len(values) else None,
        "burial_p95_m": float(values.quantile(0.95)) if len(values) else None,
        "burial_max_m": float(values.max()) if len(values) else None,
        "zero_or_negative_record_count": int((values <= 0).sum()),
        "measured_burial_state_counts": state_counts,
        "explicit_exposure_feature_count": int(profile_df["source_interpreted_exposure"].sum()),
        "canonical_reference_burial_depth_available_count": int(
            profile_df["canonical_reference_burial_depth_m"].notna().sum()
        ),
        "cover_above_asset_available_count": int(profile_df["cover_above_asset_m"].notna().sum()),
    }

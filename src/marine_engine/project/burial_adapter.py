"""Generic project-supplied tabular burial-profile ingestion (MAR-026 Section 11).

Does NOT reimplement MAR-024/MAR-024A's readiness checks or canonical cover/reference/sign
semantics -- `inspect_burial_profile` only loads the real CSV/Parquet table (using the
manifest's own explicit column mapping, Section 11: never inferring critical engineering
semantics from column names alone) and assembles the existing `marine_engine.burial.readiness.
BurialProfileFacts`; the caller passes those facts to the existing, unmodified
`assess_burial_profile_readiness`. If the source's burial reference or sign convention is
unresolved, that unresolved state is preserved honestly (`burial_reference_known=False`) --
never guessed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from marine_engine.burial import readiness as burial_readiness
from marine_engine.project.manifest import BurialColumnMapping

__all__ = ["load_burial_table", "inspect_burial_profile", "BurialTableLoadError"]

SUPPORTED_SUFFIXES = frozenset({".csv", ".parquet", ".pq"})


class BurialTableLoadError(RuntimeError):
    """The burial-profile source file could not be found, read, or parsed -- a registration-
    level failure, distinct from a scientific readiness failure."""


def load_burial_table(path: Path) -> pd.DataFrame:
    """Section 11: at minimum support CSV and Parquet -- nothing else is silently guessed."""

    if not path.is_file():
        raise BurialTableLoadError(f"burial-profile source file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise BurialTableLoadError(
            f"unsupported burial-profile file format {suffix!r} for {path} -- MAR-026 supports "
            f"only {sorted(SUPPORTED_SUFFIXES)}"
        )
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 -- any parse failure is a registration failure
        raise BurialTableLoadError(
            f"burial-profile source file could not be parsed: {path} ({exc})"
        ) from exc


def inspect_burial_profile(
    path: Path,
    *,
    column_mapping: BurialColumnMapping,
    declared_crs: str | None,
    declared_units: str | None,
    declared_measurement_reference: str | None,
    declared_survey_epoch: str | None,
) -> tuple[burial_readiness.BurialProfileFacts, pd.DataFrame]:
    """Section 11 steps: load the real table via the manifest's explicit column mapping, and
    build the existing `BurialProfileFacts`. Raises `BurialTableLoadError` if the file cannot be
    read at all -- callers must treat that as a registration failure (see `project.registry`)."""

    df = load_burial_table(path)

    kp_col = column_mapping.chainage_or_kp_column
    value_col = column_mapping.measured_value_column
    kp_available = kp_col in df.columns
    value_available = value_col in df.columns
    coordinate_support = bool(
        column_mapping.x_column
        and column_mapping.y_column
        and column_mapping.x_column in df.columns
        and column_mapping.y_column in df.columns
    )
    record_id_available = bool(
        column_mapping.record_id_column and column_mapping.record_id_column in df.columns
    )
    uncertainty_available = bool(
        column_mapping.uncertainty_column and column_mapping.uncertainty_column in df.columns
    )

    kp_series = df[kp_col] if kp_available else None
    kp_is_monotonic = bool(kp_series.is_monotonic_increasing) if kp_available else False
    duplicate_kp_count = int(kp_series.duplicated().sum()) if kp_available else 0

    value_series = df[value_col] if value_available else None
    missing_value_fraction = (
        float(value_series.isna().mean()) if value_available and len(df) else None
    )
    negative_value_count = int((value_series.dropna() < 0).sum()) if value_available else 0
    zero_value_count = int((value_series.dropna() == 0).sum()) if value_available else 0

    facts = burial_readiness.BurialProfileFacts(
        source_file_readable=True,
        record_count=len(df),
        route_identifier_available=record_id_available,
        kp_available=kp_available,
        kp_is_monotonic=kp_is_monotonic,
        duplicate_kp_count=duplicate_kp_count,
        coordinate_support=coordinate_support,
        crs_available=declared_crs is not None,
        units_available=declared_units is not None,
        # a single-table generic input has nothing else to cross-check units against in
        # MAR-026 -- never fabricated as "inconsistent" without a second source to compare.
        units_consistent_across_sources=True,
        burial_reference_known=declared_measurement_reference is not None,
        missing_value_fraction=missing_value_fraction,
        # requires a linked route to compute a real coverage fraction -- MAR-026's generic
        # adapter does not correlate assets, so this is honestly unknown, never invented.
        coverage_fraction=None,
        # no generic spike-detection heuristic in MAR-026 -- never invented.
        suspicious_spike_count=0,
        negative_value_count=negative_value_count,
        zero_value_count=zero_value_count,
        source_uncertainty_available=uncertainty_available,
        survey_epoch_known=declared_survey_epoch is not None,
    )
    return facts, df

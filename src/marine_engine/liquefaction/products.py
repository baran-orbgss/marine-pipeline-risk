"""GIS-ready product construction for earthquake CPT liquefaction triggering (MAR-033 Sections
21-23, 37).

Two products, kept structurally distinct:

* the full row-level triggering profile (`earthquake_triggering.evaluate_triggering_profile`'s
  own output) -- the auditable evidence-role-adjacent product;
* a `POINT_ANALYSIS` summary, one row per CPT test, that a GIS layer can render directly. This is
  explicitly NOT `POINT_EVIDENCE` (Section 22): it is a derived visualization/indexing product
  over the profile, never itself measured evidence, never an interpolated area hazard surface
  (Section 23 -- no spatial interpolation, no Voronoi zones, no area-wide coverage claim).
"""

from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
import pandas as pd

from marine_engine.liquefaction import contract

__all__ = [
    "POINT_ANALYSIS_COLUMNS",
    "SPATIAL_SUPPORT_STATEMENT",
    "build_point_summary_df",
    "build_point_analysis_gdf",
    "profile_df_for_parquet",
]

SPATIAL_SUPPORT_STATEMENT = (
    "POINT_ANALYSIS: a model factor-of-safety screening summary at CPT test locations only. Not "
    "an interpolated area hazard surface; no Voronoi zone, raster, or area-wide coverage claim is "
    "made or implied."
)

POINT_ANALYSIS_COLUMNS: tuple[str, ...] = (
    "test_id",
    "scenario_id",
    "minimum_model_fs_liq",
    "depth_at_minimum_model_fs_liq_m",
    "evaluated_row_count",
    "not_evaluable_row_count",
    "deep_rd_limitation_present",
)

_EVALUATED_STATES = (contract.MODEL_FS_BELOW_1, contract.MODEL_FS_AT_1, contract.MODEL_FS_ABOVE_1)


def build_point_summary_df(profile_df: pd.DataFrame) -> pd.DataFrame:
    """Section 22: one row per (test_id, scenario_id), summarizing -- never interpolating -- its
    own profile rows. `minimum_model_fs_liq` is null when a test has no evaluable row rather than
    a fabricated value."""

    rows: list[dict[str, Any]] = []
    for (test_id, scenario_id), group in profile_df.groupby(["test_id", "scenario_id"], sort=True):
        evaluated = group[group["evaluation_state"].isin(_EVALUATED_STATES)]
        not_evaluable = group[group["evaluation_state"] == contract.NOT_EVALUABLE]
        deep_limitation = bool(
            group["limitations"]
            .apply(lambda reasons: contract.RD_DEEP_EXTRAPOLATION_LIMITATION in reasons)
            .any()
        )
        if evaluated.empty:
            min_fs = None
            depth_at_min = None
        else:
            idx = evaluated["FS_liq"].idxmin()
            min_fs = float(evaluated.loc[idx, "FS_liq"])
            depth_at_min = float(evaluated.loc[idx, "depth_bsf_m"])
        rows.append(
            {
                "test_id": test_id,
                "scenario_id": scenario_id,
                "minimum_model_fs_liq": min_fs,
                "depth_at_minimum_model_fs_liq_m": depth_at_min,
                "evaluated_row_count": int(len(evaluated)),
                "not_evaluable_row_count": int(len(not_evaluable)),
                "deep_rd_limitation_present": deep_limitation,
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(POINT_ANALYSIS_COLUMNS))
    return pd.DataFrame(rows, columns=list(POINT_ANALYSIS_COLUMNS))


def build_point_analysis_gdf(
    point_summary_df: pd.DataFrame, locations_gdf: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Join the point summary onto CPT test locations (Section 22). Only a test with a resolvable
    location becomes a GIS point; a test without one is reported as a finding, never silently
    dropped."""

    geometry_col = locations_gdf.geometry.name
    merged = point_summary_df.merge(
        locations_gdf[["test_id", geometry_col]], on="test_id", how="left"
    )
    missing_mask = merged[geometry_col].isna()
    findings: list[str] = []
    if missing_mask.any():
        missing_ids = sorted(merged.loc[missing_mask, "test_id"].unique().tolist())
        findings.append(
            f"{int(missing_mask.sum())} test(s) have a point-analysis summary but no resolvable "
            f"CPT location: {missing_ids}"
        )
    matched = merged.loc[~missing_mask].copy()
    gdf = gpd.GeoDataFrame(matched, geometry=geometry_col, crs=locations_gdf.crs)
    return gdf, findings


def profile_df_for_parquet(profile_df: pd.DataFrame) -> pd.DataFrame:
    """`limitations` becomes a JSON-encoded string column, matching the repository's established
    convention for list-valued fields in a flat parquet product (e.g.
    `project.model.build_asset_linkage_df`'s `route_linkage_findings`)."""

    out = profile_df.copy()
    out["limitations"] = out["limitations"].apply(lambda reasons: json.dumps(list(reasons)))
    return out

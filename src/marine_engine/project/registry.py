"""Per-asset registration orchestration: identity, provenance-vs-observed conflict detection,
and readiness delegation (MAR-026 Sections 5, 8, 12, 13).

Two-tier status, kept structurally distinct:

- REGISTRATION status (`REGISTERED` / `REGISTRATION_FAILED`) -- can the source file even be
  found, hashed, and opened/parsed at all. A totally missing/corrupt file never reaches the
  scientific readiness checks (which, for the reused `terrain.readiness`/`burial.readiness`
  modules, structurally assume a successfully-opened source).
- READINESS status -- delegated to the existing accepted readiness module for the three
  implemented categories (`READY`/`READY_WITH_LIMITATIONS`/`NOT_READY`), or the explicit
  `REGISTERED_READINESS_NOT_IMPLEMENTED` for every other category. A future category is NEVER
  reported READY merely because its file exists (Section 7).

This module never computes an aggregate project-wide hazard-readiness score -- see
`PROJECT_HAZARD_READINESS_DISCLAIMER` (Section 13).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError

from marine_engine.burial import readiness as burial_readiness
from marine_engine.project import bathymetry_adapter, burial_adapter, route_adapter
from marine_engine.project.categories import (
    BATHYMETRY_RASTER,
    BURIAL_PROFILE,
    PIPELINE_ROUTE,
    REGISTERED_READINESS_NOT_IMPLEMENTED,
)
from marine_engine.project.identity import compute_file_identity
from marine_engine.project.manifest import AssetEntry, ProjectManifest, resolve_asset_path
from marine_engine.terrain import readiness as terrain_readiness

__all__ = [
    "REGISTERED",
    "REGISTRATION_FAILED",
    "PROJECT_HAZARD_READINESS_DISCLAIMER",
    "AssetRegistration",
    "AssetRegistrationResult",
    "ProjectRegistrationSummary",
    "register_asset",
    "register_project",
    "build_asset_registry_df",
    "build_project_readiness_dict",
]

REGISTERED = "REGISTERED"
REGISTRATION_FAILED = "REGISTRATION_FAILED"

# Section 13: a project is not "READY FOR MARINE ANALYSIS" because one dataset passed QA.
PROJECT_HAZARD_READINESS_DISCLAIMER = (
    "PROJECT REGISTRATION STATUS DESCRIBES STRUCTURAL REGISTRATION AND PER-ASSET READINESS "
    "ONLY. IT DOES NOT IMPLY THE PROJECT CONTAINS SUFFICIENT DATA FOR SCOUR, FREE-SPAN, "
    "LIQUEFACTION, SHALLOW GAS, OR ANY OTHER MARINE GEOHAZARD ANALYSIS."
)


def _crs_conflict(declared: str | None, observed: str | None) -> str | None:
    """Section 5: a material declared-vs-observed CRS conflict is recorded explicitly, never
    silently resolved by picking one or reprojecting."""

    if not declared or not observed:
        return None
    try:
        declared_crs = CRS.from_user_input(declared)
        observed_crs = CRS.from_user_input(observed)
    except CRSError:
        return None
    if declared_crs != observed_crs:
        return f"declared CRS {declared!r} does not match observed/embedded CRS {observed!r}"
    return None


def _crs_kind_conflict(declared: str | None, *, observed_is_geographic: bool | None) -> str | None:
    """A coarser conflict check usable even when the observed side is only known as
    geographic-vs-projected (e.g. a raster's embedded CRS, not exposed as a full string by
    `RasterFacts`) -- still a real, honestly-detectable conflict, never fabricated."""

    if not declared or observed_is_geographic is None:
        return None
    try:
        declared_crs = CRS.from_user_input(declared)
    except CRSError:
        return None
    if declared_crs.is_geographic != observed_is_geographic:
        return (
            f"declared CRS {declared!r} is_geographic={declared_crs.is_geographic} but observed "
            f"source CRS is_geographic={observed_is_geographic}"
        )
    return None


@dataclass(frozen=True)
class AssetRegistration:
    asset_id: str
    category: str
    evidence_role: str
    resolved_path: str
    filename: str
    byte_size: int | None
    sha256: str | None
    registration_status: str
    registration_detail: str
    provenance_declared: dict[str, Any]
    observed_facts: dict[str, Any]
    conflicts: list[str]
    readiness_status: str
    readiness_result: dict[str, Any] | None


@dataclass(frozen=True)
class AssetRegistrationResult:
    registration: AssetRegistration
    canonical_route_gdf: gpd.GeoDataFrame | None = None
    burial_table_df: pd.DataFrame | None = None


def _failed_registration(
    asset: AssetEntry,
    *,
    resolved_path: Path,
    byte_size: int | None,
    sha256: str | None,
    detail: str,
    provenance_declared: dict[str, Any],
) -> AssetRegistrationResult:
    return AssetRegistrationResult(
        registration=AssetRegistration(
            asset_id=asset.asset_id,
            category=asset.category,
            evidence_role=asset.evidence_role,
            resolved_path=str(resolved_path),
            filename=resolved_path.name,
            byte_size=byte_size,
            sha256=sha256,
            registration_status=REGISTRATION_FAILED,
            registration_detail=detail,
            provenance_declared=provenance_declared,
            observed_facts={},
            conflicts=[],
            readiness_status=terrain_readiness.NOT_READY,
            readiness_result=None,
        )
    )


def _register_route(
    asset: AssetEntry,
    resolved_path: Path,
    byte_size: int,
    sha256: str,
    provenance_declared: dict[str, Any],
    working_crs: str,
) -> AssetRegistrationResult:
    facts, gdf, canonical_line, source_crs = route_adapter.inspect_route_source(
        resolved_path,
        layer=asset.layer,
        working_crs=working_crs,
        declared_survey_epoch=asset.provenance.survey_epoch,
    )
    result = route_adapter.assess_route_readiness(facts)

    conflicts = []
    conflict = _crs_conflict(asset.provenance.horizontal_crs_declared, source_crs)
    if conflict:
        conflicts.append(conflict)

    observed_facts = {
        "geometry_types": list(facts.geometry_types),
        "crs_is_present": facts.crs_is_present,
        "crs_is_geographic": facts.crs_is_geographic,
        "observed_crs": source_crs,
        "is_single_continuous_line": facts.is_single_continuous_line,
        "disconnected_part_count": facts.disconnected_part_count,
        "source_geometry_length": facts.source_geometry_length,
        "feature_count": int(len(gdf)) if gdf is not None else None,
    }

    canonical_route_gdf = None
    if canonical_line is not None and source_crs is not None:
        canonical_route_gdf = route_adapter.build_canonical_project_route(
            canonical_line,
            asset_id=asset.asset_id,
            source_route_name=resolved_path.name,
            source_crs=source_crs,
            working_crs=working_crs,
        )

    registration = AssetRegistration(
        asset_id=asset.asset_id,
        category=asset.category,
        evidence_role=asset.evidence_role,
        resolved_path=str(resolved_path),
        filename=resolved_path.name,
        byte_size=byte_size,
        sha256=sha256,
        registration_status=REGISTERED,
        registration_detail="route source opened and inspected",
        provenance_declared=provenance_declared,
        observed_facts=observed_facts,
        conflicts=conflicts,
        readiness_status=result.status,
        readiness_result=result.to_dict(),
    )
    return AssetRegistrationResult(
        registration=registration, canonical_route_gdf=canonical_route_gdf
    )


def _register_bathymetry(
    asset: AssetEntry,
    resolved_path: Path,
    byte_size: int,
    sha256: str,
    provenance_declared: dict[str, Any],
) -> AssetRegistrationResult:
    try:
        facts = bathymetry_adapter.inspect_bathymetry_raster(
            resolved_path,
            declared_vertical_datum=asset.provenance.vertical_datum_declared,
            declared_survey_epoch=asset.provenance.survey_epoch,
        )
    except bathymetry_adapter.RasterOpenError as exc:
        return _failed_registration(
            asset,
            resolved_path=resolved_path,
            byte_size=byte_size,
            sha256=sha256,
            detail=str(exc),
            provenance_declared=provenance_declared,
        )

    result = terrain_readiness.assess_bathymetry_readiness(facts)

    conflicts = []
    conflict = _crs_kind_conflict(
        asset.provenance.horizontal_crs_declared, observed_is_geographic=facts.crs_is_geographic
    )
    if conflict:
        conflicts.append(conflict)

    observed_facts = {
        "band_count": facts.band_count,
        "dtype": facts.dtype,
        "width": facts.width,
        "height": facts.height,
        "pixel_size_x_m": facts.pixel_size_x_m,
        "pixel_size_y_m": facts.pixel_size_y_m,
        "bounds": list(facts.bounds) if facts.bounds else None,
        "nodata_value": facts.nodata_value,
        "crs_is_present": facts.crs_is_present,
        "crs_is_geographic": facts.crs_is_geographic,
        "crs_linear_units": facts.crs_linear_units,
        "data_min": facts.data_min,
        "data_max": facts.data_max,
        "valid_cell_fraction": facts.valid_cell_fraction,
    }

    registration = AssetRegistration(
        asset_id=asset.asset_id,
        category=asset.category,
        evidence_role=asset.evidence_role,
        resolved_path=str(resolved_path),
        filename=resolved_path.name,
        byte_size=byte_size,
        sha256=sha256,
        registration_status=REGISTERED,
        registration_detail="raster opened and inspected",
        provenance_declared=provenance_declared,
        observed_facts=observed_facts,
        conflicts=conflicts,
        readiness_status=result.status,
        readiness_result=result.to_dict(),
    )
    return AssetRegistrationResult(registration=registration)


def _register_burial(
    asset: AssetEntry,
    resolved_path: Path,
    byte_size: int,
    sha256: str,
    provenance_declared: dict[str, Any],
) -> AssetRegistrationResult:
    if asset.burial_columns is None:
        return _failed_registration(
            asset,
            resolved_path=resolved_path,
            byte_size=byte_size,
            sha256=sha256,
            detail="BURIAL_PROFILE asset requires an explicit burial_columns mapping in the "
            "manifest -- critical engineering semantics are never inferred from column names "
            "alone (Section 11)",
            provenance_declared=provenance_declared,
        )

    try:
        facts, df = burial_adapter.inspect_burial_profile(
            resolved_path,
            column_mapping=asset.burial_columns,
            declared_crs=asset.provenance.horizontal_crs_declared,
            declared_units=asset.provenance.units_declared,
            declared_measurement_reference=asset.provenance.measurement_reference_declared,
            declared_survey_epoch=asset.provenance.survey_epoch,
        )
    except burial_adapter.BurialTableLoadError as exc:
        return _failed_registration(
            asset,
            resolved_path=resolved_path,
            byte_size=byte_size,
            sha256=sha256,
            detail=str(exc),
            provenance_declared=provenance_declared,
        )

    result = burial_readiness.assess_burial_profile_readiness(facts)
    observed_facts = {
        "record_count": facts.record_count,
        "columns": list(df.columns),
        "kp_available": facts.kp_available,
        "kp_is_monotonic": facts.kp_is_monotonic,
        "duplicate_kp_count": facts.duplicate_kp_count,
        "coordinate_support": facts.coordinate_support,
        "missing_value_fraction": facts.missing_value_fraction,
        "negative_value_count": facts.negative_value_count,
        "zero_value_count": facts.zero_value_count,
    }

    registration = AssetRegistration(
        asset_id=asset.asset_id,
        category=asset.category,
        evidence_role=asset.evidence_role,
        resolved_path=str(resolved_path),
        filename=resolved_path.name,
        byte_size=byte_size,
        sha256=sha256,
        registration_status=REGISTERED,
        registration_detail="burial table loaded and inspected",
        provenance_declared=provenance_declared,
        observed_facts=observed_facts,
        conflicts=[],
        readiness_status=result.status,
        readiness_result=result.to_dict(),
    )
    return AssetRegistrationResult(registration=registration, burial_table_df=df)


def register_asset(
    asset: AssetEntry, *, manifest_dir: Path, working_crs: str
) -> AssetRegistrationResult:
    """Section 8: compute content-based identity first (never mutating the source file), then
    dispatch to the category's adapter, or -- for a category with no implemented adapter --
    register it honestly as `REGISTERED_READINESS_NOT_IMPLEMENTED` (Section 7)."""

    resolved_path = resolve_asset_path(asset, manifest_dir)
    provenance_declared = asset.provenance.model_dump()

    try:
        identity = compute_file_identity(resolved_path)
    except (FileNotFoundError, OSError) as exc:
        return _failed_registration(
            asset,
            resolved_path=resolved_path,
            byte_size=None,
            sha256=None,
            detail=f"source file not found or unreadable: {exc}",
            provenance_declared=provenance_declared,
        )

    if asset.category == PIPELINE_ROUTE:
        return _register_route(
            asset,
            identity.resolved_path,
            identity.byte_size,
            identity.sha256,
            provenance_declared,
            working_crs,
        )
    if asset.category == BATHYMETRY_RASTER:
        return _register_bathymetry(
            asset, identity.resolved_path, identity.byte_size, identity.sha256, provenance_declared
        )
    if asset.category == BURIAL_PROFILE:
        return _register_burial(
            asset, identity.resolved_path, identity.byte_size, identity.sha256, provenance_declared
        )

    registration = AssetRegistration(
        asset_id=asset.asset_id,
        category=asset.category,
        evidence_role=asset.evidence_role,
        resolved_path=str(identity.resolved_path),
        filename=identity.filename,
        byte_size=identity.byte_size,
        sha256=identity.sha256,
        registration_status=REGISTERED,
        registration_detail=f"file registered; no readiness adapter implemented for category "
        f"{asset.category!r}",
        provenance_declared=provenance_declared,
        observed_facts={},
        conflicts=[],
        readiness_status=REGISTERED_READINESS_NOT_IMPLEMENTED,
        readiness_result=None,
    )
    return AssetRegistrationResult(registration=registration)


@dataclass(frozen=True)
class ProjectRegistrationSummary:
    project_id: str
    project_name: str
    working_crs: str
    primary_route_asset_id: str | None
    asset_results: list[AssetRegistrationResult] = field(default_factory=list)

    def readiness_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.asset_results:
            status = r.registration.readiness_status
            counts[status] = counts.get(status, 0) + 1
        return counts

    def registration_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.asset_results:
            status = r.registration.registration_status
            counts[status] = counts.get(status, 0) + 1
        return counts


def register_project(manifest: ProjectManifest, manifest_dir: Path) -> ProjectRegistrationSummary:
    working_crs = manifest.project.working_crs
    asset_results = [
        register_asset(asset, manifest_dir=manifest_dir, working_crs=working_crs)
        for asset in manifest.assets
    ]
    return ProjectRegistrationSummary(
        project_id=manifest.project.id,
        project_name=manifest.project.name,
        working_crs=working_crs,
        primary_route_asset_id=manifest.primary_route_asset_id,
        asset_results=asset_results,
    )


def build_asset_registry_df(summary: ProjectRegistrationSummary) -> pd.DataFrame:
    """Section 14: `project_asset_registry.parquet` -- one row per registered asset."""

    rows = []
    for r in summary.asset_results:
        reg = r.registration
        rows.append(
            {
                "asset_id": reg.asset_id,
                "category": reg.category,
                "evidence_role": reg.evidence_role,
                "resolved_path": reg.resolved_path,
                "filename": reg.filename,
                "byte_size": reg.byte_size,
                "sha256": reg.sha256,
                "registration_status": reg.registration_status,
                "registration_detail": reg.registration_detail,
                "provenance_declared_json": json.dumps(reg.provenance_declared, default=str),
                "observed_facts_json": json.dumps(reg.observed_facts, default=str),
                "conflicts": "; ".join(reg.conflicts) or None,
                "readiness_status": reg.readiness_status,
            }
        )
    columns = [
        "asset_id",
        "category",
        "evidence_role",
        "resolved_path",
        "filename",
        "byte_size",
        "sha256",
        "registration_status",
        "registration_detail",
        "provenance_declared_json",
        "observed_facts_json",
        "conflicts",
        "readiness_status",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_project_readiness_dict(summary: ProjectRegistrationSummary) -> dict[str, Any]:
    """Section 14: `project_readiness.json` -- structured per-asset results, blocking vs.
    limitation distinctions preserved exactly as the delegated readiness modules produced them."""

    return {
        "scientific_role": "PROJECT_STRUCTURAL_REGISTRATION_AND_PER_ASSET_READINESS",
        "project_id": summary.project_id,
        "project_hazard_readiness_disclaimer": PROJECT_HAZARD_READINESS_DISCLAIMER,
        "registration_status_counts": summary.registration_status_counts(),
        "readiness_status_counts": summary.readiness_status_counts(),
        "assets": {
            r.registration.asset_id: {
                "category": r.registration.category,
                "evidence_role": r.registration.evidence_role,
                "registration_status": r.registration.registration_status,
                "registration_detail": r.registration.registration_detail,
                "conflicts": r.registration.conflicts,
                "readiness_status": r.registration.readiness_status,
                "readiness_result": r.registration.readiness_result,
            }
            for r in summary.asset_results
        },
    }

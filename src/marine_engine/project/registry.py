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

MAR-026A adds a THIRD distinction on top of readiness: intrinsic vs. effective. The delegated
readiness modules (`terrain.readiness.assess_bathymetry_readiness`,
`burial.readiness.assess_burial_profile_readiness`) and the new `route_adapter.
assess_route_readiness` each answer one question in isolation -- is *this file's own data*
usable. That is the INTRINSIC result, and it is never mutated (`readiness_status_intrinsic`,
`readiness_result`). Project-level integration concerns -- e.g. a declared CRS that contradicts
the file's own observed/embedded CRS -- sit ABOVE that delegated result and can force an asset's
EFFECTIVE readiness (`readiness_status_effective`, and the backward-compatible
`readiness_status` alias) to `NOT_READY` even when the intrinsic result was `READY`. Both values
are always exposed side by side -- a material conflict is never resolved by silently falsifying
the delegated module's own conclusion.
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
from marine_engine.geotechnical import cpt_readiness
from marine_engine.project import bathymetry_adapter, burial_adapter, cpt_adapter, route_adapter
from marine_engine.project.categories import (
    BATHYMETRY_RASTER,
    BURIAL_PROFILE,
    CPT,
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
    """Section 5 / MAR-026A Problem A: a material declared-vs-observed CRS conflict is recorded
    explicitly, never silently resolved by picking one or reprojecting. Comparison is by
    `pyproj.CRS` equality, never raw string equality, so two different textual representations
    of the same CRS never produce a false conflict -- while two different projected CRSs (e.g.
    `EPSG:32631` vs `EPSG:32632`) are always caught, even though neither is geographic.

    A missing declared or observed value is NOT itself a conflict (there is nothing to compare
    against; a missing embedded CRS is its own, separately handled readiness concern). A
    declared value that fails to parse as a CRS at all is never silently ignored, though -- it is
    itself reported as a conflict against a known observed CRS, rather than treated as "no
    conflict found"."""

    if not declared or not observed:
        return None
    try:
        declared_crs = CRS.from_user_input(declared)
    except CRSError:
        return f"declared CRS {declared!r} is not a valid/recognized CRS identifier"
    try:
        observed_crs = CRS.from_user_input(observed)
    except CRSError:  # pragma: no cover -- defensive; observed values are always adapter-derived
        return f"observed CRS {observed!r} could not be parsed for comparison"
    if declared_crs != observed_crs:
        return f"declared CRS {declared!r} does not match observed/embedded CRS {observed!r}"
    return None


def _effective_readiness_status(intrinsic_status: str, conflicts: list[str]) -> str:
    """MAR-026A Section 3: an unresolved declared-vs-observed conflict is a BLOCKING
    project-integration condition -- the asset's EFFECTIVE readiness becomes `NOT_READY`
    regardless of what the delegated/intrinsic readiness module concluded on the file's data
    alone. The intrinsic result itself is never mutated or falsified to reach this; it stays
    available separately as `readiness_status_intrinsic`/`readiness_result`."""

    if conflicts:
        return terrain_readiness.NOT_READY
    return intrinsic_status


def _project_working_crs_findings(working_crs: str) -> list[str]:
    """MAR-026A Section 6: the project's declared `working_crs` must be a valid, projected,
    metric CRS. Checked centrally, once per project, rather than only incidentally through a
    route asset's own readiness checks (`route_adapter.assess_route_readiness` already has its
    own `working_crs_valid`/`working_crs_projected_metric` checks, but those only run when a
    route asset happens to be present) -- so a bathymetry-only or burial-only project can never
    carry an obviously invalid working CRS unnoticed. Never infers or falls back to a
    replacement CRS: an invalid working_crs is only ever reported, so the operator corrects the
    manifest."""

    findings: list[str] = []
    if not route_adapter.is_valid_crs(working_crs):
        findings.append(f"project working_crs {working_crs!r} is not a valid CRS identifier")
    elif not route_adapter.is_projected_metric_crs(working_crs):
        findings.append(
            f"project working_crs {working_crs!r} is not a projected, metric CRS -- required "
            "for pipeline engineering operations"
        )
    return findings


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
    readiness_status_intrinsic: str
    readiness_status_effective: str
    readiness_result: dict[str, Any] | None

    @property
    def readiness_status(self) -> str:
        """Backward-compatible alias (MAR-026A Section 9): equals `readiness_status_effective`,
        i.e. it already accounts for any project-level declared-vs-observed CRS conflict --
        never only the delegated/intrinsic result in isolation. Use
        `readiness_status_intrinsic` for the untouched, delegated module's own conclusion."""
        return self.readiness_status_effective


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
            readiness_status_intrinsic=terrain_readiness.NOT_READY,
            readiness_status_effective=terrain_readiness.NOT_READY,
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

    # MAR-026A Problem B: canonical route construction is gated structurally on every
    # prerequisite the reprojection itself needs -- a resolved line, a known source CRS, a
    # working CRS that is both valid and projected/metric -- plus (Section 3) no unresolved
    # declared-vs-observed CRS conflict. This is checked BEFORE `build_canonical_project_route`
    # is ever called, so an invalid `working_crs` (e.g. "NOT_A_REAL_CRS") or a non-metric one
    # (e.g. "EPSG:4326") can never reach `GeoDataFrame.to_crs(...)` and never needs a broad
    # try/except around the reprojection to stay safe.
    can_build_canonical_route = (
        canonical_line is not None
        and source_crs is not None
        and facts.working_crs_valid
        and facts.working_crs_is_projected_metric
        and not conflicts
    )
    canonical_route_gdf = None
    if can_build_canonical_route:
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
        readiness_status_intrinsic=result.status,
        readiness_status_effective=_effective_readiness_status(result.status, conflicts),
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
        facts, observed_crs = bathymetry_adapter.inspect_bathymetry_raster(
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

    # MAR-026A Problem A: compared by exact `pyproj.CRS` identity (`_crs_conflict`), not merely
    # geographic-vs-projected -- two different projected CRSs (e.g. EPSG:32631 vs EPSG:32632)
    # must conflict even though neither is geographic.
    conflicts = []
    conflict = _crs_conflict(asset.provenance.horizontal_crs_declared, observed_crs)
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
        "observed_crs": observed_crs,
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
        readiness_status_intrinsic=result.status,
        readiness_status_effective=_effective_readiness_status(result.status, conflicts),
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
        readiness_status_intrinsic=result.status,
        readiness_status_effective=_effective_readiness_status(result.status, []),
        readiness_result=result.to_dict(),
    )
    return AssetRegistrationResult(registration=registration, burial_table_df=df)


def _register_cpt(
    asset: AssetEntry,
    resolved_path: Path,
    byte_size: int,
    sha256: str,
    provenance_declared: dict[str, Any],
) -> AssetRegistrationResult:
    """MAR-032 Section 25: inspect the registered CPT file's real bytes (`cpt_adapter`) and
    DELEGATE to the generic `geotechnical.cpt_readiness.assess_cpt_readiness`. A documentary or
    unrecognised file is registered but BLOCKING on `DIGITAL_PROFILE`, so path existence alone
    can never yield READY. The declared evidence role is passed through untouched.

    MAR-032A: the SHA-256 this registry actually computed is handed to the adapter (no checksum
    Boolean is manufactured), and the declared evidence role is handed over ONLY as an input of
    the measured-evidence-role gate. Declared role, observed canonical-product marker and observed
    structural schema stay three separate facts in the registration."""

    try:
        facts, observed_facts = cpt_adapter.inspect_cpt_asset(
            resolved_path, declared_evidence_role=asset.evidence_role, registered_sha256=sha256
        )
    except cpt_adapter.CptAssetLoadError as exc:
        return _failed_registration(
            asset,
            resolved_path=resolved_path,
            byte_size=byte_size,
            sha256=sha256,
            detail=str(exc),
            provenance_declared=provenance_declared,
        )

    result = cpt_readiness.assess_cpt_readiness(facts)
    observed_facts = {
        **observed_facts,
        "machine_readable_cpt_profile_available": facts.machine_readable_profile_available,
        "canonical_profile_created": facts.canonical_profile_created,
        "canonical_cpt_product_identity_verified": facts.canonical_product_identity_verified,
        "measured_evidence_role_verified": facts.measured_evidence_role_verified,
        "measured_cpt_profile_verified": facts.measured_cpt_profile_verified,
        "registered_asset_sha256": facts.registered_asset_sha256,
        "row_count": facts.row_count,
        "depth_reference": facts.depth_reference,
        "notes": list(facts.notes),
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
        registration_detail=(
            "CPT asset inspected; readiness delegated to geotechnical.cpt_readiness"
        ),
        provenance_declared=provenance_declared,
        observed_facts=observed_facts,
        conflicts=[],
        readiness_status_intrinsic=result.status,
        readiness_status_effective=_effective_readiness_status(result.status, []),
        readiness_result=result.to_dict(),
    )
    return AssetRegistrationResult(registration=registration)


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
    if asset.category == CPT:
        return _register_cpt(
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
        readiness_status_intrinsic=REGISTERED_READINESS_NOT_IMPLEMENTED,
        readiness_status_effective=REGISTERED_READINESS_NOT_IMPLEMENTED,
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
    working_crs_findings: list[str] = field(default_factory=list)

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
    # MAR-026A Section 6: checked centrally, once, regardless of which (if any) asset categories
    # are present -- a bathymetry-only or burial-only project must not carry an obviously
    # invalid working CRS unnoticed just because no route asset happened to trigger the check.
    working_crs_findings = _project_working_crs_findings(working_crs)
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
        working_crs_findings=working_crs_findings,
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
                "readiness_status_intrinsic": reg.readiness_status_intrinsic,
                "readiness_status_effective": reg.readiness_status_effective,
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
        "readiness_status_intrinsic",
        "readiness_status_effective",
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
        "working_crs": summary.working_crs,
        "working_crs_findings": summary.working_crs_findings,
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
                "readiness_status_intrinsic": r.registration.readiness_status_intrinsic,
                "readiness_status_effective": r.registration.readiness_status_effective,
                "readiness_result": r.registration.readiness_result,
            }
            for r in summary.asset_results
        },
    }

"""Canonical project model and explicit cross-asset route linkage (MAR-027).

Sits ABOVE `project.registry` (MAR-026/026A registration, intrinsic/effective readiness) and
`project.route_reference` (primary route, canonical route-reference grid). It adds exactly ONE
new, separate axis -- route LINKAGE -- and never overloads `readiness_status_intrinsic` /
`readiness_status_effective` with new meanings (Section 13). The linkage status answers only:
can the manifest-DECLARED asset -> route relationship be represented safely? It is not a hazard,
risk, or readiness score, and there is no universal project status (Section 27).

Independent dimensions kept structurally separate (Section 5):

    source file identity != declared provenance != observed metadata != measured data !=
    source interpretation != derived output != route relationship != spatial coverage !=
    temporal compatibility != hazard inference

A declared relationship never proves coverage, representativeness, or temporal compatibility.
Relationships are never inferred from filenames, shared directories, matching CRSs, or asset
categories -- an asset without a manifest-declared relationship is simply `NOT_APPLICABLE`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely
from pyproj import CRS
from shapely.geometry import LineString, box
from shapely.geometry.base import BaseGeometry

from marine_engine.project import route_reference
from marine_engine.project.categories import BATHYMETRY_RASTER, BURIAL_PROFILE
from marine_engine.project.manifest import AssetEntry, ProjectManifest
from marine_engine.project.registry import (
    PROJECT_HAZARD_READINESS_DISCLAIMER,
    REGISTERED,
    AssetRegistrationResult,
    ProjectRegistrationSummary,
)

__all__ = [
    "LINKED",
    "LINKED_WITH_LIMITATIONS",
    "UNRESOLVED",
    "NOT_APPLICABLE",
    "ROUTE_LINKAGE_STATUSES",
    "CHAINAGE_CORRELATION_RESOLVED_BY_EXPLICIT_LINEAR_REFERENCE",
    "CHAINAGE_CORRELATION_UNRESOLVED_NO_LINEAR_REFERENCE",
    "CHAINAGE_CORRELATION_UNRESOLVED_NO_NUMERIC_CHAINAGE",
    "SCIENTIFIC_ROLE",
    "ASSET_LINKAGE_COLUMNS",
    "AssetRouteLinkage",
    "CanonicalProjectModel",
    "assess_asset_route_linkage",
    "build_canonical_project_model",
    "build_canonical_project_model_dict",
    "build_asset_linkage_df",
]

# --- Section 13: route-linkage vocabulary -- a separate axis, never a readiness/hazard score ----

LINKED = "LINKED"
LINKED_WITH_LIMITATIONS = "LINKED_WITH_LIMITATIONS"
UNRESOLVED = "UNRESOLVED"
NOT_APPLICABLE = "NOT_APPLICABLE"

ROUTE_LINKAGE_STATUSES = frozenset({LINKED, LINKED_WITH_LIMITATIONS, UNRESOLVED, NOT_APPLICABLE})

# --- Sections 15-17: explicit numeric-chainage correlation states ------------------------------

CHAINAGE_CORRELATION_RESOLVED_BY_EXPLICIT_LINEAR_REFERENCE = "RESOLVED_BY_EXPLICIT_LINEAR_REFERENCE"
CHAINAGE_CORRELATION_UNRESOLVED_NO_LINEAR_REFERENCE = "UNRESOLVED_NO_LINEAR_REFERENCE_DECLARATION"
CHAINAGE_CORRELATION_UNRESOLVED_NO_NUMERIC_CHAINAGE = "UNRESOLVED_NO_NUMERIC_CHAINAGE_VALUES"
CHAINAGE_CORRELATION_UNRESOLVED_COLUMN_UNAVAILABLE = "UNRESOLVED_CHAINAGE_COLUMN_UNAVAILABLE"

SCIENTIFIC_ROLE = "CANONICAL_PROJECT_ROUTE_REFERENCE_MODEL_AND_ASSET_LINKAGE"

# Section 17: a min/max range overlap is never continuous measurement coverage.
CHAINAGE_RANGE_OVERLAP_NOTE = (
    "chainage_range_overlap_* describes only the overlap between the source's min/max numeric "
    "chainage range and the canonical route range [0, route_length_m]; it is NOT continuous "
    "measurement coverage between samples, and the accepted intrinsic burial readiness "
    "coverage_fraction is left untouched."
)
# Section 19: raster bounds are never valid-data coverage.
RASTER_EXTENT_NOTE = (
    "raster_extent_* describes only the raster's observed bounding extent placed relative to the "
    "canonical route; it is NOT valid-cell (nodata-aware) bathymetry coverage along the route."
)
# Section 20: epochs are reported side by side only.
TEMPORAL_NOTE = (
    "declared survey epochs are reported side by side only; temporal compatibility between "
    "assets is not assessed and no allowable separation is defined."
)
NO_RELATIONSHIP_FINDING = (
    "no route_relationship declared in the manifest; a relationship is never inferred from a "
    "filename, shared directory, matching CRS, or asset category"
)

ROUTE_LINKAGE_STATUS_DISCLAIMER = (
    "ROUTE LINKAGE STATUS DESCRIBES ONLY WHETHER A MANIFEST-DECLARED ASSET -> ROUTE RELATIONSHIP "
    "CAN BE REPRESENTED SAFELY. IT IS NOT A HAZARD, RISK, OR READINESS SCORE, AND IT DOES NOT "
    "IMPLY THE ASSET COVERS, IS REPRESENTATIVE OF, OR IS TEMPORALLY COMPATIBLE WITH ANY PART OF "
    "THE ROUTE."
)

ASSET_LINKAGE_COLUMNS: tuple[str, ...] = (
    "project_id",
    "asset_id",
    "category",
    "evidence_role",
    "registration_status",
    "readiness_status_intrinsic",
    "readiness_status_effective",
    "declared_route_asset_id",
    "declared_relationship_type",
    "declared_linear_reference_json",
    "route_linkage_status",
    "route_linkage_findings",
    "linkage_facts_json",
    "asset_survey_epoch_declared",
    "route_survey_epoch_declared",
)


@dataclass(frozen=True)
class AssetRouteLinkage:
    """Section 14: one asset's linkage facts. MAR-026A registration/readiness fields are carried
    through verbatim -- linkage never modifies them."""

    asset_id: str
    category: str
    evidence_role: str
    registration_status: str
    readiness_status_intrinsic: str
    readiness_status_effective: str
    conflicts: list[str]
    declared_route_asset_id: str | None
    declared_relationship_type: str | None
    declared_linear_reference: dict[str, str] | None
    route_linkage_status: str
    route_linkage_findings: list[str]
    linkage_facts: dict[str, Any]
    asset_survey_epoch_declared: str | None
    route_survey_epoch_declared: str | None


def _extent_in_working_crs(
    extent: BaseGeometry, *, observed_crs: str, working_crs: str
) -> BaseGeometry:
    """Places an OBSERVED raster extent in the project working CRS for a derived extent-vs-route
    fact. Only ever called for an asset with no unresolved declared-vs-observed CRS conflict, so
    no conflict is being silently decided here; the source raster itself is never touched."""

    if CRS.from_user_input(observed_crs) == CRS.from_user_input(working_crs):
        return extent
    minx, miny, maxx, maxy = extent.bounds
    segment = max(maxx - minx, maxy - miny) / 100.0
    densified = shapely.segmentize(extent, segment) if segment > 0 else extent
    return gpd.GeoSeries([densified], crs=observed_crs).to_crs(working_crs).iloc[0]


def _raster_extent_linkage(
    result: AssetRegistrationResult, *, route_line: LineString, working_crs: str
) -> tuple[str, list[str], dict[str, Any]]:
    """Section 19: extent-only facts, explicitly NOT valid-cell coverage."""

    reg = result.registration
    facts: dict[str, Any] = {"raster_extent_note": RASTER_EXTENT_NOTE}
    bounds = reg.observed_facts.get("bounds")
    observed_crs = reg.observed_facts.get("observed_crs")
    if not bounds or not observed_crs:
        return (
            UNRESOLVED,
            [
                "raster observed bounds or embedded CRS unavailable; the extent cannot be placed "
                "relative to the canonical route"
            ],
            facts,
        )

    extent = _extent_in_working_crs(
        box(*bounds), observed_crs=str(observed_crs), working_crs=working_crs
    )
    intersects = bool(extent.intersects(route_line))
    overlap_length_m = float(route_line.intersection(extent).length) if intersects else 0.0
    facts.update(
        {
            "raster_observed_crs": str(observed_crs),
            "raster_extent_intersects_route": intersects,
            "raster_extent_route_overlap_length_m": overlap_length_m,
            "route_length_m": float(route_line.length),
        }
    )
    if not intersects:
        return (
            LINKED_WITH_LIMITATIONS,
            ["raster observed extent does not intersect the canonical route"],
            facts,
        )
    return LINKED, [], facts


def _burial_linkage(
    asset: AssetEntry, result: AssetRegistrationResult, *, route_line: LineString
) -> tuple[str, list[str], dict[str, Any]]:
    """Sections 15-17: numeric chainage is correlated with the canonical route ONLY under an
    explicit, supported `linear_reference` declaration. A column named chainage/KP, a similar
    value range, or matching units never substitute for that declaration."""

    rel = asset.route_relationship
    assert rel is not None
    facts: dict[str, Any] = {}
    findings: list[str] = []

    if rel.linear_reference is None:
        facts["chainage_correlation_status"] = CHAINAGE_CORRELATION_UNRESOLVED_NO_LINEAR_REFERENCE
        findings.append(
            "burial_columns.chainage_or_kp_column alone does not establish that its numeric values "
            "share the canonical route's chainage definition (units, origin, direction); no "
            "linear_reference was declared, so numeric chainage correlation is left unresolved"
        )
        return LINKED_WITH_LIMITATIONS, findings, facts

    facts["linear_reference_basis"] = rel.linear_reference.basis
    facts["linear_reference_units"] = rel.linear_reference.units

    df = result.burial_table_df
    column = asset.burial_columns.chainage_or_kp_column if asset.burial_columns else None
    if df is None or column is None or column not in df.columns:
        facts["chainage_correlation_status"] = CHAINAGE_CORRELATION_UNRESOLVED_COLUMN_UNAVAILABLE
        findings.append(f"declared chainage column {column!r} is not available in the loaded table")
        return UNRESOLVED, findings, facts

    numeric = pd.to_numeric(df[column], errors="coerce")
    valid = numeric.dropna()
    route_length_m = float(route_line.length)
    facts.update(
        {
            "route_length_m": route_length_m,
            "record_count": int(len(df)),
            "records_with_numeric_chainage": int(len(valid)),
        }
    )
    if valid.empty:
        facts["chainage_correlation_status"] = CHAINAGE_CORRELATION_UNRESOLVED_NO_NUMERIC_CHAINAGE
        findings.append(f"no numeric values in declared chainage column {column!r}")
        return UNRESOLVED, findings, facts

    source_min = float(valid.min())
    source_max = float(valid.max())
    within = int(((valid >= 0.0) & (valid <= route_length_m)).sum())
    outside = int(len(valid) - within)
    overlap_m = max(0.0, min(source_max, route_length_m) - max(source_min, 0.0))
    duplicate_count = int(valid.duplicated().sum())
    non_numeric_count = int(len(df) - len(valid))
    resolved = CHAINAGE_CORRELATION_RESOLVED_BY_EXPLICIT_LINEAR_REFERENCE
    facts.update(
        {
            "chainage_correlation_status": resolved,
            "source_chainage_min_m": source_min,
            "source_chainage_max_m": source_max,
            "records_within_route_chainage_range": within,
            "records_outside_route_chainage_range": outside,
            "chainage_range_overlap_m": overlap_m,
            "chainage_range_overlap_fraction": overlap_m / route_length_m,
            "duplicate_numeric_chainage_count": duplicate_count,
            "chainage_range_overlap_note": CHAINAGE_RANGE_OVERLAP_NOTE,
        }
    )

    status = LINKED
    if outside:
        status = LINKED_WITH_LIMITATIONS
        findings.append(
            f"{outside} record(s) have numeric chainage outside the canonical route range "
            f"[0, {route_length_m}] m -- reported, never clipped or dropped"
        )
    if duplicate_count:
        status = LINKED_WITH_LIMITATIONS
        findings.append(
            f"{duplicate_count} duplicate numeric chainage value(s) in the source -- kept "
            "visible, never collapsed"
        )
    if non_numeric_count:
        status = LINKED_WITH_LIMITATIONS
        findings.append(
            f"{non_numeric_count} record(s) have a non-numeric/missing chainage value and could "
            "not be correlated"
        )
    if overlap_m <= 0.0:
        status = LINKED_WITH_LIMITATIONS
        findings.append("source chainage range does not overlap the canonical route range")
    return status, findings, facts


def assess_asset_route_linkage(
    asset: AssetEntry,
    result: AssetRegistrationResult,
    *,
    results_by_id: dict[str, AssetRegistrationResult],
    working_crs: str,
    working_crs_findings: list[str],
) -> AssetRouteLinkage:
    """Sections 8, 13-20: derive one asset's linkage row from the manifest-declared relationship
    and the already-computed registration results. Nothing about the asset's evidence role,
    intrinsic readiness, or effective readiness is recomputed or altered."""

    reg = result.registration
    rel = asset.route_relationship
    common = {
        "asset_id": reg.asset_id,
        "category": reg.category,
        "evidence_role": reg.evidence_role,
        "registration_status": reg.registration_status,
        "readiness_status_intrinsic": reg.readiness_status_intrinsic,
        "readiness_status_effective": reg.readiness_status_effective,
        "conflicts": list(reg.conflicts),
        "asset_survey_epoch_declared": asset.provenance.survey_epoch,
    }

    if rel is None:
        return AssetRouteLinkage(
            **common,
            declared_route_asset_id=None,
            declared_relationship_type=None,
            declared_linear_reference=None,
            route_linkage_status=NOT_APPLICABLE,
            route_linkage_findings=[NO_RELATIONSHIP_FINDING],
            linkage_facts={},
            route_survey_epoch_declared=None,
        )

    route_result = results_by_id.get(rel.route_asset_id)
    route_epoch = (
        route_result.registration.provenance_declared.get("survey_epoch")
        if route_result is not None
        else None
    )
    declared = {
        "declared_route_asset_id": rel.route_asset_id,
        "declared_relationship_type": rel.relationship_type,
        "declared_linear_reference": (
            rel.linear_reference.model_dump() if rel.linear_reference is not None else None
        ),
        "route_survey_epoch_declared": route_epoch,
    }

    def _unresolved(findings: list[str]) -> AssetRouteLinkage:
        return AssetRouteLinkage(
            **common,
            **declared,
            route_linkage_status=UNRESOLVED,
            route_linkage_findings=findings,
            linkage_facts={"temporal_note": TEMPORAL_NOTE},
        )

    if reg.registration_status != REGISTERED:
        return _unresolved(
            [
                f"asset did not register ({reg.registration_detail}); no observed facts are "
                "available to represent the declared relationship"
            ]
        )
    if reg.conflicts:
        return _unresolved(
            [
                "asset has an unresolved declared-vs-observed conflict; the software does not "
                f"decide which side is correct, so it cannot be placed relative to the route: {c}"
                for c in reg.conflicts
            ]
        )
    if route_result is None:
        return _unresolved(
            [f"referenced route asset {rel.route_asset_id!r} matches no registered asset"]
        )
    route_findings = route_reference.evaluate_canonical_route_usability(
        route_result, working_crs_findings=working_crs_findings
    )
    if route_findings:
        return _unresolved(
            [
                f"referenced route asset {rel.route_asset_id!r} cannot provide a canonical "
                f"route: {finding}"
                for finding in route_findings
            ]
        )

    assert route_result.canonical_route_gdf is not None
    route_line = route_result.canonical_route_gdf.geometry.iloc[0]

    if reg.category == BURIAL_PROFILE:
        status, findings, facts = _burial_linkage(asset, result, route_line=route_line)
    elif reg.category == BATHYMETRY_RASTER:
        status, findings, facts = _raster_extent_linkage(
            result, route_line=route_line, working_crs=working_crs
        )
    else:
        status = LINKED_WITH_LIMITATIONS
        findings = [
            f"no category-specific route-correlation facts are implemented for category "
            f"{reg.category!r} in MAR-027; the declared relationship is recorded only"
        ]
        facts = {}
    facts["temporal_note"] = TEMPORAL_NOTE

    return AssetRouteLinkage(
        **common,
        **declared,
        route_linkage_status=status,
        route_linkage_findings=findings,
        linkage_facts=facts,
    )


@dataclass(frozen=True)
class CanonicalProjectModel:
    project_id: str
    project_name: str
    project_description: str
    working_crs: str
    working_crs_findings: list[str]
    route_reference: route_reference.RouteReferenceResult
    asset_linkages: list[AssetRouteLinkage]

    def route_linkage_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for linkage in self.asset_linkages:
            counts[linkage.route_linkage_status] = counts.get(linkage.route_linkage_status, 0) + 1
        return dict(sorted(counts.items()))

    def evidence_role_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for linkage in self.asset_linkages:
            counts[linkage.evidence_role] = counts.get(linkage.evidence_role, 0) + 1
        return dict(sorted(counts.items()))


def build_canonical_project_model(
    manifest: ProjectManifest, summary: ProjectRegistrationSummary
) -> CanonicalProjectModel:
    """Section 21: assemble the canonical project model from the manifest and the already-run
    MAR-026/026A registration summary -- registration is never re-implemented here."""

    rr = route_reference.build_project_route_reference(manifest, summary)
    results_by_id = {r.registration.asset_id: r for r in summary.asset_results}
    linkages = [
        assess_asset_route_linkage(
            asset,
            results_by_id[asset.asset_id],
            results_by_id=results_by_id,
            working_crs=summary.working_crs,
            working_crs_findings=summary.working_crs_findings,
        )
        for asset in manifest.assets
    ]
    return CanonicalProjectModel(
        project_id=manifest.project.id,
        project_name=manifest.project.name,
        project_description=manifest.project.description,
        working_crs=summary.working_crs,
        working_crs_findings=list(summary.working_crs_findings),
        route_reference=rr,
        asset_linkages=linkages,
    )


def _linkage_to_dict(linkage: AssetRouteLinkage) -> dict[str, Any]:
    return {
        "category": linkage.category,
        "evidence_role": linkage.evidence_role,
        "registration_status": linkage.registration_status,
        "readiness_status_intrinsic": linkage.readiness_status_intrinsic,
        "readiness_status_effective": linkage.readiness_status_effective,
        "conflicts": list(linkage.conflicts),
        "declared_route_asset_id": linkage.declared_route_asset_id,
        "declared_relationship_type": linkage.declared_relationship_type,
        "declared_linear_reference": linkage.declared_linear_reference,
        "route_linkage_status": linkage.route_linkage_status,
        "route_linkage_findings": list(linkage.route_linkage_findings),
        "linkage_facts": dict(linkage.linkage_facts),
        "asset_survey_epoch_declared": linkage.asset_survey_epoch_declared,
        "route_survey_epoch_declared": linkage.route_survey_epoch_declared,
    }


def build_canonical_project_model_dict(model: CanonicalProjectModel) -> dict[str, Any]:
    """Section 21: `canonical_project_model.json` -- deterministic model/linkage METADATA (no
    source tables copied in, no timestamps, no machine-specific paths), with component statuses
    exposed separately and no aggregate project status or numeric score (Section 27)."""

    rr = model.route_reference
    built = rr.status == route_reference.ROUTE_REFERENCE_BUILT
    return {
        "scientific_role": SCIENTIFIC_ROLE,
        "project": {
            "id": model.project_id,
            "name": model.project_name,
            "description": model.project_description,
        },
        "working_crs": model.working_crs,
        "working_crs_findings": list(model.working_crs_findings),
        "primary_route": {
            "status": rr.primary_route.status,
            "primary_route_asset_id": rr.primary_route.primary_route_asset_id,
            "findings": list(rr.primary_route.findings),
        },
        "route_reference": {
            "status": rr.status,
            "findings": list(rr.findings),
            "configured_interval_m": rr.configured_interval_m,
            "interval_units": "m",
            "chainage_origin_basis": rr.chainage_origin_basis,
            "route_length_m": rr.route_length_m,
            "station_count": rr.station_count,
            "regular_station_count": rr.regular_station_count,
            "terminal_residual_m": rr.terminal_residual_m,
            "gis_layer": route_reference.ROUTE_REFERENCE_LAYER if built else None,
            "disclaimer": route_reference.ROUTE_REFERENCE_GRID_DISCLAIMER,
        },
        "asset_count": len(model.asset_linkages),
        "evidence_role_counts": model.evidence_role_counts(),
        "route_linkage_status_counts": model.route_linkage_status_counts(),
        "assets": {linkage.asset_id: _linkage_to_dict(linkage) for linkage in model.asset_linkages},
        "route_linkage_status_disclaimer": ROUTE_LINKAGE_STATUS_DISCLAIMER,
        "project_hazard_readiness_disclaimer": PROJECT_HAZARD_READINESS_DISCLAIMER,
        "explicit_limitations": [
            "no universal project readiness, hazard, or risk status is produced; primary-route, "
            "route-reference, working-CRS, and per-asset linkage findings are exposed separately",
            TEMPORAL_NOTE,
            CHAINAGE_RANGE_OVERLAP_NOTE,
            RASTER_EXTENT_NOTE,
            "coordinate-based projection of burial records onto the route "
            "(preprocessing.chainage.project_point_to_route) is not implemented in MAR-027",
            "valid-cell (nodata-aware) bathymetry coverage along the route is not implemented in "
            "MAR-027",
            "burial measurement-reference and sign-convention semantics are neither resolved nor "
            "altered by route linkage; the accepted intrinsic burial readiness result is untouched",
        ],
    }


def build_asset_linkage_df(model: CanonicalProjectModel) -> pd.DataFrame:
    """Section 23: `project_asset_linkage.parquet` -- one row per manifest asset. List/structured
    fields are JSON-encoded strings so they stay deterministic, lossless, and serializable."""

    rows = [
        {
            "project_id": model.project_id,
            "asset_id": linkage.asset_id,
            "category": linkage.category,
            "evidence_role": linkage.evidence_role,
            "registration_status": linkage.registration_status,
            "readiness_status_intrinsic": linkage.readiness_status_intrinsic,
            "readiness_status_effective": linkage.readiness_status_effective,
            "declared_route_asset_id": linkage.declared_route_asset_id,
            "declared_relationship_type": linkage.declared_relationship_type,
            "declared_linear_reference_json": (
                json.dumps(linkage.declared_linear_reference, sort_keys=True)
                if linkage.declared_linear_reference is not None
                else None
            ),
            "route_linkage_status": linkage.route_linkage_status,
            "route_linkage_findings": json.dumps(list(linkage.route_linkage_findings)),
            "linkage_facts_json": json.dumps(linkage.linkage_facts, sort_keys=True, default=str),
            "asset_survey_epoch_declared": linkage.asset_survey_epoch_declared,
            "route_survey_epoch_declared": linkage.route_survey_epoch_declared,
        }
        for linkage in model.asset_linkages
    ]
    if not rows:
        return pd.DataFrame(columns=list(ASSET_LINKAGE_COLUMNS))
    return pd.DataFrame(rows, columns=list(ASSET_LINKAGE_COLUMNS))

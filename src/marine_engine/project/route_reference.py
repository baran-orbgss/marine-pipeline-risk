"""Canonical project route-reference grid and primary-route operationalization (MAR-027).

Sits ABOVE `project.registry` (MAR-026/026A registration + intrinsic/effective readiness): it
consumes an already-registered project and decides, with explicit findings, whether the declared
primary route may drive a CANONICAL ROUTE-REFERENCE GRID -- a deterministic linear/spatial
indexing framework along the canonical route for downstream data alignment. Grid points are
never "supports": no physical pipeline-support claim is made or implied (Section 6).

Primary-route honesty (Section 9): `primary_route_asset_id` is the ONLY source of primary-route
identity. When it is absent, no route is auto-selected even if exactly one route exists; when it
references a route that registered/assessed as unusable, a controlled finding is produced and no
other route is substituted.

Chainage honesty (Section 10): chainage 0 is the canonical route geometry's own start vertex.
The basis is read from the accepted canonical route (`geometry_direction_semantics`, i.e.
`SOURCE_GEOMETRY_ORDER`) -- never inferred from a filename, project name, or pipe name.

Linear-referencing mathematics is NOT rewritten here (Section 11): stations come from the
accepted MAR-004 primitives `preprocessing.chainage.compute_chainage_stations` and
`format_kp_label`; this module is only the project-specific wrapper around them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from marine_engine.preprocessing import chainage as chainage_primitives
from marine_engine.project import route_adapter
from marine_engine.project.categories import PIPELINE_ROUTE
from marine_engine.project.manifest import ProjectManifest
from marine_engine.project.registry import (
    REGISTERED,
    AssetRegistrationResult,
    ProjectRegistrationSummary,
)

__all__ = [
    "PRIMARY_ROUTE_AVAILABLE",
    "PRIMARY_ROUTE_UNUSABLE",
    "PRIMARY_ROUTE_NOT_DECLARED",
    "ROUTE_REFERENCE_BUILT",
    "ROUTE_REFERENCE_NOT_BUILT",
    "ROUTE_REFERENCE_LAYER",
    "ROUTE_REFERENCE_COLUMNS",
    "ROUTE_REFERENCE_GRID_DISCLAIMER",
    "PrimaryRouteResolution",
    "RouteReferenceGrid",
    "RouteReferenceResult",
    "evaluate_canonical_route_usability",
    "resolve_primary_route",
    "build_route_reference_grid",
    "build_project_route_reference",
    "write_route_reference_gpkg",
]

PRIMARY_ROUTE_AVAILABLE = "PRIMARY_ROUTE_AVAILABLE"
PRIMARY_ROUTE_UNUSABLE = "PRIMARY_ROUTE_UNUSABLE"
PRIMARY_ROUTE_NOT_DECLARED = "PRIMARY_ROUTE_NOT_DECLARED"

ROUTE_REFERENCE_BUILT = "ROUTE_REFERENCE_BUILT"
ROUTE_REFERENCE_NOT_BUILT = "ROUTE_REFERENCE_NOT_BUILT"

ROUTE_REFERENCE_LAYER = "route_reference_points"

ROUTE_REFERENCE_COLUMNS: tuple[str, ...] = (
    "project_id",
    "route_asset_id",
    "station_index",
    "chainage_m",
    "kp_label",
    "fraction_along_route",
    "is_terminal",
    "route_reference_interval_m",
    "chainage_origin_basis",
    "geometry",
)

# Section 6: the grid is an indexing framework, never a physical-support statement.
ROUTE_REFERENCE_GRID_DISCLAIMER = (
    "THE CANONICAL ROUTE-REFERENCE GRID IS A DETERMINISTIC LINEAR/SPATIAL INDEXING FRAMEWORK "
    "ALONG THE CANONICAL ROUTE FOR DOWNSTREAM DATA ALIGNMENT ONLY. ITS POINTS ARE NOT PIPELINE "
    "SUPPORTS OR SUPPORT LOCATIONS AND CARRY NO PHYSICAL-SUPPORT CLAIM; ITS INTERVAL IS AN "
    "INDEXING RESOLUTION, NOT A SURVEY ACCURACY OR ENGINEERING RESOLUTION."
)

_ROUTE_USABLE_EFFECTIVE_STATUSES = frozenset(
    {route_adapter.READY, route_adapter.READY_WITH_LIMITATIONS}
)


def evaluate_canonical_route_usability(
    result: AssetRegistrationResult, *, working_crs_findings: list[str]
) -> list[str]:
    """Section 9: every prerequisite a registered route must meet before it may drive canonical
    route-referenced work. Returns the unmet-prerequisite findings (empty means usable). Never
    repairs, reprojects, or substitutes anything -- it only reports."""

    reg = result.registration
    if reg.category != PIPELINE_ROUTE:
        return [
            f"asset {reg.asset_id!r} has category {reg.category!r}, not {PIPELINE_ROUTE!r} -- it "
            "cannot serve as a route"
        ]
    if reg.registration_status != REGISTERED:
        return [f"route asset {reg.asset_id!r} did not register: {reg.registration_detail}"]

    findings: list[str] = []
    findings.extend(
        f"route asset {reg.asset_id!r} has an unresolved declared-vs-observed conflict: {conflict}"
        for conflict in reg.conflicts
    )
    if reg.readiness_status_effective not in _ROUTE_USABLE_EFFECTIVE_STATUSES:
        findings.append(
            f"route asset {reg.asset_id!r} effective readiness is "
            f"{reg.readiness_status_effective!r}; canonical route generation requires "
            f"{route_adapter.READY!r} or {route_adapter.READY_WITH_LIMITATIONS!r}"
        )
    findings.extend(f"project working_crs: {finding}" for finding in working_crs_findings)
    if result.canonical_route_gdf is None:
        findings.append(
            f"route asset {reg.asset_id!r} produced no canonical route geometry in the project "
            "working CRS"
        )
    return findings


@dataclass(frozen=True)
class PrimaryRouteResolution:
    status: str
    primary_route_asset_id: str | None
    findings: list[str]
    canonical_route_gdf: gpd.GeoDataFrame | None


def resolve_primary_route(summary: ProjectRegistrationSummary) -> PrimaryRouteResolution:
    """Section 9: operationalize `primary_route_asset_id` -- never guessed, never substituted."""

    primary_id = summary.primary_route_asset_id
    route_count = sum(1 for r in summary.asset_results if r.registration.category == PIPELINE_ROUTE)
    if primary_id is None:
        return PrimaryRouteResolution(
            status=PRIMARY_ROUTE_NOT_DECLARED,
            primary_route_asset_id=None,
            findings=[
                f"no primary_route_asset_id declared in the manifest; {route_count} "
                f"{PIPELINE_ROUTE} asset(s) registered and none is auto-selected as primary"
            ],
            canonical_route_gdf=None,
        )

    result = next((r for r in summary.asset_results if r.registration.asset_id == primary_id), None)
    if result is None:
        return PrimaryRouteResolution(
            status=PRIMARY_ROUTE_UNUSABLE,
            primary_route_asset_id=primary_id,
            findings=[f"primary_route_asset_id {primary_id!r} matches no registered asset"],
            canonical_route_gdf=None,
        )

    findings = evaluate_canonical_route_usability(
        result, working_crs_findings=summary.working_crs_findings
    )
    if findings:
        return PrimaryRouteResolution(
            status=PRIMARY_ROUTE_UNUSABLE,
            primary_route_asset_id=primary_id,
            findings=findings,
            canonical_route_gdf=None,
        )
    return PrimaryRouteResolution(
        status=PRIMARY_ROUTE_AVAILABLE,
        primary_route_asset_id=primary_id,
        findings=[],
        canonical_route_gdf=result.canonical_route_gdf,
    )


@dataclass(frozen=True)
class RouteReferenceGrid:
    gdf: gpd.GeoDataFrame
    route_length_m: float
    regular_station_count: int
    terminal_residual_m: float
    chainage_origin_basis: str


def build_route_reference_grid(
    canonical_route_gdf: gpd.GeoDataFrame,
    *,
    project_id: str,
    route_asset_id: str,
    interval_m: float,
) -> RouteReferenceGrid:
    """Section 12: deterministic stations from chainage 0 (canonical route geometry start) to the
    exact route terminus, in the canonical route's own (project working) CRS. Stations come from
    the accepted MAR-004 `compute_chainage_stations` (regular interval + exact terminus, strictly
    increasing, no duplicates); `kp_label` is a display field only -- `chainage_m` is retained
    unrounded."""

    line = canonical_route_gdf.geometry.iloc[0]
    basis = str(canonical_route_gdf["geometry_direction_semantics"].iloc[0])
    stations = chainage_primitives.compute_chainage_stations(line, interval_m)
    total_length_m = float(stations.total_length_m)

    records = [
        {
            "project_id": project_id,
            "route_asset_id": route_asset_id,
            "station_index": station.station_index,
            "chainage_m": float(station.chainage_m),
            "kp_label": chainage_primitives.format_kp_label(station.chainage_m),
            "fraction_along_route": float(station.chainage_m) / total_length_m,
            "is_terminal": bool(station.is_terminal),
            "route_reference_interval_m": float(interval_m),
            "chainage_origin_basis": basis,
        }
        for station in stations.stations
    ]
    gdf = gpd.GeoDataFrame(
        pd.DataFrame(records, columns=list(ROUTE_REFERENCE_COLUMNS[:-1])),
        geometry=[station.point for station in stations.stations],
        crs=canonical_route_gdf.crs,
    )
    return RouteReferenceGrid(
        gdf=gdf,
        route_length_m=total_length_m,
        regular_station_count=int(stations.regular_station_count),
        terminal_residual_m=float(stations.terminal_residual_m),
        chainage_origin_basis=basis,
    )


@dataclass(frozen=True)
class RouteReferenceResult:
    status: str
    findings: list[str]
    primary_route: PrimaryRouteResolution
    configured_interval_m: float | None
    chainage_origin_basis: str | None
    route_length_m: float | None
    station_count: int
    regular_station_count: int
    terminal_residual_m: float | None
    grid_gdf: gpd.GeoDataFrame | None


def build_project_route_reference(
    manifest: ProjectManifest, summary: ProjectRegistrationSummary
) -> RouteReferenceResult:
    """Sections 7, 9, 12: build the canonical route-reference grid only when EVERY prerequisite
    holds -- an explicitly configured positive `route_reference.interval_m` (no default spacing
    is ever invented) and a usable declared primary route. Otherwise the result explains exactly
    why no grid was produced, and no grid geometry is emitted."""

    primary = resolve_primary_route(summary)
    findings = list(primary.findings)
    configured_interval_m = (
        manifest.route_reference.interval_m if manifest.route_reference is not None else None
    )
    if configured_interval_m is None:
        findings.append(
            "no route_reference.interval_m configured in the manifest; no default grid spacing "
            "is invented"
        )

    if primary.status != PRIMARY_ROUTE_AVAILABLE or configured_interval_m is None:
        return RouteReferenceResult(
            status=ROUTE_REFERENCE_NOT_BUILT,
            findings=findings,
            primary_route=primary,
            configured_interval_m=configured_interval_m,
            chainage_origin_basis=None,
            route_length_m=None,
            station_count=0,
            regular_station_count=0,
            terminal_residual_m=None,
            grid_gdf=None,
        )

    assert primary.canonical_route_gdf is not None  # guaranteed by PRIMARY_ROUTE_AVAILABLE
    assert primary.primary_route_asset_id is not None
    grid = build_route_reference_grid(
        primary.canonical_route_gdf,
        project_id=manifest.project.id,
        route_asset_id=primary.primary_route_asset_id,
        interval_m=configured_interval_m,
    )
    return RouteReferenceResult(
        status=ROUTE_REFERENCE_BUILT,
        findings=[],
        primary_route=primary,
        configured_interval_m=configured_interval_m,
        chainage_origin_basis=grid.chainage_origin_basis,
        route_length_m=grid.route_length_m,
        station_count=int(len(grid.gdf)),
        regular_station_count=grid.regular_station_count,
        terminal_residual_m=grid.terminal_residual_m,
        grid_gdf=grid.gdf,
    )


def write_route_reference_gpkg(gdf: gpd.GeoDataFrame, output_path: Path) -> Path:
    """Section 22: `project_route_reference.gpkg`, layer `route_reference_points`, carrying the
    canonical project working CRS. Callers must only invoke this when a grid was actually built."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=ROUTE_REFERENCE_LAYER)
    return output_path

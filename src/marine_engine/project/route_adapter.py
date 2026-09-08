"""Generic project-supplied pipeline route ingestion and readiness (MAR-026 Section 9).

Mirrors the established `ReadinessCheck`/`ReadinessResult` pattern (imported directly from
`marine_engine.terrain.readiness`, never duplicated) with a route-specific fact/check set.
Canonical route direction is always `SOURCE_GEOMETRY_ORDER` -- this module never infers which
end is a semantic platform/landfall endpoint. A topologically ambiguous (disconnected,
non-mergeable multi-part) route is reported `NOT_READY` rather than having connectivity
invented -- this POC never silently merges a disconnected route network into one pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

from marine_engine.burial import route as generic_route
from marine_engine.terrain.readiness import (
    BLOCKING,
    LIMITATION,
    NOT_READY,
    READY,
    READY_WITH_LIMITATIONS,
    ReadinessCheck,
    ReadinessResult,
)

__all__ = [
    "READY",
    "READY_WITH_LIMITATIONS",
    "NOT_READY",
    "BLOCKING",
    "LIMITATION",
    "ReadinessCheck",
    "ReadinessResult",
    "SOURCE_GEOMETRY_ORDER",
    "RouteFacts",
    "assess_route_readiness",
    "inspect_route_source",
    "build_canonical_project_route",
    "is_valid_crs",
    "is_projected_metric_crs",
]

# Section 9: canonical route direction -- never a guessed semantic endpoint identity.
SOURCE_GEOMETRY_ORDER = "SOURCE_GEOMETRY_ORDER"

MIN_PLAUSIBLE_PROJECTED_COORD_M = 1000.0
MAX_PLAUSIBLE_PROJECTED_COORD_M = 1.0e8


def is_valid_crs(crs_string: str | None) -> bool:
    if not crs_string:
        return False
    try:
        CRS.from_user_input(crs_string)
        return True
    except CRSError:
        return False


def is_projected_metric_crs(crs_string: str | None) -> bool:
    """Section 9: the project's working CRS must be projected and metric for pipeline
    engineering operations -- geographic (degree-based) CRSs are never accepted here."""

    if not crs_string:
        return False
    try:
        crs = CRS.from_user_input(crs_string)
    except CRSError:
        return False
    if crs.is_geographic:
        return False
    axis_units = {axis.unit_name for axis in crs.axis_info}
    return bool(axis_units) and axis_units <= {"metre", "meter"}


@dataclass(frozen=True)
class RouteFacts:
    """Plain facts about an already-loaded route source -- the caller does its own I/O (via
    GeoPandas/Fiona), so this module stays trivially unit-testable with synthetic values."""

    file_readable: bool
    layer_exists: bool | None  # None when no specific layer was requested
    is_empty: bool
    geometry_types: tuple[str, ...]
    all_line_geometry: bool
    crs_is_present: bool
    crs_is_geographic: bool | None
    coordinate_magnitude_plausible: bool
    is_single_continuous_line: bool
    disconnected_part_count: int
    source_geometry_length: float | None  # in the SOURCE CRS's own units -- sign check only
    working_crs_valid: bool
    working_crs_is_projected_metric: bool
    survey_epoch_known: bool


def assess_route_readiness(facts: RouteFacts) -> ReadinessResult:
    """Section 9's required checks. Never a numeric score."""

    checks: list[ReadinessCheck] = []

    checks.append(
        ReadinessCheck(
            "file_readable",
            facts.file_readable,
            BLOCKING,
            "route source file could not be opened"
            if not facts.file_readable
            else "route source file opened successfully",
        )
    )
    if facts.layer_exists is not None:
        checks.append(
            ReadinessCheck(
                "layer_exists",
                facts.layer_exists,
                BLOCKING,
                "requested layer not found in the source file"
                if not facts.layer_exists
                else "requested layer found",
            )
        )
    if not facts.file_readable or facts.layer_exists is False:
        checks_t = tuple(checks)
        return ReadinessResult(status=NOT_READY, checks=checks_t)

    checks.append(
        ReadinessCheck(
            "non_empty",
            not facts.is_empty,
            BLOCKING,
            "route source contains zero features" if facts.is_empty else "route source non-empty",
        )
    )
    checks.append(
        ReadinessCheck(
            "geometry_is_line_type",
            facts.all_line_geometry,
            BLOCKING,
            f"route source geometry type(s) {facts.geometry_types} are not all "
            "LineString/MultiLineString"
            if not facts.all_line_geometry
            else f"geometry type(s): {facts.geometry_types}",
        )
    )
    checks.append(
        ReadinessCheck(
            "crs_present",
            facts.crs_is_present,
            BLOCKING,
            "no CRS found on the route source" if not facts.crs_is_present else "CRS is present",
        )
    )
    checks.append(
        ReadinessCheck(
            "coordinate_magnitude_plausible",
            facts.coordinate_magnitude_plausible,
            BLOCKING,
            "route coordinates do not look plausible for the declared/observed CRS"
            if not facts.coordinate_magnitude_plausible
            else "route coordinate magnitude is plausible",
        )
    )
    checks.append(
        ReadinessCheck(
            "topologically_unambiguous",
            facts.is_single_continuous_line,
            BLOCKING,
            f"route source resolves to {facts.disconnected_part_count} disconnected part(s) -- "
            "refusing to invent connectivity between them"
            if not facts.is_single_continuous_line
            else "route source merges into one continuous line",
        )
    )
    length_positive = facts.source_geometry_length is not None and facts.source_geometry_length > 0
    checks.append(
        ReadinessCheck(
            "positive_length",
            length_positive,
            BLOCKING,
            f"route length is not positive: {facts.source_geometry_length}"
            if not length_positive
            else f"route length {facts.source_geometry_length} (source CRS units) is positive",
        )
    )
    checks.append(
        ReadinessCheck(
            "working_crs_valid",
            facts.working_crs_valid,
            BLOCKING,
            "project working_crs is not a valid CRS identifier"
            if not facts.working_crs_valid
            else "project working_crs is valid",
        )
    )
    checks.append(
        ReadinessCheck(
            "working_crs_projected_metric",
            facts.working_crs_is_projected_metric,
            BLOCKING,
            "project working_crs is not a projected, metric CRS -- required for pipeline "
            "engineering operations"
            if not facts.working_crs_is_projected_metric
            else "project working_crs is projected and metric",
        )
    )
    checks.append(
        ReadinessCheck(
            "survey_epoch_known",
            facts.survey_epoch_known,
            LIMITATION,
            "no declared survey/digitization epoch for the route source"
            if not facts.survey_epoch_known
            else "survey/digitization epoch declared",
        )
    )

    checks_t = tuple(checks)
    if any(not c.passed and c.severity == BLOCKING for c in checks_t):
        status = NOT_READY
    elif any(not c.passed and c.severity == LIMITATION for c in checks_t):
        status = READY_WITH_LIMITATIONS
    else:
        status = READY
    return ReadinessResult(status=status, checks=checks_t)


def _resolve_single_line(geoms: list) -> tuple[LineString | None, int]:
    """Never invents connectivity: only a genuinely single LineString, or a MultiLineString/
    list of LineStrings that `shapely.ops.linemerge` can merge into ONE continuous line,
    resolves. Returns `(line_or_None, disconnected_part_count)`."""

    if len(geoms) == 1 and isinstance(geoms[0], LineString):
        return geoms[0], 1
    merged = linemerge(geoms)
    if isinstance(merged, LineString):
        return merged, 1
    if isinstance(merged, MultiLineString):
        return None, len(merged.geoms)
    return None, len(geoms)


def inspect_route_source(
    path: Path,
    *,
    layer: str | None,
    working_crs: str,
    declared_survey_epoch: str | None,
) -> tuple[RouteFacts, gpd.GeoDataFrame | None, LineString | None, str | None]:
    """Opens the real route file, extracts `RouteFacts`, and -- only if safe and unambiguous --
    resolves one canonical `LineString` in `SOURCE_GEOMETRY_ORDER`. Returns
    `(facts, raw_gdf_or_None, canonical_line_or_None, source_crs_or_None)`."""

    working_crs_valid = is_valid_crs(working_crs)
    working_crs_projected_metric = is_projected_metric_crs(working_crs)

    if not path.is_file():
        facts = RouteFacts(
            file_readable=False,
            layer_exists=None,
            is_empty=True,
            geometry_types=(),
            all_line_geometry=False,
            crs_is_present=False,
            crs_is_geographic=None,
            coordinate_magnitude_plausible=False,
            is_single_continuous_line=False,
            disconnected_part_count=0,
            source_geometry_length=None,
            working_crs_valid=working_crs_valid,
            working_crs_is_projected_metric=working_crs_projected_metric,
            survey_epoch_known=declared_survey_epoch is not None,
        )
        return facts, None, None, None

    try:
        gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        layer_exists: bool | None = True
    except Exception:
        facts = RouteFacts(
            file_readable=True,
            layer_exists=False if layer else None,
            is_empty=True,
            geometry_types=(),
            all_line_geometry=False,
            crs_is_present=False,
            crs_is_geographic=None,
            coordinate_magnitude_plausible=False,
            is_single_continuous_line=False,
            disconnected_part_count=0,
            source_geometry_length=None,
            working_crs_valid=working_crs_valid,
            working_crs_is_projected_metric=working_crs_projected_metric,
            survey_epoch_known=declared_survey_epoch is not None,
        )
        return facts, None, None, None

    is_empty = bool(gdf.empty)
    geometry_types = tuple(sorted(gdf.geom_type.dropna().unique().tolist())) if not is_empty else ()
    all_line_geometry = bool(
        not is_empty and geometry_types and set(geometry_types) <= {"LineString", "MultiLineString"}
    )
    crs_is_present = gdf.crs is not None
    crs_is_geographic = bool(gdf.crs.is_geographic) if gdf.crs is not None else None
    source_crs = gdf.crs.to_string() if gdf.crs is not None else None

    coordinate_magnitude_plausible = True
    if not is_empty and crs_is_present:
        minx, miny, maxx, maxy = gdf.total_bounds
        if crs_is_geographic:
            coordinate_magnitude_plausible = (
                (-180.0 <= minx <= 180.0)
                and (-180.0 <= maxx <= 180.0)
                and (-90.0 <= miny <= 90.0)
                and (-90.0 <= maxy <= 90.0)
            )
        else:
            coordinate_magnitude_plausible = all(
                MIN_PLAUSIBLE_PROJECTED_COORD_M <= abs(v) <= MAX_PLAUSIBLE_PROJECTED_COORD_M
                for v in (minx, miny, maxx, maxy)
                if v != 0
            )

    canonical_line: LineString | None = None
    is_single_continuous_line = False
    disconnected_part_count = 0
    source_geometry_length: float | None = None
    if all_line_geometry:
        canonical_line, disconnected_part_count = _resolve_single_line(list(gdf.geometry))
        is_single_continuous_line = canonical_line is not None
        if canonical_line is not None:
            source_geometry_length = float(canonical_line.length)

    facts = RouteFacts(
        file_readable=True,
        layer_exists=layer_exists,
        is_empty=is_empty,
        geometry_types=geometry_types,
        all_line_geometry=all_line_geometry,
        crs_is_present=crs_is_present,
        crs_is_geographic=crs_is_geographic,
        coordinate_magnitude_plausible=coordinate_magnitude_plausible,
        is_single_continuous_line=is_single_continuous_line,
        disconnected_part_count=disconnected_part_count,
        source_geometry_length=source_geometry_length,
        working_crs_valid=working_crs_valid,
        working_crs_is_projected_metric=working_crs_projected_metric,
        survey_epoch_known=declared_survey_epoch is not None,
    )
    return facts, gdf, canonical_line, source_crs


def build_canonical_project_route(
    line: LineString, *, asset_id: str, source_route_name: str, source_crs: str, working_crs: str
) -> gpd.GeoDataFrame:
    """Section 9: "If safe and unambiguous, produce a canonical project route in the project's
    working CRS." Reuses `burial.route.build_canonical_asset_route` (already proven generic and
    dataset-agnostic) rather than duplicating it, then reprojects into `working_crs` and records
    the reprojected route's own true metric length -- the source geometry and source CRS are
    preserved separately as plain columns, never overwritten."""

    source_gdf = generic_route.build_canonical_asset_route(
        route=line,
        asset_id=asset_id,
        source_route_name=source_route_name,
        source_crs=source_crs,
        geometry_direction_semantics=SOURCE_GEOMETRY_ORDER,
    )
    canonical_gdf = source_gdf.to_crs(working_crs)
    canonical_gdf["canonical_route_length_m"] = float(canonical_gdf.geometry.iloc[0].length)
    return canonical_gdf

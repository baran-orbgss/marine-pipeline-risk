"""Generic canonical linear-asset route (MAR-024 Section 6).

Zero dependency on any specific project or dataset -- the caller supplies
an already-loaded real route LineString (never invented here) plus its
own source-identity facts. `project_measurements_to_chainage` is the only
piece of geometry this module performs itself: measuring distance-ALONG an
already-real line via `shapely`'s own `line.project()`, never fitting or
smoothing a new centreline.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

CANONICAL_ROUTE_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "source_route_name",
    "source_crs",
    "geometry_direction_semantics",
    "source_length_m",
    "geometry",
)


class InvalidAssetRouteError(RuntimeError):
    """The real source route geometry could not be resolved to one continuous line."""


def resolve_route_linestring(geometry: Any) -> LineString:
    """A real source route delivered as a single-row (Multi)LineString, merged into one
    continuous LineString if needed -- never a fitted/invented centreline (Section 6)."""

    if isinstance(geometry, LineString):
        return geometry
    if isinstance(geometry, MultiLineString):
        merged = linemerge(geometry)
        if isinstance(merged, LineString):
            return merged
        raise InvalidAssetRouteError(
            "the real source route geometry is a MultiLineString whose parts do not merge "
            "into one continuous line -- refusing to invent a connection between them"
        )
    raise InvalidAssetRouteError(f"unsupported source route geometry type: {geometry.geom_type}")


def build_canonical_asset_route(
    *,
    route: LineString,
    asset_id: str,
    source_route_name: str,
    source_crs: str,
    geometry_direction_semantics: str,
) -> gpd.GeoDataFrame:
    """Section 6: one-row canonical route GeoDataFrame, preserving source asset ID, source
    route name, and an explicit statement of what the geometry's own start->end direction
    means (never assumed from geographic position)."""

    record = {
        "asset_id": asset_id,
        "source_route_name": source_route_name,
        "source_crs": source_crs,
        "geometry_direction_semantics": geometry_direction_semantics,
        "source_length_m": float(route.length),
        "geometry": route,
    }
    return gpd.GeoDataFrame(
        [record], geometry="geometry", crs=source_crs, columns=list(CANONICAL_ROUTE_COLUMNS)
    )


def write_canonical_asset_route_gpkg(
    gdf: gpd.GeoDataFrame, output_path: Any, layer: str = "asset_route"
) -> Any:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer)
    return output_path


def project_measurements_to_chainage(route: LineString, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """`chainage_m[i] = route.project(Point(x[i], y[i]))` -- the standard, non-invented way to
    measure distance-along an ALREADY-REAL route line from a real measured position. This is
    never centreline fitting: the route geometry itself came directly from the source."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    points = gpd.points_from_xy(x, y)
    return np.array([route.project(point) for point in points], dtype=float)


def compute_kp_to_chainage_lookup(
    points_gdf: gpd.GeoDataFrame, *, kp_column: str, kp_units_to_m: float, route: LineString
) -> pd.DataFrame:
    """A real, source-derived KP(km-or-m) -> chainage_m(along the canonical route) lookup table
    built from the route-position-list's OWN per-station points -- never an assumed linear KP
    scaling. Returns columns `source_kp`, `source_kp_m`, `chainage_m`, sorted by chainage."""

    source_kp = points_gdf[kp_column].to_numpy(dtype=float)
    source_kp_m = source_kp * kp_units_to_m
    chainage_m = project_measurements_to_chainage(
        route, points_gdf.geometry.x.to_numpy(), points_gdf.geometry.y.to_numpy()
    )
    lookup = pd.DataFrame(
        {"source_kp": source_kp, "source_kp_m": source_kp_m, "chainage_m": chainage_m}
    )
    return lookup.sort_values("chainage_m").reset_index(drop=True)

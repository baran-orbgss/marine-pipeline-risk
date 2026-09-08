"""Generic natural-vs-anthropogenic tile validation status (MAR-022
Section 8).

Zero dependency on any specific project or dataset. Canonical NATURAL
bedform validation tiles must not be dominated by mapped anthropogenic
disturbance -- this module answers that question using ONLY exact
geometric intersection against a caller-supplied anthropogenic-context
layer (see `marine_engine.bedforms.interpretation`), never an invented
buffer distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

NATURAL_SEABED_ELIGIBLE = "NATURAL_SEABED_ELIGIBLE"
ANTHROPOGENIC_DISTURBANCE_PRESENT = "ANTHROPOGENIC_DISTURBANCE_PRESENT"
INFRASTRUCTURE_CONTEXT_INSUFFICIENT = "INFRASTRUCTURE_CONTEXT_INSUFFICIENT"

NATURAL_BEDFORM_VALIDATION_STATUSES = frozenset(
    {
        NATURAL_SEABED_ELIGIBLE,
        ANTHROPOGENIC_DISTURBANCE_PRESENT,
        INFRASTRUCTURE_CONTEXT_INSUFFICIENT,
    }
)


@dataclass(frozen=True)
class NaturalContextAssessment:
    status: str
    intersecting_feature_count: int
    intersecting_source_layers: tuple[str, ...]
    intersecting_descriptions: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "intersecting_feature_count": self.intersecting_feature_count,
            "intersecting_source_layers": list(self.intersecting_source_layers),
            "intersecting_descriptions": list(self.intersecting_descriptions),
            "reason": self.reason,
        }


def assess_natural_bedform_validation_status(
    tile_geometry: BaseGeometry,
    *,
    anthropogenic_context_gdf: gpd.GeoDataFrame | None,
    interpretation_available: bool,
) -> NaturalContextAssessment:
    """Section 8: classifies ONE canonical tile footprint. `interpretation_
    available=False` means the operator's own interpretation package could
    not be acquired/loaded at all -- a genuine 'we do not know' state,
    always `INFRASTRUCTURE_CONTEXT_INSUFFICIENT`, never silently treated
    as eligible. When interpretation IS available, a tile with zero
    intersecting anthropogenic-context features is `NATURAL_SEABED_
    ELIGIBLE` -- the absence of any mapped disturbance within a tile that
    real interpretation coverage speaks to is itself real, positive
    evidence, not merely a lack of information. No buffer is applied to
    `tile_geometry` or to any interpretation feature -- exact intersection
    only (Section 8: 'Do not invent huge arbitrary buffers')."""

    if not interpretation_available:
        return NaturalContextAssessment(
            status=INFRASTRUCTURE_CONTEXT_INSUFFICIENT,
            intersecting_feature_count=0,
            intersecting_source_layers=(),
            intersecting_descriptions=(),
            reason="source interpretation package could not be acquired/loaded -- infrastructure "
            "context is unknown, never assumed natural",
        )

    if anthropogenic_context_gdf is None or anthropogenic_context_gdf.empty:
        return NaturalContextAssessment(
            status=NATURAL_SEABED_ELIGIBLE,
            intersecting_feature_count=0,
            intersecting_source_layers=(),
            intersecting_descriptions=(),
            reason="source interpretation is available and contains zero anthropogenic-context "
            "features anywhere in the site -- this tile is not mapped as disturbed",
        )

    hits = anthropogenic_context_gdf[anthropogenic_context_gdf.intersects(tile_geometry)]
    if hits.empty:
        return NaturalContextAssessment(
            status=NATURAL_SEABED_ELIGIBLE,
            intersecting_feature_count=0,
            intersecting_source_layers=(),
            intersecting_descriptions=(),
            reason="source interpretation is available; zero mapped anthropogenic-context "
            "features intersect this tile's exact footprint",
        )

    return NaturalContextAssessment(
        status=ANTHROPOGENIC_DISTURBANCE_PRESENT,
        intersecting_feature_count=int(len(hits)),
        intersecting_source_layers=tuple(sorted(hits["source_layer"].unique().tolist())),
        intersecting_descriptions=tuple(sorted(hits["raw_description"].dropna().unique().tolist())),
        reason=f"{len(hits)} mapped anthropogenic-context feature(s) intersect this tile's exact "
        "footprint",
    )

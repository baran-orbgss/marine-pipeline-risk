"""Generic source-interpretation inventory and classification (MAR-022
Sections 6-7).

Zero dependency on any specific project or dataset -- the actual
descriptor vocabulary (e.g. an operator's own "Jackup location" or
"Sandwave crest" labels) is always a caller-supplied mapping, never
hardcoded here. This module never seeds a bedform DETECTOR from these
interpretations (see `marine_engine.bedforms.matching` for the
independent, non-seeded detector) -- it only classifies and inventories
what an operator's own GIS package already contains, for later use as an
`INDEPENDENT_SOURCE_INTERPRETATION_COMPARATOR` (Section 7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import pandas as pd

NATURAL_BEDFORM_INTERPRETATION = "NATURAL_BEDFORM_INTERPRETATION"
ANTHROPOGENIC_DISTURBANCE_CONTEXT = "ANTHROPOGENIC_DISTURBANCE_CONTEXT"
UNCLASSIFIED_INTERPRETATION_FEATURE = "UNCLASSIFIED_INTERPRETATION_FEATURE"

INTERPRETATION_CATEGORIES = frozenset(
    {
        NATURAL_BEDFORM_INTERPRETATION,
        ANTHROPOGENIC_DISTURBANCE_CONTEXT,
        UNCLASSIFIED_INTERPRETATION_FEATURE,
    }
)


def _normalize_descriptor(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


@dataclass(frozen=True)
class InterpretationLayer:
    """One caller-loaded GIS layer plus which of its own columns holds the
    human-readable feature-class description used for classification."""

    layer_name: str
    gdf: gpd.GeoDataFrame
    description_column: str


def build_feature_inventory(layers: list[InterpretationLayer]) -> pd.DataFrame:
    """Section 6: 'Record all feature classes and attributes. Do NOT
    assume any class exists before inspection.' A purely descriptive
    inventory -- every column of every real layer, every distinct value
    actually observed, and its count. Builds no classification or
    exclusion decision; see `classify_features` for that."""

    rows: list[dict[str, Any]] = []
    for layer in layers:
        gdf = layer.gdf
        for column in gdf.columns:
            if column == gdf.geometry.name:
                continue
            value_counts = gdf[column].value_counts(dropna=False)
            for value, count in value_counts.items():
                # `value` may be a string, a number, or a nodata sentinel depending on which real
                # attribute column it came from -- a single parquet column cannot mix those types
                # (a real, demonstrated `pyarrow.lib.ArrowTypeError` on this exact heterogeneous
                # inventory). Stringifying every non-null value keeps this purely descriptive
                # inventory fully readable while making the column type-homogeneous.
                is_null = isinstance(value, float) and pd.isna(value)
                rows.append(
                    {
                        "layer_name": layer.layer_name,
                        "attribute": column,
                        "value": None if is_null else str(value),
                        "count": int(count),
                        "feature_count_in_layer": len(gdf),
                        "geometry_types_in_layer": ",".join(
                            sorted(str(t) for t in gdf.geom_type.dropna().unique())
                        ),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "layer_name",
            "attribute",
            "value",
            "count",
            "feature_count_in_layer",
            "geometry_types_in_layer",
        ],
    )


def classify_features(
    layers: list[InterpretationLayer],
    *,
    category_by_normalized_descriptor: dict[str, str],
) -> gpd.GeoDataFrame:
    """Section 7/8: concatenates every layer's features into one long-form
    GeoDataFrame carrying `source_layer`, `raw_description`, and
    `interpretation_category`. `category_by_normalized_descriptor` maps a
    normalized (lower-cased, stripped) description string to one of
    `INTERPRETATION_CATEGORIES` -- any real value NOT present in that
    mapping is `UNCLASSIFIED_INTERPRETATION_FEATURE`, NEVER silently
    assumed natural or anthropogenic (Section 6's inspection-first
    principle applies to classification too)."""

    bad_categories = set(category_by_normalized_descriptor.values()) - INTERPRETATION_CATEGORIES
    if bad_categories:
        raise ValueError(f"unknown interpretation categories in mapping: {bad_categories}")

    frames: list[gpd.GeoDataFrame] = []
    for layer in layers:
        gdf = layer.gdf
        if gdf.empty:
            continue
        raw = gdf[layer.description_column]
        normalized = raw.map(_normalize_descriptor)
        category = normalized.map(
            lambda n: category_by_normalized_descriptor.get(n, UNCLASSIFIED_INTERPRETATION_FEATURE)
        )
        frame = gpd.GeoDataFrame(
            {
                "source_layer": layer.layer_name,
                "raw_description": raw.astype(object).where(raw.notna(), None),
                "interpretation_category": category,
                "geometry": gdf.geometry,
            },
            geometry="geometry",
            crs=gdf.crs,
        )
        frames.append(frame)

    if not frames:
        return gpd.GeoDataFrame(
            columns=["source_layer", "raw_description", "interpretation_category", "geometry"],
            geometry="geometry",
        )

    crs = frames[0].crs
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)
    return combined


def extract_category(classified: gpd.GeoDataFrame, category: str) -> gpd.GeoDataFrame:
    if category not in INTERPRETATION_CATEGORIES:
        raise ValueError(f"unknown interpretation category: {category!r}")
    if classified.empty:
        return classified
    return classified[classified["interpretation_category"] == category].copy()


SOURCE_BEDFORM_COMPARATOR_NOT_AVAILABLE = "SOURCE_BEDFORM_COMPARATOR_NOT_AVAILABLE"
SOURCE_BEDFORM_COMPARATOR_AVAILABLE = "SOURCE_BEDFORM_COMPARATOR_AVAILABLE"


def _line_azimuth_deg(geometry) -> float | None:
    """Undirected (0-180) azimuth of a LineString/MultiLineString's own
    start->end chord -- the same 0=north/90=east convention used
    throughout this project. A real source shapefile row can carry a
    missing/None geometry (confirmed: the real LinearFeatures layer has
    exactly this) -- that row simply has no orientation to report,
    never an error."""

    if geometry is None:
        return None
    geoms = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
    longest = max(geoms, key=lambda g: g.length)
    coords = list(longest.coords)
    if len(coords) < 2:
        return None
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dx, dy)) % 180.0


def compare_detected_crests_to_interpretation(
    detected_crests: gpd.GeoDataFrame,
    natural_interpretation: gpd.GeoDataFrame,
    *,
    proximity_threshold_m: float = 50.0,
) -> tuple[str, pd.DataFrame]:
    """Section 18: descriptive-only comparison of this engine's own
    independently-detected crests against an operator's own natural-
    bedform interpretation (e.g. mapped sand-wave crests) -- spatial
    proximity and orientation agreement, NEVER an accuracy score.
    `detected_crests` must carry `x`/`y`/`crest_azimuth_deg` columns.
    Returns `SOURCE_BEDFORM_COMPARATOR_NOT_AVAILABLE` when no relevant
    interpretation exists (Section 18: 'is correct', not an error)."""

    if detected_crests.empty or natural_interpretation.empty:
        return SOURCE_BEDFORM_COMPARATOR_NOT_AVAILABLE, pd.DataFrame(
            columns=[
                "detected_point_id",
                "nearest_distance_m",
                "within_proximity_threshold",
                "interpretation_azimuth_deg",
                "orientation_difference_deg",
            ]
        )

    rows: list[dict[str, Any]] = []
    interp_azimuths = natural_interpretation.geometry.map(_line_azimuth_deg)
    for _, point_row in detected_crests.iterrows():
        point = point_row.geometry
        distances = natural_interpretation.geometry.distance(point)
        nearest_idx = distances.idxmin()
        nearest_distance = float(distances.loc[nearest_idx])
        interp_azimuth = interp_azimuths.loc[nearest_idx]
        orientation_diff = (
            min(
                abs(point_row["crest_azimuth_deg"] - interp_azimuth) % 180.0,
                180.0 - (abs(point_row["crest_azimuth_deg"] - interp_azimuth) % 180.0),
            )
            if interp_azimuth is not None
            else None
        )
        rows.append(
            {
                "detected_point_id": point_row.get("point_id"),
                "nearest_distance_m": nearest_distance,
                "within_proximity_threshold": nearest_distance <= proximity_threshold_m,
                "interpretation_azimuth_deg": interp_azimuth,
                "orientation_difference_deg": orientation_diff,
            }
        )

    return SOURCE_BEDFORM_COMPARATOR_AVAILABLE, pd.DataFrame(rows)

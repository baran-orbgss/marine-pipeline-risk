"""Generic real observed-scour-evidence ingestion (MAR-023 Track B, Sections 9-13).

Scientifically separate from Track A (`susceptibility.py`)
------------------------------------------------------------
This module ingests and classifies an operator's own GIS interpretation of
seabed survey targets. It never feeds the pipeline scour-onset physics
engine, never seeds a detector, and is never used to validate or tune
`scour_onset.py`/`susceptibility.py` unless the asset physics and required
engineering inputs genuinely match (Section 12) -- which a windfarm
integrity survey covering monopiles, cables, and protection generally does
not, for a subsea PIPELINE scour-onset model.

Reuses the generic MAR-022 inventory builder unchanged; classification is
its own local implementation (Section 2's "do not duplicate" discipline
applies to the PHYSICS equations, not this bookkeeping)
-----------------------------------------------------------------------------
`build_feature_inventory` in `marine_engine.bedforms.interpretation` is
purely descriptive (every column, every distinct value, a count -- no
category validation at all), so it is dataset-and-domain agnostic and is
imported directly. `classify_features`/`extract_category` in that same
module, however, validate against `bedforms.interpretation
.INTERPRETATION_CATEGORIES` -- a frozenset hardcoded to the THREE bedform-
specific category strings. Reusing them here would either reject every one
of this module's own (differently-named) scour category constants, or
force scour evidence into bedform-shaped categories that do not mean
anything for a windfarm target survey (a "Cable" feature is not a "natural
bedform"). `classify_scour_features`/`extract_scour_category` below mirror
that function's structure exactly, validating against this module's own
`SCOUR_EVIDENCE_CATEGORIES` instead -- a deliberate, small, non-duplicative
local implementation, not a second copy of any scientific equation.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from marine_engine.bedforms.interpretation import (
    InterpretationLayer,
    build_feature_inventory,
)


def _normalize_descriptor(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


__all__ = [
    "InterpretationLayer",
    "build_feature_inventory",
    "classify_scour_features",
    "extract_scour_category",
    "SCOUR_EVIDENCE_CATEGORIES",
    "SOURCE_INTERPRETED_OBSERVED_SCOUR_EVIDENCE",
    "SOURCE_INTERPRETED_EXPOSURE_EVIDENCE",
    "ANTHROPOGENIC_DISTURBANCE_CONTEXT",
    "ASSET_INFRASTRUCTURE_CONTEXT",
    "SEABED_OBJECT_CONTEXT",
    "UNCLASSIFIED_INTERPRETATION_FEATURE",
    "ASSET_PHYSICS_NOT_EQUIVALENT_TO_PIPELINE_SCOUR_MODEL",
    "OBSERVED_SCOUR_MORPHOMETRY_NOT_AVAILABLE_FROM_SOURCE_PACKAGE",
    "NO_EXPLICIT_SOURCE_INTERPRETED_SCOUR_FEATURE_CLASS_PRESENT",
    "build_observed_evidence_table",
    "summarize_observed_evidence",
]

# --- Section 11: feature-class vocabulary -------------------------------------------------
# A literal scour-named class is classified as SOURCE_INTERPRETED_OBSERVED_SCOUR_EVIDENCE if
# and only if the source itself uses that word -- never inferred. "Exposure" is kept as its
# own, separately labelled category rather than folded into scour evidence: a cable/asset
# exposure is a common scour-ADJACENT indicator, but the source attributes it to nothing, so
# this module does not either (Section 11's "do not infer ... unless source attribution
# states it").
SOURCE_INTERPRETED_OBSERVED_SCOUR_EVIDENCE = "SOURCE_INTERPRETED_OBSERVED_SCOUR_EVIDENCE"
SOURCE_INTERPRETED_EXPOSURE_EVIDENCE = "SOURCE_INTERPRETED_EXPOSURE_EVIDENCE"
ANTHROPOGENIC_DISTURBANCE_CONTEXT = "ANTHROPOGENIC_DISTURBANCE_CONTEXT"
ASSET_INFRASTRUCTURE_CONTEXT = "ASSET_INFRASTRUCTURE_CONTEXT"
SEABED_OBJECT_CONTEXT = "SEABED_OBJECT_CONTEXT"
UNCLASSIFIED_INTERPRETATION_FEATURE = "UNCLASSIFIED_INTERPRETATION_FEATURE"

SCOUR_EVIDENCE_CATEGORIES = frozenset(
    {
        SOURCE_INTERPRETED_OBSERVED_SCOUR_EVIDENCE,
        SOURCE_INTERPRETED_EXPOSURE_EVIDENCE,
        ANTHROPOGENIC_DISTURBANCE_CONTEXT,
        ASSET_INFRASTRUCTURE_CONTEXT,
        SEABED_OBJECT_CONTEXT,
        UNCLASSIFIED_INTERPRETATION_FEATURE,
    }
)

# --- Section 12: asset-physics separation ---------------------------------------------------
ASSET_PHYSICS_NOT_EQUIVALENT_TO_PIPELINE_SCOUR_MODEL = (
    "ASSET_PHYSICS_NOT_EQUIVALENT_TO_PIPELINE_SCOUR_MODEL"
)
ASSET_PHYSICS_DISCLAIMER = (
    "Observed evidence from monopiles, cables, and other windfarm infrastructure "
    "demonstrates operator interpreted-data ingestion, observed-scour GIS representation, and "
    "map/report integration -- it is NOT a validation dataset for the pipeline-specific Marini "
    "scour-onset model, and asset-type association is preserved exactly as source-stated, "
    "never inferred."
)

# --- Section 13: optional descriptive morphometry ---------------------------------------------
OBSERVED_SCOUR_MORPHOMETRY_NOT_AVAILABLE_FROM_SOURCE_PACKAGE = (
    "OBSERVED_SCOUR_MORPHOMETRY_NOT_AVAILABLE_FROM_SOURCE_PACKAGE"
)

# --- MAR-023A: two separate concepts, never conflated (Section 4) ---------------------------
# "An operator's interpretation package was ingested" and "that package contains an explicit
# scour feature class" are independent facts -- a survey can genuinely ingest 746 real,
# classified features and still contain zero literal scour observations (exactly what
# Sheringham 2024 demonstrates). Neither implies the other.
NO_EXPLICIT_SOURCE_INTERPRETED_SCOUR_FEATURE_CLASS_PRESENT = (
    "NO_EXPLICIT_SOURCE_INTERPRETED_SCOUR_FEATURE_CLASS_PRESENT"
)


def classify_scour_features(
    layers: list[InterpretationLayer],
    *,
    category_by_normalized_descriptor: dict[str, str],
) -> gpd.GeoDataFrame:
    """Mirrors `bedforms.interpretation.classify_features`'s structure exactly, validated
    against THIS module's own `SCOUR_EVIDENCE_CATEGORIES` -- any real value not present in
    `category_by_normalized_descriptor` is `UNCLASSIFIED_INTERPRETATION_FEATURE`, never
    silently assumed (Section 10's inspection-first principle applies to classification too)."""

    bad_categories = set(category_by_normalized_descriptor.values()) - SCOUR_EVIDENCE_CATEGORIES
    if bad_categories:
        raise ValueError(f"unknown scour evidence categories in mapping: {bad_categories}")

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


def extract_scour_category(classified: gpd.GeoDataFrame, category: str) -> gpd.GeoDataFrame:
    if category not in SCOUR_EVIDENCE_CATEGORIES:
        raise ValueError(f"unknown scour evidence category: {category!r}")
    if classified.empty:
        return classified
    return classified[classified["interpretation_category"] == category].copy()


OBSERVED_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "source_id",
    "interpretation_category",
    "raw_descriptor",
    "asset_association_source_stated",
    "survey_epoch",
    "source_length_m",
    "source_width_m",
    "source_height_m",
    "source_water_depth_m",
    "geometry",
)


def build_observed_evidence_table(
    layer: InterpretationLayer,
    *,
    category_by_normalized_descriptor: dict[str, str],
    survey_epoch: str,
    source_id_column: str,
    asset_association_column: str | None = None,
    length_column: str | None = None,
    width_column: str | None = None,
    height_column: str | None = None,
    water_depth_column: str | None = None,
) -> gpd.GeoDataFrame:
    """Section 11: preserve source descriptor, geometry, asset association if source-stated,
    survey epoch, source ID, and dimensions if source-stated -- a straight pass-through of
    whatever the source package actually provides, never a derived/computed dimension.

    Classifies via `classify_scour_features`, then reattaches `layer`'s other original
    columns (which classification itself does not preserve) by row position -- safe here
    because exactly one layer is classified and `classify_scour_features` never reorders,
    filters, or duplicates rows for a single input layer.
    """

    classified = classify_scour_features(
        [layer], category_by_normalized_descriptor=category_by_normalized_descriptor
    )
    if classified.empty:
        return gpd.GeoDataFrame(columns=list(OBSERVED_EVIDENCE_COLUMNS), geometry="geometry")

    original = layer.gdf.reset_index(drop=True)
    classified = classified.reset_index(drop=True)

    def _column_or_none(column: str | None) -> pd.Series:
        if column is None or column not in original.columns:
            return pd.Series([None] * len(classified), index=classified.index)
        return original[column].reset_index(drop=True)

    result = gpd.GeoDataFrame(
        {
            "source_id": _column_or_none(source_id_column),
            "interpretation_category": classified["interpretation_category"],
            "raw_descriptor": classified["raw_description"],
            "asset_association_source_stated": _column_or_none(asset_association_column),
            "survey_epoch": survey_epoch,
            "source_length_m": _column_or_none(length_column),
            "source_width_m": _column_or_none(width_column),
            "source_height_m": _column_or_none(height_column),
            "source_water_depth_m": _column_or_none(water_depth_column),
            "geometry": classified.geometry,
        },
        geometry="geometry",
        crs=classified.crs,
    )
    return result[list(OBSERVED_EVIDENCE_COLUMNS)]


def summarize_observed_evidence(evidence_gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Section 11/24: feature counts by category and by source-stated asset association --
    descriptive only, never a morphometric or physics claim.

    MAR-023A Section 4: `operator_interpretation_package_ingested` (did real, classified
    operator data arrive at all) and `explicit_source_interpreted_scour_evidence_present`
    (does the source literally use the word "scour" for at least one feature) are recorded as
    two INDEPENDENT booleans -- a non-empty `evidence_gdf` full of exposure/debris/cable
    context proves the first without proving the second.
    """

    if evidence_gdf.empty:
        return {
            "total_feature_count": 0,
            "count_by_category": {},
            "count_by_asset_association": {},
            "explicit_scour_feature_count": 0,
            "morphometry_status": OBSERVED_SCOUR_MORPHOMETRY_NOT_AVAILABLE_FROM_SOURCE_PACKAGE,
            "operator_interpretation_package_ingested": False,
            "explicit_source_interpreted_scour_evidence_present": False,
            "explicit_scour_absence_reason": (
                NO_EXPLICIT_SOURCE_INTERPRETED_SCOUR_FEATURE_CLASS_PRESENT
            ),
        }

    count_by_category = {
        str(k): int(v) for k, v in evidence_gdf["interpretation_category"].value_counts().items()
    }
    asset_series = evidence_gdf["asset_association_source_stated"]
    count_by_asset_association = {
        str(k): int(v) for k, v in asset_series.value_counts(dropna=False).items()
    }
    explicit_scour_count = count_by_category.get(SOURCE_INTERPRETED_OBSERVED_SCOUR_EVIDENCE, 0)
    explicit_scour_present = explicit_scour_count > 0

    return {
        "total_feature_count": int(len(evidence_gdf)),
        "count_by_category": count_by_category,
        "count_by_asset_association": count_by_asset_association,
        "explicit_scour_feature_count": explicit_scour_count,
        # No literal scour-morphology feature class means there is nothing scour-specific to
        # measure morphometry FOR -- reported explicitly rather than measuring dimensions of
        # a different (e.g. exposure/debris) feature class and calling it scour morphometry.
        "morphometry_status": (
            OBSERVED_SCOUR_MORPHOMETRY_NOT_AVAILABLE_FROM_SOURCE_PACKAGE
            if not explicit_scour_present
            else None
        ),
        # A real, non-empty, classified table is itself proof that operator data was ingested
        # -- independent of what any of it turned out to be classified as.
        "operator_interpretation_package_ingested": True,
        "explicit_source_interpreted_scour_evidence_present": explicit_scour_present,
        "explicit_scour_absence_reason": (
            None
            if explicit_scour_present
            else NO_EXPLICIT_SOURCE_INTERPRETED_SCOUR_FEATURE_CLASS_PRESENT
        ),
    }

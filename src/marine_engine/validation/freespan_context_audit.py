"""Positive-only freespan context audit + evidence-resolution diagnosis (MAR-015).

An ANALYTICAL EVIDENCE AUDIT, never model validation (Section 1)
------------------------------------------------------------------------
Asks whether MAR-007/008/010/011A/012/013/014's independently-developed
context variables show any obvious spatial distinction at the 8 official
2018 PL854/PL855 corridor freespan events -- and what evidence/resolution
gaps currently prevent a defensible predictive model. This module never
creates a susceptibility score, probability, classifier, ROC/AUC metric, or
risk score; every statistic here is descriptive only.

Positive-only, corridor-level, never a supervised label (Section 2-3)
-------------------------------------------------------------------------------
The 8 events are `CORRIDOR_LEVEL_OBSERVED_FREESPAN_EVENTS` --
`asset_scope=PL854_PL855_PIGGYBACK_CORRIDOR` and
`individual_line_attribution=UNRESOLVED` (MAR-014A/C), never rewritten to a
PL854-only or PL855-only positive label. A route section with no tabulated
2018 event is `NO_TABULATED_2018_EVENT_IN_THIS_SUPPORT_SECTION` -- never
NO_FREESPAN/NEGATIVE/STABLE/SAFE, and this module never fabricates a formal
negative-label dataset (`negative_labels_created = False` always).

Event independence (Section 4)
------------------------------------
MAR-010/011A/012/013/014 all resolve the PL854 route into the SAME 14
hydro-pair sections (empirically confirmed: identical chainage boundaries
across all five segment tables) -- a route section's own model variable is
ONE independent spatial support value, not one value per event sharing
that section. The 8 events occupy only a handful of these 14 sections;
every audit here is computed at the SECTION level, never double-counted
per event.
"""

import sys
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import pandas as pd

SCIENTIFIC_ROLE = "POSITIVE_ONLY_FREESPAN_CONTEXT_AUDIT"
OBSERVATION_ROLE = "CORRIDOR_LEVEL_OBSERVED_FREESPAN_EVENTS"

NO_EVENT_STATUS = "NO_TABULATED_2018_EVENT_IN_THIS_SUPPORT_SECTION"
EVENT_PRESENT_STATUS = "TABULATED_2018_EVENT_PRESENT"
NON_EVENT_BACKGROUND_LABEL = "NON_EVENT_CONTAINING_ROUTE_BACKGROUND"

# --- Section 6: temporal status vocabulary ----------------------------------------------
CONTEMPORANEOUS_WITH_2018 = "CONTEMPORANEOUS_WITH_2018_OBSERVATION"
LONG_TERM_INCLUDES_2018 = "LONG_TERM_CONTEXT_INCLUDES_2018"
POSTDATES_2018 = "POSTDATES_2018_OBSERVATION"
PREDATES_2018 = "PREDATES_2018_OBSERVATION"
REGIONAL_NOT_SINGLE_EPOCH = "REGIONAL_CONTEXT_NOT_SINGLE_SURVEY_EPOCH"
VARIES_BY_SAMPLE = "VARIES_BY_ACTUAL_SAMPLE_YEAR"

# --- Section 15: evidence-readiness descriptor vocabulary -------------------------------
DESCRIPTIVE_CONTEXT_ONLY = "DESCRIPTIVE_CONTEXT_ONLY"
TEMPORALLY_MISMATCHED = "TEMPORALLY_MISMATCHED"
SPATIALLY_COARSE = "SPATIALLY_COARSE_RELATIVE_TO_EVENT_SCALE"
LEGACY_REGIONAL_CONTEXT = "LEGACY_REGIONAL_CONTEXT"
SPARSE_POINT_EVIDENCE = "SPARSE_POINT_EVIDENCE"
SPATIALLY_UNIFORM = "SPATIALLY_UNIFORM_AT_CURRENT_SUPPORT"
LOW_SPATIAL_CARDINALITY = "LOW_SPATIAL_CARDINALITY_AT_CURRENT_SUPPORT"

OBSERVED_SPAN_LENGTH_RANGE_M = (0.2, 23.0)
PIPELINE_LENGTH_APPROX_M = 23500.0
CURRENT_WAVE_SUPPORT_APPROX_M = "~1.5-2 km"
MORPHOLOGY_SOURCE_NOMINAL_RESOLUTION_M = 115.0
MORPHOLOGY_ANALYSIS_GRID_SPACING_M = 100.0


@dataclass(frozen=True)
class AuditFeature:
    """One independently-audited model/context variable's provenance."""

    key: str
    label: str
    column: str
    is_categorical: bool
    temporal_status: str
    temporal_note: str
    spatial_support_m: float | None
    spatial_support_note: str
    readiness_descriptors: tuple[str, ...]
    site_specific: bool
    continuous_along_route: bool


# The 8 families Section 19's audit-table figure names explicitly, plus one
# additional point-evidence family (PSA D50) carried in the event/section
# tables and the readiness diagnostic, but never forced into that 8-row figure.
FEATURE_REGISTRY: tuple[AuditFeature, ...] = (
    AuditFeature(
        key="current_p95",
        label="Current p95 (MAR-010)",
        column="current_reference_speed_p95_m_s",
        is_categorical=False,
        temporal_status=POSTDATES_2018,
        temporal_note=(
            "MAR-010 primary current forcing interval (2024-2026) postdates the 2018 survey."
        ),
        spatial_support_m=1500.0,
        spatial_support_note="~1.5 km native current model grid cell.",
        readiness_descriptors=(SPATIALLY_COARSE, TEMPORALLY_MISMATCHED),
        site_specific=False,
        continuous_along_route=True,
    ),
    AuditFeature(
        key="wave_orbital_p95",
        label="Wave orbital RMS p95 (MAR-011A)",
        column="orbital_rms_p95_m_s",
        is_categorical=False,
        temporal_status=LONG_TERM_INCLUDES_2018,
        temporal_note=(
            "MAR-011A wave reanalysis statistics (1980-2026) are a long-term "
            "climatology that includes 2018."
        ),
        spatial_support_m=1750.0,
        spatial_support_note=(
            "~1.5-2 km native wave model grid cell (1.9 +/- 0.4 km lon x 1.5 km lat)."
        ),
        readiness_descriptors=(SPATIALLY_COARSE,),
        site_specific=False,
        continuous_along_route=True,
    ),
    AuditFeature(
        key="combined_shear_p95",
        label="Combined bed shear p95 max (MAR-012)",
        column="tau_max_p95_sensitivity_max_pa",
        is_categorical=False,
        temporal_status=POSTDATES_2018,
        temporal_note=(
            "MAR-012 combined wave-current overlap (2024-2026, contemporaneous "
            "current-wave overlap) postdates 2018."
        ),
        spatial_support_m=1500.0,
        spatial_support_note="Same paired hydrodynamic (current+wave) support as MAR-010/011A.",
        readiness_descriptors=(SPATIALLY_COARSE, TEMPORALLY_MISMATCHED),
        site_specific=False,
        continuous_along_route=True,
    ),
    AuditFeature(
        key="mobility_capacity",
        label="Largest passing D50, p95 (MAR-013)",
        column="largest_tested_d50_with_p95_mobility_ratio_ge_1_mm",
        is_categorical=False,
        temporal_status=POSTDATES_2018,
        temporal_note="Inherits MAR-012's current-wave overlap window; postdates 2018.",
        spatial_support_m=1500.0,
        spatial_support_note="Same hydrodynamic support as MAR-012.",
        readiness_descriptors=(SPATIALLY_COARSE, TEMPORALLY_MISMATCHED),
        site_specific=False,
        continuous_along_route=True,
    ),
    AuditFeature(
        key="mar014_embedment_class",
        label="p95 required embedment upper class (MAR-014)",
        column="p95_required_embedment_upper_class",
        is_categorical=True,
        temporal_status=POSTDATES_2018,
        temporal_note="Inherits MAR-012's current-wave overlap window; postdates 2018.",
        spatial_support_m=1500.0,
        spatial_support_note="Same hydrodynamic support as MAR-012.",
        readiness_descriptors=(SPATIALLY_UNIFORM,),
        site_specific=False,
        continuous_along_route=True,
    ),
    AuditFeature(
        key="local_relief",
        label="Local relief, 1000 m radius (MAR-007)",
        column="local_relief_1000m_median_m",
        is_categorical=False,
        temporal_status=PREDATES_2018,
        temporal_note=(
            "Underlying EMODnet source bathymetry class acquisition years 1991-1992 predate 2018."
        ),
        spatial_support_m=MORPHOLOGY_ANALYSIS_GRID_SPACING_M,
        spatial_support_note=(
            f"{MORPHOLOGY_ANALYSIS_GRID_SPACING_M:.0f} m analysis grid derived from "
            f"~{MORPHOLOGY_SOURCE_NOMINAL_RESOLUTION_M:.0f} m nominal source-class bathymetry."
        ),
        readiness_descriptors=(LEGACY_REGIONAL_CONTEXT, TEMPORALLY_MISMATCHED),
        site_specific=False,
        continuous_along_route=True,
    ),
    AuditFeature(
        key="slope",
        label="Slope, 500 m radius (MAR-007)",
        column="slope_500m_median_deg",
        is_categorical=False,
        temporal_status=PREDATES_2018,
        temporal_note=(
            "Underlying EMODnet source bathymetry class acquisition years 1991-1992 predate 2018."
        ),
        spatial_support_m=MORPHOLOGY_ANALYSIS_GRID_SPACING_M,
        spatial_support_note=(
            f"{MORPHOLOGY_ANALYSIS_GRID_SPACING_M:.0f} m analysis grid derived from "
            f"~{MORPHOLOGY_SOURCE_NOMINAL_RESOLUTION_M:.0f} m nominal source-class bathymetry."
        ),
        readiness_descriptors=(LEGACY_REGIONAL_CONTEXT, TEMPORALLY_MISMATCHED),
        site_specific=False,
        continuous_along_route=True,
    ),
    AuditFeature(
        key="regional_folk_class",
        label="Regional mapped Folk class (BGS 250K)",
        column="mapped_250k_folk_class",
        is_categorical=True,
        temporal_status=REGIONAL_NOT_SINGLE_EPOCH,
        temporal_note=(
            "BGS Seabed Sediments 250k is regional mapped context, not "
            "survey-synchronous route truth."
        ),
        spatial_support_m=None,
        spatial_support_note="Regional 1:250,000 mapping polygons.",
        readiness_descriptors=(LEGACY_REGIONAL_CONTEXT,),
        site_specific=False,
        continuous_along_route=True,
    ),
    AuditFeature(
        key="psa_d50",
        label="Nearest valid PSA D50 (MAR-008)",
        column="nearest_valid_psa_d50_mm",
        is_categorical=False,
        temporal_status=VARIES_BY_SAMPLE,
        temporal_note=(
            "Quantitative PSA sample years vary by observation; preserved "
            "individually, never assumed 2018."
        ),
        spatial_support_m=None,
        spatial_support_note="Point observation.",
        readiness_descriptors=(SPARSE_POINT_EVIDENCE,),
        site_specific=True,
        continuous_along_route=False,
    ),
)

FEATURE_BY_KEY = {feature.key: feature for feature in FEATURE_REGISTRY}
# The exact 8 families Section 19's figure names -- `psa_d50` is carried
# everywhere else but deliberately excluded from that one figure.
AUDIT_TABLE_FEATURE_KEYS = tuple(f.key for f in FEATURE_REGISTRY if f.key != "psa_d50")


# --- Section 4: merge the 5 independent segment tables on their SHARED section grid ----


def merge_all_segment_tables(
    *,
    current_segments_gdf: gpd.GeoDataFrame,
    wave_segments_gdf: gpd.GeoDataFrame,
    combined_segments_gdf: gpd.GeoDataFrame,
    mobility_segments_gdf: gpd.GeoDataFrame,
    scour_segments_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """One row per hydro-pair section (14 for the real PL854 route), carrying
    every FEATURE_REGISTRY column.

    Joined on `segment_id` -- safe ONLY because MAR-010/011A/012/013/014's
    segment tables are empirically confirmed to share IDENTICAL chainage
    boundaries (the real PL854 current and wave model grids coincide), never
    assumed for an arbitrary route. A hydro-pair section's own value is ONE
    independent spatial support value, never duplicated per event sharing it
    (Section 4).
    """

    base = combined_segments_gdf[
        [
            "segment_id",
            "start_chainage_m",
            "end_chainage_m",
            "kp_start",
            "kp_end",
            "hydro_pair_id",
            "tau_max_p95_sensitivity_min_pa",
            "tau_max_p95_sensitivity_max_pa",
            "tau_max_p95_sensitivity_width_pa",
        ]
    ].copy()
    base["section_length_m"] = base["end_chainage_m"] - base["start_chainage_m"]

    base = base.merge(
        current_segments_gdf[
            [
                "segment_id",
                "current_node_id",
                "current_reference_speed_p95_m_s",
                "source_grid_nominal_resolution_m",
            ]
        ],
        on="segment_id",
        how="left",
    )
    base = base.merge(
        wave_segments_gdf[["segment_id", "wave_node_id", "orbital_rms_p95_m_s"]],
        on="segment_id",
        how="left",
    )
    base = base.merge(
        mobility_segments_gdf[
            [
                "segment_id",
                "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm",
                "mapped_250k_folk_class",
                "nearest_valid_psa_d50_mm",
                "nearest_valid_psa_distance_m",
                "nearest_valid_psa_sample_year",
            ]
        ],
        on="segment_id",
        how="left",
    )
    base = base.merge(
        scour_segments_gdf[
            [
                "segment_id",
                "p95_required_embedment_lower_class",
                "p95_required_embedment_upper_class",
                "slope_500m_median_deg",
                "slope_1000m_median_deg",
                "tpi_1000m_median_m",
                "local_relief_1000m_median_m",
                "terrain_std_1000m_median_m",
            ]
        ],
        on="segment_id",
        how="left",
    )
    return base


# --- Section 10: section-level audit table ----------------------------------------------

SECTION_CONTEXT_COLUMNS = (
    "hydro_pair_id",
    "segment_id",
    "start_chainage_m",
    "end_chainage_m",
    "kp_start",
    "kp_end",
    "section_length_m",
    "observed_2018_event_count",
    "observed_2018_total_length_m",
    "observed_2018_max_length_m",
    "observed_2018_max_height_m",
    "observation_status",
    # Not individually registered as their own FEATURE_REGISTRY family, but
    # needed alongside "combined_shear_p95" (tau_max_p95_sensitivity_max_pa)
    # to draw the full sensitivity envelope (Section 18 Panel A).
    "tau_max_p95_sensitivity_min_pa",
    "tau_max_p95_sensitivity_width_pa",
) + tuple(feature.column for feature in FEATURE_REGISTRY)


def build_section_level_context_table(
    merged_segments_df: pd.DataFrame, segment_event_counts_df: pd.DataFrame
) -> pd.DataFrame:
    """One row per hydro-pair section (Section 10). `observation_status` is
    `TABULATED_2018_EVENT_PRESENT` or `NO_TABULATED_2018_EVENT_IN_THIS_SUPPORT_
    SECTION` -- never a negative/control label (Section 3)."""

    df = merged_segments_df.merge(segment_event_counts_df, on="hydro_pair_id", how="left")
    df["any_2018_freespan"] = df["any_2018_freespan"].fillna(False)
    df["observation_status"] = df["any_2018_freespan"].map(
        {True: EVENT_PRESENT_STATUS, False: NO_EVENT_STATUS}
    )
    df = df.rename(
        columns={
            "freespan_2018_count": "observed_2018_event_count",
            "freespan_2018_total_length_m": "observed_2018_total_length_m",
            "freespan_2018_max_length_m": "observed_2018_max_length_m",
            "freespan_2018_max_height_m": "observed_2018_max_height_m",
        }
    )
    df["observed_2018_event_count"] = df["observed_2018_event_count"].fillna(0).astype(int)
    return df[list(SECTION_CONTEXT_COLUMNS)]


# --- Section 9: event-level audit table (containing-section lookup) --------------------


def _find_containing_section(mid_chainage_m: float, section_df: pd.DataFrame) -> pd.Series:
    within = section_df[
        (section_df["start_chainage_m"] <= mid_chainage_m)
        & (mid_chainage_m < section_df["end_chainage_m"])
    ]
    if not within.empty:
        return within.iloc[0]
    return section_df.sort_values("end_chainage_m").iloc[-1]


def _empirical_percentile(value: Any, all_values: list) -> float | None:
    """Section 13: `route_section_empirical_percentile_context` -- a DESCRIPTIVE
    rank among the (real: 14) support sections, never a probability/risk/hazard
    percentile. With only 14 sections this is deliberately coarse-grained."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    valid = [v for v in all_values if pd.notna(v)]
    if len(valid) <= 1:
        return None
    rank = sum(1 for v in valid if v <= value)
    return 100.0 * (rank - 1) / (len(valid) - 1)


def build_event_level_context_table(
    events_2018_df: pd.DataFrame,
    section_df: pd.DataFrame,
    temporal_relationship_df: pd.DataFrame,
    all_events_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per official 2018 event (Section 9) -- observation identity,
    hydrodynamic support, every FEATURE_REGISTRY variable plus its temporal
    status / support-scale-to-event-length ratio / route percentile context,
    and (Section 16) any UNAMBIGUOUS source-stated temporal relationship. No
    score column anywhere.
    """

    previous_length_by_id = all_events_df.set_index("event_id")["source_length_m"].to_dict()

    records = []
    for _, event in events_2018_df.iterrows():
        section = _find_containing_section(event["canonical_mid_chainage_m"], section_df)
        record: dict[str, Any] = {
            "event_id": event["event_id"],
            "canonical_kp_min": event["canonical_kp_min"],
            "canonical_kp_max": event["canonical_kp_max"],
            "canonical_mid_kp": event["canonical_mid_kp"],
            "source_length_m": event["source_length_m"],
            "source_height_m": event["source_height_m"],
            "source_comment": event.get("source_comment"),
            "asset_scope": event["asset_scope"],
            "individual_line_attribution": event["individual_line_attribution"],
            "observation_role": OBSERVATION_ROLE,
            "hydro_pair_id": section["hydro_pair_id"],
            "current_node_id": section.get("current_node_id"),
            "wave_node_id": section.get("wave_node_id"),
            "hydrodynamic_support_nominal_resolution_m": section.get(
                "source_grid_nominal_resolution_m"
            ),
        }

        for feature in FEATURE_REGISTRY:
            value = section.get(feature.column)
            record[feature.key] = value
            record[f"{feature.key}_temporal_status"] = feature.temporal_status
            record[f"{feature.key}_spatial_support_m"] = feature.spatial_support_m
            if (
                feature.spatial_support_m is not None
                and pd.notna(event["source_length_m"])
                and event["source_length_m"] > 0
            ):
                record[f"{feature.key}_support_scale_to_event_length_ratio"] = (
                    feature.spatial_support_m / event["source_length_m"]
                )
            else:
                record[f"{feature.key}_support_scale_to_event_length_ratio"] = None
            if not feature.is_categorical:
                record[f"{feature.key}_route_section_empirical_percentile_context"] = (
                    _empirical_percentile(value, section_df[feature.column].tolist())
                )

        # Section 16: only an UNAMBIGUOUS source-stated relationship, never
        # inferred cross-survey matching.
        related = temporal_relationship_df[
            (temporal_relationship_df["event_id_b"] == event["event_id"])
            & temporal_relationship_df["event_id_a"].notna()
        ]
        length_change_rows = related[related["relationship_type"] == "SOURCE_STATED_LENGTH_CHANGE"]
        chosen = (
            length_change_rows.iloc[0]
            if not length_change_rows.empty
            else (related.iloc[0] if not related.empty else None)
        )
        if chosen is not None:
            record["source_stated_previous_survey_year"] = chosen["survey_year_a"]
            record["source_stated_previous_event_id"] = chosen["event_id_a"]
            record["source_stated_previous_length_m"] = previous_length_by_id.get(
                chosen["event_id_a"]
            )
            record["source_stated_2018_length_m"] = event["source_length_m"]
            record["source_stated_length_change_statement"] = chosen["source_statement"]
        else:
            record["source_stated_previous_survey_year"] = None
            record["source_stated_previous_event_id"] = None
            record["source_stated_previous_length_m"] = None
            record["source_stated_2018_length_m"] = event["source_length_m"]
            record["source_stated_length_change_statement"] = None

        records.append(record)
    return pd.DataFrame(records)


# --- Section 4: event-independence summary -----------------------------------------------


def compute_event_independence_summary(section_df: pd.DataFrame) -> dict[str, Any]:
    """The 8 observed events occupy a HANDFUL of independent hydro-pair sections
    (Section 4) -- never one independent spatial support value per event."""

    event_sections = section_df[section_df["observation_status"] == EVENT_PRESENT_STATUS]
    return {
        "observed_event_count": int(section_df["observed_2018_event_count"].sum()),
        "independent_hydrodynamic_support_section_count": int(len(event_sections)),
        "total_route_support_section_count": int(len(section_df)),
        "events_per_support_section": {
            str(row["hydro_pair_id"]): int(row["observed_2018_event_count"])
            for _, row in event_sections.iterrows()
        },
    }


# --- Sections 11-12: descriptive contrast + range-overlap diagnostic -------------------


def _numeric_stats(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna()
    if clean.empty:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(len(clean)),
        "min": float(clean.min()),
        "median": float(clean.median()),
        "max": float(clean.max()),
    }


def compute_descriptive_contrast(section_df: pd.DataFrame, feature_key: str) -> dict[str, Any]:
    """Section 11-12: event-containing vs `NON_EVENT_CONTAINING_ROUTE_BACKGROUND`
    sections -- count/min/median/max and a raw true/false range-overlap flag
    only. Never a p-value, significance test, odds ratio, or effect size.

    Callable on any numeric-valued column, including `mobility_capacity`
    (a D50 mm value drawn from a small fixed discrete set) -- Section 14
    explicitly wants that feature audited BOTH ways, not gated on the
    registry's `is_categorical` flag.
    """

    feature = FEATURE_BY_KEY[feature_key]
    event_df = section_df[section_df["observation_status"] == EVENT_PRESENT_STATUS]
    background_df = section_df[section_df["observation_status"] == NO_EVENT_STATUS]

    event_stats = _numeric_stats(event_df[feature.column])
    background_stats = _numeric_stats(background_df[feature.column])
    route_values = section_df[feature.column].dropna()
    route_median = float(route_values.median()) if not route_values.empty else None

    absolute_median_difference = None
    if event_stats["median"] is not None and background_stats["median"] is not None:
        absolute_median_difference = abs(event_stats["median"] - background_stats["median"])

    range_overlap = None
    if event_stats["min"] is not None and background_stats["min"] is not None:
        range_overlap = not (
            event_stats["max"] < background_stats["min"]
            or background_stats["max"] < event_stats["min"]
        )

    return {
        "feature_key": feature_key,
        "label": feature.label,
        "event_containing_sections": event_stats,
        NON_EVENT_BACKGROUND_LABEL.lower(): background_stats,
        "route_median": route_median,
        "event_section_median": event_stats["median"],
        "absolute_median_difference": absolute_median_difference,
        "event_range": [event_stats["min"], event_stats["max"]],
        "background_range": [background_stats["min"], background_stats["max"]],
        "range_overlap": range_overlap,
    }


# --- Section 14: categorical / low-cardinality feature audit ---------------------------


def compute_categorical_audit(section_df: pd.DataFrame, feature_key: str) -> dict[str, Any]:
    """Section 14: honest cardinality audit for MAR-013/014-style discrete
    features. `SPATIALLY_UNIFORM_AT_CURRENT_SUPPORT` for a route-uniform
    feature (MAR-014's 0.03D is expected here); `LOW_SPATIAL_CARDINALITY_AT_
    CURRENT_SUPPORT` for exactly two route classes. Never an artificial
    continuous contrast."""

    feature = FEATURE_BY_KEY[feature_key]
    values = section_df[feature.column]
    distinct_values = sorted(str(v) for v in values.dropna().unique())
    sections_per_value = {v: int((values.astype(str) == v).sum()) for v in distinct_values}

    event_df = section_df[section_df["observation_status"] == EVENT_PRESENT_STATUS]
    event_values = event_df[feature.column].astype(str)
    event_sections_per_value = {v: int((event_values == v).sum()) for v in distinct_values}

    if len(distinct_values) <= 1:
        classification = SPATIALLY_UNIFORM
    elif len(distinct_values) == 2:
        classification = LOW_SPATIAL_CARDINALITY
    else:
        classification = None

    return {
        "feature_key": feature_key,
        "label": feature.label,
        "route_distinct_value_count": len(distinct_values),
        "route_distinct_values": distinct_values,
        "sections_per_value": sections_per_value,
        "event_containing_sections_per_value": event_sections_per_value,
        "classification": classification,
    }


# --- Section 20: literal, data-derived scientific interpretation -----------------------

_TOP_QUARTILE_PERCENTILE_THRESHOLD = 75.0


def answer_spatial_discrimination_question(section_df: pd.DataFrame, feature_key: str) -> str:
    """One literal, data-derived statement (Section 20) -- e.g. "NO -- observed
    events occur across the X-Y percentile range." or "NO SPATIAL
    DISCRIMINATION -- feature is route-uniform." Never a model-performance
    score; a plain description of where the events actually fall."""

    feature = FEATURE_BY_KEY[feature_key]
    event_df = section_df[section_df["observation_status"] == EVENT_PRESENT_STATUS]

    distinct_values = section_df[feature.column].dropna().nunique()
    if distinct_values <= 1:
        return f"NO SPATIAL DISCRIMINATION -- {feature.label} is route-uniform at current support."
    if distinct_values == 2:
        audit = compute_categorical_audit(section_df, feature_key)
        return (
            f"LOW SPATIAL CARDINALITY -- {feature.label} has only 2 route classes "
            f"({audit['route_distinct_values']}); event-containing sections show "
            f"{audit['event_containing_sections_per_value']}."
        )

    all_values = section_df[feature.column].dropna().tolist()
    event_values = event_df[feature.column].dropna()
    if event_values.empty:
        return f"NO DATA -- {feature.label} is missing at every event-containing section."

    event_percentiles = [
        p for p in (_empirical_percentile(v, all_values) for v in event_values) if p is not None
    ]
    if not event_percentiles:
        return f"NO DATA -- {feature.label} could not be ranked (insufficient route values)."

    min_p, max_p = min(event_percentiles), max(event_percentiles)
    if min_p >= _TOP_QUARTILE_PERCENTILE_THRESHOLD:
        return (
            f"YES -- observed events occur only in the {min_p:.0f}-{max_p:.0f} percentile "
            f"range (top quartile) of {feature.label}."
        )
    return (
        f"NO -- observed events occur across the {min_p:.0f}-{max_p:.0f} "
        f"percentile range of {feature.label}."
    )


def answer_all_interpretation_questions(section_df: pd.DataFrame) -> dict[str, str]:
    """Section 20 questions A-F, answered literally from the real audit data."""

    return {
        "A_highest_current_sections_only": answer_spatial_discrimination_question(
            section_df, "current_p95"
        ),
        "B_highest_wave_sections_only": answer_spatial_discrimination_question(
            section_df, "wave_orbital_p95"
        ),
        "C_highest_combined_stress_sections_only": answer_spatial_discrimination_question(
            section_df, "combined_shear_p95"
        ),
        "D_mobility_capacity_uniquely_identifies_event_sections": (
            answer_spatial_discrimination_question(section_df, "mobility_capacity")
        ),
        "E_scour_onset_class_spatially_distinguishes_event_sections": (
            answer_spatial_discrimination_question(section_df, "mar014_embedment_class")
        ),
        "F_morphology_shows_obvious_separation": (
            answer_spatial_discrimination_question(section_df, "local_relief")
            + " | "
            + answer_spatial_discrimination_question(section_df, "slope")
        ),
    }


# --- Section 21: resolution-gap interpretation ------------------------------------------


def compute_resolution_gap_statements(events_2018_df: pd.DataFrame) -> list[str]:
    """Section 21: actual ratios, plainly worded -- "the model has hydrodynamic
    forcing context for that location, NOT N m hydrodynamic resolution."""

    statements = []
    for _, event in events_2018_df.iterrows():
        length_m = event["source_length_m"]
        if not pd.notna(length_m) or length_m <= 0:
            continue
        hydro_ratio = 1500.0 / length_m
        morphology_ratio = MORPHOLOGY_ANALYSIS_GRID_SPACING_M / length_m
        statements.append(
            f"{event['event_id']} ({length_m:.2f} m observed span): the ~1500 m current/wave "
            f"model cell is {hydro_ratio:.0f}x larger than the observed span -- the model has "
            f"hydrodynamic forcing CONTEXT for this location, NOT {length_m:.2f} m hydrodynamic "
            f"resolution. The ~{MORPHOLOGY_ANALYSIS_GRID_SPACING_M:.0f} m morphology grid is "
            f"{morphology_ratio:.0f}x larger than the observed span."
        )
    return statements


# --- Section 22: demonstrated data gaps (non-ranked) ------------------------------------


def compute_demonstrated_data_gaps(
    *, event_independence: dict[str, Any], categorical_audits: list[dict[str, Any]]
) -> list[str]:
    """A non-ranked list of gaps the audit itself demonstrates -- never an
    arbitrary priority weighting, never a gap the audit does not support."""

    gaps = [
        "2018-contemporaneous near-bed current forcing: MAR-010/012/013/014 all postdate the "
        "2018 survey (2024-2026); no current/combined-shear/mobility/scour-onset feature audited "
        "here is contemporaneous with the observed events.",
        "Pipeline-scale/high-resolution bathymetry or seabed morphology: MAR-007's morphology "
        f"is derived from ~{MORPHOLOGY_SOURCE_NOMINAL_RESOLUTION_M:.0f} m nominal source-class "
        f"bathymetry (1991-1992), analysed on a {MORPHOLOGY_ANALYSIS_GRID_SPACING_M:.0f} m grid -- "
        "far coarser than the ~0.2-23 m observed span scale.",
        "Observed route embedment/burial profile: no continuous, route-wide embedment/burial "
        "observation exists to compare against MAR-014's screening output.",
        "Continuous quantitative sediment properties: PSA D50 is a sparse point observation "
        "(MAR-008); BGS 250K Folk class is regional mapped context, not survey-synchronous route "
        "truth -- neither is a continuous, route-wide quantitative property.",
        "PL854-vs-PL855 event attribution: MAR-014C found no NSTA line-specific cross-source "
        "match; individual_line_attribution remains UNRESOLVED for all 8 events.",
    ]
    if event_independence["independent_hydrodynamic_support_section_count"] <= 5:
        gaps.append(
            "Sample size: only "
            f"{event_independence['independent_hydrodynamic_support_section_count']} independent "
            "hydrodynamic support sections contain an observed event, out of "
            f"{event_independence['total_route_support_section_count']} total -- too few "
            "independent spatial samples to support any descriptive contrast beyond what is "
            "reported here."
        )
    for audit in categorical_audits:
        if audit["classification"] == SPATIALLY_UNIFORM:
            gaps.append(
                f"{audit['label']} is spatially uniform at current support -- it cannot "
                "distinguish event from non-event sections at all, by construction."
            )
    return gaps


# --- Section 15: evidence-readiness diagnostic -------------------------------------------


def build_evidence_readiness(
    categorical_audits_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Section 15: independent flags per feature family -- never a numeric score."""

    features = {}
    for feature in FEATURE_REGISTRY:
        entry = {
            "contemporaneous_with_2018": feature.temporal_status == CONTEMPORANEOUS_WITH_2018,
            "temporal_status": feature.temporal_status,
            "temporal_note": feature.temporal_note,
            "spatial_support_m": feature.spatial_support_m,
            "spatial_support_note": feature.spatial_support_note,
            "support_comparable_to_observed_span_scale": (
                feature.spatial_support_m is not None
                and feature.spatial_support_m <= OBSERVED_SPAN_LENGTH_RANGE_M[1] * 2.0
            ),
            "site_specific": feature.site_specific,
            "continuous_along_route": feature.continuous_along_route,
            "observed_vs_model_independence": True,
            "known_major_limitation": feature.spatial_support_note,
            "readiness_descriptors": list(feature.readiness_descriptors),
        }
        cardinality_audit = categorical_audits_by_key.get(feature.key)
        if cardinality_audit is not None and cardinality_audit["classification"] is not None:
            descriptors = set(entry["readiness_descriptors"])
            descriptors.add(cardinality_audit["classification"])
            entry["readiness_descriptors"] = sorted(descriptors)
        features[feature.key] = entry

    return {"scientific_role": SCIENTIFIC_ROLE, "features": features}


# --- Section 25: metadata ------------------------------------------------------------------


def build_audit_metadata(
    *,
    event_independence: dict[str, Any],
    demonstrated_gaps: list[str],
    interpretation_answers: dict[str, str],
) -> dict[str, Any]:
    """Every Section 25 required field -- explicit `False` flags for every
    forbidden output type, so a reader never has to infer their absence."""

    return {
        "scientific_role": SCIENTIFIC_ROLE,
        "observation_scope": "PL854_PL855_PIGGYBACK_CORRIDOR",
        "observation_role": OBSERVATION_ROLE,
        "individual_line_attribution_unresolved": True,
        "negative_labels_created": False,
        "model_validation_performed": False,
        "classifier_fitted": False,
        "score_created": False,
        "observed_event_count": event_independence["observed_event_count"],
        "independent_hydrodynamic_support_section_count": event_independence[
            "independent_hydrodynamic_support_section_count"
        ],
        "total_route_support_section_count": event_independence[
            "total_route_support_section_count"
        ],
        "temporal_provenance_by_feature": {
            feature.key: {"temporal_status": feature.temporal_status, "note": feature.temporal_note}
            for feature in FEATURE_REGISTRY
        },
        "spatial_support_by_feature": {
            feature.key: {
                "spatial_support_m": feature.spatial_support_m,
                "note": feature.spatial_support_note,
            }
            for feature in FEATURE_REGISTRY
        },
        "feature_audit_semantics": (
            "Every feature is audited independently and descriptively; features are never "
            "fused into a combined index or score."
        ),
        "source_stated_temporal_relationships_used_only_descriptively": True,
        "current_model_forcing_postdates_observation": True,
        "legacy_morphology_predates_observation": True,
        "route_section_empirical_percentile_context_note": (
            "With only "
            f"{event_independence['total_route_support_section_count']} support sections, "
            "route_section_empirical_percentile_context is a deliberately coarse-grained "
            "descriptive rank -- never a probability, risk, or hazard percentile."
        ),
        "interpretation_answers": interpretation_answers,
        "limitations": [
            "The observed 2018 freespan events are positive-only, corridor-level condition "
            "evidence -- never PL854-specific supervised labels.",
            "No formal negative-label dataset exists or was created; survey completeness does "
            "not establish ground truth for the absence of a freespan.",
            "All hydrodynamic/sediment/morphology context variables are independently computed "
            "and were never fitted or validated against the observed events.",
            "A 1500 m hydrodynamic model cell provides forcing CONTEXT at an observed span's "
            "location, never resolution AT the observed span's own ~0.2-23 m scale.",
        ],
        "demonstrated_data_gaps": demonstrated_gaps,
    }


# --- Report printer (Section 29) -----------------------------------------------------------


def print_audit_report(
    *,
    event_independence: dict[str, Any],
    contrasts: list[dict[str, Any]],
    categorical_audits: list[dict[str, Any]],
    interpretation_answers: dict[str, str],
    readiness: dict[str, Any],
    demonstrated_gaps: list[str],
    outputs: dict[str, Any],
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854/PL855 Positive-Only Freespan Context Audit (MAR-015) ===", ""]

    lines.append("## Observation support")
    lines.append(f"  Official 2018 event count: {event_independence['observed_event_count']}")
    lines.append(
        "  Independent hydrodynamic support sections containing an event: "
        f"{event_independence['independent_hydrodynamic_support_section_count']} of "
        f"{event_independence['total_route_support_section_count']}"
    )
    for hydro_pair_id, count in event_independence["events_per_support_section"].items():
        lines.append(f"    {hydro_pair_id}: {count} event(s)")
    lines.append("")

    lines.append("## Descriptive contrast (event-containing vs non-event-containing sections)")
    for contrast in contrasts:
        lines.append(f"  {contrast['label']}:")
        lines.append(f"    event-containing: {contrast['event_containing_sections']}")
        lines.append(
            f"    {NON_EVENT_BACKGROUND_LABEL}: {contrast[NON_EVENT_BACKGROUND_LABEL.lower()]}"
        )
        lines.append(
            f"    route median={contrast['route_median']}, "
            f"absolute median difference={contrast['absolute_median_difference']}, "
            f"range overlap={contrast['range_overlap']}"
        )
    lines.append("")

    lines.append("## Categorical / low-cardinality audit")
    for audit in categorical_audits:
        lines.append(
            f"  {audit['label']}: {audit['route_distinct_value_count']} distinct route value(s) "
            f"{audit['route_distinct_values']} -- classification={audit['classification']}"
        )
        lines.append(
            "    event-containing sections per value: "
            f"{audit['event_containing_sections_per_value']}"
        )
    lines.append("")

    lines.append("## Questions A-F (Section 20)")
    for key, answer in interpretation_answers.items():
        lines.append(f"  {key}: {answer}")
    lines.append("")

    lines.append("## Evidence readiness")
    for key, entry in readiness["features"].items():
        lines.append(f"  {key}: {entry['readiness_descriptors']}")
    lines.append("")

    lines.append("## Demonstrated data gaps (non-ranked)")
    for gap in demonstrated_gaps:
        lines.append(f"  - {gap}")
    lines.append("")

    lines.append("## Outputs")
    for key, value in outputs.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    lines.append(
        "THE 2018 EVENTS ARE POSITIVE-ONLY CORRIDOR-LEVEL CONDITION EVIDENCE, NOT "
        "PL854-SPECIFIC SUPERVISED LABELS."
    )
    lines.append(
        "NO FREESPAN SUSCEPTIBILITY SCORE, CLASSIFIER, PROBABILITY OR MODEL ACCURACY METRIC "
        "HAS BEEN CREATED."
    )
    lines.append(
        "THE PURPOSE OF MAR-015 IS TO DETERMINE WHAT THE CURRENT EVIDENCE CAN AND CANNOT "
        "SUPPORT BEFORE BUILDING A PREDICTIVE MODEL."
    )

    print("\n".join(lines), file=file)

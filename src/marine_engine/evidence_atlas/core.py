"""PL854 engineering evidence atlas: section evidence table + GIS layers (MAR-018).

Map-first packaging, not new science
-------------------------------------
Every value assembled here is READ from an already-accepted MAR-007/010/011A/
012/013/014/014A/014B/014C/015/016 output and, at most, renamed/joined/relabelled
for presentation. No equation, threshold, or statistic is recomputed. The 14
hydro-pair support sections are a pre-existing shared grid (MAR-010/011A/012/
013/014 all resolve to the identical `segment_id`/`hydro_pair_id` boundaries,
confirmed by `validation.freespan_context_audit`) -- this module only verifies
that fact (via `validate="one_to_one"` merges) rather than assuming it.

MAR-017 analog outputs are never read by this module: Section 27 of the ticket
makes them report-context only, never a PL854 scientific-field dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from marine_engine.validation import highres_seabed_survey_inventory

SCIENTIFIC_ROLE = "PL854_ENGINEERING_EVIDENCE_ATLAS"

# --- Section 3: evidence-type vocabulary --------------------------------------------------

EVIDENCE_TYPE_AUTHORITATIVE_GEOMETRY = "AUTHORITATIVE_GEOMETRY"
EVIDENCE_TYPE_OFFICIAL_OBSERVED_CONDITION = "OFFICIAL_OBSERVED_CONDITION"
EVIDENCE_TYPE_PHYSICS_BASED_MODEL_OUTPUT = "PHYSICS_BASED_MODEL_OUTPUT"
EVIDENCE_TYPE_EMPIRICAL_ENGINEERING_SCREENING = "EMPIRICAL_ENGINEERING_SCREENING"
EVIDENCE_TYPE_PRIMARY_OBSERVATIONAL_SEDIMENT_EVIDENCE = "PRIMARY_OBSERVATIONAL_SEDIMENT_EVIDENCE"
EVIDENCE_TYPE_REGIONAL_MAPPED_CONTEXT = "REGIONAL_MAPPED_CONTEXT"
EVIDENCE_TYPE_LEGACY_REGIONAL_MORPHOLOGY_CONTEXT = "LEGACY_REGIONAL_MORPHOLOGY_CONTEXT"
EVIDENCE_TYPE_DATA_AVAILABILITY_METADATA = "DATA_AVAILABILITY_METADATA"
EVIDENCE_TYPE_KNOWN_DATA_GAP = "KNOWN_DATA_GAP"

EVIDENCE_TYPES = frozenset(
    {
        EVIDENCE_TYPE_AUTHORITATIVE_GEOMETRY,
        EVIDENCE_TYPE_OFFICIAL_OBSERVED_CONDITION,
        EVIDENCE_TYPE_PHYSICS_BASED_MODEL_OUTPUT,
        EVIDENCE_TYPE_EMPIRICAL_ENGINEERING_SCREENING,
        EVIDENCE_TYPE_PRIMARY_OBSERVATIONAL_SEDIMENT_EVIDENCE,
        EVIDENCE_TYPE_REGIONAL_MAPPED_CONTEXT,
        EVIDENCE_TYPE_LEGACY_REGIONAL_MORPHOLOGY_CONTEXT,
        EVIDENCE_TYPE_DATA_AVAILABILITY_METADATA,
        EVIDENCE_TYPE_KNOWN_DATA_GAP,
    }
)

# Forbidden vocabulary (Section 2/6/20): a fused score, confidence, or risk value must never
# appear as a column name or a status value anywhere in this package.
FORBIDDEN_TERM_FRAGMENTS = (
    "risk_score",
    "susceptibility",
    "confidence_score",
    "probability",
    "hotspot",
    "priority_score",
    "safety",
)

# --- Section 6: observed-condition status vocabulary (never NEGATIVE/SAFE/STABLE) --------

OBSERVED_EVENT_PRESENT_STATUS = "TABULATED_2018_CORRIDOR_EVENT_PRESENT"
NO_OBSERVED_EVENT_STATUS = "NO_TABULATED_2018_CORRIDOR_EVENT_IN_SECTION"

GEOMETRY_SUPPORT_SEMANTICS = (
    "14 combined current+wave hydro-pair support sections tiling the full canonical route "
    "(0-23,480.67 m) with no gaps or overlaps; section boundaries follow hydrodynamic "
    "source-grid-cell proximity (MAR-012), not equal-KP divisions."
)

SECTION_EVIDENCE_COLUMNS = (
    "pipeline_id",
    "segment_id",
    "hydro_pair_id",
    "start_chainage_m",
    "end_chainage_m",
    "kp_start",
    "kp_end",
    "geometry_support_semantics",
    "observed_2018_freespan_count",
    "observed_2018_freespan_total_length_m",
    "observed_2018_freespan_max_length_m",
    "observed_2018_freespan_max_height_m",
    "observed_condition_status",
    "current_reference_p95_m_s",
    "wave_orbital_rms_p95_m_s",
    "combined_tau_max_p95_lower_pa",
    "combined_tau_max_p95_upper_pa",
    "combined_tau_max_p95_width_pa",
    "mobility_capacity_p95_d50_mm",
    "mobility_ratio_p95_d50_500um",
    "mobility_ratio_p95_d50_1000um",
    "p95_required_embedment_lower_class",
    "p95_required_embedment_upper_class",
    "slope_500m_median_deg",
    "slope_1000m_median_deg",
    "tpi_1000m_median_m",
    "local_relief_1000m_median_m",
    "terrain_std_1000m_median_m",
    "mapped_250k_folk_class",
    "nearest_valid_psa_id",
    "nearest_valid_psa_d50_mm",
    "nearest_valid_psa_distance_to_pipeline_m",
    "nearest_valid_psa_sample_year",
)


class MissingUpstreamEvidenceError(Exception):
    """Raised when an accepted upstream MAR-007/010-016 output is missing."""


# --- Section 27: upstream provenance/integrity --------------------------------------------


def required_upstream_paths(study_dir: Path) -> dict[str, Path]:
    """Every accepted upstream output MAR-018 reads. MAR-017 analog outputs are
    deliberately absent: Section 27 makes them report-context only, never a
    PL854 scientific-field dependency."""

    return {
        "route_pipeline_gpkg": study_dir / "pipeline.gpkg",
        "chainage_25m_gpkg": study_dir / "chainage_25m.gpkg",
        "MAR-007_chainage_regional_morphology": (
            study_dir / "morphology" / "chainage_regional_morphology.parquet"
        ),
        "MAR-010_current_reference_segments": study_dir
        / "metocean"
        / "current_reference_segments.gpkg",
        "MAR-011A_wave_orbital_reference_segments": (
            study_dir / "metocean" / "wave_orbital_reference_segments.gpkg"
        ),
        "MAR-012_combined_bed_shear_segments": study_dir
        / "metocean"
        / "combined_bed_shear_segments.gpkg",
        "MAR-012_combined_bed_shear_stats": study_dir
        / "metocean"
        / "combined_bed_shear_stats.parquet",
        "MAR-013_noncohesive_mobility_capacity_segments": (
            study_dir / "sediment" / "noncohesive_mobility_capacity_segments.gpkg"
        ),
        "MAR-013_observed_d50_context": study_dir / "sediment" / "observed_d50_context.parquet",
        "MAR-014_scour_onset_embedment_segments": study_dir
        / "scour"
        / "scour_onset_embedment_segments.gpkg",
        "MAR-014A_anglia_freespan_spatial_evidence_gpkg": (
            study_dir / "freespan_evidence" / "anglia_freespan_spatial_evidence.gpkg"
        ),
        "MAR-014A_freespan_segment_event_counts_2018": (
            study_dir / "freespan_evidence" / "freespan_segment_event_counts_2018.parquet"
        ),
        "MAR-014A_condition_benchmark": (
            study_dir / "pipeline_condition" / "anglia_2018_condition_benchmark.json"
        ),
        "MAR-014B_freespan_temporal_relationship_evidence": (
            study_dir
            / "pipeline_condition"
            / "anglia_freespan_temporal_relationship_evidence.parquet"
        ),
        "MAR-014C_nsta_freespan_registry_gpkg": (
            study_dir / "pipeline_condition" / "nsta_pl854_pl855_freespan_registry.gpkg"
        ),
        "MAR-015_freespan_section_context_audit": (
            study_dir / "validation" / "freespan_section_context_audit.parquet"
        ),
        "MAR-015_freespan_context_audit_metadata": (
            study_dir / "validation" / "freespan_context_audit_metadata.json"
        ),
        "MAR-016_high_resolution_survey_inventory": (
            study_dir / "seabed_data" / "high_resolution_survey_inventory.parquet"
        ),
        "MAR-016_seabed_data_access_gap": study_dir / "seabed_data" / "seabed_data_access_gap.json",
    }


def check_upstream_integrity(study_dir: Path) -> list[str]:
    """`f"{name}: {path}"` for every required upstream file that does not exist.

    Empty list means every accepted dependency is present."""

    return [
        f"{name}: {path}"
        for name, path in required_upstream_paths(study_dir).items()
        if not path.exists()
    ]


# --- Section 6: canonical section evidence table -------------------------------------------


def build_section_evidence_table(study_dir: Path) -> pd.DataFrame:
    """One row per real hydro-pair support section, joined ONLY on `segment_id`/
    `hydro_pair_id` -- every merge uses `validate="one_to_one"` so a silent
    schema drift in any upstream ticket fails loudly here rather than
    corrupting the atlas."""

    combined = gpd.read_file(
        study_dir / "metocean" / "combined_bed_shear_segments.gpkg",
        layer="combined_bed_shear_segments",
    )
    current = gpd.read_file(
        study_dir / "metocean" / "current_reference_segments.gpkg",
        layer="current_reference_segments",
    )
    wave = gpd.read_file(
        study_dir / "metocean" / "wave_orbital_reference_segments.gpkg",
        layer="wave_orbital_reference_segments",
    )
    mobility = gpd.read_file(
        study_dir / "sediment" / "noncohesive_mobility_capacity_segments.gpkg",
        layer="noncohesive_mobility_capacity_segments",
    )
    scour = gpd.read_file(
        study_dir / "scour" / "scour_onset_embedment_segments.gpkg",
        layer="scour_onset_embedment_segments",
    )
    freespan_counts = pd.read_parquet(
        study_dir / "freespan_evidence" / "freespan_segment_event_counts_2018.parquet"
    )

    df = pd.DataFrame(combined.drop(columns="geometry"))[
        [
            "pipeline_id",
            "segment_id",
            "hydro_pair_id",
            "start_chainage_m",
            "end_chainage_m",
            "kp_start",
            "kp_end",
            "tau_max_p95_sensitivity_min_pa",
            "tau_max_p95_sensitivity_max_pa",
            "tau_max_p95_sensitivity_width_pa",
        ]
    ].rename(
        columns={
            "tau_max_p95_sensitivity_min_pa": "combined_tau_max_p95_lower_pa",
            "tau_max_p95_sensitivity_max_pa": "combined_tau_max_p95_upper_pa",
            "tau_max_p95_sensitivity_width_pa": "combined_tau_max_p95_width_pa",
        }
    )
    df["geometry_support_semantics"] = GEOMETRY_SUPPORT_SEMANTICS

    df = df.merge(
        pd.DataFrame(current[["segment_id", "current_reference_speed_p95_m_s"]]).rename(
            columns={"current_reference_speed_p95_m_s": "current_reference_p95_m_s"}
        ),
        on="segment_id",
        how="left",
        validate="one_to_one",
    )
    df = df.merge(
        pd.DataFrame(wave[["segment_id", "orbital_rms_p95_m_s"]]).rename(
            columns={"orbital_rms_p95_m_s": "wave_orbital_rms_p95_m_s"}
        ),
        on="segment_id",
        how="left",
        validate="one_to_one",
    )

    mobility_slice = pd.DataFrame(
        mobility[
            [
                "segment_id",
                "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm",
                "mobility_ratio_p95_d50_500um",
                "mobility_ratio_p95_d50_1000um",
                "mapped_250k_folk_class",
                "nearest_valid_psa_id",
                "nearest_valid_psa_d50_mm",
                "nearest_valid_psa_distance_m",
                "nearest_valid_psa_sample_year",
            ]
        ]
    ).rename(
        columns={
            "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": "mobility_capacity_p95_d50_mm",
            "nearest_valid_psa_distance_m": "nearest_valid_psa_distance_to_pipeline_m",
        }
    )
    df = df.merge(mobility_slice, on="segment_id", how="left", validate="one_to_one")

    scour_slice = pd.DataFrame(
        scour[
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
        ]
    )
    df = df.merge(scour_slice, on="segment_id", how="left", validate="one_to_one")

    freespan_slice = freespan_counts.rename(
        columns={
            "freespan_2018_count": "observed_2018_freespan_count",
            "freespan_2018_total_length_m": "observed_2018_freespan_total_length_m",
            "freespan_2018_max_length_m": "observed_2018_freespan_max_length_m",
            "freespan_2018_max_height_m": "observed_2018_freespan_max_height_m",
        }
    )[
        [
            "hydro_pair_id",
            "observed_2018_freespan_count",
            "observed_2018_freespan_total_length_m",
            "observed_2018_freespan_max_length_m",
            "observed_2018_freespan_max_height_m",
            "any_2018_freespan",
        ]
    ]
    df = df.merge(freespan_slice, on="hydro_pair_id", how="left", validate="one_to_one")
    df["observed_condition_status"] = np.where(
        df["any_2018_freespan"].fillna(False),
        OBSERVED_EVENT_PRESENT_STATUS,
        NO_OBSERVED_EVENT_STATUS,
    )
    df = df.drop(columns=["any_2018_freespan"])

    return df[list(SECTION_EVIDENCE_COLUMNS)].sort_values("segment_id").reset_index(drop=True)


# --- Section 7-8: GIS atlas layers ----------------------------------------------------------


def build_pipeline_route_layer(study_dir: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(study_dir / "pipeline.gpkg", layer="pipeline")


def build_engineering_support_sections_layer(
    study_dir: Path, section_df: pd.DataFrame
) -> gpd.GeoDataFrame:
    """The real 14-segment route geometry (MAR-012), carrying the full section
    evidence table as attributes -- one authoritative GIS layer for every
    section-level fact rather than one layer per upstream ticket."""

    combined = gpd.read_file(
        study_dir / "metocean" / "combined_bed_shear_segments.gpkg",
        layer="combined_bed_shear_segments",
    )
    geometry_only = combined[["segment_id", "geometry"]]
    merged = geometry_only.merge(section_df, on="segment_id", how="left", validate="one_to_one")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=combined.crs)


def build_observed_freespans_2018_layer(study_dir: Path) -> gpd.GeoDataFrame:
    """The TRUE route-substring geometry for exactly the 8 official 2018 events
    -- never widened, never a symbol/point standing in for the real length."""

    all_events = gpd.read_file(
        study_dir / "freespan_evidence" / "anglia_freespan_spatial_evidence.gpkg",
        layer="historical_freespans",
    )
    events_2018 = all_events[all_events["survey_year"] == 2018].reset_index(drop=True)
    return events_2018


def build_historical_freespans_layer(study_dir: Path) -> gpd.GeoDataFrame:
    """All 17 tabulated events (2012/2014/2018), reused verbatim -- the same
    accepted MAR-014A/B geometry, unmodified."""

    return gpd.read_file(
        study_dir / "freespan_evidence" / "anglia_freespan_spatial_evidence.gpkg",
        layer="historical_freespans",
    )


def build_observed_psa_d50_points_layer(
    study_dir: Path, interim_pl854_dir: Path
) -> gpd.GeoDataFrame:
    """The 5 valid-D50 PSA point observations (MAR-008): attribute values come
    from the accepted `observed_d50_context.parquet`; point geometry (simple
    already-surveyed coordinates, no science) comes from the interim PSA
    sample table since the processed tier does not itself carry geometry."""

    d50_df = pd.read_parquet(study_dir / "sediment" / "observed_d50_context.parquet")
    psa_points = gpd.read_file(interim_pl854_dir / "sediment" / "bgs_psa_observations.gpkg")
    coords = psa_points[["psa_data_id", "geometry"]]
    merged = d50_df.merge(coords, on="psa_data_id", how="left", validate="one_to_one")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=psa_points.crs)


def load_cached_bgs_survey_features(manifest_path: Path) -> list[dict[str, Any]]:
    """Offline reconstruction of every real BGS feature behind MAR-016's survey
    inventory, read entirely from already-cached raw responses recorded in
    `acquisition_manifest.json` -- no network access, no BGS re-query."""

    if not manifest_path.exists():
        return []
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    features: list[dict[str, Any]] = []
    for entry in entries:
        raw_path = Path(entry["raw_file_path"])
        if not raw_path.exists():
            continue
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        features.extend(payload.get("features", []))
    return features


def build_highres_survey_inventory_layer(
    *, manifest_path: Path, inventory_path: Path, working_crs: str
) -> gpd.GeoDataFrame:
    """Real footprint polygons for the 14 MAR-016 survey-inventory records,
    rebuilt offline (never fabricated, never re-queried) via MAR-016's own
    `build_survey_footprints_gdf`, then joined to the accepted flat inventory
    table. Returns an empty (but correctly-typed) layer if either the cached
    raw evidence or the inventory table is unavailable -- a KNOWN_DATA_GAP,
    never a fabricated footprint."""

    features = load_cached_bgs_survey_features(manifest_path)
    if not features or not inventory_path.exists():
        return gpd.GeoDataFrame(
            columns=["survey_id", "geometry"], geometry="geometry", crs=working_crs
        )

    footprints = highres_seabed_survey_inventory.build_survey_footprints_gdf(
        features, working_crs=working_crs
    )
    footprints = footprints.drop_duplicates(subset="survey_id", keep="first")
    inventory_df = pd.read_parquet(inventory_path)
    merged = footprints.merge(inventory_df, on="survey_id", how="inner", validate="one_to_one")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=footprints.crs)


def build_chainage_reference_points_layer(study_dir: Path) -> gpd.GeoDataFrame:
    """A small cartographic reference set (KP 0/5/10/15/20/terminus), each
    snapped to its nearest real 25 m chainage station -- never a fabricated
    coordinate. Distinct from the raw 941-station `chainage_points` layer,
    which stays untouched in `chainage_25m.gpkg`."""

    points = gpd.read_file(study_dir / "chainage_25m.gpkg", layer="chainage_points")
    total_length_m = float(points["chainage_m"].max())
    targets = [
        (0.0, "KP 0 (route start)"),
        (5000.0, "KP 5"),
        (10000.0, "KP 10"),
        (15000.0, "KP 15"),
        (20000.0, "KP 20"),
        (total_length_m, "Route terminus"),
    ]

    selected_indices: list[Any] = []
    reference_labels: list[str] = []
    for target_chainage_m, label in targets:
        nearest_pos = int((points["chainage_m"] - target_chainage_m).abs().to_numpy().argmin())
        idx = points.index[nearest_pos]
        if idx not in selected_indices:
            selected_indices.append(idx)
            reference_labels.append(label)

    reference_df = points.loc[selected_indices].copy()
    reference_df["reference_label"] = reference_labels
    return gpd.GeoDataFrame(reference_df, geometry="geometry", crs=points.crs).reset_index(
        drop=True
    )


ATLAS_PROJECTED_LAYER_NAMES = (
    "pipeline_route",
    "engineering_support_sections",
    "observed_freespans_2018",
    "historical_freespans_2012_2018",
    "observed_psa_d50_points",
    "highres_survey_inventory",
    "chainage_reference_points",
)


def write_evidence_atlas_gpkg(output_path: Path, layers: dict[str, gpd.GeoDataFrame]) -> Path:
    """Delete-if-exists, then one `to_file(..., layer=name)` call per non-empty
    layer -- matches the project's established multi-layer GeoPackage idiom
    (`_cmd_build_analog_sandwave_morphometry`). Never writes a layer with no
    rows (Section 23: no null geometry where geometry is required)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    for layer_name, gdf in layers.items():
        if gdf is None or gdf.empty:
            continue
        gdf.to_file(output_path, driver="GPKG", layer=layer_name)
    return output_path


def verify_projected_layers_crs(
    layers: dict[str, gpd.GeoDataFrame], *, expected_epsg: int = 32631
) -> list[str]:
    """Names of any non-empty layer whose CRS is not EPSG:`expected_epsg`."""

    mismatched = []
    for name, gdf in layers.items():
        if gdf is None or gdf.empty:
            continue
        if gdf.crs is None or gdf.crs.to_epsg() != expected_epsg:
            mismatched.append(name)
    return mismatched

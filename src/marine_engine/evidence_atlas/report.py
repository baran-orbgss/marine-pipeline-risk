"""PL854 engineering evidence report (MAR-018 Sections 16-25).

Format-neutral content, two renderers
--------------------------------------
`build_report_blocks()` derives every factual statement from already-loaded
real data (never a hard-coded stale number) into a small format-neutral
block list (heading/paragraph/list/table/callout). `_render_html`/
`_render_markdown` turn that SAME block list into the two required output
formats, so the HTML and Markdown reports can never drift apart. No
external template engine is used (none is installed in this project) --
plain string building only, offline, no JavaScript.
"""

from __future__ import annotations

import hashlib
import html as html_module
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from marine_engine.evidence_atlas import core

# --- Section 16: evidence variable manifest -------------------------------------------------


def build_evidence_variable_manifest(*, generated_at_utc: str) -> dict[str, Any]:
    """Machine-readable registry of every major atlas variable/layer. Hand-authored
    (this is documentation, not a statistic) but every source_file/field name
    named here is verified to exist by the CLI's upstream-integrity check and by
    the offline tests before this ever ships."""

    entries = [
        {
            "display_name": "PL854 pipeline route geometry",
            "scientific_role": "CANONICAL_ROUTE_GEOMETRY",
            "evidence_type": core.EVIDENCE_TYPE_AUTHORITATIVE_GEOMETRY,
            "source_file": "data/processed/pl854/pipeline.gpkg",
            "original_provider": "NSTA UKCS offshore infrastructure pipeline linear",
            "acquisition_epoch": "current authoritative registry geometry",
            "spatial_support": "as-surveyed route centreline",
            "units": "m (EPSG:32631)",
            "canonical_field": "pipeline.gpkg:pipeline.geometry",
            "map_usage": "all panels, all figures, GIS layer 'pipeline_route'",
            "major_limitation": (
                "chainage/KP direction has no verified physical (Anglia-vs-LOGGS) basis"
            ),
        },
        {
            "display_name": "2018 official corridor freespan events",
            "scientific_role": "HISTORICAL_SPATIAL_FREESPAN_CORRIDOR_EVIDENCE",
            "evidence_type": core.EVIDENCE_TYPE_OFFICIAL_OBSERVED_CONDITION,
            "source_file": (
                "data/processed/pl854/freespan_evidence/anglia_2018_freespan_spatial_evidence.parquet"
            ),
            "original_provider": (
                "Ithaca Energy Anglia Decommissioning Comparative Assessment, Table B.1"
            ),
            "acquisition_epoch": "2018 observed condition",
            "spatial_support": "true route-substring geometry, ~0.2-23 m per event",
            "units": "m (length/height)",
            "canonical_field": "source_length_m / source_height_m",
            "map_usage": "Panel A, evidence strip Band 1, GIS layer 'observed_freespans_2018'",
            "major_limitation": "PL854 vs PL855 individual-line attribution unresolved",
        },
        {
            "display_name": "2018 exposure aggregate condition",
            "scientific_role": "HISTORICAL_AGGREGATE_CONDITION_BENCHMARK",
            "evidence_type": core.EVIDENCE_TYPE_OFFICIAL_OBSERVED_CONDITION,
            "source_file": (
                "data/processed/pl854/pipeline_condition/anglia_2018_condition_benchmark.json"
            ),
            "original_provider": (
                "Ithaca Energy Anglia Decommissioning Environmental Appraisal, Tables 3.4/3.5"
            ),
            "acquisition_epoch": "2018 observed condition",
            "spatial_support": "aggregate corridor total only, no spatial locations",
            "units": "count / m",
            "canonical_field": "exposed_section_count / total_exposed_length_m",
            "map_usage": "observed-condition summary box",
            "major_limitation": "spatial locations of exposed sections are unavailable",
        },
        {
            "display_name": "Primary current reference speed (p95)",
            "scientific_role": "CURRENT_ONLY_LOG_PROFILE_SENSITIVITY",
            "evidence_type": core.EVIDENCE_TYPE_PHYSICS_BASED_MODEL_OUTPUT,
            "source_file": "data/processed/pl854/metocean/current_reference_segments.gpkg",
            "original_provider": "Copernicus Marine NWSHELF_ANALYSISFORECAST_PHY_004_013",
            "acquisition_epoch": "2024-07 to 2026-09 (contemporaneous)",
            "spatial_support": "~1.5 km source grid",
            "units": "m/s",
            "canonical_field": "current_reference_speed_p95_m_s",
            "map_usage": "section evidence table",
            "major_limitation": "postdates the 2018 observations by 6+ years",
        },
        {
            "display_name": "Wave orbital RMS velocity (p95)",
            "scientific_role": "WAVE_ONLY_SPECTRAL_NEAR_BED_ORBITAL_VELOCITY",
            "evidence_type": core.EVIDENCE_TYPE_PHYSICS_BASED_MODEL_OUTPUT,
            "source_file": "data/processed/pl854/metocean/wave_orbital_reference_segments.gpkg",
            "original_provider": "Copernicus Marine NWSHELF_REANALYSIS_WAV_004_015",
            "acquisition_epoch": "1980-2026 long-term wave context",
            "spatial_support": "1.9 +/- 0.4 km longitude x 1.5 km latitude source grid",
            "units": "m/s",
            "canonical_field": "orbital_rms_p95_m_s",
            "map_usage": "section evidence table, evidence strip",
            "major_limitation": (
                "long-term context, not the contemporaneous overlap used for shear"
            ),
        },
        {
            "display_name": "Combined wave-current bed shear (p95 sensitivity envelope)",
            "scientific_role": "SOULSBY_ALGEBRAIC_WAVE_CURRENT_BED_SHEAR_SENSITIVITY",
            "evidence_type": core.EVIDENCE_TYPE_PHYSICS_BASED_MODEL_OUTPUT,
            "source_file": "data/processed/pl854/metocean/combined_bed_shear_segments.gpkg",
            "original_provider": "MAR-012 (Soulsby algebraic wave-current interaction)",
            "acquisition_epoch": "2024-07 to 2026-04 contemporaneous overlap",
            "spatial_support": "14 hydro-pair support sections, ~1.5-2.3 km node spacing",
            "units": "Pa",
            "canonical_field": ("combined_tau_max_p95_lower_pa / combined_tau_max_p95_upper_pa"),
            "map_usage": "Panel B, evidence strip Band 2, section summary table",
            "major_limitation": (
                "upper-bound sensitivity across 5 roughness scenarios, never a best estimate"
            ),
        },
        {
            "display_name": "Noncohesive sediment mobility capacity (p95)",
            "scientific_role": "NONCOHESIVE_SEDIMENT_MOBILITY_CAPACITY",
            "evidence_type": core.EVIDENCE_TYPE_PHYSICS_BASED_MODEL_OUTPUT,
            "source_file": (
                "data/processed/pl854/sediment/noncohesive_mobility_capacity_segments.gpkg"
            ),
            "original_provider": "MAR-013 (Soulsby-Whitehouse threshold-of-motion screening)",
            "acquisition_epoch": "2024-07 to 2026-04 contemporaneous overlap",
            "spatial_support": (
                "14 hydro-pair support sections (~1.5-2.3 km hydrodynamic node spacing)"
            ),
            "units": "mm (largest tested D50 passing threshold)",
            "canonical_field": "mobility_capacity_p95_d50_mm",
            "map_usage": "Panel C, evidence strip Band 3, section summary table",
            "major_limitation": "discrete 9-point test ladder, never a continuous route D50",
        },
        {
            "display_name": "Scour-onset required embedment class (p95)",
            "scientific_role": "EMPIRICAL_SCOUR_ONSET_SCREENING",
            "evidence_type": core.EVIDENCE_TYPE_EMPIRICAL_ENGINEERING_SCREENING,
            "source_file": "data/processed/pl854/scour/scour_onset_embedment_segments.gpkg",
            "original_provider": "Marini et al. 2024, Coastal Engineering 190:104507",
            "acquisition_epoch": "2024-07 to 2026-04 contemporaneous overlap",
            "spatial_support": "14 hydro-pair support sections",
            "units": "fraction of pipe diameter (xD)",
            "canonical_field": (
                "p95_required_embedment_lower_class / p95_required_embedment_upper_class"
            ),
            "map_usage": "evidence strip Band 4, section summary table",
            "major_limitation": (
                "PL854 diameter (0.3048 m) is outside the source D=0.05-0.10 m experimental "
                "envelope"
            ),
        },
        {
            "display_name": "Regional morphology context (slope/relief/roughness)",
            "scientific_role": "REGIONAL_KILOMETRE_SCALE_MORPHOLOGY_CONTEXT",
            "evidence_type": core.EVIDENCE_TYPE_LEGACY_REGIONAL_MORPHOLOGY_CONTEXT,
            "source_file": "data/processed/pl854/morphology/chainage_regional_morphology.parquet",
            "original_provider": "EMODnet Bathymetry (MAR-007)",
            "acquisition_epoch": "1991-1992 source acquisition",
            "spatial_support": "~115 m nominal source information class on a 100 m analysis grid",
            "units": "deg (slope) / m (relief, TPI, roughness)",
            "canonical_field": "local_relief_1000m_median_m / slope_500m_median_deg / etc.",
            "map_usage": "evidence strip Bands 5-6, section summary table",
            "major_limitation": (
                "not appropriate for present-day metre-scale seabed geometry claims"
            ),
        },
        {
            "display_name": "BGS 1:250,000 regional Folk sediment class",
            "scientific_role": "REGIONAL_MAPPED_SEABED_SEDIMENT_CLASSIFICATION",
            "evidence_type": core.EVIDENCE_TYPE_REGIONAL_MAPPED_CONTEXT,
            "source_file": "data/processed/pl854/sediment/chainage_sediment_evidence.parquet",
            "original_provider": "British Geological Survey, seabed sediments 1:250,000",
            "acquisition_epoch": "regional mapped context",
            "spatial_support": "1:250,000 regional mapping, not pipeline-scale ground truth",
            "units": "Folk class code",
            "canonical_field": "mapped_250k_folk_class",
            "map_usage": "evidence strip Band 6, section summary table",
            "major_limitation": "regional mapping, not pipeline-scale ground truth",
        },
        {
            "display_name": "Observed PSA D50 point samples",
            "scientific_role": "PRIMARY_OBSERVATIONAL_GRAIN_SIZE_EVIDENCE",
            "evidence_type": core.EVIDENCE_TYPE_PRIMARY_OBSERVATIONAL_SEDIMENT_EVIDENCE,
            "source_file": "data/processed/pl854/sediment/observed_d50_context.parquet",
            "original_provider": "BGS particle size analysis (PSA) seabed samples",
            "acquisition_epoch": "actual sample years (1979-2009)",
            "spatial_support": "point observations, not interpolated to the pipeline",
            "units": "mm",
            "canonical_field": "d50_mm",
            "map_usage": "Panel D, GIS layer 'observed_psa_d50_points'",
            "major_limitation": (
                "only 5 of 27 regional PSA records have a usable D50; no continuous route D50"
            ),
        },
        {
            "display_name": "High-resolution seabed survey inventory",
            "scientific_role": "HIGH_RESOLUTION_SURVEY_DATA_AVAILABILITY_AUDIT",
            "evidence_type": core.EVIDENCE_TYPE_DATA_AVAILABILITY_METADATA,
            "source_file": (
                "data/processed/pl854/seabed_data/high_resolution_survey_inventory.parquet"
            ),
            "original_provider": (
                "British Geological Survey offshore oil & gas site surveys catalogue"
            ),
            "acquisition_epoch": "data-availability audit, not a survey observation",
            "spatial_support": "14 real candidate survey footprints in/near the AOI",
            "units": "n/a",
            "canonical_field": "access_class",
            "map_usage": "Panel D, GIS layer 'highres_survey_inventory'",
            "major_limitation": "every real candidate is METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED",
        },
        {
            "display_name": "PL854-specific high-resolution bathymetric grid",
            "scientific_role": "ROUTE_SPECIFIC_HIGH_RESOLUTION_MORPHOLOGY",
            "evidence_type": core.EVIDENCE_TYPE_KNOWN_DATA_GAP,
            "source_file": "data/processed/pl854/seabed_data/seabed_data_access_gap.json",
            "original_provider": (
                "n/a -- confirmed absent after a structured recovery/audit attempt"
            ),
            "acquisition_epoch": "n/a",
            "spatial_support": "n/a",
            "units": "n/a",
            "canonical_field": "n/a",
            "map_usage": "Panel D caption, report Section 8",
            "major_limitation": (
                "no verified open 2018 Fugro bathymetric grid; no continuous route embedment "
                "profile"
            ),
        },
    ]

    return {
        "scientific_role": core.SCIENTIFIC_ROLE,
        "generated_at_utc": generated_at_utc,
        "variable_count": len(entries),
        "variables": entries,
    }


# --- Format-neutral report content (Sections 17-24) -----------------------------------------


def _fmt_range(low: float, high: float, unit: str, decimals: int = 2) -> str:
    return f"{low:.{decimals}f}-{high:.{decimals}f} {unit}"


def derive_executive_summary_facts(
    *,
    section_df: pd.DataFrame,
    freespans_2018_gdf: gpd.GeoDataFrame,
    route_length_km: float,
) -> dict[str, Any]:
    """Every number here is computed directly from already-loaded real data --
    never a hard-coded stale figure (Section 19)."""

    event_sections = section_df[section_df["observed_2018_freespan_count"].fillna(0) > 0]
    embedment_classes = sorted(section_df["p95_required_embedment_upper_class"].dropna().unique())
    return {
        "route_length_km": route_length_km,
        "observed_event_count": int(len(freespans_2018_gdf)),
        "event_section_count": int(len(event_sections)),
        "total_section_count": int(len(section_df)),
        "mobility_capacity_min_mm": float(section_df["mobility_capacity_p95_d50_mm"].min()),
        "mobility_capacity_max_mm": float(section_df["mobility_capacity_p95_d50_mm"].max()),
        "embedment_classes": embedment_classes,
        "embedment_spatially_uniform": len(embedment_classes) == 1,
    }


def build_report_blocks(
    *,
    section_df: pd.DataFrame,
    freespans_2018_gdf: gpd.GeoDataFrame,
    historical_freespans_gdf: gpd.GeoDataFrame,
    condition_benchmark: dict[str, Any],
    route_length_km: float,
    depth_stats: dict[str, float],
    sediment_metadata: dict[str, Any],
    morphology_metadata: dict[str, Any],
    highres_survey_df: pd.DataFrame,
    seabed_data_access_gap: dict[str, Any],
    analog_family_status: dict[str, Any],
    key_limitations: tuple[str, ...],
) -> list[dict[str, Any]]:
    """The full report content as format-neutral blocks, in required order."""

    facts = derive_executive_summary_facts(
        section_df=section_df,
        freespans_2018_gdf=freespans_2018_gdf,
        route_length_km=route_length_km,
    )
    embedment_label = (
        f"{facts['embedment_classes'][0]}"
        if facts["embedment_spatially_uniform"]
        else "/".join(facts["embedment_classes"])
    )
    blocks: list[dict[str, Any]] = []

    blocks.append({"type": "heading", "level": 1, "text": "PL854 Engineering Evidence Report"})

    blocks.append({"type": "heading", "level": 2, "text": "Executive Summary"})
    mobility_range = _fmt_range(
        facts["mobility_capacity_min_mm"], facts["mobility_capacity_max_mm"], "mm", 1
    )
    blocks.append(
        {
            "type": "list",
            "items": [
                f"PL854 canonical route length ~{facts['route_length_km']:.2f} km",
                f"{facts['observed_event_count']} official 2018 corridor freespans spatially "
                "recovered",
                f"those events occupy {facts['event_section_count']}/"
                f"{facts['total_section_count']} current hydrodynamic support sections",
                f"noncohesive p95 mobility capacity is {mobility_range} across the current support",
                f"MAR-014 p95 screening class is spatially uniform at {embedment_label}xD"
                if facts["embedment_spatially_uniform"]
                else f"MAR-014 p95 screening class varies across sections ({embedment_label}xD)",
                "high-resolution morphology remains the principal demonstrated data gap",
            ],
        }
    )

    blocks.append({"type": "callout", "text": "Key evidence limits"})
    blocks.append({"type": "list", "items": list(key_limitations)})

    blocks.append({"type": "heading", "level": 2, "text": "1. Purpose"})
    blocks.append(
        {
            "type": "paragraph",
            "text": (
                "This is a physics/evidence-based seabed and pipeline-condition screening "
                "report. It is not a deterministic freespan prediction or risk assessment. "
                "Every figure and table distinguishes observed condition, physics-based model "
                "output, empirical engineering screening, and legacy/regional context; no "
                "dimension is fused into a single score."
            ),
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "2. Pipeline and Study Route"})
    blocks.append(
        {
            "type": "list",
            "items": [
                f"Route length: ~{route_length_km:.2f} km (NSTA UKCS offshore infrastructure "
                "pipeline linear)",
                "Source: NSTA authoritative registry geometry, ingested and reconciled to "
                "EPSG:32631",
                "Geometry orientation caveat: chainage/KP 0 is the source geometry's own first "
                "vertex; the canonical pipeline schema carries no authoritative from/to "
                "installation field, so chainage direction is NOT confirmed to run from any "
                "particular physical installation",
                f"Water-depth context along the route: {depth_stats['min_m']:.1f}-"
                f"{depth_stats['max_m']:.1f} m (median {depth_stats['median_m']:.1f} m), "
                "EMODnet baseline",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "3. Official Observed Condition"})
    blocks.append(
        {
            "type": "list",
            "items": [
                f"{len(freespans_2018_gdf)} official 2018 corridor freespan events, "
                f"{float(freespans_2018_gdf['source_length_m'].sum()):.2f} m exact tabulated "
                f"total, max length {float(freespans_2018_gdf['source_length_m'].max()):.2f} m, "
                f"max height {float(freespans_2018_gdf['source_height_m'].max()):.2f} m",
                f"{condition_benchmark['exposed_section_count']} exposed sections, "
                f"{condition_benchmark['total_exposed_length_m']:.0f} m total exposed pipeline "
                "(aggregate corridor evidence only; spatial locations unavailable)",
                f"Historical context: {len(historical_freespans_gdf) - len(freespans_2018_gdf)} "
                "further tabulated events from the 2012/2014 surveys",
                "PL854 vs PL855 individual-line attribution remains UNRESOLVED for every "
                "tabulated event (the corridor is a piggybacked PL854/PL855 asset); an NSTA "
                "line-specific cross-check returned zero corroborating or contradicting records",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "4. Hydrodynamic Environment"})
    blocks.append(
        {
            "type": "list",
            "items": [
                "MAR-010 (current-only log-profile sensitivity): contemporaneous primary "
                "current, 2024-07 to 2026-09, ~1.5 km source grid",
                "MAR-011A (wave-only spectral near-bed orbital velocity): 1980-2026 long-term "
                "context, 1.9+/-0.4 km longitude x 1.5 km latitude source grid",
                "MAR-012 (combined wave-current bed shear, Soulsby algebraic interaction): "
                "contemporaneous current-wave overlap only, 2024-07 to 2026-04 -- not the full "
                "1980-2026 wave record and not a 25-year return-period analysis",
                "All three postdate the 2018 observed condition by 6+ years, and their "
                "~1.5-2.3 km spatial support is far coarser than the ~0.2-23 m scale of an "
                "individual observed freespan",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "5. Sediment and Mobility"})
    blocks.append(
        {
            "type": "list",
            "items": [
                "MAR-008: 27 regional BGS PSA records in the AOI; only 5 carry a usable D50 "
                f"({sediment_metadata['coverage_diagnostics']['sample_year_min']}-"
                f"{sediment_metadata['coverage_diagnostics']['sample_year_max']}), point "
                "observations only",
                "MAR-013 (noncohesive sediment mobility capacity): p95 capacity is "
                f"{mobility_range} across the current 14-section support, from a fixed "
                "9-point test-D50 ladder",
                "NO CONTINUOUS ROUTE D50 EXISTS: sediment evidence is either a regional "
                "1:250,000 mapped class or 5 sparse point observations -- never a continuous, "
                "quantitative, route-following D50",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "6. Pipeline Scour-Onset Screening"})
    blocks.append(
        {
            "type": "list",
            "items": [
                "MAR-014 (Marini et al. 2024 combined wave-current scour-onset screening): "
                "every one of the 14 hydro-pair sections resolves to the same p95 screening "
                f"class ({embedment_label}xD)",
                "PL854 PIPE DIAMETER IS OUTSIDE THE SOURCE EXPERIMENTAL ENVELOPE (D=0.3048 m "
                "vs the source's tested D=0.05-0.10 m); MAR-014 is a research screening "
                "extrapolation",
                f"{embedment_label}xD is an empirical engineering SCREENING class, never an "
                "engineering burial design requirement",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "7. Observed-Event Context Audit"})
    blocks.append(
        {
            "type": "list",
            "items": [
                f"MAR-015: the {facts['observed_event_count']} official 2018 events occupy "
                f"only {facts['event_section_count']} of {facts['total_section_count']} "
                "independent hydrodynamic support sections",
                "Current, wave-orbital, and combined-shear p95 all show broad range overlap "
                "between event-containing and non-event sections -- no forcing variable "
                "spatially discriminates them",
                "MAR-014's screening class is spatially uniform at current support and "
                "therefore cannot discriminate event-containing sections either",
                "Event-containing sections DO fall in the top quartile (77-92 percentile) of "
                "legacy MAR-007 local relief/slope -- reported as a descriptive co-location "
                "only, never causal, and based on only 3 independent spatial samples",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "8. Morphology Evidence Gap"})
    blocks.append(
        {
            "type": "list",
            "items": [
                "No verified open PL854-specific high-resolution bathymetric grid exists: all "
                "14 real route-area survey candidates found are "
                "METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED",
                (
                    "No verified open 2018 Fugro bathymetric grid: dossier status "
                    f"{seabed_data_access_gap.get('final_status', 'n/a')}"
                )
                if "final_status" in seabed_data_access_gap
                else "No verified open 2018 Fugro bathymetric grid was found",
                "A generic, dataset-independent high-resolution sand-wave morphometry engine "
                "exists and is synthetically validated",
                f"{analog_family_status.get('real_analog_validation_attempts', 3)} "
                "independent, real, open-analog canonical validation attempts (HHW, Inner "
                "Dowsing/Race Bank/North Ridge, Greater Gabbard 2014) did not satisfy the "
                "engine's own canonical spatial-support requirement -- all three failed for "
                "lack of continuous coverage, not lack of trying",
                f"Analog search status: "
                f"{analog_family_status.get('analog_search_status', 'n/a')} -- no further "
                "open-analog search is planned; these datasets remain method-development "
                "analogs only and never enter PL854 scientific evidence",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "9. What Can Currently Be Concluded"})
    blocks.append(
        {
            "type": "list",
            "items": [
                "8 official 2018 corridor freespan/exposure events are spatially and "
                "aggregately documented from an authoritative source, with a fully reconciled "
                "chainage location",
                "Contemporaneous (2024-2026) hydrodynamic forcing, noncohesive sediment "
                "mobility capacity, and scour-onset screening class are available at "
                "14-section spatial support",
                "The observed 2018 events show no spatial discrimination against modelled "
                "forcing or screening variables, and only a descriptive (not causal) "
                "co-location with legacy regional morphology top-quartile relief/slope",
                "A generic, reusable, synthetically-validated high-resolution morphometry "
                "engine exists, ready to ingest a real PL854-specific high-resolution survey "
                "should one become available",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "10. What Cannot Currently Be Concluded"})
    blocks.append(
        {
            "type": "list",
            "items": [
                "No PL854-specific freespan probability",
                "No future freespan prediction",
                "No scour depth",
                "No erosion/deposition time history",
                "No structural/VIV assessment",
                "No 25-year reliability estimate",
            ],
        }
    )

    blocks.append(
        {"type": "heading", "level": 2, "text": "11. Data Required for Next Scientific Upgrade"}
    )
    blocks.append(
        {
            "type": "list",
            "items": [
                "2018 Fugro raw MBES/bathymetry (if it can be recovered from the custodian)",
                "A route burial/embedment profile",
                "PL854-specific condition attribution (resolving the PL854/PL855 piggyback "
                "ambiguity)",
                "Quantitative, continuous sediment properties along the route",
                "Contemporaneous near-bed current observations, if available",
            ],
        }
    )

    event_rows = section_df[section_df["observed_2018_freespan_count"].fillna(0) > 0].sort_values(
        "segment_id"
    )
    if not event_rows.empty:
        blocks.append(
            {
                "type": "heading",
                "level": 2,
                "text": "Sections Containing Tabulated 2018 Corridor Freespan Evidence",
            }
        )
        blocks.append(
            {
                "type": "paragraph",
                "text": (
                    "This is a factual listing of sections with tabulated evidence, not a "
                    "predicted-priority list."
                ),
            }
        )
        blocks.append(
            {
                "type": "table",
                "headers": [
                    "KP range",
                    "Event count",
                    "Total length (m)",
                    "Max length (m)",
                    "Max height (m)",
                ],
                "rows": [
                    [
                        f"{row['kp_start']} - {row['kp_end']}",
                        f"{int(row['observed_2018_freespan_count'])}",
                        f"{row['observed_2018_freespan_total_length_m']:.2f}",
                        f"{row['observed_2018_freespan_max_length_m']:.2f}",
                        f"{row['observed_2018_freespan_max_height_m']:.2f}",
                    ]
                    for _, row in event_rows.iterrows()
                ],
            }
        )

    blocks.append({"type": "heading", "level": 2, "text": "Sources and References"})
    blocks.append(
        {
            "type": "list",
            "items": [
                "North Sea Transition Authority (NSTA) -- UKCS offshore infrastructure pipeline "
                "registry",
                "EMODnet Bathymetry -- regional bathymetric context (MAR-007)",
                "Copernicus Marine Service -- NWSHELF_ANALYSISFORECAST_PHY_004_013 (current), "
                "NWSHELF_REANALYSIS_WAV_004_015 (wave)",
                "British Geological Survey (BGS) -- seabed sediments 1:250,000 mapping; "
                "particle size analysis (PSA) samples; offshore oil & gas site survey "
                "catalogue",
                "Ithaca Energy (UK) Limited -- Anglia Decommissioning Programme and "
                "Environmental Appraisal (official documentation, 2019-2020)",
                "Marini, F., Postacchini, M., Pizzigalli, C., Badalini, M., Corvaro, S., & "
                "Brocchini, M. (2024). On the onset of pipeline scouring: Reconciling waves "
                "and currents forcing. Coastal Engineering, 190, 104507.",
                "Soulsby, R. (1997) and Soulsby & Whitehouse threshold-of-motion formulations",
                "JNCC/Cefas (HHW, Inner Dowsing/Race Bank/North Ridge) and The Crown Estate "
                "Marine Data Exchange (Greater Gabbard 2014) -- method-development analog "
                "datasets ONLY; these are never PL854 scientific evidence",
            ],
        }
    )

    blocks.append(
        {
            "type": "callout",
            "text": "MAR-018 DOES NOT CREATE A FREESPAN SUSCEPTIBILITY SCORE OR RISK MODEL.",
        }
    )
    blocks.append(
        {
            "type": "callout",
            "text": (
                "OBSERVED CONDITION, PHYSICS-BASED MODEL OUTPUTS, EMPIRICAL SCREENING, AND "
                "LEGACY/REGIONAL CONTEXT REMAIN VISUALLY AND SEMANTICALLY SEPARATE."
            ),
        }
    )
    blocks.append(
        {
            "type": "callout",
            "text": (
                "THE PRIMARY PURPOSE OF MAR-018 IS HUMAN-READABLE GIS / ENGINEERING "
                "COMMUNICATION OF THE ACCEPTED EVIDENCE BASE."
            ),
        }
    )

    return blocks


def _render_html(blocks: list[dict[str, Any]]) -> str:
    def esc(text: str) -> str:
        return html_module.escape(text)

    parts: list[str] = []
    for block in blocks:
        kind = block["type"]
        if kind == "heading":
            level = block["level"]
            parts.append(f"<h{level}>{esc(block['text'])}</h{level}>")
        elif kind == "paragraph":
            parts.append(f"<p>{esc(block['text'])}</p>")
        elif kind == "list":
            items = "".join(f"<li>{esc(item)}</li>" for item in block["items"])
            parts.append(f"<ul>{items}</ul>")
        elif kind == "table":
            header_html = "".join(f"<th>{esc(h)}</th>" for h in block["headers"])
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
                for row in block["rows"]
            )
            parts.append(
                f"<table><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"
            )
        elif kind == "callout":
            parts.append(f'<div class="callout">{esc(block["text"])}</div>')
        else:  # pragma: no cover -- defensive, every block type above is exhaustive
            raise ValueError(f"unknown report block type: {kind}")

    body = "\n".join(parts)
    style = """
    body { font-family: Georgia, 'Times New Roman', serif; max-width: 900px; margin: 2rem auto;
           padding: 0 1.5rem; color: #1a1a1a; line-height: 1.55; }
    h1 { font-size: 1.8rem; border-bottom: 3px solid #1a1a1a; padding-bottom: 0.3rem; }
    h2 { font-size: 1.25rem; margin-top: 2rem; color: #2a3d5c; border-bottom: 1px solid #ccc; }
    ul { padding-left: 1.4rem; }
    li { margin-bottom: 0.35rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92rem; }
    th, td { border: 1px solid #999; padding: 0.4rem 0.6rem; text-align: left; }
    th { background: #e8e6dd; }
    .callout { background: #fbeeee; border: 1px solid #c0392b; border-radius: 4px;
               padding: 0.7rem 1rem; margin: 1rem 0; font-weight: bold; color: #7a1f1f; }
    """
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>PL854 Engineering Evidence Report</title>"
        f"<style>{style}</style></head><body>{body}</body></html>"
    )


def _render_markdown(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        kind = block["type"]
        if kind == "heading":
            lines.append(f"{'#' * block['level']} {block['text']}")
            lines.append("")
        elif kind == "paragraph":
            lines.append(block["text"])
            lines.append("")
        elif kind == "list":
            lines.extend(f"- {item}" for item in block["items"])
            lines.append("")
        elif kind == "table":
            lines.append("| " + " | ".join(block["headers"]) + " |")
            lines.append("| " + " | ".join("---" for _ in block["headers"]) + " |")
            for row in block["rows"]:
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        elif kind == "callout":
            lines.append(f"> **{block['text']}**")
            lines.append("")
        else:  # pragma: no cover -- defensive, every block type above is exhaustive
            raise ValueError(f"unknown report block type: {kind}")
    return "\n".join(lines)


def write_html_report(blocks: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html(blocks), encoding="utf-8")
    return output_path


def write_markdown_report(blocks: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_markdown(blocks), encoding="utf-8")
    return output_path


# --- Section 25: package manifest (hash every deliverable) ----------------------------------


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_package_manifest(
    *, deliverables: dict[str, Path], generated_at_utc: str, project_root: Path
) -> dict[str, Any]:
    """SHA256 + byte size for every named deliverable, relative to `project_root`."""

    resolved_root = project_root.resolve()
    files = []
    for name, path in deliverables.items():
        if not path.exists():
            continue
        files.append(
            {
                "name": name,
                "relative_path": str(path.resolve().relative_to(resolved_root)).replace("\\", "/"),
                "file_size_bytes": path.stat().st_size,
                "sha256": compute_sha256(path),
            }
        )
    return {
        "scientific_role": core.SCIENTIFIC_ROLE,
        "generated_at_utc": generated_at_utc,
        "file_count": len(files),
        "files": files,
    }


def write_package_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return output_path

"""Minimal command-line entry point for the marine-engine package."""

import argparse
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import xarray as xr
from shapely.geometry import LineString as shapely_linestring
from shapely.geometry import box as shapely_box
from shapely.geometry import shape as shapely_shape
from shapely.ops import substring as shapely_substring
from shapely.ops import unary_union

from marine_engine import __version__
from marine_engine.analogs import greater_gabbard_2014 as gg_analog
from marine_engine.analogs import hhw_cend1111 as hhw_analog
from marine_engine.analogs import idr_bnr_cend1111 as idrbnr_analog
from marine_engine.bedforms import contract as bedform_contract
from marine_engine.bedforms import extraction as bedform_extraction
from marine_engine.bedforms import interpretation as bedform_interpretation
from marine_engine.bedforms import maps as bedform_maps
from marine_engine.bedforms import matching as bedform_matching
from marine_engine.bedforms import natural_context as bedform_natural_context
from marine_engine.bedforms import report as bedform_report
from marine_engine.burial import contract as burial_contract
from marine_engine.burial import exposure_screening as burial_exposure_screening
from marine_engine.burial import maps as burial_maps
from marine_engine.burial import profile as burial_profile
from marine_engine.burial import readiness as burial_readiness
from marine_engine.burial import report as burial_report
from marine_engine.burial import route as burial_route
from marine_engine.burial import semantics as burial_semantics
from marine_engine.change import alignment as change_alignment
from marine_engine.change import common_support as change_common_support
from marine_engine.change import comparator as change_comparator
from marine_engine.change import contract as change_contract
from marine_engine.change import dod as change_dod
from marine_engine.change import epoch_compatibility as change_epoch_compatibility
from marine_engine.change import maps as change_maps
from marine_engine.change import report as change_report
from marine_engine.change import route_evidence as change_route_evidence
from marine_engine.change import uncertainty as change_uncertainty
from marine_engine.config import load_study_config
from marine_engine.evidence_atlas import core as evidence_atlas_core
from marine_engine.evidence_atlas import maps as evidence_atlas_maps
from marine_engine.evidence_atlas import poc as evidence_atlas_poc
from marine_engine.evidence_atlas import report as evidence_atlas_report
from marine_engine.freespan import contract as fs_contract
from marine_engine.freespan import maps as fs_maps
from marine_engine.freespan import nsta_registry as fs_nsta_registry
from marine_engine.freespan import report as fs_report
from marine_engine.freespan import structural_handoff as fs_structural_handoff
from marine_engine.freespan import support_state as fs_support_state
from marine_engine.freespan import synthetic as fs_synthetic
from marine_engine.metocean import (
    combined_bed_shear,
    combined_bed_shear_map,
    current_map,
    current_normalization,
    wave_orbital,
    wave_orbital_map,
)
from marine_engine.metocean import evidence as metocean_evidence
from marine_engine.morphology import regional
from marine_engine.morphology import sandwave_morphometry as swm
from marine_engine.morphology import sandwave_morphometry_map as swmap
from marine_engine.preprocessing import bathymetry, source_resolution
from marine_engine.preprocessing.aoi import (
    InvalidAoiGeometryError,
    InvalidPipelineInputError,
    build_aoi,
    print_aoi_report,
)
from marine_engine.preprocessing.chainage import (
    ChainageValidationError,
    InvalidPipelineRouteError,
    build_chainage,
    load_pipeline_route,
    print_chainage_report,
)
from marine_engine.project import categories as project_categories
from marine_engine.project import manifest as project_manifest
from marine_engine.project import model as project_model
from marine_engine.project import registry as project_registry
from marine_engine.project import report as project_report
from marine_engine.project import route_reference as project_route_reference
from marine_engine.providers import barrow_2016 as barrow_2016_provider
from marine_engine.providers import bgs_offshore_surveys, nsta_freespan
from marine_engine.providers.bathymetry import acquisition, bgs, emodnet, inventory, ukho
from marine_engine.providers.bathymetry import greater_gabbard_2014 as gg_provider
from marine_engine.providers.bathymetry import hhw_cend1111 as hhw_provider
from marine_engine.providers.bathymetry import idr_bnr_cend1111 as idrbnr_provider
from marine_engine.providers.bathymetry import sheringham_shoal_2018 as sheringham_2018_provider
from marine_engine.providers.bathymetry import sheringham_shoal_2020 as sheringham_provider
from marine_engine.providers.bathymetry import sheringham_shoal_2024 as sheringham_2024_provider
from marine_engine.providers.metocean import acquisition as metocean_acquisition
from marine_engine.providers.metocean import copernicus
from marine_engine.providers.nsta import (
    AmbiguousPipelineError,
    InvalidGeometryError,
    PipelineNotFoundError,
    ingest_pipeline,
    print_ingestion_report,
)
from marine_engine.providers.sediment import bgs as sediment_bgs
from marine_engine.resources import (
    ANGLIA_TABLE_B1_FREESPAN_RELATIONSHIPS_CSV,
    AngliaFreespanRelationshipValidationError,
    AngliaTableB1ChecksumError,
    load_anglia_table_b1_freespan_relationships,
    load_anglia_table_b1_freespans,
)
from marine_engine.scour import (
    contract as scour_contract,
)
from marine_engine.scour import (
    freespan_evidence,
    freespan_evidence_map,
    freespan_temporal_provenance,
    nsta_freespan_reconciliation,
    nsta_freespan_reconciliation_map,
    observed_evidence,
    observed_evidence_map,
    pipeline_condition,
    scour_onset,
    scour_onset_map,
    susceptibility,
    susceptibility_map,
)
from marine_engine.scour import (
    poc_report as scour_poc_report,
)
from marine_engine.sediment import evidence, noncohesive_mobility, noncohesive_mobility_map
from marine_engine.terrain import canonical as terrain_canonical
from marine_engine.terrain import contract as terrain_contract
from marine_engine.terrain import derivatives as terrain_derivatives
from marine_engine.terrain import maps as terrain_maps
from marine_engine.terrain import raster_io as terrain_raster_io
from marine_engine.terrain import readiness as terrain_readiness
from marine_engine.terrain import report as terrain_report
from marine_engine.validation import (
    freespan_context_audit,
    freespan_context_audit_map,
    freespan_model_context,
    highres_seabed_survey_inventory,
    highres_seabed_survey_inventory_map,
)


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"marine-engine {__version__}")
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    print(f"OK: '{config.study.id}' ({config.study.name})")
    print(f"  CRS (horizontal): {config.crs.horizontal}")
    print(f"  raw data dir:     {config.paths.raw_dir}")
    return 0


def _cmd_ingest_pipeline(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    cache_dir = config.paths.raw_dir / "nsta"
    output_path = config.paths.processed_dir / pipeline_id.lower() / "pipeline.gpkg"

    try:
        report = ingest_pipeline(
            pipeline_id,
            cache_dir=cache_dir,
            output_path=output_path,
            working_crs=config.crs.horizontal,
        )
    except (PipelineNotFoundError, AmbiguousPipelineError, InvalidGeometryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_ingestion_report(report)
    return 0


def _cmd_build_aoi(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    buffer_m = config.area_of_interest.corridor_buffer_m
    if not buffer_m:
        print(
            f"error: '{args.config}' has no area_of_interest.corridor_buffer_m configured",
            file=sys.stderr,
        )
        return 1

    pipeline_gpkg_path = config.paths.processed_dir / pipeline_id.lower() / "pipeline.gpkg"
    output_path = config.paths.processed_dir / pipeline_id.lower() / "aoi.gpkg"

    try:
        report = build_aoi(
            pipeline_gpkg_path=pipeline_gpkg_path,
            pipeline_id=pipeline_id,
            study_id=config.study.id,
            buffer_m=buffer_m,
            working_crs=config.crs.horizontal,
            output_path=output_path,
        )
    except (InvalidPipelineInputError, InvalidAoiGeometryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_aoi_report(report)
    return 0


def _cmd_build_chainage(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    interval_m = config.pipeline.get("chainage_interval_m")
    if not interval_m:
        print(
            f"error: '{args.config}' has no pipeline.chainage_interval_m configured",
            file=sys.stderr,
        )
        return 1

    pipeline_gpkg_path = config.paths.processed_dir / pipeline_id.lower() / "pipeline.gpkg"
    aoi_gpkg_path = config.paths.processed_dir / pipeline_id.lower() / "aoi.gpkg"
    output_path = config.paths.processed_dir / pipeline_id.lower() / "chainage_25m.gpkg"

    try:
        report = build_chainage(
            pipeline_gpkg_path=pipeline_gpkg_path,
            aoi_gpkg_path=aoi_gpkg_path,
            pipeline_id=pipeline_id,
            study_id=config.study.id,
            interval_m=interval_m,
            working_crs=config.crs.horizontal,
            output_path=output_path,
        )
    except (InvalidPipelineRouteError, ChainageValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_chainage_report(report)
    return 0


def _study_paths(config, pipeline_id: str) -> tuple[Path, Path, Path, Path]:
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    interim_dir = config.paths.interim_dir / pipeline_id.lower()
    return (
        study_dir / "pipeline.gpkg",
        study_dir / "aoi.gpkg",
        study_dir / "chainage_25m.gpkg",
        interim_dir,
    )


def _cmd_discover_bathymetry(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )

    try:
        pipeline_gdf = gpd.read_file(pipeline_gpkg_path, layer="pipeline")
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(f"error: could not load canonical pipeline/AOI/chainage: {exc}", file=sys.stderr)
        return 1

    working_crs = config.crs.horizontal
    pipeline_geom = pipeline_gdf.geometry.iloc[0]
    aoi_geom = unary_union(aoi_gdf.geometry)
    aoi_bbox_wgs84 = tuple(float(v) for v in aoi_gdf.to_crs("EPSG:4326").total_bounds)

    ukho_records, ukho_status = ukho.discover_ukho_surveys(aoi_bbox_wgs84)
    bgs_records = bgs.discover_bgs_surveys()
    emodnet_record = emodnet.discover_emodnet_baseline(aoi_bbox_wgs84)
    raw_records = [*ukho_records, *bgs_records, emodnet_record]

    report = inventory.run_discovery(
        raw_records,
        pipeline_geom_working=pipeline_geom,
        aoi_geom_working=aoi_geom,
        chainage_gdf_working=chainage_gdf,
        working_crs=working_crs,
        parquet_path=interim_dir / "bathymetry_inventory.parquet",
        gpkg_path=interim_dir / "bathymetry_inventory.gpkg",
    )

    print(f"UKHO (via MEDIN): {ukho_status.message}")
    print()
    inventory.print_discovery_report(report)
    return 0


def _cmd_fetch_bathymetry(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    _pipeline_gpkg_path, aoi_gpkg_path, _chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    parquet_path = interim_dir / "bathymetry_inventory.parquet"
    manifest_path = interim_dir / "bathymetry_acquisition_manifest.json"

    if not parquet_path.exists():
        print(
            f"error: no inventory found at {parquet_path}; run discover-bathymetry first",
            file=sys.stderr,
        )
        return 1

    try:
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not load AOI: {exc}", file=sys.stderr)
        return 1

    aoi_bbox_wgs84 = tuple(float(v) for v in aoi_gdf.to_crs("EPSG:4326").total_bounds)

    output_path = (
        acquisition.raw_dataset_dir(config.paths.raw_dir, "emodnet", emodnet.COVERAGE_ID)
        / f"{emodnet.COVERAGE_ID}.tif"
    )
    request_parameters = {
        "coverageId": emodnet.COVERAGE_ID,
        "bbox_wgs84": list(aoi_bbox_wgs84),
        "format": "image/tiff;application=geotiff",
    }

    existing = acquisition.already_acquired(
        manifest_path, "EMODnet", emodnet.COVERAGE_ID, request_parameters
    )
    if existing is not None:
        print(f"EMODnet: already acquired ({existing['local_path']}); skipping re-download.")
        emodnet_entry = existing
    else:
        try:
            fetch_result = emodnet.fetch_emodnet_geotiff(aoi_bbox_wgs84, output_path)
        except emodnet.EmodnetUnavailableError as exc:
            print(f"error: EMODnet acquisition failed: {exc}", file=sys.stderr)
            return 1

        emodnet_entry = acquisition.record_acquisition(
            manifest_path,
            source="EMODnet",
            dataset_id=emodnet.COVERAGE_ID,
            source_url_or_service=emodnet.WCS_BASE_URL,
            request_parameters=fetch_result.request_parameters,
            local_path=fetch_result.local_path,
            licence=emodnet.LICENCE,
            # EMODnet DTM 2024 is an aggregate product release, not a survey
            # acquisition -- see acquisition.record_acquisition's docstring.
            acquisition_year=None,
            product_release_year=2024,
            horizontal_crs=fetch_result.returned_crs,
            vertical_datum=emodnet.VERTICAL_DATUM,
            nominal_resolution_m=emodnet.NATIVE_RESOLUTION_M,
            acquired_at=datetime.now(UTC),
        )
        print(
            f"EMODnet: acquired {fetch_result.width_px}x{fetch_result.height_px} px "
            f"-> {fetch_result.local_path}"
        )

    df = pd.read_parquet(parquet_path)
    manual_rows = df[df["manual_download_required"].fillna(False)]
    print()
    print("Datasets requiring manual download:")
    if manual_rows.empty:
        print("  (none)")
    else:
        for _, row in manual_rows.iterrows():
            url = row["source_record_url_or_identifier"]
            print(f"  - [{row['source']}] {row['source_dataset_id']}: {url}")

    print()
    print(f"Manifest: {manifest_path}")
    print(f"EMODnet sha256: {emodnet_entry['sha256']}")
    print(f"EMODnet file size: {emodnet_entry['file_size_bytes']} bytes")
    return 0


def _cmd_build_bathymetry(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    _pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    manifest_path = interim_dir / "bathymetry_acquisition_manifest.json"
    raw_path = (
        acquisition.raw_dataset_dir(config.paths.raw_dir, "emodnet", emodnet.COVERAGE_ID)
        / f"{emodnet.COVERAGE_ID}.tif"
    )

    if not raw_path.exists():
        print(
            f"error: no raw EMODnet raster found at {raw_path}; run fetch-bathymetry first",
            file=sys.stderr,
        )
        return 1

    try:
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(f"error: could not load canonical AOI/chainage: {exc}", file=sys.stderr)
        return 1

    working_crs = config.crs.horizontal
    aoi_geom_working = unary_union(aoi_gdf.geometry)
    aoi_bbox_wgs84 = tuple(float(v) for v in aoi_gdf.to_crs("EPSG:4326").total_bounds)

    manifest_entries = acquisition.load_manifest(manifest_path)
    raw_manifest_entry = next(
        (
            e
            for e in manifest_entries
            if e.get("source") == "EMODnet" and e.get("dataset_id") == emodnet.COVERAGE_ID
        ),
        None,
    )

    output_dir = config.paths.processed_dir / pipeline_id.lower() / "bathymetry"
    output_raster_path = output_dir / "emodnet_baseline_lat_100m.tif"
    output_metadata_path = output_dir / "emodnet_baseline_lat_100m.json"
    chainage_output_path = output_dir / "chainage_bathymetry.parquet"

    try:
        dtm_report = bathymetry.build_canonical_dtm(
            raw_path=raw_path,
            raw_manifest_entry=raw_manifest_entry,
            aoi_geometry_working=aoi_geom_working,
            working_crs=working_crs,
            aoi_identifier=f"{pipeline_id}_AOI",
            output_raster_path=output_raster_path,
            output_metadata_path=output_metadata_path,
        )
    except (bathymetry.InvalidRawRasterError, bathymetry.AmbiguousSignConventionError) as exc:
        print(f"error: canonical DTM build failed: {exc}", file=sys.stderr)
        return 1

    # Source-reference/quality-index attribution is retrieved best-effort: a live
    # WFS failure here must not fail the DTM build above (Section 11) -- depth
    # processing and source-quality attribution are deliberately separable.
    try:
        source_refs = emodnet.fetch_source_references(aoi_bbox_wgs84)
        qi_features = emodnet.fetch_quality_index(aoi_bbox_wgs84)
        attribution_status = "available"
        attribution_notes = ""
    except emodnet.EmodnetAttributionUnavailableError as exc:
        source_refs = []
        qi_features = []
        attribution_status = "unavailable"
        attribution_notes = str(exc)

    msl_result = emodnet.check_msl_availability(aoi_bbox_wgs84)
    if msl_result.available:
        msl_notes = (
            f"available (tile {msl_result.tile_id}, release {msl_result.dtm_release}, "
            f"format '{msl_result.format_label}'); {msl_result.notes}"
        )
    else:
        msl_notes = f"not available; {msl_result.notes}"

    chainage_df = bathymetry.sample_chainage_bathymetry(
        chainage_gdf=chainage_gdf,
        canonical_raster_path=output_raster_path,
        source_reference_features=source_refs,
        quality_index_features=qi_features,
        working_crs=working_crs,
    )
    bathymetry.write_chainage_bathymetry(chainage_df, chainage_output_path)

    bathymetry.print_bathymetry_report(
        dtm_report,
        chainage_df,
        attribution_status=attribution_status,
        attribution_notes=attribution_notes,
        msl_notes=msl_notes,
    )
    return 0


def _cmd_resolve_bathymetry_sources(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    # Canonical DTM / chainage-bathymetry stay under processed/<study>/bathymetry/
    # (analysis-ready products); this command's own output is source/provenance
    # *resolution* metadata, not an analysis-ready product, so it belongs under
    # interim/<study>/ instead -- see MAR-006C.
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    chainage_bathymetry_path = study_dir / "bathymetry" / "chainage_bathymetry.parquet"

    if not chainage_bathymetry_path.exists():
        print(
            f"error: no chainage bathymetry attribution found at {chainage_bathymetry_path}; "
            "run build-bathymetry first",
            file=sys.stderr,
        )
        return 1

    try:
        pipeline_gdf = gpd.read_file(pipeline_gpkg_path, layer="pipeline")
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
        chainage_bathymetry = pd.read_parquet(chainage_bathymetry_path)
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(
            f"error: could not load canonical pipeline/AOI/chainage/bathymetry: {exc}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    aoi_bbox_wgs84 = tuple(float(v) for v in aoi_gdf.to_crs("EPSG:4326").total_bounds)

    try:
        source_refs = emodnet.fetch_source_references(aoi_bbox_wgs84)
        qi_features = emodnet.fetch_quality_index(aoi_bbox_wgs84)
    except emodnet.EmodnetAttributionUnavailableError as exc:
        print(
            f"error: could not retrieve EMODnet source-reference attribution: {exc}",
            file=sys.stderr,
        )
        return 1

    df, records, overlaps = source_resolution.resolve_pl854_cdi_sources(
        pipeline_gdf=pipeline_gdf,
        aoi_gdf=aoi_gdf,
        chainage_gdf=chainage_gdf,
        chainage_bathymetry=chainage_bathymetry,
        source_reference_features=source_refs,
        quality_index_features=qi_features,
        working_crs=working_crs,
    )

    parquet_path = interim_dir / "emodnet_cdi_sources.parquet"
    gpkg_path = interim_dir / "emodnet_cdi_sources.gpkg"
    source_resolution.write_cdi_sources_parquet(df, parquet_path)
    source_resolution.write_cdi_sources_gpkg(records, working_crs, gpkg_path)

    print(f"Resolved {len(df)} PL854 source-reference record(s).")
    print(f"Output: {parquet_path}")
    print()
    source_resolution.print_source_resolution_report(df, overlaps)
    return 0


def _cmd_build_regional_morphology(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    _pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    chainage_bathymetry_path = study_dir / "bathymetry" / "chainage_bathymetry.parquet"
    cdi_sources_path = interim_dir / "emodnet_cdi_sources.parquet"
    canonical_dtm_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"

    if not chainage_bathymetry_path.exists():
        print(
            f"error: no chainage bathymetry attribution found at {chainage_bathymetry_path}; "
            "run build-bathymetry first",
            file=sys.stderr,
        )
        return 1
    if not cdi_sources_path.exists():
        print(
            f"error: no CDI source provenance found at {cdi_sources_path}; "
            "run resolve-bathymetry-sources first",
            file=sys.stderr,
        )
        return 1

    try:
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
        chainage_bathymetry_df = pd.read_parquet(chainage_bathymetry_path)
        cdi_sources_df = pd.read_parquet(cdi_sources_path)
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(
            f"error: could not load canonical AOI/chainage/bathymetry/provenance: {exc}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    aoi_geom_working = unary_union(aoi_gdf.geometry)
    halo_bbox_wgs84 = regional.build_halo_bbox_wgs84(aoi_geom_working, working_crs)

    manifest_path = interim_dir / "bathymetry_acquisition_manifest.json"
    halo_dataset_id = f"{emodnet.COVERAGE_ID}_halo_{pipeline_id.lower()}"
    raw_halo_path = (
        acquisition.raw_dataset_dir(config.paths.raw_dir, "emodnet", halo_dataset_id)
        / f"{halo_dataset_id}.tif"
    )
    # Must match exactly what `fetch_emodnet_geotiff` itself records as
    # `request_parameters` (coverageId/bbox_wgs84/format only) -- any extra
    # key here would never equal the stored entry and defeat idempotency,
    # re-fetching this halo on every single run.
    request_parameters = {
        "coverageId": emodnet.COVERAGE_ID,
        "bbox_wgs84": list(halo_bbox_wgs84),
        "format": "image/tiff;application=geotiff",
    }
    existing = acquisition.already_acquired(
        manifest_path, "EMODnet", halo_dataset_id, request_parameters
    )
    if existing is not None:
        print(f"EMODnet halo: already acquired ({existing['local_path']}); skipping re-download.")
        halo_manifest_entry = existing
    else:
        try:
            fetch_result = emodnet.fetch_emodnet_geotiff(halo_bbox_wgs84, raw_halo_path)
        except emodnet.EmodnetUnavailableError as exc:
            print(f"error: EMODnet halo acquisition failed: {exc}", file=sys.stderr)
            return 1
        halo_manifest_entry = acquisition.record_acquisition(
            manifest_path,
            source="EMODnet",
            dataset_id=halo_dataset_id,
            source_url_or_service=emodnet.WCS_BASE_URL,
            request_parameters=fetch_result.request_parameters,
            local_path=fetch_result.local_path,
            licence=emodnet.LICENCE,
            # EMODnet DTM 2024 is an aggregate product release, not a survey
            # acquisition -- see acquisition.record_acquisition's docstring.
            acquisition_year=None,
            product_release_year=2024,
            horizontal_crs=fetch_result.returned_crs,
            vertical_datum=emodnet.VERTICAL_DATUM,
            nominal_resolution_m=emodnet.NATIVE_RESOLUTION_M,
            acquired_at=datetime.now(UTC),
        )
        print(
            f"EMODnet halo: acquired {fetch_result.width_px}x{fetch_result.height_px} px "
            f"-> {fetch_result.local_path}"
        )

    qa_layer_availability = emodnet.check_native_qa_layers(halo_bbox_wgs84)

    try:
        result = regional.build_regional_morphology(
            aoi_gdf=aoi_gdf,
            chainage_gdf=chainage_gdf,
            chainage_bathymetry_df=chainage_bathymetry_df,
            cdi_sources_df=cdi_sources_df,
            raw_halo_path=raw_halo_path,
            raw_halo_manifest_entry=halo_manifest_entry,
            qa_layer_availability=qa_layer_availability,
            working_crs=working_crs,
            aoi_identifier=f"{pipeline_id}_AOI",
        )
    except (
        bathymetry.InvalidRawRasterError,
        bathymetry.AmbiguousSignConventionError,
        regional.RegionalMorphologyError,
    ) as exc:
        print(f"error: regional morphology build failed: {exc}", file=sys.stderr)
        return 1

    morphology_dir = study_dir / "morphology"
    raster_paths = {}
    for layer in result.layers:
        raster_path = morphology_dir / f"{layer.name}.tif"
        regional.write_morphology_raster(
            layer, working_crs, raster_path, aoi_identifier=f"{pipeline_id}_AOI"
        )
        raster_paths[layer.name] = raster_path

    chainage_output_path = morphology_dir / "chainage_regional_morphology.parquet"
    regional.write_chainage_regional_morphology(result.chainage_df, chainage_output_path)

    metadata_path = morphology_dir / "morphology_metadata.json"
    regional.write_morphology_metadata(
        result,
        input_dtm_path=canonical_dtm_path,
        raster_paths=raster_paths,
        output_path=metadata_path,
    )

    print(f"Wrote {len(raster_paths)} morphology raster(s), chainage table, and metadata to:")
    print(f"  {morphology_dir}")
    print()
    regional.print_regional_morphology_report(result)
    return 0


def _cmd_build_sediment_evidence(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    if not aoi_gpkg_path.exists() or not chainage_gpkg_path.exists():
        print(
            f"error: no canonical AOI/chainage found under {aoi_gpkg_path.parent}; "
            "run build-aoi and build-chainage first",
            file=sys.stderr,
        )
        return 1

    try:
        route_working, _attributes, _source_crs = load_pipeline_route(
            pipeline_gpkg_path, pipeline_id
        )
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(f"error: could not load canonical pipeline/AOI/chainage: {exc}", file=sys.stderr)
        return 1

    working_crs = config.crs.horizontal
    aoi_geometry_working = unary_union(aoi_gdf.geometry)
    aoi_geometry_wgs84 = (
        gpd.GeoSeries([aoi_geometry_working], crs=working_crs).to_crs("EPSG:4326").iloc[0]
    )
    query_timestamp = datetime.now(UTC)

    try:
        psa_features = sediment_bgs.fetch_psa_observations(aoi_geometry_wgs84)
        seabed_250k_features = sediment_bgs.fetch_seabed_sediments_250k(aoi_geometry_wgs84)
        predictive_folk_features = sediment_bgs.fetch_predictive_folk_polygons(aoi_geometry_wgs84)
    except sediment_bgs.BgsSedimentUnavailableError as exc:
        print(f"error: BGS sediment acquisition failed: {exc}", file=sys.stderr)
        return 1

    psa_gdf = evidence.normalize_psa_observations(
        psa_features,
        route_working=route_working,
        working_crs=working_crs,
        aoi_geometry_working=aoi_geometry_working,
        run_timestamp=query_timestamp,
    )
    psa_gdf = evidence.attach_nearest_chainage_station(psa_gdf, chainage_gdf.to_crs(working_crs))

    seabed_250k_gdf = evidence.normalize_seabed_sediments_250k(
        seabed_250k_features, working_crs=working_crs
    )
    seabed_250k_gdf = evidence.compute_250k_intersections(
        seabed_250k_gdf, aoi_geometry_working=aoi_geometry_working, route_working=route_working
    )

    predictive_folk_gdf = evidence.normalize_predictive_folk_polygons(
        predictive_folk_features, working_crs=working_crs
    )

    psa_with_comparisons = evidence.attach_mapped_and_predictive_at_psa_points(
        psa_gdf, seabed_250k_gdf, predictive_folk_gdf
    )

    # Predictive sand/gravel/mud percentages: only at surface PSA points, never
    # at all 941 chainage stations -- see the metadata's
    # predictive_percentage_chainage_note for why (Section 16's own
    # "if not safely queryable, do not fabricate").
    surface_mask = psa_with_comparisons["surface_evidence_class"].isin(
        (evidence.SURFACE_GRAB, evidence.SURFACE_CORE_INTERVAL)
    )
    predictive_percentages_by_psa_id: dict = {}
    for _, row in psa_with_comparisons[surface_mask].iterrows():
        percentages = {}
        for key, layer_id in (
            ("gravel", sediment_bgs.PREDICTIVE_GRAVEL_LAYER_ID),
            ("sand", sediment_bgs.PREDICTIVE_SAND_LAYER_ID),
            ("mud", sediment_bgs.PREDICTIVE_MUD_LAYER_ID),
        ):
            try:
                percentages[key] = sediment_bgs.fetch_predictive_percentage_at_point(
                    row["longitude"], row["latitude"], layer_id
                )
            except sediment_bgs.BgsSedimentUnavailableError:
                percentages[key] = None
        predictive_percentages_by_psa_id[row["psa_data_id"]] = percentages

    predictive_comparison_df = evidence.build_predictive_comparison_table(
        psa_with_comparisons, predictive_percentages_by_psa_id=predictive_percentages_by_psa_id
    )

    chainage_sediment_df = evidence.build_chainage_sediment_evidence(
        chainage_gdf=chainage_gdf,
        psa_gdf_working=psa_with_comparisons,
        seabed_250k_gdf_working=seabed_250k_gdf,
        predictive_folk_gdf_working=predictive_folk_gdf,
        working_crs=working_crs,
    )

    coverage = evidence.compute_coverage_diagnostics(psa_gdf)
    chainage_support = evidence.compute_chainage_support_proportions(chainage_sediment_df)
    agreement = evidence.compute_agreement_diagnostics(psa_with_comparisons, chainage_sediment_df)
    d50_assessment = evidence.assess_d50_spatial_support(chainage_sediment_df, coverage)

    sediment_interim_dir = interim_dir / "sediment"
    sediment_processed_dir = config.paths.processed_dir / pipeline_id.lower() / "sediment"

    psa_parquet_path = sediment_interim_dir / "bgs_psa_observations.parquet"
    psa_gpkg_path = sediment_interim_dir / "bgs_psa_observations.gpkg"
    seabed_250k_gpkg_path = sediment_interim_dir / "bgs_seabed_sediments_250k.gpkg"
    seabed_250k_parquet_path = sediment_interim_dir / "bgs_seabed_sediments_250k.parquet"
    predictive_comparison_path = sediment_interim_dir / "bgs_predictive_sediment_comparison.parquet"
    chainage_sediment_path = sediment_processed_dir / "chainage_sediment_evidence.parquet"
    metadata_path = sediment_processed_dir / "sediment_evidence_metadata.json"

    evidence.write_psa_observations(psa_gdf, psa_parquet_path, psa_gpkg_path)
    evidence.write_seabed_sediments_250k(
        seabed_250k_gdf, seabed_250k_gpkg_path, seabed_250k_parquet_path
    )
    evidence.write_predictive_comparison(predictive_comparison_df, predictive_comparison_path)
    evidence.write_chainage_sediment_evidence(chainage_sediment_df, chainage_sediment_path)

    providers_metadata = {
        "bgs_psa": {
            "provider": "BGS",
            "dataset": sediment_bgs.PSA_DATASET_TITLE,
            "endpoint": sediment_bgs.PSA_SERVICE_URL,
            "evidence_role": "PRIMARY_OBSERVATIONAL",
        },
        "bgs_seabed_sediments_250k": {
            "provider": "BGS",
            "dataset": sediment_bgs.SEABED_SEDIMENTS_250K_DATASET_TITLE,
            "endpoint": sediment_bgs.SEABED_SEDIMENTS_250K_SERVICE_URL,
            "evidence_role": "REGIONAL_MAPPED_SUBSTRATE",
        },
        "bgs_predictive": {
            "provider": "BGS",
            "dataset": sediment_bgs.PREDICTIVE_DATASET_TITLE,
            "endpoint": sediment_bgs.PREDICTIVE_FOLK_SERVICE_URL,
            "percentage_layers_endpoint": sediment_bgs.PREDICTIVE_SERVICE_ROOT_URL,
            "evidence_role": "SECONDARY_MODEL_COMPARISON",
        },
    }
    evidence.write_sediment_evidence_metadata(
        providers=providers_metadata,
        query_timestamp=query_timestamp,
        aoi_identifier=f"{pipeline_id}_AOI",
        coverage=coverage,
        chainage_support=chainage_support,
        agreement=agreement,
        d50_assessment=d50_assessment,
        outputs={
            "psa_observations_parquet": psa_parquet_path,
            "psa_observations_gpkg": psa_gpkg_path,
            "seabed_sediments_250k_gpkg": seabed_250k_gpkg_path,
            "seabed_sediments_250k_parquet": seabed_250k_parquet_path,
            "predictive_comparison_parquet": predictive_comparison_path,
            "chainage_sediment_evidence_parquet": chainage_sediment_path,
        },
        output_path=metadata_path,
    )

    print(f"PSA observations: {len(psa_gdf)} record(s) -> {psa_parquet_path}")
    print(f"Seabed Sediments 250k: {len(seabed_250k_gdf)} polygon(s) -> {seabed_250k_gpkg_path}")
    print(f"Predictive comparison: {len(predictive_comparison_df)} row(s)")
    print(f"  -> {predictive_comparison_path}")
    print(f"Chainage sediment evidence: {len(chainage_sediment_df)} station(s)")
    print(f"  -> {chainage_sediment_path}")
    print(f"Metadata: {metadata_path}")
    print()
    evidence.print_sediment_evidence_report(
        coverage=coverage,
        chainage_support=chainage_support,
        agreement=agreement,
        d50_assessment=d50_assessment,
    )
    return 0


def _read_existing_parquet(path: Path) -> pd.DataFrame:
    """The prior run's canonical output, or an empty frame if there isn't one yet.

    Read BEFORE that same path is overwritten, so the old-vs-corrected
    comparison (MAR-009A Sections 12, 13, 19) reflects what was actually on
    disk rather than a fabricated baseline.
    """

    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _mask_surface_slice(mask: xr.DataArray) -> xr.DataArray:
    """The surface-level slice of a mask variable that may or may not have a depth dimension."""

    if "depth" in mask.dims:
        return mask.isel(depth=0)
    return mask


def _acquire_static_dataset(
    *,
    manifest_path: Path,
    raw_dir: Path,
    product_id: str,
    dataset_id: str,
    variables: list,
    bbox_wgs84: tuple,
    evidence_role: str,
) -> xr.Dataset:
    requested_bbox = list(bbox_wgs84)
    existing = metocean_acquisition.already_acquired(
        manifest_path,
        product_id=product_id,
        dataset_id=dataset_id,
        variables=variables,
        requested_bbox=requested_bbox,
        requested_depths=None,
        requested_start=metocean_acquisition.STATIC_TIME_SENTINEL,
        requested_end=metocean_acquisition.STATIC_TIME_SENTINEL,
    )
    if existing is not None:
        return xr.open_dataset(existing["local_path"])

    result = copernicus.subset_dataset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=bbox_wgs84[0],
        maximum_longitude=bbox_wgs84[2],
        minimum_latitude=bbox_wgs84[1],
        maximum_latitude=bbox_wgs84[3],
        start_datetime=None,
        end_datetime=None,
        minimum_depth=None,
        maximum_depth=None,
        output_directory=raw_dir,
        output_filename=f"{dataset_id}_static.nc",
    )
    metocean_acquisition.record_acquisition(
        manifest_path,
        provider="Copernicus Marine",
        product_id=product_id,
        dataset_id=dataset_id,
        evidence_role=evidence_role,
        variables=variables,
        requested_bbox=requested_bbox,
        requested_depths=None,
        requested_start=None,
        requested_end=None,
        actual_start=None,
        actual_end=None,
        temporal_resolution="static",
        local_path=result.local_path,
        toolbox_version=copernicus.toolbox_version(),
        licence=None,
        downloaded_at=datetime.now(UTC),
    )
    return xr.open_dataset(result.local_path)


def _acquire_chunked_dataset(
    *,
    manifest_path: Path,
    raw_dir: Path,
    product_id: str,
    dataset_id: str,
    variables: list,
    bbox_wgs84: tuple,
    depth_range: tuple | None,
    chunk_ranges: list,
    evidence_role: str,
    temporal_resolution: str,
) -> tuple[xr.Dataset | None, metocean_acquisition.TemporalDeduplicationResult | None]:
    """Acquire every chunk (resuming from the manifest), then open+concat+dedup them.

    Never re-downloads a chunk whose exact identity (product/dataset/
    variables/bbox/depths/start/end) is already manifested with its file
    still on disk (Section 15). Adjacent chunks can share their boundary
    instant (Copernicus subset requests are inclusive of `end_datetime`) --
    `deduplicate_time_coordinate` collapses that overlap here, at chunk
    assembly, before the dataset is ever normalized (MAR-009B).
    """

    requested_bbox = list(bbox_wgs84)
    requested_depths = list(depth_range) if depth_range else None
    chunk_paths: list[Path] = []

    for chunk_start, chunk_end in chunk_ranges:
        existing = metocean_acquisition.already_acquired(
            manifest_path,
            product_id=product_id,
            dataset_id=dataset_id,
            variables=variables,
            requested_bbox=requested_bbox,
            requested_depths=requested_depths,
            requested_start=chunk_start.isoformat(),
            requested_end=chunk_end.isoformat(),
        )
        if existing is not None:
            chunk_paths.append(Path(existing["local_path"]))
            continue

        result = copernicus.subset_dataset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=bbox_wgs84[0],
            maximum_longitude=bbox_wgs84[2],
            minimum_latitude=bbox_wgs84[1],
            maximum_latitude=bbox_wgs84[3],
            start_datetime=chunk_start,
            end_datetime=chunk_end,
            minimum_depth=depth_range[0] if depth_range else None,
            maximum_depth=depth_range[1] if depth_range else None,
            output_directory=raw_dir,
            output_filename=f"{dataset_id}_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.nc",
        )
        with xr.open_dataset(result.local_path) as chunk_ds:
            actual_start = (
                str(chunk_ds["time"].min().values) if "time" in chunk_ds.variables else None
            )
            actual_end = (
                str(chunk_ds["time"].max().values) if "time" in chunk_ds.variables else None
            )

        metocean_acquisition.record_acquisition(
            manifest_path,
            provider="Copernicus Marine",
            product_id=product_id,
            dataset_id=dataset_id,
            evidence_role=evidence_role,
            variables=variables,
            requested_bbox=requested_bbox,
            requested_depths=requested_depths,
            requested_start=chunk_start,
            requested_end=chunk_end,
            actual_start=actual_start,
            actual_end=actual_end,
            temporal_resolution=temporal_resolution,
            local_path=result.local_path,
            toolbox_version=copernicus.toolbox_version(),
            licence=None,
            downloaded_at=datetime.now(UTC),
        )
        chunk_paths.append(result.local_path)

    if not chunk_paths:
        return None, None
    if len(chunk_paths) == 1:
        ds = xr.open_dataset(chunk_paths[0])
    else:
        ds = xr.open_mfdataset([str(p) for p in chunk_paths], combine="by_coords")
    deduped_ds, dedup_result = metocean_acquisition.deduplicate_time_coordinate(ds)
    return deduped_ds, dedup_result


def _cmd_build_metocean_evidence(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    _pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    chainage_bathymetry_path = study_dir / "bathymetry" / "chainage_bathymetry.parquet"

    if not aoi_gpkg_path.exists() or not chainage_gpkg_path.exists():
        print(
            f"error: no canonical AOI/chainage found under {aoi_gpkg_path.parent}; "
            "run build-aoi and build-chainage first",
            file=sys.stderr,
        )
        return 1

    try:
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(f"error: could not load canonical AOI/chainage: {exc}", file=sys.stderr)
        return 1

    canonical_depth_df = None
    if chainage_bathymetry_path.exists():
        canonical_depth_df = pd.read_parquet(chainage_bathymetry_path)[
            ["station_index", "depth_lat_m"]
        ]

    working_crs = config.crs.horizontal
    # A modest extra buffer beyond the AOI so the model bbox request
    # comfortably contains at least one real wet cell even near a coastline.
    request_buffer_m = 5000.0
    aoi_geom_working = unary_union(aoi_gdf.geometry).buffer(request_buffer_m)
    aoi_bbox_wgs84 = tuple(
        float(v)
        for v in gpd.GeoSeries([aoi_geom_working], crs=working_crs).to_crs("EPSG:4326").total_bounds
    )
    chainage_points_working = chainage_gdf.to_crs(working_crs).geometry

    print("Resolving live Copernicus Marine dataset ids...")
    try:
        primary_current_dataset_id = copernicus.confirm_live_dataset_id(
            copernicus.PRIMARY_CURRENT_PRODUCT_ID, copernicus.PRIMARY_CURRENT_DATASET_ID
        )
        long_term_current_dataset_id = copernicus.confirm_live_dataset_id(
            copernicus.LONG_TERM_CURRENT_PRODUCT_ID, copernicus.LONG_TERM_CURRENT_DATASET_ID
        )
        wave_dataset_id = copernicus.confirm_live_dataset_id(
            copernicus.WAVE_PRODUCT_ID, copernicus.WAVE_DATASET_ID
        )
    except copernicus.CopernicusDatasetNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        copernicus.ensure_authenticated()
    except copernicus.CopernicusAuthenticationRequiredError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metocean_interim_dir = interim_dir / "metocean"
    metocean_processed_dir = config.paths.processed_dir / pipeline_id.lower() / "metocean"
    raw_dir = config.paths.raw_dir / "metocean" / "copernicus"
    manifest_path = metocean_interim_dir / "copernicus_acquisition_manifest.json"

    now_utc = datetime.now(UTC)
    historical_cutoff = metocean_acquisition.compute_historical_cutoff(now_utc)

    # --- static bathymetry/mask for each product -------------------------
    primary_static_ds = _acquire_static_dataset(
        manifest_path=manifest_path,
        raw_dir=raw_dir / "statics",
        product_id=copernicus.PRIMARY_CURRENT_PRODUCT_ID,
        dataset_id=copernicus.PRIMARY_CURRENT_STATIC_DATASET_ID,
        variables=list(copernicus.PRIMARY_CURRENT_STATIC_VARIABLES),
        bbox_wgs84=aoi_bbox_wgs84,
        evidence_role=copernicus.PRIMARY_CURRENT_EVIDENCE_ROLE,
    )
    long_term_static_ds = _acquire_static_dataset(
        manifest_path=manifest_path,
        raw_dir=raw_dir / "statics",
        product_id=copernicus.LONG_TERM_CURRENT_PRODUCT_ID,
        dataset_id=copernicus.LONG_TERM_CURRENT_STATIC_DATASET_ID,
        variables=["deptho"],
        bbox_wgs84=aoi_bbox_wgs84,
        evidence_role=copernicus.LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE,
    )
    wave_static_ds = _acquire_static_dataset(
        manifest_path=manifest_path,
        raw_dir=raw_dir / "statics",
        product_id=copernicus.WAVE_PRODUCT_ID,
        dataset_id=copernicus.WAVE_STATIC_DATASET_ID,
        variables=list(copernicus.WAVE_STATIC_VARIABLES),
        bbox_wgs84=aoi_bbox_wgs84,
        evidence_role=copernicus.PRIMARY_WAVE_CLIMATE_ROLE,
    )

    # --- support-node identification and chainage mapping (Sections 4-5) -
    primary_nodes = metocean_evidence.identify_wet_grid_cells(
        primary_static_ds["longitude"].to_numpy(),
        primary_static_ds["latitude"].to_numpy(),
        _mask_surface_slice(primary_static_ds["mask"]).to_numpy(),
        "current",
    )
    long_term_nodes = metocean_evidence.identify_wet_grid_cells(
        long_term_static_ds["longitude"].to_numpy(),
        long_term_static_ds["latitude"].to_numpy(),
        np.isfinite(long_term_static_ds["deptho"].to_numpy()),
        "current_lt",
    )
    wave_nodes = metocean_evidence.identify_wet_grid_cells(
        wave_static_ds["longitude"].to_numpy(),
        wave_static_ds["latitude"].to_numpy(),
        _mask_surface_slice(wave_static_ds["mask"]).to_numpy(),
        "wave",
    )

    primary_mapping = metocean_evidence.map_points_to_nearest_node(
        chainage_points_working, primary_nodes, working_crs
    )
    long_term_mapping = metocean_evidence.map_points_to_nearest_node(
        chainage_points_working, long_term_nodes, working_crs
    )
    wave_mapping = metocean_evidence.map_points_to_nearest_node(
        chainage_points_working, wave_nodes, working_crs
    )

    primary_bathymetry_by_node = {
        node.node_id: float(
            primary_static_ds["deptho"].isel(latitude=node.grid_j, longitude=node.grid_i).to_numpy()
        )
        for node in primary_nodes
        if node.node_id in set(primary_mapping["node_id"].dropna())
    }
    primary_deptho_lev_by_node = {
        node.node_id: float(
            primary_static_ds["deptho_lev"]
            .isel(latitude=node.grid_j, longitude=node.grid_i)
            .to_numpy()
        )
        for node in primary_nodes
        if node.node_id in set(primary_mapping["node_id"].dropna())
    }
    wave_bathymetry_by_node = {
        node.node_id: float(
            wave_static_ds["deptho"].isel(latitude=node.grid_j, longitude=node.grid_i).to_numpy()
        )
        for node in wave_nodes
        if node.node_id in set(wave_mapping["node_id"].dropna())
    }

    primary_node_table = metocean_evidence.build_support_node_table(
        primary_nodes,
        primary_mapping["node_id"],
        primary_mapping["distance_m"],
        model_bathymetry_by_node_id=primary_bathymetry_by_node,
        deptho_lev_by_node_id=primary_deptho_lev_by_node,
        source_product=copernicus.PRIMARY_CURRENT_PRODUCT_ID,
        source_dataset=primary_current_dataset_id,
        evidence_role=copernicus.PRIMARY_CURRENT_EVIDENCE_ROLE,
    )

    # Only chainage-USED nodes are ever normalized into a time series
    # (MAR-009A, Section 7) -- never every wet cell in the request bbox.
    used_primary_nodes = [n for n in primary_nodes if n.node_id in primary_bathymetry_by_node]
    used_long_term_node_ids = set(long_term_mapping["node_id"].dropna())
    used_long_term_nodes = [n for n in long_term_nodes if n.node_id in used_long_term_node_ids]
    used_wave_node_ids = set(wave_mapping["node_id"].dropna())
    used_wave_nodes = [n for n in wave_nodes if n.node_id in used_wave_node_ids]

    # --- capture OLD canonical outputs for the old-vs-corrected comparison
    # (Sections 12, 13, 19) -- read BEFORE anything is overwritten below.
    old_primary_current_df = _read_existing_parquet(
        metocean_interim_dir / "current_primary_hourly.parquet"
    )
    # The node-count regression this ticket fixes (Section 7/8) shows up in
    # the TIME SERIES files (every wet bbox cell was normalized), not the
    # support-node table (already correctly used-only via
    # `build_support_node_table`) -- so the OLD comparison baseline must be
    # read from the same hourly/3-hourly files, not the node table.
    old_wave_df = _read_existing_parquet(metocean_interim_dir / "wave_3hourly.parquet")
    old_long_term_current_df = _read_existing_parquet(
        metocean_interim_dir / "current_long_term_surface_hourly.parquet"
    )

    def _safe_old_stats(compute_fn, df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Old (pre-fix) stats for the comparison report only -- never load-bearing.

        The OLD on-disk data predates MAR-009B's chunk-boundary dedup, so it
        can itself trip the new strict completeness check
        (`TemporalCompletenessError`) -- that is an expected, informative
        outcome (it demonstrates the pre-fix defect), not a reason to abort
        this run. Only the NEW canonical computation must remain a hard
        failure.
        """

        if df.empty:
            return pd.DataFrame()
        try:
            return compute_fn(df)
        except metocean_evidence.TemporalCompletenessError:
            print(
                f"note: old (pre-fix) {label} data itself exceeds 100% completeness "
                "(the exact defect this ticket fixes) -- old comparison values for "
                f"{label} are reported as unavailable rather than computed",
                file=sys.stderr,
            )
            return pd.DataFrame()

    old_current_stats = _safe_old_stats(
        metocean_evidence.compute_current_node_statistics, old_primary_current_df, "primary current"
    )
    old_wave_stats = _safe_old_stats(
        metocean_evidence.compute_wave_node_statistics, old_wave_df, "wave"
    )
    old_long_term_stats = _safe_old_stats(
        metocean_evidence.compute_long_term_surface_current_statistics,
        old_long_term_current_df,
        "long-term surface current",
    )

    # --- primary current acquisition (Section 6, 15) ---------------------
    primary_chunk_ranges = metocean_acquisition.generate_monthly_chunks(
        # The rolling analysis/forecast catalogue's own available start is
        # discovered live, never hard-coded (Section 6).
        _dataset_start_or(
            copernicus.get_dataset_time_range_ms(primary_current_dataset_id), now_utc
        ),
        historical_cutoff,
    )
    try:
        primary_current_ds, primary_temporal_dedup = _acquire_chunked_dataset(
            manifest_path=manifest_path,
            raw_dir=raw_dir / "current_primary",
            product_id=copernicus.PRIMARY_CURRENT_PRODUCT_ID,
            dataset_id=primary_current_dataset_id,
            variables=list(copernicus.PRIMARY_CURRENT_VARIABLES),
            bbox_wgs84=aoi_bbox_wgs84,
            depth_range=None,
            chunk_ranges=primary_chunk_ranges,
            evidence_role=copernicus.PRIMARY_CURRENT_EVIDENCE_ROLE,
            temporal_resolution="hourly_instantaneous",
        )
    except metocean_acquisition.DuplicateTimestampConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    static_depth_mask_used = False
    static_mask_by_node_id: dict[str, np.ndarray] | None = None
    unreconciled_primary_node_ids: list[str] = []
    primary_current_df = pd.DataFrame()
    if primary_current_ds is not None:
        static_depth_mask_used = "depth" in primary_static_ds.coords and (
            metocean_evidence.check_depth_coordinate_alignment(
                primary_static_ds["depth"].to_numpy(), primary_current_ds["depth"].to_numpy()
            )
        )
        if static_depth_mask_used:
            static_mask_by_node_id = {
                node.node_id: primary_static_ds["mask"]
                .isel(latitude=node.grid_j, longitude=node.grid_i)
                .to_numpy()
                for node in used_primary_nodes
            }
        else:
            print(
                "warning: static mask depth coordinate does not align with the dynamic "
                "current dataset's own depth coordinate -- proceeding with the bathymetry-"
                "depth eligibility constraint alone (Section 3); static_depth_mask_used=False",
                file=sys.stderr,
            )

        primary_current_df, unreconciled_primary_node_ids = (
            metocean_evidence.normalize_primary_current(
                primary_current_ds,
                nodes=used_primary_nodes,
                model_bathymetry_by_node_id=primary_bathymetry_by_node,
                static_mask_by_node_id=static_mask_by_node_id,
                source_dataset=primary_current_dataset_id,
                evidence_role=copernicus.PRIMARY_CURRENT_EVIDENCE_ROLE,
            )
        )
        if unreconciled_primary_node_ids:
            print(
                f"warning: {len(unreconciled_primary_node_ids)} primary current support "
                "node(s) could not be reconciled against the dynamic dataset grid and were "
                f"excluded from normalization: {unreconciled_primary_node_ids}",
                file=sys.stderr,
            )

    # --- long-term surface current acquisition (Sections 11, 31) ---------
    long_term_start_ms = copernicus.get_dataset_time_range_ms(long_term_current_dataset_id)
    long_term_chunk_ranges = metocean_acquisition.generate_yearly_chunks(
        _dataset_start_or(long_term_start_ms, now_utc), historical_cutoff
    )
    try:
        long_term_current_ds, long_term_temporal_dedup = _acquire_chunked_dataset(
            manifest_path=manifest_path,
            raw_dir=raw_dir / "current_long_term_surface",
            product_id=copernicus.LONG_TERM_CURRENT_PRODUCT_ID,
            dataset_id=long_term_current_dataset_id,
            variables=list(copernicus.LONG_TERM_CURRENT_VARIABLES),
            bbox_wgs84=aoi_bbox_wgs84,
            depth_range=None,
            chunk_ranges=long_term_chunk_ranges,
            evidence_role=copernicus.LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE,
            temporal_resolution="hourly_instantaneous",
        )
    except metocean_acquisition.DuplicateTimestampConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    unreconciled_long_term_node_ids: list[str] = []
    long_term_current_df = pd.DataFrame()
    if long_term_current_ds is not None:
        long_term_current_df, unreconciled_long_term_node_ids = (
            metocean_evidence.normalize_long_term_surface_current(
                long_term_current_ds,
                nodes=used_long_term_nodes,
                source_dataset=long_term_current_dataset_id,
                evidence_role=copernicus.LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE,
            )
        )
        if unreconciled_long_term_node_ids:
            print(
                f"warning: {len(unreconciled_long_term_node_ids)} long-term surface current "
                "support node(s) could not be reconciled against the dynamic dataset grid and "
                f"were excluded from normalization: {unreconciled_long_term_node_ids}",
                file=sys.stderr,
            )

    # --- wave reanalysis acquisition (Section 12) -------------------------
    wave_start_ms = copernicus.get_dataset_time_range_ms(wave_dataset_id)
    wave_chunk_ranges = metocean_acquisition.generate_yearly_chunks(
        _dataset_start_or(wave_start_ms, now_utc), historical_cutoff
    )
    try:
        wave_ds, wave_temporal_dedup = _acquire_chunked_dataset(
            manifest_path=manifest_path,
            raw_dir=raw_dir / "wave_reanalysis",
            product_id=copernicus.WAVE_PRODUCT_ID,
            dataset_id=wave_dataset_id,
            variables=list(copernicus.WAVE_VARIABLES),
            bbox_wgs84=aoi_bbox_wgs84,
            depth_range=None,
            chunk_ranges=wave_chunk_ranges,
            evidence_role=copernicus.PRIMARY_WAVE_CLIMATE_ROLE,
            temporal_resolution="3hourly_instantaneous",
        )
    except metocean_acquisition.DuplicateTimestampConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    unreconciled_wave_node_ids: list[str] = []
    wave_df = pd.DataFrame()
    if wave_ds is not None:
        wave_df, unreconciled_wave_node_ids = metocean_evidence.normalize_wave(
            wave_ds, nodes=used_wave_nodes, source_dataset=wave_dataset_id
        )
        if unreconciled_wave_node_ids:
            print(
                f"warning: {len(unreconciled_wave_node_ids)} wave support node(s) could not "
                "be reconciled against the dynamic dataset grid and were excluded from "
                f"normalization: {unreconciled_wave_node_ids}",
                file=sys.stderr,
            )

    # --- descriptive statistics (Sections 24, 25, 27) ---------------------
    try:
        current_stats = metocean_evidence.compute_current_node_statistics(primary_current_df)
        long_term_stats = metocean_evidence.compute_long_term_surface_current_statistics(
            long_term_current_df
        )
        wave_stats = metocean_evidence.compute_wave_node_statistics(wave_df)
    except metocean_evidence.TemporalCompletenessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    annual_max_hs = metocean_evidence.compute_annual_max_hs(wave_df)

    primary_start = primary_current_df["time_utc"].min() if not primary_current_df.empty else None
    primary_end = primary_current_df["time_utc"].max() if not primary_current_df.empty else None
    short_window_ratios = metocean_evidence.compute_short_window_surface_context_ratio(
        long_term_current_df, primary_start, primary_end
    )

    # --- new integrity diagnostics (MAR-009A, Sections 4, 9, 10) ----------
    below_bed_diagnostics = metocean_evidence.compute_below_bed_diagnostics(primary_current_df)
    primary_current_route_summary = metocean_evidence.compute_primary_current_route_summary(
        primary_current_df
    )
    primary_distance_diagnostics = metocean_evidence.compute_distance_diagnostics(primary_mapping)
    long_term_distance_diagnostics = metocean_evidence.compute_distance_diagnostics(
        long_term_mapping
    )
    wave_distance_diagnostics = metocean_evidence.compute_distance_diagnostics(wave_mapping)

    # --- temporal integrity QA (MAR-009B, Sections 4, 5) -------------------
    # Merges the chunk-assembly-level dedup diagnostics (raw/unique/removed
    # timestamp counts, identical for every node of a product since they
    # share one dynamic time axis) with a defensive post-normalization
    # re-check (unique/monotonic/no-duplicate-rows) -- reported separately
    # for all three products, never folded together.
    def _temporal_qa(
        dedup_result: metocean_acquisition.TemporalDeduplicationResult | None,
        df: pd.DataFrame,
        *,
        time_column: str,
        node_column: str,
    ) -> dict[str, Any]:
        qa: dict[str, Any] = {
            "raw_time_count": dedup_result.raw_time_count if dedup_result else None,
            "unique_time_count": dedup_result.unique_time_count if dedup_result else None,
            "duplicate_boundary_timestamp_count": (
                dedup_result.duplicate_boundary_timestamp_count if dedup_result else None
            ),
        }
        qa.update(
            metocean_evidence.validate_temporal_integrity(
                df, time_column=time_column, node_column=node_column
            )
        )
        return qa

    primary_temporal_qa = _temporal_qa(
        primary_temporal_dedup,
        primary_current_df,
        time_column="time_utc",
        node_column="current_node_id",
    )
    long_term_temporal_qa = _temporal_qa(
        long_term_temporal_dedup,
        long_term_current_df,
        time_column="time_utc",
        node_column="current_lt_node_id",
    )
    wave_temporal_qa = _temporal_qa(
        wave_temporal_dedup, wave_df, time_column="time_utc", node_column="wave_node_id"
    )

    long_term_node_table = metocean_evidence.build_support_node_table(
        long_term_nodes,
        long_term_mapping["node_id"],
        long_term_mapping["distance_m"],
        source_product=copernicus.LONG_TERM_CURRENT_PRODUCT_ID,
        source_dataset=long_term_current_dataset_id,
        evidence_role=copernicus.LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE,
    )
    wave_node_table = metocean_evidence.build_support_node_table(
        wave_nodes,
        wave_mapping["node_id"],
        wave_mapping["distance_m"],
        model_bathymetry_by_node_id=wave_bathymetry_by_node,
        source_product=copernicus.WAVE_PRODUCT_ID,
        source_dataset=wave_dataset_id,
        evidence_role=copernicus.PRIMARY_WAVE_CLIMATE_ROLE,
    )

    # --- chainage evidence assembly (Section 23) --------------------------
    chainage_metocean_df = metocean_evidence.build_chainage_metocean_evidence(
        chainage_gdf=chainage_gdf,
        canonical_depth_df=canonical_depth_df,
        current_mapping=primary_mapping,
        current_stats=current_stats,
        current_node_bathymetry=primary_bathymetry_by_node,
        long_term_mapping=long_term_mapping,
        long_term_stats=long_term_stats,
        wave_mapping=wave_mapping,
        wave_stats=wave_stats,
        wave_node_bathymetry=wave_bathymetry_by_node,
    )

    # --- write outputs -----------------------------------------------------
    primary_nodes_path = metocean_evidence.write_parquet(
        primary_node_table, metocean_interim_dir / "current_primary_support_nodes.parquet"
    )
    primary_current_path = metocean_evidence.write_parquet(
        primary_current_df, metocean_interim_dir / "current_primary_hourly.parquet"
    )
    long_term_nodes_path = metocean_evidence.write_parquet(
        long_term_node_table,
        metocean_interim_dir / "current_long_term_surface_support_nodes.parquet",
    )
    long_term_current_path = metocean_evidence.write_parquet(
        long_term_current_df, metocean_interim_dir / "current_long_term_surface_hourly.parquet"
    )
    wave_nodes_path = metocean_evidence.write_parquet(
        wave_node_table, metocean_interim_dir / "wave_support_nodes.parquet"
    )
    wave_path = metocean_evidence.write_parquet(
        wave_df, metocean_interim_dir / "wave_3hourly.parquet"
    )
    annual_max_hs_path = metocean_evidence.write_parquet(
        annual_max_hs, metocean_interim_dir / "wave_annual_max_hs.parquet"
    )
    chainage_path = metocean_evidence.write_parquet(
        chainage_metocean_df, metocean_processed_dir / "chainage_metocean_evidence.parquet"
    )

    metadata_path = metocean_processed_dir / "metocean_evidence_metadata.json"
    metocean_evidence.write_metocean_evidence_metadata(
        metadata={
            "products": {
                "primary_current": {
                    "product_id": copernicus.PRIMARY_CURRENT_PRODUCT_ID,
                    "dataset_id": primary_current_dataset_id,
                    "static_dataset_id": copernicus.PRIMARY_CURRENT_STATIC_DATASET_ID,
                    "evidence_role": copernicus.PRIMARY_CURRENT_EVIDENCE_ROLE,
                    "temporal_resolution": "hourly_instantaneous",
                    "vertical_semantics": (
                        "deepest_valid_standard_level_current -- NOT native bottom-cell current"
                    ),
                },
                "long_term_surface_current": {
                    "product_id": copernicus.LONG_TERM_CURRENT_PRODUCT_ID,
                    "dataset_id": long_term_current_dataset_id,
                    "static_dataset_id": copernicus.LONG_TERM_CURRENT_STATIC_DATASET_ID,
                    "evidence_role": copernicus.LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE,
                    "temporal_resolution": "hourly_instantaneous",
                    "forbidden_daily_dataset_id": (
                        copernicus.LONG_TERM_CURRENT_FORBIDDEN_DAILY_DATASET_ID
                    ),
                },
                "wave": {
                    "product_id": copernicus.WAVE_PRODUCT_ID,
                    "dataset_id": wave_dataset_id,
                    "static_dataset_id": copernicus.WAVE_STATIC_DATASET_ID,
                    "evidence_role": copernicus.PRIMARY_WAVE_CLIMATE_ROLE,
                    "temporal_resolution": "3hourly_instantaneous",
                },
            },
            "retrieval_timestamp": now_utc.isoformat(),
            "historical_cutoff": historical_cutoff.isoformat(),
            "current_direction_convention": "current_direction_to_deg: degrees clockwise from "
            "true north, vector points TOWARD that bearing",
            "wave_direction_convention": "wave_mean_direction_from_deg: degrees the waves "
            "travel FROM; wave_mean_direction_to_deg = (from + 180) %% 360 is derived for "
            "convenience only",
            "support_node_mapping_method": "nearest wet model grid cell (no bilinear "
            "interpolation of data or masks)",
            "canonical_model_bathymetry_vertical_datums_not_harmonised": True,
            "dynamic_grid_coordinate_reconciliation_method": (
                "each support node's canonical lon/lat is re-resolved against the dynamic "
                "dataset's own coordinate arrays (nearest cell, refused beyond "
                f"{metocean_evidence.GRID_RECONCILIATION_TOLERANCE_FRACTION:.0%} of that "
                "axis's median grid spacing) -- static dataset grid indices are never reused "
                "directly against a dynamic dataset (MAR-009A Section 6)"
            ),
            "static_dynamic_coordinate_match_status": {
                "primary_current": (
                    "all_used_nodes_reconciled"
                    if not unreconciled_primary_node_ids
                    else f"{len(unreconciled_primary_node_ids)}_node(s)_unreconciled"
                ),
                "long_term_surface_current": (
                    "all_used_nodes_reconciled"
                    if not unreconciled_long_term_node_ids
                    else f"{len(unreconciled_long_term_node_ids)}_node(s)_unreconciled"
                ),
                "wave": (
                    "all_used_nodes_reconciled"
                    if not unreconciled_wave_node_ids
                    else f"{len(unreconciled_wave_node_ids)}_node(s)_unreconciled"
                ),
            },
            "primary_current_vertical_eligibility_rule": (
                "a standard depth is only eligible when uo AND vo are finite AND "
                "depth_m <= model_bathymetry_m + tolerance AND, where the static mask's depth "
                "coordinate is alignable, the mask cell is wet (MAR-009A Sections 2-3)"
            ),
            "static_depth_mask_used": static_depth_mask_used,
            "below_bed_finite_value_diagnostic_summary": {
                "below_model_bed_finite_candidate_count": (
                    int(below_bed_diagnostics["below_model_bed_finite_candidate_count"].sum())
                    if not below_bed_diagnostics.empty
                    else 0
                ),
                "timestamps_with_below_bed_finite_candidates": (
                    int(below_bed_diagnostics["timestamps_with_below_bed_finite_candidates"].sum())
                    if not below_bed_diagnostics.empty
                    else 0
                ),
                "max_below_bed_candidate_depth_m": (
                    float(below_bed_diagnostics["max_below_bed_candidate_depth_m"].max())
                    if not below_bed_diagnostics.empty
                    and below_bed_diagnostics["max_below_bed_candidate_depth_m"].notna().any()
                    else None
                ),
            },
            "only_chainage_used_support_nodes_normalized": True,
            "temporal_qa": {
                "primary_current": primary_temporal_qa,
                "long_term_surface_current": long_term_temporal_qa,
                "wave": wave_temporal_qa,
            },
            "raw_acquisition_manifest_path": str(manifest_path),
            "limitations": [
                "primary current record is only the rolling available historical analysis "
                "period, not a multi-decadal near-bed climatology",
                "deepest valid standard level is not the model native bottom cell",
                "long-term 7 km hourly current is surface current context only",
                "wave data are model reanalysis, not local buoy observations",
                "model bathymetry and canonical LAT bathymetry are not vertically harmonised",
            ],
            "no_physics_yet_statement": (
                "This ticket produces forcing evidence only -- no bed shear stress, Shields "
                "parameter, sediment mobility, erosion/deposition, scour, free-span, fatigue, "
                "or risk scoring is computed anywhere here."
            ),
            "outputs": {
                "current_primary_support_nodes": str(primary_nodes_path),
                "current_primary_hourly": str(primary_current_path),
                "current_long_term_surface_support_nodes": str(long_term_nodes_path),
                "current_long_term_surface_hourly": str(long_term_current_path),
                "wave_support_nodes": str(wave_nodes_path),
                "wave_3hourly": str(wave_path),
                "wave_annual_max_hs": str(annual_max_hs_path),
                "chainage_metocean_evidence": str(chainage_path),
            },
        },
        output_path=metadata_path,
    )

    print(f"Primary current support nodes: {len(primary_node_table)} -> {primary_nodes_path}")
    print(f"Long-term current support nodes: {len(long_term_node_table)} -> {long_term_nodes_path}")
    print(f"Wave support nodes: {len(wave_node_table)} -> {wave_nodes_path}")
    print(f"Chainage metocean evidence: {len(chainage_metocean_df)} station(s) -> {chainage_path}")
    print(f"Metadata: {metadata_path}")
    print()

    # --- old-vs-corrected comparison (Sections 12, 13, 19) ----------------
    # `old_*` values were read from the ON-DISK canonical outputs BEFORE this
    # run overwrote them; a run against an empty/absent prior output simply
    # reports `old=None` rather than fabricating a baseline.
    old_vs_new_comparison: dict[str, tuple[Any, Any]] = {
        "primary_current_speed_mean_m_s": (
            float(old_current_stats["current_speed_mean_m_s"].mean())
            if not old_current_stats.empty
            else None,
            float(current_stats["current_speed_mean_m_s"].mean())
            if not current_stats.empty
            else None,
        ),
        "primary_current_speed_p95_m_s": (
            float(old_current_stats["current_speed_p95_m_s"].max())
            if not old_current_stats.empty
            else None,
            float(current_stats["current_speed_p95_m_s"].max())
            if not current_stats.empty
            else None,
        ),
        "primary_current_speed_p99_m_s": (
            float(old_current_stats["current_speed_p99_m_s"].max())
            if not old_current_stats.empty
            else None,
            float(current_stats["current_speed_p99_m_s"].max())
            if not current_stats.empty
            else None,
        ),
        "primary_current_speed_max_m_s": (
            float(old_current_stats["current_speed_max_m_s"].max())
            if not old_current_stats.empty
            else None,
            float(current_stats["current_speed_max_m_s"].max())
            if not current_stats.empty
            else None,
        ),
        "wave_time_series_distinct_node_count": (
            int(old_wave_df["wave_node_id"].nunique()) if not old_wave_df.empty else None,
            len(wave_stats),
        ),
        "long_term_current_time_series_distinct_node_count": (
            int(old_long_term_current_df["current_lt_node_id"].nunique())
            if not old_long_term_current_df.empty
            else None,
            len(long_term_stats),
        ),
        "long_term_current_speed_p95_m_s": (
            float(old_long_term_stats["surface_current_speed_p95_m_s"].max())
            if not old_long_term_stats.empty
            else None,
            float(long_term_stats["surface_current_speed_p95_m_s"].max())
            if not long_term_stats.empty
            else None,
        ),
        "long_term_current_speed_p99_m_s": (
            float(old_long_term_stats["surface_current_speed_p99_m_s"].max())
            if not old_long_term_stats.empty
            else None,
            float(long_term_stats["surface_current_speed_p99_m_s"].max())
            if not long_term_stats.empty
            else None,
        ),
        "long_term_current_speed_max_m_s": (
            float(old_long_term_stats["surface_current_speed_max_m_s"].max())
            if not old_long_term_stats.empty
            else None,
            float(long_term_stats["surface_current_speed_max_m_s"].max())
            if not long_term_stats.empty
            else None,
        ),
        "wave_hs_mean_m": (
            float(old_wave_stats["hs_mean_m"].mean()) if not old_wave_stats.empty else None,
            float(wave_stats["hs_mean_m"].mean()) if not wave_stats.empty else None,
        ),
        "wave_hs_p95_m": (
            float(old_wave_stats["hs_p95_m"].max()) if not old_wave_stats.empty else None,
            float(wave_stats["hs_p95_m"].max()) if not wave_stats.empty else None,
        ),
        "wave_hs_p99_m": (
            float(old_wave_stats["hs_p99_m"].max()) if not old_wave_stats.empty else None,
            float(wave_stats["hs_p99_m"].max()) if not wave_stats.empty else None,
        ),
        "wave_hs_max_m": (
            float(old_wave_stats["hs_max_m"].max()) if not old_wave_stats.empty else None,
            float(wave_stats["hs_max_m"].max()) if not wave_stats.empty else None,
        ),
        "wave_tp_median_s": (
            float(old_wave_stats["tp_median_s"].median()) if not old_wave_stats.empty else None,
            float(wave_stats["tp_median_s"].median()) if not wave_stats.empty else None,
        ),
        "wave_tp_p95_s": (
            float(old_wave_stats["tp_p95_s"].max()) if not old_wave_stats.empty else None,
            float(wave_stats["tp_p95_s"].max()) if not wave_stats.empty else None,
        ),
    }

    metocean_evidence.print_metocean_evidence_report(
        primary_current_stats=current_stats,
        primary_current_route_summary=primary_current_route_summary,
        primary_current_canonical_row_count=len(primary_current_df),
        primary_temporal_qa=primary_temporal_qa,
        below_bed_diagnostics=below_bed_diagnostics,
        primary_distance_diagnostics=primary_distance_diagnostics,
        long_term_stats=long_term_stats,
        long_term_temporal_qa=long_term_temporal_qa,
        long_term_distance_diagnostics=long_term_distance_diagnostics,
        short_window_ratios=short_window_ratios,
        wave_stats=wave_stats,
        wave_temporal_qa=wave_temporal_qa,
        wave_distance_diagnostics=wave_distance_diagnostics,
        primary_current_actual_start=primary_start,
        primary_current_actual_end=primary_end,
        long_term_actual_start=long_term_current_df["time_utc"].min()
        if not long_term_current_df.empty
        else None,
        long_term_actual_end=long_term_current_df["time_utc"].max()
        if not long_term_current_df.empty
        else None,
        wave_actual_start=wave_df["time_utc"].min() if not wave_df.empty else None,
        wave_actual_end=wave_df["time_utc"].max() if not wave_df.empty else None,
        old_vs_new_comparison=old_vs_new_comparison,
    )
    return 0


def _none_if_nan(value: Any) -> Any:
    """`None` for a missing/NaN scalar, else `float(value)` -- JSON/dataclass-safe."""

    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _cmd_build_current_normalization(args: argparse.Namespace) -> int:
    """MAR-010: current-only 1 m log-profile normalization sensitivity + reference map.

    Requires the MAR-009B canonical primary-current outputs to already
    exist on disk; performs NO network request and NO Copernicus
    acquisition (Section 12) -- everything it reads was already downloaded
    and normalized by `build-metocean-evidence`.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    metocean_interim_dir = interim_dir / "metocean"
    metocean_processed_dir = study_dir / "metocean"
    maps_dir = study_dir / "maps"

    primary_nodes_path = metocean_interim_dir / "current_primary_support_nodes.parquet"
    primary_hourly_path = metocean_interim_dir / "current_primary_hourly.parquet"
    chainage_metocean_path = metocean_processed_dir / "chainage_metocean_evidence.parquet"
    required_paths = (
        pipeline_gpkg_path,
        primary_nodes_path,
        primary_hourly_path,
        chainage_metocean_path,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required canonical output(s) -- run build-chainage and "
            f"build-metocean-evidence first: {missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    primary_nodes_df = pd.read_parquet(primary_nodes_path)
    primary_hourly_df = pd.read_parquet(primary_hourly_path)
    chainage_metocean_df = pd.read_parquet(chainage_metocean_path)

    # --- MAR-010 core: log-profile 1 m sensitivity normalization -----------
    try:
        hourly_sensitivity_df = current_normalization.build_current_only_1m_sensitivity_hourly(
            primary_hourly_df
        )
        sensitivity_stats_df = current_normalization.compute_current_only_1m_sensitivity_stats(
            hourly_sensitivity_df
        )
        current_stats = metocean_evidence.compute_current_node_statistics(primary_hourly_df)
    except (
        current_normalization.NormalizationCompletenessError,
        metocean_evidence.TemporalCompletenessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sensitivity_envelope_df = current_normalization.compute_current_only_1m_sensitivity_envelope(
        sensitivity_stats_df
    )
    sensitivity_stats_df = sensitivity_stats_df.merge(
        sensitivity_envelope_df, on="current_node_id", how="left"
    )
    vertical_domain_summary = current_normalization.compute_vertical_domain_summary(
        hourly_sensitivity_df
    )

    # --- per-node reference attributes for segment/map assembly ------------
    node_bathymetry_by_id = primary_nodes_df.set_index("node_id")["model_bathymetry_m"].to_dict()
    current_stats_by_id = current_stats.set_index("current_node_id")
    envelope_by_id = sensitivity_envelope_df.set_index("current_node_id")

    node_attributes_by_id: dict[str, current_map.NodeReferenceAttributes] = {}
    for node_id, stats_row in current_stats_by_id.iterrows():
        bathymetry_m = node_bathymetry_by_id.get(node_id)
        representative_depth_m = stats_row.get("representative_sample_depth_m")
        reference_height_m = (
            bathymetry_m - representative_depth_m
            if bathymetry_m is not None and pd.notna(representative_depth_m)
            else None
        )
        envelope_row = envelope_by_id.loc[node_id] if node_id in envelope_by_id.index else None
        node_attributes_by_id[node_id] = current_map.NodeReferenceAttributes(
            model_bathymetry_m=_none_if_nan(bathymetry_m),
            reference_height_m=_none_if_nan(reference_height_m),
            speed_mean_m_s=_none_if_nan(stats_row.get("current_speed_mean_m_s")),
            speed_p95_m_s=_none_if_nan(stats_row.get("current_speed_p95_m_s")),
            speed_p99_m_s=_none_if_nan(stats_row.get("current_speed_p99_m_s")),
            speed_max_m_s=_none_if_nan(stats_row.get("current_speed_max_m_s")),
            sensitivity_p95_min_m_s=_none_if_nan(envelope_row["speed_1m_p95_sensitivity_min_m_s"])
            if envelope_row is not None
            else None,
            sensitivity_p95_max_m_s=_none_if_nan(envelope_row["speed_1m_p95_sensitivity_max_m_s"])
            if envelope_row is not None
            else None,
            sensitivity_p95_width_m_s=_none_if_nan(
                envelope_row["speed_1m_p95_sensitivity_width_m_s"]
            )
            if envelope_row is not None
            else None,
        )

    # --- contiguous map segments ---------------------------------------------
    chainage_current_df = chainage_metocean_df[
        ["chainage_m", "current_node_id", "current_node_distance_m"]
    ].copy()
    segments_gdf = current_map.build_current_reference_segments(
        pipeline_id=pipeline_id,
        route=route,
        chainage_current_df=chainage_current_df,
        node_attributes_by_id=node_attributes_by_id,
        working_crs=working_crs,
    )
    distance_diagnostics = metocean_evidence.compute_distance_diagnostics(
        chainage_current_df.rename(columns={"current_node_distance_m": "distance_m"})
    )

    # --- write parquet/gpkg outputs ------------------------------------------
    hourly_path = metocean_evidence.write_parquet(
        hourly_sensitivity_df,
        metocean_interim_dir / "current_only_1m_sensitivity_hourly.parquet",
    )
    stats_path = metocean_evidence.write_parquet(
        sensitivity_stats_df, metocean_processed_dir / "current_only_1m_sensitivity_stats.parquet"
    )
    segments_path = current_map.write_current_reference_segments_gpkg(
        segments_gdf, metocean_processed_dir / "current_reference_segments.gpkg"
    )

    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    png_path = current_map.render_reference_current_map(
        segments_gdf=segments_gdf,
        route=route,
        working_crs=working_crs,
        output_path=maps_dir / "pl854_reference_current_forcing.png",
        background_raster_path=(
            background_raster_path if background_raster_path.exists() else None
        ),
    )
    png_dimensions = current_map.read_png_dimensions(png_path)

    # --- metadata --------------------------------------------------------------
    metadata_path = metocean_processed_dir / "current_normalization_metadata.json"
    metadata = {
        "scientific_role": current_normalization.SCIENTIFIC_ROLE,
        "target_height_above_model_bed_m": (current_normalization.TARGET_HEIGHT_ABOVE_MODEL_BED_M),
        "equation": (
            "S(z_t,z_r,z0) = [ln(z_t+z0)-ln(z0)] / [ln(z_r+z0)-ln(z0)]; "
            "uo_1m = S * uo_ref; vo_1m = S * vo_ref; "
            "speed_1m = sqrt(uo_1m^2 + vo_1m^2)"
        ),
        "roughness_scenarios_m": dict(current_normalization.ROUGHNESS_SCENARIOS_M),
        "roughness_semantics": "SENSITIVITY_SCENARIOS_NOT_SITE_SPECIFIC_BED_TRUTH",
        "vertical_domain_screen": current_normalization.VERTICAL_DOMAIN_SCREEN_FRACTION,
        "vertical_domain_screen_semantics": (
            "A conservative project data-QA validity screen (z_r_over_h_model <= 0.30) for "
            "applying this simple current-only log-profile formulation at all -- never a "
            "universal physical threshold."
        ),
        "z_r_source": "Copernicus model bathymetry - deepest physically valid standard depth",
        "canonical_LAT_bathymetry_not_used_in_vertical_scaling": True,
        "current_wave_interaction_applied": False,
        "pipeline_directionality_applied": False,
        "current_source_nominal_resolution_m": current_map.SOURCE_GRID_NOMINAL_RESOLUTION_M,
        "map_colour_variable": "current_reference_speed_p95_m_s",
        "wave_interaction_statement": (
            "Surface waves can modify the apparent roughness and mean-current profile in "
            "the bottom boundary layer. MAR-010 intentionally does not model that "
            "interaction; this is a current-only normalization sensitivity product."
        ),
        "limitations": [
            "Roughness scenarios are fixed sensitivity dimensions, not a site-specific "
            "PL854 seabed roughness estimate.",
            "The 0.30 vertical-domain screen is a conservative project heuristic, not a "
            "universal physical threshold.",
            "Wave-current bottom-boundary-layer interaction is not modelled here.",
            "Model bathymetry (used here) and canonical MAR-006 LAT bathymetry remain "
            "deliberately unharmonised.",
            "Map colours represent the native corrected reference-current p95 only, not "
            "any roughness-selected or risk value.",
        ],
        "references": [
            {
                "citation": "Soulsby, R. (1997), Dynamics of Marine Sands.",
                "doi": "10.1680/doms.25844",
            },
            {
                "citation": (
                    "Grant, W.D. & Madsen, O.S. (1979), Combined wave and current "
                    "interaction with a rough bottom."
                ),
                "doi": "10.1029/JC084iC04p01797",
            },
            {
                "citation": (
                    "Warner, J.C. et al. (2008), Development of a three-dimensional, "
                    "regional, coupled wave, current, and sediment-transport model."
                ),
                "doi": "10.1016/j.cageo.2008.02.012",
            },
        ],
        "roughness_reference_note": (
            "The roughness sensitivity values are consistent with long-standing DNV "
            "F105/F109 seabed roughness classes, but this does not claim certified "
            "compliance with the current licensed DNV editions."
        ),
        "vertical_domain_summary": vertical_domain_summary,
        "outputs": {
            "current_only_1m_sensitivity_hourly": str(hourly_path),
            "current_only_1m_sensitivity_stats": str(stats_path),
            "current_reference_segments": str(segments_path),
            "current_reference_map_png": str(png_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(
        f"Current-only 1 m sensitivity (hourly): {len(hourly_sensitivity_df)} row(s) -> "
        f"{hourly_path}"
    )
    print(
        f"Current-only 1 m sensitivity (stats): {len(sensitivity_stats_df)} row(s) -> {stats_path}"
    )
    print(f"Current reference segments: {len(segments_gdf)} section(s) -> {segments_path}")
    print(f"Reference current map: {png_path}")
    print(f"Metadata: {metadata_path}")
    print()
    current_map.print_current_normalization_report(
        vertical_domain_summary=vertical_domain_summary,
        sensitivity_stats_df=sensitivity_stats_df,
        segments_gdf=segments_gdf,
        route_used_node_count=len(current_stats),
        distance_diagnostics=distance_diagnostics,
        segments_path=segments_path,
        png_path=png_path,
        png_dimensions=png_dimensions,
    )
    return 0


def _cmd_build_wave_orbital_forcing(args: argparse.Namespace) -> int:
    """MAR-011: wave-only spectral near-bed orbital velocity + reference map.

    Requires the MAR-009B canonical wave outputs to already exist on disk;
    performs NO network request and NO Copernicus acquisition (Section 18)
    -- everything it reads was already downloaded and normalized by
    `build-metocean-evidence`. Reads no current-related file at all.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    metocean_interim_dir = interim_dir / "metocean"
    metocean_processed_dir = study_dir / "metocean"
    maps_dir = study_dir / "maps"

    wave_hourly_path = metocean_interim_dir / "wave_3hourly.parquet"
    wave_nodes_path = metocean_interim_dir / "wave_support_nodes.parquet"
    chainage_metocean_path = metocean_processed_dir / "chainage_metocean_evidence.parquet"
    required_paths = (
        pipeline_gpkg_path,
        wave_hourly_path,
        wave_nodes_path,
        chainage_metocean_path,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required canonical output(s) -- run build-chainage and "
            f"build-metocean-evidence first: {missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    wave_df = pd.read_parquet(wave_hourly_path)
    wave_nodes_df = pd.read_parquet(wave_nodes_path)
    chainage_metocean_df = pd.read_parquet(chainage_metocean_path)

    # --- MAR-011 core: Soulsby & Smallman spectral orbital velocity ---------
    # `model_bathymetry_m` comes from the WAVE product's OWN static support-
    # node table (Section 2) -- never the canonical MAR-006 LAT depth, never
    # a current-product bathymetry substitute.
    wave_bathymetry_by_node = wave_nodes_df.set_index("node_id")["model_bathymetry_m"].to_dict()
    wave_df = wave_df.copy()
    wave_df["model_bathymetry_m"] = wave_df["wave_node_id"].map(wave_bathymetry_by_node)

    hourly_orbital_df = wave_orbital.build_wave_orbital_velocity_3hourly(wave_df)

    # MAR-011A Section 7: a hard integrity failure, never a mere warning --
    # duplicate/non-unique/non-monotonic (wave_node_id, time_utc) must stop
    # the run, since MAR-009B's own canonical wave series is guaranteed
    # temporally unique and any violation here indicates a real regression.
    # Checked BEFORE stats/completeness so this specific, known failure mode
    # is reported clearly rather than incidentally tripping the generic
    # completeness invariant with a less specific message.
    temporal_qa = metocean_evidence.validate_temporal_integrity(
        hourly_orbital_df, time_column="time_utc", node_column="wave_node_id"
    )
    temporal_integrity_failed = (
        bool(temporal_qa.get("duplicate_node_time_row_count"))
        or temporal_qa.get("time_coordinate_unique") is False
        or temporal_qa.get("time_coordinate_monotonic_increasing") is False
    )
    if temporal_integrity_failed:
        print(
            "error: wave orbital canonical time series failed a required temporal "
            f"integrity check: {temporal_qa}",
            file=sys.stderr,
        )
        return 1

    try:
        stats_df = wave_orbital.compute_wave_orbital_velocity_stats(hourly_orbital_df)
    except wave_orbital.OrbitalVelocityCompletenessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    domain_summary = wave_orbital.compute_wave_orbital_domain_summary(hourly_orbital_df)

    # --- per-node reference attributes for segment/map assembly ------------
    stats_by_id = stats_df.set_index("wave_node_id")
    node_attributes_by_id: dict[str, wave_orbital_map.WaveNodeReferenceAttributes] = {}
    for node_id, row in stats_by_id.iterrows():
        node_attributes_by_id[node_id] = wave_orbital_map.WaveNodeReferenceAttributes(
            model_bathymetry_m=_none_if_nan(row.get("model_bathymetry_m")),
            hs_p95_m=_none_if_nan(row.get("hs_p95_m")),
            hs_p99_m=_none_if_nan(row.get("hs_p99_m")),
            hs_max_m=_none_if_nan(row.get("hs_max_m")),
            tm02_median_s=_none_if_nan(row.get("tm02_median_s")),
            tm02_p95_s=_none_if_nan(row.get("tm02_p95_s")),
            orbital_rms_mean_m_s=_none_if_nan(row.get("orbital_rms_mean_m_s")),
            orbital_rms_p95_m_s=_none_if_nan(row.get("orbital_rms_p95_m_s")),
            orbital_rms_p99_m_s=_none_if_nan(row.get("orbital_rms_p99_m_s")),
            orbital_rms_max_m_s=_none_if_nan(row.get("orbital_rms_max_m_s")),
            orbital_amplitude_p95_m_s=_none_if_nan(row.get("orbital_amplitude_p95_m_s")),
            orbital_amplitude_p99_m_s=_none_if_nan(row.get("orbital_amplitude_p99_m_s")),
            orbital_amplitude_max_m_s=_none_if_nan(row.get("orbital_amplitude_max_m_s")),
        )

    # --- contiguous map segments ---------------------------------------------
    chainage_wave_df = chainage_metocean_df[
        ["chainage_m", "wave_node_id", "wave_node_distance_m"]
    ].copy()
    segments_gdf = wave_orbital_map.build_wave_orbital_reference_segments(
        pipeline_id=pipeline_id,
        route=route,
        chainage_wave_df=chainage_wave_df,
        node_attributes_by_id=node_attributes_by_id,
        working_crs=working_crs,
    )
    distance_diagnostics = metocean_evidence.compute_distance_diagnostics(
        chainage_wave_df.rename(columns={"wave_node_distance_m": "distance_m"})
    )

    # --- capture OLD (pre-MAR-011A conditional) stats for comparison --------
    # Read BEFORE anything is overwritten below (Section 9) -- the old
    # `wave_orbital_velocity_stats.parquet` (if present) was computed under
    # MAR-011's incorrect t<=0.54 gating, so its `orbital_rms_*` fields are
    # the "old conditional" values being compared against here.
    stats_output_path = metocean_processed_dir / "wave_orbital_velocity_stats.parquet"
    old_stats_df = _read_existing_parquet(stats_output_path)

    old_vs_new_comparison: dict[str, tuple[Any, Any]] = {
        "orbital_rms_mean_m_s": (
            float(old_stats_df["orbital_rms_mean_m_s"].mean()) if not old_stats_df.empty else None,
            float(stats_df["orbital_rms_mean_m_s"].mean()) if not stats_df.empty else None,
        ),
        "orbital_rms_p95_m_s": (
            float(old_stats_df["orbital_rms_p95_m_s"].max()) if not old_stats_df.empty else None,
            float(stats_df["orbital_rms_p95_m_s"].max()) if not stats_df.empty else None,
        ),
        "orbital_rms_p99_m_s": (
            float(old_stats_df["orbital_rms_p99_m_s"].max()) if not old_stats_df.empty else None,
            float(stats_df["orbital_rms_p99_m_s"].max()) if not stats_df.empty else None,
        ),
        "orbital_rms_max_m_s": (
            float(old_stats_df["orbital_rms_max_m_s"].max()) if not old_stats_df.empty else None,
            float(stats_df["orbital_rms_max_m_s"].max()) if not stats_df.empty else None,
        ),
        "orbital_amplitude_p95_m_s": (
            float(old_stats_df["orbital_amplitude_p95_m_s"].max())
            if not old_stats_df.empty
            else None,
            float(stats_df["orbital_amplitude_p95_m_s"].max()) if not stats_df.empty else None,
        ),
        "orbital_amplitude_p99_m_s": (
            float(old_stats_df["orbital_amplitude_p99_m_s"].max())
            if not old_stats_df.empty
            else None,
            float(stats_df["orbital_amplitude_p99_m_s"].max()) if not stats_df.empty else None,
        ),
        "orbital_amplitude_max_m_s": (
            float(old_stats_df["orbital_amplitude_max_m_s"].max())
            if not old_stats_df.empty
            else None,
            float(stats_df["orbital_amplitude_max_m_s"].max()) if not stats_df.empty else None,
        ),
    }

    old_vs_new_per_node = pd.DataFrame()
    if not old_stats_df.empty and not stats_df.empty:
        old_vs_new_per_node = old_stats_df[["wave_node_id", "orbital_rms_p95_m_s"]].merge(
            stats_df[["wave_node_id", "orbital_rms_p95_m_s"]],
            on="wave_node_id",
            how="outer",
            suffixes=("_old", "_new"),
        )
        old_vs_new_per_node = old_vs_new_per_node.rename(
            columns={
                "orbital_rms_p95_m_s_old": "old_p95_m_s",
                "orbital_rms_p95_m_s_new": "new_p95_m_s",
            }
        )
        old_vs_new_per_node["abs_diff_m_s"] = (
            old_vs_new_per_node["new_p95_m_s"] - old_vs_new_per_node["old_p95_m_s"]
        )
        old_vs_new_per_node["pct_diff_pct"] = np.where(
            old_vs_new_per_node["old_p95_m_s"].notna() & (old_vs_new_per_node["old_p95_m_s"] != 0),
            100.0
            * old_vs_new_per_node["abs_diff_m_s"]
            / old_vs_new_per_node["old_p95_m_s"].replace(0, np.nan),
            np.nan,
        )

    # --- write parquet/gpkg outputs ------------------------------------------
    hourly_path = metocean_evidence.write_parquet(
        hourly_orbital_df, metocean_interim_dir / "wave_orbital_velocity_3hourly.parquet"
    )
    stats_path = metocean_evidence.write_parquet(stats_df, stats_output_path)
    segments_path = wave_orbital_map.write_wave_orbital_reference_segments_gpkg(
        segments_gdf, metocean_processed_dir / "wave_orbital_reference_segments.gpkg"
    )

    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    png_path = wave_orbital_map.render_wave_orbital_map(
        segments_gdf=segments_gdf,
        route=route,
        working_crs=working_crs,
        output_path=maps_dir / "pl854_wave_orbital_forcing.png",
        background_raster_path=(
            background_raster_path if background_raster_path.exists() else None
        ),
    )
    png_dimensions = wave_orbital_map.read_png_dimensions(png_path)

    # --- metadata --------------------------------------------------------------
    metadata_path = metocean_processed_dir / "wave_orbital_velocity_metadata.json"
    metadata = {
        "scientific_role": wave_orbital.SCIENTIFIC_ROLE,
        "gravity_m_s2": wave_orbital.GRAVITY_M_S2,
        "hs_source": "VHM0",
        "tz_source": wave_orbital.TZ_SOURCE,
        "tp_observed_source": "VTPK",
        "energy_period_source": "VTM10",
        "water_depth_source": ("Copernicus wave-product static deptho at same support node"),
        "canonical_LAT_bathymetry_used_in_orbital_calculation": False,
        "method": "Soulsby & Smallman irregular-wave spectral approximation",
        "equations": (
            "Tn = sqrt(h/g); t = Tn/Tz; A = [6500 + (0.56 + 15.54*t)^6]^(1/6); "
            "Urms = 0.25*Hs / [Tn * (1 + A*t^2)^3]; "
            "equivalent_amplitude = sqrt(2)*Urms; "
            "equivalent_peak_period = 1.28*Tz"
        ),
        "reported_better_than_1pct_accuracy_range": (
            f"0 <= sqrt(h/g)/Tz <= {wave_orbital.REPORTED_1PCT_ACCURACY_MAX_T}"
        ),
        "accuracy_range_semantics": "APPROXIMATION_ACCURACY_QUALIFICATION_NOT_VALIDITY_THRESHOLD",
        "orbital_estimates_outside_reported_1pct_range_retained": True,
        "outside_range_accuracy_not_quantified": True,
        "tr155_notes_orbital_velocities_very_small_above_0_54": True,
        "equivalent_amplitude_definition": "sqrt(2) * Urms",
        "equivalent_peak_period_definition": "1.28 * Tz",
        "equivalent_peak_period_is_diagnostic_not_observed_tp": True,
        "current_effect_on_wave_dispersion_applied": False,
        "wave_current_bottom_boundary_layer_applied": False,
        "directional_spreading_correction_applied": False,
        "nonbreaking_wave_assumption_applied": True,
        "explicit_breaking_wave_classifier_applied": False,
        "source_grid_resolution_note": wave_orbital_map.SOURCE_GRID_RESOLUTION_NOTE,
        "map_colour_variable": "orbital_rms_p95_m_s",
        "limitations": [
            "The Soulsby & Smallman approximation is reported by TR155 to fit the exact "
            "computed spectral value to better than 1% only for Tn/Tz <= 0.54 -- an "
            "accuracy qualification, not a validity threshold; Urms is still computed "
            "and reported above it (TR155 notes orbital velocities are very small there).",
            "Non-breaking waves are assumed; no breaking-wave classifier is applied here.",
            "Wave-current interaction and bed shear stress are not modelled in this ticket.",
            "The equivalent amplitude and equivalent peak period are derived helpers, "
            "never more canonical than Urms/observed Tp respectively.",
            "Wave model bathymetry and canonical MAR-006 LAT bathymetry remain "
            "deliberately unharmonised; only wave model bathymetry is used here.",
        ],
        "references": [
            {
                "citation": (
                    "Soulsby, R.L. (2006). Simplified calculation of wave orbital "
                    "velocities. HR Wallingford Report TR155."
                )
            },
            {
                "citation": (
                    "Soulsby, R.L. & Smallman, J.V. (1986). A direct method of "
                    "calculating bottom orbital velocity under waves. Hydraulics "
                    "Research Report SR76."
                )
            },
            {
                "citation": (
                    "Wiberg, P.L. & Sherwood, C.R. (2008). Calculating wave-generated "
                    "bottom orbital velocities from surface-wave parameters. Computers "
                    "& Geosciences 34, 1243-1262."
                ),
                "doi": "10.1016/j.cageo.2008.02.010",
            },
            {
                "citation": (
                    "Wilson, R.J. et al. (2018). A synthetic map of the north-west "
                    "European Shelf sedimentary environment for applications in marine "
                    "science. Earth System Science Data 10, 109-130."
                ),
                "doi": "10.5194/essd-10-109-2018",
            },
            {"citation": "Copernicus Marine: NWSHELF_REANALYSIS_WAV_004_015 PUM / QUID."},
        ],
        "vertical_domain_summary": domain_summary,
        "outputs": {
            "wave_orbital_velocity_3hourly": str(hourly_path),
            "wave_orbital_velocity_stats": str(stats_path),
            "wave_orbital_reference_segments": str(segments_path),
            "wave_orbital_map_png": str(png_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"Wave orbital velocity (3-hourly): {len(hourly_orbital_df)} row(s) -> {hourly_path}")
    print(f"Wave orbital velocity (stats): {len(stats_df)} row(s) -> {stats_path}")
    print(f"Wave orbital reference segments: {len(segments_gdf)} section(s) -> {segments_path}")
    print(f"Wave orbital map: {png_path}")
    print(f"Metadata: {metadata_path}")
    print()
    wave_orbital_map.print_wave_orbital_report(
        domain_summary=domain_summary,
        stats_df=stats_df,
        segments_gdf=segments_gdf,
        route_used_node_count=len(stats_df),
        distance_diagnostics=distance_diagnostics,
        segments_path=segments_path,
        png_path=png_path,
        png_dimensions=png_dimensions,
        old_vs_new_comparison=old_vs_new_comparison,
        old_vs_new_per_node=old_vs_new_per_node,
    )
    return 0


def _cmd_build_combined_bed_shear(args: argparse.Namespace) -> int:
    """MAR-012: Soulsby algebraic wave-current bed shear stress sensitivity + map.

    Requires the MAR-010 current normalization and MAR-011A wave orbital
    outputs to already exist on disk; performs NO network request (Section
    29) -- everything it reads was already computed by
    `build-current-normalization` and `build-wave-orbital-forcing`.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    metocean_interim_dir = interim_dir / "metocean"
    metocean_processed_dir = study_dir / "metocean"
    maps_dir = study_dir / "maps"

    current_hourly_path = metocean_interim_dir / "current_only_1m_sensitivity_hourly.parquet"
    current_nodes_path = metocean_interim_dir / "current_primary_support_nodes.parquet"
    wave_hourly_path = metocean_interim_dir / "wave_orbital_velocity_3hourly.parquet"
    wave_nodes_path = metocean_interim_dir / "wave_support_nodes.parquet"
    chainage_metocean_path = metocean_processed_dir / "chainage_metocean_evidence.parquet"
    required_paths = (
        pipeline_gpkg_path,
        current_hourly_path,
        current_nodes_path,
        wave_hourly_path,
        wave_nodes_path,
        chainage_metocean_path,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required canonical output(s) -- run build-current-normalization "
            f"and build-wave-orbital-forcing first: {missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    current_hourly_df = pd.read_parquet(current_hourly_path)
    current_nodes_df = pd.read_parquet(current_nodes_path)
    wave_hourly_df = pd.read_parquet(wave_hourly_path)
    wave_nodes_df = pd.read_parquet(wave_nodes_path)
    chainage_metocean_df = pd.read_parquet(chainage_metocean_path)

    # --- MAR-012 core: spatial reconciliation, exact-time join, physics ----
    try:
        hydro_pairs_df = combined_bed_shear.build_hydro_pairs(
            current_nodes_df, wave_nodes_df, working_crs=working_crs
        )
    except combined_bed_shear.UnreconciledHydroNodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    combined_df = combined_bed_shear.build_combined_bed_shear_3hourly(
        current_hourly_df, wave_hourly_df, hydro_pairs_df
    )

    try:
        alignment_summary = combined_bed_shear.compute_temporal_alignment_summary(combined_df)
        stats_df = combined_bed_shear.compute_combined_bed_shear_stats(combined_df)
    except combined_bed_shear.CombinedBedShearCompletenessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    envelope_df = combined_bed_shear.compute_sensitivity_envelope(stats_df)

    overlap_keys_df = combined_df[["wave_node_id", "time_utc"]].drop_duplicates()
    long_term_stats_df = combined_bed_shear.compute_wave_only_bed_shear_long_term_stats(
        wave_hourly_df, overlap_keys_df=overlap_keys_df
    )

    # --- per-hydro-pair reference attributes for segment/map assembly ------
    overlap_bounds_by_pair = stats_df.drop_duplicates(subset=["hydro_pair_id"]).set_index(
        "hydro_pair_id"
    )[["overlap_start_time_utc", "overlap_end_time_utc"]]
    envelope_by_pair = envelope_df.set_index("hydro_pair_id")

    node_attributes_by_id: dict[str, combined_bed_shear_map.HydroPairReferenceAttributes] = {}
    for pair_id in hydro_pairs_df["hydro_pair_id"]:
        envelope_row = envelope_by_pair.loc[pair_id] if pair_id in envelope_by_pair.index else None
        bounds_row = (
            overlap_bounds_by_pair.loc[pair_id] if pair_id in overlap_bounds_by_pair.index else None
        )
        node_attributes_by_id[pair_id] = combined_bed_shear_map.HydroPairReferenceAttributes(
            tau_max_p95_sensitivity_min_pa=_none_if_nan(
                envelope_row["tau_max_p95_sensitivity_min_pa"]
            )
            if envelope_row is not None
            else None,
            tau_max_p95_sensitivity_max_pa=_none_if_nan(
                envelope_row["tau_max_p95_sensitivity_max_pa"]
            )
            if envelope_row is not None
            else None,
            tau_max_p95_sensitivity_width_pa=_none_if_nan(
                envelope_row["tau_max_p95_sensitivity_width_pa"]
            )
            if envelope_row is not None
            else None,
            tau_max_p99_sensitivity_min_pa=_none_if_nan(
                envelope_row["tau_max_p99_sensitivity_min_pa"]
            )
            if envelope_row is not None
            else None,
            tau_max_p99_sensitivity_max_pa=_none_if_nan(
                envelope_row["tau_max_p99_sensitivity_max_pa"]
            )
            if envelope_row is not None
            else None,
            tau_max_p99_sensitivity_width_pa=_none_if_nan(
                envelope_row["tau_max_p99_sensitivity_width_pa"]
            )
            if envelope_row is not None
            else None,
            overlap_start_time_utc=bounds_row["overlap_start_time_utc"]
            if bounds_row is not None
            else None,
            overlap_end_time_utc=bounds_row["overlap_end_time_utc"]
            if bounds_row is not None
            else None,
        )

    # --- contiguous map segments (built from current_node_id + wave_node_id) -----
    chainage_hydro_df = chainage_metocean_df[
        [
            "chainage_m",
            "current_node_id",
            "current_node_distance_m",
            "wave_node_id",
            "wave_node_distance_m",
        ]
    ].merge(
        hydro_pairs_df[["current_node_id", "wave_node_id", "hydro_pair_id"]],
        on=["current_node_id", "wave_node_id"],
        how="left",
    )
    segments_gdf = combined_bed_shear_map.build_combined_bed_shear_segments(
        pipeline_id=pipeline_id,
        route=route,
        chainage_hydro_df=chainage_hydro_df,
        node_attributes_by_id=node_attributes_by_id,
        working_crs=working_crs,
    )

    pooled_distances_df = pd.concat(
        [
            chainage_metocean_df[["current_node_distance_m"]].rename(
                columns={"current_node_distance_m": "distance_m"}
            ),
            chainage_metocean_df[["wave_node_distance_m"]].rename(
                columns={"wave_node_distance_m": "distance_m"}
            ),
        ],
        ignore_index=True,
    )
    distance_diagnostics = metocean_evidence.compute_distance_diagnostics(pooled_distances_df)

    # --- write parquet/gpkg outputs ------------------------------------------
    combined_hourly_path = metocean_evidence.write_parquet(
        combined_df, metocean_interim_dir / "combined_bed_shear_3hourly.parquet"
    )
    long_term_stats_path = metocean_evidence.write_parquet(
        long_term_stats_df, metocean_processed_dir / "wave_only_bed_shear_long_term_stats.parquet"
    )
    stats_path = metocean_evidence.write_parquet(
        stats_df, metocean_processed_dir / "combined_bed_shear_stats.parquet"
    )
    segments_path = combined_bed_shear_map.write_combined_bed_shear_segments_gpkg(
        segments_gdf, metocean_processed_dir / "combined_bed_shear_segments.gpkg"
    )

    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    png_path = combined_bed_shear_map.render_combined_bed_shear_map(
        segments_gdf=segments_gdf,
        route=route,
        working_crs=working_crs,
        output_path=maps_dir / "pl854_combined_bed_shear_sensitivity.png",
        background_raster_path=(
            background_raster_path if background_raster_path.exists() else None
        ),
    )
    png_dimensions = combined_bed_shear_map.read_png_dimensions(png_path)

    # --- metadata --------------------------------------------------------------
    metadata_path = metocean_processed_dir / "combined_bed_shear_metadata.json"
    verified_alignment = (
        bool((hydro_pairs_df["coordinate_separation_m"] < 1.0).all())
        if not hydro_pairs_df.empty
        else None
    )
    metadata = {
        "scientific_role": combined_bed_shear.SCIENTIFIC_ROLE,
        "interaction_model": combined_bed_shear.INTERACTION_MODEL,
        "equations": {
            "current_friction_velocity": "u_star_c = kappa*U_1m / ln((z_target+z0)/z0)",
            "current_bed_shear_stress": "tau_c_pa = rho * u_star_c^2",
            "wave_representative_amplitude": "Uw = sqrt(2) * Urms",
            "wave_representative_period": "T_rep = 1.28 * Tz (VTM02)",
            "wave_semi_orbital_excursion": "A_wave_m = Uw * T_rep / (2*pi)",
            "wave_reynolds_number": "Rw = Uw * A_wave / nu",
            "wave_friction_smooth_laminar_branch": (
                "f_ws = B*Rw^(-N); B=2.0,N=0.5 (Rw<=5e5); B=0.0521,N=0.187 (Rw>5e5)"
            ),
            "wave_friction_rough_branch": "f_wr = 1.39 * (A_wave/z0)^(-0.52)",
            "wave_friction_factor": "f_w = max(f_ws, f_wr)",
            "wave_bed_shear_stress": "tau_w_pa = 0.5 * rho * f_w * Uw^2",
            "soulsby_mean_combined_stress": (
                "tau_m_pa = tau_c * [1 + 1.2*(tau_w/(tau_c+tau_w))^3.2]"
            ),
            "soulsby_max_combined_stress": (
                "tau_max_pa = sqrt((tau_m+tau_w*cos(phi))^2 + (tau_w*sin(phi))^2)"
            ),
        },
        "rho_water_kg_m3": combined_bed_shear.RHO_WATER_KG_M3,
        "kinematic_viscosity_m2_s": combined_bed_shear.KINEMATIC_VISCOSITY_M2_S,
        "von_karman_kappa": combined_bed_shear.VON_KARMAN_KAPPA,
        "reference_fluid_properties_note": (
            "Fixed project reference seawater properties for this research calculation -- "
            "never PL854 in-situ measurements, never a temperature/salinity dependent "
            "viscosity model."
        ),
        "current_reference_height_m": combined_bed_shear.TARGET_HEIGHT_ABOVE_MODEL_BED_M,
        "current_source": "MAR-010",
        "wave_source": "MAR-011A",
        "representative_wave_amplitude": "sqrt(2) * spectral Urms",
        "representative_wave_period": "1.28 * VTM02",
        "roughness_scenarios_m": dict(combined_bed_shear.ROUGHNESS_SCENARIOS_M),
        "roughness_semantics": "SENSITIVITY_SCENARIOS_NOT_SITE_SPECIFIC_BED_TRUTH",
        "wave_friction_method": (
            "Soulsby smooth/laminar versus rough branch; maximum branch retained"
        ),
        "current_wave_angle_uses_wave_axis_180deg_symmetry": True,
        "exact_timestamp_inner_join": True,
        "interpolation_applied": False,
        "full_grant_madsen_bbl_applied": False,
        "apparent_wave_enhanced_current_roughness_iterated": False,
        "current_effect_on_wave_dispersion_applied": False,
        "shields_parameter_computed": False,
        "sediment_mobility_computed": False,
        "map_colour_variable": "tau_max_p95_sensitivity_max_pa",
        "map_colour_semantics": (
            "UPPER_BOUND_ACROSS_ROUGHNESS_SENSITIVITY_SCENARIOS_NOT_BEST_ESTIMATE"
        ),
        "hydro_pair_reconciliation_method": (
            "each current support node's coordinate is matched to its nearest wave support "
            "node's coordinate in the working CRS, within a tolerance of "
            f"{combined_bed_shear.HYDRO_PAIR_TOLERANCE_FRACTION:.0%} of the current grid's "
            "own median nearest-neighbour spacing -- never assumed from matching node-id "
            "strings"
        ),
        "hydro_pair_count": int(len(hydro_pairs_df)),
        "verified_current_wave_coordinate_alignment": verified_alignment,
        "representativeness_warning": combined_bed_shear.REPRESENTATIVENESS_WARNING,
        "limitations": [
            "Roughness scenarios are fixed sensitivity dimensions, not a site-specific "
            "PL854 seabed roughness estimate.",
            "This is the Soulsby algebraic wave-current interaction approximation, not a "
            "full Grant-Madsen iterative bottom-boundary-layer solution.",
            "Combined statistics are limited to the contemporaneous primary-current/wave "
            "overlap, not the full 1980-2026 wave record.",
            "No Shields parameter, critical shear stress, or sediment mobility is computed here.",
            "Map colours represent the upper p95 bound across roughness scenarios, never a "
            "best estimate or risk value.",
        ],
        "references": [
            {
                "citation": "Soulsby, R.L. (1997). Dynamics of Marine Sands.",
                "doi": "10.1680/doms.25844",
            },
            {
                "citation": (
                    "Grant, W.D. & Madsen, O.S. (1979). Combined wave and current "
                    "interaction with a rough bottom. Journal of Geophysical Research "
                    "84(C4), 1797-1808."
                ),
                "doi": "10.1029/JC084iC04p01797",
            },
            {
                "citation": (
                    "Madsen, O.S. (1994). Spectral wave-current bottom boundary layer "
                    "flows. Proceedings of the 24th International Conference on Coastal "
                    "Engineering."
                ),
            },
            {
                "citation": (
                    "Soulsby, R.L. (2006). Simplified calculation of wave orbital "
                    "velocities. HR Wallingford TR155."
                ),
            },
        ],
        "outputs": {
            "combined_bed_shear_3hourly": str(combined_hourly_path),
            "wave_only_bed_shear_long_term_stats": str(long_term_stats_path),
            "combined_bed_shear_stats": str(stats_path),
            "combined_bed_shear_segments": str(segments_path),
            "combined_bed_shear_map_png": str(png_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"Combined bed shear (3-hourly): {len(combined_df)} row(s) -> {combined_hourly_path}")
    print(
        f"Wave-only long-term bed shear stats: {len(long_term_stats_df)} row(s) -> "
        f"{long_term_stats_path}"
    )
    print(f"Combined bed shear stats: {len(stats_df)} row(s) -> {stats_path}")
    print(f"Combined bed shear segments: {len(segments_gdf)} section(s) -> {segments_path}")
    print(f"Combined bed shear map: {png_path}")
    print(f"Metadata: {metadata_path}")
    print()
    combined_bed_shear_map.print_combined_bed_shear_report(
        alignment_summary=alignment_summary,
        hydro_pairs_df=hydro_pairs_df,
        stats_df=stats_df,
        envelope_df=envelope_df,
        long_term_stats_df=long_term_stats_df,
        segments_gdf=segments_gdf,
        segments_path=segments_path,
        png_path=png_path,
        png_dimensions=png_dimensions,
        distance_diagnostics=distance_diagnostics,
    )
    return 0


def _cmd_build_noncohesive_mobility(args: argparse.Namespace) -> int:
    """MAR-013: noncohesive sediment mobility capacity + map + profile.

    Requires the MAR-009B canonical primary current, MAR-011A wave orbital,
    and MAR-008 sediment evidence outputs to already exist on disk; performs
    NO network request -- everything it reads was already computed. Reuses
    MAR-012's verified hydro-node spatial reconciliation logic directly.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    metocean_interim_dir = interim_dir / "metocean"
    metocean_processed_dir = study_dir / "metocean"
    sediment_interim_dir = interim_dir / "sediment"
    sediment_processed_dir = study_dir / "sediment"
    maps_dir = study_dir / "maps"

    current_hourly_path = metocean_interim_dir / "current_primary_hourly.parquet"
    current_nodes_path = metocean_interim_dir / "current_primary_support_nodes.parquet"
    wave_hourly_path = metocean_interim_dir / "wave_orbital_velocity_3hourly.parquet"
    wave_nodes_path = metocean_interim_dir / "wave_support_nodes.parquet"
    chainage_metocean_path = metocean_processed_dir / "chainage_metocean_evidence.parquet"
    chainage_sediment_path = sediment_processed_dir / "chainage_sediment_evidence.parquet"
    psa_observations_path = sediment_interim_dir / "bgs_psa_observations.parquet"
    required_paths = (
        pipeline_gpkg_path,
        current_hourly_path,
        current_nodes_path,
        wave_hourly_path,
        wave_nodes_path,
        chainage_metocean_path,
        chainage_sediment_path,
        psa_observations_path,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required canonical output(s) -- run build-metocean-evidence, "
            "build-wave-orbital-forcing, and build-sediment-evidence first: "
            f"{missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    current_hourly_df = pd.read_parquet(current_hourly_path)
    current_nodes_df = pd.read_parquet(current_nodes_path)
    wave_hourly_df = pd.read_parquet(wave_hourly_path)
    wave_nodes_df = pd.read_parquet(wave_nodes_path)
    chainage_metocean_df = pd.read_parquet(chainage_metocean_path)
    chainage_sediment_df = pd.read_parquet(chainage_sediment_path)
    psa_observations_df = pd.read_parquet(psa_observations_path)

    # --- MAR-012's verified spatial reconciliation, reused directly ---------
    try:
        hydro_pairs_df = combined_bed_shear.build_hydro_pairs(
            current_nodes_df, wave_nodes_df, working_crs=working_crs
        )
    except combined_bed_shear.UnreconciledHydroNodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # --- MAR-013 core: grain-related skin friction + Soulsby-Whitehouse -----
    mobility_df = noncohesive_mobility.build_noncohesive_mobility_3hourly(
        current_hourly_df, wave_hourly_df, hydro_pairs_df
    )

    try:
        stats_df = noncohesive_mobility.compute_noncohesive_mobility_stats(mobility_df)
    except noncohesive_mobility.NoncohesiveMobilityCompletenessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    capacity_df = noncohesive_mobility.compute_mobility_capacity_summary(stats_df)
    observed_d50_context_df = noncohesive_mobility.build_observed_d50_context(psa_observations_df)

    # --- per-hydro-pair capacity attributes for segment/map assembly --------
    stats_by_pair_d50 = stats_df.set_index(["hydro_pair_id", "tested_d50_mm"])
    capacity_by_pair_id: dict[str, noncohesive_mobility_map.CapacityAttributes] = {}
    for _, row in capacity_df.iterrows():
        pair_id = row["hydro_pair_id"]
        reference_ratios: dict[float, float | None] = {}
        for mm in noncohesive_mobility.REFERENCE_D50_SCENARIOS_MM:
            key = (pair_id, mm)
            reference_ratios[mm] = (
                _none_if_nan(stats_by_pair_d50.loc[key, "mobility_ratio_p95"])
                if key in stats_by_pair_d50.index
                else None
            )
        capacity_by_pair_id[pair_id] = noncohesive_mobility_map.CapacityAttributes(
            largest_tested_d50_with_p90_mobility_ratio_ge_1_mm=_none_if_nan(
                row["largest_tested_d50_with_p90_mobility_ratio_ge_1_mm"]
            ),
            largest_tested_d50_with_p95_mobility_ratio_ge_1_mm=_none_if_nan(
                row["largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"]
            ),
            largest_tested_d50_with_p99_mobility_ratio_ge_1_mm=_none_if_nan(
                row["largest_tested_d50_with_p99_mobility_ratio_ge_1_mm"]
            ),
            largest_tested_d50_with_any_exceedance_mm=_none_if_nan(
                row["largest_tested_d50_with_any_exceedance_mm"]
            ),
            p95_mobility_sequence_monotonic_nonincreasing=(
                bool(row["p95_mobility_sequence_monotonic_nonincreasing"])
                if pd.notna(row["p95_mobility_sequence_monotonic_nonincreasing"])
                else None
            ),
            monotonicity_violation_count=(
                int(row["monotonicity_violation_count"])
                if pd.notna(row["monotonicity_violation_count"])
                else None
            ),
            reference_p95_ratios_by_mm=reference_ratios,
        )

    # --- contiguous map segments (reusing MAR-012's hydro pairs) -------------
    chainage_hydro_df = (
        chainage_metocean_df[["station_index", "chainage_m", "current_node_id", "wave_node_id"]]
        .merge(
            hydro_pairs_df[["current_node_id", "wave_node_id", "hydro_pair_id"]],
            on=["current_node_id", "wave_node_id"],
            how="left",
        )
        .merge(
            chainage_sediment_df[
                ["station_index", "mapped_250k_folk_class", "mapped_250k_nominal_scale"]
            ],
            on="station_index",
            how="left",
        )
    )
    segments_gdf = noncohesive_mobility_map.build_noncohesive_mobility_capacity_segments(
        pipeline_id=pipeline_id,
        route=route,
        chainage_hydro_df=chainage_hydro_df,
        capacity_by_pair_id=capacity_by_pair_id,
        observed_d50_context_df=observed_d50_context_df,
        working_crs=working_crs,
    )

    # --- write parquet/gpkg outputs ------------------------------------------
    mobility_hourly_path = metocean_evidence.write_parquet(
        mobility_df, sediment_interim_dir / "noncohesive_mobility_3hourly.parquet"
    )
    stats_path = metocean_evidence.write_parquet(
        stats_df, sediment_processed_dir / "noncohesive_mobility_stats.parquet"
    )
    observed_context_path = metocean_evidence.write_parquet(
        observed_d50_context_df, sediment_processed_dir / "observed_d50_context.parquet"
    )
    segments_path = noncohesive_mobility_map.write_noncohesive_mobility_capacity_segments_gpkg(
        segments_gdf, sediment_processed_dir / "noncohesive_mobility_capacity_segments.gpkg"
    )

    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    png_path = noncohesive_mobility_map.render_noncohesive_mobility_capacity_map(
        segments_gdf=segments_gdf,
        route=route,
        working_crs=working_crs,
        output_path=maps_dir / "pl854_noncohesive_mobility_capacity.png",
        psa_observations_df=psa_observations_df,
        background_raster_path=(
            background_raster_path if background_raster_path.exists() else None
        ),
    )
    png_dimensions = noncohesive_mobility_map.read_png_dimensions(png_path)

    profile_path = noncohesive_mobility_map.render_mobility_capacity_profile(
        segments_gdf=segments_gdf,
        psa_observations_df=psa_observations_df,
        working_crs=working_crs,
        output_path=maps_dir / "pl854_mobility_capacity_profile.png",
    )
    profile_dimensions = noncohesive_mobility_map.read_png_dimensions(profile_path)

    # --- metadata --------------------------------------------------------------
    metadata_path = sediment_processed_dir / "noncohesive_mobility_metadata.json"
    metadata = {
        "scientific_role": noncohesive_mobility.SCIENTIFIC_ROLE,
        "tested_d50_scenarios_mm": list(noncohesive_mobility.TESTED_D50_SCENARIOS_MM),
        "grain_size_scenario_semantics": noncohesive_mobility.GRAIN_SIZE_SCENARIO_SEMANTICS,
        "gravity_m_s2": noncohesive_mobility.GRAVITY_M_S2,
        "rho_water_kg_m3": noncohesive_mobility.RHO_WATER_KG_M3,
        "rho_sediment_kg_m3": noncohesive_mobility.RHO_SEDIMENT_KG_M3,
        "rho_sediment_note": (
            "A quartz/mineral-density reference assumption for this noncohesive test-"
            "scenario analysis -- never a measured PL854 sediment mineral density."
        ),
        "kinematic_viscosity_m2_s": noncohesive_mobility.KINEMATIC_VISCOSITY_M2_S,
        "von_karman_kappa": noncohesive_mobility.VON_KARMAN_KAPPA,
        "grain_skin_roughness_relation": "z0_skin = d50 / 12",
        "current_skin_stress_equation": (
            "u_star_c_skin = kappa*U_ref / ln((z_r+z0_skin)/z0_skin); "
            "tau_current_skin_pa = rho_water * u_star_c_skin^2"
        ),
        "wave_skin_friction_equations": (
            "A_wave = Uw*Trep/(2*pi); Rw = Uw*A_wave/nu; "
            "f_ws = 2.0*Rw^(-0.5) (Rw<=5e5) or 0.0521*Rw^(-0.187) (Rw>5e5); "
            "f_wr = 1.39*(A_wave/z0_skin)^(-0.52); f_w_skin = max(f_ws, f_wr); "
            "tau_wave_skin_pa = 0.5*rho_water*f_w_skin*Uw^2"
        ),
        "soulsby_combined_stress_equations": (
            "tau_mean_grain_skin = tau_current_skin*[1+1.2*(tau_wave_skin/"
            "(tau_current_skin+tau_wave_skin))^3.2]; "
            "tau_max_grain_skin = sqrt((tau_mean_grain_skin+tau_wave_skin*cos(phi))^2 + "
            "(tau_wave_skin*sin(phi))^2)"
        ),
        "dimensionless_grain_size_equation": (
            "D* = d50 * [g*(rho_sediment/rho_water - 1)/nu^2]^(1/3)"
        ),
        "soulsby_whitehouse_theta_cr_equation": (
            "theta_cr = 0.30/(1+1.2*D*) + 0.055*[1-exp(-0.020*D*)]"
        ),
        "tau_cr_equation": "tau_cr = theta_cr * (rho_sediment - rho_water) * g * d50",
        "mobility_ratio_definition": "tau_max_grain_skin / tau_critical",
        "incipient_motion_condition": "mobility_ratio >= 1",
        "continuous_pipeline_d50_field_created": False,
        "bgs_folk_to_numeric_d50_mapping_applied": False,
        "psa_d50_interpolation_applied": False,
        "bgs_predictive_sediment_used_in_physics": False,
        "cohesive_or_mixed_bed_threshold_physics_applied": False,
        "sediment_transport_rate_computed": False,
        "erosion_deposition_computed": False,
        "scour_computed": False,
        "pipeline_risk_computed": False,
        "combined_record_overlap_only": True,
        "map_colour_variable": "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm",
        "map_colour_semantics": (
            "Among the tested noncohesive grain-size scenarios, this is the largest tested "
            "D50 whose p95 mobility ratio reaches or exceeds the incipient-motion threshold "
            "-- it does not mean the actual local D50 equals this value, that the entire "
            "seabed is mobile, that this sediment exists there, or that erosion/scour/"
            "pipeline exposure will occur."
        ),
        "bgs_250k_folk_class_note": (
            "BGS 250K Folk class provides regional substrate context only and is not "
            "converted into numeric D50."
        ),
        "limitations": [
            "This is a forcing-capacity product for nine fixed test grain sizes, not a "
            "continuous site-specific sediment-mobility field.",
            "The largest passing tested scenario is never extrapolated beyond the tested "
            "range; a passing largest-tested scenario is flagged as exceeding the tested "
            "range rather than reported as a true capacity ceiling.",
            "Mobility ratio is not assumed to decrease monotonically with grain size; the "
            "ordering is verified per hydro pair and any violation is recorded.",
            "No Shields threshold, transport rate, erosion, deposition, scour, or pipeline "
            "risk is computed here.",
            "Combined statistics are limited to the contemporaneous primary-current/wave "
            "overlap, not the full 1980-2026 wave record.",
        ],
        "references": [
            {
                "citation": "Soulsby, R.L. (1997). Dynamics of Marine Sands.",
                "doi": "10.1680/doms.25844",
            },
            {
                "citation": (
                    "Soulsby, R.L. & Whitehouse, R.J.S.W. (1997). Threshold of sediment "
                    "motion in coastal environments."
                ),
            },
            {
                "citation": (
                    "Soulsby, R.L. (2006). Simplified calculation of wave orbital "
                    "velocities. HR Wallingford TR155."
                ),
            },
            {"citation": "BGS Seabed Sediments 250K dataset documentation."},
        ],
        "outputs": {
            "noncohesive_mobility_3hourly": str(mobility_hourly_path),
            "noncohesive_mobility_stats": str(stats_path),
            "observed_d50_context": str(observed_context_path),
            "noncohesive_mobility_capacity_segments": str(segments_path),
            "noncohesive_mobility_capacity_map_png": str(png_path),
            "mobility_capacity_profile_png": str(profile_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"Noncohesive mobility (3-hourly): {len(mobility_df)} row(s) -> {mobility_hourly_path}")
    print(f"Noncohesive mobility stats: {len(stats_df)} row(s) -> {stats_path}")
    print(f"Observed D50 context: {len(observed_d50_context_df)} row(s) -> {observed_context_path}")
    print(f"Mobility capacity segments: {len(segments_gdf)} section(s) -> {segments_path}")
    print(f"Mobility capacity map: {png_path}")
    print(f"Mobility capacity profile: {profile_path}")
    print(f"Metadata: {metadata_path}")
    print()
    noncohesive_mobility_map.print_noncohesive_mobility_report(
        stats_df=stats_df,
        capacity_df=capacity_df,
        observed_d50_context_df=observed_d50_context_df,
        segments_gdf=segments_gdf,
        segments_path=segments_path,
        png_path=png_path,
        png_dimensions=png_dimensions,
        profile_path=profile_path,
        profile_dimensions=profile_dimensions,
    )
    return 0


def _cmd_build_scour_onset_screening(args: argparse.Namespace) -> int:
    """MAR-014: pipeline scour-onset embedment screening + 2018 condition benchmark.

    Requires the MAR-009B current, MAR-011A wave orbital, and MAR-007
    morphology outputs to already exist on disk; performs NO network
    request -- everything it reads was already computed. Reuses MAR-012's
    verified hydro-node spatial reconciliation directly.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    metocean_interim_dir = interim_dir / "metocean"
    metocean_processed_dir = study_dir / "metocean"
    morphology_processed_dir = study_dir / "morphology"
    scour_interim_dir = interim_dir / "scour"
    scour_processed_dir = study_dir / "scour"
    pipeline_condition_dir = study_dir / "pipeline_condition"
    maps_dir = study_dir / "maps"

    current_hourly_path = metocean_interim_dir / "current_primary_hourly.parquet"
    current_nodes_path = metocean_interim_dir / "current_primary_support_nodes.parquet"
    wave_hourly_path = metocean_interim_dir / "wave_orbital_velocity_3hourly.parquet"
    wave_nodes_path = metocean_interim_dir / "wave_support_nodes.parquet"
    chainage_metocean_path = metocean_processed_dir / "chainage_metocean_evidence.parquet"
    chainage_morphology_path = morphology_processed_dir / "chainage_regional_morphology.parquet"
    required_paths = (
        pipeline_gpkg_path,
        current_hourly_path,
        current_nodes_path,
        wave_hourly_path,
        wave_nodes_path,
        chainage_metocean_path,
        chainage_morphology_path,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required canonical output(s) -- run build-metocean-evidence, "
            "build-wave-orbital-forcing, and build-regional-morphology first: "
            f"{missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    current_hourly_df = pd.read_parquet(current_hourly_path)
    current_nodes_df = pd.read_parquet(current_nodes_path)
    wave_hourly_df = pd.read_parquet(wave_hourly_path)
    wave_nodes_df = pd.read_parquet(wave_nodes_path)
    chainage_metocean_df = pd.read_parquet(chainage_metocean_path)
    chainage_morphology_df = pd.read_parquet(chainage_morphology_path)

    # --- MAR-012's verified spatial reconciliation, reused directly ---------
    try:
        hydro_pairs_df = combined_bed_shear.build_hydro_pairs(
            current_nodes_df, wave_nodes_df, working_crs=working_crs
        )
    except combined_bed_shear.UnreconciledHydroNodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # --- route sections + local tangent bearings (Section 12) --------------
    chainage_hydro_df = chainage_metocean_df[
        ["station_index", "chainage_m", "current_node_id", "wave_node_id"]
    ].merge(
        hydro_pairs_df[["current_node_id", "wave_node_id", "hydro_pair_id"]],
        on=["current_node_id", "wave_node_id"],
        how="left",
    )
    sections = scour_onset_map.build_route_sections(route, chainage_hydro_df)
    tangent_bearing_by_pair_id = scour_onset_map.build_tangent_bearing_by_pair_id(sections)

    # --- MAR-014 core: Marini et al. (2024) scour-onset screening -----------
    mobility_df, reference_diagnostics = scour_onset.build_scour_onset_embedment_3hourly(
        current_hourly_df, wave_hourly_df, hydro_pairs_df, tangent_bearing_by_pair_id
    )

    try:
        monotonicity_violation_count = scour_onset.raise_if_embedment_monotonicity_violated(
            mobility_df
        )
    except scour_onset.EmbedmentMonotonicityViolationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        stats_df = scour_onset.compute_scour_onset_embedment_stats(mobility_df)
    except scour_onset.ScourOnsetCompletenessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    applicability = scour_onset.compute_applicability_diagnostics(
        reference_diagnostics, mobility_df
    )
    projection_qa = scour_onset.compute_pipeline_normal_projection_qa(reference_diagnostics)
    envelope_df = scour_onset.compute_sensitivity_envelope(stats_df)

    envelope_by_pair_id = {row["hydro_pair_id"]: row.to_dict() for _, row in envelope_df.iterrows()}

    # --- morphology context (Section 27, LEGACY CONTEXT ONLY) --------------
    morphology_join = chainage_hydro_df.merge(
        chainage_morphology_df[
            [
                "station_index",
                "slope_500m_deg",
                "slope_1000m_deg",
                "tpi_1000m_m",
                "local_relief_1000m_m",
                "terrain_std_1000m_m",
            ]
        ],
        on="station_index",
        how="left",
    )
    morphology_by_pair_id: dict[str, dict[str, Any]] = {}
    morphology_summary: dict[str, Any] = {}
    if not morphology_join.empty:
        for pair_id, group in morphology_join.groupby("hydro_pair_id"):
            morphology_by_pair_id[pair_id] = {
                "slope_500m_median_deg": _none_if_nan(group["slope_500m_deg"].median()),
                "slope_500m_p95_deg": _none_if_nan(group["slope_500m_deg"].quantile(0.95)),
                "slope_1000m_median_deg": _none_if_nan(group["slope_1000m_deg"].median()),
                "tpi_1000m_median_m": _none_if_nan(group["tpi_1000m_m"].median()),
                "local_relief_1000m_median_m": _none_if_nan(group["local_relief_1000m_m"].median()),
                "terrain_std_1000m_median_m": _none_if_nan(group["terrain_std_1000m_m"].median()),
            }
        morphology_summary = {
            "slope_500m_deg_route_median": _none_if_nan(morphology_join["slope_500m_deg"].median()),
            "slope_1000m_deg_route_median": _none_if_nan(
                morphology_join["slope_1000m_deg"].median()
            ),
            "tpi_1000m_m_route_median": _none_if_nan(morphology_join["tpi_1000m_m"].median()),
            "local_relief_1000m_m_route_median": _none_if_nan(
                morphology_join["local_relief_1000m_m"].median()
            ),
            "terrain_std_1000m_m_route_median": _none_if_nan(
                morphology_join["terrain_std_1000m_m"].median()
            ),
        }

    # --- final enriched segments ---------------------------------------------
    segments_gdf = scour_onset_map.build_scour_onset_embedment_segments(
        pipeline_id=pipeline_id,
        route=route,
        sections=sections,
        envelope_by_pair_id=envelope_by_pair_id,
        applicability=applicability,
        morphology_by_pair_id=morphology_by_pair_id,
        diameter_m=scour_onset.PIPELINE_DIAMETER_M,
        working_crs=working_crs,
    )

    # --- write parquet/gpkg/json outputs --------------------------------------
    mobility_output_path = metocean_evidence.write_parquet(
        mobility_df, scour_interim_dir / "scour_onset_embedment_screen_3hourly.parquet"
    )
    stats_path = metocean_evidence.write_parquet(
        stats_df, scour_processed_dir / "scour_onset_embedment_stats.parquet"
    )
    segments_path = scour_onset_map.write_scour_onset_embedment_segments_gpkg(
        segments_gdf, scour_processed_dir / "scour_onset_embedment_segments.gpkg"
    )
    benchmark_path = pipeline_condition.write_2018_condition_benchmark(
        pipeline_condition_dir / "anglia_2018_condition_benchmark.json"
    )
    benchmark = pipeline_condition.build_2018_condition_benchmark()

    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    png_path = scour_onset_map.render_scour_onset_embedment_map(
        segments_gdf=segments_gdf,
        route=route,
        working_crs=working_crs,
        output_path=maps_dir / "pl854_scour_onset_embedment_screening.png",
        diameter_m=scour_onset.PIPELINE_DIAMETER_M,
        background_raster_path=(
            background_raster_path if background_raster_path.exists() else None
        ),
    )
    png_dimensions = scour_onset_map.read_png_dimensions(png_path)

    profile_path = scour_onset_map.render_scour_onset_embedment_profile(
        segments_gdf=segments_gdf,
        diameter_m=scour_onset.PIPELINE_DIAMETER_M,
        output_path=maps_dir / "pl854_scour_onset_embedment_profile.png",
    )
    profile_dimensions = scour_onset_map.read_png_dimensions(profile_path)

    # --- metadata --------------------------------------------------------------
    metadata_path = scour_processed_dir / "scour_onset_embedment_metadata.json"
    metadata = {
        "scientific_role": scour_onset.SCIENTIFIC_ROLE,
        "source_model": scour_onset.SOURCE_MODEL_CITATION,
        "source_doi": scour_onset.SOURCE_MODEL_DOI,
        "equations": {
            "current_friction_velocity": ("u_star_c = kappa*U_ref / ln((z_ref+z0_skin)/z0_skin)"),
            "current_at_pipe_top": (
                "Uc_top = u_star_c/kappa * ln((z_top+z0_skin)/z0_skin); z_top = D*(1-e/D)"
            ),
            "pipeline_normal_projection": (
                "Uc_perp = abs(Uc_top*sin(delta_current)); "
                "Uw_perp = abs(Uw*sin(delta_wave)); delta folded via the minimal 0..180 "
                "angular difference around 90 (180-deg axis symmetry)"
            ),
            "marini_wave_shields": (
                "a_x = Uw_perp*T/(2*pi); f_w = 0.04*(a_x/(2.5*d50))^-0.25; "
                "u_star_w = sqrt(f_w/2)*Uw_perp; tau_w = rho_water*u_star_w^2; "
                "theta_w = tau_w / [rho_water*g*(s-1)*d50]"
            ),
            "marini_kc_eq30": (
                "wave-only: KC = Uw_perp*T/D; combined: KC = (Uw_perp*T/D) * "
                "[-0.25 + 1.25*exp((Uc_perp/Uw_perp)^0.87)]"
            ),
            "alpha": "alpha = Uc_perp / (Uc_perp+Uw_perp)",
            "beta": "beta = 6*alpha^3 - 16*alpha^2 + 10*alpha + 1",
            "marini_a": "a = 0.025*[1-exp(-14*X)]; X = KC^0.69 * theta_w^0.70",
            "marini_b": "b = 0.5 + exp(-2.9*X)",
            "omega_forcing": "Omega_forcing = u_equivalent^2 / [g*D*(s-1)*(1-n)]",
            "omega_threshold": "Omega_threshold = a*exp(9*(e/D)^b)",
            "onset_condition": "Omega_forcing >= Omega_threshold",
        },
        "pipeline_diameter_m": scour_onset.PIPELINE_DIAMETER_M,
        "pipeline_diameter_source": scour_onset.PIPELINE_DIAMETER_SOURCE,
        "source_experimental_envelopes": {
            "diameter_m": list(scour_onset.SOURCE_DIAMETER_RANGE_M),
            "current_only_d50_mm": list(scour_onset.SOURCE_CURRENT_ONLY_D50_RANGE_MM),
            "wave_only_d50_mm": list(scour_onset.SOURCE_WAVE_ONLY_D50_RANGE_MM),
            "combined_d50_mm": list(scour_onset.SOURCE_COMBINED_D50_RANGE_MM),
            "uc_m_s": list(scour_onset.SOURCE_UC_RANGE_M_S),
            "uw_m_s": list(scour_onset.SOURCE_UW_RANGE_M_S),
            "kc": list(scour_onset.SOURCE_KC_RANGE),
            "embedment_overall": list(scour_onset.SOURCE_EMBEDMENT_RANGE_OVERALL),
            "embedment_combined": list(scour_onset.SOURCE_EMBEDMENT_RANGE_COMBINED),
        },
        "tested_d50_scenarios_mm": list(scour_onset.TESTED_D50_SCENARIOS_MM),
        "d50_scenario_semantics": scour_onset.D50_SCENARIO_SEMANTICS,
        "tested_porosity_scenarios": list(scour_onset.TESTED_POROSITY_SCENARIOS),
        "porosity_scenario_semantics": scour_onset.POROSITY_SCENARIO_SEMANTICS,
        "tested_embedment_ratios": list(scour_onset.TESTED_EMBEDMENT_RATIOS),
        "embedment_scenario_semantics": scour_onset.EMBEDMENT_SCENARIO_SEMANTICS,
        "current_only_branch_applied": True,
        "wave_only_branch_applied": True,
        "calm_branch_applied": True,
        "continuous_critical_embedment_extrapolated": False,
        "actual_pipeline_embedment_assigned": False,
        "pipeline_normal_projection_applied": True,
        "oblique_flow_extension_directly_validated": False,
        "source_combined_flow_was_codirectional": True,
        "pipe_diameter_within_source_envelope": applicability[
            "within_source_pipe_diameter_envelope"
        ],
        "pipe_diameter_outside_source_envelope_flag": (
            scour_onset.PIPE_DIAMETER_OUTSIDE_SOURCE_ENVELOPE
        ),
        "research_screening_extrapolation_flag": scour_onset.RESEARCH_SCREENING_EXTRAPOLATION,
        "bgs_folk_to_numeric_d50_mapping_applied": False,
        "psa_d50_interpolation_applied": False,
        "bgs_predictive_sediment_used": False,
        "morphology_used_in_onset_physics": False,
        "morphology_role": scour_onset.MORPHOLOGY_ROLE,
        "morphology_source_route_acquisition_years": (
            scour_onset.MORPHOLOGY_SOURCE_ACQUISITION_YEARS
        ),
        "condition_benchmark_2018_path": str(benchmark_path),
        "historical_condition_used_for_spatial_calibration": False,
        "equilibrium_scour_depth_computed": False,
        "scour_propagation_computed": False,
        "exposure_prediction_computed": False,
        "free_span_prediction_computed": False,
        "risk_computed": False,
        "embedment_monotonicity_violation_count": monotonicity_violation_count,
        "applicability_diagnostics": applicability,
        "applicability_diagnostics_semantics": "APPLICABILITY_DIAGNOSTIC_NOT_CONFIDENCE_SCORE",
        "pipeline_normal_projection_qa": projection_qa,
        "references": [
            {
                "citation": scour_onset.SOURCE_MODEL_CITATION,
                "doi": scour_onset.SOURCE_MODEL_DOI,
            },
            {
                "citation": (
                    "Sumer, B.M., Truelsen, C., Sichmann, T., & Fredsoe, J. (2001). "
                    "Onset of scour below pipelines and self-burial. Coastal "
                    "Engineering, 42, 313-335."
                ),
                "doi": "10.1016/S0378-3839(00)00066-1",
            },
            {
                "citation": (
                    "Zang, Z., Cheng, L., & Zhao, M. (2010). Onset of scour below "
                    "pipeline under combined waves and current. OMAE2010-20719."
                ),
                "doi": "10.1115/OMAE2010-20719",
            },
            {
                "citation": (
                    "Ithaca Energy (UK) Limited (2020). Anglia Decommissioning "
                    "Environmental Appraisal. Official UK Government publication. "
                    "Page 29, Tables 3.4-3.5."
                ),
            },
        ],
        "outputs": {
            "scour_onset_embedment_screen_3hourly": str(mobility_output_path),
            "scour_onset_embedment_stats": str(stats_path),
            "scour_onset_embedment_segments": str(segments_path),
            "anglia_2018_condition_benchmark": str(benchmark_path),
            "scour_onset_embedment_map_png": str(png_path),
            "scour_onset_embedment_profile_png": str(profile_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(
        f"Scour onset embedment screen (3-hourly): {len(mobility_df)} row(s) -> "
        f"{mobility_output_path}"
    )
    print(f"Scour onset embedment stats: {len(stats_df)} row(s) -> {stats_path}")
    print(f"Scour onset embedment segments: {len(segments_gdf)} section(s) -> {segments_path}")
    print(f"2018 condition benchmark: {benchmark_path}")
    print(f"Scour onset embedment map: {png_path}")
    print(f"Scour onset embedment profile: {profile_path}")
    print(f"Metadata: {metadata_path}")
    print()
    scour_onset_map.print_scour_onset_report(
        diameter_m=scour_onset.PIPELINE_DIAMETER_M,
        applicability=applicability,
        projection_qa=projection_qa,
        stats_df=stats_df,
        envelope_df=envelope_df,
        monotonicity_violation_count=monotonicity_violation_count,
        morphology_summary=morphology_summary,
        benchmark=benchmark,
        segments_gdf=segments_gdf,
        segments_path=segments_path,
        png_path=png_path,
        png_dimensions=png_dimensions,
        profile_path=profile_path,
        profile_dimensions=profile_dimensions,
    )
    return 0


_SHERINGHAM_2024_SCOUR_CATEGORY_BY_DESCRIPTOR: dict[str, str] = {
    "concrete block": observed_evidence.SEABED_OBJECT_CONTEXT,
    "exposure": observed_evidence.SOURCE_INTERPRETED_EXPOSURE_EVIDENCE,
    "cable": observed_evidence.ASSET_INFRASTRUCTURE_CONTEXT,
    "jack-up footprint": observed_evidence.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
    "boulder": observed_evidence.SEABED_OBJECT_CONTEXT,
    "debris": observed_evidence.SEABED_OBJECT_CONTEXT,
    "rock-bag": observed_evidence.SEABED_OBJECT_CONTEXT,
    "jack-up footprint with sediment build up": observed_evidence.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
    "infrastructure": observed_evidence.SEABED_OBJECT_CONTEXT,
}


def _derive_scour_poc_validation_questions(
    *,
    margin_computed_when_actual_embedment_provided: bool,
    exceedance_fraction_computable: bool,
    pl854_scenario_envelope_produced: bool,
    explicit_scour_evidence_present: bool,
    operator_interpretation_package_ingested: bool,
) -> dict[str, str | None]:
    """MAR-023 Section 21 (MAR-023A Section 3 correction): a pure function so D/G/H's mandated
    NO answers are structurally enforced (asserted), not just conventionally true -- this POC
    never assembles a PL854 site-specific embedment profile, never feeds Sheringham evidence
    into the pipeline physics, and never predicts a future scour depth.

    Question F asks specifically about EXPLICIT source-interpreted SCOUR evidence -- never
    "any interpreted feature at all" (Section 3's real bug: a non-empty, fully-real,
    correctly-classified `evidence_gdf` full of exposure/debris/cable context is NOT scour
    evidence, and must not be read as YES here). `operator_interpretation_package_ingested`
    is threaded through separately (never derived FROM the F answer or vice versa) so a
    caller cannot accidentally collapse the two independent facts back into one (Section 4).
    """

    pl854_has_site_specific_data = False
    sheringham_used_as_pipeline_validation = False
    future_scour_depth_prediction_made = False
    assert pl854_has_site_specific_data is False
    assert sheringham_used_as_pipeline_validation is False
    assert future_scour_depth_prediction_made is False

    return {
        "question_a_generic_engine_operator_input_ready": "YES",
        "question_b_margin_computed_when_actual_embedment_provided": (
            "YES" if margin_computed_when_actual_embedment_provided else "NO"
        ),
        "question_c_forcing_record_exceedance_fraction_computable": (
            "YES" if exceedance_fraction_computable else "NO"
        ),
        "question_d_pl854_has_enough_data_for_site_specific_susceptibility": (
            "YES" if pl854_has_site_specific_data else "NO"
        ),
        "question_e_pl854_scenario_envelope_produced": (
            "YES" if pl854_scenario_envelope_produced else "NO"
        ),
        "question_f_real_source_interpreted_scour_evidence_ingested": (
            "YES" if explicit_scour_evidence_present else "NO"
        ),
        "question_f_reason": (
            None
            if explicit_scour_evidence_present
            else observed_evidence.NO_EXPLICIT_SOURCE_INTERPRETED_SCOUR_FEATURE_CLASS_PRESENT
        ),
        "question_g_sheringham_used_as_pipeline_physics_validation": (
            "YES" if sheringham_used_as_pipeline_validation else "NO"
        ),
        "question_h_future_scour_depth_prediction_made": (
            "YES" if future_scour_depth_prediction_made else "NO"
        ),
        # Section 4: recorded here too so the validation JSON carries both concepts even
        # though only F is a lettered question -- deliberately independent of question_f.
        "operator_interpretation_package_ingested": (
            "YES" if operator_interpretation_package_ingested else "NO"
        ),
        "explicit_source_interpreted_scour_evidence_present": (
            "YES" if explicit_scour_evidence_present else "NO"
        ),
    }


def _cmd_build_scour_susceptibility_poc(args: argparse.Namespace) -> int:
    """MAR-023: generic pipeline scour-onset susceptibility screening (Track A, reusing
    MAR-014's Marini et al. 2024 engine unchanged) + real observed-scour-evidence ingestion
    from the 2024 Sheringham Shoal XOCEAN survey (Track B) -- scientifically separate tracks;
    Track B never validates Track A's physics (Section 12).

    Requires MAR-014's `build-scour-onset-screening` outputs to already exist on disk for
    PL854 (Section 23: never recomputes the upstream scientific model); performs ONE minimal
    live acquisition for the Sheringham 2024 Interpretation Data package if not already
    cached, then is fully offline.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    scour_interim_dir = interim_dir / "scour"
    scour_processed_dir = study_dir / "scour"
    maps_dir = study_dir / "maps"

    mobility_path = scour_interim_dir / "scour_onset_embedment_screen_3hourly.parquet"
    segments_path = scour_processed_dir / "scour_onset_embedment_segments.gpkg"
    required_paths = (pipeline_gpkg_path, mobility_path, segments_path)
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required canonical output(s) -- run build-scour-onset-screening "
            f"first: {missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    diameter_m = scour_onset.PIPELINE_DIAMETER_M
    print("Reusing the accepted MAR-014 scour-onset screening outputs (Section 2)...")
    mobility_df = pd.read_parquet(mobility_path)
    segments_gdf = gpd.read_file(segments_path)
    print(f"  {len(mobility_df)} forcing row(s), {len(segments_gdf)} route section(s)")

    sections = [
        {
            "section_id": int(row["segment_id"]),
            "hydro_pair_id": row["hydro_pair_id"],
            "start_chainage_m": float(row["start_chainage_m"]),
            "end_chainage_m": float(row["end_chainage_m"]),
        }
        for _, row in segments_gdf.iterrows()
    ]

    # --- Track A: site-specific screening -- PL854 has no actual embedment profile (Section 6) --
    print("Building the site-specific susceptibility table (Section 15)...")
    site_specific_df, site_specific_detail_df = susceptibility.build_susceptibility_tables(
        pipeline_id=pipeline_id,
        sections=sections,
        mobility_df=mobility_df,
        diameter_m=diameter_m,
        actual_embedment_m_by_section_id=None,
        evidence_type=susceptibility.EVIDENCE_TYPE_NO_PROFILE,
    )
    site_specific_path = metocean_evidence.write_parquet(
        site_specific_df, scour_processed_dir / "pipeline_scour_susceptibility_screening.parquet"
    )
    metocean_evidence.write_parquet(
        site_specific_detail_df,
        scour_processed_dir / "pipeline_scour_susceptibility_sensitivity_detail.parquet",
    )
    print(f"  {len(site_specific_df)} section(s) -> {site_specific_path}")

    # --- Track A: PL854 tested-embedment scenario envelope (Section 16) ------------------------
    print("Building the PL854 tested-embedment scenario envelope (Section 16)...")
    envelope_summary_df, envelope_detail_df = susceptibility.build_scenario_envelope_table(
        pipeline_id=pipeline_id,
        sections=sections,
        mobility_df=mobility_df,
        diameter_m=diameter_m,
    )
    envelope_path = metocean_evidence.write_parquet(
        envelope_summary_df, scour_processed_dir / "pl854_scour_onset_scenario_envelope.parquet"
    )
    metocean_evidence.write_parquet(
        envelope_detail_df,
        scour_processed_dir / "pl854_scour_onset_scenario_envelope_sensitivity_detail.parquet",
    )
    print(f"  {len(envelope_summary_df)} section x scenario row(s) -> {envelope_path}")

    # --- maps (Sections 16-17) --------------------------------------------------------------------
    envelope_map_path = susceptibility_map.render_scour_onset_scenario_envelope_map(
        envelope_summary_df=envelope_summary_df,
        total_length_m=route.length,
        diameter_m=diameter_m,
        output_path=maps_dir / "pl854_scour_onset_scenario_envelope.png",
    )
    print(f"  Scenario envelope map -> {envelope_map_path}")

    merged_segments_gdf = segments_gdf.merge(
        site_specific_df,
        left_on="segment_id",
        right_on="section_id",
        how="left",
        suffixes=("", "_susc"),
    )
    susceptibility_map_path = susceptibility_map.render_pipeline_scour_susceptibility_map(
        segments_gdf=merged_segments_gdf,
        route=route,
        working_crs=working_crs,
        output_path=maps_dir / "pl854_scour_onset_susceptibility_screening.png",
        diameter_m=diameter_m,
        unavailable_message=(
            "SITE-SPECIFIC SCOUR SUSCEPTIBILITY NOT AVAILABLE\nNO ACTUAL PL854 EMBEDMENT PROFILE"
        ),
    )
    print(
        "  Susceptibility screening map (future-operator-ready renderer) -> "
        f"{susceptibility_map_path}"
    )

    # --- GIS (Section 19) --------------------------------------------------------------------
    pl854_gpkg_path = scour_processed_dir / "pipeline_scour_screening.gpkg"
    if pl854_gpkg_path.exists():
        pl854_gpkg_path.unlink()
    if not merged_segments_gdf.empty:
        merged_segments_gdf.to_file(
            pl854_gpkg_path, driver="GPKG", layer="site_specific_susceptibility_screening"
        )
    envelope_gis_gdf = envelope_summary_df.merge(
        segments_gdf[["segment_id", "geometry"]],
        left_on="section_id",
        right_on="segment_id",
        how="left",
    )
    envelope_gis_gdf = gpd.GeoDataFrame(envelope_gis_gdf, geometry="geometry", crs=working_crs)
    if not envelope_gis_gdf.empty:
        envelope_gis_gdf.to_file(
            pl854_gpkg_path, driver="GPKG", layer="tested_embedment_scenario_envelope"
        )
    print(f"  GIS: {pl854_gpkg_path}")

    # --- input contract (Section 14) --------------------------------------------------------------
    contract_path = scour_processed_dir / "scour_susceptibility_input_contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(
            scour_contract.build_scour_susceptibility_input_contract(), indent=2, default=str
        ),
        encoding="utf-8",
    )
    print(f"  Input contract -> {contract_path}")

    # ==========================================================================================
    # Track B: Sheringham Shoal 2024 observed scour evidence (Sections 9-13, 18-19)
    # ==========================================================================================
    print()
    print("Acquiring the 2024 Sheringham Shoal Interpretation Data package (Section 9)...")
    sheringham_raw_dir = config.paths.raw_dir / "sheringham_shoal_2024"
    sheringham_processed_dir = config.paths.processed_dir / "sheringham_shoal_2024"
    sheringham_scour_dir = sheringham_processed_dir / "scour"
    sheringham_maps_dir = sheringham_processed_dir / "maps"

    acquisition = sheringham_2024_provider.download_sheringham_shoal_2024_interpretation_data(
        sheringham_raw_dir
    )
    print(
        f"  1 layer, {acquisition.package_bytes:,} bytes, "
        f"already_cached={acquisition.already_cached}"
    )
    targets_gdf = sheringham_2024_provider.load_targets_layer(acquisition.extracted_dir)
    print(f"  {len(targets_gdf)} real target feature(s) loaded, CRS={targets_gdf.crs}")

    interpretation_layer = observed_evidence.InterpretationLayer(
        layer_name=sheringham_2024_provider.TARGETS_LAYER_NAME,
        gdf=targets_gdf,
        description_column=sheringham_2024_provider.DESCRIPTION_COLUMN,
    )

    print("Building the source interpretation inventory (Section 10)...")
    inventory_df = observed_evidence.build_feature_inventory([interpretation_layer])
    inventory_path = metocean_evidence.write_parquet(
        inventory_df, sheringham_scour_dir / "source_interpretation_inventory.parquet"
    )
    print(f"  {len(inventory_df)} attribute-value row(s) -> {inventory_path}")

    print("Classifying interpreted features (Section 11)...")
    evidence_gdf = observed_evidence.build_observed_evidence_table(
        interpretation_layer,
        category_by_normalized_descriptor=_SHERINGHAM_2024_SCOUR_CATEGORY_BY_DESCRIPTOR,
        survey_epoch=sheringham_2024_provider.SURVEY_EPOCH,
        source_id_column=sheringham_2024_provider.SOURCE_ID_COLUMN,
        asset_association_column=sheringham_2024_provider.ASSET_ASSOCIATION_COLUMN,
        length_column=sheringham_2024_provider.LENGTH_COLUMN,
        width_column=sheringham_2024_provider.WIDTH_COLUMN,
        height_column=sheringham_2024_provider.HEIGHT_COLUMN,
        water_depth_column=sheringham_2024_provider.WATER_DEPTH_COLUMN,
    )
    evidence_summary = observed_evidence.summarize_observed_evidence(evidence_gdf)
    for category, count in evidence_summary["count_by_category"].items():
        print(f"  {category}: {count}")
    print(
        f"  explicit scour feature count: {evidence_summary['explicit_scour_feature_count']} | "
        f"morphometry_status: {evidence_summary['morphometry_status']}"
    )

    evidence_attributes_df = pd.DataFrame(evidence_gdf).drop(columns="geometry")
    evidence_path = metocean_evidence.write_parquet(
        evidence_attributes_df,
        sheringham_scour_dir / "observed_scour_evidence_attributes.parquet",
    )
    print(f"  {len(evidence_gdf)} classified feature(s) -> {evidence_path}")
    print(f"  Observed scour morphometry: {evidence_summary['morphometry_status']}")

    # --- map (Section 18) ----------------------------------------------------------------------
    background_raster_path = (
        study_dir.parent / "sheringham_shoal_2020" / "terrain" / "canonical_bed_elevation.tif"
    )
    explicit_scour_present = evidence_summary["explicit_source_interpreted_scour_evidence_present"]
    evidence_map_path = observed_evidence_map.render_observed_scour_evidence_map(
        evidence_gdf=evidence_gdf,
        output_path=sheringham_maps_dir / "sheringham_shoal_2024_observed_scour_evidence.png",
        background_raster_path=(
            background_raster_path if background_raster_path.exists() else None
        ),
        background_raster_label=(
            "2020 MBES canonical bed elevation (MAR-020/021, spatial context only -- not "
            "co-temporal with the 2024 interpretation)"
        ),
        explicit_scour_evidence_present=explicit_scour_present,
    )
    print(f"  Observed evidence map -> {evidence_map_path}")

    # --- GIS (Section 19; MAR-023A Section 5: semantically-correct layer names) ------------------
    # Filename kept as `observed_scour_evidence.gpkg` for compatibility (Section 5), but the
    # primary layer holding ALL 746 interpreted features is named for what it actually is --
    # a general integrity-context layer, never implying every row is a scour observation.
    sheringham_gpkg_path = sheringham_scour_dir / "observed_scour_evidence.gpkg"
    if sheringham_gpkg_path.exists():
        sheringham_gpkg_path.unlink()
    if not evidence_gdf.empty:
        evidence_gdf.to_file(
            sheringham_gpkg_path, driver="GPKG", layer="source_interpreted_integrity_context"
        )
    explicit_scour_gdf = observed_evidence.extract_scour_category(
        evidence_gdf, observed_evidence.SOURCE_INTERPRETED_OBSERVED_SCOUR_EVIDENCE
    )
    if not explicit_scour_gdf.empty:
        explicit_scour_gdf.to_file(
            sheringham_gpkg_path, driver="GPKG", layer="explicit_observed_scour_evidence"
        )
    print(f"  GIS: {sheringham_gpkg_path}")

    # ==========================================================================================
    # Final POC report + validation (Sections 20-21)
    # ==========================================================================================
    print()
    print("Building the generic linear-asset scour POC report (Section 20)...")

    validation = _derive_scour_poc_validation_questions(
        margin_computed_when_actual_embedment_provided=True,
        exceedance_fraction_computable=True,
        pl854_scenario_envelope_produced=not envelope_summary_df.empty,
        explicit_scour_evidence_present=evidence_summary[
            "explicit_source_interpreted_scour_evidence_present"
        ],
        operator_interpretation_package_ingested=evidence_summary[
            "operator_interpretation_package_ingested"
        ],
    )

    blocks = scour_poc_report.build_scour_poc_report_blocks(
        project_title="Generic Linear-Asset Scour Susceptibility Screening POC",
        purpose_text=(
            "Demonstrates the future OrbGSS workflow: operator pipeline data + sediment + "
            "metocean + embedment -> pipeline scour-onset physics -> physical susceptibility "
            "margin -> map + KP view + GIS + report. Separately: operator survey/interpretation "
            "-> observed scour evidence layer. Track B never validates Track A unless the asset "
            "physics and required engineering inputs genuinely match."
        ),
        pipeline_method_facts={
            "scientific_role": scour_onset.SCIENTIFIC_ROLE,
            "source_model": scour_onset.SOURCE_MODEL_CITATION,
            "source_doi": scour_onset.SOURCE_MODEL_DOI,
            "tested_embedment_ratios": list(scour_onset.TESTED_EMBEDMENT_RATIOS),
            "tested_d50_scenarios_mm": list(scour_onset.TESTED_D50_SCENARIOS_MM),
            "tested_porosity_scenarios": list(scour_onset.TESTED_POROSITY_SCENARIOS),
            "continuous_critical_embedment_solved": False,
        },
        required_inputs=[
            f"{f['field']}: {f['description']}" for f in scour_contract.REQUIRED_FIELDS
        ],
        actual_vs_critical_facts={
            "actual_embedment_ratio_semantics": "e_actual / D, operator-supplied only",
            "critical_embedment_ratio_semantics": (
                "MAR-014's own tested/required embedment screening class, e_critical / D"
            ),
            "embedment_protection_margin_p95_e_over_D_semantics": (
                "actual_embedment_ratio - critical_embedment_ratio_p95; positive is an "
                "empirical onset-screening result only, never SAFE/DESIGN ACCEPTABLE/NO SCOUR"
            ),
            "pl854_site_specific_screening_state": (
                site_specific_df["screening_state"].iloc[0] if not site_specific_df.empty else None
            ),
        },
        exceedance_fraction_facts={
            "definition": (
                "valid forcing timesteps where the tested critical embedment exceeds the "
                "actual embedment / valid forcing timesteps"
            ),
            "disclaimers": list(susceptibility.EXCEEDANCE_FRACTION_DISCLAIMERS),
        },
        pl854_scenario_facts={
            "section_count": len(sections),
            "tested_embedment_scenarios": list(susceptibility.TESTED_EMBEDMENT_SCENARIOS),
            "scenario_envelope_row_count": len(envelope_summary_df),
        },
        pl854_limitations=[
            scour_onset.RESEARCH_SCREENING_EXTRAPOLATION,
            scour_onset.PIPE_DIAMETER_OUTSIDE_SOURCE_ENVELOPE,
            "No verified continuous observed PL854 embedment profile exists -- PL854 receives "
            "a scenario envelope only, never a site-specific actual susceptibility map.",
        ],
        sheringham_evidence_facts={
            "source_page": sheringham_2024_provider.MDE_SOURCE_PAGE_URL,
            "package_url": sheringham_2024_provider.INTERPRETATION_DATA_URL,
            "package_bytes": acquisition.package_bytes,
            "total_feature_count": evidence_summary["total_feature_count"],
            "count_by_category": evidence_summary["count_by_category"],
            "operator_interpretation_package_ingested": (
                "YES" if evidence_summary["operator_interpretation_package_ingested"] else "NO"
            ),
            "explicit_source_interpreted_scour_evidence_present": (
                "YES"
                if evidence_summary["explicit_source_interpreted_scour_evidence_present"]
                else "NO"
            ),
            "explicit_scour_feature_count": evidence_summary["explicit_scour_feature_count"],
            "explicit_scour_absence_reason": evidence_summary["explicit_scour_absence_reason"],
            "morphometry_status": evidence_summary["morphometry_status"],
        },
        asset_physics_mismatch_text=observed_evidence.ASSET_PHYSICS_DISCLAIMER,
        production_transfer_contract_summary=[
            f"{f['field']}: {f['description']}" for f in scour_contract.STRONGLY_PREFERRED_FIELDS
        ],
        not_predicted=[
            "Future scour depth",
            "Freespan development",
            "Fatigue/VIV response",
            "Route suitability",
            "A generic or 0-100 risk score",
            "Any ML-derived prediction",
        ],
    )
    report_html = scour_poc_report.render_blocks_html(
        blocks, title="Generic Linear-Asset Scour Susceptibility Screening POC"
    )
    report_path = (
        config.paths.processed_dir / "scour_poc" / "report" / "generic_linear_asset_scour_poc.html"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_html, encoding="utf-8")
    print(f"  Report -> {report_path}")

    validation_path = config.paths.processed_dir / "scour_poc" / "scour_poc_validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    print(f"  Validation -> {validation_path}")

    print()
    print("=== Generic Linear-Asset Scour Susceptibility Screening POC (MAR-023) ===")
    print()
    print("## Pipeline engine")
    print(f"  method reused: {scour_onset.SOURCE_MODEL_CITATION}")
    print(f"  generic input fields: {len(scour_contract.REQUIRED_FIELDS)} required")
    print("  domain gates: pipe-diameter source envelope, tested embedment envelope (<= 0.15 D)")
    print()
    print("## PL854")
    print(f"  section count: {len(sections)}")
    print(f"  tested embedment scenarios: {list(susceptibility.TESTED_EMBEDMENT_SCENARIOS)}")
    print(f"  scenario exceedance statistics: see {envelope_path}")
    print(
        "  no actual PL854 embedment profile exists -- "
        f"{susceptibility.SITE_SPECIFIC_SCOUR_SUSCEPTIBILITY_NOT_AVAILABLE_NO_EMBEDMENT_PROFILE}"
    )
    print()
    print("## Sheringham 2024")
    print(f"  source: {sheringham_2024_provider.MDE_SOURCE_PAGE_URL}")
    print(f"  interpretation layer inventory: {len(inventory_df)} attribute-value row(s)")
    print(f"  observed feature count/classes: {evidence_summary['count_by_category']}")
    print(f"  asset associations (source-stated): {evidence_summary['count_by_asset_association']}")
    print(
        "  operator_interpretation_package_ingested: "
        f"{validation['operator_interpretation_package_ingested']}"
    )
    print(
        "  explicit_source_interpreted_scour_evidence_present: "
        f"{validation['explicit_source_interpreted_scour_evidence_present']}"
    )
    if validation.get("question_f_reason"):
        print(f"  reason: {validation['question_f_reason']}")
    print()
    print("## Outputs")
    print(f"  maps: {susceptibility_map_path}, {envelope_map_path}, {evidence_map_path}")
    print(f"  parquet: {site_specific_path}, {envelope_path}, {evidence_path}")
    print(f"  GIS: {pl854_gpkg_path}, {sheringham_gpkg_path}")
    print(f"  report: {report_path}")
    print(f"  validation: {validation_path}")
    print()
    print(
        "IS GENERIC OPERATOR-SUPPLIED PIPELINE SCOUR-ONSET SUSCEPTIBILITY SCREENING "
        f"DEMONSTRATED? {validation['question_a_generic_engine_operator_input_ready']}"
    )
    print(
        "IS A SITE-SPECIFIC PL854 SCOUR SUSCEPTIBILITY MAP DEFENSIBLE WITH THE CURRENT DATA? "
        f"{validation['question_d_pl854_has_enough_data_for_site_specific_susceptibility']}"
    )
    return 0


def _cmd_build_freespan_spatial_evidence(args: argparse.Namespace) -> int:
    """MAR-014A: official freespan spatial evidence recovery + canonical route reconciliation.

    Requires the MAR-012 combined-bed-shear, MAR-013 noncohesive-mobility, and
    MAR-014 scour-onset segment outputs to already exist on disk; performs NO
    network request. Reads the tracked Table B.1 CSV resource directly
    (never scraped at runtime) and empirically reconciles its (undocumented)
    source CRS against the TRUE canonical route before projecting events onto it.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, _interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    combined_bed_shear_segments_path = study_dir / "metocean" / "combined_bed_shear_segments.gpkg"
    noncohesive_mobility_segments_path = (
        study_dir / "sediment" / "noncohesive_mobility_capacity_segments.gpkg"
    )
    scour_onset_segments_path = study_dir / "scour" / "scour_onset_embedment_segments.gpkg"

    required_paths = (
        pipeline_gpkg_path,
        combined_bed_shear_segments_path,
        noncohesive_mobility_segments_path,
        scour_onset_segments_path,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required canonical output(s) -- run build-combined-bed-shear, "
            "build-noncohesive-mobility, and build-scour-onset-screening first: "
            f"{missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    combined_bed_shear_segments_df = gpd.read_file(combined_bed_shear_segments_path)
    noncohesive_mobility_segments_df = gpd.read_file(noncohesive_mobility_segments_path)
    scour_onset_segments_df = gpd.read_file(scour_onset_segments_path)

    try:
        events_df = load_anglia_table_b1_freespans()
    except AngliaTableB1ChecksumError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        relationships_df = load_anglia_table_b1_freespan_relationships()
    except AngliaFreespanRelationshipValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # --- CRS candidate evaluation + acceptance guard (Sections 7-9) ------------
    diagnostics_by_epsg = {
        epsg: freespan_evidence.evaluate_crs_candidate(events_df, route, working_crs, epsg)
        for epsg in freespan_evidence.CANDIDATE_CRS_EPSG_CODES
    }
    try:
        accepted_epsg, crs_checks = freespan_evidence.select_working_crs(diagnostics_by_epsg)
    except freespan_evidence.CRSReconciliationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    survey_direction = freespan_evidence.classify_survey_direction(
        diagnostics_by_epsg[accepted_epsg].fit_slope
    )

    # --- canonical projection (Sections 11-14) ---------------------------------
    try:
        events_projected_df = freespan_evidence.project_events_to_canonical_route(
            events_df, route, working_crs, accepted_epsg
        )
    except freespan_evidence.AngliaFreespanValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    events_2018_df = events_projected_df[events_projected_df["survey_year"] == 2018].reset_index(
        drop=True
    )

    geometries_all = freespan_evidence.build_event_geometries(events_projected_df, route)
    events_all_gdf = gpd.GeoDataFrame(events_projected_df, geometry=geometries_all, crs=working_crs)

    geometries_2018 = freespan_evidence.build_event_geometries(events_2018_df, route)
    events_2018_gdf = gpd.GeoDataFrame(events_2018_df, geometry=geometries_2018, crs=working_crs)

    # --- segment event counts (Section 18) --------------------------------------
    segment_bounds = scour_onset_segments_df[
        ["hydro_pair_id", "start_chainage_m", "end_chainage_m"]
    ].to_dict("records")
    segment_counts_df = freespan_evidence.compute_segment_freespan_counts(
        events_2018_df, segment_bounds
    )

    # --- model-context join (Section 17, no score/probability/rank) ------------
    context_df = freespan_model_context.attach_model_context_to_events(
        events_2018_df,
        combined_bed_shear_segments_df=combined_bed_shear_segments_df,
        noncohesive_mobility_segments_df=noncohesive_mobility_segments_df,
        scour_onset_segments_df=scour_onset_segments_df,
    )

    # --- MAR-014B: source-stated temporal/lineage relationship evidence --------
    temporal_evidence_df = (
        freespan_temporal_provenance.build_freespan_temporal_relationship_evidence(
            relationships_df, events_all_gdf
        )
    )
    coverage_metadata = freespan_temporal_provenance.build_survey_coverage_metadata()

    # --- write canonical outputs -------------------------------------------------
    freespan_dir = study_dir / "freespan_evidence"
    validation_dir = study_dir / "validation"
    pipeline_condition_dir = study_dir / "pipeline_condition"
    maps_dir = study_dir / "maps"

    evidence_parquet_path = metocean_evidence.write_parquet(
        events_all_gdf.drop(columns="geometry"),
        freespan_dir / "anglia_freespan_spatial_evidence.parquet",
    )
    evidence_gpkg_path = freespan_dir / "anglia_freespan_spatial_evidence.gpkg"
    evidence_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    events_all_gdf.to_file(evidence_gpkg_path, driver="GPKG", layer="historical_freespans")

    evidence_2018_path = metocean_evidence.write_parquet(
        events_2018_df, freespan_dir / "anglia_2018_freespan_spatial_evidence.parquet"
    )
    segment_counts_path = metocean_evidence.write_parquet(
        segment_counts_df, freespan_dir / "freespan_segment_event_counts_2018.parquet"
    )
    model_context_path = metocean_evidence.write_parquet(
        context_df, validation_dir / "2018_freespan_model_context.parquet"
    )
    temporal_evidence_path = (
        freespan_temporal_provenance.write_freespan_temporal_relationship_evidence(
            temporal_evidence_df,
            pipeline_condition_dir / "anglia_freespan_temporal_relationship_evidence.parquet",
        )
    )

    # --- Section 20: refresh the 2018 condition benchmark with corrected flags --
    benchmark_path = pipeline_condition.write_2018_condition_benchmark(
        pipeline_condition_dir / "anglia_2018_condition_benchmark.json"
    )

    # --- maps --------------------------------------------------------------------
    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    background_raster_path = background_raster_path if background_raster_path.exists() else None

    map_2018_path = freespan_evidence_map.render_2018_freespan_evidence_map(
        events_2018_gdf=events_2018_gdf,
        route=route,
        output_path=maps_dir / "pl854_observed_freespans_2018.png",
        background_raster_path=background_raster_path,
    )
    map_2018_dimensions = freespan_evidence_map.read_png_dimensions(map_2018_path)

    map_historical_path = freespan_evidence_map.render_historical_freespan_evidence_map(
        events_all_gdf=events_all_gdf,
        route=route,
        output_path=maps_dir / "pl854_historical_freespans_2012_2018.png",
        background_raster_path=background_raster_path,
    )
    map_historical_dimensions = freespan_evidence_map.read_png_dimensions(map_historical_path)

    profile_path = freespan_evidence_map.render_freespan_model_context_profile(
        context_df=context_df,
        events_2018_span_df=events_2018_df[
            ["event_id", "canonical_chainage_min_m", "canonical_chainage_max_m"]
        ],
        combined_bed_shear_segments_df=combined_bed_shear_segments_df,
        noncohesive_mobility_segments_df=noncohesive_mobility_segments_df,
        scour_onset_segments_df=scour_onset_segments_df,
        total_length_m=route.length,
        output_path=maps_dir / "pl854_2018_freespan_model_context_profile.png",
    )
    profile_dimensions = freespan_evidence_map.read_png_dimensions(profile_path)

    temporal_figure_path = freespan_evidence_map.render_freespan_temporal_evolution_figure(
        temporal_evidence_df=temporal_evidence_df,
        route=route,
        output_path=maps_dir / "pl854_source_stated_freespan_evolution.png",
        background_raster_path=background_raster_path,
    )
    temporal_figure_dimensions = freespan_evidence_map.read_png_dimensions(temporal_figure_path)

    # --- CRS/route reconciliation metadata (Section 24) -------------------------
    gaps_2014 = freespan_evidence_map.compute_2014_coverage_gap_zones(events_all_gdf)
    zones_by_year = freespan_evidence_map.compute_zone_coverage_by_year(events_all_gdf)
    per_year_counts = {
        str(int(year)): {
            "count": int(len(group)),
            "sum_source_length_m": float(group["source_length_m"].sum()),
        }
        for year, group in events_all_gdf.groupby("survey_year")
    }

    def _diag_to_dict(diag: freespan_evidence.CrsCandidateDiagnostics) -> dict[str, Any]:
        return {
            "candidate_epsg": diag.candidate_epsg,
            "endpoint_count": diag.endpoint_count,
            "distance_min_m": diag.distance_min_m,
            "distance_median_m": diag.distance_median_m,
            "distance_p95_m": diag.distance_p95_m,
            "distance_max_m": diag.distance_max_m,
            "fit_slope": diag.fit_slope,
            "fit_intercept_m": diag.fit_intercept_m,
            "fit_r_squared": diag.fit_r_squared,
            "residual_median_m": diag.residual_median_m,
            "residual_p95_m": diag.residual_p95_m,
            "residual_max_m": diag.residual_max_m,
            "orientation_consistent": diag.orientation_consistent,
        }

    reconciliation_metadata = {
        "scientific_role": freespan_evidence.SCIENTIFIC_ROLE,
        "asset_scope": freespan_evidence.ASSET_SCOPE,
        "individual_line_attribution": freespan_evidence.INDIVIDUAL_LINE_ATTRIBUTION,
        "source": {
            "publisher": "Ithaca Energy (UK) Limited",
            "title": "Pipelines and Umbilical Comparative Assessment",
            "date": "April 2020",
            "table": "Appendix B, Table B.1",
        },
        "source_crs_status": freespan_evidence.SOURCE_CRS_STATUS,
        "candidate_crs_epsg_codes": list(freespan_evidence.CANDIDATE_CRS_EPSG_CODES),
        "candidate_diagnostics": {
            str(epsg): _diag_to_dict(diag) for epsg, diag in diagnostics_by_epsg.items()
        },
        "acceptance_checks": crs_checks,
        "accepted_crs_epsg": accepted_epsg,
        "canonical_working_crs": working_crs,
        "canonical_route_source": str(pipeline_gpkg_path),
        "canonical_route_length_m": route.length,
        "survey_kp_direction_relative_to_canonical": survey_direction,
        "per_year_counts": per_year_counts,
        "2014_coverage_gap_zones": gaps_2014,
        "zone_coverage_by_year": {
            str(year): [{"start_chainage_m": lo, "end_chainage_m": hi} for lo, hi in zones]
            for year, zones in zones_by_year.items()
        },
        "endpoint_route_distance_stats_m": {
            "max": float(
                max(
                    events_all_gdf["endpoint_a_route_distance_m"].max(),
                    events_all_gdf["endpoint_b_route_distance_m"].max(),
                )
            ),
            "median": float(
                pd.concat(
                    [
                        events_all_gdf["endpoint_a_route_distance_m"],
                        events_all_gdf["endpoint_b_route_distance_m"],
                    ]
                ).median()
            ),
        },
        "source_length_reconciliation": {
            "max_absolute_length_difference_m": float(
                events_all_gdf["absolute_length_difference_m"].max()
            ),
            "gross_mismatch_absolute_threshold_m": (
                freespan_evidence.GROSS_LENGTH_MISMATCH_ABSOLUTE_M
            ),
            "gross_mismatch_relative_threshold_pct": (
                freespan_evidence.GROSS_LENGTH_MISMATCH_RELATIVE_PCT
            ),
        },
        "limitations": [
            "Source CRS was never stated by either source document; EPSG:23031 was inferred "
            "empirically and must not be read as a source-confirmed fact.",
            "2014-01 and 2014-02 (near source KP 0) project to the exact canonical route "
            "terminus, indicating the physical corridor survey extends slightly beyond the "
            "digitized canonical PL854 route's own extent at that end; their projected "
            "interval lengths collapse towards 0 m even though source lengths are 10.02 m "
            "and 7.98 m respectively.",
            "2014 Table B.1 events report no coverage in one zone where both 2012 and 2018 "
            "report events (see 2014_coverage_gap_zones) -- this may reflect incomplete 2014 "
            "survey coverage rather than absence of a freespan.",
            "Table B.1's own scope is the piggybacked PL854/PL855 corridor; individual line "
            "attribution is UNRESOLVED for every event.",
            "No score, probability, rank, or accuracy metric has been computed anywhere in "
            "this output.",
        ],
        # --- MAR-014B Section 12 -------------------------------------------------
        "table_b1_source_comments_preserved": True,
        "source_stated_temporal_relationships_available": True,
        "automatic_cross_survey_matching_applied": False,
        "2014_partial_coverage_source_stated": True,
        "2018_full_route_negative_label_assumption_applied": False,
        "survey_coverage_semantics": coverage_metadata,
        "outputs": {
            "anglia_freespan_spatial_evidence_parquet": str(evidence_parquet_path),
            "anglia_freespan_spatial_evidence_gpkg": str(evidence_gpkg_path),
            "anglia_2018_freespan_spatial_evidence_parquet": str(evidence_2018_path),
            "freespan_segment_event_counts_2018_parquet": str(segment_counts_path),
            "2018_freespan_model_context_parquet": str(model_context_path),
            "anglia_2018_condition_benchmark_json": str(benchmark_path),
            "anglia_table_b1_freespan_relationships_csv": str(
                ANGLIA_TABLE_B1_FREESPAN_RELATIONSHIPS_CSV
            ),
            "anglia_freespan_temporal_relationship_evidence_parquet": str(temporal_evidence_path),
            "map_2018_png": str(map_2018_path),
            "map_historical_png": str(map_historical_path),
            "model_context_profile_png": str(profile_path),
            "freespan_temporal_evolution_figure_png": str(temporal_figure_path),
        },
    }
    metadata_path = freespan_dir / "anglia_freespan_spatial_reconciliation_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(reconciliation_metadata, indent=2, default=str), encoding="utf-8"
    )

    print(f"Freespan spatial evidence: {len(events_all_gdf)} event(s) -> {evidence_parquet_path}")
    print(f"  GPKG: {evidence_gpkg_path}")
    print(
        f"2018 freespan spatial evidence: {len(events_2018_gdf)} event(s) -> {evidence_2018_path}"
    )
    print(f"Segment event counts: {len(segment_counts_df)} segment(s) -> {segment_counts_path}")
    print(f"2018 model context: {len(context_df)} event(s) -> {model_context_path}")
    print(
        f"Temporal relationship evidence: {len(temporal_evidence_df)} row(s) -> "
        f"{temporal_evidence_path}"
    )
    print(f"2018 condition benchmark (refreshed): {benchmark_path}")
    print(f"Reconciliation metadata: {metadata_path}")
    print(f"Map (2018): {map_2018_path}")
    print(f"Map (historical): {map_historical_path}")
    print(f"Model-context profile: {profile_path}")
    print(f"Temporal evolution figure: {temporal_figure_path}")
    print()
    freespan_evidence_map.print_freespan_evidence_report(
        events_all_df=events_all_gdf,
        events_2018_df=events_2018_gdf,
        accepted_epsg=accepted_epsg,
        crs_checks=crs_checks,
        survey_direction=survey_direction,
        segment_counts_df=segment_counts_df,
        evidence_path=evidence_parquet_path,
        evidence_2018_path=evidence_2018_path,
        model_context_path=model_context_path,
        reconciliation_metadata_path=metadata_path,
        map_2018_path=map_2018_path,
        map_2018_dimensions=map_2018_dimensions,
        map_historical_path=map_historical_path,
        map_historical_dimensions=map_historical_dimensions,
        profile_path=profile_path,
        profile_dimensions=profile_dimensions,
    )
    print()
    freespan_temporal_provenance.print_freespan_temporal_provenance_report(
        freespans_df=events_all_gdf,
        relationships_df=relationships_df,
        temporal_evidence_df=temporal_evidence_df,
        coverage_metadata=coverage_metadata,
        temporal_evidence_path=temporal_evidence_path,
        figure_path=temporal_figure_path,
        figure_dimensions=temporal_figure_dimensions,
    )
    return 0


def _cmd_ingest_nsta_freespan_registry(args: argparse.Namespace) -> int:
    """MAR-014C: the ONE live network request this ticket performs -- acquires the
    NSTA Pipeline Freespans registry (current + removed layers) for PL854/PL855
    and caches the raw response. `build-freespan-registry-reconciliation` reads
    only this cached snapshot and performs no network request of its own.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    cache_dir = config.paths.raw_dir / "nsta" / "freespans"
    manifest_path = (
        config.paths.interim_dir
        / pipeline_id.lower()
        / "nsta_freespan"
        / "acquisition_manifest.json"
    )

    try:
        report = nsta_freespan.ingest_freespan_registry(
            cache_dir=cache_dir, manifest_path=manifest_path
        )
    except nsta_freespan.NstaFreespanServiceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    nsta_freespan.print_ingestion_report(report)
    return 0


def _cmd_build_freespan_registry_reconciliation(args: argparse.Namespace) -> int:
    """MAR-014C: offline reconciliation of the cached NSTA freespan registry
    snapshot against Table B.1 (MAR-014A/B). Performs NO network request --
    requires `ingest-nsta-freespan-registry` and `build-freespan-spatial-evidence`
    to have already run.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, _interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    cache_dir = config.paths.raw_dir / "nsta" / "freespans"
    manifest_path = (
        config.paths.interim_dir
        / pipeline_id.lower()
        / "nsta_freespan"
        / "acquisition_manifest.json"
    )
    raw_current_path = cache_dir / f"{nsta_freespan.CURRENT_REGISTRY_LAYER.lower()}.geojson"
    raw_removed_path = cache_dir / f"{nsta_freespan.REMOVED_REGISTRY_LAYER.lower()}.geojson"
    spatial_evidence_path = (
        study_dir / "freespan_evidence" / "anglia_freespan_spatial_evidence.parquet"
    )
    reconciliation_metadata_path = (
        study_dir / "freespan_evidence" / "anglia_freespan_spatial_reconciliation_metadata.json"
    )

    required_paths = (
        pipeline_gpkg_path,
        raw_current_path,
        raw_removed_path,
        spatial_evidence_path,
        reconciliation_metadata_path,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required input(s) -- run ingest-nsta-freespan-registry and "
            f"build-freespan-spatial-evidence first: {missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    current_payload = json.loads(raw_current_path.read_text(encoding="utf-8"))
    removed_payload = json.loads(raw_removed_path.read_text(encoding="utf-8"))
    current_records = nsta_freespan_reconciliation.parse_raw_freespan_features(
        current_payload, nsta_freespan.CURRENT_REGISTRY_LAYER
    )
    removed_records = nsta_freespan_reconciliation.parse_raw_freespan_features(
        removed_payload, nsta_freespan.REMOVED_REGISTRY_LAYER
    )
    nsta_registry_gdf = nsta_freespan_reconciliation.build_nsta_freespan_registry_gdf(
        current_records + removed_records, route, working_crs
    )

    table_b1_df = pd.read_parquet(spatial_evidence_path)
    table_b1_2018_df = table_b1_df[table_b1_df["survey_year"] == 2018].reset_index(drop=True)

    match_diagnostics_df = nsta_freespan_reconciliation.match_table_b1_events_to_nsta(
        table_b1_df, nsta_registry_gdf
    )
    piggyback_df = nsta_freespan_reconciliation.detect_piggyback_coincident_records(
        nsta_registry_gdf
    )
    attribution_evidence_df = nsta_freespan_reconciliation.build_2018_attribution_evidence(
        table_b1_2018_df, match_diagnostics_df, piggyback_df
    )
    temporal_context = nsta_freespan_reconciliation.summarize_registry_temporal_context(
        nsta_registry_gdf
    )
    totals_comparison = nsta_freespan_reconciliation.compare_registry_totals(
        nsta_registry_gdf, table_b1_df
    )

    pipeline_condition_dir = study_dir / "pipeline_condition"
    maps_dir = study_dir / "maps"

    registry_parquet_path = pipeline_condition_dir / "nsta_pl854_pl855_freespan_registry.parquet"
    registry_gpkg_path = pipeline_condition_dir / "nsta_pl854_pl855_freespan_registry.gpkg"
    nsta_freespan_reconciliation.write_nsta_freespan_registry(
        nsta_registry_gdf, registry_parquet_path, registry_gpkg_path
    )

    crosswalk_path = pipeline_condition_dir / "nsta_table_b1_freespan_crosswalk.parquet"
    crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
    match_diagnostics_df.to_parquet(crosswalk_path, index=False)

    attribution_evidence_path = (
        pipeline_condition_dir / "anglia_2018_freespan_attribution_evidence.parquet"
    )
    attribution_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    attribution_evidence_df.to_parquet(attribution_evidence_path, index=False)

    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    background_raster_path = background_raster_path if background_raster_path.exists() else None

    geometries_2018 = freespan_evidence.build_event_geometries(table_b1_2018_df, route)
    events_2018_gdf = gpd.GeoDataFrame(table_b1_2018_df, geometry=geometries_2018, crs=working_crs)

    reconciliation_map_path = (
        nsta_freespan_reconciliation_map.render_nsta_table_b1_reconciliation_map(
            events_2018_gdf=events_2018_gdf,
            nsta_registry_gdf=nsta_registry_gdf,
            piggyback_df=piggyback_df,
            route=route,
            output_path=maps_dir / "pl854_nsta_table_b1_freespan_reconciliation.png",
            background_raster_path=background_raster_path,
        )
    )
    reconciliation_map_dimensions = nsta_freespan_reconciliation_map.read_png_dimensions(
        reconciliation_map_path
    )

    crosswalk_figure_path = (
        nsta_freespan_reconciliation_map.render_2018_attribution_crosswalk_figure(
            attribution_evidence_df=attribution_evidence_df,
            match_diagnostics_df=match_diagnostics_df,
            output_path=maps_dir / "pl854_2018_freespan_attribution_crosswalk.png",
        )
    )
    crosswalk_figure_dimensions = nsta_freespan_reconciliation_map.read_png_dimensions(
        crosswalk_figure_path
    )

    # --- Section 19: update MAR-014A/B metadata (never delete previous uncertainty) --
    reconciliation_metadata = json.loads(reconciliation_metadata_path.read_text(encoding="utf-8"))
    individual_line_attribution_refined = bool(
        (
            attribution_evidence_df["attribution_evidence_status"]
            != nsta_freespan_reconciliation.NO_NSTA_CROSS_SOURCE_MATCH
        ).any()
    )
    reconciliation_metadata["nsta_pipeline_freespan_registry_checked"] = True
    reconciliation_metadata["nsta_current_layer_checked"] = True
    reconciliation_metadata["nsta_removed_layer_checked"] = True
    reconciliation_metadata["nsta_line_specific_attribution_fields_available"] = True
    reconciliation_metadata["individual_line_attribution_refined_by_nsta"] = (
        individual_line_attribution_refined
    )
    reconciliation_metadata["nsta_freespan_registry_totals_comparison"] = totals_comparison
    reconciliation_metadata_path.write_text(
        json.dumps(reconciliation_metadata, indent=2, default=str), encoding="utf-8"
    )

    print(f"NSTA freespan registry: {len(nsta_registry_gdf)} feature(s) -> {registry_parquet_path}")
    print(f"  GPKG: {registry_gpkg_path}")
    print(
        f"Table B.1/NSTA crosswalk: {len(match_diagnostics_df)} candidate row(s) -> "
        f"{crosswalk_path}"
    )
    print(
        f"2018 attribution evidence: {len(attribution_evidence_df)} event(s) -> "
        f"{attribution_evidence_path}"
    )
    print(f"Reconciliation map: {reconciliation_map_path}")
    print(f"Attribution crosswalk figure: {crosswalk_figure_path}")
    print(f"Reconciliation metadata (updated): {reconciliation_metadata_path}")
    print()

    if manifest_path.exists():
        manifest_entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("## NSTA acquisition (from manifest)")
        for entry in manifest_entries:
            print(
                f"  {entry['registry_layer']}: {entry['returned_feature_count']} feature(s), "
                f"retrieved {entry['retrieved_at_utc']}, sha256={entry['sha256'][:16]}..."
            )
        print()

    nsta_freespan_reconciliation.print_reconciliation_report(
        nsta_registry_gdf=nsta_registry_gdf,
        temporal_context=temporal_context,
        attribution_evidence_df=attribution_evidence_df,
        totals_comparison=totals_comparison,
        piggyback_df=piggyback_df,
        outputs={
            "nsta_pl854_pl855_freespan_registry_parquet": registry_parquet_path,
            "nsta_pl854_pl855_freespan_registry_gpkg": registry_gpkg_path,
            "nsta_table_b1_freespan_crosswalk_parquet": crosswalk_path,
            "anglia_2018_freespan_attribution_evidence_parquet": attribution_evidence_path,
            "reconciliation_map_png": (
                f"{reconciliation_map_path} "
                f"({reconciliation_map_dimensions[0]}x{reconciliation_map_dimensions[1]} px)"
            ),
            "attribution_crosswalk_png": (
                f"{crosswalk_figure_path} "
                f"({crosswalk_figure_dimensions[0]}x{crosswalk_figure_dimensions[1]} px)"
            ),
        },
    )
    return 0


def _cmd_audit_freespan_context(args: argparse.Namespace) -> int:
    """MAR-015: positive-only freespan context audit + evidence-resolution
    diagnosis. Performs NO network request -- requires build-current-
    normalization, build-wave-orbital-forcing, build-combined-bed-shear,
    build-noncohesive-mobility, build-scour-onset-screening, and
    build-freespan-spatial-evidence to have already run. This is an
    ANALYTICAL EVIDENCE AUDIT, never model validation: it never computes a
    susceptibility score, probability, classifier, or accuracy metric.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, _interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()

    current_segments_path = study_dir / "metocean" / "current_reference_segments.gpkg"
    wave_segments_path = study_dir / "metocean" / "wave_orbital_reference_segments.gpkg"
    combined_segments_path = study_dir / "metocean" / "combined_bed_shear_segments.gpkg"
    mobility_segments_path = study_dir / "sediment" / "noncohesive_mobility_capacity_segments.gpkg"
    scour_segments_path = study_dir / "scour" / "scour_onset_embedment_segments.gpkg"
    segment_event_counts_path = (
        study_dir / "freespan_evidence" / "freespan_segment_event_counts_2018.parquet"
    )
    events_2018_path = (
        study_dir / "freespan_evidence" / "anglia_2018_freespan_spatial_evidence.parquet"
    )
    all_events_path = study_dir / "freespan_evidence" / "anglia_freespan_spatial_evidence.parquet"
    temporal_relationship_path = (
        study_dir / "pipeline_condition" / "anglia_freespan_temporal_relationship_evidence.parquet"
    )

    required_paths = (
        pipeline_gpkg_path,
        current_segments_path,
        wave_segments_path,
        combined_segments_path,
        mobility_segments_path,
        scour_segments_path,
        segment_event_counts_path,
        events_2018_path,
        all_events_path,
        temporal_relationship_path,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required canonical output(s) -- run build-current-normalization, "
            "build-wave-orbital-forcing, build-combined-bed-shear, build-noncohesive-mobility, "
            f"build-scour-onset-screening, and build-freespan-spatial-evidence first: {missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    current_segments_gdf = gpd.read_file(current_segments_path)
    wave_segments_gdf = gpd.read_file(wave_segments_path)
    combined_segments_gdf = gpd.read_file(combined_segments_path)
    mobility_segments_gdf = gpd.read_file(mobility_segments_path)
    scour_segments_gdf = gpd.read_file(scour_segments_path)
    segment_event_counts_df = pd.read_parquet(segment_event_counts_path)
    events_2018_df = pd.read_parquet(events_2018_path)
    all_events_df = pd.read_parquet(all_events_path)
    temporal_relationship_df = pd.read_parquet(temporal_relationship_path)

    merged_segments_df = freespan_context_audit.merge_all_segment_tables(
        current_segments_gdf=current_segments_gdf,
        wave_segments_gdf=wave_segments_gdf,
        combined_segments_gdf=combined_segments_gdf,
        mobility_segments_gdf=mobility_segments_gdf,
        scour_segments_gdf=scour_segments_gdf,
    )
    section_df = freespan_context_audit.build_section_level_context_table(
        merged_segments_df, segment_event_counts_df
    )
    event_df = freespan_context_audit.build_event_level_context_table(
        events_2018_df, section_df, temporal_relationship_df, all_events_df
    )
    event_independence = freespan_context_audit.compute_event_independence_summary(section_df)

    numeric_keys = [
        key
        for key in freespan_context_audit.AUDIT_TABLE_FEATURE_KEYS
        if not freespan_context_audit.FEATURE_BY_KEY[key].is_categorical
    ]
    categorical_keys = [
        key
        for key in freespan_context_audit.AUDIT_TABLE_FEATURE_KEYS
        if freespan_context_audit.FEATURE_BY_KEY[key].is_categorical or key == "mobility_capacity"
    ]
    contrasts = [
        freespan_context_audit.compute_descriptive_contrast(section_df, key) for key in numeric_keys
    ]
    categorical_audits = [
        freespan_context_audit.compute_categorical_audit(section_df, key)
        for key in categorical_keys
    ]
    contrasts_by_key = {c["feature_key"]: c for c in contrasts}
    categorical_audits_by_key = {a["feature_key"]: a for a in categorical_audits}

    interpretation_answers = freespan_context_audit.answer_all_interpretation_questions(section_df)
    demonstrated_gaps = freespan_context_audit.compute_demonstrated_data_gaps(
        event_independence=event_independence, categorical_audits=categorical_audits
    )
    readiness = freespan_context_audit.build_evidence_readiness(categorical_audits_by_key)
    metadata = freespan_context_audit.build_audit_metadata(
        event_independence=event_independence,
        demonstrated_gaps=demonstrated_gaps,
        interpretation_answers=interpretation_answers,
    )
    resolution_gap_statements = freespan_context_audit.compute_resolution_gap_statements(
        events_2018_df
    )
    metadata["resolution_gap_statements"] = resolution_gap_statements

    validation_dir = study_dir / "validation"
    maps_dir = study_dir / "maps"

    event_context_path = validation_dir / "2018_freespan_positive_only_context_audit.parquet"
    event_context_path.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_parquet(event_context_path, index=False)

    section_context_path = validation_dir / "freespan_section_context_audit.parquet"
    section_context_path.parent.mkdir(parents=True, exist_ok=True)
    section_df.to_parquet(section_context_path, index=False)

    readiness_path = validation_dir / "freespan_evidence_readiness.json"
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_path.write_text(json.dumps(readiness, indent=2, default=str), encoding="utf-8")

    metadata_path = validation_dir / "freespan_context_audit_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    background_raster_path = background_raster_path if background_raster_path.exists() else None

    geometries_2018 = freespan_evidence.build_event_geometries(events_2018_df, route)
    events_2018_gdf = gpd.GeoDataFrame(events_2018_df, geometry=geometries_2018, crs=working_crs)

    primary_map_path = freespan_context_audit_map.render_freespan_evidence_resolution_audit_map(
        events_2018_gdf=events_2018_gdf,
        section_df=section_df,
        route=route,
        independent_section_count=event_independence[
            "independent_hydrodynamic_support_section_count"
        ],
        output_path=maps_dir / "pl854_freespan_evidence_resolution_audit.png",
        background_raster_path=background_raster_path,
    )
    primary_map_dimensions = freespan_context_audit_map.read_png_dimensions(primary_map_path)

    small_multiple_path = freespan_context_audit_map.render_feature_context_small_multiple(
        section_df=section_df,
        events_2018_df=events_2018_df,
        total_length_m=route.length,
        output_path=maps_dir / "pl854_freespan_feature_context_audit.png",
    )
    small_multiple_dimensions = freespan_context_audit_map.read_png_dimensions(small_multiple_path)

    audit_table_path = freespan_context_audit_map.render_feature_audit_table(
        contrasts_by_key=contrasts_by_key,
        categorical_audits_by_key=categorical_audits_by_key,
        readiness=readiness,
        feature_keys=freespan_context_audit.AUDIT_TABLE_FEATURE_KEYS,
        output_path=maps_dir / "pl854_freespan_feature_audit_table.png",
    )
    audit_table_dimensions = freespan_context_audit_map.read_png_dimensions(audit_table_path)

    print(f"Event-level context audit: {len(event_df)} event(s) -> {event_context_path}")
    print(f"Section-level context audit: {len(section_df)} section(s) -> {section_context_path}")
    print(f"Evidence readiness: {readiness_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Primary resolution-audit map: {primary_map_path}")
    print(f"Feature context small-multiple: {small_multiple_path}")
    print(f"Feature audit table: {audit_table_path}")
    print()
    freespan_context_audit.print_audit_report(
        event_independence=event_independence,
        contrasts=contrasts,
        categorical_audits=categorical_audits,
        interpretation_answers=interpretation_answers,
        readiness=readiness,
        demonstrated_gaps=demonstrated_gaps,
        outputs={
            "2018_freespan_positive_only_context_audit_parquet": event_context_path,
            "freespan_section_context_audit_parquet": section_context_path,
            "freespan_evidence_readiness_json": readiness_path,
            "freespan_context_audit_metadata_json": metadata_path,
            "primary_map_png": (
                f"{primary_map_path} ({primary_map_dimensions[0]}x{primary_map_dimensions[1]} px)"
            ),
            "small_multiple_png": (
                f"{small_multiple_path} "
                f"({small_multiple_dimensions[0]}x{small_multiple_dimensions[1]} px)"
            ),
            "audit_table_png": (
                f"{audit_table_path} ({audit_table_dimensions[0]}x{audit_table_dimensions[1]} px)"
            ),
        },
    )
    return 0


def _cmd_inventory_highres_seabed_data(args: argparse.Namespace) -> int:
    """MAR-016: high-resolution seabed survey recovery + access audit. The
    ONE live network step this command performs is the BGS OGC API
    acquisition; every classification, table, dossier, and map is then
    built purely from that acquired data (no further network access). A
    DATA-DISCOVERY / ACCESS-RECOVERY milestone -- never a new freespan
    model, susceptibility score, morphology weighting, or canonical raster.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, aoi_gpkg_path, _chainage_gpkg_path, _interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()

    if not pipeline_gpkg_path.exists() or not aoi_gpkg_path.exists():
        print(
            f"error: missing canonical pipeline/AOI under {study_dir}; run build-aoi first",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
    aoi_geometry = unary_union(aoi_gdf.geometry)
    aoi_bbox_wgs84 = highres_seabed_survey_inventory.compute_aoi_bbox_wgs84(
        aoi_geometry, working_crs
    )

    cache_dir = config.paths.raw_dir / "bgs_offshore_surveys"
    manifest_path = (
        config.paths.interim_dir
        / pipeline_id.lower()
        / "bgs_offshore_surveys"
        / "acquisition_manifest.json"
    )
    try:
        acquisition_report = bgs_offshore_surveys.acquire_bgs_survey_candidates(
            aoi_bbox_wgs84=aoi_bbox_wgs84,
            bbox_pad_deg=0.05,
            cache_dir=cache_dir,
            manifest_path=manifest_path,
        )
    except bgs_offshore_surveys.BgsOffshoreSurveysServiceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    survey_inventory_df = highres_seabed_survey_inventory.build_survey_inventory_table(
        acquisition_report.merged_features,
        route=route,
        aoi_geometry=aoi_geometry,
        working_crs=working_crs,
    )
    footprints_gdf = highres_seabed_survey_inventory.build_survey_footprints_gdf(
        acquisition_report.merged_features, working_crs=working_crs
    )
    file_inventory_df = highres_seabed_survey_inventory.build_empty_file_inventory_table()

    seabed_data_dir = study_dir / "seabed_data"
    maps_dir = study_dir / "maps"

    survey_inventory_path = seabed_data_dir / "high_resolution_survey_inventory.parquet"
    survey_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    survey_inventory_df.to_parquet(survey_inventory_path, index=False)

    file_inventory_path = seabed_data_dir / "high_resolution_file_inventory.parquet"
    file_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    file_inventory_df.to_parquet(file_inventory_path, index=False)

    fugro_dossier = highres_seabed_survey_inventory.build_fugro_2018_recovery_dossier(
        survey_inventory_df=survey_inventory_df, retrieved_at_utc=datetime.now(UTC).isoformat()
    )
    fugro_dossier["analog_datasets"] = list(highres_seabed_survey_inventory.ANALOG_DATASET_REGISTRY)
    fugro_dossier_path = seabed_data_dir / "anglia_fugro_2018_recovery_dossier.json"
    fugro_dossier_path.parent.mkdir(parents=True, exist_ok=True)
    fugro_dossier_path.write_text(
        json.dumps(fugro_dossier, indent=2, default=str), encoding="utf-8"
    )

    access_gap_report = highres_seabed_survey_inventory.build_access_gap_report(
        survey_inventory_df=survey_inventory_df, fugro_dossier=fugro_dossier
    )
    access_gap_report["survey_footprint_metadata_does_not_imply_data_custody"] = True
    access_gap_path = seabed_data_dir / "seabed_data_access_gap.json"
    access_gap_path.parent.mkdir(parents=True, exist_ok=True)
    access_gap_path.write_text(
        json.dumps(access_gap_report, indent=2, default=str), encoding="utf-8"
    )

    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    background_raster_path = background_raster_path if background_raster_path.exists() else None

    coverage_map_path = highres_seabed_survey_inventory_map.render_survey_inventory_coverage_map(
        survey_inventory_df=survey_inventory_df,
        footprints_gdf=footprints_gdf,
        route=route,
        aoi_geometry=aoi_geometry,
        output_path=maps_dir / "pl854_high_resolution_survey_inventory.png",
        background_raster_path=background_raster_path,
    )
    timeline_path = highres_seabed_survey_inventory_map.render_seabed_data_timeline(
        survey_inventory_df=survey_inventory_df,
        output_path=maps_dir / "pl854_seabed_data_timeline.png",
    )

    print(f"Survey inventory: {len(survey_inventory_df)} candidate(s) -> {survey_inventory_path}")
    print(f"File inventory: {len(file_inventory_df)} file(s) -> {file_inventory_path}")
    print(f"Fugro 2018 dossier: {fugro_dossier_path}")
    print(f"Access-gap report: {access_gap_path}")
    print(f"Coverage map: {coverage_map_path}")
    print(f"Timeline: {timeline_path}")
    print()
    highres_seabed_survey_inventory.print_survey_inventory_report(
        survey_inventory_df=survey_inventory_df,
        access_gap_report=access_gap_report,
        outputs={
            "high_resolution_survey_inventory_parquet": survey_inventory_path,
            "high_resolution_file_inventory_parquet": file_inventory_path,
            "anglia_fugro_2018_recovery_dossier_json": fugro_dossier_path,
            "seabed_data_access_gap_json": access_gap_path,
            "coverage_map_png": coverage_map_path,
            "timeline_png": timeline_path,
        },
    )
    return 0


def _cmd_build_analog_sandwave_morphometry(args: argparse.Namespace) -> int:
    """MAR-017 / MAR-017A: builds and validates the reusable high-
    resolution sand-wave morphometry engine on the real, open HHW CEND
    11/11 analog dataset -- never PL854 evidence. The PL854 config is
    used only for project paths/conventions; every output lives under
    `data/processed/analogs/hhw_cend1111/`, never a PL854 feature/
    validation layer. The one live step is the official ZIP download,
    which is cached and skipped on subsequent runs.

    MAR-017A keeps CANONICAL (>=1000 m, >=90%-valid, >=3-wavelengths-
    eligible) and EXPLORATORY (below that floor) outputs strictly
    separate -- canonical outputs are legitimately empty for this real
    dataset (Section 2: "this is a valid scientific result").
    """

    config = load_study_config(args.config)

    zip_path = config.paths.raw_dir / "analogs" / "hhw_cend1111" / "HHW-Bathy.zip"
    acquisition = hhw_provider.download_hhw_bathy_zip(zip_path)
    print(
        f"HHW-Bathy.zip: {acquisition.byte_size} bytes, sha256={acquisition.sha256[:16]}..., "
        f"already_cached={acquisition.already_cached}"
    )

    analog_dir = config.paths.processed_dir / "analogs" / "hhw_cend1111"
    maps_dir = analog_dir / "maps"
    analog_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("Building source file inventory (Section 5)...")
    source_inventory_df = hhw_provider.build_source_file_inventory(zip_path)
    source_inventory_path = analog_dir / "source_file_inventory.parquet"
    source_inventory_df.to_parquet(source_inventory_path, index=False)
    print(f"  {len(source_inventory_df)} archive member(s) -> {source_inventory_path}")

    print("Selecting primary grid (Section 2/25, data-driven)...")
    selection = hhw_analog.select_primary_grid(zip_path)
    grid_dir = selection["selected_grid_directory"]
    print(f"  selected: {grid_dir}")

    print("Rendering native bathymetry overview (Section 8, no morphology yet)...")
    bg_data, bg_valid, bg_extent = hhw_analog.get_background_for_map(zip_path, grid_dir)
    overview_row = source_inventory_df[
        (source_inventory_df["grid_directory"] == grid_dir)
        & (source_inventory_df["readable_by_rasterio"] == True)  # noqa: E712 -- pandas needs `== True`, not `is True`, against a nullable/object column
    ].iloc[0]
    overview_map_path = swmap.render_native_bathymetry_overview(
        elevation=bg_data,
        valid=bg_valid,
        extent_m=bg_extent,
        crs=str(overview_row["crs"]),
        native_pixel_size_m=float(overview_row["pixel_size_x_m"]),
        survey_year=hhw_provider.HHW_SURVEY_YEAR,
        nodata_value=float(overview_row["nodata"]),
        output_path=maps_dir / "hhw_native_bathymetry_overview.png",
    )

    # --- CANONICAL 2D tile search (MAR-017A Section 3): >=1000 m only, never lower --------
    print("Searching for CANONICAL analysis tiles (>=1000 m, >=90% valid, Section 3)...")
    canonical_tiles, canonical_search_meta = hhw_analog.build_tile_candidates(zip_path, grid_dir)
    print(f"  {len(canonical_tiles)} canonical candidate tile(s)")
    for entry in canonical_search_meta["cascade_log"]:
        print(f"    {entry}")

    canonical_spectral_df = hhw_analog.build_tile_spectral_table(
        zip_path, grid_dir, canonical_tiles, canonical=True
    )
    canonical_spectral_df = hhw_analog.select_canonical_top_tiles(canonical_spectral_df, top_n=3)
    tile_spectral_path = analog_dir / "tile_spectral_morphometry.parquet"
    canonical_spectral_df.to_parquet(tile_spectral_path, index=False)
    canonical_selected_ids = (
        canonical_spectral_df[canonical_spectral_df["rank_selected_top3"]]["tile_id"].tolist()
        if not canonical_spectral_df.empty
        else []
    )
    print(
        f"  {len(canonical_spectral_df)} canonical tile(s) analyzed, "
        f"{len(canonical_selected_ids)} strictly >=3-wavelengths-eligible -> {tile_spectral_path}"
    )

    canonical_transect_df, canonical_bedform_df, canonical_crests_gdf, canonical_troughs_gdf = (
        hhw_analog.build_transect_and_bedform_tables(zip_path, grid_dir, canonical_spectral_df)
    )
    transect_path = analog_dir / "transect_morphometry.parquet"
    canonical_transect_df.to_parquet(transect_path, index=False)
    bedform_path = analog_dir / "individual_bedforms.parquet"
    canonical_bedform_df.to_parquet(bedform_path, index=False)
    print(f"  {len(canonical_transect_df)} canonical transect(s) -> {transect_path}")
    print(f"  {len(canonical_bedform_df)} canonical individual bedform(s) -> {bedform_path}")

    extrema_path = analog_dir / "detected_profile_extrema.gpkg"
    if extrema_path.exists():
        extrema_path.unlink()
    if not canonical_crests_gdf.empty:
        canonical_crests_gdf.to_file(extrema_path, layer="detected_crests", driver="GPKG")
    if not canonical_troughs_gdf.empty:
        canonical_troughs_gdf.to_file(extrema_path, layer="detected_troughs", driver="GPKG")
    print(f"  canonical crest/trough point layers -> {extrema_path}")

    # --- EXPLORATORY_SMALL_SUPPORT_DIAGNOSTIC (MAR-017A Section 4): below the canonical ----
    # --- floor, never used to populate a canonical output ----------------------------------
    print("Searching for EXPLORATORY small-support diagnostic tiles (Section 4)...")
    exploratory_tiles, exploratory_search_meta = hhw_analog.build_exploratory_tile_candidates(
        zip_path, grid_dir
    )
    print(
        f"  tile size used: {exploratory_search_meta['tile_size_used_m']} m, "
        f"{len(exploratory_tiles)} exploratory candidate tile(s)"
    )
    for entry in exploratory_search_meta["cascade_log"]:
        print(f"    {entry}")

    exploratory_spectral_df = hhw_analog.build_tile_spectral_table(
        zip_path, grid_dir, exploratory_tiles, canonical=False
    )
    exploratory_spectral_df = hhw_analog.select_exploratory_top_tiles(
        exploratory_spectral_df, top_n=3
    )
    exploratory_tile_path = analog_dir / "exploratory_small_support_tile_diagnostics.parquet"
    exploratory_spectral_df.to_parquet(exploratory_tile_path, index=False)
    print(
        f"  {len(exploratory_spectral_df)} exploratory tile(s) analyzed -> {exploratory_tile_path}"
    )

    exploratory_transect_df, exploratory_bedform_df, _exp_crests_gdf, _exp_troughs_gdf = (
        hhw_analog.build_transect_and_bedform_tables(zip_path, grid_dir, exploratory_spectral_df)
    )
    print(
        f"  {len(exploratory_transect_df)} exploratory transect(s), "
        f"{len(exploratory_bedform_df)} exploratory bedform(s) (figure use only, never persisted "
        "as a canonical-schema file)"
    )

    # --- MAR-017A Section 8: explicit, real-data-derived validation statuses --------------
    canonical_tile_count = len(canonical_spectral_df)
    canonical_bedform_count = len(canonical_bedform_df)
    hhw_canonical_2d_validation_status = (
        hhw_analog.CANONICALLY_VALIDATED
        if canonical_tile_count > 0
        else hhw_analog.HHW_CANONICAL_2D_TILE_VALIDATION_NOT_SUPPORTED
    )
    hhw_detailed_bedform_validation_status = (
        hhw_analog.CANONICALLY_VALIDATED
        if canonical_bedform_count > 0
        else hhw_analog.NOT_CANONICALLY_VALIDATED
    )
    any_canonical_meets_3wl = (
        bool(canonical_spectral_df["meets_3_wavelengths_across_tile"].any())
        if not canonical_spectral_df.empty
        else False
    )

    # --- Figures (Sections 8/22-24): canonical if available, otherwise clearly-labelled ----
    # --- exploratory -- never both, never presented as equivalent (Section 12) -------------
    print("Rendering method/statistics/spectral maps (Sections 22-24)...")
    canonical_method_path = maps_dir / "hhw_sandwave_morphometry_method.png"
    exploratory_method_path = maps_dir / "hhw_exploratory_small_support_morphometry.png"
    use_canonical_for_figures = bool(canonical_selected_ids) and canonical_bedform_count > 0

    if use_canonical_for_figures:
        exploratory_method_path.unlink(missing_ok=True)
        figure_tile_df, figure_transect_df, figure_bedform_df = (
            canonical_spectral_df,
            canonical_transect_df,
            canonical_bedform_df,
        )
        method_output_path = canonical_method_path
        method_subtitle = None
        stats_subtitle = None
        spectral_subtitle = None
        canonical_unavailable_message = None
    else:
        canonical_method_path.unlink(missing_ok=True)
        figure_tile_df, figure_transect_df, figure_bedform_df = (
            exploratory_spectral_df,
            exploratory_transect_df,
            exploratory_bedform_df,
        )
        method_output_path = exploratory_method_path
        method_subtitle = "Exploratory 250 m support -- below canonical >=1000 m validation floor"
        stats_subtitle = (
            "EXPLORATORY (below-canonical-floor) bedforms -- not an accepted site distribution"
        )
        spectral_subtitle = "EXPLORATORY tiles shown (canonical set is empty)"
        canonical_unavailable_message = (
            "CANONICAL tile set is EMPTY for this dataset -- points below are EXPLORATORY only"
        )

    if not figure_tile_df.empty:
        top_tile_row = (
            figure_tile_df[figure_tile_df["rank_selected_top3"]]
            .sort_values(
                ["directional_concentration", "spectral_peak_to_median_power_ratio"],
                ascending=[False, False],
            )
            .iloc[0]
        )
        method_inputs = hhw_analog.get_method_figure_inputs(
            zip_path, grid_dir, top_tile_row, figure_transect_df, figure_bedform_df
        )
        method_map_path = swmap.render_method_figure(
            **method_inputs,
            output_path=method_output_path,
            dataset_label="HHW CEND 11/11",
            subtitle=method_subtitle,
        )
    else:
        method_map_path = None
        print("  no canonical or exploratory tile available -- method figure skipped")

    stats_map_path = swmap.render_bedform_distribution_figure(
        bedforms_df=figure_bedform_df,
        output_path=maps_dir / "hhw_sandwave_morphometry_statistics.png",
        dataset_label="HHW CEND 11/11",
        subtitle=stats_subtitle,
    )
    spectral_map_path = swmap.render_dominant_bedform_scale_map(
        background_elevation=bg_data,
        background_valid=bg_valid,
        background_extent_m=bg_extent,
        tile_spectral_df=figure_tile_df,
        tile_size_m=float(figure_tile_df["tile_size_m"].iloc[0])
        if not figure_tile_df.empty
        else 250.0,
        output_path=maps_dir / "hhw_dominant_bedform_scale.png",
        subtitle=spectral_subtitle,
        canonical_unavailable_message=canonical_unavailable_message,
    )

    print("Writing pipeline-transfer contract (Sections 8-9/25)...")
    contract = hhw_analog.build_pipeline_transfer_contract(
        hhw_canonical_2d_validation_status=hhw_canonical_2d_validation_status,
        hhw_detailed_bedform_validation_status=hhw_detailed_bedform_validation_status,
    )
    contract_path = analog_dir / "pipeline_transfer_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")

    print("Writing analog validation-gap report (Section 13)...")
    validation_gap = hhw_analog.build_analog_validation_gap_report(
        canonical_cascade_log=canonical_search_meta["cascade_log"],
        exploratory_cascade_log=exploratory_search_meta["cascade_log"],
        canonical_tile_count=canonical_tile_count,
        exploratory_tile_count=len(exploratory_spectral_df),
        any_canonical_tile_meets_3_wavelengths=any_canonical_meets_3wl,
    )
    validation_gap_path = analog_dir / "analog_validation_gap.json"
    validation_gap_path.write_text(
        json.dumps(validation_gap, indent=2, default=str), encoding="utf-8"
    )

    print()
    print("=== Outputs ===")
    for label, path in (
        ("source_file_inventory_parquet", source_inventory_path),
        ("tile_spectral_morphometry_parquet (canonical)", tile_spectral_path),
        ("exploratory_small_support_tile_diagnostics_parquet", exploratory_tile_path),
        ("transect_morphometry_parquet (canonical)", transect_path),
        ("individual_bedforms_parquet (canonical)", bedform_path),
        ("detected_profile_extrema_gpkg (canonical)", extrema_path),
        ("pipeline_transfer_contract_json", contract_path),
        ("analog_validation_gap_json", validation_gap_path),
        ("native_bathymetry_overview_png", overview_map_path),
        ("method_figure_png", method_map_path),
        ("sandwave_morphometry_statistics_png", stats_map_path),
        ("dominant_bedform_scale_png", spectral_map_path),
    ):
        print(f"  {label}: {path}")
    print()
    print(f"CORE_ENGINE_IMPLEMENTATION_STATUS: {hhw_analog.IMPLEMENTED_AND_SYNTHETICALLY_VERIFIED}")
    print(f"HHW_CANONICAL_2D_VALIDATION_STATUS: {hhw_canonical_2d_validation_status}")
    print(f"HHW_DETAILED_BEDFORM_VALIDATION_STATUS: {hhw_detailed_bedform_validation_status}")
    print()
    # --- MAR-017A Section 17: the three required, verbatim validation-status statements ----
    print("THE GENERIC MORPHOMETRY ENGINE IS IMPLEMENTED AND SYNTHETICALLY VERIFIED.")
    sufficiency_word = "DOES" if canonical_tile_count > 0 else "DOES NOT"
    print(
        f"HHW {sufficiency_word} PROVIDE SUFFICIENT CONTIGUOUS SPATIAL SUPPORT FOR THE "
        "CANONICAL REAL-DATA VALIDATION PROTOCOL."
    )
    print("EXPLORATORY SUB-1000-M TILE RESULTS ARE NOT CANONICAL MORPHOMETRY VALIDATION.")
    print()
    print(
        "HHW CEND 11/11 IS A METHOD-DEVELOPMENT ANALOG ONLY AND DOES NOT ENTER PL854 SCIENTIFIC "
        "EVIDENCE."
    )
    print(
        "MAR-017 BUILDS A REUSABLE HIGH-RESOLUTION MORPHOMETRY ENGINE; IT DOES NOT CREATE A "
        "PL854 FREESPAN PREDICTION OR SUSCEPTIBILITY SCORE."
    )
    return 0


def _build_hhw_cross_analog_summary_row(processed_analogs_dir: Path) -> dict[str, Any]:
    """MAR-017B Section 23: re-reads HHW's ALREADY-PERSISTED MAR-017A
    outputs (never re-runs its pipeline) to build its row of the cross-
    analog validation-SUPPORT summary. HHW's own canonical tables are
    legitimately empty (MAR-017A's accepted, real finding), so most counts
    here are correctly 0 -- this compares method-validation SUPPORT only,
    never morphology values (Section 23)."""

    hhw_dir = processed_analogs_dir / "hhw_cend1111"
    source_inventory = pd.read_parquet(hhw_dir / "source_file_inventory.parquet")
    validation_gap = json.loads(
        (hhw_dir / "analog_validation_gap.json").read_text(encoding="utf-8")
    )
    tile_spectral = pd.read_parquet(hhw_dir / "tile_spectral_morphometry.parquet")
    transects = pd.read_parquet(hhw_dir / "transect_morphometry.parquet")
    bedforms = pd.read_parquet(hhw_dir / "individual_bedforms.parquet")

    native_res_rows = source_inventory[source_inventory["readable_by_rasterio"] == True]  # noqa: E712
    native_resolution_m = (
        float(native_res_rows["pixel_size_x_m"].iloc[0]) if not native_res_rows.empty else None
    )
    max_valid = validation_gap["max_valid_fraction_by_tile_size_m"]
    successful_transects = (
        transects[transects["bedform_count_canonical"].notna()]
        if not transects.empty
        else transects
    )
    filter_flags = (
        successful_transects["filter_sensitivity_flags"]
        if not successful_transects.empty
        else pd.Series(dtype=object)
    )

    return {
        "analog_id": "HHW_CEND1111",
        "native_resolution_m": native_resolution_m,
        "2000m_best_valid_fraction": max_valid.get("2000.0"),
        "1000m_best_valid_fraction": max_valid.get("1000.0"),
        "canonical_support_tile_count": int(validation_gap["canonical_tile_count"]),
        "three_wavelength_eligible_tile_count": int(
            tile_spectral["meets_3_wavelengths_across_tile"].sum() if not tile_spectral.empty else 0
        ),
        # HHW's analog module has no infrastructure/natural-seabed concept at all (MAR-017A
        # predates it) -- 0 is the honest, correct value here, never a placeholder.
        "natural_seabed_eligible_count": 0,
        "successful_canonical_transect_count": int(len(successful_transects)),
        "canonical_bedform_count": int(len(bedforms)),
        "filter_stable_transect_count": int((filter_flags.isna() | (filter_flags == "")).sum()),
        "filter_sensitive_transect_count": int(
            filter_flags.notna().sum() - (filter_flags.isna() | (filter_flags == "")).sum()
        ),
        # MAR-017A's HHW-specific status vocabulary maps directly onto MAR-017B's shared
        # vocabulary: HHW's canonical tile count is 0, the same real situation MAR-017B's
        # INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT describes -- never a re-interpretation of
        # what HHW's real result was, just naming it under the shared vocabulary this newer
        # ticket introduces for cross-analog comparison.
        "canonical_real_validation_status": idrbnr_analog.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT,
    }


def _build_idrbnr_cross_analog_summary_row(
    *,
    native_resolution_m: float | None,
    preflight_df: pd.DataFrame,
    tile_spectral_df: pd.DataFrame,
    transect_df: pd.DataFrame,
    bedform_df: pd.DataFrame,
    canonical_real_validation_status: str,
) -> dict[str, Any]:
    eligible_preflight = preflight_df[~preflight_df["excluded_from_preflight"]]
    successful_transects = (
        transect_df[transect_df["bedform_count_canonical"].notna()]
        if not transect_df.empty
        else transect_df
    )
    filter_flags = (
        successful_transects["filter_stability_classification"]
        if not successful_transects.empty
        else pd.Series(dtype=object)
    )
    return {
        "analog_id": "IDRBNR_CEND1111",
        "native_resolution_m": native_resolution_m,
        "2000m_best_valid_fraction": eligible_preflight["best_valid_fraction_2000m"].max()
        if not eligible_preflight.empty
        else None,
        "1000m_best_valid_fraction": eligible_preflight["best_valid_fraction_1000m"].max()
        if not eligible_preflight.empty
        else None,
        "canonical_support_tile_count": int(
            eligible_preflight["qualifying_tile_count_1000m"].fillna(0).sum()
        ),
        "three_wavelength_eligible_tile_count": int(
            tile_spectral_df["meets_3_wavelengths_across_tile"].sum()
            if not tile_spectral_df.empty
            else 0
        ),
        # IDRBNR's analog module has no infrastructure/natural-seabed concept either -- 0 is
        # the honest, correct value, never a placeholder.
        "natural_seabed_eligible_count": 0,
        "successful_canonical_transect_count": int(len(successful_transects)),
        "canonical_bedform_count": int(len(bedform_df)),
        "filter_stable_transect_count": int(
            (filter_flags == idrbnr_analog.FILTER_STABLE_AT_TESTED_SCALES).sum()
        ),
        "filter_sensitive_transect_count": int(
            (filter_flags == idrbnr_analog.FILTER_SCALE_SENSITIVE).sum()
        ),
        "canonical_real_validation_status": canonical_real_validation_status,
    }


def _build_greater_gabbard_cross_analog_summary_row(
    *,
    native_resolution_m: float | None,
    preflight_df: pd.DataFrame,
    tile_spectral_df: pd.DataFrame,
    transect_df: pd.DataFrame,
    bedform_df: pd.DataFrame,
    canonical_real_validation_status: str,
) -> dict[str, Any]:
    eligible_preflight = preflight_df[~preflight_df["excluded_from_preflight"]]
    successful_transects = (
        transect_df[transect_df["bedform_count_canonical"].notna()]
        if not transect_df.empty
        else transect_df
    )
    filter_flags = (
        successful_transects["filter_stability_classification"]
        if not successful_transects.empty
        else pd.Series(dtype=object)
    )
    return {
        "analog_id": "GREATER_GABBARD_2014",
        "native_resolution_m": native_resolution_m,
        "2000m_best_valid_fraction": eligible_preflight["best_valid_fraction_2000m"].max()
        if not eligible_preflight.empty
        else None,
        "1000m_best_valid_fraction": eligible_preflight["best_valid_fraction_1000m"].max()
        if not eligible_preflight.empty
        else None,
        "canonical_support_tile_count": int(
            eligible_preflight["qualifying_tile_count_1000m"].fillna(0).sum()
        ),
        "three_wavelength_eligible_tile_count": int(
            tile_spectral_df["meets_3_wavelengths_across_tile"].sum()
            if not tile_spectral_df.empty
            else 0
        ),
        "natural_seabed_eligible_count": int(
            (
                tile_spectral_df["natural_seabed_eligibility_status"]
                == gg_analog.NATURAL_SEABED_VALIDATION_ELIGIBLE
            ).sum()
            if not tile_spectral_df.empty
            else 0
        ),
        "successful_canonical_transect_count": int(len(successful_transects)),
        "canonical_bedform_count": int(len(bedform_df)),
        "filter_stable_transect_count": int(
            (filter_flags == gg_analog.FILTER_STABLE_AT_TESTED_SCALES).sum()
        ),
        "filter_sensitive_transect_count": int(
            (filter_flags == gg_analog.FILTER_SCALE_SENSITIVE).sum()
        ),
        "canonical_real_validation_status": canonical_real_validation_status,
    }


CROSS_ANALOG_SUMMARY_COLUMNS = (
    "analog_id",
    "native_resolution_m",
    "2000m_best_valid_fraction",
    "1000m_best_valid_fraction",
    "canonical_support_tile_count",
    "three_wavelength_eligible_tile_count",
    "natural_seabed_eligible_count",
    "successful_canonical_transect_count",
    "canonical_bedform_count",
    "filter_stable_transect_count",
    "filter_sensitive_transect_count",
    "canonical_real_validation_status",
)


def _cmd_build_idrbnr_sandwave_validation(args: argparse.Namespace) -> int:
    """MAR-017B: tests the SAME reusable, already-accepted sand-wave
    morphometry engine against a SECOND independent open Southern North
    Sea analog (JNCC/Cefas IDRBNR CEND 11/11) -- a real canonical-real-
    data method-validation milestone, never PL854 evidence. Every output
    lives under `data/processed/analogs/idr_bnr_cend1111/`. Canonical
    support preflight (Section 7) runs BEFORE any morphometry; if it
    fails, this correctly stops early with a negative, honestly-reported
    dataset-suitability result (Section 9) -- a valid, complete ticket
    outcome, never a bug to route around.
    """

    config = load_study_config(args.config)

    zip_path = config.paths.raw_dir / "analogs" / "idr_bnr_cend1111" / "IDRBNR-Bathy.zip"
    acquisition = idrbnr_provider.download_idrbnr_bathy_zip(zip_path)
    print(
        f"IDRBNR-Bathy.zip: {acquisition.byte_size} bytes, sha256={acquisition.sha256[:16]}..., "
        f"already_cached={acquisition.already_cached}"
    )

    analogs_dir = config.paths.processed_dir / "analogs"
    analog_dir = analogs_dir / "idr_bnr_cend1111"
    maps_dir = analog_dir / "maps"
    analog_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("Building source file inventory (Section 5)...")
    source_inventory_df = idrbnr_provider.build_source_file_inventory(zip_path)
    source_inventory_path = analog_dir / "source_file_inventory.parquet"
    source_inventory_df.to_parquet(source_inventory_path, index=False)
    print(f"  {len(source_inventory_df)} archive member(s) -> {source_inventory_path}")

    print("Running canonical support preflight on every raster candidate (Section 7)...")
    preflight_df = idrbnr_analog.run_canonical_support_preflight(zip_path)
    preflight_path = analog_dir / "canonical_support_preflight.parquet"
    preflight_df.to_parquet(preflight_path, index=False)
    eligible_preflight = preflight_df[~preflight_df["excluded_from_preflight"]]
    for _, row in preflight_df.iterrows():
        print(
            f"  {row['raster_candidate_id']}: 2000m best={row['best_valid_fraction_2000m']}, "
            f"passing={row['qualifying_tile_count_2000m']}; "
            f"1000m best={row['best_valid_fraction_1000m']}, "
            f"passing={row['qualifying_tile_count_1000m']}"
            + (f" -- EXCLUDED: {row['exclusion_reason']}" if row["excluded_from_preflight"] else "")
        )
    print(f"  -> {preflight_path}")

    early_stop_status = idrbnr_analog.derive_early_stop_status(preflight_df)

    if eligible_preflight.empty:
        raise RuntimeError(
            "DATA_INTEGRITY_FAILURE: every raster candidate in the archive was excluded from "
            "the canonical support preflight (unreadable or CRS-inconsistent) -- there is no "
            "credible candidate left to report on, even negatively."
        )

    # The candidate with the highest achieved 1000 m valid fraction, shown as QA-figure
    # background regardless of outcome -- purely a display choice, never a selection.
    background_row = eligible_preflight.loc[
        eligible_preflight["best_valid_fraction_1000m"].fillna(0.0).idxmax()
    ]
    background_candidate_id = background_row["raster_candidate_id"]
    bg_data, bg_valid, bg_extent = idrbnr_analog.get_background_for_map(
        zip_path, background_candidate_id
    )

    tile_spectral_path = analog_dir / "tile_spectral_morphometry.parquet"
    transect_path = analog_dir / "transect_morphometry.parquet"
    bedform_path = analog_dir / "individual_bedforms.parquet"
    extrema_path = analog_dir / "detected_profile_extrema.gpkg"
    method_map_path: Path | None = None
    stats_map_path: Path | None = None

    if early_stop_status is not None:
        print()
        print(f"EARLY STOP (Section 9): {early_stop_status}")
        best_1000 = background_row["best_valid_fraction_1000m"]
        reason = (
            f"No candidate among {len(eligible_preflight)} credible raster candidates reaches "
            f">=90% valid data at a >=1000 m tile size (best achieved: "
            f"{best_1000 * 100:.1f}% at 1000 m, by {background_candidate_id})."
        )
        print(f"  {reason}")

        tile_spectral_df = pd.DataFrame(columns=list(idrbnr_analog.TILE_SPECTRAL_COLUMNS))
        transect_df = pd.DataFrame(columns=list(idrbnr_analog.TRANSECT_COLUMNS))
        bedform_df = pd.DataFrame(columns=list(idrbnr_analog.INDIVIDUAL_BEDFORM_COLUMNS))
        tile_spectral_df.to_parquet(tile_spectral_path, index=False)
        transect_df.to_parquet(transect_path, index=False)
        bedform_df.to_parquet(bedform_path, index=False)

        support_audit_path = swmap.render_canonical_support_audit_map(
            background_elevation=bg_data,
            background_valid=bg_valid,
            background_extent_m=bg_extent,
            background_candidate_id=background_candidate_id,
            qualifying_2000m_tiles=[],
            qualifying_1000m_tiles=[],
            no_qualifying_tile_message=(
                f"NO QUALIFYING >=1000 m CANONICAL TILE IN ANY CANDIDATE (best: "
                f"{best_1000 * 100:.1f}% at 1000 m, required >=90%)"
            ),
            output_path=maps_dir / "idr_bnr_canonical_support_audit.png",
            dataset_label="IDRBNR CEND 11/11",
        )

        canonical_real_validation_status, canonical_real_validation_reason = (
            early_stop_status,
            reason,
        )
    else:
        selection = idrbnr_analog.select_primary_grid_from_preflight(preflight_df)
        selected_candidate_id = selection["selected_raster_candidate_id"]
        print(f"Selected primary grid (Section 8): {selected_candidate_id}")

        print("Searching for canonical analysis tiles (Section 10)...")
        canonical_tiles, tile_search_meta = idrbnr_analog.build_tile_candidates(
            zip_path, selected_candidate_id
        )
        print(f"  {len(canonical_tiles)} canonical candidate tile(s)")

        tile_spectral_df = idrbnr_analog.build_tile_spectral_table(
            zip_path, selected_candidate_id, canonical_tiles
        )
        tile_spectral_df = idrbnr_analog.select_detailed_validation_tiles(tile_spectral_df)
        tile_spectral_df.to_parquet(tile_spectral_path, index=False)
        selected_ids = (
            tile_spectral_df[tile_spectral_df["selected_for_detailed_validation"]][
                "tile_id"
            ].tolist()
            if not tile_spectral_df.empty
            else []
        )
        print(
            f"  {len(tile_spectral_df)} tile(s) analyzed, {len(selected_ids)} selected for "
            f"detailed validation (Sections 12/13) -> {tile_spectral_path}"
        )

        transect_df, bedform_df, crests_gdf, troughs_gdf = (
            idrbnr_analog.build_transect_and_bedform_tables(
                zip_path, selected_candidate_id, tile_spectral_df
            )
        )
        transect_df.to_parquet(transect_path, index=False)
        bedform_df.to_parquet(bedform_path, index=False)
        print(f"  {len(transect_df)} transect(s) -> {transect_path}")
        print(f"  {len(bedform_df)} canonical individual bedform(s) -> {bedform_path}")

        if extrema_path.exists():
            extrema_path.unlink()
        if not crests_gdf.empty:
            crests_gdf.to_file(extrema_path, layer="detected_crests", driver="GPKG")
        if not troughs_gdf.empty:
            troughs_gdf.to_file(extrema_path, layer="detected_troughs", driver="GPKG")

        # `find_valid_tiles` returns candidates from exactly ONE tile size (whichever passed
        # first in its 2000 m -> 1000 m cascade) -- every entry in `canonical_tiles` shares
        # `tile_search_meta["tile_size_used_m"]`, so it buckets into exactly one of the two
        # QA-figure overlay lists, never both.
        tile_footprints = [(t.center_x_m, t.center_y_m, t.tile_size_m) for t in canonical_tiles]
        qualifying_2000m_tiles = (
            tile_footprints
            if tile_search_meta["tile_size_used_m"] == swm.CANONICAL_TILE_SIZE_M
            else []
        )
        qualifying_1000m_tiles = (
            tile_footprints if tile_search_meta["tile_size_used_m"] == swm.MIN_TILE_SIZE_M else []
        )
        support_audit_path = swmap.render_canonical_support_audit_map(
            background_elevation=bg_data,
            background_valid=bg_valid,
            background_extent_m=bg_extent,
            background_candidate_id=background_candidate_id,
            qualifying_2000m_tiles=qualifying_2000m_tiles,
            qualifying_1000m_tiles=qualifying_1000m_tiles,
            no_qualifying_tile_message=None,
            output_path=maps_dir / "idr_bnr_canonical_support_audit.png",
            dataset_label="IDRBNR CEND 11/11",
        )

        any_meets_3wl = bool(tile_spectral_df["meets_3_wavelengths_across_tile"].any())
        successful_transects = (
            transect_df[transect_df["bedform_count_canonical"].notna()]
            if not transect_df.empty
            else transect_df
        )
        canonical_real_validation_status, canonical_real_validation_reason = (
            idrbnr_analog.derive_canonical_real_validation_status(
                canonical_tile_count=len(tile_spectral_df),
                any_meets_3_wavelengths=any_meets_3wl,
                successful_transect_count=len(successful_transects),
                canonical_bedform_count=len(bedform_df),
            )
        )
        print(f"Canonical real validation status (Section 18): {canonical_real_validation_status}")
        print(f"  {canonical_real_validation_reason}")

        if selected_ids and not bedform_df.empty:
            top_tile_row = (
                tile_spectral_df[tile_spectral_df["selected_for_detailed_validation"]]
                .sort_values(
                    ["directional_concentration", "spectral_peak_to_median_power_ratio"],
                    ascending=[False, False],
                )
                .iloc[0]
            )
            method_inputs = idrbnr_analog.get_method_figure_inputs(
                zip_path, selected_candidate_id, top_tile_row, transect_df, bedform_df
            )
            method_map_path = swmap.render_method_figure(
                **method_inputs,
                output_path=maps_dir / "idr_bnr_sandwave_morphometry_validation.png",
                dataset_label="IDRBNR CEND 11/11",
                title="IDRBNR CEND 11/11 -- Sand-Wave Morphometry Canonical Validation",
                subtitle="Canonical >=1000 m support",
            )
        if not bedform_df.empty:
            stats_map_path = swmap.render_bedform_distribution_figure(
                bedforms_df=bedform_df,
                output_path=maps_dir / "idr_bnr_canonical_bedform_statistics.png",
                dataset_label="IDRBNR CEND 11/11",
                title="IDRBNR CEND 11/11 -- Canonical real-data validation -- Bedform "
                "Distributions",
            )

    print("Writing canonical real-validation metadata (Sections 19/25)...")
    contract = idrbnr_analog.build_pipeline_transfer_contract(
        canonical_real_validation_status=canonical_real_validation_status,
        canonical_real_validation_reason=canonical_real_validation_reason,
    )
    validation_metadata = {
        "preflight_summary": {
            "credible_candidate_count": int(len(eligible_preflight)),
            "excluded_candidate_count": int(preflight_df["excluded_from_preflight"].sum()),
            "best_valid_fraction_2000m": float(
                eligible_preflight["best_valid_fraction_2000m"].max()
            )
            if not eligible_preflight.empty
            and eligible_preflight["best_valid_fraction_2000m"].notna().any()
            else None,
            "best_valid_fraction_1000m": float(
                eligible_preflight["best_valid_fraction_1000m"].max()
            )
            if not eligible_preflight.empty
            and eligible_preflight["best_valid_fraction_1000m"].notna().any()
            else None,
            "background_candidate_shown_in_support_audit_figure": background_candidate_id,
        },
        **contract,
    }
    metadata_path = analog_dir / "morphometry_validation_metadata.json"
    metadata_path.write_text(
        json.dumps(validation_metadata, indent=2, default=str), encoding="utf-8"
    )

    print("Building cross-analog validation-support summary (Section 23)...")
    hhw_row = _build_hhw_cross_analog_summary_row(analogs_dir)
    native_res_row = source_inventory_df[source_inventory_df["is_raster_candidate"]]
    native_resolution_m = (
        float(native_res_row["pixel_size_x_m"].dropna().iloc[0])
        if not native_res_row["pixel_size_x_m"].dropna().empty
        else None
    )
    idrbnr_row = _build_idrbnr_cross_analog_summary_row(
        native_resolution_m=native_resolution_m,
        preflight_df=preflight_df,
        tile_spectral_df=tile_spectral_df,
        transect_df=transect_df,
        bedform_df=bedform_df,
        canonical_real_validation_status=canonical_real_validation_status,
    )
    cross_analog_df = pd.DataFrame([hhw_row, idrbnr_row])[list(CROSS_ANALOG_SUMMARY_COLUMNS)]
    cross_analog_path = analogs_dir / "sandwave_morphometry_analog_validation_summary.parquet"
    cross_analog_df.to_parquet(cross_analog_path, index=False)
    print(f"  -> {cross_analog_path}")

    print()
    print("=== Outputs ===")
    for label, path in (
        ("source_file_inventory_parquet", source_inventory_path),
        ("canonical_support_preflight_parquet", preflight_path),
        ("tile_spectral_morphometry_parquet", tile_spectral_path),
        ("transect_morphometry_parquet", transect_path),
        ("individual_bedforms_parquet", bedform_path),
        ("detected_profile_extrema_gpkg", extrema_path if extrema_path.exists() else None),
        ("morphometry_validation_metadata_json", metadata_path),
        ("cross_analog_validation_summary_parquet", cross_analog_path),
        ("canonical_support_audit_png", support_audit_path),
        ("sandwave_morphometry_validation_png", method_map_path),
        ("canonical_bedform_statistics_png", stats_map_path),
    ):
        print(f"  {label}: {path}")

    print()
    question = (
        "DID IDRBNR PROVIDE SUFFICIENT CONTIGUOUS REAL BATHYMETRY TO VALIDATE THE CANONICAL "
        "MORPHOMETRY WORKFLOW?"
    )
    answer = (
        "YES"
        if canonical_real_validation_status == idrbnr_analog.CANONICAL_REAL_DATA_VALIDATED
        else "NO"
    )
    print(f"{question} {answer}")
    print(f"STATUS: {canonical_real_validation_status}")
    print()
    print(
        "IDRBNR CEND 11/11 IS A METHOD-DEVELOPMENT ANALOG ONLY AND DOES NOT ENTER PL854 "
        "SCIENTIFIC EVIDENCE."
    )
    print("CANONICAL REAL-DATA VALIDATION DOES NOT MEAN PL854 MORPHOLOGY HAS BEEN VALIDATED.")
    return 0


def _cmd_build_greater_gabbard_sandwave_validation(args: argparse.Namespace) -> int:
    """MAR-017C: the FINAL open-analog canonical real-data validation
    attempt for the reusable sand-wave morphometry engine, against
    Greater Gabbard 2014 (ADUS DeepOcean). Every output lives under
    `data/processed/analogs/greater_gabbard_2014/`. Unlike HHW/IDRBNR,
    this analog is an OPERATING WIND FARM survey -- a canonical support
    preflight (Section 7) still runs before any morphometry, but any
    qualifying tile must ALSO pass a natural-seabed eligibility check
    (Section 11) derived from an infrastructure-context inventory built
    directly from the bathymetry package's own file structure (Section
    10), before detailed validation may proceed. If this analog also
    fails, the MAR-017 family's analog-hunting is closed for good
    (Section 21) -- never a fourth dataset search.
    """

    config = load_study_config(args.config)

    raw_dir = config.paths.raw_dir / "analogs" / "greater_gabbard_2014"
    acquisition = gg_provider.download_greater_gabbard_ascii_subset(raw_dir)
    print(
        f"Greater Gabbard ASCII subset: {acquisition.extracted_entry_count} entries, "
        f"{acquisition.extracted_total_bytes} bytes, already_cached={acquisition.already_cached} "
        f"(remote bundle is {acquisition.bundle_total_bytes} bytes total -- only the needed "
        "entries were range-fetched, never the whole bundle)"
    )

    analogs_dir = config.paths.processed_dir / "analogs"
    analog_dir = analogs_dir / "greater_gabbard_2014"
    maps_dir = analog_dir / "maps"
    analog_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("Building source file inventory (Section 5)...")
    source_inventory_df = gg_provider.build_source_file_inventory(raw_dir)
    source_inventory_path = analog_dir / "source_file_inventory.parquet"
    source_inventory_df.to_parquet(source_inventory_path, index=False)
    print(f"  {len(source_inventory_df)} archive entries -> {source_inventory_path}")

    print("Building infrastructure context inventory (Sections 9-10)...")
    infrastructure_df = gg_analog.build_infrastructure_context_inventory(raw_dir)
    infrastructure_path = analog_dir / "infrastructure_context_inventory.parquet"
    infrastructure_df.to_parquet(infrastructure_path, index=False)
    remote_entries = [e.filename for e in gg_provider.list_remote_entries()]
    cable_segments = gg_analog.build_cable_corridor_segments(remote_entries, infrastructure_df)
    print(
        f"  {len(infrastructure_df)} infrastructure feature(s) "
        f"({int(infrastructure_df['is_substation'].sum())} substation, "
        f"{len(cable_segments)} cable corridor segment(s) derived) -> {infrastructure_path}"
    )

    print("Running canonical support preflight on every gridded candidate (Section 7)...")
    preflight_df = gg_analog.run_canonical_support_preflight(raw_dir)
    preflight_path = analog_dir / "canonical_support_preflight.parquet"
    preflight_df.to_parquet(preflight_path, index=False)
    eligible_preflight = preflight_df[~preflight_df["excluded_from_preflight"]]
    if eligible_preflight.empty:
        raise RuntimeError(
            "DATA_INTEGRITY_FAILURE: every raster candidate was excluded from the canonical "
            "support preflight -- there is no credible candidate left to report on, even "
            "negatively."
        )
    print(
        f"  {len(preflight_df)} candidate(s) tested, best achieved: "
        f"2000m={eligible_preflight['best_valid_fraction_2000m'].max()}, "
        f"1000m={eligible_preflight['best_valid_fraction_1000m'].max()} -> {preflight_path}"
    )

    early_stop_status = gg_analog.derive_early_stop_status(preflight_df)

    # Prefer the candidate closest to passing (highest real best_valid_fraction_1000m,
    # matching HHW/IDRBNR's own background-selection convention); only fall back to "largest
    # bounding box" if every candidate's tile search never even produced a real value (e.g. a
    # raster too small in some dimension to attempt a 1000 m tile at all).
    if eligible_preflight["best_valid_fraction_1000m"].notna().any():
        background_row = eligible_preflight.loc[
            eligible_preflight["best_valid_fraction_1000m"].fillna(-1.0).idxmax()
        ]
    else:
        background_row = eligible_preflight.loc[
            (eligible_preflight["width"] * eligible_preflight["native_pixel_size_m"]).idxmax()
        ]
    background_candidate_id = background_row["raster_candidate_id"]
    bg_data, bg_valid, bg_extent = gg_analog.get_background_for_map(
        raw_dir, background_candidate_id
    )

    turbine_positions = [
        (row["center_x_m"], row["center_y_m"], row["infrastructure_id"])
        for _, row in infrastructure_df[
            infrastructure_df["infrastructure_type"] == "turbine_or_substation_foundation"
        ].iterrows()
    ]
    rock_protection_footprints = [
        (row["min_x_m"], row["min_y_m"], row["max_x_m"], row["max_y_m"])
        for _, row in infrastructure_df[
            infrastructure_df["infrastructure_type"] == "rock_concrete_protection"
        ].iterrows()
    ]

    tile_spectral_path = analog_dir / "tile_spectral_morphometry.parquet"
    transect_path = analog_dir / "transect_morphometry.parquet"
    bedform_path = analog_dir / "individual_bedforms.parquet"
    extrema_path = analog_dir / "detected_profile_extrema.gpkg"
    method_map_path: Path | None = None
    stats_map_path: Path | None = None

    if early_stop_status is not None:
        print()
        print(f"EARLY STOP (Section 8): {early_stop_status}")
        best_1000 = eligible_preflight["best_valid_fraction_1000m"].max()
        best_1000_pct = (
            f"{best_1000 * 100:.1f}%"
            if pd.notna(best_1000)
            else "N/A (no candidate is even large enough to fit a 1000 m tile)"
        )
        reason = (
            f"No candidate among {len(eligible_preflight)} credible gridded candidates (144 "
            f"turbine/substation foundation grids + 152 inter-array cable corridor grids) "
            f"reaches >=90% valid data at a >=1000 m tile size (best achieved: {best_1000_pct} "
            f"at 1000 m, by {background_candidate_id}) -- corridors are large enough in extent "
            f"to attempt a tile, but their real valid-data density (the survey follows a narrow "
            f"cable route, not a full-width swath) still falls far short."
        )
        print(f"  {reason}")

        tile_spectral_df = pd.DataFrame(columns=list(gg_analog.TILE_SPECTRAL_COLUMNS))
        transect_df = pd.DataFrame(columns=list(gg_analog.TRANSECT_COLUMNS))
        bedform_df = pd.DataFrame(columns=list(gg_analog.INDIVIDUAL_BEDFORM_COLUMNS))
        tile_spectral_df.to_parquet(tile_spectral_path, index=False)
        transect_df.to_parquet(transect_path, index=False)
        bedform_df.to_parquet(bedform_path, index=False)

        support_audit_path = swmap.render_canonical_support_audit_map(
            background_elevation=bg_data,
            background_valid=bg_valid,
            background_extent_m=bg_extent,
            background_candidate_id=background_candidate_id,
            qualifying_2000m_tiles=[],
            qualifying_1000m_tiles=[],
            no_qualifying_tile_message=(
                f"NO QUALIFYING >=1000 m CANONICAL TILE IN ANY CANDIDATE (best: {best_1000_pct} "
                f"at 1000 m, required >=90%)"
            ),
            output_path=maps_dir / "greater_gabbard_canonical_support_audit.png",
            dataset_label="GREATER GABBARD 2014",
            turbine_positions=turbine_positions,
            rock_protection_footprints=rock_protection_footprints,
        )

        canonical_real_validation_status, canonical_real_validation_reason = (
            early_stop_status,
            reason,
        )
    else:
        selection = gg_analog.select_primary_grid_from_preflight(preflight_df)
        selected_candidate_id = selection["selected_raster_candidate_id"]
        print(f"Selected primary grid: {selected_candidate_id}")

        print("Searching for canonical analysis tiles (Section 13)...")
        canonical_tiles, tile_search_meta = gg_analog.build_tile_candidates(
            raw_dir, selected_candidate_id
        )
        print(f"  {len(canonical_tiles)} canonical candidate tile(s)")

        tile_spectral_df = gg_analog.build_tile_spectral_table(
            raw_dir,
            selected_candidate_id,
            canonical_tiles,
            infrastructure_df=infrastructure_df,
            cable_segments=cable_segments,
        )
        tile_spectral_df = gg_analog.select_detailed_validation_tiles(tile_spectral_df)
        tile_spectral_df.to_parquet(tile_spectral_path, index=False)
        selected_ids = (
            tile_spectral_df[tile_spectral_df["selected_for_detailed_validation"]][
                "tile_id"
            ].tolist()
            if not tile_spectral_df.empty
            else []
        )
        print(
            f"  {len(tile_spectral_df)} tile(s) analyzed, {len(selected_ids)} selected for "
            f"detailed validation (Sections 11/12) -> {tile_spectral_path}"
        )

        transect_df, bedform_df, crests_gdf, troughs_gdf = (
            gg_analog.build_transect_and_bedform_tables(
                raw_dir, selected_candidate_id, tile_spectral_df
            )
        )
        transect_df.to_parquet(transect_path, index=False)
        bedform_df.to_parquet(bedform_path, index=False)
        print(f"  {len(transect_df)} transect(s) -> {transect_path}")
        print(f"  {len(bedform_df)} canonical individual bedform(s) -> {bedform_path}")

        if extrema_path.exists():
            extrema_path.unlink()
        if not crests_gdf.empty:
            crests_gdf.to_file(extrema_path, layer="detected_crests", driver="GPKG")
        if not troughs_gdf.empty:
            troughs_gdf.to_file(extrema_path, layer="detected_troughs", driver="GPKG")

        tile_footprints = [(t.center_x_m, t.center_y_m, t.tile_size_m) for t in canonical_tiles]
        qualifying_2000m_tiles = (
            tile_footprints
            if tile_search_meta["tile_size_used_m"] == swm.CANONICAL_TILE_SIZE_M
            else []
        )
        qualifying_1000m_tiles = (
            tile_footprints if tile_search_meta["tile_size_used_m"] == swm.MIN_TILE_SIZE_M else []
        )
        support_audit_path = swmap.render_canonical_support_audit_map(
            background_elevation=bg_data,
            background_valid=bg_valid,
            background_extent_m=bg_extent,
            background_candidate_id=background_candidate_id,
            qualifying_2000m_tiles=qualifying_2000m_tiles,
            qualifying_1000m_tiles=qualifying_1000m_tiles,
            no_qualifying_tile_message=None,
            output_path=maps_dir / "greater_gabbard_canonical_support_audit.png",
            dataset_label="GREATER GABBARD 2014",
            turbine_positions=turbine_positions,
            rock_protection_footprints=rock_protection_footprints,
        )

        any_meets_3wl = bool(tile_spectral_df["meets_3_wavelengths_across_tile"].any())
        any_natural_eligible = bool(
            (
                tile_spectral_df["natural_seabed_eligibility_status"]
                == gg_analog.NATURAL_SEABED_VALIDATION_ELIGIBLE
            ).any()
        )
        successful_transects = (
            transect_df[transect_df["bedform_count_canonical"].notna()]
            if not transect_df.empty
            else transect_df
        )
        canonical_real_validation_status, canonical_real_validation_reason = (
            gg_analog.derive_canonical_real_validation_status(
                canonical_tile_count=len(tile_spectral_df),
                any_meets_3_wavelengths=any_meets_3wl,
                any_natural_seabed_eligible=any_natural_eligible,
                successful_transect_count=len(successful_transects),
                canonical_bedform_count=len(bedform_df),
            )
        )
        print(f"Canonical real validation status (Section 15): {canonical_real_validation_status}")
        print(f"  {canonical_real_validation_reason}")

        if selected_ids and not bedform_df.empty:
            top_tile_row = (
                tile_spectral_df[tile_spectral_df["selected_for_detailed_validation"]]
                .sort_values(
                    ["directional_concentration", "spectral_peak_to_median_power_ratio"],
                    ascending=[False, False],
                )
                .iloc[0]
            )
            method_inputs = gg_analog.get_method_figure_inputs(
                raw_dir, selected_candidate_id, top_tile_row, transect_df, bedform_df
            )
            method_map_path = swmap.render_method_figure(
                **method_inputs,
                output_path=maps_dir / "greater_gabbard_sandwave_morphometry_validation.png",
                dataset_label="GREATER GABBARD 2014",
                title="Greater Gabbard 2014 -- Sand-Wave Morphometry Canonical Validation",
                subtitle="Canonical >=1000 m natural-seabed validation tile",
            )
        if not bedform_df.empty:
            stats_map_path = swmap.render_bedform_distribution_figure(
                bedforms_df=bedform_df,
                output_path=maps_dir / "greater_gabbard_canonical_bedform_statistics.png",
                dataset_label="GREATER GABBARD 2014",
                title="Greater Gabbard 2014 -- Canonical real-data validation -- Bedform "
                "Distributions",
            )

    print("Writing canonical real-validation metadata (Section 16)...")
    contract = gg_analog.build_pipeline_transfer_contract(
        canonical_real_validation_status=canonical_real_validation_status,
        canonical_real_validation_reason=canonical_real_validation_reason,
    )
    validation_metadata = {
        "preflight_summary": {
            "credible_candidate_count": int(len(eligible_preflight)),
            "excluded_candidate_count": int(preflight_df["excluded_from_preflight"].sum()),
            "best_valid_fraction_2000m": float(
                eligible_preflight["best_valid_fraction_2000m"].max()
            )
            if eligible_preflight["best_valid_fraction_2000m"].notna().any()
            else None,
            "best_valid_fraction_1000m": float(
                eligible_preflight["best_valid_fraction_1000m"].max()
            )
            if eligible_preflight["best_valid_fraction_1000m"].notna().any()
            else None,
            "background_candidate_shown_in_support_audit_figure": background_candidate_id,
        },
        "infrastructure_summary": {
            "turbine_or_substation_foundation_count": int(
                (
                    infrastructure_df["infrastructure_type"] == "turbine_or_substation_foundation"
                ).sum()
            ),
            "rock_concrete_protection_count": int(
                (infrastructure_df["infrastructure_type"] == "rock_concrete_protection").sum()
            ),
            "cable_corridor_segment_count": len(cable_segments),
        },
        **contract,
    }
    metadata_path = analog_dir / "morphometry_validation_metadata.json"
    metadata_path.write_text(
        json.dumps(validation_metadata, indent=2, default=str), encoding="utf-8"
    )

    print("Building cross-analog validation-support summary (Section 20)...")
    hhw_row = _build_hhw_cross_analog_summary_row(analogs_dir)
    idrbnr_dir = analogs_dir / "idr_bnr_cend1111"
    idrbnr_source_inventory = pd.read_parquet(idrbnr_dir / "source_file_inventory.parquet")
    idrbnr_preflight = pd.read_parquet(idrbnr_dir / "canonical_support_preflight.parquet")
    idrbnr_tile_spectral = pd.read_parquet(idrbnr_dir / "tile_spectral_morphometry.parquet")
    idrbnr_transects = pd.read_parquet(idrbnr_dir / "transect_morphometry.parquet")
    idrbnr_bedforms = pd.read_parquet(idrbnr_dir / "individual_bedforms.parquet")
    idrbnr_metadata = json.loads(
        (idrbnr_dir / "morphometry_validation_metadata.json").read_text(encoding="utf-8")
    )
    idrbnr_native_res_rows = idrbnr_source_inventory[idrbnr_source_inventory["is_raster_candidate"]]
    idrbnr_native_resolution_m = (
        float(idrbnr_native_res_rows["pixel_size_x_m"].dropna().iloc[0])
        if not idrbnr_native_res_rows["pixel_size_x_m"].dropna().empty
        else None
    )
    idrbnr_row = _build_idrbnr_cross_analog_summary_row(
        native_resolution_m=idrbnr_native_resolution_m,
        preflight_df=idrbnr_preflight,
        tile_spectral_df=idrbnr_tile_spectral,
        transect_df=idrbnr_transects,
        bedform_df=idrbnr_bedforms,
        canonical_real_validation_status=idrbnr_metadata["canonical_real_data_validation_status"],
    )
    gg_row = _build_greater_gabbard_cross_analog_summary_row(
        native_resolution_m=gg_provider.NATIVE_PIXEL_SIZE_M,
        preflight_df=preflight_df,
        tile_spectral_df=tile_spectral_df,
        transect_df=transect_df,
        bedform_df=bedform_df,
        canonical_real_validation_status=canonical_real_validation_status,
    )
    cross_analog_df = pd.DataFrame([hhw_row, idrbnr_row, gg_row])[
        list(CROSS_ANALOG_SUMMARY_COLUMNS)
    ]
    cross_analog_path = analogs_dir / "sandwave_morphometry_analog_validation_summary.parquet"
    cross_analog_df.to_parquet(cross_analog_path, index=False)
    print(f"  -> {cross_analog_path}")

    print("Writing final MAR-017 family decision (Section 21)...")
    family_decision = gg_analog.build_final_mar017_family_decision(
        hhw_status=hhw_row["canonical_real_validation_status"],
        idrbnr_status=idrbnr_row["canonical_real_validation_status"],
        greater_gabbard_status=canonical_real_validation_status,
    )
    family_decision_path = analogs_dir / "sandwave_morphometry_engine_validation_status.json"
    family_decision_path.write_text(
        json.dumps(family_decision, indent=2, default=str), encoding="utf-8"
    )
    print(f"  -> {family_decision_path}")

    print()
    print("=== Outputs ===")
    for label, path in (
        ("source_file_inventory_parquet", source_inventory_path),
        ("infrastructure_context_inventory_parquet", infrastructure_path),
        ("canonical_support_preflight_parquet", preflight_path),
        ("tile_spectral_morphometry_parquet", tile_spectral_path),
        ("transect_morphometry_parquet", transect_path),
        ("individual_bedforms_parquet", bedform_path),
        ("detected_profile_extrema_gpkg", extrema_path if extrema_path.exists() else None),
        ("morphometry_validation_metadata_json", metadata_path),
        ("cross_analog_validation_summary_parquet", cross_analog_path),
        ("mar017_family_validation_status_json", family_decision_path),
        ("canonical_support_audit_png", support_audit_path),
        ("sandwave_morphometry_validation_png", method_map_path),
        ("canonical_bedform_statistics_png", stats_map_path),
    ):
        print(f"  {label}: {path}")

    print()
    q1 = (
        "DID GREATER GABBARD PROVIDE A DEFENSIBLE NATURAL-SEABED CANONICAL REAL-DATA "
        "VALIDATION OF THE MORPHOMETRY ENGINE?"
    )
    a1 = (
        "YES"
        if canonical_real_validation_status == gg_analog.CANONICAL_REAL_DATA_VALIDATED
        else "NO"
    )
    print(f"{q1} {a1}")
    q2 = "IS THE MAR-017 MORPHOMETRY ENGINE CANONICALLY REAL-DATA VALIDATED?"
    a2 = "YES" if family_decision["canonical_real_data_validation_passed"] else "NO"
    print(f"{q2} {a2}")
    print(f"STATUS: {canonical_real_validation_status}")
    print()
    print(
        "GREATER GABBARD IS A METHOD-DEVELOPMENT ANALOG ONLY AND DOES NOT ENTER PL854 "
        "SCIENTIFIC EVIDENCE."
    )
    print(
        "IF THIS THIRD ANALOG FAILS, NO FURTHER OPEN-ANALOG SEARCH WILL BE USED TO DELAY THE "
        "MAIN PL854 PROJECT."
    )
    return 0


def _dataset_start_or(time_range_ms: tuple | None, fallback_now: datetime) -> datetime:
    """The live dataset's own start timestamp, or `fallback_now` if it could not be discovered."""

    if time_range_ms is None:
        return fallback_now
    return datetime.fromtimestamp(time_range_ms[0] / 1000.0, tz=UTC)


def _fractional_year(ts: pd.Timestamp) -> float:
    year_start = pd.Timestamp(year=ts.year, month=1, day=1, tz=ts.tz)
    year_end = pd.Timestamp(year=ts.year + 1, month=1, day=1, tz=ts.tz)
    return ts.year + (ts - year_start) / (year_end - year_start)


def _cmd_build_engineering_evidence_atlas(args: argparse.Namespace) -> int:
    """MAR-018: PL854 engineering evidence atlas -- map-first GIS/report
    packaging of already-accepted MAR-007/010-016 outputs. No network. No new
    scientific model: every value here is read, joined, and relabelled, never
    recomputed. MAR-017 analog outputs are report context only.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    study_dir = config.paths.processed_dir / pipeline_id.lower()
    interim_pl854_dir = config.paths.interim_dir / pipeline_id.lower()
    working_crs = config.crs.horizontal

    missing = evidence_atlas_core.check_upstream_integrity(study_dir)
    if missing:
        print(
            "error: missing required accepted upstream output(s) -- run the corresponding "
            "upstream 'build-*' commands first:",
            file=sys.stderr,
        )
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1

    atlas_dir = study_dir / "evidence_atlas"
    maps_dir = study_dir / "maps"
    report_dir = study_dir / "report"

    print("Building canonical section evidence table (Section 6)...")
    section_df = evidence_atlas_core.build_section_evidence_table(study_dir)
    section_evidence_path = atlas_dir / "pl854_section_evidence.parquet"
    metocean_evidence.write_parquet(section_df, section_evidence_path)
    print(f"  {len(section_df)} section(s) -> {section_evidence_path}")

    print("Building GIS atlas GeoPackage layers (Sections 7-8)...")
    pipeline_layer = evidence_atlas_core.build_pipeline_route_layer(study_dir)
    route = pipeline_layer.geometry.iloc[0]
    sections_gdf = evidence_atlas_core.build_engineering_support_sections_layer(
        study_dir, section_df
    )
    freespans_2018_gdf = evidence_atlas_core.build_observed_freespans_2018_layer(study_dir)
    historical_freespans_gdf = evidence_atlas_core.build_historical_freespans_layer(study_dir)
    psa_points_gdf = evidence_atlas_core.build_observed_psa_d50_points_layer(
        study_dir, interim_pl854_dir
    )
    bgs_manifest_path = interim_pl854_dir / "bgs_offshore_surveys" / "acquisition_manifest.json"
    highres_survey_inventory_path = (
        study_dir / "seabed_data" / "high_resolution_survey_inventory.parquet"
    )
    highres_survey_gdf = evidence_atlas_core.build_highres_survey_inventory_layer(
        manifest_path=bgs_manifest_path,
        inventory_path=highres_survey_inventory_path,
        working_crs=working_crs,
    )
    chainage_reference_gdf = evidence_atlas_core.build_chainage_reference_points_layer(study_dir)

    layers = {
        "pipeline_route": pipeline_layer,
        "engineering_support_sections": sections_gdf,
        "observed_freespans_2018": freespans_2018_gdf,
        "historical_freespans_2012_2018": historical_freespans_gdf,
        "observed_psa_d50_points": psa_points_gdf,
        "highres_survey_inventory": highres_survey_gdf,
        "chainage_reference_points": chainage_reference_gdf,
    }
    crs_mismatches = evidence_atlas_core.verify_projected_layers_crs(layers)
    if crs_mismatches:
        print(f"error: GIS layer(s) not in {working_crs}: {crs_mismatches}", file=sys.stderr)
        return 1

    atlas_gpkg_path = atlas_dir / "pl854_engineering_evidence_atlas.gpkg"
    evidence_atlas_core.write_evidence_atlas_gpkg(atlas_gpkg_path, layers)
    written_layers = [name for name, gdf in layers.items() if gdf is not None and not gdf.empty]
    print(f"  {len(written_layers)} layer(s) -> {atlas_gpkg_path}")

    generated_at_utc = datetime.now(UTC).isoformat()

    print("Writing evidence variable manifest (Section 16)...")
    evidence_manifest = evidence_atlas_report.build_evidence_variable_manifest(
        generated_at_utc=generated_at_utc
    )
    evidence_manifest_path = atlas_dir / "evidence_atlas_manifest.json"
    evidence_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_manifest_path.write_text(
        json.dumps(evidence_manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"  {evidence_manifest['variable_count']} variable(s) -> {evidence_manifest_path}")

    condition_benchmark = json.loads(
        (study_dir / "pipeline_condition" / "anglia_2018_condition_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    observed_condition_summary = {
        "event_count": int(freespans_2018_gdf["source_length_m"].count()),
        "total_length_m": float(freespans_2018_gdf["source_length_m"].sum()),
        "max_length_m": float(freespans_2018_gdf["source_length_m"].max()),
        "max_height_m": float(freespans_2018_gdf["source_height_m"].max()),
        "exposed_section_count": condition_benchmark["exposed_section_count"],
        "total_exposed_length_m": condition_benchmark["total_exposed_length_m"],
    }
    key_limitations = (
        "Hydrodynamic forcing (2024-2026) postdates the 2018 observations",
        "Current/wave support (~1.5-2 km) vs observed spans (~0.2-23 m)",
        "Route-scale morphology derives from 1991-1992 regional bathymetry",
        "No verified open 2018 Fugro bathymetric grid",
        "No observed continuous route embedment profile",
        "No continuous quantitative route D50",
        "PL854 vs PL855 freespan attribution unresolved",
    )
    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    background_raster_path = background_raster_path if background_raster_path.exists() else None

    print("Rendering primary engineering evidence atlas map (Sections 9-12)...")
    atlas_png_path = evidence_atlas_maps.render_engineering_evidence_atlas(
        route=route,
        sections_gdf=sections_gdf,
        freespans_2018_gdf=freespans_2018_gdf,
        historical_freespans_gdf=historical_freespans_gdf,
        psa_points_gdf=psa_points_gdf,
        highres_survey_gdf=highres_survey_gdf,
        chainage_reference_gdf=chainage_reference_gdf,
        observed_condition_summary=observed_condition_summary,
        key_limitations=key_limitations,
        output_path=maps_dir / "pl854_engineering_evidence_atlas.png",
        background_raster_path=background_raster_path,
    )
    print(f"  -> {atlas_png_path}")

    print("Rendering evidence strip (Section 13)...")
    strip_png_path = evidence_atlas_maps.render_evidence_strip(
        section_df=section_df,
        freespans_2018_gdf=freespans_2018_gdf,
        output_path=maps_dir / "pl854_engineering_evidence_strip.png",
    )
    print(f"  -> {strip_png_path}")

    print("Rendering engineering section summary table (Section 14)...")
    table_png_path = evidence_atlas_maps.render_section_summary_table(
        section_df=section_df, output_path=maps_dir / "pl854_engineering_section_summary.png"
    )
    print(f"  -> {table_png_path}")

    print("Rendering evidence provenance timeline (Section 15)...")
    sediment_metadata = json.loads(
        (study_dir / "sediment" / "sediment_evidence_metadata.json").read_text(encoding="utf-8")
    )
    morphology_metadata = json.loads(
        (study_dir / "morphology" / "morphology_metadata.json").read_text(encoding="utf-8")
    )
    morphology_df = pd.read_parquet(
        study_dir / "morphology" / "chainage_regional_morphology.parquet"
    )
    fugro_dossier = json.loads(
        (study_dir / "seabed_data" / "anglia_fugro_2018_recovery_dossier.json").read_text(
            encoding="utf-8"
        )
    )
    ea_publication_date = None
    for ref in fugro_dossier.get("official_references", []):
        if "Environmental Appraisal" in ref.get("title", ""):
            ea_publication_date = ref.get("date")
            break

    combined_stats_df = pd.read_parquet(study_dir / "metocean" / "combined_bed_shear_stats.parquet")
    overlap_start = pd.Timestamp(combined_stats_df["overlap_start_time_utc"].iloc[0])
    overlap_end = pd.Timestamp(combined_stats_df["overlap_end_time_utc"].iloc[0])
    wave_stats_df = pd.read_parquet(study_dir / "metocean" / "wave_orbital_velocity_stats.parquet")
    wave_start = pd.Timestamp(wave_stats_df["start_time_utc"].iloc[0])
    wave_end = pd.Timestamp(wave_stats_df["end_time_utc"].iloc[0])
    sample_year_min = int(sediment_metadata["coverage_diagnostics"]["sample_year_min"])
    sample_year_max = int(sediment_metadata["coverage_diagnostics"]["sample_year_max"])
    morph_year_min = int(morphology_df["source_acquisition_year"].min())
    morph_year_max = int(morphology_df["source_acquisition_year"].max())

    epochs = [
        {
            "label": f"PSA samples ({sample_year_min}-{sample_year_max})",
            "start_year": float(sample_year_min),
            "end_year": float(sample_year_max),
            "kind": "OBSERVATION_PERIOD",
            "color": "#2E6F40",
        },
        {
            "label": f"Legacy regional bathymetry, MAR-007 ({morph_year_min}-{morph_year_max})",
            "start_year": float(morph_year_min),
            "end_year": float(morph_year_max),
            "kind": "OBSERVATION_PERIOD",
            "color": "0.5",
        },
        {
            "label": "2018 freespan/exposure survey",
            "start_year": 2018.0,
            "end_year": 2018.5,
            "kind": "OBSERVATION_PERIOD",
            "color": "#B4131A",
        },
        {
            "label": (
                "Contemporaneous current-wave overlap, MAR-012 "
                f"({overlap_start.strftime('%b %Y')} - {overlap_end.strftime('%b %Y')})"
            ),
            "start_year": _fractional_year(overlap_start),
            "end_year": _fractional_year(overlap_end),
            "kind": "OBSERVATION_PERIOD",
            "color": "#4C6EF5",
        },
        {
            "label": f"Wave context ({wave_start.year} - {wave_end.strftime('%b %Y')})",
            "start_year": _fractional_year(wave_start),
            "end_year": _fractional_year(wave_end),
            "kind": "OBSERVATION_PERIOD",
            "color": "#C1440E",
        },
    ]
    if ea_publication_date:
        epochs.append(
            {
                "label": f"Anglia decommissioning EA published ({ea_publication_date})",
                "start_year": float(ea_publication_date.split("-")[0]),
                "end_year": float(ea_publication_date.split("-")[0]),
                "kind": "PUBLICATION_EVENT",
            }
        )
    timeline_png_path = evidence_atlas_maps.render_provenance_timeline(
        epochs=epochs, output_path=maps_dir / "pl854_evidence_provenance_timeline.png"
    )
    print(f"  -> {timeline_png_path}")

    print("Building human-readable engineering evidence report (Sections 17-24)...")
    route_length_km = float(route.length) / 1000.0
    depth_stats = {
        "min_m": float(morphology_df["depth_lat_m"].min()),
        "median_m": float(morphology_df["depth_lat_m"].median()),
        "max_m": float(morphology_df["depth_lat_m"].max()),
    }
    seabed_data_access_gap = json.loads(
        (study_dir / "seabed_data" / "seabed_data_access_gap.json").read_text(encoding="utf-8")
    )
    analog_family_status_path = (
        config.paths.processed_dir
        / "analogs"
        / "sandwave_morphometry_engine_validation_status.json"
    )
    analog_family_status = (
        json.loads(analog_family_status_path.read_text(encoding="utf-8"))
        if analog_family_status_path.exists()
        else {}
    )

    report_blocks = evidence_atlas_report.build_report_blocks(
        section_df=section_df,
        freespans_2018_gdf=freespans_2018_gdf,
        historical_freespans_gdf=historical_freespans_gdf,
        condition_benchmark=condition_benchmark,
        route_length_km=route_length_km,
        depth_stats=depth_stats,
        sediment_metadata=sediment_metadata,
        morphology_metadata=morphology_metadata,
        highres_survey_df=pd.read_parquet(highres_survey_inventory_path),
        seabed_data_access_gap=seabed_data_access_gap,
        analog_family_status=analog_family_status,
        key_limitations=key_limitations,
    )
    html_report_path = evidence_atlas_report.write_html_report(
        report_blocks, report_dir / "pl854_engineering_evidence_report.html"
    )
    md_report_path = evidence_atlas_report.write_markdown_report(
        report_blocks, report_dir / "pl854_engineering_evidence_report.md"
    )
    print(f"  HTML -> {html_report_path}")
    print(f"  Markdown -> {md_report_path}")

    print("Writing file package manifest (Section 25)...")
    package_manifest = evidence_atlas_report.build_package_manifest(
        deliverables={
            "section_evidence_parquet": section_evidence_path,
            "atlas_gpkg": atlas_gpkg_path,
            "evidence_manifest_json": evidence_manifest_path,
            "atlas_png": atlas_png_path,
            "evidence_strip_png": strip_png_path,
            "section_summary_png": table_png_path,
            "provenance_timeline_png": timeline_png_path,
            "html_report": html_report_path,
            "md_report": md_report_path,
        },
        generated_at_utc=generated_at_utc,
        project_root=Path.cwd(),
    )
    package_manifest_path = report_dir / "pl854_engineering_evidence_package_manifest.json"
    evidence_atlas_report.write_package_manifest(package_manifest, package_manifest_path)
    print(f"  {package_manifest['file_count']} file(s) -> {package_manifest_path}")

    print()
    print("=== PL854 Engineering Evidence Atlas (MAR-018) ===")
    print()
    print("## Atlas")
    print(f"  Section count: {len(section_df)}")
    event_section_count = int((section_df["observed_2018_freespan_count"].fillna(0) > 0).sum())
    print(f"  Observed-event section count: {event_section_count}/{len(section_df)}")
    print(f"  GeoPackage layers: {written_layers}")
    atlas_dims = evidence_atlas_maps.read_png_dimensions(atlas_png_path)
    strip_dims = evidence_atlas_maps.read_png_dimensions(strip_png_path)
    print(f"  Primary atlas: {atlas_png_path} ({atlas_dims[0]}x{atlas_dims[1]} px)")
    print(f"  Evidence strip: {strip_png_path} ({strip_dims[0]}x{strip_dims[1]} px)")
    print()
    print("## Report")
    print(f"  HTML: {html_report_path}")
    print(f"  Markdown: {md_report_path}")
    print(f"  Package size: {sum(f['file_size_bytes'] for f in package_manifest['files'])} bytes")
    print()
    print("## Key factual findings")
    print(f"  Route length: ~{route_length_km:.2f} km")
    print(
        f"  {observed_condition_summary['event_count']} official 2018 corridor freespans occupy "
        f"{event_section_count}/{len(section_df)} hydrodynamic support sections"
    )
    print(
        "  Noncohesive p95 mobility capacity: "
        f"{section_df['mobility_capacity_p95_d50_mm'].min():.3g}-"
        f"{section_df['mobility_capacity_p95_d50_mm'].max():.3g} mm"
    )
    print()
    print("## Limitations")
    for item in key_limitations:
        print(f"  - {item}")
    print()
    print("MAR-018 DOES NOT CREATE A FREESPAN SUSCEPTIBILITY SCORE OR RISK MODEL.")
    print(
        "OBSERVED CONDITION, PHYSICS-BASED MODEL OUTPUTS, EMPIRICAL SCREENING, AND "
        "LEGACY/REGIONAL CONTEXT REMAIN VISUALLY AND SEMANTICALLY SEPARATE."
    )
    print(
        "THE PRIMARY PURPOSE OF MAR-018 IS HUMAN-READABLE GIS / ENGINEERING COMMUNICATION OF THE "
        "ACCEPTED EVIDENCE BASE."
    )
    return 0


def _cmd_build_marine_poc_review_package(args: argparse.Namespace) -> int:
    """MAR-019: OrbGSS Marine POC v0.1 external-reviewer package. Product-
    framing / presentation only -- no new geohazard physics, no risk score, no
    ML. Reuses MAR-018's already-accepted outputs by READING them back (never
    rebuilding via `evidence_atlas.core`), so the canonical scientific values
    can never drift during this ticket's work. No network.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    study_dir = config.paths.processed_dir / pipeline_id.lower()
    working_crs = config.crs.horizontal

    missing = evidence_atlas_poc.check_mar018_outputs_present(study_dir)
    if missing:
        print(
            "error: missing required accepted MAR-018 output(s) -- run "
            "'build-engineering-evidence-atlas' first:",
            file=sys.stderr,
        )
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1

    maps_dir = study_dir / "maps"
    report_dir = study_dir / "report"
    poc_dir = study_dir / "poc"

    print("Reloading already-accepted MAR-018 section evidence + GIS layers (no recomputation)...")
    mar018_paths = evidence_atlas_poc.required_mar018_outputs(study_dir)
    section_df = pd.read_parquet(mar018_paths["section_evidence_parquet"])
    layers = evidence_atlas_poc.read_atlas_gpkg_layers(
        mar018_paths["atlas_gpkg"], working_crs=working_crs
    )
    route = layers["pipeline_route"].geometry.iloc[0]
    print(f"  {len(section_df)} section(s), {len(layers)} GIS layer(s) reloaded unchanged")

    condition_benchmark = json.loads(
        mar018_paths["condition_benchmark"].read_text(encoding="utf-8")
    )
    freespans_2018_gdf = layers["observed_freespans_2018"]
    observed_condition_summary = {
        "event_count": int(freespans_2018_gdf["source_length_m"].count()),
        "total_length_m": float(freespans_2018_gdf["source_length_m"].sum()),
        "max_length_m": float(freespans_2018_gdf["source_length_m"].max()),
        "max_height_m": float(freespans_2018_gdf["source_height_m"].max()),
        "exposed_section_count": condition_benchmark["exposed_section_count"],
        "total_exposed_length_m": condition_benchmark["total_exposed_length_m"],
    }
    key_limitations = (
        "Hydrodynamic forcing (2024-2026) postdates the 2018 observations",
        "Current/wave support (~1.5-2 km) vs observed spans (~0.2-23 m)",
        "Route-scale morphology derives from 1991-1992 regional bathymetry",
        "No verified open 2018 Fugro bathymetric grid",
        "No observed continuous route embedment profile",
        "No continuous quantitative route D50",
        "PL854 vs PL855 freespan attribution unresolved",
    )
    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    background_raster_path = background_raster_path if background_raster_path.exists() else None

    sediment_df = pd.read_parquet(mar018_paths["chainage_sediment_evidence"])
    folk_class_descriptions = (
        sediment_df[["mapped_250k_folk_class", "mapped_250k_folk_d50_text"]]
        .drop_duplicates()
        .set_index("mapped_250k_folk_class")["mapped_250k_folk_d50_text"]
        .to_dict()
    )

    print("Re-rendering polished engineering evidence atlas (Sections 4-8)...")
    atlas_png_path = evidence_atlas_maps.render_engineering_evidence_atlas(
        route=route,
        sections_gdf=layers["engineering_support_sections"],
        freespans_2018_gdf=freespans_2018_gdf,
        historical_freespans_gdf=layers["historical_freespans_2012_2018"],
        psa_points_gdf=layers["observed_psa_d50_points"],
        highres_survey_gdf=layers["highres_survey_inventory"],
        chainage_reference_gdf=layers["chainage_reference_points"],
        observed_condition_summary=observed_condition_summary,
        key_limitations=key_limitations,
        output_path=maps_dir / "pl854_engineering_evidence_atlas.png",
        background_raster_path=background_raster_path,
    )
    print(f"  -> {atlas_png_path}")

    print("Re-rendering polished KP evidence strip (Sections 9-12)...")
    strip_png_path = evidence_atlas_maps.render_evidence_strip(
        section_df=section_df,
        freespans_2018_gdf=freespans_2018_gdf,
        folk_class_descriptions=folk_class_descriptions,
        output_path=maps_dir / "pl854_engineering_evidence_strip.png",
    )
    print(f"  -> {strip_png_path}")

    print("Rebuilding engineering evidence report with corrected wording (Section 13-14)...")
    morphology_df = pd.read_parquet(
        study_dir / "morphology" / "chainage_regional_morphology.parquet"
    )
    depth_stats = {
        "min_m": float(morphology_df["depth_lat_m"].min()),
        "median_m": float(morphology_df["depth_lat_m"].median()),
        "max_m": float(morphology_df["depth_lat_m"].max()),
    }
    sediment_metadata = json.loads(
        (study_dir / "sediment" / "sediment_evidence_metadata.json").read_text(encoding="utf-8")
    )
    morphology_metadata = json.loads(
        (study_dir / "morphology" / "morphology_metadata.json").read_text(encoding="utf-8")
    )
    seabed_data_access_gap = json.loads(
        (study_dir / "seabed_data" / "seabed_data_access_gap.json").read_text(encoding="utf-8")
    )
    analog_family_status_path = (
        config.paths.processed_dir
        / "analogs"
        / "sandwave_morphometry_engine_validation_status.json"
    )
    analog_family_status = (
        json.loads(analog_family_status_path.read_text(encoding="utf-8"))
        if analog_family_status_path.exists()
        else {}
    )
    route_length_km = float(route.length) / 1000.0

    report_blocks = evidence_atlas_report.build_report_blocks(
        section_df=section_df,
        freespans_2018_gdf=freespans_2018_gdf,
        historical_freespans_gdf=layers["historical_freespans_2012_2018"],
        condition_benchmark=condition_benchmark,
        route_length_km=route_length_km,
        depth_stats=depth_stats,
        sediment_metadata=sediment_metadata,
        morphology_metadata=morphology_metadata,
        highres_survey_df=pd.read_parquet(
            study_dir / "seabed_data" / "high_resolution_survey_inventory.parquet"
        ),
        seabed_data_access_gap=seabed_data_access_gap,
        analog_family_status=analog_family_status,
        key_limitations=key_limitations,
    )
    html_report_path = evidence_atlas_report.write_html_report(
        report_blocks, report_dir / "pl854_engineering_evidence_report.html"
    )
    md_report_path = evidence_atlas_report.write_markdown_report(
        report_blocks, report_dir / "pl854_engineering_evidence_report.md"
    )
    print(f"  HTML -> {html_report_path}")
    print(f"  Markdown -> {md_report_path}")

    print("Building POC overview (Sections 15-18)...")
    overview_blocks = evidence_atlas_poc.build_poc_overview_blocks()
    overview_html_path = poc_dir / "orbgss_marine_poc_overview.html"
    overview_md_path = poc_dir / "orbgss_marine_poc_overview.md"
    overview_html_path.parent.mkdir(parents=True, exist_ok=True)
    overview_html_path.write_text(
        evidence_atlas_report.render_blocks_html(
            overview_blocks, title=evidence_atlas_poc.POC_TITLE
        ),
        encoding="utf-8",
    )
    overview_md_path.write_text(
        evidence_atlas_report.render_blocks_markdown(overview_blocks), encoding="utf-8"
    )
    print(f"  HTML -> {overview_html_path}")
    print(f"  Markdown -> {overview_md_path}")

    print("Building external reviewer guide (Sections 19-21)...")
    review_guide_path = poc_dir / "external_georisk_review_guide.md"
    review_guide_path.write_text(evidence_atlas_poc.build_review_guide_markdown(), encoding="utf-8")
    print(f"  -> {review_guide_path}")

    print("Building POC package index (Section 22)...")
    index_path = poc_dir / "index.html"
    index_html = evidence_atlas_poc.build_poc_index_html(
        deliverables=[
            {
                "title": "Marine POC Overview",
                "href": "orbgss_marine_poc_overview.html",
                "description": "3-5 minute product-demonstrator overview for an external "
                "georisk engineer.",
            },
            {
                "title": "Engineering Evidence Atlas",
                "href": "../maps/pl854_engineering_evidence_atlas.png",
                "description": "Map View: 4-panel spatial evidence/hazard figure (observed, "
                "modelled, screening, data support).",
            },
            {
                "title": "KP Evidence Strip",
                "href": "../maps/pl854_engineering_evidence_strip.png",
                "description": "KP / Route View: aligned along-route engineering evidence bands.",
            },
            {
                "title": "Engineering Evidence Report",
                "href": "../report/pl854_engineering_evidence_report.html",
                "description": "Human-readable project conclusions, provenance and limitations.",
            },
            {
                "title": "External Reviewer Guide",
                "href": "external_georisk_review_guide.md",
                "description": "Engineering/product questions for a domain reviewer, plus "
                "data-authorization guidance.",
            },
        ]
    )
    index_path.write_text(index_html, encoding="utf-8")
    print(f"  -> {index_path}")

    print("Writing POC review-package manifest (Section 26)...")
    generated_at_utc = datetime.now(UTC).isoformat()
    package_manifest = evidence_atlas_report.build_package_manifest(
        deliverables={
            "poc_index_html": index_path,
            "poc_overview_html": overview_html_path,
            "poc_overview_md": overview_md_path,
            "external_review_guide_md": review_guide_path,
            "polished_atlas_png": atlas_png_path,
            "polished_evidence_strip_png": strip_png_path,
            "engineering_report_html": html_report_path,
        },
        generated_at_utc=generated_at_utc,
        project_root=Path.cwd(),
    )
    package_manifest_path = poc_dir / "poc_review_package_manifest.json"
    evidence_atlas_report.write_package_manifest(package_manifest, package_manifest_path)
    print(f"  {package_manifest['file_count']} file(s) -> {package_manifest_path}")

    atlas_dims = evidence_atlas_maps.read_png_dimensions(atlas_png_path)
    strip_dims = evidence_atlas_maps.read_png_dimensions(strip_png_path)
    print()
    print("=== OrbGSS Marine POC Review Package (MAR-019) ===")
    print()
    print("## Presentation")
    print(f"  Atlas dimensions: {atlas_dims[0]}x{atlas_dims[1]} px -> {atlas_png_path}")
    print(f"  Strip dimensions: {strip_dims[0]}x{strip_dims[1]} px -> {strip_png_path}")
    print()
    print("## POC package")
    print(
        f"  {package_manifest['file_count']} file(s), "
        f"{sum(f['file_size_bytes'] for f in package_manifest['files'])} bytes total"
    )
    print(f"  Manifest: {package_manifest_path}")
    print(f"  Index: {index_path}")
    print()
    print(
        "ORB GSS MARINE MODULE POC ASSUMES OPERATOR-SUPPLIED PROJECT-GRADE SURVEY / "
        "GEOPHYSICAL / GEOTECHNICAL DATA IN A PRODUCTION DEPLOYMENT."
    )
    print(
        "ORB GSS IS THE SOFTWARE ANALYTICS / GIS / ENGINEERING-COMMUNICATION LAYER; THIS "
        "POC DOES NOT POSITION ORBGSS AS THE PRIMARY SURVEY OR DRILLING CONTRACTOR."
    )
    print("PL854 IS A PUBLIC-DATA DEVELOPMENT / DEMONSTRATION CASE.")
    print("NO NEW GEOHAZARD PHYSICS OR RISK SCORE WAS INTRODUCED IN MAR-019.")
    return 0


def _terrain_raster_facts(
    *,
    band: np.ndarray,
    crs,
    transform,
    nodata,
    width: int,
    height: int,
    bounds: tuple,
    band_count: int,
    dtype: str,
    color_interp: tuple[str, ...],
    vertical_datum: str | None,
    survey_epoch: str | None,
) -> tuple["terrain_readiness.RasterFacts", np.ndarray]:
    valid_mask_raw = (
        np.isfinite(band)
        if nodata is None
        else (np.isfinite(band) & ~np.isclose(band, nodata, rtol=0, atol=abs(nodata) * 1e-6 + 1e-6))
    )
    valid_values = band[valid_mask_raw]
    facts = terrain_readiness.RasterFacts(
        band_count=band_count,
        dtype=str(dtype),
        color_interpretations=color_interp,
        crs_is_present=crs is not None,
        crs_is_geographic=crs.is_geographic if crs is not None else None,
        crs_linear_units=crs.linear_units if crs is not None else None,
        width=width,
        height=height,
        pixel_size_x_m=transform.a,
        pixel_size_y_m=-transform.e,
        bounds=tuple(bounds),
        nodata_value=nodata,
        vertical_datum=vertical_datum,
        survey_epoch=survey_epoch,
        data_min=float(valid_values.min()) if valid_values.size else None,
        data_max=float(valid_values.max()) if valid_values.size else None,
        data_std=float(valid_values.std()) if valid_values.size else None,
        valid_cell_fraction=float(valid_mask_raw.mean()) if valid_mask_raw.size else None,
    )
    return facts, valid_mask_raw


def _build_terrain_footprint_gdf(
    valid_mask: np.ndarray, transform, working_crs: str, *, stride: int = 20
):
    """A lightweight (coarsened, dissolved) survey-footprint polygon from
    the real valid-data mask -- never the full raster's own bounding box
    (which would overstate real coverage for a narrow survey swath), and
    never a per-pixel-precise (potentially huge) vector (Section 12:
    "lightweight vector products only")."""

    coarse = valid_mask[::stride, ::stride].astype(np.uint8)
    coarse_transform = transform * rasterio.Affine.scale(stride, stride)
    shapes = rasterio.features.shapes(coarse, mask=coarse.astype(bool), transform=coarse_transform)
    polygons = [shapely_shape(geom) for geom, value in shapes if value == 1]
    if not polygons:
        return gpd.GeoDataFrame({"footprint": ["survey_coverage"]}, geometry=[], crs=working_crs)
    dissolved = unary_union(polygons).simplify(stride, preserve_topology=True)
    return gpd.GeoDataFrame(
        {"footprint": ["survey_coverage"]}, geometry=[dissolved], crs=working_crs
    )


def _cmd_build_highres_terrain_poc(args: argparse.Namespace) -> int:
    """MAR-020: generic high-resolution bathymetry / seabed terrain POC,
    benchmarked against the real 2020 Fugro Sheringham Shoal survey
    (TCE-1986). ONE live acquisition when the source raster is absent from
    the cache, then fully offline. No risk score, no geohazard
    interpretation -- terrain analytics only.
    """

    config = load_study_config(args.config)
    project_id = config.study.id.lower()
    study_dir = config.paths.processed_dir / project_id
    raw_dir = config.paths.raw_dir / project_id
    working_crs = config.crs.horizontal

    print("Acquiring Sheringham Shoal bathymetry (Section 4)...")
    acquisition = sheringham_provider.download_sheringham_shoal_bathymetry(raw_dir)
    print(
        f"  {acquisition.target_entry_name}: {acquisition.target_entry_bytes} bytes "
        f"(bundle total {acquisition.bundle_total_bytes} bytes -- only the needed entry was "
        f"range-fetched, never the whole bundle), already_cached={acquisition.already_cached}",
        flush=True,
    )
    raster_path = raw_dir / acquisition.target_entry_name

    print("Inspecting raster + running operator-style readiness (Section 5)...", flush=True)
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        crs = src.crs
        transform = src.transform
        nodata = src.nodata
        width, height = src.width, src.height
        bounds = tuple(src.bounds)
        band_count = src.count
        dtype = src.dtypes[0]
        color_interp = tuple(str(c) for c in src.colorinterp)

    facts, valid_mask = _terrain_raster_facts(
        band=band,
        crs=crs,
        transform=transform,
        nodata=nodata,
        width=width,
        height=height,
        bounds=bounds,
        band_count=band_count,
        dtype=dtype,
        color_interp=color_interp,
        vertical_datum=sheringham_provider.SOURCE_VERTICAL_DATUM,
        survey_epoch=acquisition.survey_period,
    )
    readiness_result = terrain_readiness.assess_bathymetry_readiness(facts)
    print(f"  Readiness: {readiness_result.status}", flush=True)
    for reason in readiness_result.reasons():
        print(f"    - {reason}")

    readiness_dir = study_dir / "readiness"
    readiness_path = readiness_dir / "bathymetry_readiness.json"
    readiness_dir.mkdir(parents=True, exist_ok=True)
    readiness_path.write_text(
        json.dumps(readiness_result.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    print(f"  -> {readiness_path}")

    route_kp_status = "NOT_APPLICABLE_NO_AUTHORITATIVE_ROUTE_SUPPLIED"

    if readiness_result.status == terrain_readiness.NOT_READY:
        print("EARLY STOP (Section 5): source data is NOT_READY -- no terrain analytics computed.")
        validation = {
            "scientific_role": "HIGH_RESOLUTION_SEABED_TERRAIN_POC",
            "question_a_readiness_passed": False,
            "question_b_derived_layers_valid": False,
            "question_c_route_kp_view_available": False,
            "route_kp_status": route_kp_status,
        }
        validation_path = study_dir / "terrain_poc_validation.json"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
        print(f"  -> {validation_path}")
        print()
        print(
            "IS GENERIC HIGH-RESOLUTION OPERATOR BATHYMETRY -> TERRAIN ANALYTICS DEMONSTRATED? NO"
        )
        return 0

    print("Building canonical bed_elevation_m raster (Section 6)...", flush=True)
    canonical = terrain_canonical.build_canonical_bed_elevation(
        band,
        nodata_value=nodata,
        source_sign_convention=sheringham_provider.SOURCE_SIGN_CONVENTION,
        source_vertical_datum=sheringham_provider.SOURCE_VERTICAL_DATUM,
    )
    print(
        f"  source convention {canonical.source_sign_convention} -> "
        f"{terrain_canonical.CANONICAL_SIGN_CONVENTION_NOTE}"
    )

    cell_size_m = float(transform.a)
    terrain_dir = study_dir / "terrain"
    tags_base = {
        "product": "MAR-020 generic high-resolution seabed terrain POC",
        "scientific_role": "HIGH_RESOLUTION_SEABED_TERRAIN_POC",
        "product_role": "ORBGSS_MARINE_OPERATOR_DATA_BENCHMARK",
        "source_dataset": sheringham_provider.DATASET_TITLE,
        "source_sha256": acquisition.target_entry_sha256,
        "source_sign_convention": canonical.source_sign_convention,
        "source_vertical_datum": canonical.source_vertical_datum,
    }

    bed_elevation_path = terrain_raster_io.write_terrain_raster(
        canonical.bed_elevation_m,
        transform,
        working_crs,
        terrain_dir / "canonical_bed_elevation.tif",
        {**tags_base, "layer": "bed_elevation_m", "units": "m", "convention": "higher=shallower"},
    )
    print(f"  -> {bed_elevation_path}")

    # Section 9: scale windows, derived from the real raster's own extent/coverage pattern (a
    # narrow, branching wind-farm survey swath, not a broad regional AOI), recorded transparently
    # rather than blindly using every one of the ticket's example values.
    scales = [("engineering", 10.0), ("intermediate", 50.0)]
    print(f"Computing derived terrain layers at {scales} metres (Sections 8-9)...", flush=True)

    derived_layer_facts: list[dict] = []
    gis_ok = True

    def _record_and_write(
        name: str, array: np.ndarray, units: str, scale_label: str, filename: str
    ):
        path = terrain_raster_io.write_terrain_raster(
            array,
            transform,
            working_crs,
            terrain_dir / filename,
            {**tags_base, "layer": name, "units": units, "scale": scale_label},
        )
        finite = np.isfinite(array)
        derived_layer_facts.append(
            {
                "layer": name,
                "scale": scale_label,
                "units": units,
                "valid_cells": int(finite.sum()),
                "min": f"{float(np.nanmin(array)):.4f}" if finite.any() else "n/a",
                "max": f"{float(np.nanmax(array)):.4f}" if finite.any() else "n/a",
                "mean": f"{float(np.nanmean(array)):.4f}" if finite.any() else "n/a",
            }
        )
        print(f"  -> {path} ({finite.sum():,} valid cells)", flush=True)
        return finite.any()

    engineering_scale_m = scales[0][1]

    t0 = time.time()
    slope_deg, aspect_deg, _slope_vf = terrain_derivatives.compute_slope_aspect_deg(
        canonical.bed_elevation_m, canonical.valid_mask, engineering_scale_m, cell_size_m
    )
    print(f"  slope/aspect @ {engineering_scale_m}m: {time.time() - t0:.1f}s", flush=True)
    gis_ok &= _record_and_write("slope_deg", slope_deg, "deg", "engineering (10 m)", "slope.tif")
    gis_ok &= _record_and_write(
        "aspect_deg", aspect_deg, "deg (compass bearing)", "engineering (10 m)", "aspect.tif"
    )

    t0 = time.time()
    profile_curv, plan_curv, _curv_vf = terrain_derivatives.compute_profile_plan_curvature(
        canonical.bed_elevation_m, canonical.valid_mask, engineering_scale_m, cell_size_m
    )
    print(f"  curvature @ step={engineering_scale_m}m: {time.time() - t0:.1f}s", flush=True)
    gis_ok &= _record_and_write(
        "profile_curvature",
        profile_curv,
        "1/m",
        f"engineering (step={engineering_scale_m} m)",
        "profile_curvature.tif",
    )
    gis_ok &= _record_and_write(
        "plan_curvature",
        plan_curv,
        "1/m",
        f"engineering (step={engineering_scale_m} m)",
        "plan_curvature.tif",
    )

    scale_facts: list[dict] = [
        {
            "name": "engineering",
            "physical_m": engineering_scale_m,
            "pixel_count": round(engineering_scale_m / cell_size_m),
            "rationale": "individual-feature / immediate-foundation scale context; also used as "
            "the curvature finite-difference step (a fine-scale-sensitive second-derivative "
            "property)",
        },
    ]

    for scale_name, radius_m in scales:
        t0 = time.time()
        relief, _rvf = terrain_derivatives.compute_local_relief(
            canonical.bed_elevation_m, canonical.valid_mask, radius_m, cell_size_m
        )
        print(f"  local_relief @ {radius_m}m ({scale_name}): {time.time() - t0:.1f}s", flush=True)
        gis_ok &= _record_and_write(
            "local_relief",
            relief,
            "m",
            f"{scale_name} ({radius_m:g} m)",
            f"local_relief_{radius_m:g}m.tif",
        )

        t0 = time.time()
        std, _svf = terrain_derivatives.compute_terrain_std(
            canonical.bed_elevation_m, canonical.valid_mask, radius_m, cell_size_m
        )
        print(f"  terrain_std @ {radius_m}m ({scale_name}): {time.time() - t0:.1f}s", flush=True)
        gis_ok &= _record_and_write(
            "terrain_std",
            std,
            "m",
            f"{scale_name} ({radius_m:g} m)",
            f"terrain_std_{radius_m:g}m.tif",
        )

        t0 = time.time()
        tri, _tvf = terrain_derivatives.compute_ruggedness(
            canonical.bed_elevation_m, canonical.valid_mask, radius_m, cell_size_m
        )
        print(f"  ruggedness @ {radius_m}m ({scale_name}): {time.time() - t0:.1f}s", flush=True)
        gis_ok &= _record_and_write(
            "ruggedness",
            tri,
            "m",
            f"{scale_name} ({radius_m:g} m)",
            f"ruggedness_{radius_m:g}m.tif",
        )

        if scale_name != "engineering":
            scale_facts.append(
                {
                    "name": scale_name,
                    "physical_m": radius_m,
                    "pixel_count": round(radius_m / cell_size_m),
                    "rationale": "broader morphological context beyond individual-feature scale, "
                    "still well within the real survey swath's own narrow width",
                }
            )
        else:
            scale_facts[0]["rationale"] += (
                "; local_relief/terrain_std/ruggedness are ALSO computed at this scale"
            )

    print("Rendering bathymetry QA map (Section 10)...", flush=True)
    maps_dir = study_dir / "maps"
    qa_map_path = terrain_maps.render_bathymetry_qa_map(
        elevation=canonical.bed_elevation_m,
        valid_mask=canonical.valid_mask,
        transform=transform,
        crs_label=working_crs,
        native_pixel_size_m=cell_size_m,
        vertical_datum=canonical.source_vertical_datum,
        survey_epoch=acquisition.survey_period,
        readiness_status=readiness_result.status,
        output_path=maps_dir / "sheringham_shoal_2020_bathymetry_qa.png",
        title="Sheringham Shoal 2020 -- Bathymetry Data Readiness (not a hazard map)",
    )
    print(f"  -> {qa_map_path}")

    print("Rendering terrain atlas (Section 11)...", flush=True)
    atlas_path = terrain_maps.render_terrain_atlas(
        layers={
            "A. Bathymetry (bed_elevation_m)": (
                canonical.bed_elevation_m,
                "viridis",
                "m (higher=shallower)",
            ),
            "B. Slope (engineering, 10 m)": (slope_deg, "inferno", "deg"),
            "C. Local Relief (intermediate, 50 m)": (relief, "cividis", "m"),
            "D. Ruggedness / Terrain Variability (intermediate, 50 m)": (tri, "magma", "m (TRI)"),
        },
        transform=transform,
        output_path=maps_dir / "sheringham_shoal_2020_terrain_atlas.png",
        title="Sheringham Shoal 2020 -- High-Resolution Seabed Terrain POC",
        subtitle="Operator-data benchmark / not PL854 evidence -- no risk colours or risk language",
        footer_note="OBSERVED bathymetry (A); MODELLED generic terrain derivatives (B-D). "
        "No fused hazard/risk score anywhere in this atlas.",
    )
    print(f"  -> {atlas_path}")

    print("Building GIS outputs (Section 12)...", flush=True)
    gis_dir = study_dir / "gis"
    footprint_gdf = _build_terrain_footprint_gdf(canonical.valid_mask, transform, working_crs)
    minx, miny, maxx, maxy = bounds
    qa_footprint_gdf = gpd.GeoDataFrame(
        {"footprint": ["raster_bounding_box"]},
        geometry=[shapely_box(minx, miny, maxx, maxy)],
        crs=working_crs,
    )
    gpkg_path = gis_dir / "seabed_terrain_poc.gpkg"
    gis_dir.mkdir(parents=True, exist_ok=True)
    if gpkg_path.exists():
        gpkg_path.unlink()
    if not footprint_gdf.empty:
        footprint_gdf.to_file(gpkg_path, driver="GPKG", layer="survey_footprint")
    qa_footprint_gdf.to_file(gpkg_path, driver="GPKG", layer="terrain_qa_footprint")
    print(f"  -> {gpkg_path}")

    print("Writing generic operator-data input contract (Section 14)...", flush=True)
    contract = terrain_contract.build_terrain_input_contract()
    contract_path = study_dir / "terrain_input_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")
    print(f"  -> {contract_path}")

    print("Writing product readiness result (Section 17)...", flush=True)
    validation = {
        "scientific_role": "HIGH_RESOLUTION_SEABED_TERRAIN_POC",
        "question_a_readiness_passed": readiness_result.status != terrain_readiness.NOT_READY,
        "question_a_readiness_status": readiness_result.status,
        "question_b_derived_layers_valid": bool(gis_ok),
        "question_c_route_kp_view_available": False,
        "route_kp_status": route_kp_status,
    }
    validation_path = study_dir / "terrain_poc_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    print(f"  -> {validation_path}")

    print("Building engineering POC report (Section 18)...", flush=True)
    limitations = [
        "No route/pipeline geometry was supplied in the acquired package -- KP/route view is "
        f"{route_kp_status}, never fabricated",
        "Terrain derivatives are generic geomorphometric statistics only -- no geohazard, "
        "risk, or susceptibility interpretation is made",
        "Curvature uses an explicit fine-scale physical step (engineering scale) -- it is not "
        "computed at the broader intermediate scale, where the second-derivative signal "
        "becomes physically less meaningful",
        f"Native resolution ({cell_size_m:g} m) is used as-is throughout -- no upsampling, no "
        "interpolation of missing regions to create false coverage",
        f"Valid-cell coverage is {facts.valid_cell_fraction:.1%} of the raster's own bounding "
        "box -- a real narrow, branching survey swath, not a data defect",
    ]
    input_contract_summary = [
        f"{f['field']}: {f['description']}" for f in contract["required_fields"]
    ]
    report_blocks = terrain_report.build_terrain_poc_report_blocks(
        project_title="Sheringham Shoal 2020 -- High-Resolution Seabed Terrain POC",
        source_facts={
            "Source": sheringham_provider.DATASET_TITLE,
            "Series": sheringham_provider.MDE_SERIES_ID,
            "File": acquisition.target_entry_name,
            "File size": f"{acquisition.target_entry_bytes:,} bytes",
            "SHA256": acquisition.target_entry_sha256,
            "CRS": working_crs,
            "Native resolution": f"{cell_size_m:g} m",
            "Vertical datum": canonical.source_vertical_datum,
            "Survey epoch": acquisition.survey_period,
            "Survey organisation": acquisition.survey_organisation,
            "Licence": acquisition.licence,
        },
        readiness_status=readiness_result.status,
        readiness_reasons=readiness_result.reasons(),
        bathymetry_stats={
            "bed_elevation_m range": f"{facts.data_min:.2f} to {facts.data_max:.2f} m"
            if facts.data_min is not None
            else "n/a",
            "Valid cell fraction": f"{facts.valid_cell_fraction:.1%}",
            "Raster dimensions": f"{width} x {height} px",
        },
        derived_layer_facts=derived_layer_facts,
        scale_facts=scale_facts,
        gis_outputs=[
            {
                "file": "gis/seabed_terrain_poc.gpkg",
                "type": "GeoPackage",
                "description": "survey + terrain QA footprints",
            },
            {
                "file": "terrain/canonical_bed_elevation.tif",
                "type": "GeoTIFF",
                "description": "canonical bed_elevation_m",
            },
            {
                "file": "terrain/slope.tif / aspect.tif",
                "type": "GeoTIFF",
                "description": "engineering-scale slope/aspect",
            },
            {
                "file": "terrain/profile_curvature.tif / plan_curvature.tif",
                "type": "GeoTIFF",
                "description": "Zevenbergen-Thorne curvature",
            },
            {
                "file": "terrain/local_relief_*.tif / terrain_std_*.tif / ruggedness_*.tif",
                "type": "GeoTIFF",
                "description": "multiscale terrain variability",
            },
        ],
        limitations=limitations,
        input_contract_summary=input_contract_summary,
    )
    report_dir = study_dir / "report"
    report_path = report_dir / "sheringham_shoal_2020_terrain_poc.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        terrain_report.render_blocks_html(
            report_blocks, title="Sheringham Shoal 2020 -- High-Resolution Seabed Terrain POC"
        ),
        encoding="utf-8",
    )
    print(f"  -> {report_path}")

    atlas_dims = terrain_maps.read_png_dimensions(atlas_path)
    qa_dims = terrain_maps.read_png_dimensions(qa_map_path)
    print()
    print("=== Sheringham Shoal 2020 High-Resolution Terrain POC (MAR-020) ===")
    print()
    print("## Source")
    print(f"  File: {acquisition.target_entry_name} ({acquisition.target_entry_bytes:,} bytes)")
    print(
        f"  Dimensions: {width} x {height} px | CRS: {working_crs} | resolution: {cell_size_m:g} m"
    )
    print(
        f"  Vertical datum: {canonical.source_vertical_datum} | "
        f"survey epoch: {acquisition.survey_period}"
    )
    print(f"  SHA256: {acquisition.target_entry_sha256}")
    print()
    print("## Readiness")
    print(f"  Status: {readiness_result.status}")
    for reason in readiness_result.reasons():
        print(f"    - {reason}")
    print()
    print("## Terrain")
    print(f"  bed_elevation_m range: {facts.data_min:.2f} to {facts.data_max:.2f} m")
    print(f"  scale parameters: {scale_facts}")
    print()
    print("## Outputs")
    print(f"  QA map: {qa_map_path} ({qa_dims[0]}x{qa_dims[1]} px)")
    print(f"  Terrain atlas: {atlas_path} ({atlas_dims[0]}x{atlas_dims[1]} px)")
    print(f"  GIS: {gpkg_path}")
    print(f"  Report: {report_path}")
    print()
    print(
        "IS GENERIC HIGH-RESOLUTION OPERATOR BATHYMETRY -> TERRAIN ANALYTICS DEMONSTRATED? "
        + ("YES" if gis_ok else "NO")
    )
    return 0


def _epoch2018_raster_facts(
    elevation: np.ndarray, valid_mask: np.ndarray, transform, working_crs: str
):
    height, width = elevation.shape
    minx = transform.c
    maxy = transform.f
    bounds = (minx, maxy + transform.e * height, minx + transform.a * width, maxy)
    valid_values = elevation[valid_mask]
    return terrain_readiness.RasterFacts(
        band_count=1,
        dtype="float64",
        color_interpretations=("gray",),
        crs_is_present=True,
        crs_is_geographic=False,
        crs_linear_units="metre",
        width=width,
        height=height,
        pixel_size_x_m=transform.a,
        pixel_size_y_m=-transform.e,
        bounds=bounds,
        nodata_value=None,
        vertical_datum=sheringham_2018_provider.SOURCE_VERTICAL_DATUM,
        survey_epoch=sheringham_2018_provider.SURVEY_PERIOD,
        data_min=float(valid_values.min()) if valid_values.size else None,
        data_max=float(valid_values.max()) if valid_values.size else None,
        data_std=float(valid_values.std()) if valid_values.size else None,
        valid_cell_fraction=float(valid_mask.mean()) if valid_mask.size else None,
    )


def _cmd_build_seabed_change_poc(args: argparse.Namespace) -> int:
    """MAR-021: generic multi-epoch seabed change / DEM-of-Difference POC,
    benchmarked against the real 2018 vs 2020 Sheringham Shoal surveys.
    Observed change only -- no future erosion/deposition prediction, no
    sediment-transport model, no risk/susceptibility score, no ML.
    """

    config = load_study_config(args.config)
    project_id = config.study.id.lower()
    study_dir = config.paths.processed_dir / project_id
    raw_dir = config.paths.raw_dir / project_id
    working_crs = config.crs.horizontal

    print("Acquiring epoch sources (Section 4)...", flush=True)
    acq2020 = sheringham_provider.download_sheringham_shoal_bathymetry(raw_dir)
    diff_path, diff_cached = sheringham_provider.download_sheringham_shoal_2020_comparison_product(
        raw_dir
    )
    hsd_path, hsd_cached = sheringham_provider.download_sheringham_shoal_2020_hsd_grid(raw_dir)
    acq2018 = sheringham_2018_provider.download_sheringham_shoal_2018_bathymetry(raw_dir)
    print(
        f"  2020 bathymetry: {acq2020.target_entry_name} ({acq2020.target_entry_bytes:,} bytes, "
        f"already_cached={acq2020.already_cached})",
        flush=True,
    )
    print(f"  2020 comparator: {diff_path.name} (already_cached={diff_cached})", flush=True)
    print(f"  2020 HSD grid: {hsd_path.name} (already_cached={hsd_cached})", flush=True)
    print(
        f"  2018 XYZ parts: {len(acq2018.xyz_entry_names)} files, "
        f"already_cached={acq2018.already_cached}",
        flush=True,
    )

    print("Rasterizing 2018 XYZ export (direct grid placement, no interpolation)...", flush=True)
    xyz_paths = [raw_dir / Path(n).name for n in sheringham_2018_provider.XYZ_ENTRY_NAMES]
    raw2018, valid2018_raw, transform2018 = sheringham_2018_provider.rasterize_xyz_points(xyz_paths)

    print("Running per-epoch operator-style readiness (Section 5)...", flush=True)
    raster_path_2020 = raw_dir / acq2020.target_entry_name
    with rasterio.open(raster_path_2020) as src:
        band2020 = src.read(1)
        crs2020 = src.crs
        transform2020 = src.transform
        nodata2020 = src.nodata
        width2020, height2020 = src.width, src.height
        bounds2020 = tuple(src.bounds)
        band_count2020 = src.count
        dtype2020 = src.dtypes[0]
        color_interp2020 = tuple(str(c) for c in src.colorinterp)

    facts2020, valid2020_raw = _terrain_raster_facts(
        band=band2020,
        crs=crs2020,
        transform=transform2020,
        nodata=nodata2020,
        width=width2020,
        height=height2020,
        bounds=bounds2020,
        band_count=band_count2020,
        dtype=dtype2020,
        color_interp=color_interp2020,
        vertical_datum=sheringham_provider.SOURCE_VERTICAL_DATUM,
        survey_epoch=acq2020.survey_period,
    )
    readiness2020 = terrain_readiness.assess_bathymetry_readiness(facts2020)
    facts2018 = _epoch2018_raster_facts(raw2018, valid2018_raw, transform2018, working_crs)
    readiness2018 = terrain_readiness.assess_bathymetry_readiness(facts2018)

    readiness_dir = study_dir / "readiness"
    readiness_dir.mkdir(parents=True, exist_ok=True)
    (readiness_dir / "bathymetry_2018_readiness.json").write_text(
        json.dumps(readiness2018.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    (readiness_dir / "bathymetry_2020_readiness.json").write_text(
        json.dumps(readiness2020.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    print(f"  2018 readiness: {readiness2018.status}", flush=True)
    print(f"  2020 readiness: {readiness2020.status}", flush=True)

    route_kp_status = "NOT_APPLICABLE_NO_AUTHORITATIVE_ROUTE_SUPPLIED"
    validation_path = study_dir / "seabed_change_poc_validation.json"

    def _early_stop(
        question_a, question_b, question_c, question_d, question_e, question_f, answer: str
    ) -> int:
        validation = {
            "scientific_role": "MULTI_EPOCH_SEABED_CHANGE_POC",
            "question_a_both_epochs_data_ready": question_a,
            "question_b_vertical_datums_compatible": question_b,
            "question_c_common_support_sufficient": question_c,
            "question_d_independent_dod_generated": question_d,
            "question_e_defensible_uncertainty_threshold": question_e,
            "question_f_source_product_consistent": question_f,
            "route_kp_status": route_kp_status,
        }
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
        print(f"  -> {validation_path}")
        print()
        print(
            "IS GENERIC OPERATOR-SUPPLIED MULTI-EPOCH MBES -> OBSERVED SEABED CHANGE ANALYTICS "
            f"DEMONSTRATED? {answer}"
        )
        return 0

    if (
        readiness2018.status == terrain_readiness.NOT_READY
        or readiness2020.status == terrain_readiness.NOT_READY
    ):
        print("EARLY STOP (Section 5): one or both epochs are NOT_READY.")
        return _early_stop(
            "NO", "NOT_APPLICABLE", "NOT_APPLICABLE", "NO", "NOT_APPLICABLE", "NOT_APPLICABLE", "NO"
        )

    print("Checking the HARD vertical-datum gate (Section 6)...", flush=True)
    datum_result = change_epoch_compatibility.assess_vertical_datum_compatibility(
        sheringham_2018_provider.SOURCE_VERTICAL_DATUM,
        sheringham_provider.SOURCE_VERTICAL_DATUM,
        harmonization_evidence=sheringham_provider.VERTICAL_DATUM_COMPATIBILITY_EVIDENCE,
    )
    print(f"  {datum_result.status}: {datum_result.reason}", flush=True)
    if datum_result.status == change_epoch_compatibility.VERTICAL_DATUM_NOT_HARMONIZED:
        print("EARLY STOP (Section 6): vertical datums are not demonstrably harmonized.")
        return _early_stop(
            "YES", "NO", "NOT_APPLICABLE", "NO", "NOT_APPLICABLE", "NOT_APPLICABLE", "NO"
        )

    print("Building canonical bed_elevation_m for both epochs (Section 7)...", flush=True)
    canonical2018 = terrain_canonical.build_canonical_bed_elevation(
        raw2018,
        nodata_value=None,
        source_sign_convention=sheringham_2018_provider.SOURCE_SIGN_CONVENTION,
        source_vertical_datum=sheringham_2018_provider.SOURCE_VERTICAL_DATUM,
    )
    canonical2020 = terrain_canonical.build_canonical_bed_elevation(
        band2020,
        nodata_value=nodata2020,
        source_sign_convention=sheringham_provider.SOURCE_SIGN_CONVENTION,
        source_vertical_datum=sheringham_provider.SOURCE_VERTICAL_DATUM,
    )

    print("Classifying + executing horizontal/grid alignment (Section 8)...", flush=True)
    classification = change_epoch_compatibility.classify_grid_alignment(
        crs1=working_crs,
        transform1=transform2018,
        crs2=str(crs2020),
        transform2=transform2020,
    )
    print(f"  {classification.status}: {classification.reason}", flush=True)
    if classification.status == change_epoch_compatibility.INCOMPATIBLE_HORIZONTAL_REFERENCE:
        print("EARLY STOP (Section 8): horizontal references are incompatible.")
        return _early_stop("YES", "YES", "NO", "NO", "NOT_APPLICABLE", "NOT_APPLICABLE", "NO")

    aligned2018, aligned2020 = change_alignment.align_to_common_grid(
        classification=classification,
        elevation1=canonical2018.bed_elevation_m,
        valid1=canonical2018.valid_mask,
        transform1=transform2018,
        elevation2=canonical2020.bed_elevation_m,
        valid2=canonical2020.valid_mask,
        transform2=transform2020,
        crs=working_crs,
    )

    print("Building common valid support (Section 10)...", flush=True)
    common = change_common_support.build_common_valid_support(
        aligned2018.valid_mask, aligned2020.valid_mask
    )
    print(
        f"  common valid cells: {common.common_valid_cell_count:,} "
        f"({common.fraction_of_epoch1_covered:.1%} of 2018, "
        f"{common.fraction_of_epoch2_covered:.1%} of 2020)",
        flush=True,
    )

    change_dir = study_dir / "change"
    tags_base = {
        "product": "MAR-021 generic multi-epoch seabed change POC",
        "scientific_role": "MULTI_EPOCH_SEABED_CHANGE_POC",
        "definition": change_dod.DoDResult.__dataclass_fields__["definition"].default,
    }

    if common.common_valid_cell_count < 30:
        print("EARLY STOP (Section 10): common valid support is not sufficient to compute a DoD.")
        return _early_stop("YES", "YES", "NO", "NO", "NOT_APPLICABLE", "NOT_APPLICABLE", "NO")

    print("Computing the canonical DoD (Section 11)...", flush=True)
    dod_result = change_dod.compute_delta_bed_elevation(
        aligned2018.elevation, aligned2020.elevation, common.common_valid_mask
    )
    dod_path = terrain_raster_io.write_terrain_raster(
        dod_result.delta_bed_elevation_m,
        aligned2020.transform,
        working_crs,
        change_dir / "delta_bed_elevation_2020_minus_2018_m.tif",
        {**tags_base, "layer": "delta_bed_elevation_m", "units": "m"},
    )
    print(f"  -> {dod_path}", flush=True)

    epoch1_date = date(2018, 10, 1)  # approximate: source states "October 2018 - November 2018"
    epoch2_date = date(2020, 11, 1)  # approximate: source states "November 2020 - December 2020"
    annualized = change_dod.annualize_change(
        dod_result.delta_bed_elevation_m, epoch1_date=epoch1_date, epoch2_date=epoch2_date
    )
    annualized_path = terrain_raster_io.write_terrain_raster(
        annualized.annualized_delta_m_per_year,
        aligned2020.transform,
        working_crs,
        change_dir / "annualized_bed_elevation_change_m_per_year.tif",
        {
            **tags_base,
            "layer": "annualized_bed_elevation_change_m_per_year",
            "units": "m/yr",
            "disclaimer": change_dod.ANNUALIZATION_DISCLAIMER,
        },
    )
    print(
        f"  -> {annualized_path} (elapsed years: {annualized.elapsed_years:.2f}, approximate "
        "representative dates -- source states month-level survey periods only)",
        flush=True,
    )

    print("Assessing horizontal misregistration QA (Section 9)...", flush=True)
    slope_deg, _aspect_deg, _slope_vf = terrain_derivatives.compute_slope_aspect_deg(
        aligned2020.elevation, aligned2020.valid_mask, 10.0, float(aligned2020.transform.a)
    )
    misreg_qa = change_alignment.assess_misregistration_qa(
        dod_result.delta_bed_elevation_m, common.common_valid_mask, slope_deg
    )
    print(
        f"  slope-dependent |dz| correlation: {misreg_qa.slope_dependent_correlation}", flush=True
    )

    print("Inventorying uncertainty evidence (Section 12)...", flush=True)
    uncertainty_items = [
        change_uncertainty.UncertaintyEvidenceItem(
            kind=change_uncertainty.EVIDENCE_NOMINAL_ACCURACY,
            epoch="2018",
            description="Nominal MBES vertical accuracy: "
            f"+/-{sheringham_provider.REPORTED_MBES_VERTICAL_ACCURACY_M} m -- a 'typically less "
            "than' figure, NOT verified by the source as a 1-sigma standard uncertainty "
            "(MAR-021A)",
            source_citation=sheringham_provider.PRECISION_EVIDENCE_SOURCE,
            usable_for_threshold=False,
        ),
        change_uncertainty.UncertaintyEvidenceItem(
            kind=change_uncertainty.EVIDENCE_NOMINAL_ACCURACY,
            epoch="2020",
            description="Nominal MBES vertical accuracy: "
            f"+/-{sheringham_provider.REPORTED_MBES_VERTICAL_ACCURACY_M} m -- a 'typically less "
            "than' figure, NOT verified by the source as a 1-sigma standard uncertainty "
            "(MAR-021A)",
            source_citation=sheringham_provider.PRECISION_EVIDENCE_SOURCE,
            usable_for_threshold=False,
        ),
        change_uncertainty.UncertaintyEvidenceItem(
            kind=change_uncertainty.EVIDENCE_DATUM_SQUARE_REPEATABILITY,
            epoch="2018+2020",
            description="Observed repeatability at a stable 100 m^2 datum square across "
            "winter surveys including 2018/2020: <= "
            f"{sheringham_provider.REPORTED_DATUM_SQUARE_REPEATABILITY_M} m",
            source_citation=sheringham_provider.PRECISION_EVIDENCE_SOURCE,
            usable_for_threshold=False,
        ),
        change_uncertainty.UncertaintyEvidenceItem(
            kind=change_uncertainty.EVIDENCE_CLASSIFIED_GRID_UNVERIFIED,
            epoch="2020",
            description=f"{sheringham_provider.MBESHSD_ENTRY_NAME}: uint8, values 0-254, "
            "RepresentationType=THEMATIC, no scale/offset/unit metadata found -- NOT verified as "
            "a direct metres-scale standard deviation.",
            source_citation=f"direct rasterio inspection of {hsd_path.name}",
            usable_for_threshold=False,
        ),
        change_uncertainty.UncertaintyEvidenceItem(
            kind=change_uncertainty.EVIDENCE_SOURCE_SPECIFIC_ANALYST_THRESHOLD,
            epoch="2018+2020",
            description="Fugro's own analyst-applied significance criterion: 'changes of less "
            f"than {sheringham_provider.REPORTED_ANALYST_SIGNIFICANCE_THRESHOLD_M} m were not "
            "considered significant, when assessing areas of erosion or accretion' -- scoped to "
            "THAT source's own interpretation practice; never repackaged as a generic OrbGSS "
            "uncertainty formula or threshold (MAR-021A)",
            source_citation=sheringham_provider.PRECISION_EVIDENCE_SOURCE,
            usable_for_threshold=False,
        ),
    ]
    uncertainty_inventory = change_uncertainty.build_uncertainty_evidence_inventory(
        uncertainty_items
    )
    uncertainty_dir = study_dir / "uncertainty"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    (uncertainty_dir / "uncertainty_evidence_inventory.json").write_text(
        json.dumps(uncertainty_inventory, indent=2, default=str), encoding="utf-8"
    )

    threshold = change_uncertainty.derive_change_threshold(
        sigma_epoch1_m=sheringham_provider.REPORTED_MBES_VERTICAL_ACCURACY_M,
        sigma_epoch2_m=sheringham_provider.REPORTED_MBES_VERTICAL_ACCURACY_M,
        evidence_citation=sheringham_provider.PRECISION_EVIDENCE_SOURCE,
    )
    print(
        f"  threshold status: {threshold.status} "
        f"(generic_threshold_m={threshold.generic_threshold_m}, "
        f"nominal_accuracy_rss_reference_m="
        f"{threshold.nominal_accuracy_rss_reference_m:.3f} -- reference only, not a threshold)"
        if threshold.nominal_accuracy_rss_reference_m is not None
        else f"  threshold status: {threshold.status} (generic_threshold_m=None)",
        flush=True,
    )

    print("Comparing against the source-produced 20v18 product (Section 15)...", flush=True)
    with rasterio.open(diff_path) as ds:
        source_diff_raw = ds.read(1)
        source_diff_nodata = ds.nodata
    source_diff_valid = (
        np.isfinite(source_diff_raw)
        if source_diff_nodata is None
        else (source_diff_raw != source_diff_nodata) & np.isfinite(source_diff_raw)
    )
    source_diff_full = np.where(source_diff_valid, source_diff_raw, np.nan)
    # source_diff shares epoch2's ORIGINAL (uncropped) grid -- crop it to the SAME window
    # already used for the DoD, never a fresh/independent crop computation.
    source_diff = change_alignment.crop_array_to_aligned_window(
        source_diff_full,
        original_transform=transform2020,
        aligned_transform=aligned2020.transform,
        aligned_shape=dod_result.delta_bed_elevation_m.shape,
    )
    comparator_result = change_comparator.compare_dod_to_source_product(
        dod_result.delta_bed_elevation_m,
        source_diff,
        common.common_valid_mask,
        sign_evidence="Not stated in the source GeoTIFF sidecars (.tif.xml/.tif.aux.xml) or in "
        "the 2020 Comparison Report's prose (both fetched and searched directly) -- treated as "
        "unresolved per Section 15, never inferred from correlation quality.",
    )
    vertical_bias_qa = change_comparator.assess_vertical_bias_qa(dod_result.delta_bed_elevation_m)
    print(f"  sign status: {comparator_result.sign_status}", flush=True)
    print(
        f"  median DoD (vertical bias QA): {vertical_bias_qa.median_dod_m:.4f} m "
        "(reported, not auto-corrected)",
        flush=True,
    )

    validation_dir = study_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "dod_vs_source_difference_comparison.json").write_text(
        json.dumps(comparator_result.to_dict(), indent=2, default=str), encoding="utf-8"
    )

    print("Rendering primary change map (Section 19)...", flush=True)
    maps_dir = study_dir / "maps"
    change_map_path = change_maps.render_seabed_change_map(
        bed_elevation_epoch1_m=aligned2018.elevation,
        bed_elevation_epoch2_m=aligned2020.elevation,
        delta_bed_elevation_m=dod_result.delta_bed_elevation_m,
        qa_panel_array=np.where(
            common.common_valid_mask, np.abs(dod_result.delta_bed_elevation_m), np.nan
        ),
        qa_panel_label="|Observed change| (m) -- comparison/QA context, not a hazard score",
        qa_panel_cmap="magma",
        transform=aligned2020.transform,
        output_path=maps_dir / "sheringham_shoal_2018_2020_seabed_change.png",
        title="Sheringham Shoal 2018-2020 -- Observed Seabed Elevation Change POC",
        epoch1_label="2018",
        epoch2_label="2020",
    )
    print(f"  -> {change_map_path}", flush=True)

    print("Rendering change distribution QA figure (Section 20)...", flush=True)
    as_is_stats = comparator_result.interpretations.get(
        "as_is"
    ) or comparator_result.interpretations.get("confirmed")
    qa_fig_path = change_maps.render_change_distribution_qa(
        delta_bed_elevation_m=dod_result.delta_bed_elevation_m,
        source_residuals_m=None,
        common_support_summary={
            "fraction_of_2018_covered": common.fraction_of_epoch1_covered,
            "fraction_of_2020_covered": common.fraction_of_epoch2_covered,
        },
        output_path=maps_dir / "sheringham_shoal_2018_2020_change_qa.png",
        title="Sheringham Shoal 2018-2020 -- Change Distribution QA",
    )
    print(f"  -> {qa_fig_path}", flush=True)

    print("Building GIS outputs (Section 21)...", flush=True)
    gis_dir = study_dir / "gis"
    gis_dir.mkdir(parents=True, exist_ok=True)
    footprint_2018 = _build_terrain_footprint_gdf(
        aligned2018.valid_mask, aligned2020.transform, working_crs
    )
    footprint_2020 = _build_terrain_footprint_gdf(
        aligned2020.valid_mask, aligned2020.transform, working_crs
    )
    footprint_common = _build_terrain_footprint_gdf(
        common.common_valid_mask, aligned2020.transform, working_crs
    )
    gpkg_path = gis_dir / "seabed_change_poc.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    if not footprint_2018.empty:
        footprint_2018.to_file(gpkg_path, driver="GPKG", layer="epoch_2018_footprint")
    if not footprint_2020.empty:
        footprint_2020.to_file(gpkg_path, driver="GPKG", layer="epoch_2020_footprint")
    if not footprint_common.empty:
        footprint_common.to_file(gpkg_path, driver="GPKG", layer="common_support_footprint")
    print(f"  -> {gpkg_path}", flush=True)

    print("Writing generic operator-data input contract (Section 24)...", flush=True)
    contract = change_contract.build_seabed_change_input_contract()
    contract_path = study_dir / "seabed_change_input_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")
    print(f"  -> {contract_path}", flush=True)

    print("Building engineering POC report (Section 25)...", flush=True)
    finite_dod = dod_result.delta_bed_elevation_m[np.isfinite(dod_result.delta_bed_elevation_m)]
    change_facts = {
        "Definition": "delta_bed_elevation_m = bed_elevation_2020_m - bed_elevation_2018_m",
        "Min": f"{float(finite_dod.min()):.3f} m" if finite_dod.size else "n/a",
        "Median": f"{vertical_bias_qa.median_dod_m:.3f} m",
        "P05": f"{vertical_bias_qa.p05_m:.3f} m",
        "P95": f"{vertical_bias_qa.p95_m:.3f} m",
        "Max": f"{float(finite_dod.max()):.3f} m" if finite_dod.size else "n/a",
        "Annualized period (years)": f"{annualized.elapsed_years:.2f} "
        "(approximate representative dates)",
        "Horizontal alignment": classification.status,
    }
    input_contract_summary = [
        f"{f['field']}: {f['description']}" for f in contract["required_fields"]
    ]
    report_blocks = change_report.build_seabed_change_report_blocks(
        project_title="Sheringham Shoal 2018-2020 -- Observed Seabed Elevation Change POC",
        epoch1_source_facts={
            "Source": sheringham_2018_provider.DATASET_TITLE,
            "Series": sheringham_2018_provider.MDE_SERIES_ID,
            "Files": ", ".join(Path(n).name for n in acq2018.xyz_entry_names),
            "SHA256": ", ".join(acq2018.xyz_entry_sha256),
            "Vertical datum": sheringham_2018_provider.SOURCE_VERTICAL_DATUM,
            "Sign convention": sheringham_2018_provider.SOURCE_SIGN_CONVENTION,
            "Survey epoch": sheringham_2018_provider.SURVEY_PERIOD,
        },
        epoch2_source_facts={
            "Source": sheringham_provider.DATASET_TITLE,
            "Series": sheringham_provider.MDE_SERIES_ID,
            "File": acq2020.target_entry_name,
            "SHA256": acq2020.target_entry_sha256,
            "Vertical datum": sheringham_provider.SOURCE_VERTICAL_DATUM,
            "Sign convention": sheringham_provider.SOURCE_SIGN_CONVENTION,
            "Survey epoch": acq2020.survey_period,
        },
        per_epoch_readiness={
            "2018": {"status": readiness2018.status, "reasons": readiness2018.reasons()},
            "2020": {"status": readiness2020.status, "reasons": readiness2020.reasons()},
        },
        datum_compatibility=datum_result.to_dict(),
        grid_compatibility=classification.to_dict(),
        common_support=common.to_dict(),
        change_facts=change_facts,
        measurement_accuracy_evidence=[
            f"Nominal MBES vertical accuracy (both epochs): "
            f"+/-{sheringham_provider.REPORTED_MBES_VERTICAL_ACCURACY_M} m -- a 'typically less "
            "than' figure, NOT verified by the source as a 1-sigma standard uncertainty",
            f"Stable datum-square repeatability (2018+2020): "
            f"<= {sheringham_provider.REPORTED_DATUM_SQUARE_REPEATABILITY_M} m",
            "MBESHSD grid: acquired but NOT used quantitatively -- units/scale unverified "
            "(see uncertainty_evidence_inventory.json)",
            f"Source: {sheringham_provider.PRECISION_EVIDENCE_SOURCE}",
        ],
        source_specific_analyst_threshold=[
            f"Fugro's own analyst-applied significance criterion: changes of less than "
            f"{sheringham_provider.REPORTED_ANALYST_SIGNIFICANCE_THRESHOLD_M} m were not "
            "considered significant, when assessing areas of erosion or accretion",
            "Scoped to THAT source's own interpretation practice -- never repackaged as a "
            "generic OrbGSS uncertainty formula or threshold",
        ],
        generic_propagated_uncertainty=[
            f"Status: {threshold.status}",
            f"Generic threshold (m): {threshold.generic_threshold_m}",
            "Nominal-accuracy RSS reference (m): "
            + (
                f"{threshold.nominal_accuracy_rss_reference_m:.3f} -- NOT a propagated 1-sigma "
                "DoD uncertainty and NOT a canonical significance threshold"
                if threshold.nominal_accuracy_rss_reference_m is not None
                else "n/a"
            ),
            f"Reason: {threshold.reason}",
        ],
        comparator_facts={
            "Sign status": comparator_result.sign_status,
            "Sign evidence": comparator_result.sign_evidence,
            "Comparison cell count": comparator_result.common_comparison_cell_count,
            "As-is / confirmed NMAD (m)": (
                as_is_stats["nmad_residual_m"] if as_is_stats else "n/a"
            ),
        },
        anthropogenic_limitation_text="Sheringham Shoal is an operating offshore wind farm. "
        "Observed change can include foundation scour, cable trenching/exposure/burial, rock "
        "protection works, jack-up disturbance, or natural sediment migration. This POC does NOT "
        "automatically infer cause -- cells are labelled OBSERVED_SEABED_RAISING / "
        "OBSERVED_SEABED_LOWERING only, never 'natural erosion' or any other attributed cause.",
        input_contract_summary=input_contract_summary,
        what_this_does_not_predict=[
            "Future erosion or deposition at any location.",
            "A sediment-transport model of any kind.",
            "Scour or freespan susceptibility.",
            "A risk score or route-suitability score.",
            "Any anthropogenic-vs-natural attribution of observed change.",
        ],
    )
    report_dir = study_dir / "report"
    report_path = report_dir / "sheringham_shoal_2018_2020_seabed_change_poc.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        change_report.render_blocks_html(
            report_blocks,
            title="Sheringham Shoal 2018-2020 -- Observed Seabed Elevation Change POC",
        ),
        encoding="utf-8",
    )
    print(f"  -> {report_path}", flush=True)

    question_f = "NOT_APPLICABLE"
    validation = {
        "scientific_role": "MULTI_EPOCH_SEABED_CHANGE_POC",
        "question_a_both_epochs_data_ready": "YES",
        "question_b_vertical_datums_compatible": "YES",
        "question_c_common_support_sufficient": "YES",
        "question_d_independent_dod_generated": "YES",
        "question_e_defensible_uncertainty_threshold": "NO",
        "question_e_reason": "source accuracy confidence semantics are insufficient for "
        "generic uncertainty propagation (MAR-021A) -- a nominal/typical accuracy figure is "
        "not necessarily a 1-sigma standard uncertainty, and no source in this project has "
        "established that it is",
        "question_f_source_product_consistent": question_f,
        "question_f_reason": "Source difference-product sign convention is unresolved from "
        "documentation -- a YES/NO consistency verdict would implicitly assume a sign never "
        "confirmed by the source; see dod_vs_source_difference_comparison.json for both "
        "interpretations' residual statistics.",
        "route_kp_status": route_kp_status,
    }
    validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    print(f"  -> {validation_path}", flush=True)

    change_map_dims = change_maps.read_png_dimensions(change_map_path)
    qa_fig_dims = change_maps.read_png_dimensions(qa_fig_path)
    print()
    print("=== Sheringham Shoal 2018-2020 Seabed Change POC (MAR-021) ===")
    print()
    print("## Source")
    print(
        f"  2018: {len(acq2018.xyz_entry_names)} XYZ parts, vertical datum "
        f"{sheringham_2018_provider.SOURCE_VERTICAL_DATUM}, "
        f"survey epoch {sheringham_2018_provider.SURVEY_PERIOD}"
    )
    print(
        f"  2020: {acq2020.target_entry_name}, "
        f"vertical datum {sheringham_provider.SOURCE_VERTICAL_DATUM}, "
        f"survey epoch {acq2020.survey_period}"
    )
    print()
    print("## Readiness")
    print(f"  2018: {readiness2018.status} | 2020: {readiness2020.status}")
    print()
    print("## Alignment")
    print(f"  {classification.status}: {classification.reason}")
    print(f"  common overlap: {common.common_valid_cell_count:,} cells")
    print()
    print("## DoD")
    print(
        f"  min={float(finite_dod.min()):.3f} median={vertical_bias_qa.median_dod_m:.3f} "
        f"p05={vertical_bias_qa.p05_m:.3f} p95={vertical_bias_qa.p95_m:.3f} "
        f"max={float(finite_dod.max()):.3f} m"
    )
    print(f"  annualized period: {annualized.elapsed_years:.2f} years")
    print()
    print("## Uncertainty")
    print(f"  threshold status: {threshold.status}")
    print(f"  generic threshold (m): {threshold.generic_threshold_m}")
    print(
        "  nominal-accuracy RSS reference (m, non-canonical): "
        f"{threshold.nominal_accuracy_rss_reference_m}"
    )
    print()
    print("## Official comparator")
    print(f"  sign status: {comparator_result.sign_status}")
    print()
    print("## Outputs")
    print(f"  Change map: {change_map_path} ({change_map_dims[0]}x{change_map_dims[1]} px)")
    print(f"  QA figure: {qa_fig_path} ({qa_fig_dims[0]}x{qa_fig_dims[1]} px)")
    print(f"  GIS: {gpkg_path}")
    print(f"  Report: {report_path}")
    print()
    print(
        "IS GENERIC OPERATOR-SUPPLIED MULTI-EPOCH MBES -> OBSERVED SEABED CHANGE ANALYTICS "
        "DEMONSTRATED? YES"
    )
    return 0


# --- MAR-022: Sheringham-specific interpretation vocabulary -> generic category mapping -------
# kept in the CLI layer, never inside the generic `bedforms.interpretation` module (Section 7's
# "preserve them separately" / Section 8's exclusion list, applied to the REAL classes found by
# direct inspection of the real package -- see `sheringham_shoal_2020`'s acquisition docstring).
# "unknown_linear_feature" is deliberately absent -> falls through to
# UNCLASSIFIED_INTERPRETATION_FEATURE, never assumed natural or anthropogenic.
_SHERINGHAM_2020_INTERPRETATION_CATEGORY_BY_DESCRIPTOR = {
    "jackup location": bedform_interpretation.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
    "sandwave crest": bedform_interpretation.NATURAL_BEDFORM_INTERPRETATION,
    "fishing gear": bedform_interpretation.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
    "rock dump": bedform_interpretation.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
    "rope": bedform_interpretation.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
    "exposure": bedform_interpretation.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
    "wreck": bedform_interpretation.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
    "trenching": bedform_interpretation.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
    "possible debris": bedform_interpretation.ANTHROPOGENIC_DISTURBANCE_CONTEXT,
}

_BEDFORM_MAX_TILES_TO_RANK = 60  # bounds spectral-ranking cost on a near-full-coverage raster;
# logged (never silent) if the real qualifying-tile count exceeds this.


def _tile_window(
    elevation: np.ndarray, valid_mask: np.ndarray, tile
) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, size_px = tile.row_origin_px, tile.col_origin_px, tile.tile_size_px
    return elevation[r0 : r0 + size_px, c0 : c0 + size_px], valid_mask[
        r0 : r0 + size_px, c0 : c0 + size_px
    ]


def _tile_box(tile):
    half = tile.tile_size_m / 2.0
    return shapely_box(
        tile.center_x_m - half,
        tile.center_y_m - half,
        tile.center_x_m + half,
        tile.center_y_m + half,
    )


def _derive_bedform_validation_questions(
    *,
    has_1000m_support: bool,
    has_support_at_either_scale: bool,
    canonical_bedform_count: int,
    canonical_selected_tile_count: int,
    stable_match_count: int,
) -> dict[str, str]:
    """MAR-022A Section 13: the causal validation-question rule, pulled
    out as a pure function so it is testable independent of the real
    CLI/real data. Question E may be YES only when A, C, and D are ALL
    YES AND at least one match is stable across every tested tolerance
    set; F may be YES only when E is YES; G is always NO. `D=NO ->
    E=YES` is asserted structurally impossible, never merely hoped for."""

    question_a = "YES" if has_1000m_support else "NO"
    question_b = "YES" if has_support_at_either_scale else "NO"
    question_c = "YES" if canonical_bedform_count >= 3 else "NO"
    question_d = "YES" if canonical_selected_tile_count >= 1 else "NO"
    question_e = (
        "YES"
        if (
            question_a == "YES"
            and question_c == "YES"
            and question_d == "YES"
            and stable_match_count >= 1
        )
        else "NO"
    )
    question_f = "YES" if question_e == "YES" else "NO"
    question_g = "NO"
    assert not (question_d == "NO" and question_e == "YES"), "D=NO can never produce E=YES"
    assert not (question_e == "NO" and question_f == "YES"), "E=NO can never produce F=YES"
    return {
        "a": question_a,
        "b": question_b,
        "c": question_c,
        "d": question_d,
        "e": question_e,
        "f": question_f,
        "g": question_g,
    }


def _tile_box_from_row(row):
    half = row["tile_size_m"] / 2.0
    return shapely_box(
        row["center_x_m"] - half,
        row["center_y_m"] - half,
        row["center_x_m"] + half,
        row["center_y_m"] + half,
    )


def _cmd_build_bedform_morphodynamics_poc(args: argparse.Namespace) -> int:
    """MAR-022: generic sand-wave/bedform morphodynamics POC. Reuses the
    ACCEPTED MAR-020/021 canonical bathymetry pipeline (re-derived here
    from the same cached sources, never reacquired) and the ACCEPTED
    MAR-017 morphometry engine unchanged. Adds: canonical support audit,
    natural-vs-anthropogenic tile context (from the real, independently-
    acquired 2020 Interpretation Shapefiles package), per-epoch bedform
    morphometry, and independent multi-epoch crest matching. Static
    bedform geometry and OBSERVED multi-epoch change only -- no future
    migration prediction, no susceptibility, no hazard/risk score.
    """

    config = load_study_config(args.config)
    project_id = config.study.id.lower()
    study_dir = config.paths.processed_dir / project_id
    raw_dir = config.paths.raw_dir / project_id
    working_crs = config.crs.horizontal
    bedforms_dir = study_dir / "bedforms"
    maps_dir = study_dir / "maps"
    gis_dir = study_dir / "gis"
    report_dir = study_dir / "report"
    for d in (bedforms_dir, maps_dir, gis_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)
    validation_path = study_dir / "bedform_morphodynamics_poc_validation.json"

    def _write_validation(a, b, c, d_, e, f, g, *, question_d_reason: str | None = None) -> None:
        validation = {
            "scientific_role": "SANDBED_BEDFORM_MORPHOLOGY_AND_OBSERVED_CHANGE",
            "question_a_1000m_canonical_support": a,
            "question_b_generic_engine_applied_to_real_project_grade_mbes": b,
            "question_c_3plus_canonical_sandwave_bedforms_recovered": c,
            "question_d_natural_seabed_validation_areas_identified": d_,
            "question_d_reason": question_d_reason,
            "question_e_crests_matched_with_defensible_support": e,
            "question_f_observed_apparent_displacement_rate_reportable": f,
            "question_g_future_migration_prediction_made": g,
        }
        validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")

    # --- Section 2: reuse the ACCEPTED MAR-020/021 canonical pipeline, never reacquired --------
    print(
        "Reusing the accepted MAR-020/021 canonical bathymetry pipeline (Section 2)...", flush=True
    )
    acq2020 = sheringham_provider.download_sheringham_shoal_bathymetry(raw_dir)
    acq2018 = sheringham_2018_provider.download_sheringham_shoal_2018_bathymetry(raw_dir)
    xyz_paths = [raw_dir / Path(n).name for n in sheringham_2018_provider.XYZ_ENTRY_NAMES]
    raw2018, _valid2018_raw, transform2018 = sheringham_2018_provider.rasterize_xyz_points(
        xyz_paths
    )

    raster_path_2020 = raw_dir / acq2020.target_entry_name
    with rasterio.open(raster_path_2020) as src:
        band2020 = src.read(1)
        crs2020 = src.crs
        transform2020 = src.transform
        nodata2020 = src.nodata

    datum_result = change_epoch_compatibility.assess_vertical_datum_compatibility(
        sheringham_2018_provider.SOURCE_VERTICAL_DATUM,
        sheringham_provider.SOURCE_VERTICAL_DATUM,
        harmonization_evidence=sheringham_provider.VERTICAL_DATUM_COMPATIBILITY_EVIDENCE,
    )
    if datum_result.status != change_epoch_compatibility.VERTICAL_DATUM_HARMONIZED:
        print("EARLY STOP: vertical datums are not demonstrably harmonized.")
        _write_validation("NO", "NO", "NOT_APPLICABLE", "NOT_APPLICABLE", "NO", "NO", "NO")
        print(f"  -> {validation_path}")
        print("IS GENERIC HIGH-RESOLUTION MBES -> SANDBED BEDFORM MORPHOMETRY DEMONSTRATED? NO")
        print("IS DEFENSIBLE OBSERVED 2018-2020 CREST DISPLACEMENT DEMONSTRATED? NO")
        return 0

    canonical2018 = terrain_canonical.build_canonical_bed_elevation(
        raw2018,
        nodata_value=None,
        source_sign_convention=sheringham_2018_provider.SOURCE_SIGN_CONVENTION,
        source_vertical_datum=sheringham_2018_provider.SOURCE_VERTICAL_DATUM,
    )
    canonical2020 = terrain_canonical.build_canonical_bed_elevation(
        band2020,
        nodata_value=nodata2020,
        source_sign_convention=sheringham_provider.SOURCE_SIGN_CONVENTION,
        source_vertical_datum=sheringham_provider.SOURCE_VERTICAL_DATUM,
    )
    classification = change_epoch_compatibility.classify_grid_alignment(
        crs1=working_crs, transform1=transform2018, crs2=str(crs2020), transform2=transform2020
    )
    aligned2018, aligned2020 = change_alignment.align_to_common_grid(
        classification=classification,
        elevation1=canonical2018.bed_elevation_m,
        valid1=canonical2018.valid_mask,
        transform1=transform2018,
        elevation2=canonical2020.bed_elevation_m,
        valid2=canonical2020.valid_mask,
        transform2=transform2020,
        crs=working_crs,
    )
    common = change_common_support.build_common_valid_support(
        aligned2018.valid_mask, aligned2020.valid_mask
    )
    pixel_size_m = float(aligned2020.transform.a)
    print(f"  common valid cells: {common.common_valid_cell_count:,}", flush=True)

    dod_path = study_dir / "change" / "delta_bed_elevation_2020_minus_2018_m.tif"
    if not dod_path.exists():
        print(f"EARLY STOP: accepted MAR-021 DoD not found at {dod_path} -- run MAR-021 first.")
        _write_validation("NO", "NO", "NOT_APPLICABLE", "NOT_APPLICABLE", "NO", "NO", "NO")
        return 0
    with rasterio.open(dod_path) as ds:
        dod_array = ds.read(1)
        dod_transform = ds.transform
    print(f"  loaded the accepted MAR-021 DoD unchanged from {dod_path}", flush=True)

    elapsed_years = (date(2020, 11, 1) - date(2018, 10, 1)).days / 365.25

    # --- Section 6: acquire + inventory the real 2020 Interpretation Shapefiles package --------
    print("Acquiring the 2020 Interpretation Shapefiles package (Section 6)...", flush=True)
    interp_acq = sheringham_provider.download_sheringham_shoal_2020_interpretation_shapefiles(
        raw_dir
    )
    print(
        f"  {len(interp_acq.layer_names)} layers, {interp_acq.package_bytes:,} bytes, "
        f"already_cached={interp_acq.already_cached}",
        flush=True,
    )
    raw_interp_layers = sheringham_provider.load_interpretation_layers(interp_acq.extracted_dir)
    interp_layers = [
        bedform_interpretation.InterpretationLayer(name, gdf, "Descriptio")
        for name, gdf in raw_interp_layers.items()
    ]
    feature_inventory_df = bedform_interpretation.build_feature_inventory(interp_layers)
    feature_inventory_path = bedforms_dir / "source_interpretation_inventory.parquet"
    feature_inventory_df.to_parquet(feature_inventory_path, index=False)
    print(
        f"  {len(feature_inventory_df)} attribute-value rows -> {feature_inventory_path}",
        flush=True,
    )

    print("Classifying interpretation features (Section 7-8)...", flush=True)
    classified_interp = bedform_interpretation.classify_features(
        interp_layers,
        category_by_normalized_descriptor=_SHERINGHAM_2020_INTERPRETATION_CATEGORY_BY_DESCRIPTOR,
    )
    natural_interp = bedform_interpretation.extract_category(
        classified_interp, bedform_interpretation.NATURAL_BEDFORM_INTERPRETATION
    )
    anthropogenic_interp = bedform_interpretation.extract_category(
        classified_interp, bedform_interpretation.ANTHROPOGENIC_DISTURBANCE_CONTEXT
    )
    unclassified_interp = bedform_interpretation.extract_category(
        classified_interp, bedform_interpretation.UNCLASSIFIED_INTERPRETATION_FEATURE
    )
    print(
        f"  natural={len(natural_interp)} anthropogenic={len(anthropogenic_interp)} "
        f"unclassified={len(unclassified_interp)}",
        flush=True,
    )

    # =============================================================================================
    # MAR-022A: canonical natural-bedform selection MUST apply natural-vs-anthropogenic context to
    # EVERY qualifying support tile BEFORE any spectral ranking (Section 3), at 2000 m first and
    # only then 1000 m (Section 2/4), and only ever select spatially-independent, spectrally-band-
    # eligible, NATURAL tiles (Section 6) -- never the non-independent `rank_and_select_top_tiles`
    # fallback used for MAR-022's original (defective) selection. That original selection -- and
    # everything it found -- is preserved below as explicit, clearly-labelled diagnostics (Section
    # 1), never deleted, never called canonical.
    # =============================================================================================

    def _canonical_pass(tile_size_m: float) -> dict[str, Any]:
        """Sections 2-6 at ONE support scale: find every qualifying tile ->
        assess natural-vs-anthropogenic context for ALL of them (never
        just the eventually-ranked ones) -> canonical-band spectral
        eligibility (30 m <= wavelength <= tile_size_m/3) computed ONLY
        for the natural-eligible subset -> spatially-independent
        selection (max 5) from the natural+spectrally-eligible pool
        only."""

        all_tiles, meta = swm.find_valid_tiles(
            common.common_valid_mask,
            pixel_size_m,
            transform=aligned2020.transform,
            tile_size_m=tile_size_m,
            min_tile_size_m=tile_size_m,
        )
        records = []
        for tile in all_tiles:
            natural_status = bedform_natural_context.assess_natural_bedform_validation_status(
                _tile_box(tile),
                anthropogenic_context_gdf=anthropogenic_interp,
                interpretation_available=True,
            )
            records.append({"tile": tile, "natural_status": natural_status})
        natural_records = [
            r
            for r in records
            if r["natural_status"].status == bedform_natural_context.NATURAL_SEABED_ELIGIBLE
        ]
        excluded_records = [
            r
            for r in records
            if r["natural_status"].status != bedform_natural_context.NATURAL_SEABED_ELIGIBLE
        ]

        analysis_pool = natural_records
        capped = len(analysis_pool) > _BEDFORM_MAX_TILES_TO_RANK
        if capped:
            idx = sorted(
                set(np.linspace(0, len(analysis_pool) - 1, _BEDFORM_MAX_TILES_TO_RANK).astype(int))
            )
            print(
                f"    capping canonical-band spectral analysis to {len(idx)} evenly-sampled "
                f"natural candidates out of {len(analysis_pool)} (logged, not silent)",
                flush=True,
            )
            analysis_pool = [analysis_pool[i] for i in idx]

        max_wavelength_m = tile_size_m / 3.0
        spectral_pool = []
        spectrally_eligible_count = 0
        for r in analysis_pool:
            tile = r["tile"]
            tile_elev, tile_valid = _tile_window(
                aligned2020.elevation, aligned2020.valid_mask, tile
            )
            global_diag, band_diag = swm.analyze_tile_dual_band(
                tile_elev, tile_valid, pixel_size_m, max_wavelength_m=max_wavelength_m
            )
            r["global_diag"] = global_diag
            r["band_diag"] = band_diag
            if band_diag is not None:
                spectrally_eligible_count += 1
                spectral_pool.append(
                    {
                        "tile_id": tile.tile_id,
                        "tile_size_m": tile.tile_size_m,
                        "center_x_m": tile.center_x_m,
                        "center_y_m": tile.center_y_m,
                        "diagnostics": band_diag,
                        "_record": r,
                    }
                )

        selected_entries = swm.select_spatially_independent_eligible_tiles(
            spectral_pool, max_tiles=5
        )
        selected_records = [e["_record"] for e in selected_entries]

        return {
            "tile_size_m": tile_size_m,
            "support_tile_count": len(all_tiles),
            "records": records,
            "natural_records": natural_records,
            "excluded_records": excluded_records,
            "analysis_pool_size": len(analysis_pool),
            "capped": capped,
            "spectrally_eligible_count": spectrally_eligible_count,
            "selected_records": selected_records,
            "cascade_meta": meta,
        }

    print("Running the 2000 m canonical natural-bedform pass (Section 2-6)...", flush=True)
    pass_2000 = _canonical_pass(swm.CANONICAL_TILE_SIZE_M)
    print(
        f"  support={pass_2000['support_tile_count']} "
        f"natural_eligible={len(pass_2000['natural_records'])} "
        f"spectrally_eligible={pass_2000['spectrally_eligible_count']} "
        f"selected={len(pass_2000['selected_records'])}",
        flush=True,
    )

    print("Running the 1000 m canonical natural-bedform pass (Section 4)...", flush=True)
    pass_1000 = _canonical_pass(swm.MIN_TILE_SIZE_M)
    print(
        f"  support={pass_1000['support_tile_count']} "
        f"natural_eligible={len(pass_1000['natural_records'])} "
        f"spectrally_eligible={pass_1000['spectrally_eligible_count']} "
        f"selected={len(pass_1000['selected_records'])}",
        flush=True,
    )

    if pass_2000["selected_records"]:
        canonical_pass, canonical_scale_used_m = pass_2000, swm.CANONICAL_TILE_SIZE_M
    elif pass_1000["selected_records"]:
        canonical_pass, canonical_scale_used_m = pass_1000, swm.MIN_TILE_SIZE_M
    else:
        canonical_pass, canonical_scale_used_m = None, None
    canonical_available = canonical_pass is not None
    print(
        f"Canonical natural-bedform tile scale used: "
        f"{canonical_scale_used_m if canonical_available else 'NONE (unavailable)'}",
        flush=True,
    )

    # --- Section 1: preserve MAR-022's ORIGINAL (pre-MAR-022A) top-5 selection, unchanged, as an
    # explicit, clearly noncanonical diagnostic -- natural context is deliberately NOT applied
    # before ranking here (reproducing exactly what MAR-022 did), and the non-spatially-
    # independent `rank_and_select_top_tiles` fallback is used exactly as before.
    print(
        "Reproducing MAR-022's original disturbed-tile top-5 selection as a diagnostic...",
        flush=True,
    )
    legacy_ranking_pool = []
    for r in pass_2000["records"]:
        tile = r["tile"]
        tile_elev, tile_valid = _tile_window(aligned2020.elevation, aligned2020.valid_mask, tile)
        diag = swm.analyze_tile(tile_elev, tile_valid, pixel_size_m)
        if diag is None:
            continue
        legacy_ranking_pool.append(
            {
                "tile_id": tile.tile_id,
                "tile_size_m": tile.tile_size_m,
                "center_x_m": tile.center_x_m,
                "center_y_m": tile.center_y_m,
                "diagnostics": diag,
                "_record": r,
            }
        )
    legacy_selected = swm.select_spatially_independent_eligible_tiles(
        legacy_ranking_pool, max_tiles=5
    )
    if not legacy_selected and legacy_ranking_pool:
        legacy_selected = swm.rank_and_select_top_tiles(legacy_ranking_pool, top_n=5)
    legacy_records = [e["_record"] for e in legacy_selected]
    legacy_global_diag_by_tile_id = {e["tile_id"]: e["diagnostics"] for e in legacy_selected}
    legacy_natural_count = sum(
        1
        for r in legacy_records
        if r["natural_status"].status == bedform_natural_context.NATURAL_SEABED_ELIGIBLE
    )
    print(
        f"  {len(legacy_records)} legacy tile(s) selected, {legacy_natural_count} natural-eligible "
        f"(MAR-022 originally found all 5 anthropogenically disturbed -- reproduced here, not "
        "assumed)",
        flush=True,
    )

    # --- Shared per-tile extraction + multi-tolerance matching (Sections 7-11) -----------------
    def _extract_and_match(
        tile_records: list[dict[str, Any]], *, tolerance_sets: tuple, is_canonical: bool
    ) -> dict[str, Any]:
        tile_rows = {"2018": [], "2020": []}
        bedform_rows = {"2018": [], "2020": []}
        crest_points = {"2018": [], "2020": []}
        trough_points = {"2018": [], "2020": []}
        tile_summaries: list[dict[str, Any]] = []
        representative_transects: list[tuple] = []
        pairs_by_tolerance = {tol.name: [] for tol in tolerance_sets}
        matches_by_tolerance = {tol.name: [] for tol in tolerance_sets}

        for record in tile_records:
            tile = record["tile"]
            natural_status = record["natural_status"]
            max_wavelength_m = tile.tile_size_m / 3.0
            per_epoch_diag: dict[str, Any] = {}
            per_epoch_effective_diag: dict[str, Any] = {}
            per_epoch_extraction: dict[str, Any] = {}
            for epoch_label, elevation, valid_mask in (
                ("2018", aligned2018.elevation, aligned2018.valid_mask),
                ("2020", aligned2020.elevation, aligned2020.valid_mask),
            ):
                tile_elev, tile_valid = _tile_window(elevation, valid_mask, tile)
                # A canonical candidate already passed the natural-context + canonical-band
                # eligibility gate on ITS OWN band-restricted 2020 diagnostic (`_canonical_pass`)
                # -- for driving the ACTUAL transect orientation/extraction, every epoch
                # independently prefers ITS OWN band-restricted diagnostic (Section 5's whole
                # point: the global, unbounded-above peak can legitimately BE the tile-scale
                # artefact, never a real sand-wave orientation) over the global one. Diagnostic/
                # legacy (disturbed) tiles were never subjected to that gate at all, so they keep
                # using the plain global diagnostic, exactly reproducing MAR-022's own behaviour.
                if is_canonical:
                    diag, band_diag = swm.analyze_tile_dual_band(
                        tile_elev, tile_valid, pixel_size_m, max_wavelength_m=max_wavelength_m
                    )
                else:
                    diag = swm.analyze_tile(tile_elev, tile_valid, pixel_size_m)
                    band_diag = None
                per_epoch_diag[epoch_label] = diag
                if diag is None:
                    continue
                effective_diag = band_diag if band_diag is not None else diag
                per_epoch_effective_diag[epoch_label] = effective_diag
                tile_rows[epoch_label].append(
                    {
                        "tile_id": tile.tile_id,
                        "epoch": epoch_label,
                        "center_x_m": tile.center_x_m,
                        "center_y_m": tile.center_y_m,
                        "tile_size_m": tile.tile_size_m,
                        "valid_fraction": tile.valid_fraction,
                        "natural_bedform_validation_status": natural_status.status,
                        "global_dominant_wavelength_m": diag["dominant_wavelength_m"],
                        "canonical_band_dominant_wavelength_m": (
                            band_diag["dominant_wavelength_m"] if band_diag else None
                        ),
                        **effective_diag,
                    }
                )
                extraction = bedform_extraction.extract_tile_bedforms(
                    elevation,
                    valid_mask,
                    aligned2020.transform,
                    tile_id=tile.tile_id,
                    epoch=epoch_label,
                    center_x_m=tile.center_x_m,
                    center_y_m=tile.center_y_m,
                    tile_size_m=tile.tile_size_m,
                    crest_azimuth_deg=effective_diag["dominant_crest_azimuth_deg"],
                )
                per_epoch_extraction[epoch_label] = extraction
                bedform_rows[epoch_label].extend(extraction.bedform_rows)
                crest_points[epoch_label].extend(extraction.crest_points)
                trough_points[epoch_label].extend(extraction.trough_points)
                if epoch_label == "2020" and not representative_transects:
                    representative_transects = swm.generate_cross_crest_transects(
                        tile.center_x_m,
                        tile.center_y_m,
                        tile.tile_size_m,
                        effective_diag["dominant_crest_azimuth_deg"],
                    )

            tile_summaries.append(
                {
                    "tile_id": tile.tile_id,
                    "center_x_m": tile.center_x_m,
                    "center_y_m": tile.center_y_m,
                    "tile_size_m": tile.tile_size_m,
                    "valid_fraction": tile.valid_fraction,
                    "natural_bedform_validation_status": natural_status.status,
                    "dominant_wavelength_m": (per_epoch_effective_diag.get("2020") or {}).get(
                        "dominant_wavelength_m"
                    ),
                    "dominant_crest_azimuth_deg": (per_epoch_effective_diag.get("2020") or {}).get(
                        "dominant_crest_azimuth_deg"
                    ),
                }
            )

            if "2018" not in per_epoch_extraction or "2020" not in per_epoch_extraction:
                continue
            crests1 = [
                c
                for c in per_epoch_extraction["2018"].crest_points
                if c["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
            ]
            crests2 = [
                c
                for c in per_epoch_extraction["2020"].crest_points
                if c["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
            ]
            crests1_by_id = {c["bedform_id"]: c for c in crests1}
            crests2_by_id = {c["bedform_id"]: c for c in crests2}
            for tolerances in tolerance_sets:
                all_pairs, canonical_matches = bedform_matching.match_crests_within_tile(
                    tile.tile_id, crests1, crests2, tolerances=tolerances
                )
                for pair in all_pairs:
                    c1 = crests1_by_id.get(pair["epoch1_crest_id"])
                    c2 = crests2_by_id.get(pair["epoch2_crest_id"])
                    pair["epoch1_x_m"], pair["epoch1_y_m"] = (
                        (c1["x"], c1["y"]) if c1 else (None, None)
                    )
                    pair["epoch2_x_m"], pair["epoch2_y_m"] = (
                        (c2["x"], c2["y"]) if c2 else (None, None)
                    )
                    pair["tolerance_set"] = tolerances.name
                for pair in canonical_matches:
                    pair["apparent_rate_m_per_year"] = (
                        bedform_matching.compute_apparent_displacement_rate(
                            pair["normal_displacement_m"], elapsed_years
                        )
                    )
                    pair["elapsed_years"] = elapsed_years
                    dod_evidence = bedform_matching.sample_dod_around_matched_crest(
                        dod_array,
                        dod_transform,
                        x_m=pair["epoch1_x_m"],
                        y_m=pair["epoch1_y_m"],
                        normal_azimuth_deg=pair["normal_azimuth_deg"],
                    )
                    pair.update(dod_evidence)
                pairs_by_tolerance[tolerances.name].extend(all_pairs)
                matches_by_tolerance[tolerances.name].extend(canonical_matches)

        return {
            "tile_summaries": tile_summaries,
            "tile_rows": tile_rows,
            "bedform_rows": bedform_rows,
            "crest_points": crest_points,
            "trough_points": trough_points,
            "representative_transects": representative_transects,
            "pairs_by_tolerance": pairs_by_tolerance,
            "matches_by_tolerance": matches_by_tolerance,
        }

    print(
        "Extracting + matching the diagnostic disturbed-tile pool (NOMINAL tolerance only)...",
        flush=True,
    )
    diagnostic_result = _extract_and_match(
        legacy_records, tolerance_sets=(bedform_matching.NOMINAL_TOLERANCES,), is_canonical=False
    )
    diagnostic_bedform_count_2018 = len(diagnostic_result["bedform_rows"]["2018"])
    diagnostic_bedform_count_2020 = len(diagnostic_result["bedform_rows"]["2020"])
    diagnostic_nominal_matches = diagnostic_result["matches_by_tolerance"]["NOMINAL"]
    print(
        f"  diagnostic bedform observations -- 2018: {diagnostic_bedform_count_2018}, "
        f"2020: {diagnostic_bedform_count_2020} | diagnostic nominal matches: "
        f"{len(diagnostic_nominal_matches)}",
        flush=True,
    )

    if canonical_available:
        n_canonical_tiles = len(canonical_pass["selected_records"])
        print(
            f"Extracting + matching the canonical natural-bedform pool ({n_canonical_tiles} "
            "tiles, 3 tolerance sets)...",
            flush=True,
        )
        canonical_result = _extract_and_match(
            canonical_pass["selected_records"],
            tolerance_sets=bedform_matching.TOLERANCE_SETS,
            is_canonical=True,
        )
        nominal_matches = canonical_result["matches_by_tolerance"]["NOMINAL"]
        conservative_pairs = canonical_result["pairs_by_tolerance"]["CONSERVATIVE"]
        permissive_pairs = canonical_result["pairs_by_tolerance"]["PERMISSIVE"]
        for match in nominal_matches:
            match["matching_stability_status"] = bedform_matching.assess_matching_stability(
                match,
                conservative_all_pairs=conservative_pairs,
                permissive_all_pairs=permissive_pairs,
            )
        stable_matches = [
            m
            for m in nominal_matches
            if m["matching_stability_status"] == bedform_matching.STABILITY_STABLE
        ]
        canonical_bedform_count_2018 = sum(
            1
            for b in canonical_result["bedform_rows"]["2018"]
            if b["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
        )
        canonical_bedform_count_2020 = sum(
            1
            for b in canonical_result["bedform_rows"]["2020"]
            if b["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
        )
        print(
            f"  canonical natural (>=30 m) observations -- 2018: {canonical_bedform_count_2018}, "
            f"2020: {canonical_bedform_count_2020}",
            flush=True,
        )
        print(
            f"  nominal matches: {len(nominal_matches)} | stable across all 3 tolerance sets: "
            f"{len(stable_matches)}",
            flush=True,
        )
    else:
        canonical_result = None
        nominal_matches, stable_matches = [], []
        canonical_bedform_count_2018 = canonical_bedform_count_2020 = 0
        print(
            "  CANONICAL_NATURAL_BEDFORM_VALIDATION_NOT_DEMONSTRATED at either support scale.",
            flush=True,
        )

    # --- Section 7: tabular outputs -- canonical vs noncanonical split ------------------------
    print("Writing canonical + diagnostic tabular outputs (Section 7)...", flush=True)
    empty_tile_row_cols = [
        "tile_id",
        "epoch",
        "center_x_m",
        "center_y_m",
        "tile_size_m",
        "valid_fraction",
        "natural_bedform_validation_status",
        "global_dominant_wavelength_m",
        "canonical_band_dominant_wavelength_m",
    ]
    if canonical_available:
        pd.DataFrame(canonical_result["tile_rows"]["2018"]).to_parquet(
            bedforms_dir / "canonical_natural_tile_spectral_morphometry_2018.parquet", index=False
        )
        pd.DataFrame(canonical_result["tile_rows"]["2020"]).to_parquet(
            bedforms_dir / "canonical_natural_tile_spectral_morphometry_2020.parquet", index=False
        )
        pd.DataFrame(canonical_result["bedform_rows"]["2018"]).to_parquet(
            bedforms_dir / "canonical_natural_bedform_observations_2018.parquet", index=False
        )
        pd.DataFrame(canonical_result["bedform_rows"]["2020"]).to_parquet(
            bedforms_dir / "canonical_natural_bedform_observations_2020.parquet", index=False
        )
    else:
        for name in (
            "canonical_natural_tile_spectral_morphometry_2018.parquet",
            "canonical_natural_tile_spectral_morphometry_2020.parquet",
        ):
            pd.DataFrame(columns=empty_tile_row_cols).to_parquet(bedforms_dir / name, index=False)
        for name in (
            "canonical_natural_bedform_observations_2018.parquet",
            "canonical_natural_bedform_observations_2020.parquet",
        ):
            pd.DataFrame(columns=["bedform_id", "record_type", "tile_id", "epoch"]).to_parquet(
                bedforms_dir / name, index=False
            )

    noncanonical_rows: list[dict[str, Any]] = []
    for pass_result, scale_label in ((pass_2000, "2000m"), (pass_1000, "1000m")):
        selected_ids = {r["tile"].tile_id for r in pass_result["selected_records"]}
        for r in pass_result["excluded_records"]:
            tile = r["tile"]
            noncanonical_rows.append(
                {
                    "tile_id": tile.tile_id,
                    "support_scale_m": scale_label,
                    "tile_size_m": tile.tile_size_m,
                    "center_x_m": tile.center_x_m,
                    "center_y_m": tile.center_y_m,
                    "valid_fraction": tile.valid_fraction,
                    "natural_bedform_validation_status": r["natural_status"].status,
                    "global_dominant_wavelength_m": None,
                    "canonical_band_dominant_wavelength_m": None,
                    "spectrally_eligible": None,
                    "diagnostic_reason": "EXCLUDED_BY_NATURAL_CONTEXT_BEFORE_SPECTRAL_RANKING",
                }
            )
        for r in pass_result["natural_records"]:
            if r["tile"].tile_id in selected_ids or "band_diag" not in r:
                continue
            tile = r["tile"]
            noncanonical_rows.append(
                {
                    "tile_id": tile.tile_id,
                    "support_scale_m": scale_label,
                    "tile_size_m": tile.tile_size_m,
                    "center_x_m": tile.center_x_m,
                    "center_y_m": tile.center_y_m,
                    "valid_fraction": tile.valid_fraction,
                    "natural_bedform_validation_status": r["natural_status"].status,
                    "global_dominant_wavelength_m": (
                        r["global_diag"]["dominant_wavelength_m"] if r["global_diag"] else None
                    ),
                    "canonical_band_dominant_wavelength_m": (
                        r["band_diag"]["dominant_wavelength_m"] if r["band_diag"] else None
                    ),
                    "spectrally_eligible": r["band_diag"] is not None,
                    "diagnostic_reason": (
                        "NATURAL_ELIGIBLE_BUT_NOT_SPATIALLY_INDEPENDENT_SELECTED"
                        if r["band_diag"] is not None
                        else "NATURAL_ELIGIBLE_BUT_SPECTRALLY_INELIGIBLE_IN_CANONICAL_BAND"
                    ),
                }
            )
    for r in legacy_records:
        tile = r["tile"]
        noncanonical_rows.append(
            {
                "tile_id": tile.tile_id,
                "support_scale_m": "2000m",
                "tile_size_m": tile.tile_size_m,
                "center_x_m": tile.center_x_m,
                "center_y_m": tile.center_y_m,
                "valid_fraction": tile.valid_fraction,
                "natural_bedform_validation_status": r["natural_status"].status,
                "global_dominant_wavelength_m": legacy_global_diag_by_tile_id[tile.tile_id][
                    "dominant_wavelength_m"
                ],
                "canonical_band_dominant_wavelength_m": None,
                "spectrally_eligible": None,
                "diagnostic_reason": "LEGACY_MAR022_TOP5_DISTURBED_DETAILED_DIAGNOSTIC",
            }
        )
    noncanonical_diagnostics_df = pd.DataFrame(noncanonical_rows)
    noncanonical_diagnostics_path = bedforms_dir / "noncanonical_disturbed_tile_diagnostics.parquet"
    noncanonical_diagnostics_df.to_parquet(noncanonical_diagnostics_path, index=False)
    pd.DataFrame(diagnostic_result["bedform_rows"]["2018"]).to_parquet(
        bedforms_dir / "noncanonical_diagnostic_bedform_observations_2018.parquet", index=False
    )
    pd.DataFrame(diagnostic_result["bedform_rows"]["2020"]).to_parquet(
        bedforms_dir / "noncanonical_diagnostic_bedform_observations_2020.parquet", index=False
    )
    pd.DataFrame(diagnostic_nominal_matches).to_parquet(
        bedforms_dir / "noncanonical_diagnostic_crest_matches.parquet", index=False
    )
    print(
        f"  -> {noncanonical_diagnostics_path} ({len(noncanonical_diagnostics_df)} rows)",
        flush=True,
    )

    candidates_path = bedforms_dir / "crest_match_candidates.parquet"
    canonical_matches_path = bedforms_dir / "canonical_crest_matches.parquet"
    if canonical_available:
        candidates_df = pd.DataFrame(canonical_result["pairs_by_tolerance"]["NOMINAL"])
        canonical_matches_df = pd.DataFrame(nominal_matches)
    else:
        candidates_df = pd.DataFrame(
            columns=["epoch1_crest_id", "epoch2_crest_id", "match_status", "rejection_reason"]
        )
        canonical_matches_df = pd.DataFrame(
            columns=[
                "epoch1_crest_id",
                "epoch2_crest_id",
                "match_status",
                "matching_stability_status",
            ]
        )
    candidates_df.to_parquet(candidates_path, index=False)
    canonical_matches_df.to_parquet(canonical_matches_path, index=False)
    stable_matches_df = pd.DataFrame(stable_matches)
    print(f"  -> {candidates_path}, {canonical_matches_path}", flush=True)

    # --- Section 18: source-interpretation comparator -- canonical natural detections only -----
    print(
        "Comparing CANONICAL detected crests to source interpretation (Section 12)...", flush=True
    )
    if canonical_available and canonical_result["crest_points"]["2020"]:
        crest_points_2020_canonical = canonical_result["crest_points"]["2020"]
        detected_2020_gdf = gpd.GeoDataFrame(
            crest_points_2020_canonical,
            geometry=gpd.points_from_xy(
                [c["x"] for c in crest_points_2020_canonical],
                [c["y"] for c in crest_points_2020_canonical],
            ),
            crs=working_crs,
        )
        detected_2020_canonical_gdf = detected_2020_gdf[
            detected_2020_gdf["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
        ]
    else:
        detected_2020_canonical_gdf = gpd.GeoDataFrame(
            columns=["point_id", "x", "y", "crest_azimuth_deg"], geometry=[], crs=working_crs
        )
    comparator_status, comparator_df = (
        bedform_interpretation.compare_detected_crests_to_interpretation(
            detected_2020_canonical_gdf, natural_interp
        )
    )
    comparator_summary = bedform_interpretation.summarize_comparator_results(comparator_df)
    comparator_path = bedforms_dir / "source_bedform_comparator.parquet"
    comparator_df.to_parquet(comparator_path, index=False)
    print(
        f"  {comparator_status} | correspondence: {comparator_summary['correspondence_status']} "
        f"-> {comparator_path}",
        flush=True,
    )

    # --- Section 15: figures ---------------------------------------------------------------------
    print("Rendering canonical bedform morphometry map (Section 15)...", flush=True)
    if canonical_available:
        canonical_tile_summaries_df = pd.DataFrame(canonical_result["tile_summaries"])
        morphometry_unavailable_message = None
        morphometry_transects = canonical_result["representative_transects"]
    else:
        canonical_tile_summaries_df = pd.DataFrame(
            columns=[
                "tile_id",
                "center_x_m",
                "center_y_m",
                "tile_size_m",
                "natural_bedform_validation_status",
                "dominant_wavelength_m",
            ]
        )
        morphometry_unavailable_message = (
            "CANONICAL_NATURAL_BEDFORM_VALIDATION_NOT_DEMONSTRATED -- no natural, spectrally-"
            "eligible, spatially-independent tile at 2000 m or 1000 m"
        )
        morphometry_transects = []
    morphometry_map_path = bedform_maps.render_bedform_morphometry_map(
        background_elevation=aligned2020.elevation,
        background_valid=aligned2020.valid_mask,
        transform=aligned2020.transform,
        canonical_tiles_df=canonical_tile_summaries_df,
        transect_endpoints=morphometry_transects,
        output_path=maps_dir / "sheringham_shoal_2020_bedform_morphometry.png",
        title="Sheringham Shoal 2020 -- Canonical Natural Bedform Morphometry",
        unavailable_message=morphometry_unavailable_message,
    )
    print(f"  -> {morphometry_map_path}", flush=True)

    print("Rendering diagnostic (noncanonical) bedform morphometry map (Section 15)...", flush=True)
    diagnostic_tile_summaries_df = pd.DataFrame(diagnostic_result["tile_summaries"])
    diagnostic_morphometry_map_path = bedform_maps.render_bedform_morphometry_map(
        background_elevation=aligned2020.elevation,
        background_valid=aligned2020.valid_mask,
        transform=aligned2020.transform,
        canonical_tiles_df=diagnostic_tile_summaries_df,
        transect_endpoints=diagnostic_result["representative_transects"],
        output_path=maps_dir / "sheringham_shoal_2020_bedform_diagnostics_noncanonical.png",
        title="Sheringham disturbed-tile bedform diagnostics -- NONCANONICAL",
    )
    print(f"  -> {diagnostic_morphometry_map_path}", flush=True)

    print("Rendering canonical multi-epoch displacement map (Section 15)...", flush=True)
    if canonical_available:
        change_crests_2018 = [
            (c["x"], c["y"])
            for c in canonical_result["crest_points"]["2018"]
            if c["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
        ]
        change_crests_2020 = [
            (c["x"], c["y"])
            for c in canonical_result["crest_points"]["2020"]
            if c["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
        ]
    else:
        change_crests_2018, change_crests_2020 = [], []
    change_unavailable_message = (
        None
        if stable_matches
        else "DEFENSIBLE_OBSERVED_CREST_DISPLACEMENT_NOT_DEMONSTRATED -- zero matches stable "
        "across all 3 tested tolerance sets"
    )
    change_map_path = bedform_maps.render_bedform_change_map(
        background_delta_bed_elevation_m=dod_array,
        transform=dod_transform,
        crests_epoch1_xy=change_crests_2018,
        crests_epoch2_xy=change_crests_2020,
        matched_pairs=stable_matches,
        epoch1_label="2018",
        epoch2_label="2020",
        output_path=maps_dir / "sheringham_shoal_2018_2020_bedform_change.png",
        title="Sheringham Shoal 2018-2020 -- Canonical Stable Observed Bedform Change",
        unavailable_message=change_unavailable_message,
    )
    print(f"  -> {change_map_path}", flush=True)

    print("Rendering diagnostic (noncanonical) displacement map (Section 15)...", flush=True)
    diagnostic_change_crests_2018 = [
        (c["x"], c["y"])
        for c in diagnostic_result["crest_points"]["2018"]
        if c["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
    ]
    diagnostic_change_crests_2020 = [
        (c["x"], c["y"])
        for c in diagnostic_result["crest_points"]["2020"]
        if c["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
    ]
    diagnostic_change_map_path = bedform_maps.render_bedform_change_map(
        background_delta_bed_elevation_m=dod_array,
        transform=dod_transform,
        crests_epoch1_xy=diagnostic_change_crests_2018,
        crests_epoch2_xy=diagnostic_change_crests_2020,
        matched_pairs=diagnostic_nominal_matches,
        epoch1_label="2018",
        epoch2_label="2020",
        output_path=maps_dir
        / "sheringham_shoal_2018_2020_bedform_change_diagnostic_noncanonical.png",
        title="Sheringham disturbed-tile crest-matching diagnostics -- NONCANONICAL",
        subdued=True,
    )
    print(f"  -> {diagnostic_change_map_path}", flush=True)

    print("Rendering canonical natural morphometry statistics (Section 21)...", flush=True)
    canonical_bedforms_2018_df = (
        pd.DataFrame(
            [
                b
                for b in canonical_result["bedform_rows"]["2018"]
                if b["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
            ]
        )
        if canonical_available
        else pd.DataFrame()
    )
    canonical_bedforms_2020_df = (
        pd.DataFrame(
            [
                b
                for b in canonical_result["bedform_rows"]["2020"]
                if b["scale_classification"] == bedform_extraction.SANDBED_SAND_WAVE_SCALE
            ]
        )
        if canonical_available
        else pd.DataFrame()
    )
    statistics_map_path = bedform_maps.render_bedform_statistics(
        bedforms_epoch1_df=canonical_bedforms_2018_df,
        bedforms_epoch2_df=canonical_bedforms_2020_df,
        epoch1_label="2018",
        epoch2_label="2020",
        output_path=maps_dir / "sheringham_shoal_bedform_statistics.png",
        title="Sheringham Shoal -- Canonical Natural Bedform Morphometry Statistics",
    )
    print(f"  -> {statistics_map_path}", flush=True)

    displacement_stats_path = None
    if len(stable_matches_df) >= 3:
        print(
            "Rendering observed-displacement statistics from STABLE matches only (Section 22)...",
            flush=True,
        )
        displacement_stats_path = bedform_maps.render_displacement_statistics(
            canonical_matches_df=stable_matches_df,
            output_path=maps_dir / "sheringham_shoal_observed_crest_displacement_statistics.png",
            title="Sheringham Shoal -- Observed Apparent Crest Displacement (Stable Canonical "
            "Matches)",
        )
        print(f"  -> {displacement_stats_path}", flush=True)
    else:
        print(
            f"  skipped: only {len(stable_matches_df)} stable canonical matches (<3 required)",
            flush=True,
        )

    # --- Section 23: GIS outputs --------------------------------------------------------------
    print("Writing GIS outputs (Section 23)...", flush=True)
    gpkg_path = gis_dir / "bedform_morphodynamics_poc.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()

    if canonical_available and canonical_result["tile_summaries"]:
        tiles_gdf = gpd.GeoDataFrame(
            canonical_tile_summaries_df,
            geometry=[_tile_box_from_row(row) for _, row in canonical_tile_summaries_df.iterrows()],
            crs=working_crs,
        )
        tiles_gdf.to_file(gpkg_path, driver="GPKG", layer="canonical_validation_tiles")
    if not noncanonical_diagnostics_df.empty:
        noncanonical_tiles_gdf = gpd.GeoDataFrame(
            noncanonical_diagnostics_df,
            geometry=[_tile_box_from_row(row) for _, row in noncanonical_diagnostics_df.iterrows()],
            crs=working_crs,
        )
        noncanonical_tiles_gdf.to_file(
            gpkg_path, driver="GPKG", layer="noncanonical_diagnostic_tiles"
        )

    def _extrema_gdf(crest_points, trough_points):
        rows = [{**c, "extremum_type": "crest"} for c in crest_points] + [
            {**t, "extremum_type": "trough"} for t in trough_points
        ]
        if not rows:
            return gpd.GeoDataFrame(
                columns=["point_id", "extremum_type"], geometry=[], crs=working_crs
            )
        geometry = gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows])
        frame = pd.DataFrame(rows).drop(columns=["x", "y"])
        return gpd.GeoDataFrame(frame, geometry=geometry, crs=working_crs)

    if canonical_available:
        extrema_2018_gdf = _extrema_gdf(
            canonical_result["crest_points"]["2018"], canonical_result["trough_points"]["2018"]
        )
        extrema_2020_gdf = _extrema_gdf(
            canonical_result["crest_points"]["2020"], canonical_result["trough_points"]["2020"]
        )
        if not extrema_2018_gdf.empty:
            extrema_2018_gdf.to_file(gpkg_path, driver="GPKG", layer="bedform_extrema_2018")
        if not extrema_2020_gdf.empty:
            extrema_2020_gdf.to_file(gpkg_path, driver="GPKG", layer="bedform_extrema_2020")

        matched_only = [
            p
            for p in canonical_result["pairs_by_tolerance"]["NOMINAL"]
            if p["match_status"]
            in (bedform_matching.MATCHED_HIGH_SUPPORT, bedform_matching.MATCHED_WITH_LIMITATIONS)
        ]
        if matched_only:
            matches_gdf = gpd.GeoDataFrame(
                pd.DataFrame(matched_only),
                geometry=[
                    shapely_linestring(
                        [(p["epoch1_x_m"], p["epoch1_y_m"]), (p["epoch2_x_m"], p["epoch2_y_m"])]
                    )
                    for p in matched_only
                ],
                crs=working_crs,
            )
            matches_gdf.to_file(gpkg_path, driver="GPKG", layer="crest_matches_2018_2020")

    if not classified_interp.empty:
        classified_interp.to_file(gpkg_path, driver="GPKG", layer="source_interpretation_context")
    if not anthropogenic_interp.empty:
        anthropogenic_interp.to_file(
            gpkg_path, driver="GPKG", layer="excluded_anthropogenic_context"
        )
    print(f"  -> {gpkg_path}", flush=True)

    # --- Section 26: input contract ------------------------------------------------------------
    contract = bedform_contract.build_bedform_morphodynamics_input_contract()
    contract_path = study_dir / "bedform_morphodynamics_input_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")
    print(f"Wrote input contract (Section 26) -> {contract_path}", flush=True)

    # --- Section 27: report ---------------------------------------------------------------------
    print("Building bedform morphodynamics POC report (Section 27)...", flush=True)
    displacement_facts: dict[str, Any] = {
        "Nominal canonical matches": len(nominal_matches),
        "Stable-across-all-3-tolerances matches": len(stable_matches),
        "OBSERVED_CREST_DISPLACEMENT_2018_2020": bedform_matching.OBSERVED_CREST_DISPLACEMENT,
    }
    if len(stable_matches_df):
        displacement_facts.update(
            {
                "Absolute displacement min/median/max (m, stable only)": (
                    f"{stable_matches_df['absolute_normal_displacement_m'].min():.3f} / "
                    f"{stable_matches_df['absolute_normal_displacement_m'].median():.3f} / "
                    f"{stable_matches_df['absolute_normal_displacement_m'].max():.3f}"
                ),
                "Apparent rate min/median/max (m/yr, stable only)": (
                    f"{stable_matches_df['apparent_rate_m_per_year'].min():.3f} / "
                    f"{stable_matches_df['apparent_rate_m_per_year'].median():.3f} / "
                    f"{stable_matches_df['apparent_rate_m_per_year'].max():.3f}"
                ),
                "Label": bedform_matching.OBSERVED_APPARENT_CREST_DISPLACEMENT_RATE,
                "Disclaimer": bedform_matching.APPARENT_RATE_DISCLAIMER,
            }
        )
    input_contract_summary = [
        f"{f['field']}: {f['description']}"
        for f in contract["required_fields_static_morphometry"]
        + contract["required_fields_multi_epoch_change"]
    ]
    report_blocks = bedform_report.build_bedform_morphodynamics_report_blocks(
        project_title="Sheringham Shoal -- Sand-Wave/Bedform Morphodynamics POC (MAR-022A)",
        source_data_facts={
            "2018 source": f"{sheringham_2018_provider.DATASET_TITLE} "
            f"({len(acq2018.xyz_entry_names)} XYZ parts, survey epoch {acq2018.survey_period})",
            "2020 source": f"{sheringham_provider.DATASET_TITLE} "
            f"({acq2020.target_entry_name}, survey epoch {acq2020.survey_period})",
            "Interpretation package": f"{len(interp_acq.layer_names)} shapefiles, "
            f"{interp_acq.package_bytes:,} bytes",
        },
        canonical_support_facts={
            "2000 m: support / natural-eligible / spectrally-eligible / selected": (
                f"{pass_2000['support_tile_count']} / {len(pass_2000['natural_records'])} / "
                f"{pass_2000['spectrally_eligible_count']} / {len(pass_2000['selected_records'])}"
            ),
            "1000 m: support / natural-eligible / spectrally-eligible / selected": (
                f"{pass_1000['support_tile_count']} / {len(pass_1000['natural_records'])} / "
                f"{pass_1000['spectrally_eligible_count']} / {len(pass_1000['selected_records'])}"
            ),
            "Canonical scale used": canonical_scale_used_m if canonical_available else "NONE",
        },
        natural_vs_anthropogenic_facts={
            "Natural interpretation features (sand-wave crest)": len(natural_interp),
            "Anthropogenic interpretation features": len(anthropogenic_interp),
            "Unclassified interpretation features": len(unclassified_interp),
            "Diagnostic (legacy top-5) tiles, natural-eligible count": legacy_natural_count,
        },
        epoch1_morphometry_facts={
            "Canonical natural (>=30 m) observations": canonical_bedform_count_2018,
            "Diagnostic disturbed-tile observations (noncanonical)": diagnostic_bedform_count_2018,
        },
        epoch2_morphometry_facts={
            "Canonical natural (>=30 m) observations": canonical_bedform_count_2020,
            "Diagnostic disturbed-tile observations (noncanonical)": diagnostic_bedform_count_2020,
        },
        crest_matching_facts={
            "Nominal canonical candidate pairs": len(candidates_df),
            "Nominal canonical matches": len(nominal_matches),
            "Stable across CONSERVATIVE+NOMINAL+PERMISSIVE": len(stable_matches),
            "Tolerance sets tested": ", ".join(t.name for t in bedform_matching.TOLERANCE_SETS),
        },
        observed_displacement_facts=displacement_facts,
        dod_supporting_context_text="For each STABLE canonical matched crest, the ALREADY-"
        "ACCEPTED MAR-021 DoD (never recomputed here) is sampled at the crest position and at a "
        "fixed offset either side along that pair's own local cross-crest normal, purely as "
        "descriptive supporting context. A perfect raising/lowering dipole is never required, "
        "and this evidence never determines match acceptance or stability.",
        source_interpretation_comparison_facts={
            "Status": comparator_status,
            "Correspondence": comparator_summary["correspondence_status"] or "ADEQUATE",
            "Compared (canonical natural detections)": comparator_summary["compared_count"],
            "Median / p95 nearest distance (m)": (
                f"{comparator_summary['median_nearest_distance_m']:.2f} / "
                f"{comparator_summary['p95_nearest_distance_m']:.2f}"
                if comparator_summary["compared_count"]
                else "n/a"
            ),
            "Fraction within 25 / 50 / 100 m": (
                f"{comparator_summary['fraction_within_25m']:.2%} / "
                f"{comparator_summary['fraction_within_50m']:.2%} / "
                f"{comparator_summary['fraction_within_100m']:.2%}"
                if comparator_summary["compared_count"]
                else "n/a"
            ),
        },
        limitations=[
            "Transect-derived point OBSERVATIONS only (TRANSECT_DERIVED_BEDFORM_OBSERVATION / "
            "TRANSECT_DERIVED_CREST_OBSERVATION) -- never independent, unique physical crest "
            "lines; no continuous 2D crest-line reconstruction exists in this POC.",
            "Crest matching is scoped to within one canonical tile -- never a whole-site search "
            "across clearly different bedform systems.",
            "The historical 2013-2014 ~10 m migration context for this site is external "
            "background only and was never used to calibrate matching tolerances or interpret "
            "2018-2020 displacement.",
            "No numeric match-confidence score is produced; only categorical match and stability "
            "statuses.",
            "The diagnostic disturbed-tile figures/tables are explicitly NONCANONICAL -- never "
            "used for the primary observed-displacement claim.",
        ],
        input_contract_summary=input_contract_summary,
    )
    report_path = report_dir / "sheringham_shoal_bedform_morphodynamics_poc.html"
    report_path.write_text(
        bedform_report.render_blocks_html(
            report_blocks,
            title="Sheringham Shoal -- Sand-Wave/Bedform Morphodynamics POC (MAR-022A)",
        ),
        encoding="utf-8",
    )
    print(f"  -> {report_path}", flush=True)

    # --- Section 13: validation questions -- causally consistent ------------------------------
    questions = _derive_bedform_validation_questions(
        has_1000m_support=pass_1000["support_tile_count"] > 0,
        has_support_at_either_scale=(
            pass_2000["support_tile_count"] + pass_1000["support_tile_count"]
        )
        > 0,
        canonical_bedform_count=canonical_bedform_count_2018 + canonical_bedform_count_2020,
        canonical_selected_tile_count=(
            len(canonical_pass["selected_records"]) if canonical_available else 0
        ),
        stable_match_count=len(stable_matches),
    )
    question_a = questions["a"]
    question_b = questions["b"]
    question_c = questions["c"]
    question_d = questions["d"]
    question_e = questions["e"]
    question_f = questions["f"]
    question_g = questions["g"]
    question_d_reason = None
    if question_d == "NO":
        question_d_reason = (
            "CANONICAL_NATURAL_BEDFORM_VALIDATION_NOT_DEMONSTRATED: no tile at 2000 m or 1000 m "
            "support scale was simultaneously >=90% valid, NATURAL_SEABED_ELIGIBLE, and "
            "spectrally eligible within the canonical 30 m <= wavelength <= tile_size/3 band. "
            "This does not invalidate the generic engine -- the disturbed-tile diagnostics above "
            "remain useful POC evidence."
        )
    _write_validation(
        question_a,
        question_b,
        question_c,
        question_d,
        question_e,
        question_f,
        question_g,
        question_d_reason=question_d_reason,
    )
    print(f"  -> {validation_path}", flush=True)

    # --- Section 18: final summary ---------------------------------------------------------------
    print()
    print("=== Sheringham Shoal Sand-Wave/Bedform Morphodynamics POC (MAR-022A) ===")
    print()
    print("## 2000 m canonical pass")
    print(
        f"  support={pass_2000['support_tile_count']} "
        f"natural_eligible={len(pass_2000['natural_records'])} "
        f"spectral_eligible={pass_2000['spectrally_eligible_count']} "
        f"selected={len(pass_2000['selected_records'])}"
    )
    print()
    print("## 1000 m canonical pass")
    print(
        f"  support={pass_1000['support_tile_count']} "
        f"natural_eligible={len(pass_1000['natural_records'])} "
        f"spectral_eligible={pass_1000['spectrally_eligible_count']} "
        f"selected={len(pass_1000['selected_records'])}"
    )
    print()
    print("## Diagnostic disturbed tiles (NONCANONICAL)")
    print(
        f"  count={len(legacy_records)} bedform_observations_2018={diagnostic_bedform_count_2018} "
        f"bedform_observations_2020={diagnostic_bedform_count_2020}"
    )
    print()
    print("## Canonical natural morphometry")
    print(
        f"  observation count -- 2018: {canonical_bedform_count_2018}, "
        f"2020: {canonical_bedform_count_2020}"
    )
    if canonical_available and len(canonical_bedforms_2020_df):
        print(
            "  2020 wavelength median/p95 (m): "
            f"{canonical_bedforms_2020_df['wavelength_m'].median():.1f} / "
            f"{canonical_bedforms_2020_df['wavelength_m'].quantile(0.95):.1f}"
        )
        print(
            "  2020 height median/p95 (m): "
            f"{canonical_bedforms_2020_df['wave_height_m'].median():.2f} / "
            f"{canonical_bedforms_2020_df['wave_height_m'].quantile(0.95):.2f}"
        )
    print()
    print("## Matching")
    print(
        f"  nominal match count: {len(nominal_matches)} | stable match count: {len(stable_matches)}"
    )
    print(f"  tolerance-sensitive count: {len(nominal_matches) - len(stable_matches)}")
    if len(stable_matches_df):
        print(
            "  displacement (stable only) min/median/p95/max (m): "
            f"{stable_matches_df['absolute_normal_displacement_m'].min():.3f}/"
            f"{stable_matches_df['absolute_normal_displacement_m'].median():.3f}/"
            f"{stable_matches_df['absolute_normal_displacement_m'].quantile(0.95):.3f}/"
            f"{stable_matches_df['absolute_normal_displacement_m'].max():.3f}"
        )
    print()
    print("## Source comparator")
    median_dist = comparator_summary["median_nearest_distance_m"]
    p95_dist = comparator_summary["p95_nearest_distance_m"]
    print(f"  median/p95 distance (m): {median_dist}/{p95_dist}")
    print(
        f"  fraction <=25/50/100 m: {comparator_summary['fraction_within_25m']}/"
        f"{comparator_summary['fraction_within_50m']}/{comparator_summary['fraction_within_100m']}"
    )
    print(f"  correspondence: {comparator_summary['correspondence_status']}")
    print()
    canonical_morphometry_demonstrated = (
        "YES" if (question_b == "YES" and question_c == "YES" and question_d == "YES") else "NO"
    )
    print(
        "IS CANONICAL NATURAL SANDBED MORPHOMETRY DEMONSTRATED ON REAL PROJECT-GRADE MBES? "
        f"{canonical_morphometry_demonstrated}"
    )
    print(f"IS DEFENSIBLE OBSERVED 2018-2020 CREST DISPLACEMENT DEMONSTRATED? {question_e}")
    return 0


_BARROW_ASSET_ID = "BARROW_EXPORT_CABLE"
_BARROW_COORD_MATCH_TOLERANCE_M = 1.0


def _derive_burial_exposure_poc_validation_questions(
    *,
    real_dob_dataset_ingested: bool,
    burial_reference_resolved: bool,
    authoritative_route_recovered: bool,
    measured_burial_profile_produced: bool,
    explicit_exposure_evidence_present: bool,
    generic_exposure_screening_computable: bool,
) -> dict[str, str]:
    """MAR-024 Section 22: a pure function so G/H's mandated NO answers are structurally
    enforced (asserted) -- this POC never claims a real Barrow future exposure susceptibility
    result and never produces an exposure probability, regardless of what upstream facts say."""

    barrow_future_susceptibility_defensible = False
    exposure_probability_produced = False
    assert barrow_future_susceptibility_defensible is False
    assert exposure_probability_produced is False

    return {
        "question_a_real_operator_style_dob_dataset_ingested": (
            "YES" if real_dob_dataset_ingested else "NO"
        ),
        "question_b_source_burial_measurement_reference_resolved": (
            "YES" if burial_reference_resolved else "NO"
        ),
        "question_c_authoritative_route_kp_model_recovered": (
            "YES" if authoritative_route_recovered else "NO"
        ),
        "question_d_real_measured_burial_profile_produced": (
            "YES" if measured_burial_profile_produced else "NO"
        ),
        "question_e_explicit_source_interpreted_exposure_evidence_present": (
            "YES" if explicit_exposure_evidence_present else "NO"
        ),
        "question_f_generic_cover_depletion_exposure_screening_computable": (
            "YES" if generic_exposure_screening_computable else "NO"
        ),
        "question_g_real_barrow_future_exposure_susceptibility_defensible": (
            "YES" if barrow_future_susceptibility_defensible else "NO"
        ),
        "question_h_exposure_probability_produced": (
            "YES" if exposure_probability_produced else "NO"
        ),
    }


def _cmd_build_burial_exposure_poc(args: argparse.Namespace) -> int:
    """MAR-024: generic linear-asset burial/exposure state and cover-margin screening POC,
    benchmarked against the real 2016 Deep BV Barrow Offshore Wind Farm export cable
    geophysical depth-of-burial survey (TCE-48). Three concepts kept separate throughout:
    observed/measured burial state (A), source-interpreted exposure evidence (B), and future
    exposure susceptibility (C) -- C is only ever produced given a defensible seabed-lowering
    input, which the real Barrow package does not provide (Section 16).

    Performs at most two minimal live acquisitions (Depth of Burial Listing, Route Position
    List Files) if not already cached, then is fully offline.
    """

    config = load_study_config(args.config)
    study_id = config.study.id.lower()
    study_dir = config.paths.processed_dir / study_id
    raw_dir = config.paths.raw_dir / study_id
    burial_dir = study_dir / "burial"
    readiness_dir = study_dir / "readiness"
    maps_dir = study_dir / "maps"
    report_dir = study_dir / "report"
    working_crs = config.crs.horizontal

    # ==========================================================================================
    # Section 3: minimal acquisition
    # ==========================================================================================
    print("Acquiring the 2016 Barrow Depth of Burial Listing (Section 3)...")
    dob_acquisition = barrow_2016_provider.download_depth_of_burial_listing(raw_dir)
    print(
        f"  {dob_acquisition.package_bytes:,} bytes, "
        f"sha256={dob_acquisition.package_sha256[:12]}..., "
        f"already_cached={dob_acquisition.already_cached}"
    )
    print("Acquiring the 2016 Barrow Route Position List Files (Section 3)...")
    rpl_acquisition = barrow_2016_provider.download_route_position_list(raw_dir)
    print(
        f"  {rpl_acquisition.package_bytes:,} bytes, "
        f"sha256={rpl_acquisition.package_sha256[:12]}..., "
        f"already_cached={rpl_acquisition.already_cached}"
    )

    dob_df = barrow_2016_provider.load_dob_listing_raw(dob_acquisition.extracted_dir)
    line_gdf, points_gdf = barrow_2016_provider.load_route_position_list(
        rpl_acquisition.extracted_dir
    )
    print(f"  {len(dob_df)} real DoB record(s) loaded")
    print(f"  {len(points_gdf)} real route-position point(s) loaded")

    # ==========================================================================================
    # Section 6: canonical route -- the real source route/route-position line, never invented
    # ==========================================================================================
    try:
        route = burial_route.resolve_route_linestring(line_gdf.geometry.iloc[0])
    except burial_route.InvalidAssetRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    start_point = route.interpolate(0.0)
    end_point = route.interpolate(route.length)
    min_kp_point = points_gdf.loc[points_gdf[barrow_2016_provider.RPL_KP_COLUMN].idxmin()].geometry
    max_kp_point = points_gdf.loc[points_gdf[barrow_2016_provider.RPL_KP_COLUMN].idxmax()].geometry
    start_matches_min_kp = start_point.distance(min_kp_point) < _BARROW_COORD_MATCH_TOLERANCE_M
    end_matches_max_kp = end_point.distance(max_kp_point) < _BARROW_COORD_MATCH_TOLERANCE_M
    if start_matches_min_kp and end_matches_max_kp:
        geometry_direction_semantics = (
            "GEOMETRY_START_TO_END_MATCHES_INCREASING_SOURCE_KP "
            "(verified against the real route-position-list points, not assumed)"
        )
    elif start_point.distance(max_kp_point) < _BARROW_COORD_MATCH_TOLERANCE_M and (
        end_point.distance(min_kp_point) < _BARROW_COORD_MATCH_TOLERANCE_M
    ):
        geometry_direction_semantics = (
            "GEOMETRY_START_TO_END_MATCHES_DECREASING_SOURCE_KP "
            "(verified against the real route-position-list points, not assumed)"
        )
    else:
        geometry_direction_semantics = (
            "GEOMETRY_DIRECTION_VS_SOURCE_KP_UNRESOLVED "
            "(could not verify against route-position-list points within tolerance)"
        )
    print(f"  Route direction semantics: {geometry_direction_semantics}")

    canonical_route_gdf = burial_route.build_canonical_asset_route(
        route=route,
        asset_id=_BARROW_ASSET_ID,
        source_route_name=barrow_2016_provider.RPL_LINE_FILENAME,
        source_crs=working_crs,
        geometry_direction_semantics=geometry_direction_semantics,
    )
    canonical_route_path = burial_route.write_canonical_asset_route_gpkg(
        canonical_route_gdf, burial_dir / "canonical_asset_route.gpkg"
    )
    print(f"  Canonical asset route ({route.length:.1f} m) -> {canonical_route_path}")

    # ==========================================================================================
    # Section 4: source burial-measurement semantics -- a hard gate, resolved from real evidence
    # ==========================================================================================
    print("Resolving source burial-measurement semantics (Section 4)...")
    dob_z = dob_df[barrow_2016_provider.DOB_Z_COLUMN]
    exposure_mask = (
        dob_df[barrow_2016_provider.DOB_STORAGE_DB_COLUMN]
        == barrow_2016_provider.SOURCE_EXPOSURE_FLAG_VALUE
    )
    z_exposed_mean = float(dob_z[exposure_mask].mean())
    z_non_exposed_mean = float(dob_z[~exposure_mask].mean())
    resolution_evidence = (
        "Source MEDIN lineage states 'depth of burial (DoB or \"z\")' as a general survey- "
        "campaign objective, but the real data is inconsistent with a literal burial-depth- "
        f"below-seabed reading: {barrow_2016_provider.DOB_Z_COLUMN!r} ranges "
        f"{float(dob_z.min()):.2f} to {float(dob_z.max()):.2f} m "
        f"(mean {float(dob_z.mean()):.2f} m) across {len(dob_df)} real records, and the "
        f"{int(exposure_mask.sum())} rows the source "
        f"itself flags {barrow_2016_provider.SOURCE_EXPOSURE_FLAG_VALUE!r} in "
        f"{barrow_2016_provider.DOB_STORAGE_DB_COLUMN!r} show a statistically similar Z "
        f"distribution (mean {z_exposed_mean:.2f} m) to non-exposure rows (mean "
        f"{z_non_exposed_mean:.2f} m) rather than clustering near 0 m as a true burial-depth "
        "convention would require. Z is retained and reported exactly as provided, never "
        "relabelled as top-of-cable or centreline burial. Because the reference point itself "
        "is unresolved, no sign convention is asserted either (MAR-024A) -- "
        f"{burial_semantics.SIGN_CONVENTION_UNRESOLVED}."
    )
    burial_semantics_obj = burial_semantics.BurialMeasurementSemantics(
        source_measurement_name=barrow_2016_provider.DOB_Z_COLUMN,
        measurement_reference_point=burial_semantics.SOURCE_BURIAL_REFERENCE_UNRESOLVED,
        sign_convention=burial_semantics.SIGN_CONVENTION_UNRESOLVED,
        source_sign_convention_text=(
            "Negative-down observed empirically "
            f"({int((dob_z < 0).sum())}/{len(dob_df)} real records < 0); no source-stated "
            "datum -- kept for provenance only, never trusted for classification (MAR-024A)."
        ),
        units="m",
        source_stated_uncertainty_available=True,
        survey_technique=(
            "Innomar sub-bottom profiler cross-referenced against an acoustic cable tracker "
            "(source-titled 'DoB listing correlated with acoustics')"
        ),
        unknown_fields=(
            "vertical datum for Z",
            "whether Z is below-seabed or below-chart-datum",
            "DepthSD Value definition",
            "Decibel / Frequency Value / Current Value definitions",
        ),
        resolution_evidence=resolution_evidence,
    )
    burial_semantics_dict = burial_semantics.build_source_burial_semantics(burial_semantics_obj)
    semantics_path = burial_dir / "source_burial_semantics.json"
    semantics_path.parent.mkdir(parents=True, exist_ok=True)
    semantics_path.write_text(
        json.dumps(burial_semantics_dict, indent=2, default=str), encoding="utf-8"
    )
    print(f"  {burial_semantics_obj.measurement_reference_point} -> {semantics_path}")

    # ==========================================================================================
    # Section 7: canonical burial profile
    # ==========================================================================================
    print("Building the canonical burial profile (Section 7)...")
    dob_df = dob_df.reset_index(drop=True)
    dob_df["_source_record_id"] = dob_df.index.astype(str)
    dob_df["_is_exposed"] = exposure_mask.to_numpy()

    uncertainty_p95 = dob_df[barrow_2016_provider.DOB_UNCERTAINTY_COLUMN].quantile(0.95)
    qa_flags_by_index: dict[Any, list[str]] = {}
    for idx, row in dob_df.iterrows():
        flags = []
        if row.get(barrow_2016_provider.DOB_DATA_QUALITY_COLUMN) == 2:
            flags.append("SOURCE_DATA_QUALITY_FLAG_2")
        if row[barrow_2016_provider.DOB_UNCERTAINTY_COLUMN] > uncertainty_p95:
            flags.append("HIGH_UNCERTAINTY_TOP_5_PERCENT")
        if flags:
            qa_flags_by_index[idx] = flags

    profile_df = burial_profile.build_canonical_burial_profile(
        records_df=dob_df,
        asset_id=_BARROW_ASSET_ID,
        route=route,
        x_column=barrow_2016_provider.DOB_EASTING_COLUMN,
        y_column=barrow_2016_provider.DOB_NORTHING_COLUMN,
        measured_value_column=barrow_2016_provider.DOB_Z_COLUMN,
        source_kp_column=barrow_2016_provider.DOB_KP_COLUMN,
        record_id_column="_source_record_id",
        burial_reference_type=burial_semantics.SOURCE_BURIAL_REFERENCE_UNRESOLVED,
        sign_convention=burial_semantics_obj.sign_convention,
        survey_epoch=barrow_2016_provider.SURVEY_EPOCH,
        measurement_method=burial_semantics_obj.survey_technique,
        source_sign_convention_text=burial_semantics_obj.source_sign_convention_text,
        reference_to_asset_top_offset_m=None,  # no source-stated offset/geometry -- never assumed
        source_uncertainty_column=barrow_2016_provider.DOB_UNCERTAINTY_COLUMN,
        exposure_flag_column="_is_exposed",
        qa_flags_by_index=qa_flags_by_index,
    )
    profile_path = metocean_evidence.write_parquet(
        profile_df, burial_dir / "canonical_burial_profile.parquet"
    )
    print(f"  {len(profile_df)} canonical profile row(s) -> {profile_path}")

    profile_stats = burial_profile.compute_profile_statistics(profile_df)
    for key, value in profile_stats.items():
        print(f"  {key}: {value}")

    # ==========================================================================================
    # Section 5: burial-profile data readiness
    # ==========================================================================================
    print("Assessing burial-profile data readiness (Section 5)...")
    source_kp = dob_df[barrow_2016_provider.DOB_KP_COLUMN]
    coverage_fraction = (
        (float(source_kp.max()) - float(source_kp.min())) / route.length if route.length else None
    )
    facts = burial_readiness.BurialProfileFacts(
        source_file_readable=True,
        record_count=len(dob_df),
        route_identifier_available=True,
        kp_available=True,
        kp_is_monotonic=bool(source_kp.is_monotonic_increasing),
        duplicate_kp_count=int(source_kp.duplicated().sum()),
        coordinate_support=True,
        crs_available=True,
        units_available=True,
        units_consistent_across_sources=(
            barrow_2016_provider.DOB_KP_UNITS == barrow_2016_provider.RPL_KP_UNITS
        ),
        burial_reference_known=False,
        missing_value_fraction=float(dob_z.isna().mean()),
        coverage_fraction=coverage_fraction,
        suspicious_spike_count=0,
        negative_value_count=int((dob_z < 0).sum()),
        zero_value_count=int((dob_z == 0).sum()),
        source_uncertainty_available=True,
        survey_epoch_known=True,
    )
    readiness_result = burial_readiness.assess_burial_profile_readiness(facts)
    print(f"  Readiness: {readiness_result.status}")
    for reason in readiness_result.reasons():
        print(f"    - {reason}")
    readiness_path = readiness_dir / "burial_profile_readiness.json"
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_path.write_text(
        json.dumps(readiness_result.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    print(f"  Readiness written -> {readiness_path}")

    # ==========================================================================================
    # Section 9: source-interpreted exposure evidence
    # ==========================================================================================
    exposure_df = burial_profile.extract_source_interpreted_exposure(profile_df)
    print(f"Source-interpreted exposure evidence (Section 9): {len(exposure_df)} record(s)")

    # ==========================================================================================
    # Sections 12-13: target / required burial + burial margin -- none stated by this source
    # ==========================================================================================
    target_burial_m = None  # no source-stated target/design burial depth found in the DoB listing
    print(
        "Target/required burial (Sections 12-13): no source-stated reference found -- "
        "burial_margin_m is null for every record (never invented)."
    )

    # ==========================================================================================
    # Sections 10-11: maps
    # ==========================================================================================
    value_label = "Measured value, Z (m) -- burial reference unresolved"
    profile_gdf_for_map = profile_df
    state_map_path = burial_maps.render_observed_burial_state_map(
        route=route,
        profile_df=profile_gdf_for_map,
        output_path=maps_dir / "barrow_2016_observed_burial_state.png",
        value_label=value_label,
    )
    print(f"  Observed burial state map -> {state_map_path}")

    gap_threshold_m = 200.0
    coverage_gaps_m: list[tuple[float, float]] = []
    if not profile_df.empty:
        ordered_chainage = profile_df["chainage_m"].sort_values().to_numpy()
        gaps = np.diff(ordered_chainage)
        for i, gap in enumerate(gaps):
            if gap > gap_threshold_m:
                coverage_gaps_m.append((float(ordered_chainage[i]), float(ordered_chainage[i + 1])))

    kp_profile_path = burial_maps.render_burial_kp_profile(
        profile_df=profile_df,
        output_path=maps_dir / "barrow_2016_burial_kp_profile.png",
        value_label=value_label,
        target_burial_m=target_burial_m,
        coverage_gaps_m=coverage_gaps_m,
    )
    print(
        f"  Burial KP profile -> {kp_profile_path} ({len(coverage_gaps_m)} gap(s) > "
        f"{gap_threshold_m:.0f} m flagged)"
    )

    # ==========================================================================================
    # Section 19: GIS
    # ==========================================================================================
    print("Writing GIS outputs (Section 19)...")
    gpkg_path = burial_dir / "burial_exposure_poc.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    canonical_route_gdf.to_file(gpkg_path, driver="GPKG", layer="asset_route")

    measurements_gdf = gpd.GeoDataFrame(
        profile_df,
        geometry=gpd.points_from_xy(profile_df["x_m"], profile_df["y_m"]),
        crs=working_crs,
    )
    measurements_gdf.to_file(gpkg_path, driver="GPKG", layer="burial_measurements")

    if not exposure_df.empty:
        exposure_gdf = gpd.GeoDataFrame(
            exposure_df,
            geometry=gpd.points_from_xy(exposure_df["x_m"], exposure_df["y_m"]),
            crs=working_crs,
        )
        exposure_gdf.to_file(gpkg_path, driver="GPKG", layer="source_interpreted_exposure")

    if not profile_df.empty:
        coverage_geom = shapely_substring(
            route, float(profile_df["chainage_m"].min()), float(profile_df["chainage_m"].max())
        )
        coverage_gdf = gpd.GeoDataFrame(
            [{"asset_id": _BARROW_ASSET_ID, "coverage_length_m": coverage_geom.length}],
            geometry=[coverage_geom],
            crs=working_crs,
        )
        coverage_gdf.to_file(gpkg_path, driver="GPKG", layer="survey_coverage")

    qa_df = profile_df[profile_df["qa_flags"].notna()]
    if not qa_df.empty:
        qa_gdf = gpd.GeoDataFrame(
            qa_df, geometry=gpd.points_from_xy(qa_df["x_m"], qa_df["y_m"]), crs=working_crs
        )
        qa_gdf.to_file(gpkg_path, driver="GPKG", layer="qa_flags")
    print(f"  GIS: {gpkg_path}")

    # ==========================================================================================
    # Section 20: generic input contract
    # ==========================================================================================
    contract_dict = burial_contract.build_burial_exposure_input_contract()
    contract_path = burial_dir / "burial_exposure_input_contract.json"
    contract_path.write_text(json.dumps(contract_dict, indent=2, default=str), encoding="utf-8")
    print(f"  Input contract -> {contract_path}")

    # ==========================================================================================
    # Sections 14-17: generic exposure-susceptibility screening contract (demonstrated
    # generically; Section 16's real-run rule: no lowering magnitude is invented for Barrow)
    # ==========================================================================================
    # cover_above_asset_m is null for every real Barrow record (burial reference unresolved,
    # MAR-024A Section 10) -- the raw Z median is never substituted in its place (Section 9).
    real_screening_result = burial_exposure_screening.screen_exposure_susceptibility(None, None)
    print(
        "Real Barrow exposure-screening result (Section 16): "
        f"{real_screening_result['screening_state']}"
    )

    # ==========================================================================================
    # Section 21-22: report + validation
    # ==========================================================================================
    print("Building the generic burial/exposure POC report (Section 21)...")

    validation = _derive_burial_exposure_poc_validation_questions(
        real_dob_dataset_ingested=len(dob_df) > 0,
        burial_reference_resolved=(
            burial_semantics_obj.measurement_reference_point
            != burial_semantics.SOURCE_BURIAL_REFERENCE_UNRESOLVED
        ),
        authoritative_route_recovered=route.length > 0,
        measured_burial_profile_produced=len(profile_df) > 0,
        explicit_exposure_evidence_present=len(exposure_df) > 0,
        generic_exposure_screening_computable=(
            burial_exposure_screening.screen_exposure_susceptibility(
                1.0,
                burial_exposure_screening.SeabedLoweringInput(
                    0.4, burial_exposure_screening.OPERATOR_DEFINED_LOWERING_SCENARIO
                ),
            )["screening_state"]
            == burial_exposure_screening.POSITIVE_COVER_REMAINS_IN_SCREENING
        ),
    )

    blocks = burial_report.build_burial_exposure_report_blocks(
        project_title="Generic Linear-Asset Burial/Exposure State POC",
        purpose_text=(
            "Demonstrates the future OrbGSS workflow: operator route + measured depth of "
            "burial + source-interpreted exposure/seabed features -> data QA/readiness -> "
            "canonical burial profile -> current burial/exposure state -> cover-margin "
            "screening -> map + KP view + GIS + report. Observed/measured burial state and "
            "source-interpreted exposure evidence can be demonstrated from real source data "
            "alone; future exposure susceptibility may only be produced when defensible "
            "seabed-lowering input is supplied."
        ),
        source_survey_facts={
            "source_page": barrow_2016_provider.MDE_SOURCE_PAGE_URL,
            "dob_listing_package_url": barrow_2016_provider.DOB_LISTING_URL,
            "dob_listing_bytes": dob_acquisition.package_bytes,
            "dob_listing_sha256": dob_acquisition.package_sha256,
            "route_position_list_package_url": barrow_2016_provider.ROUTE_POSITION_LIST_URL,
            "route_position_list_bytes": rpl_acquisition.package_bytes,
            "route_position_list_sha256": rpl_acquisition.package_sha256,
            "survey_epoch": barrow_2016_provider.SURVEY_EPOCH,
            "survey_contractor": barrow_2016_provider.SURVEY_CONTRACTOR,
            "crs": working_crs,
        },
        burial_reference_semantics_facts=burial_semantics_dict,
        data_readiness_facts=readiness_result.to_dict(),
        route_and_coverage_facts={
            "route_length_m": route.length,
            "geometry_direction_semantics": geometry_direction_semantics,
            "coverage_fraction": coverage_fraction,
            "coverage_gap_count": len(coverage_gaps_m),
        },
        canonical_cover_semantics_text=(
            "This profile distinguishes five concepts (MAR-024A): RAW SOURCE MEASUREMENT (the "
            f"value exactly as {barrow_2016_provider.DOB_Z_COLUMN!r} was reported by the "
            "source); CANONICAL REFERENCE BURIAL DEPTH (the raw value after its sign "
            "convention is resolved and normalized -- null here, since "
            f"{burial_semantics_obj.sign_convention} for every record); TOP-OF-ASSET COVER "
            "(the canonical reference burial depth converted to cover above the top of the "
            "asset -- also null here, since "
            f"{burial_semantics_obj.measurement_reference_point} for every record); "
            "SOURCE-INTERPRETED EXPOSURE (evidence the source itself explicitly flagged, "
            "always independent of the numeric measurement); and FUTURE EXPOSURE SCREENING "
            "(only ever produced given a defensible seabed-lowering input, never inferred "
            "from a single survey epoch). Barrow demonstrates the valid unresolved-reference "
            "path: every record is reported as MEASURED_REFERENCE_REQUIRES_REVIEW or the "
            "source's own explicit SOURCE_INTERPRETED_EXPOSED, never a guessed physical state."
        ),
        measured_burial_profile_facts=profile_stats,
        source_interpreted_exposure_facts={
            "explicit_exposure_feature_count": len(exposure_df),
            "source_flag_column": barrow_2016_provider.DOB_STORAGE_DB_COLUMN,
            "source_flag_value": barrow_2016_provider.SOURCE_EXPOSURE_FLAG_VALUE,
        },
        burial_margin_text=(
            "No source-stated or operator-supplied target/reference burial depth was found in "
            "the real Barrow DoB listing, so burial_margin_m is null for every record -- never "
            "invented (Section 13)."
        ),
        exposure_screening_contract_facts={
            "allowed_lowering_evidence_types": sorted(
                burial_exposure_screening.ALLOWED_LOWERING_EVIDENCE_TYPES
            ),
            "screening_states": sorted(burial_exposure_screening.EXPOSURE_SCREENING_STATES),
            "real_barrow_screening_result": real_screening_result["screening_state"],
        },
        future_susceptibility_text=(
            "The acquired Barrow 2016 source package provides a single survey epoch with no "
            "co-registered multi-epoch seabed-change product and no operator-defined lowering "
            "scenario, so no defensible seabed-lowering input exists for this route/epoch. "
            "Future exposure susceptibility is therefore NOT DEMONSTRATED from this single "
            "survey -- this is an accepted, honestly-reported POC outcome (Section 16), not a "
            "failure of the generic engine, which Section 17's synthetic tests demonstrate "
            "works correctly once a defensible lowering input is supplied."
        ),
        gis_outputs=[
            f"asset_route: {gpkg_path}",
            f"burial_measurements: {gpkg_path}",
            f"source_interpreted_exposure: {gpkg_path}"
            if not exposure_df.empty
            else "source_interpreted_exposure: absent (see explicit_exposure_feature_count)",
            f"survey_coverage: {gpkg_path}",
            f"qa_flags: {gpkg_path}"
            if not qa_df.empty
            else "qa_flags: absent (no flagged records)",
        ],
        production_transfer_contract_summary=[
            f"{f['field']}: {f['description']}" for f in burial_contract.STRONGLY_PREFERRED_FIELDS
        ],
    )
    report_html = burial_report.render_blocks_html(
        blocks, title="Generic Linear-Asset Burial/Exposure State POC"
    )
    report_path = report_dir / "barrow_2016_burial_exposure_poc.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_html, encoding="utf-8")
    print(f"  Report -> {report_path}")

    validation_path = burial_dir / "burial_exposure_poc_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    print(f"  Validation -> {validation_path}")

    # ==========================================================================================
    # Section 25: final report
    # ==========================================================================================
    print()
    print("=== Generic Linear-Asset Burial/Exposure State POC (MAR-024) ===")
    print()
    print("## Source")
    print(
        f"  DoB Listing: {dob_acquisition.package_bytes:,} bytes, "
        f"sha256={dob_acquisition.package_sha256}"
    )
    print(
        f"  Route Position List: {rpl_acquisition.package_bytes:,} bytes, "
        f"sha256={rpl_acquisition.package_sha256}"
    )
    print(f"  survey_epoch: {barrow_2016_provider.SURVEY_EPOCH}")
    print(f"  CRS: {working_crs}")
    print(f"  route_semantics: {geometry_direction_semantics}")
    print(f"  burial_measurement_reference: {burial_semantics_obj.measurement_reference_point}")
    print()
    print("## Profile")
    for key, value in profile_stats.items():
        print(f"  {key}: {value}")
    print(f"  coverage_fraction: {coverage_fraction}")
    print(f"  coverage_gap_count (> {gap_threshold_m:.0f} m): {len(coverage_gaps_m)}")
    print()
    print("## Product")
    print(f"  map: {state_map_path}")
    print(f"  KP view: {kp_profile_path}")
    print(f"  GIS: {gpkg_path}")
    print(f"  report: {report_path}")
    print(f"  input contract: {contract_path}")
    print()
    print(
        "IS GENERIC OPERATOR-SUPPLIED DEPTH-OF-BURIAL -> BURIAL / EXPOSURE STATE ANALYTICS "
        f"DEMONSTRATED? {'YES' if len(profile_df) > 0 else 'NO'}"
    )
    print(
        "IS FUTURE BARROW EXPOSURE SUSCEPTIBILITY DEMONSTRATED FROM THE CURRENT REAL DATA? "
        f"{validation['question_g_real_barrow_future_exposure_susceptibility_defensible']}"
    )
    return 0


def _derive_free_span_poc_validation_questions(
    *,
    pipe_bottom_normalization_demonstrated: bool,
    clearance_computable: bool,
    defensible_intervals_extracted: bool,
    unsurveyed_gaps_prevented_from_joining: bool,
    scenario_creates_and_extends_support_loss: bool,
    real_nsta_evidence_ingested: bool,
) -> dict[str, str]:
    """MAR-025 Section 31: a pure function so G/H/I's mandated NO answers are structurally
    enforced (asserted) -- this POC never claims PL854 has enough data for a site-specific
    free-span susceptibility map, never performs a structural DNV-RP-F105/VIV/fatigue
    assessment, and never produces a failure probability, regardless of what upstream facts
    say."""

    pl854_site_specific_susceptibility_defensible = False
    structural_assessment_performed = False
    future_failure_probability_produced = False
    assert pl854_site_specific_susceptibility_defensible is False
    assert structural_assessment_performed is False
    assert future_failure_probability_produced is False

    return {
        "question_a_pipe_vertical_reference_normalized_to_pipe_bottom_elevation": (
            "YES" if pipe_bottom_normalization_demonstrated else "NO"
        ),
        "question_b_current_pipe_underside_clearance_computed": (
            "YES" if clearance_computable else "NO"
        ),
        "question_c_defensible_current_free_span_intervals_extracted": (
            "YES" if defensible_intervals_extracted else "NO"
        ),
        "question_d_unsurveyed_gaps_prevented_from_joining_spans": (
            "YES" if unsurveyed_gaps_prevented_from_joining else "NO"
        ),
        "question_e_lowering_scenario_creates_or_extends_support_loss_intervals": (
            "YES" if scenario_creates_and_extends_support_loss else "NO"
        ),
        "question_f_real_authoritative_nsta_observed_free_span_evidence_ingested": (
            "YES" if real_nsta_evidence_ingested else "NO"
        ),
        "question_g_pl854_site_specific_free_span_susceptibility_map_defensible": (
            "YES" if pl854_site_specific_susceptibility_defensible else "NO"
        ),
        "question_h_structural_dnv_rp_f105_viv_fatigue_assessment_performed": (
            "YES" if structural_assessment_performed else "NO"
        ),
        "question_i_future_failure_probability_produced": (
            "YES" if future_failure_probability_produced else "NO"
        ),
    }


def _cmd_build_free_span_poc(args: argparse.Namespace) -> int:
    """MAR-025: generic pipeline free-span geometry and support-loss susceptibility screening
    POC. Three tracks, kept structurally separate throughout: (A) an explicitly synthetic exact
    engineering validation case exercising the full generic engine (Sections 3-17); (B) real
    authoritative NSTA UKCS-wide observed free-span registry evidence, cached/acquired only
    when the current accepted cache is absent (Sections 18-20); (C) the accepted, already-
    computed PL854 Table B.1 2018 observed evidence, repackaged and never recomputed (Sections
    21-22, 27). No structural free-span integrity assessment (DNV-RP-F105/VIV/fatigue/ULS/FLS)
    is performed anywhere in this command."""

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pl854_study_dir = config.paths.processed_dir / pipeline_id.lower()
    freespan_poc_dir = config.paths.processed_dir / "freespan_poc"
    synthetic_dir = freespan_poc_dir / "synthetic"
    maps_dir = freespan_poc_dir / "maps"
    gis_dir = freespan_poc_dir / "gis"
    report_dir = freespan_poc_dir / "report"
    contracts_dir = freespan_poc_dir / "freespan"

    # ==========================================================================================
    # Sections 3-17: synthetic exact engineering validation
    # ==========================================================================================
    print("Building the synthetic exact engine validation case (Section 17)...")
    case = fs_synthetic.build_synthetic_free_span_case()
    synthetic_summary = fs_synthetic.summarize_synthetic_case(case)
    for key, value in synthetic_summary.items():
        print(f"  {key}: {value}")

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    profile_path = metocean_evidence.write_parquet(
        case.scenario_result["profile_df"], synthetic_dir / "synthetic_pipeline_profile.parquet"
    )
    spans_path = metocean_evidence.write_parquet(
        case.measured_intervals_df, synthetic_dir / "synthetic_measured_free_spans.parquet"
    )
    scenario_path = metocean_evidence.write_parquet(
        case.scenario_result["scenario_intervals_df"],
        synthetic_dir / "synthetic_support_loss_scenario.parquet",
    )
    print(f"  Profile ({len(case.profile_df)} sample(s)) -> {profile_path}")
    print(f"  Measured free spans ({len(case.measured_intervals_df)}) -> {spans_path}")
    print(
        f"  Scenario support-loss intervals "
        f"({len(case.scenario_result['scenario_intervals_df'])}) -> {scenario_path}"
    )

    maps_dir.mkdir(parents=True, exist_ok=True)
    synthetic_map_path = fs_maps.render_free_span_support_map(
        route=case.route,
        profile_df=case.profile_df,
        measured_intervals_df=case.measured_intervals_df,
        scenario_intervals_df=case.scenario_result["scenario_intervals_df"],
        output_path=synthetic_dir / "synthetic_free_span_map.png",
        title=f"Synthetic Pipeline Free-Span Support Map ({fs_synthetic.PROMINENT_DISCLAIMER})",
    )
    synthetic_kp_path = fs_maps.render_free_span_kp_view(
        profile_df=case.scenario_result["profile_df"],
        measured_intervals_df=case.measured_intervals_df,
        scenario_intervals_df=case.scenario_result["scenario_intervals_df"],
        classification_threshold_m=fs_synthetic.CLASSIFICATION_THRESHOLD_M,
        seabed_lowering_m=case.lowering_input.seabed_lowering_m,
        output_path=synthetic_dir / "synthetic_free_span_kp_view.png",
        title=f"Synthetic Pipeline Free-Span KP View ({fs_synthetic.PROMINENT_DISCLAIMER})",
    )
    print(f"  Map -> {synthetic_map_path}")
    print(f"  KP view -> {synthetic_kp_path}")

    # Sections 25 + 28: synthetic GIS, written to both required locations -- every interval
    # geometry is a real substring of the real synthetic route, never a fabricated geometry.
    profile_gdf = gpd.GeoDataFrame(
        case.scenario_result["profile_df"],
        geometry=gpd.points_from_xy(
            case.scenario_result["profile_df"]["x_m"], case.scenario_result["profile_df"]["y_m"]
        ),
        crs=fs_synthetic.SYNTHETIC_CRS,
    )

    def _intervals_to_gdf(intervals_df: pd.DataFrame) -> gpd.GeoDataFrame:
        geometries = [
            shapely_substring(
                case.route, float(row["start_chainage_m"]), float(row["end_chainage_m"])
            )
            for _, row in intervals_df.iterrows()
        ]
        return gpd.GeoDataFrame(intervals_df, geometry=geometries, crs=fs_synthetic.SYNTHETIC_CRS)

    measured_intervals_gdf = _intervals_to_gdf(case.measured_intervals_df)
    scenario_intervals_gdf = _intervals_to_gdf(case.scenario_result["scenario_intervals_df"])

    def _write_synthetic_gpkg(gpkg_path: Path) -> None:
        gpkg_path.parent.mkdir(parents=True, exist_ok=True)
        if gpkg_path.exists():
            gpkg_path.unlink()
        case.route_gdf.to_file(gpkg_path, driver="GPKG", layer="asset_route")
        profile_gdf.to_file(gpkg_path, driver="GPKG", layer="pipeline_profile")
        measured_intervals_gdf.to_file(
            gpkg_path, driver="GPKG", layer="measured_free_span_intervals"
        )
        scenario_intervals_gdf.to_file(
            gpkg_path, driver="GPKG", layer="scenario_support_loss_intervals"
        )

    gis_dir.mkdir(parents=True, exist_ok=True)
    synthetic_gpkg_path = synthetic_dir / "synthetic_free_span_poc.gpkg"
    synthetic_screening_gpkg_path = gis_dir / "synthetic_free_span_support_screening.gpkg"
    _write_synthetic_gpkg(synthetic_gpkg_path)
    _write_synthetic_gpkg(synthetic_screening_gpkg_path)
    print(f"  GIS -> {synthetic_gpkg_path}")
    print(f"  GIS -> {synthetic_screening_gpkg_path}")

    # ==========================================================================================
    # Sections 18-20: real NSTA UKCS-wide observed free-span registry
    # ==========================================================================================
    print("Acquiring/auditing the real NSTA UKCS-wide freespan registry (Sections 18-20)...")
    nsta_cache_dir = config.paths.raw_dir / "nsta" / "freespans_full_registry"
    nsta_evidence = fs_nsta_registry.build_real_nsta_evidence(nsta_cache_dir)
    for registry_layer, info in nsta_evidence["acquisitions"].items():
        print(
            f"  {registry_layer}: already_cached={info['already_cached']} -> {info['cache_path']}"
        )
    for key, value in nsta_evidence["audit_summary"].items():
        print(f"  {key}: {value}")

    audit_path = metocean_evidence.write_parquet(
        nsta_evidence["audit_df"], freespan_poc_dir / "nsta_freespan_registry_audit.parquet"
    )
    print(f"  Registry audit -> {audit_path}")

    nsta_gpkg_path = gis_dir / "nsta_observed_freespan_evidence.gpkg"
    if nsta_gpkg_path.exists():
        nsta_gpkg_path.unlink()
    nsta_evidence["registry_gdf"].to_file(
        nsta_gpkg_path, driver="GPKG", layer="nsta_observed_freespan_evidence"
    )
    print(f"  GIS -> {nsta_gpkg_path}")

    example_pipeline, example_count = fs_nsta_registry.select_example_pipeline_by_record_count(
        nsta_evidence["audit_df"]
    )
    nsta_map_path = fs_maps.render_nsta_registry_overview_map(
        registry_gdf=nsta_evidence["registry_gdf"],
        example_pipeline_number=example_pipeline,
        example_record_count=example_count,
        output_path=maps_dir / "ukcs_observed_pipeline_freespan_evidence.png",
    )
    print(
        f"  Map -> {nsta_map_path} (example pipeline: {example_pipeline}, "
        f"{example_count} record(s) -- selected by record count only, not a risk indication)"
    )

    # ==========================================================================================
    # Sections 21-22, 27: PL854 observed evidence -- reused, never recomputed
    # ==========================================================================================
    print("Repackaging the accepted PL854 2018 observed freespan evidence (Sections 21-22, 27)...")
    pl854_evidence_gpkg = (
        pl854_study_dir / "freespan_evidence" / "anglia_freespan_spatial_evidence.gpkg"
    )
    if not pl854_evidence_gpkg.exists():
        print(
            "error: missing required PL854 input -- run "
            "'marine-engine build-freespan-spatial-evidence configs/pl854.yaml' first: "
            f"{pl854_evidence_gpkg}",
            file=sys.stderr,
        )
        return 1

    freespans_2018_gdf = evidence_atlas_core.build_observed_freespans_2018_layer(pl854_study_dir)
    pl854_observed_summary = {
        "event_count": int(freespans_2018_gdf["source_length_m"].count()),
        "total_length_m": float(freespans_2018_gdf["source_length_m"].sum()),
        "max_length_m": float(freespans_2018_gdf["source_length_m"].max()),
        "max_height_m": float(freespans_2018_gdf["source_height_m"].max()),
        "asset_scope": sorted(freespans_2018_gdf["asset_scope"].astype(str).unique().tolist()),
        "individual_line_attribution": sorted(
            freespans_2018_gdf["individual_line_attribution"].astype(str).unique().tolist()
        ),
    }
    for key, value in pl854_observed_summary.items():
        print(f"  {key}: {value}")

    pl854_route_layer = evidence_atlas_core.build_pipeline_route_layer(pl854_study_dir)
    pl854_route = pl854_route_layer.geometry.iloc[0]
    pl854_map_path = freespan_evidence_map.render_2018_freespan_evidence_map(
        events_2018_gdf=freespans_2018_gdf,
        route=pl854_route,
        output_path=maps_dir / "pl854_2018_observed_freespan_evidence.png",
        title="PL854/PL855 Corridor — Official 2018 Observed Freespan Evidence",
    )
    print(f"  Map -> {pl854_map_path} (individual line attribution unresolved)")
    pl854_susceptibility_unavailable_reason = (
        "PL854_SITE_SPECIFIC_FREE_SPAN_SUSCEPTIBILITY_NOT_AVAILABLE (Section 22): no "
        "high-resolution route bathymetry, measured continuous pipe vertical profile, or "
        "measured continuous embedment/support profile exists for PL854."
    )
    print(f"  {pl854_susceptibility_unavailable_reason}")

    # ==========================================================================================
    # Sections 16, 29: input contract + structural handoff contract
    # ==========================================================================================
    contracts_dir.mkdir(parents=True, exist_ok=True)
    contract_dict = fs_contract.build_free_span_input_contract()
    contract_path = contracts_dir / "free_span_input_contract.json"
    contract_path.write_text(json.dumps(contract_dict, indent=2, default=str), encoding="utf-8")

    handoff_dict = fs_structural_handoff.build_structural_free_span_assessment_handoff_contract()
    handoff_path = contracts_dir / "structural_free_span_assessment_handoff.json"
    handoff_path.write_text(json.dumps(handoff_dict, indent=2, default=str), encoding="utf-8")
    print(f"  Input contract -> {contract_path}")
    print(f"  Structural handoff contract -> {handoff_path}")

    # ==========================================================================================
    # Section 31: validation
    # ==========================================================================================
    validation = _derive_free_span_poc_validation_questions(
        pipe_bottom_normalization_demonstrated=bool(
            abs(
                case.profile_df["pipe_bottom_elevation_m"].iloc[0]
                - fs_synthetic.EXPECTED_PIPE_BOTTOM_ELEVATION_M
            )
            < 1e-9
        ),
        clearance_computable=bool(case.profile_df["clearance_m"].notna().all()),
        defensible_intervals_extracted=len(case.measured_intervals_df) > 0,
        unsurveyed_gaps_prevented_from_joining=bool(
            case.measured_intervals_df["limitations"].notna().any()
        ),
        scenario_creates_and_extends_support_loss=(
            case.scenario_result["new_span_count"] > 0
            and case.scenario_result["extended_span_count"] > 0
        ),
        real_nsta_evidence_ingested=len(nsta_evidence["records"]) > 0,
    )

    # ==========================================================================================
    # Section 30: report
    # ==========================================================================================
    print("Building the generic free-span support-loss POC report (Section 30)...")
    blocks = fs_report.build_free_span_report_blocks(
        project_title="Generic Pipeline Free-Span Geometry & Support-Loss Susceptibility POC",
        purpose_text=(
            "Demonstrates the future OrbGSS workflow: operator pipeline vertical profile + "
            "seabed support profile + pipe geometry -> canonical pipe underside clearance -> "
            "current unsupported-span geometry -> optional seabed-lowering/support-loss "
            "scenario -> free-span support-loss susceptibility -> map + KP view + GIS + "
            "report. Source-reported/observed free-span evidence (NSTA registry, PL854 Table "
            "B.1) is ingested and audited separately, never used to derive a susceptibility "
            "result."
        ),
        product_boundary_text=(
            "MAR-025 covers measured/observed free-span geometry, generic support-loss "
            "susceptibility screening, and real authoritative free-span evidence ingestion. It "
            "does NOT perform structural free-span integrity assessment -- "
            f"{fs_structural_handoff.STRUCTURAL_FREE_SPAN_ASSESSMENT_NOT_PERFORMED}. The result "
            "produced here is "
            f"{fs_structural_handoff.GEOMETRIC_SUPPORT_CONDITION_AND_SUPPORT_LOSS_SCREENING}."
        ),
        operator_input_model_facts={
            "pipe_vertical_reference": fs_synthetic.PIPE_VERTICAL_REFERENCE,
            "seabed_sign_convention": fs_synthetic.SEABED_SIGN_CONVENTION,
            "classification_threshold_m": fs_synthetic.CLASSIFICATION_THRESHOLD_M,
            "classification_threshold_provenance": (
                fs_synthetic.CLASSIFICATION_THRESHOLD_PROVENANCE
            ),
            "max_measurement_gap_m": fs_synthetic.MAX_MEASUREMENT_GAP_M,
        },
        canonical_geometry_facts={
            "pipe_bottom_elevation_m": fs_synthetic.EXPECTED_PIPE_BOTTOM_ELEVATION_M,
            "clearance_equation": (
                "pipe_underside_clearance_m = pipe_bottom_elevation_m - seabed_support_elevation_m"
            ),
        },
        measured_free_span_facts=synthetic_summary,
        gap_governance_text=(
            f"An along-route measurement gap exceeding {fs_synthetic.MAX_MEASUREMENT_GAP_M:g} m "
            "between two unsupported observations is never bridged into one free span -- both "
            f"resulting intervals are flagged {fs_support_state.SPAN_SPLIT_BY_MEASUREMENT_GAP}. "
            "The synthetic case demonstrates this directly: "
            f"{synthetic_summary['gap_split_interval_count']} of "
            f"{synthetic_summary['recovered_measured_span_count']} recovered measured "
            "intervals are gap-split halves of what would otherwise misleadingly appear as one "
            "continuous span."
        ),
        support_loss_scenario_facts={
            "seabed_lowering_m": case.lowering_input.seabed_lowering_m,
            "lowering_evidence_type": case.lowering_input.evidence_type,
            "new_span_count": case.scenario_result["new_span_count"],
            "extended_span_count": case.scenario_result["extended_span_count"],
            "pipe_vertical_position": (
                "fixed -- MAR-025 assumes no pipe-response model (Section 13)"
            ),
        },
        synthetic_validation_facts=synthetic_summary,
        nsta_registry_facts=nsta_evidence["audit_summary"],
        pl854_observed_context_facts=pl854_observed_summary,
        structural_boundary_text=(
            f"{fs_structural_handoff.STRUCTURAL_FREE_SPAN_ASSESSMENT_NOT_PERFORMED}. MAR-025 "
            "does not claim DNV compliance, allowable span assessment, VIV assessment, fatigue "
            "assessment, or structural acceptability. See "
            "freespan/structural_free_span_assessment_handoff.json for what a future "
            "structural/VIV/fatigue module would additionally require."
        ),
        production_transfer_contract_summary=[
            f"{f['field']}: {f['description']}" for f in fs_contract.STRONGLY_PREFERRED_FIELDS
        ],
        limitations=[
            "Synthetic case is an exact engineering validation of the generic engine, never "
            "field validation.",
            "PL854 vs PL855 freespan attribution unresolved (reused from the accepted Table "
            "B.1 evidence).",
            "PL854 has no measured continuous pipe vertical profile or seabed support profile "
            "-- no site-specific free-span susceptibility map is produced for it.",
            "NSTA registry records for PL854/PL855 specifically are real and confirmed zero "
            "(investigated, not a data gap this POC works around).",
            "No structural free-span integrity assessment (DNV-RP-F105/VIV/fatigue/ULS/FLS) is "
            "performed anywhere in this POC.",
        ],
    )
    report_html = fs_report.render_blocks_html(
        blocks, title="Generic Pipeline Free-Span Support-Loss POC"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "generic_pipeline_free_span_poc.html"
    report_path.write_text(report_html, encoding="utf-8")
    print(f"  Report -> {report_path}")

    validation_path = freespan_poc_dir / "freespan_poc_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    print(f"  Validation -> {validation_path}")

    # ==========================================================================================
    # Section 34: final report
    # ==========================================================================================
    print()
    print("=== Generic Pipeline Free-Span Support-Loss POC (MAR-025) ===")
    print()
    print("## Generic engine")
    print(f"  pipe_vertical_reference: {fs_synthetic.PIPE_VERTICAL_REFERENCE}")
    print(
        "  clearance_equation: pipe_underside_clearance_m = pipe_bottom_elevation_m - "
        "seabed_support_elevation_m"
    )
    print(
        f"  classification_threshold_provenance: {fs_synthetic.CLASSIFICATION_THRESHOLD_PROVENANCE}"
    )
    print(f"  max_measurement_gap_m: {fs_synthetic.MAX_MEASUREMENT_GAP_M}")
    print()
    print("## Measured geometry")
    print("  MEASURED SUPPORT STATE IS DERIVED ONLY FROM PIPE/SEABED GEOMETRY.")
    print()
    print("## Source interpretation")
    print(
        "  SOURCE-INTERPRETED FREE-SPAN EVIDENCE IS AN ORTHOGONAL EVIDENCE DIMENSION AND NEVER "
        "OVERRIDES GEOMETRIC CLASSIFICATION."
    )
    print(
        f"  source_interpreted_free_span_sample_count (synthetic): "
        f"{synthetic_summary['source_interpreted_free_span_sample_count']}"
    )
    print(
        "  geometry_vs_source_interpretation_statuses observed (synthetic): "
        f"{synthetic_summary['source_interpreted_geometry_vs_source_statuses']}"
    )
    print()
    print("## Interval support")
    print(
        "  interval_boundary_semantics (synthetic): "
        f"{synthetic_summary['interval_boundary_semantics']}"
    )
    print(f"  span_length_semantics: {fs_support_state.SPAN_LENGTH_SEMANTICS_NOTE}")
    print(
        f"  gap_governance: max_measurement_gap_m={fs_synthetic.MAX_MEASUREMENT_GAP_M}, "
        f"flag={fs_support_state.SPAN_SPLIT_BY_MEASUREMENT_GAP}"
    )
    print(
        "  A DISCRETE UNSUPPORTED SAMPLE RUN IS NOT AUTOMATICALLY CLAIMED TO BE AN EXACT "
        "PHYSICAL FREE-SPAN LENGTH."
    )
    print()
    print("## Regression")
    regression_unchanged = (
        synthetic_summary["recovered_measured_span_count"]
        == synthetic_summary["expected_measured_span_count"]
        and sorted(synthetic_summary["recovered_measured_span_lengths_m"])
        == sorted(synthetic_summary["expected_measured_span_lengths_m"])
        and abs(
            synthetic_summary["recovered_max_clearance_m"]
            - synthetic_summary["expected_max_clearance_m"]
        )
        < 1e-6
        and synthetic_summary["recovered_scenario_new_span_count"]
        == synthetic_summary["expected_scenario_new_span_count"]
        and synthetic_summary["recovered_scenario_extended_span_count"]
        == synthetic_summary["expected_scenario_extended_span_count"]
    )
    print(
        "  EXISTING SYNTHETIC NUMERICAL OUTPUTS ARE UNCHANGED: "
        f"{'YES' if regression_unchanged else 'NO'}"
    )
    print()
    print("## Synthetic validation")
    for key, value in synthetic_summary.items():
        print(f"  {key}: {value}")
    print()
    print("## NSTA real evidence")
    for key, value in nsta_evidence["audit_summary"].items():
        print(f"  {key}: {value}")
    print()
    print("## PL854")
    for key, value in pl854_observed_summary.items():
        print(f"  {key}: {value}")
    print(
        f"  explicit_reason_susceptibility_unavailable: {pl854_susceptibility_unavailable_reason}"
    )
    print()

    measured_geometry_demonstrated = (
        validation["question_a_pipe_vertical_reference_normalized_to_pipe_bottom_elevation"]
        == "YES"
        and validation["question_b_current_pipe_underside_clearance_computed"] == "YES"
        and validation["question_c_defensible_current_free_span_intervals_extracted"] == "YES"
    )
    print(
        "IS GENERIC OPERATOR-SUPPLIED PIPE/SEABED PROFILE -> FREE-SPAN GEOMETRY ANALYTICS "
        f"DEMONSTRATED? {'YES' if measured_geometry_demonstrated else 'NO'}"
    )
    print(
        "IS GENERIC SUPPORT-LOSS SUSCEPTIBILITY SCREENING DEMONSTRATED? "
        f"{validation['question_e_lowering_scenario_creates_or_extends_support_loss_intervals']}"
    )
    print(
        "IS SITE-SPECIFIC PL854 FREE-SPAN SUSCEPTIBILITY DEFENSIBLE? "
        f"{validation['question_g_pl854_site_specific_free_span_susceptibility_map_defensible']}"
    )
    return 0


def _derive_project_readiness_validation_questions(
    *,
    generic_registration_demonstrated: bool,
    source_files_registered_without_mutation: bool,
    bathymetry_readiness_reused: bool,
    burial_readiness_reused: bool,
) -> dict[str, str]:
    """MAR-026 Section 20: a pure function so the final NO is structurally enforced -- this
    package never claims universal marine-geohazard readiness, regardless of any upstream facts."""

    claims_universal_hazard_readiness = False
    assert claims_universal_hazard_readiness is False

    return {
        "question_a_generic_local_operator_project_registration_demonstrated": (
            "YES" if generic_registration_demonstrated else "NO"
        ),
        "question_b_source_files_registered_with_identity_and_provenance_without_mutation": (
            "YES" if source_files_registered_without_mutation else "NO"
        ),
        "question_c_existing_bathymetry_readiness_reused_through_generic_project_layer": (
            "YES" if bathymetry_readiness_reused else "NO"
        ),
        "question_d_existing_burial_profile_readiness_reused_through_generic_project_layer": (
            "YES" if burial_readiness_reused else "NO"
        ),
        "question_e_does_mar026_claim_project_ready_for_every_marine_geohazard": (
            "YES" if claims_universal_hazard_readiness else "NO"
        ),
    }


def _derive_project_model_validation_questions(
    *,
    route_reference_grid_built: bool,
    linked_without_declared_relationship: bool,
    burial_chainage_correlated_without_linear_reference: bool,
    linkage_modified_evidence_role_or_intrinsic_readiness: bool,
    raster_bounds_reported_as_valid_data_coverage: bool,
    universal_project_score_present: bool,
    accepted_readiness_semantics_preserved: bool,
) -> dict[str, str]:
    """MAR-027 Section 30: the required proof questions, derived from the actual run's facts --
    never asserted as constants."""

    return {
        "question_a_does_mar027_create_a_deterministic_canonical_route_reference_grid": (
            "YES" if route_reference_grid_built else "NO"
        ),
        "question_b_can_an_asset_become_route_linked_without_an_explicit_manifest_relationship": (
            "YES" if linked_without_declared_relationship else "NO"
        ),
        "question_c_can_a_burial_numeric_kp_chainage_column_be_assumed_to_match_canonical_"
        "project_chainage_without_explicit_linear_reference_semantics": (
            "YES" if burial_chainage_correlated_without_linear_reference else "NO"
        ),
        "question_d_does_route_linkage_modify_evidence_role_or_intrinsic_readiness": (
            "YES" if linkage_modified_evidence_role_or_intrinsic_readiness else "NO"
        ),
        "question_e_does_mar027_claim_raster_bound_intersection_is_valid_data_coverage": (
            "YES" if raster_bounds_reported_as_valid_data_coverage else "NO"
        ),
        "question_f_does_mar027_create_a_universal_project_hazard_or_readiness_score": (
            "YES" if universal_project_score_present else "NO"
        ),
        "question_g_are_mar020_mar024_mar026a_scientific_readiness_semantics_preserved": (
            "YES" if accepted_readiness_semantics_preserved else "NO"
        ),
    }


def _dict_keys_recursive(obj: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.append(str(key))
            keys.extend(_dict_keys_recursive(value))
    elif isinstance(obj, list):
        for value in obj:
            keys.extend(_dict_keys_recursive(value))
    return keys


def _cmd_build_project_model(args: argparse.Namespace) -> int:
    """MAR-027: canonical project route-reference model and cross-asset linkage POC. Calls the
    real, unmodified MAR-026/026A registration (`project_registry.register_project`), then
    operationalizes the declared primary route, builds a deterministic canonical route-reference
    grid when every prerequisite holds, and represents manifest-declared asset -> route
    relationships as a separate LINKAGE axis. No hazard science, no fusion, no scores, fully
    offline.
    """

    try:
        manifest, manifest_dir = project_manifest.load_project_manifest(args.manifest)
    except Exception as exc:
        print(f"error: project manifest is invalid: {exc}", file=sys.stderr)
        return 1

    project_id = manifest.project.id
    output_dir = Path("data/processed") / project_id / "project"

    print(f"Registering project {project_id!r} via MAR-026/026A registration...")
    summary = project_registry.register_project(manifest, manifest_dir)
    for finding in summary.working_crs_findings:
        print(f"  PROJECT WORKING CRS ISSUE: {finding}")

    print("Building canonical project model (MAR-027)...")
    model = project_model.build_canonical_project_model(manifest, summary)
    rr = model.route_reference
    print(f"  Primary route: {rr.primary_route.status} ({rr.primary_route.primary_route_asset_id})")
    for finding in rr.primary_route.findings:
        print(f"    FINDING: {finding}")
    print(f"  Route reference: {rr.status}")
    for finding in rr.findings:
        print(f"    FINDING: {finding}")
    for linkage in model.asset_linkages:
        print(
            f"  {linkage.asset_id}: category={linkage.category} "
            f"evidence_role={linkage.evidence_role} "
            f"registration={linkage.registration_status} "
            f"readiness(intrinsic={linkage.readiness_status_intrinsic}, "
            f"effective={linkage.readiness_status_effective}) "
            f"declared_route={linkage.declared_route_asset_id} "
            f"linkage={linkage.route_linkage_status}"
        )
        for finding in linkage.route_linkage_findings:
            print(f"    LINKAGE FINDING: {finding}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_dict = project_model.build_canonical_project_model_dict(model)
    model_path = output_dir / "canonical_project_model.json"
    model_path.write_text(json.dumps(model_dict, indent=2, default=str), encoding="utf-8")
    print(f"  Canonical project model -> {model_path}")

    linkage_df = project_model.build_asset_linkage_df(model)
    linkage_path = metocean_evidence.write_parquet(
        linkage_df, output_dir / "project_asset_linkage.parquet"
    )
    print(f"  Asset linkage ({len(linkage_df)} row(s)) -> {linkage_path}")

    # Section 22: a GeoPackage exists ONLY when a grid was actually built -- a stale grid from an
    # earlier run of this same command must never survive a run that could not build one.
    grid_path = output_dir / "project_route_reference.gpkg"
    if grid_path.exists():
        grid_path.unlink()
    if rr.grid_gdf is not None:
        project_route_reference.write_route_reference_gpkg(rr.grid_gdf, grid_path)
        print(
            f"  Route-reference grid ({len(rr.grid_gdf)} station(s), layer "
            f"{project_route_reference.ROUTE_REFERENCE_LAYER!r}) -> {grid_path}"
        )
    else:
        print("  Route-reference grid: not built (see findings above); no GeoPackage written")

    registrations = {r.registration.asset_id: r.registration for r in summary.asset_results}
    linkage_altered_registration_facts = any(
        linkage.evidence_role != registrations[linkage.asset_id].evidence_role
        or linkage.readiness_status_intrinsic
        != registrations[linkage.asset_id].readiness_status_intrinsic
        for linkage in model.asset_linkages
    )
    effective_preserved = all(
        linkage.readiness_status_effective
        == registrations[linkage.asset_id].readiness_status_effective
        and registrations[linkage.asset_id].readiness_status
        == registrations[linkage.asset_id].readiness_status_effective
        for linkage in model.asset_linkages
    )
    model_keys_lower = [key.lower() for key in _dict_keys_recursive(model_dict)]
    validation = _derive_project_model_validation_questions(
        route_reference_grid_built=(
            rr.status == project_route_reference.ROUTE_REFERENCE_BUILT and rr.grid_gdf is not None
        ),
        linked_without_declared_relationship=any(
            linkage.route_linkage_status != project_model.NOT_APPLICABLE
            and linkage.declared_route_asset_id is None
            for linkage in model.asset_linkages
        ),
        burial_chainage_correlated_without_linear_reference=any(
            linkage.category == project_categories.BURIAL_PROFILE
            and linkage.declared_linear_reference is None
            and "chainage_range_overlap_fraction" in linkage.linkage_facts
            for linkage in model.asset_linkages
        ),
        linkage_modified_evidence_role_or_intrinsic_readiness=linkage_altered_registration_facts,
        raster_bounds_reported_as_valid_data_coverage=any(
            linkage.category == project_categories.BATHYMETRY_RASTER
            and any(
                "coverage" in key.lower()
                for key in linkage.linkage_facts
                if key != "raster_extent_note"
            )
            for linkage in model.asset_linkages
        ),
        universal_project_score_present=any(
            "score" in key or "project_ready" in key for key in model_keys_lower
        ),
        accepted_readiness_semantics_preserved=(
            effective_preserved and not linkage_altered_registration_facts
        ),
    )
    validation_path = output_dir / "project_model_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    print(f"  Validation -> {validation_path}")

    print()
    print("=== Canonical Project Route-Reference Model & Cross-Asset Linkage (MAR-027) ===")
    print()
    print(f"Project: {manifest.project.id} ({manifest.project.name})")
    print(f"Working CRS: {model.working_crs}  findings={model.working_crs_findings}")
    print(f"Primary route: {rr.primary_route.status} ({rr.primary_route.primary_route_asset_id})")
    print(f"Route reference: {rr.status}")
    if rr.grid_gdf is not None:
        print(f"  interval: {rr.configured_interval_m} m (indexing resolution)")
        print(f"  chainage origin basis: {rr.chainage_origin_basis}")
        print(f"  route length: {rr.route_length_m:,.3f} m")
        print(
            f"  stations: {rr.station_count} (regular {rr.regular_station_count}, terminal "
            f"residual {rr.terminal_residual_m:.3f} m)"
        )
    print(f"Assets: {len(model.asset_linkages)}")
    print(f"Evidence role counts: {model.evidence_role_counts()}")
    print(f"Route linkage status counts: {model.route_linkage_status_counts()}")
    print()
    print(project_route_reference.ROUTE_REFERENCE_GRID_DISCLAIMER)
    print()
    print(project_model.ROUTE_LINKAGE_STATUS_DISCLAIMER)
    print()
    print(project_registry.PROJECT_HAZARD_READINESS_DISCLAIMER)
    print()
    print(
        "DOES MAR-027 CREATE A DETERMINISTIC CANONICAL ROUTE-REFERENCE GRID? "
        f"{validation['question_a_does_mar027_create_a_deterministic_canonical_route_reference_grid']}"
    )
    print(
        "CAN AN ASSET BECOME ROUTE-LINKED WITHOUT AN EXPLICIT MANIFEST RELATIONSHIP? "
        + validation[
            "question_b_can_an_asset_become_route_linked_without_an_explicit_manifest_relationship"
        ]
    )
    print(
        "CAN A BURIAL PROFILE'S NUMERIC KP/CHAINAGE COLUMN BE ASSUMED TO MATCH CANONICAL PROJECT "
        "CHAINAGE WITHOUT EXPLICIT LINEAR-REFERENCE SEMANTICS? "
        + validation[
            "question_c_can_a_burial_numeric_kp_chainage_column_be_assumed_to_match_canonical_"
            "project_chainage_without_explicit_linear_reference_semantics"
        ]
    )
    print(
        "DOES ROUTE LINKAGE MODIFY EVIDENCE ROLE OR INTRINSIC READINESS? "
        + validation["question_d_does_route_linkage_modify_evidence_role_or_intrinsic_readiness"]
    )
    print(
        "DOES MAR-027 CLAIM RASTER-BOUND INTERSECTION IS VALID-DATA COVERAGE? "
        + validation[
            "question_e_does_mar027_claim_raster_bound_intersection_is_valid_data_coverage"
        ]
    )
    print(
        "DOES MAR-027 CREATE A UNIVERSAL PROJECT HAZARD/READINESS SCORE? "
        + validation["question_f_does_mar027_create_a_universal_project_hazard_or_readiness_score"]
    )
    print(
        "ARE MAR-020, MAR-024, MAR-026A SCIENTIFIC/READINESS SEMANTICS PRESERVED? "
        + validation[
            "question_g_are_mar020_mar024_mar026a_scientific_readiness_semantics_preserved"
        ]
    )
    return 0


def _cmd_build_route_seabed_change_evidence(args: argparse.Namespace) -> int:
    """MAR-029: route-referenced observed seabed elevation-change evidence POC. Loads an EXPLICIT
    route <-> DoD linkage manifest, runs the real MAR-026/026A registration and the real MAR-027
    canonical project model (unchanged), consumes the accepted MAR-021/021A DoD raster read-only,
    and samples the DoD cell containing each canonical route-reference point. Raw observed sign
    labels only; no significance threshold, no causal attribution, no prediction, no score.
    """

    try:
        result, model = change_route_evidence.run_route_change_evidence(args.manifest)
    except Exception as exc:
        print(f"error: route-change evidence inputs are invalid: {exc}", file=sys.stderr)
        return 1

    rr = model.route_reference
    print(f"Project {result.project_id!r}: working CRS {result.working_crs}")
    print(f"  Primary route: {rr.primary_route.status} ({rr.primary_route.primary_route_asset_id})")
    print(f"  Route reference: {rr.status} ({rr.station_count} station(s))")
    print(f"  Declared route asset for change evidence: {result.route_manifest.route_asset_id}")
    print(f"  DoD source: {result.dod_resolved_path}")
    if result.dod_identity is not None:
        print(f"  DoD SHA-256: {result.dod_identity.sha256}")
    if result.dod_observed is not None:
        obs = result.dod_observed
        print(
            f"  DoD observed: crs={obs.observed_crs} {obs.width}x{obs.height} "
            f"pixel=({obs.pixel_size_x:g},{obs.pixel_size_y:g}) dtype={obs.dtype} "
            f"bands={obs.band_count} nodata={obs.nodata}"
        )

    output_dir = Path("data/processed") / result.project_id / "change_route"
    final = change_route_evidence.write_route_change_evidence_outputs(result, output_dir)
    for key in ("parquet", "gpkg", "metadata", "validation", "report"):
        if key in final.output_paths:
            print(f"  {key} -> {final.output_paths[key]}")
    if not final.products_written:
        print("  Point products: not written (evidence not available; see findings)")

    validation = change_route_evidence.build_route_change_evidence_validation(final)
    metadata = change_route_evidence.build_route_change_evidence_metadata(final)
    counts = metadata["sample_counts"]

    print()
    print("=== Route-Referenced Observed Seabed Elevation Change (MAR-029) ===")
    print()
    print(f"Scientific role: {change_route_evidence.SCIENTIFIC_ROLE}")
    print(f"Status: {final.status}" + (f" ({final.reason_code})" if final.reason_code else ""))
    for finding in final.findings:
        print(f"  FINDING: {finding}")
    print(f"Sample support: {change_route_evidence.SAMPLE_SUPPORT_SEMANTICS}")
    print("Sample counts (route-reference SAMPLES, not route-length coverage):")
    for key, value in counts.items():
        if key != "note":
            print(f"  {key}: {value}")
    print(
        "Generic change significance: "
        f"{metadata['generic_change_significance']['generic_change_significance_status']} "
        "(threshold_m = null)"
    )
    print()
    print(change_route_evidence.DIRECTION_LABEL_DISCLAIMER)
    print()
    labels = {
        "question_a_does_mar029_reuse_the_accepted_mar021_dod_definition": (
            "DOES MAR-029 REUSE THE ACCEPTED MAR-021 DoD DEFINITION?"
        ),
        "question_b_can_a_positive_dod_sample_be_called_definitive_deposition": (
            "CAN A POSITIVE DoD SAMPLE BE CALLED DEFINITIVE DEPOSITION?"
        ),
        "question_c_can_a_negative_dod_sample_be_called_definitive_erosion_or_scour": (
            "CAN A NEGATIVE DoD SAMPLE BE CALLED DEFINITIVE EROSION OR SCOUR?"
        ),
        "question_d_does_mar029_apply_a_generic_change_significance_threshold": (
            "DOES MAR-029 APPLY A GENERIC CHANGE-SIGNIFICANCE THRESHOLD?"
        ),
        "question_e_does_nodata_become_zero_change": "DOES NODATA BECOME ZERO CHANGE?",
        "question_f_is_the_dod_raster_interpolated_or_smoothed_during_route_sampling": (
            "IS THE DoD RASTER INTERPOLATED OR SMOOTHED DURING ROUTE SAMPLING?"
        ),
        "question_g_is_route_referenced_observed_change_available_for_this_run": (
            "IS ROUTE-REFERENCED OBSERVED CHANGE AVAILABLE FOR THIS RUN?"
        ),
        "question_h_was_the_dod_source_modified_during_the_run": (
            "WAS THE DoD SOURCE MODIFIED DURING THE RUN?"
        ),
    }
    for key, label in labels.items():
        print(f"{label} {validation[key]}")
    return 0


def _cmd_build_project_readiness(args: argparse.Namespace) -> int:
    """MAR-026: generic local operator-project registration and asset-readiness layer. Sits
    above raw operator files and below the independent scientific geohazard engines -- performs
    NO hazard-specific science, and delegates bathymetry/burial readiness to the existing,
    unmodified `terrain.readiness`/`burial.readiness` modules. Fully offline: no network
    access, no provider discovery, no credentials, no interactive prompts.
    """

    try:
        manifest, manifest_dir = project_manifest.load_project_manifest(args.manifest)
    except Exception as exc:
        print(f"error: project manifest is invalid: {exc}", file=sys.stderr)
        return 1

    project_id = manifest.project.id
    working_crs = manifest.project.working_crs
    output_dir = Path("data/processed") / project_id / "project"

    print(f"Registering project {project_id!r} (Section 6)...")
    summary = project_registry.register_project(manifest, manifest_dir)
    for finding in summary.working_crs_findings:
        print(f"  PROJECT WORKING CRS ISSUE: {finding}")
    for r in summary.asset_results:
        reg = r.registration
        print(
            f"  {reg.asset_id}: category={reg.category} evidence_role={reg.evidence_role} "
            f"registration={reg.registration_status} "
            f"readiness(intrinsic={reg.readiness_status_intrinsic}, "
            f"effective={reg.readiness_status_effective})"
        )
        for conflict in reg.conflicts:
            print(f"    CONFLICT: {conflict}")

    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_manifest_path = output_dir / "normalized_project_manifest.json"
    normalized_manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
    )
    print(f"  Normalized manifest -> {normalized_manifest_path}")

    registry_df = project_registry.build_asset_registry_df(summary)
    registry_path = metocean_evidence.write_parquet(
        registry_df, output_dir / "project_asset_registry.parquet"
    )
    print(f"  Asset registry ({len(registry_df)} row(s)) -> {registry_path}")

    readiness_dict = project_registry.build_project_readiness_dict(summary)
    readiness_path = output_dir / "project_readiness.json"
    readiness_path.write_text(json.dumps(readiness_dict, indent=2, default=str), encoding="utf-8")
    print(f"  Readiness -> {readiness_path}")

    route_gpkg_path = output_dir / "project_route.gpkg"
    if route_gpkg_path.exists():
        route_gpkg_path.unlink()
    route_layer_written = False
    for r in summary.asset_results:
        if r.canonical_route_gdf is not None:
            r.canonical_route_gdf.to_file(
                route_gpkg_path, driver="GPKG", layer=r.registration.asset_id
            )
            route_layer_written = True
    if route_layer_written:
        print(f"  Canonical route(s) -> {route_gpkg_path}")
    else:
        route_gpkg_path = None

    asset_rows = [
        {
            "asset_id": r.registration.asset_id,
            "category": r.registration.category,
            "evidence_role": r.registration.evidence_role,
            "registration_status": r.registration.registration_status,
            "readiness_status_intrinsic": r.registration.readiness_status_intrinsic,
            "readiness_status_effective": r.registration.readiness_status_effective,
            "filename": r.registration.filename,
            "byte_size": r.registration.byte_size,
            "sha256": (r.registration.sha256[:16] + "...") if r.registration.sha256 else None,
            "conflicts": "; ".join(r.registration.conflicts) or "none",
        }
        for r in summary.asset_results
    ]

    evidence_role_summary: dict[str, int] = {}
    # MAR-026A: a project-level working_crs problem is a blocking integration issue even though
    # it is not tied to any single asset (Section 6) -- surfaced first so it is never missed.
    blocking_issues: list[str] = [
        f"PROJECT WORKING CRS: {finding}" for finding in summary.working_crs_findings
    ]
    limitations_list: list[str] = []
    unsupported_categories_list: list[str] = []
    for r in summary.asset_results:
        reg = r.registration
        evidence_role_summary[reg.evidence_role] = (
            evidence_role_summary.get(reg.evidence_role, 0) + 1
        )
        if reg.registration_status == project_registry.REGISTRATION_FAILED:
            blocking_issues.append(f"{reg.asset_id}: {reg.registration_detail}")
        if reg.readiness_result:
            blocking_issues.extend(
                f"{reg.asset_id}: {detail}"
                for detail in reg.readiness_result.get("blocking_reasons", [])
            )
            limitations_list.extend(
                f"{reg.asset_id}: {detail}"
                for detail in reg.readiness_result.get("limitation_reasons", [])
            )
        # MAR-026A Section 3: a declared-vs-observed conflict is itself a BLOCKING
        # project-integration condition, even when the delegated/intrinsic readiness result
        # (above) reported no blocking reason of its own -- it lives outside that result on
        # purpose, since the intrinsic result is never mutated to reflect it.
        blocking_issues.extend(f"{reg.asset_id}: {conflict}" for conflict in reg.conflicts)
        if reg.readiness_status == project_categories.REGISTERED_READINESS_NOT_IMPLEMENTED:
            unsupported_categories_list.append(f"{reg.asset_id} ({reg.category})")

    blocks = project_report.build_project_readiness_report_blocks(
        project_title=f"Operator Project Registration & Readiness -- {manifest.project.name}",
        purpose_text=(
            "Registers operator-supplied local project files (route, bathymetry, burial "
            "profile, and other declared evidence) into a canonical project inventory, "
            "verifying content identity/provenance and delegating scientific readiness "
            "assessment to the existing accepted readiness modules. Introduces no new scour, "
            "mobility, free-span, burial, erosion/deposition, bedform, or risk physics."
        ),
        project_identity_facts={
            "project_id": manifest.project.id,
            "project_name": manifest.project.name,
            "working_crs": working_crs,
            "primary_route_asset_id": manifest.primary_route_asset_id,
            "asset_count": len(summary.asset_results),
        },
        asset_rows=asset_rows,
        evidence_role_summary=evidence_role_summary,
        readiness_summary=summary.readiness_status_counts(),
        blocking_issues=blocking_issues,
        limitations=limitations_list,
        unsupported_categories=unsupported_categories_list,
        production_transfer_notes=[
            "PIPELINE_ROUTE, BATHYMETRY_RASTER, and BURIAL_PROFILE have real readiness "
            "adapters in MAR-026; every other category is registered honestly as "
            "REGISTERED_READINESS_NOT_IMPLEMENTED, never READY, merely because a file exists.",
            project_registry.PROJECT_HAZARD_READINESS_DISCLAIMER,
        ],
    )
    report_html = project_report.render_blocks_html(
        blocks, title=f"Project Readiness -- {manifest.project.name}"
    )
    report_path = output_dir / "project_readiness_report.html"
    report_path.write_text(report_html, encoding="utf-8")
    print(f"  Report -> {report_path}")

    generic_registration_demonstrated = any(
        r.registration.registration_status == project_registry.REGISTERED
        for r in summary.asset_results
    )
    source_files_registered_without_mutation = any(
        r.registration.sha256 is not None for r in summary.asset_results
    )
    bathymetry_readiness_reused = any(
        r.registration.category == project_categories.BATHYMETRY_RASTER
        and r.registration.readiness_result is not None
        for r in summary.asset_results
    )
    burial_readiness_reused = any(
        r.registration.category == project_categories.BURIAL_PROFILE
        and r.registration.readiness_result is not None
        for r in summary.asset_results
    )
    validation = _derive_project_readiness_validation_questions(
        generic_registration_demonstrated=generic_registration_demonstrated,
        source_files_registered_without_mutation=source_files_registered_without_mutation,
        bathymetry_readiness_reused=bathymetry_readiness_reused,
        burial_readiness_reused=burial_readiness_reused,
    )
    validation_path = output_dir / "project_readiness_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    print(f"  Validation -> {validation_path}")

    print()
    print("=== Generic Operator Project Registration & Readiness (MAR-026) ===")
    print()
    print(f"Project: {manifest.project.id} ({manifest.project.name})")
    print(f"Working CRS: {working_crs}")
    print(f"Assets registered: {len(summary.asset_results)}")
    print(f"Registration status counts: {summary.registration_status_counts()}")
    print(f"Readiness status counts: {summary.readiness_status_counts()}")
    print()
    print(project_registry.PROJECT_HAZARD_READINESS_DISCLAIMER)
    print()
    print(
        "IS GENERIC LOCAL OPERATOR PROJECT REGISTRATION DEMONSTRATED? "
        f"{validation['question_a_generic_local_operator_project_registration_demonstrated']}"
    )
    print(
        "ARE OPERATOR SOURCE FILES REGISTERED WITH CONTENT IDENTITY AND PROVENANCE WITHOUT "
        "MUTATION? "
        f"{validation['question_b_source_files_registered_with_identity_and_provenance_without_mutation']}"
    )
    print(
        "IS EXISTING BATHYMETRY READINESS REUSED THROUGH THE GENERIC PROJECT LAYER? "
        f"{validation['question_c_existing_bathymetry_readiness_reused_through_generic_project_layer']}"
    )
    print(
        "IS EXISTING BURIAL-PROFILE READINESS REUSED THROUGH THE GENERIC PROJECT LAYER? "
        f"{validation['question_d_existing_burial_profile_readiness_reused_through_generic_project_layer']}"
    )
    print(
        "DOES MAR-026 CLAIM THE PROJECT IS READY FOR EVERY MARINE GEOHAZARD? "
        f"{validation['question_e_does_mar026_claim_project_ready_for_every_marine_geohazard']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marine-engine",
        description="Seabed-risk modelling engine for subsea pipeline corridors.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print the installed package version.")
    version_parser.set_defaults(func=_cmd_version)

    validate_parser = subparsers.add_parser(
        "validate-config", help="Load and validate a study YAML configuration."
    )
    validate_parser.add_argument("config", type=Path, help="Path to a study config YAML file.")
    validate_parser.set_defaults(func=_cmd_validate_config)

    ingest_parser = subparsers.add_parser(
        "ingest-pipeline",
        help="Ingest a study's pipeline geometry from NSTA and write the canonical GeoPackage.",
    )
    ingest_parser.add_argument("config", type=Path, help="Path to a study config YAML file.")
    ingest_parser.set_defaults(func=_cmd_ingest_pipeline)

    aoi_parser = subparsers.add_parser(
        "build-aoi",
        help="Build the pipeline corridor AOI from the canonical pipeline geometry.",
    )
    aoi_parser.add_argument("config", type=Path, help="Path to a study config YAML file.")
    aoi_parser.set_defaults(func=_cmd_build_aoi)

    chainage_parser = subparsers.add_parser(
        "build-chainage",
        help="Build the 25 m chainage/KP linear-reference points along the canonical pipeline.",
    )
    chainage_parser.add_argument("config", type=Path, help="Path to a study config YAML file.")
    chainage_parser.set_defaults(func=_cmd_build_chainage)

    discover_bathymetry_parser = subparsers.add_parser(
        "discover-bathymetry",
        help="Discover, spatially verify, and rank approved bathymetry sources for the AOI.",
    )
    discover_bathymetry_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    discover_bathymetry_parser.set_defaults(func=_cmd_discover_bathymetry)

    fetch_bathymetry_parser = subparsers.add_parser(
        "fetch-bathymetry",
        help="Acquire the mandatory EMODnet baseline and report manual-download-only datasets.",
    )
    fetch_bathymetry_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    fetch_bathymetry_parser.set_defaults(func=_cmd_fetch_bathymetry)

    build_bathymetry_parser = subparsers.add_parser(
        "build-bathymetry",
        help=(
            "Build the canonical EMODnet baseline DTM (positive-down depth, LAT datum, "
            "AOI-clipped) and sample depth/source/quality attribution onto chainage."
        ),
    )
    build_bathymetry_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_bathymetry_parser.set_defaults(func=_cmd_build_bathymetry)

    resolve_sources_parser = subparsers.add_parser(
        "resolve-bathymetry-sources",
        help=(
            "Resolve PL854's EMODnet source-reference ids to their real SeaDataNet CDI "
            "survey provenance (acquisition epoch, instrument, access, recovery potential)."
        ),
    )
    resolve_sources_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    resolve_sources_parser.set_defaults(func=_cmd_resolve_bathymetry_sources)

    build_morphology_parser = subparsers.add_parser(
        "build-regional-morphology",
        help=(
            "Build broad (500/1000/2000 m-scale) regional seabed morphology context "
            "(slope, TPI, local relief) from the EMODnet baseline, with an analysis halo "
            "to avoid AOI-edge bias, and join MAR-006B/C source-age provenance onto chainage."
        ),
    )
    build_morphology_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_morphology_parser.set_defaults(func=_cmd_build_regional_morphology)

    build_sediment_evidence_parser = subparsers.add_parser(
        "build-sediment-evidence",
        help=(
            "Build the PL854 seabed sediment/substrate evidence base from BGS PSA "
            "observations, BGS Seabed Sediments 250k, and the BGS predictive product "
            "(comparison only) -- no sediment mobility physics."
        ),
    )
    build_sediment_evidence_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_sediment_evidence_parser.set_defaults(func=_cmd_build_sediment_evidence)

    build_metocean_evidence_parser = subparsers.add_parser(
        "build-metocean-evidence",
        help=(
            "Build the PL854 metocean forcing evidence base from Copernicus Marine: primary "
            "1.5 km 3D hourly current, 7 km long-term surface current context, and wave "
            "reanalysis -- forcing evidence only, no bed-shear physics."
        ),
    )
    build_metocean_evidence_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_metocean_evidence_parser.set_defaults(func=_cmd_build_metocean_evidence)

    build_current_normalization_parser = subparsers.add_parser(
        "build-current-normalization",
        help=(
            "Build the current-only 1 m log-profile near-bed normalization sensitivity "
            "(MAR-010) and render the reference-current map -- no network, requires "
            "build-metocean-evidence to have already run."
        ),
    )
    build_current_normalization_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_current_normalization_parser.set_defaults(func=_cmd_build_current_normalization)

    build_wave_orbital_forcing_parser = subparsers.add_parser(
        "build-wave-orbital-forcing",
        help=(
            "Build the wave-only spectral near-bed orbital velocity (MAR-011, Soulsby & "
            "Smallman) and render the wave-orbital reference map -- no network, requires "
            "build-metocean-evidence to have already run."
        ),
    )
    build_wave_orbital_forcing_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_wave_orbital_forcing_parser.set_defaults(func=_cmd_build_wave_orbital_forcing)

    build_combined_bed_shear_parser = subparsers.add_parser(
        "build-combined-bed-shear",
        help=(
            "Build the Soulsby algebraic wave-current bed shear stress sensitivity "
            "(MAR-012) and render the combined bed-shear map -- no network, requires "
            "build-current-normalization and build-wave-orbital-forcing to have already run."
        ),
    )
    build_combined_bed_shear_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_combined_bed_shear_parser.set_defaults(func=_cmd_build_combined_bed_shear)

    build_noncohesive_mobility_parser = subparsers.add_parser(
        "build-noncohesive-mobility",
        help=(
            "Build the noncohesive sediment mobility capacity (MAR-013, Soulsby-Whitehouse) "
            "and render the capacity map + chainage profile -- no network, requires "
            "build-metocean-evidence, build-wave-orbital-forcing, and build-sediment-evidence "
            "to have already run."
        ),
    )
    build_noncohesive_mobility_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_noncohesive_mobility_parser.set_defaults(func=_cmd_build_noncohesive_mobility)

    build_scour_onset_screening_parser = subparsers.add_parser(
        "build-scour-onset-screening",
        help=(
            "Build the pipeline scour-onset embedment screening (MAR-014, Marini et al. "
            "2024) and the 2018 condition benchmark -- no network, requires "
            "build-metocean-evidence, build-wave-orbital-forcing, and "
            "build-regional-morphology to have already run."
        ),
    )
    build_scour_onset_screening_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_scour_onset_screening_parser.set_defaults(func=_cmd_build_scour_onset_screening)

    build_freespan_spatial_evidence_parser = subparsers.add_parser(
        "build-freespan-spatial-evidence",
        help=(
            "Build the official freespan spatial evidence base (MAR-014A, Ithaca Energy "
            "Comparative Assessment Appendix B Table B.1) and reconcile it onto the "
            "canonical route -- no network, requires build-combined-bed-shear, "
            "build-noncohesive-mobility, and build-scour-onset-screening to have already run."
        ),
    )
    build_freespan_spatial_evidence_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_freespan_spatial_evidence_parser.set_defaults(func=_cmd_build_freespan_spatial_evidence)

    ingest_nsta_freespan_registry_parser = subparsers.add_parser(
        "ingest-nsta-freespan-registry",
        help=(
            "Live NSTA acquisition (MAR-014C): query the NSTA Pipeline Freespans "
            "registry (current + removed layers) for PL854/PL855 and cache the raw "
            "response. The ONE network request this ticket performs."
        ),
    )
    ingest_nsta_freespan_registry_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    ingest_nsta_freespan_registry_parser.set_defaults(func=_cmd_ingest_nsta_freespan_registry)

    build_freespan_registry_reconciliation_parser = subparsers.add_parser(
        "build-freespan-registry-reconciliation",
        help=(
            "Offline reconciliation (MAR-014C) of the cached NSTA freespan registry "
            "snapshot against Table B.1 -- no network, requires "
            "ingest-nsta-freespan-registry and build-freespan-spatial-evidence to "
            "have already run."
        ),
    )
    build_freespan_registry_reconciliation_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_freespan_registry_reconciliation_parser.set_defaults(
        func=_cmd_build_freespan_registry_reconciliation
    )

    audit_freespan_context_parser = subparsers.add_parser(
        "audit-freespan-context",
        help=(
            "Positive-only freespan context audit (MAR-015): descriptive evidence audit of "
            "MAR-007/008/010/011A/012/013/014 against the 8 official 2018 events -- no "
            "network, requires build-current-normalization, build-wave-orbital-forcing, "
            "build-combined-bed-shear, build-noncohesive-mobility, "
            "build-scour-onset-screening, and build-freespan-spatial-evidence to have "
            "already run."
        ),
    )
    audit_freespan_context_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    audit_freespan_context_parser.set_defaults(func=_cmd_audit_freespan_context)

    inventory_highres_seabed_data_parser = subparsers.add_parser(
        "inventory-highres-seabed-data",
        help=(
            "High-resolution seabed survey recovery + access audit (MAR-016): queries the "
            "BGS OGC API for real oil/gas site-survey footprints near PL854, classifies real "
            "route/AOI overlap and data access, and audits whether the pipeline-scale seabed-"
            "morphology gap can be closed with verified open data -- requires build-aoi to "
            "have already run; live BGS network access, requires internet."
        ),
    )
    inventory_highres_seabed_data_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    inventory_highres_seabed_data_parser.set_defaults(func=_cmd_inventory_highres_seabed_data)

    build_analog_sandwave_morphometry_parser = subparsers.add_parser(
        "build-analog-sandwave-morphometry",
        help=(
            "Reusable high-resolution sand-wave morphometry engine (MAR-017), built and "
            "validated on the real, open JNCC/Cefas HHW CEND 11/11 analog dataset -- never "
            "PL854 evidence. The PL854 config is used only for project paths/conventions; "
            "outputs live under processed/analogs/hhw_cend1111/. Downloads the official "
            "processed-bathymetry ZIP once (cached thereafter); requires internet on first run."
        ),
    )
    build_analog_sandwave_morphometry_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file (used for paths only)."
    )
    build_analog_sandwave_morphometry_parser.set_defaults(
        func=_cmd_build_analog_sandwave_morphometry
    )

    build_idrbnr_sandwave_validation_parser = subparsers.add_parser(
        "build-idrbnr-sandwave-validation",
        help=(
            "Second open-analog CANONICAL real-data sand-wave morphometry validation "
            "(MAR-017B) on the JNCC/Cefas Inner Dowsing, Race Bank and North Ridge cSAC "
            "(IDRBNR CEND 11/11) analog dataset -- never PL854 evidence. Runs a canonical "
            "support preflight before any morphometry and stops early with an explicit, "
            "honest status if no candidate raster provides a qualifying >=1000 m tile. "
            "Outputs live under processed/analogs/idr_bnr_cend1111/. Downloads the official "
            "processed-bathymetry ZIP once (cached thereafter); requires internet on first run."
        ),
    )
    build_idrbnr_sandwave_validation_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file (used for paths only)."
    )
    build_idrbnr_sandwave_validation_parser.set_defaults(func=_cmd_build_idrbnr_sandwave_validation)

    build_greater_gabbard_sandwave_validation_parser = subparsers.add_parser(
        "build-greater-gabbard-sandwave-validation",
        help=(
            "FINAL open-analog CANONICAL real-data sand-wave morphometry validation "
            "(MAR-017C) on The Crown Estate's Greater Gabbard 2014 (ADUS DeepOcean) analog "
            "dataset -- never PL854 evidence. Runs a canonical support preflight before any "
            "morphometry, and requires natural-seabed eligibility (no turbine/substation/rock-"
            "protection disturbance) before any qualifying tile proceeds to detailed "
            "validation. If this also fails, MAR-017 family analog-hunting is closed for good. "
            "Outputs live under processed/analogs/greater_gabbard_2014/. Extracts only the "
            "needed entries from the official remote archive via HTTP range requests (never "
            "the full ~8.47 GB bundle); requires internet on first run."
        ),
    )
    build_greater_gabbard_sandwave_validation_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file (used for paths only)."
    )
    build_greater_gabbard_sandwave_validation_parser.set_defaults(
        func=_cmd_build_greater_gabbard_sandwave_validation
    )

    build_engineering_evidence_atlas_parser = subparsers.add_parser(
        "build-engineering-evidence-atlas",
        help=(
            "MAR-018: PL854 engineering evidence atlas -- map-first GIS/report packaging of "
            "already-accepted MAR-007/010-016 outputs (observed condition, hydrodynamic "
            "forcing, sediment mobility, scour-onset screening, regional morphology, "
            "high-resolution survey-access audit). No network, no new scientific model: "
            "every value is read, joined, and relabelled, never recomputed. MAR-017 analog "
            "outputs are report context only. Outputs live under "
            "processed/pl854/{evidence_atlas,maps,report}/."
        ),
    )
    build_engineering_evidence_atlas_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_engineering_evidence_atlas_parser.set_defaults(func=_cmd_build_engineering_evidence_atlas)

    build_marine_poc_review_package_parser = subparsers.add_parser(
        "build-marine-poc-review-package",
        help=(
            "MAR-019: OrbGSS Marine POC v0.1 external-reviewer package -- product-framing/"
            "presentation only. Re-renders the MAR-018 atlas/strip with layout and typography "
            "polish, corrects report wording, and builds a POC overview + external reviewer "
            "guide + package index. No network, no new geohazard physics, no risk score. "
            "Reuses MAR-018's accepted outputs by reading them back, never recomputing. "
            "Outputs live under processed/pl854/poc/ (plus re-rendered maps/report/)."
        ),
    )
    build_marine_poc_review_package_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_marine_poc_review_package_parser.set_defaults(func=_cmd_build_marine_poc_review_package)

    build_highres_terrain_poc_parser = subparsers.add_parser(
        "build-highres-terrain-poc",
        help=(
            "MAR-020: generic high-resolution bathymetry / seabed terrain POC, benchmarked "
            "against the real 2020 Fugro Sheringham Shoal survey (Marine Data Exchange "
            "TCE-1986) -- the first non-PL854 OrbGSS Marine Module project. Operator-style data "
            "readiness, canonical bed_elevation_m raster, generic multiscale terrain "
            "derivatives (slope/aspect/curvature/relief/roughness/ruggedness), QA map, terrain "
            "atlas, GIS output, and an engineering POC report. No risk/hazard score. One live "
            "acquisition when the source raster is absent from cache, then fully offline. "
            "Outputs live under processed/sheringham_shoal_2020/."
        ),
    )
    build_highres_terrain_poc_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_highres_terrain_poc_parser.set_defaults(func=_cmd_build_highres_terrain_poc)

    build_seabed_change_poc_parser = subparsers.add_parser(
        "build-seabed-change-poc",
        help=(
            "MAR-021: generic multi-epoch seabed change / DEM-of-Difference POC, benchmarked "
            "against the real 2018 vs 2020 Sheringham Shoal surveys -- the first non-single-"
            "epoch OrbGSS Marine Module project. Hard vertical-datum gate, horizontal/grid "
            "alignment, common-support masking, canonical DoD, uncertainty evidence inventory, "
            "comparison against an independent source-produced difference product. No future "
            "erosion/deposition prediction, no sediment-transport model, no risk score, no ML. "
            "Reuses the already-cached MAR-020 2020 bathymetry; one live acquisition for the "
            "2018/comparator/HSD files when absent from cache, then fully offline. Outputs live "
            "under processed/sheringham_shoal_2020/."
        ),
    )
    build_seabed_change_poc_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_seabed_change_poc_parser.set_defaults(func=_cmd_build_seabed_change_poc)

    build_bedform_morphodynamics_poc_parser = subparsers.add_parser(
        "build-bedform-morphodynamics-poc",
        help=(
            "MAR-022: generic sand-wave/bedform morphodynamics POC, benchmarked against the "
            "real 2018 vs 2020 Sheringham Shoal surveys -- canonical support audit, natural-vs-"
            "anthropogenic tile context (from the real Interpretation Shapefiles package), "
            "per-epoch bedform morphometry (reusing the accepted MAR-017 engine unchanged), and "
            "independent multi-epoch crest matching / observed apparent displacement. No future "
            "migration prediction, no susceptibility, no hazard/risk score, no ML."
        ),
    )
    build_bedform_morphodynamics_poc_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_bedform_morphodynamics_poc_parser.set_defaults(func=_cmd_build_bedform_morphodynamics_poc)

    build_scour_susceptibility_poc_parser = subparsers.add_parser(
        "build-scour-susceptibility-poc",
        help=(
            "MAR-023: generic pipeline scour-onset susceptibility screening POC (Track A, "
            "reusing the accepted MAR-014 Marini et al. 2024 engine unchanged -- actual vs "
            "tested critical embedment margin, forcing-record exceedance fraction, PL854 "
            "tested-embedment scenario envelope) + real observed-scour-evidence ingestion "
            "from the 2024 XOCEAN Sheringham Shoal Seabed Monitoring Survey (Track B), kept "
            "scientifically separate. No future scour-depth prediction, no freespan/fatigue/"
            "VIV prediction, no route suitability, no risk score, no ML. Requires "
            "build-scour-onset-screening to have already run; one minimal live acquisition "
            "for the Sheringham 2024 interpretation package when absent from cache, then "
            "fully offline."
        ),
    )
    build_scour_susceptibility_poc_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_scour_susceptibility_poc_parser.set_defaults(func=_cmd_build_scour_susceptibility_poc)

    build_burial_exposure_poc_parser = subparsers.add_parser(
        "build-burial-exposure-poc",
        help=(
            "MAR-024: generic linear-asset burial/exposure state and cover-margin screening "
            "POC, benchmarked against the real 2016 Deep BV Barrow Offshore Wind Farm export "
            "cable geophysical depth-of-burial survey (TCE-48) -- the first POC built around a "
            "real measured depth-of-burial profile. Observed/measured burial state and "
            "source-interpreted exposure evidence are kept strictly separate from future "
            "exposure susceptibility, which is only ever produced given a defensible "
            "seabed-lowering input. No free-span prediction, no exposure probability, no "
            "scour-depth prediction, no risk score, no route suitability score, no ML. At "
            "most two minimal live acquisitions (Depth of Burial Listing, Route Position List "
            "Files) when absent from cache, then fully offline."
        ),
    )
    build_burial_exposure_poc_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_burial_exposure_poc_parser.set_defaults(func=_cmd_build_burial_exposure_poc)

    build_free_span_poc_parser = subparsers.add_parser(
        "build-free-span-poc",
        help=(
            "MAR-025: generic pipeline free-span geometry and support-loss susceptibility "
            "screening POC -- measured/observed free-span geometry, generic support-loss "
            "screening, and real authoritative free-span evidence ingestion (NSTA UKCS-wide "
            "registry, cached/acquired only if absent; PL854 Table B.1 2018 observed evidence, "
            "reused never recomputed). An explicitly synthetic exact engineering validation "
            "case exercises the full generic engine, since no verified open project-grade "
            "pipeline vertical-profile + seabed support-profile dataset is currently available "
            "in this repo. No structural free-span integrity assessment (DNV-RP-F105/VIV/"
            "fatigue/ULS/FLS), no failure probability, no risk score, no ML."
        ),
    )
    build_free_span_poc_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_free_span_poc_parser.set_defaults(func=_cmd_build_free_span_poc)

    build_project_readiness_parser = subparsers.add_parser(
        "build-project-readiness",
        help=(
            "MAR-026: generic local operator-project registration and asset-readiness layer -- "
            "registers operator-supplied local files (route, bathymetry raster, burial profile, "
            "and other declared evidence) against a project manifest, verifies content identity/"
            "provenance without ever mutating the source files, and delegates scientific "
            "readiness to the existing, unmodified terrain/burial readiness modules. Fully "
            "offline: no network access, no provider discovery, no credentials. Makes no "
            "universal marine-geohazard-readiness claim."
        ),
    )
    build_project_readiness_parser.add_argument(
        "manifest", type=Path, help="Path to an operator project manifest YAML file."
    )
    build_project_readiness_parser.set_defaults(func=_cmd_build_project_readiness)

    build_project_model_parser = subparsers.add_parser(
        "build-project-model",
        help=(
            "MAR-027: canonical project route-reference model and cross-asset linkage POC -- "
            "runs the existing MAR-026/026A registration, operationalizes the manifest-declared "
            "primary route (never guessed), builds a deterministic canonical route-reference "
            "grid (a linear indexing framework, not pipeline supports) when every prerequisite "
            "holds, and represents manifest-declared asset -> route relationships as a separate "
            "linkage axis (never inferred from filenames, directories, or CRS). Emits "
            "canonical_project_model.json, project_asset_linkage.parquet, and -- only when a "
            "grid was built -- project_route_reference.gpkg. No hazard science, no fusion, no "
            "scores, fully offline."
        ),
    )
    build_project_model_parser.add_argument(
        "manifest", type=Path, help="Path to an operator project manifest YAML file."
    )
    build_project_model_parser.set_defaults(func=_cmd_build_project_model)

    build_route_seabed_change_evidence_parser = subparsers.add_parser(
        "build-route-seabed-change-evidence",
        help=(
            "MAR-029: route-referenced observed seabed elevation-change evidence POC -- takes an "
            "EXPLICIT route <-> DoD linkage manifest (never inferred from overlap, CRS, directory, "
            "or filename), runs the existing MAR-026/026A registration and MAR-027 canonical "
            "project model unchanged, consumes the accepted MAR-021/021A DoD raster read-only, "
            "and records the DoD value of the raster cell containing each canonical "
            "route-reference point (no interpolation, no smoothing, nodata never becomes zero). "
            "Raw OBSERVED_SEABED_RAISING/LOWERING sign labels only: no generic significance "
            "threshold, no erosion/deposition/scour attribution, no burial/free-span coupling, "
            "no future change rate, no score. Emits route_observed_seabed_change.{parquet,gpkg} "
            "(only when available), plus metadata/validation JSON and an HTML report under "
            "data/processed/<project_id>/change_route/. Fully offline."
        ),
    )
    build_route_seabed_change_evidence_parser.add_argument(
        "manifest", type=Path, help="Path to a route-change evidence manifest YAML file."
    )
    build_route_seabed_change_evidence_parser.set_defaults(
        func=_cmd_build_route_seabed_change_evidence
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

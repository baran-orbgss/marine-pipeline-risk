"""Minimal command-line entry point for the marine-engine package."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.ops import unary_union

from marine_engine import __version__
from marine_engine.analogs import hhw_cend1111 as hhw_analog
from marine_engine.config import load_study_config
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
from marine_engine.providers import bgs_offshore_surveys, nsta_freespan
from marine_engine.providers.bathymetry import acquisition, bgs, emodnet, inventory, ukho
from marine_engine.providers.bathymetry import hhw_cend1111 as hhw_provider
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
    freespan_evidence,
    freespan_evidence_map,
    freespan_temporal_provenance,
    nsta_freespan_reconciliation,
    nsta_freespan_reconciliation_map,
    pipeline_condition,
    scour_onset,
    scour_onset_map,
)
from marine_engine.sediment import evidence, noncohesive_mobility, noncohesive_mobility_map
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
            **method_inputs, output_path=method_output_path, subtitle=method_subtitle
        )
    else:
        method_map_path = None
        print("  no canonical or exploratory tile available -- method figure skipped")

    stats_map_path = swmap.render_bedform_distribution_figure(
        bedforms_df=figure_bedform_df,
        output_path=maps_dir / "hhw_sandwave_morphometry_statistics.png",
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


def _dataset_start_or(time_range_ms: tuple | None, fallback_now: datetime) -> datetime:
    """The live dataset's own start timestamp, or `fallback_now` if it could not be discovered."""

    if time_range_ms is None:
        return fallback_now
    return datetime.fromtimestamp(time_range_ms[0] / 1000.0, tz=UTC)


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

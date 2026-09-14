"""Generic `bedform_morphodynamics` capability runtime (MAR-034 Sections 19-21).

Registers the accepted MAR-017/MAR-022 bedform-morphometry engine
(`morphology.sandwave_morphometry`, `bedforms.extraction`, `bedforms.natural_context`,
`bedforms.matching`) behind the generic capability-orchestration runtime -- no new detection
formula, tolerance, or eligibility threshold is introduced here.

Automatic tile DISCOVERY (the standalone `build-bedform-morphodynamics-poc` command's own
2000 m/1000 m cascading search, `morphology.sandwave_morphometry.find_valid_tiles`) is not
reimplemented: choosing where to place a canonical validation tile and its crest azimuth is not
something this ticket can derive from raster values alone without guessing, so it is instead a
genuinely missing, DECLARED fact (Section 19: "if mandatory bedform parameters are not available
automatically, expose them as exact capability blockers -- do not invent defaults"). A
`BedformDeclaration` (this module's own capability-specific declaration, loaded through the
generic `declaration_adapter` mechanism -- Section 8/48) names each tile's centre, size, and
crest azimuth explicitly.

Per-tile CANONICAL-SCALE eligibility (Section 20: coarse regional bathymetry must never be
treated as high-resolution bedform evidence) is checked with the EXACT accepted rule,
`sandwave_morphometry.meets_wavelengths_across_tile` (>=3 wavelengths must fit across the tile),
applied to the tile's OWN detected median wavelength -- never a new threshold, and reported
transparently rather than silently excluded (mirroring the accepted MAR-022 POC's own "report
`ANTHROPOGENIC_DISTURBANCE_PRESENT` transparently, never re-tune until eligible" precedent).
Natural-vs-anthropogenic context similarly reports `INFRASTRUCTURE_CONTEXT_INSUFFICIENT` (an
existing, honest "we do not know" state) when no interpretation layer was declared -- never
silently treated as natural.

When exactly two canonical terrain epochs are available,
`bedforms.matching.match_crests_within_tile` additionally runs per declared tile -- the SAME
independent, DoD-blind crest matching the standalone command uses, never a new comparison method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, box

from marine_engine.bedforms import contract as bedforms_contract
from marine_engine.bedforms import extraction, matching, natural_context
from marine_engine.morphology import sandwave_morphometry as swm
from marine_engine.orchestration.capability import BEDFORM_MORPHODYNAMICS, CAPABILITY_REGISTRY
from marine_engine.orchestration.context import PlanningContext
from marine_engine.orchestration.declaration import DeclarationRequest
from marine_engine.orchestration.planner import (
    AVAILABLE,
    BLOCKED_MISSING_INPUT,
    NOT_APPLICABLE,
    CapabilityPlan,
)
from marine_engine.orchestration.product_manifest import (
    AnalysisProductManifest,
    write_product_manifest,
)
from marine_engine.orchestration.runtime import (
    CapabilityExecutionOutcome,
    CapabilityRuntime,
    register_capability_runtime,
)
from marine_engine.slope_stability import contract as slope_stability_contract

__all__ = [
    "NO_DECLARED_BEDFORM_TILES",
    "BedformTileDeclaration",
    "BedformDeclaration",
    "bedform_declaration_adapter",
    "bedform_morphodynamics_readiness_adapter",
    "bedform_morphodynamics_planner_adapter",
    "bedform_morphodynamics_executor",
    "register_bedform_morphodynamics_runtime",
]

_METHOD_ID = "MAR-022_BEDFORM_MORPHODYNAMICS"

# Section 19: the exact, named blocker when no tile centre/size/crest-azimuth has been declared --
# never guessed, never auto-discovered by this capability.
NO_DECLARED_BEDFORM_TILES = "NO_DECLARED_BEDFORM_TILES"


@dataclass(frozen=True)
class BedformTileDeclaration:
    tile_id: str
    center_x_m: float
    center_y_m: float
    tile_size_m: float
    crest_azimuth_deg: float


@dataclass(frozen=True)
class BedformDeclaration:
    """This capability's own declaration (Section 48), loaded from the run manifest's
    `capabilities.bedform_morphodynamics` section -- never parsed by the generic CLI."""

    tiles: tuple[BedformTileDeclaration, ...] = ()
    anthropogenic_context_paths: tuple[Path, ...] = ()


def bedform_declaration_adapter(request: DeclarationRequest) -> BedformDeclaration | None:
    section = request.raw_capability_section
    if not section:
        return None
    base_dir = request.manifest_dir or Path(".")
    tiles = tuple(
        BedformTileDeclaration(
            tile_id=str(raw["tile_id"]),
            center_x_m=float(raw["center_x_m"]),
            center_y_m=float(raw["center_y_m"]),
            tile_size_m=float(raw.get("tile_size_m", swm.CANONICAL_TILE_SIZE_M)),
            crest_azimuth_deg=float(raw["crest_azimuth_deg"]),
        )
        for raw in (section.get("tiles") or [])
    )
    raw_context_paths = section.get("anthropogenic_context_paths") or []
    context_paths = tuple((base_dir / raw_path).resolve() for raw_path in raw_context_paths)
    return BedformDeclaration(tiles=tiles, anthropogenic_context_paths=context_paths)


class _TerrainSource:
    __slots__ = ("source_asset_ids", "raster_path")

    def __init__(self, source_asset_ids: tuple[str, ...], raster_path: Path) -> None:
        self.source_asset_ids = source_asset_ids
        self.raster_path = raster_path


def _canonical_terrain_sources(context: PlanningContext) -> tuple[_TerrainSource, ...]:
    sources: list[_TerrainSource] = []
    seen: set[Path] = set()
    for asset in context.assets_with_role(slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED):
        sources.append(_TerrainSource((asset.asset_id,), asset.path))
        seen.add(asset.path.resolve())
    for manifest in context.manifests_with_role(
        slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED
    ):
        if not manifest.geometry_or_raster_path:
            continue
        raster_path = Path(manifest.geometry_or_raster_path)
        if raster_path.resolve() in seen:
            continue
        sources.append(_TerrainSource(manifest.source_asset_ids, raster_path))
        seen.add(raster_path.resolve())
    return tuple(sources)


@dataclass(frozen=True)
class BedformReadinessFacts:
    terrain_sources: tuple[_TerrainSource, ...]
    declaration: BedformDeclaration | None


def bedform_morphodynamics_readiness_adapter(
    context: PlanningContext,
) -> BedformReadinessFacts | None:
    sources = _canonical_terrain_sources(context)
    if not sources:
        return None
    declaration = context.capability_declarations.get(BEDFORM_MORPHODYNAMICS)
    return BedformReadinessFacts(terrain_sources=sources, declaration=declaration)


def bedform_morphodynamics_planner_adapter(
    context: PlanningContext, readiness_facts: BedformReadinessFacts | None
) -> CapabilityPlan:
    if readiness_facts is None:
        return CapabilityPlan(
            BEDFORM_MORPHODYNAMICS,
            NOT_APPLICABLE,
            (
                f"no {slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED} product is "
                "available yet",
            ),
            (f"a {slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED} product",),
        )
    if readiness_facts.declaration is None or not readiness_facts.declaration.tiles:
        return CapabilityPlan(
            BEDFORM_MORPHODYNAMICS,
            BLOCKED_MISSING_INPUT,
            (
                f"{NO_DECLARED_BEDFORM_TILES}: no tile centre/size/crest_azimuth_deg was "
                "declared for this capability -- automatic tile discovery is out of scope, this "
                "is a genuinely missing declared fact, not something inferred from the raster",
            ),
            (NO_DECLARED_BEDFORM_TILES,),
        )
    return CapabilityPlan(BEDFORM_MORPHODYNAMICS, AVAILABLE, (), ())


def _load_anthropogenic_context(paths: tuple[Path, ...]) -> gpd.GeoDataFrame | None:
    if not paths:
        return None
    frames = []
    for path in paths:
        gdf = gpd.read_file(path)
        gdf["source_layer"] = gdf.get("source_layer", path.stem)
        if "raw_description" not in gdf.columns:
            gdf["raw_description"] = None
        frames.append(gdf)
    if not frames:
        return None
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def bedform_morphodynamics_executor(
    context: PlanningContext, plan: CapabilityPlan
) -> CapabilityExecutionOutcome:
    readiness_facts = bedform_morphodynamics_readiness_adapter(context)
    declaration = readiness_facts.declaration if readiness_facts is not None else None
    if readiness_facts is None or declaration is None or not declaration.tiles:
        return CapabilityExecutionOutcome(
            capability_id=BEDFORM_MORPHODYNAMICS, plan=plan, manifests=(), details=None
        )

    anthropogenic_gdf = _load_anthropogenic_context(declaration.anthropogenic_context_paths)
    interpretation_available = bool(declaration.anthropogenic_context_paths)

    extractions_by_epoch: dict[str, dict[str, extraction.TileBedformExtraction]] = {}
    tile_status_rows: list[dict] = []
    all_bedform_rows: list[dict] = []
    all_transect_rows: list[dict] = []
    all_crest_rows: list[dict] = []
    all_trough_rows: list[dict] = []
    source_asset_ids: list[str] = []

    for source in readiness_facts.terrain_sources:
        source_asset_ids.extend(source.source_asset_ids)
        epoch = source.source_asset_ids[0] if source.source_asset_ids else source.raster_path.stem
        with rasterio.open(source.raster_path) as src:
            band = src.read(1)
            transform = src.transform
        valid_mask = np.isfinite(band)

        extractions_by_epoch[epoch] = {}
        for tile in declaration.tiles:
            result = extraction.extract_tile_bedforms(
                band,
                valid_mask,
                transform,
                tile_id=tile.tile_id,
                epoch=epoch,
                center_x_m=tile.center_x_m,
                center_y_m=tile.center_y_m,
                tile_size_m=tile.tile_size_m,
                crest_azimuth_deg=tile.crest_azimuth_deg,
            )
            extractions_by_epoch[epoch][tile.tile_id] = result
            all_bedform_rows.extend(result.bedform_rows)
            all_transect_rows.extend(result.transect_rows)
            all_crest_rows.extend(result.crest_points)
            all_trough_rows.extend(result.trough_points)

            wavelengths = [row["wavelength_m"] for row in result.bedform_rows]
            median_wavelength = float(pd.Series(wavelengths).median()) if wavelengths else None
            canonical_scale_eligible = (
                median_wavelength is not None
                and swm.meets_wavelengths_across_tile(median_wavelength, tile.tile_size_m)
            )
            tile_geometry = box(
                tile.center_x_m - tile.tile_size_m / 2,
                tile.center_y_m - tile.tile_size_m / 2,
                tile.center_x_m + tile.tile_size_m / 2,
                tile.center_y_m + tile.tile_size_m / 2,
            )
            natural_status = natural_context.assess_natural_bedform_validation_status(
                tile_geometry,
                anthropogenic_context_gdf=anthropogenic_gdf,
                interpretation_available=interpretation_available,
            )
            tile_status_rows.append(
                {
                    "tile_id": tile.tile_id,
                    "epoch": epoch,
                    "bedform_count": len(result.bedform_rows),
                    "median_wavelength_m": median_wavelength,
                    "canonical_scale_eligible": canonical_scale_eligible,
                    "natural_bedform_validation_status": natural_status.status,
                    "natural_context_reason": natural_status.reason,
                    "geometry": tile_geometry,
                }
            )

    crest_match_rows: list[dict] = []
    epochs = list(extractions_by_epoch)
    if len(epochs) == 2:
        epoch1, epoch2 = epochs
        for tile in declaration.tiles:
            extraction1 = extractions_by_epoch[epoch1].get(tile.tile_id)
            extraction2 = extractions_by_epoch[epoch2].get(tile.tile_id)
            if extraction1 is None or extraction2 is None:
                continue
            _all_pairs, canonical_matches = matching.match_crests_within_tile(
                tile.tile_id, extraction1.crest_points, extraction2.crest_points
            )
            crest_match_rows.extend(canonical_matches)

    if not all_bedform_rows and not tile_status_rows:
        return CapabilityExecutionOutcome(
            capability_id=BEDFORM_MORPHODYNAMICS, plan=plan, manifests=(), details=None
        )

    first_asset_id = source_asset_ids[0] if source_asset_ids else "bedforms"
    out_dir = Path("data/processed") / first_asset_id / "bedforms"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[AnalysisProductManifest] = []

    observations_path = out_dir / "bedform_observations.parquet"
    pd.DataFrame(all_bedform_rows).to_parquet(observations_path, index=False)
    manifests.append(
        _write_manifest(
            product_id=f"{first_asset_id}__bedform_observations",
            support_type="TABLE",
            source_asset_ids=tuple(dict.fromkeys(source_asset_ids)),
            units={"wavelength_m": "m", "wave_height_m": "m"},
            primary_value_fields=("wavelength_m", "wave_height_m", "scale_classification"),
            geometry_or_raster_path=str(observations_path),
            limitations=(
                "One row per profile crossing, not a unique 2D crest-line count (Section 21).",
            ),
            display_name=f"Bedform Observations -- {first_asset_id}",
            out_dir=out_dir,
        )
    )

    if all_crest_rows or all_trough_rows:
        points_path = out_dir / "bedform_points.gpkg"
        if all_crest_rows:
            crest_gdf = gpd.GeoDataFrame(
                pd.DataFrame(all_crest_rows),
                geometry=[Point(row["x"], row["y"]) for row in all_crest_rows],
                crs=_source_crs(readiness_facts.terrain_sources[0].raster_path),
            )
            crest_gdf.drop(columns=["x", "y"]).to_file(points_path, driver="GPKG", layer="crests")
        if all_trough_rows:
            trough_gdf = gpd.GeoDataFrame(
                pd.DataFrame(all_trough_rows),
                geometry=[Point(row["x"], row["y"]) for row in all_trough_rows],
                crs=_source_crs(readiness_facts.terrain_sources[0].raster_path),
            )
            trough_gdf.drop(columns=["x", "y"]).to_file(points_path, driver="GPKG", layer="troughs")
        manifests.append(
            _write_manifest(
                product_id=f"{first_asset_id}__bedform_points",
                support_type="POINT",
                source_asset_ids=tuple(dict.fromkeys(source_asset_ids)),
                units={},
                primary_value_fields=("scale_classification",),
                geometry_or_raster_path=str(points_path),
                limitations=(),
                display_name=f"Bedform Crest/Trough Points -- {first_asset_id}",
                out_dir=out_dir,
            )
        )

    if crest_match_rows:
        matches_path = out_dir / "canonical_crest_matches.parquet"
        pd.DataFrame(crest_match_rows).to_parquet(matches_path, index=False)
        manifests.append(
            _write_manifest(
                product_id=f"{first_asset_id}__canonical_crest_matches",
                support_type="TABLE",
                source_asset_ids=tuple(dict.fromkeys(source_asset_ids)),
                units={},
                primary_value_fields=("match_status",),
                geometry_or_raster_path=str(matches_path),
                limitations=(matching.APPARENT_RATE_DISCLAIMER,),
                display_name=f"Canonical Crest Matches -- {first_asset_id}",
                out_dir=out_dir,
            )
        )

    tile_status_path = out_dir / "tile_validation_status.parquet"
    tile_status_records = [
        {key: value for key, value in row.items() if key != "geometry"} for row in tile_status_rows
    ]
    pd.DataFrame(tile_status_records).to_parquet(tile_status_path, index=False)
    manifests.append(
        _write_manifest(
            product_id=f"{first_asset_id}__tile_validation_status",
            support_type="TABLE",
            source_asset_ids=tuple(dict.fromkeys(source_asset_ids)),
            units={},
            primary_value_fields=("natural_bedform_validation_status", "canonical_scale_eligible"),
            geometry_or_raster_path=str(tile_status_path),
            limitations=(
                "canonical_scale_eligible reuses the accepted >=3-wavelengths-across-tile rule "
                "applied to this tile's own median detected wavelength.",
            ),
            display_name=f"Bedform Tile Validation Status -- {first_asset_id}",
            out_dir=out_dir,
        )
    )

    return CapabilityExecutionOutcome(
        capability_id=BEDFORM_MORPHODYNAMICS, plan=plan, manifests=tuple(manifests), details=None
    )


def _write_manifest(
    *,
    product_id,
    support_type,
    source_asset_ids,
    units,
    primary_value_fields,
    geometry_or_raster_path,
    limitations,
    display_name,
    out_dir,
) -> AnalysisProductManifest:
    manifest = AnalysisProductManifest(
        product_id=product_id,
        capability_id=BEDFORM_MORPHODYNAMICS,
        scientific_role=bedforms_contract.SANDBED_BEDFORM_MORPHOLOGY_AND_OBSERVED_CHANGE,
        evidence_role="DERIVED",
        support_type=support_type,
        source_asset_ids=source_asset_ids,
        scenario_id=None,
        method_id=_METHOD_ID,
        units=units,
        primary_value_fields=primary_value_fields,
        geometry_or_raster_path=geometry_or_raster_path,
        readiness=AVAILABLE,
        limitations=limitations,
        display_name=display_name,
    )
    write_product_manifest(manifest, out_dir / f"{manifest.product_id}.product_manifest.json")
    return manifest


def _source_crs(raster_path: Path) -> str:
    with rasterio.open(raster_path) as src:
        return str(src.crs)


def register_bedform_morphodynamics_runtime() -> None:
    register_capability_runtime(
        CapabilityRuntime(
            definition=CAPABILITY_REGISTRY[BEDFORM_MORPHODYNAMICS],
            readiness_adapter=bedform_morphodynamics_readiness_adapter,
            planner_adapter=bedform_morphodynamics_planner_adapter,
            executor=bedform_morphodynamics_executor,
            declaration_adapter=bedform_declaration_adapter,
        )
    )

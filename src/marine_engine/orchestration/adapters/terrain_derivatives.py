"""Generic `terrain_derivatives` capability runtime (MAR-034 Sections 16-18).

Registers the accepted MAR-020 `terrain.derivatives` windowed-moment terrain engine (slope,
aspect, profile/plan curvature, local relief, terrain roughness) behind the generic
capability-orchestration runtime -- the SAME functions the standalone `build-highres-terrain-poc`
command already calls, no new formula or scale added "just because it would be convenient"
(Section 16).

Depends on `canonicalize_bathymetry` (Section 17): its canonical terrain source may come from
EITHER a recognized input asset (a previously-canonicalized file fed back in) OR a manifest
`canonicalize_bathymetry` just produced earlier in the SAME `execute_plan` run -- proving MAR-033A's
generated-product-unlocks-a-dependent-capability chaining with real science, not a synthetic
stand-in. No AOI is required (Section 18): the raster's own valid-data extent is the processing
extent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from marine_engine.orchestration.capability import CAPABILITY_REGISTRY, TERRAIN_DERIVATIVES
from marine_engine.orchestration.context import PlanningContext
from marine_engine.orchestration.planner import AVAILABLE, NOT_APPLICABLE, CapabilityPlan
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
from marine_engine.terrain import derivatives, product_roles, raster_io

__all__ = [
    "TerrainDerivativeSource",
    "terrain_derivatives_readiness_adapter",
    "terrain_derivatives_planner_adapter",
    "terrain_derivatives_executor",
    "register_terrain_derivatives_runtime",
]

_METHOD_ID = "MAR-020_TERRAIN_DERIVATIVES"
_ENGINEERING_SCALE_M = 10.0


class TerrainDerivativeSource:
    __slots__ = ("source_asset_ids", "raster_path")

    def __init__(self, source_asset_ids: tuple[str, ...], raster_path: Path) -> None:
        self.source_asset_ids = source_asset_ids
        self.raster_path = raster_path


def _canonical_terrain_sources(context: PlanningContext) -> tuple[TerrainDerivativeSource, ...]:
    """Every available canonical bed-elevation source this run currently has -- from a recognized
    input asset, from an already-produced manifest, or both (Section 17)."""

    sources: list[TerrainDerivativeSource] = []
    for asset in context.assets_with_role(slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED):
        sources.append(TerrainDerivativeSource((asset.asset_id,), asset.path))
    for manifest in context.manifests_with_role(
        slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED
    ):
        if manifest.geometry_or_raster_path:
            raster_path = Path(manifest.geometry_or_raster_path)
            sources.append(TerrainDerivativeSource(manifest.source_asset_ids, raster_path))
    return tuple(sources)


def terrain_derivatives_readiness_adapter(
    context: PlanningContext,
) -> tuple[TerrainDerivativeSource, ...] | None:
    sources = _canonical_terrain_sources(context)
    return sources or None


def terrain_derivatives_planner_adapter(
    context: PlanningContext, readiness_facts: tuple[TerrainDerivativeSource, ...] | None
) -> CapabilityPlan:
    if readiness_facts is None:
        return CapabilityPlan(
            TERRAIN_DERIVATIVES,
            NOT_APPLICABLE,
            (
                f"no {slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED} product is "
                "available yet (neither a recognized input nor one produced earlier in this run)",
            ),
            (f"a {slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED} product",),
        )
    return CapabilityPlan(TERRAIN_DERIVATIVES, AVAILABLE, (), ())


def _write_layer(
    *,
    name: str,
    array: np.ndarray,
    units: str,
    transform,
    crs,
    out_dir: Path,
    filename: str,
    tags_base: dict[str, str],
    source_asset_ids: tuple[str, ...],
    display_name: str,
) -> AnalysisProductManifest:
    scale_label = f"engineering ({_ENGINEERING_SCALE_M:g} m)"
    path = raster_io.write_terrain_raster(
        array,
        transform,
        str(crs),
        out_dir / filename,
        {**tags_base, "layer": name, "units": units, "scale": scale_label},
    )
    manifest = AnalysisProductManifest(
        product_id=f"{source_asset_ids[0]}__{name}",
        capability_id=TERRAIN_DERIVATIVES,
        scientific_role=product_roles.TERRAIN_DERIVATIVE_PRODUCT,
        evidence_role="DERIVED",
        support_type="AREA_SURFACE",
        source_asset_ids=source_asset_ids,
        scenario_id=None,
        method_id=_METHOD_ID,
        units={name: units},
        primary_value_fields=(name,),
        geometry_or_raster_path=str(path),
        readiness=AVAILABLE,
        limitations=(),
        display_name=display_name,
    )
    write_product_manifest(manifest, out_dir / f"{manifest.product_id}.product_manifest.json")
    return manifest


def terrain_derivatives_executor(
    context: PlanningContext, plan: CapabilityPlan
) -> CapabilityExecutionOutcome:
    manifests: list[AnalysisProductManifest] = []
    for source in _canonical_terrain_sources(context):
        with rasterio.open(source.raster_path) as src:
            band = src.read(1)
            crs = src.crs
            transform = src.transform
            nodata = src.nodata
        valid_mask = np.isfinite(band)
        if nodata is not None and np.isfinite(nodata):
            # A NaN nodata sentinel (every raster this capability itself writes via
            # `raster_io.write_terrain_raster`) is already excluded by `np.isfinite` above; only a
            # real numeric sentinel (e.g. a raw provider raster's own -9999) needs this extra check.
            valid_mask &= ~np.isclose(band, nodata, rtol=0, atol=abs(nodata) * 1e-6 + 1e-6)
        bed_elevation_m = np.where(valid_mask, band, np.nan)
        cell_size_m = float(transform.a)
        source_asset_ids = source.source_asset_ids or (source.raster_path.stem,)
        out_dir = Path("data/processed") / source_asset_ids[0] / "terrain"
        tags_base = {
            "product": "MAR-034 generic terrain_derivatives capability",
            "scientific_role": product_roles.TERRAIN_DERIVATIVE_PRODUCT,
            "source_asset_id": source_asset_ids[0],
        }

        slope_deg, aspect_deg, _slope_vf = derivatives.compute_slope_aspect_deg(
            bed_elevation_m, valid_mask, _ENGINEERING_SCALE_M, cell_size_m
        )
        profile_curv, plan_curv, _curv_vf = derivatives.compute_profile_plan_curvature(
            bed_elevation_m, valid_mask, _ENGINEERING_SCALE_M, cell_size_m
        )
        relief, _relief_vf = derivatives.compute_local_relief(
            bed_elevation_m, valid_mask, _ENGINEERING_SCALE_M, cell_size_m
        )
        terrain_std, _std_vf = derivatives.compute_terrain_std(
            bed_elevation_m, valid_mask, _ENGINEERING_SCALE_M, cell_size_m
        )
        ruggedness, _tri_vf = derivatives.compute_ruggedness(
            bed_elevation_m, valid_mask, _ENGINEERING_SCALE_M, cell_size_m
        )

        layers = (
            ("slope_deg", slope_deg, "deg", "slope.tif"),
            ("aspect_deg", aspect_deg, "deg (compass bearing)", "aspect.tif"),
            ("profile_curvature", profile_curv, "1/m", "profile_curvature.tif"),
            ("plan_curvature", plan_curv, "1/m", "plan_curvature.tif"),
            ("local_relief", relief, "m", "local_relief.tif"),
            ("terrain_std", terrain_std, "m", "terrain_std.tif"),
            ("ruggedness_tri", ruggedness, "m", "ruggedness.tif"),
        )
        for name, array, units, filename in layers:
            manifests.append(
                _write_layer(
                    name=name,
                    array=array,
                    units=units,
                    transform=transform,
                    crs=crs,
                    out_dir=out_dir,
                    filename=filename,
                    tags_base=tags_base,
                    source_asset_ids=source_asset_ids,
                    display_name=f"Terrain Derivative -- {name} -- {source_asset_ids[0]}",
                )
            )

    return CapabilityExecutionOutcome(
        capability_id=TERRAIN_DERIVATIVES, plan=plan, manifests=tuple(manifests), details=None
    )


def register_terrain_derivatives_runtime() -> None:
    register_capability_runtime(
        CapabilityRuntime(
            definition=CAPABILITY_REGISTRY[TERRAIN_DERIVATIVES],
            readiness_adapter=terrain_derivatives_readiness_adapter,
            planner_adapter=terrain_derivatives_planner_adapter,
            executor=terrain_derivatives_executor,
        )
    )

"""Generic `canonicalize_bathymetry` capability runtime (MAR-034 Sections 11-15).

Registers the accepted MAR-020 `terrain.readiness`/`terrain.canonical` science behind the generic
capability-orchestration runtime. No new formula, threshold, or sign-convention rule is
introduced here: this module only assembles `terrain.readiness.RasterFacts` from a recognized
asset's own observed raster bytes plus its DECLARED `AssetDeclaration`
(vertical_datum/survey_epoch/source_sign_convention), then calls
`terrain.readiness.assess_bathymetry_readiness` / `terrain.canonical.build_canonical_bed_elevation`
exactly as the standalone MAR-020 `build-highres-terrain-poc` command already does, and writes
the output through the SAME `terrain.raster_io.write_terrain_raster` tagging convention (so a
canonical product this capability writes is recognized identically to one written by the
standalone command -- see `intake.recognition.recognize_canonical_terrain_product`).

Every candidate `BATHYMETRY_RASTER` asset is canonicalized independently (Section 22: multiple
survey epochs must each become their own canonical product, never merged). The plan is AVAILABLE
if at least one candidate asset is ready; the executor independently re-checks each asset itself
(never trusting the aggregate plan alone) and writes a product only for the ones that actually
pass -- an asset that is not ready is silently skipped, its exact blocker already reported by the
plan.

Sign convention is a hard scientific boundary (Section 13): never inferred from raster values or
a filename. `assess_bathymetry_readiness` has no sign-convention concept at all (it is a MAR-020
raster-quality check, not a MAR-034 orchestration gate); this capability's OWN plan additionally
requires an explicit `AssetDeclaration.source_sign_convention` per asset before canonicalization
is available for it. The intrinsic terrain readiness result is reported unmodified alongside this
extra gate -- mirroring MAR-026A's intrinsic/effective readiness distinction (CLAUDE.md Section
12), never silently folded into one status.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from marine_engine.intake.declaration import AssetDeclaration
from marine_engine.orchestration.capability import CANONICALIZE_BATHYMETRY, CAPABILITY_REGISTRY
from marine_engine.orchestration.context import PlanningContext, RecognizedAsset
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
from marine_engine.project import categories as project_categories
from marine_engine.slope_stability import contract as slope_stability_contract
from marine_engine.terrain import canonical, raster_io, readiness

__all__ = [
    "SOURCE_SIGN_CONVENTION_REQUIRED",
    "BathymetryAssetReadiness",
    "canonicalize_bathymetry_readiness_adapter",
    "canonicalize_bathymetry_planner_adapter",
    "canonicalize_bathymetry_executor",
    "register_canonicalize_bathymetry_runtime",
]

# Section 13: the exact, named blocker when an otherwise-plausible bathymetry asset has no
# declared source sign convention -- never guessed from raster values or a filename.
SOURCE_SIGN_CONVENTION_REQUIRED = "SOURCE_SIGN_CONVENTION_REQUIRED"

_METHOD_ID = "MAR-020_CANONICAL_BED_ELEVATION"


@dataclass(frozen=True)
class BathymetryAssetReadiness:
    """One candidate bathymetry asset's own facts. `intrinsic_readiness` is
    `terrain.readiness.assess_bathymetry_readiness`'s own, UNMODIFIED verdict (Section 12)."""

    asset: RecognizedAsset
    declaration: AssetDeclaration | None
    raster_facts: readiness.RasterFacts
    intrinsic_readiness: readiness.ReadinessResult

    @property
    def sign_convention_declared(self) -> bool:
        return bool(self.declaration and self.declaration.source_sign_convention)

    @property
    def effectively_ready(self) -> bool:
        return (
            self.intrinsic_readiness.status != readiness.NOT_READY and self.sign_convention_declared
        )


def _read_raster_facts(
    path: Path, declaration: AssetDeclaration | None
) -> tuple[readiness.RasterFacts, np.ndarray, Any, Any, float | None]:
    """OBSERVED facts read directly from the raster's own bytes (Section 7) -- never merged with
    `declaration`'s DECLARED vertical_datum/survey_epoch, only assembled alongside them."""

    with rasterio.open(path) as src:
        band = src.read(1)
        crs = src.crs
        transform = src.transform
        nodata = src.nodata
        width, height = src.width, src.height
        bounds = tuple(src.bounds)
        band_count = src.count
        dtype = src.dtypes[0]
        color_interp = tuple(str(c) for c in src.colorinterp)

    valid_mask_raw = np.isfinite(band)
    if nodata is not None and np.isfinite(nodata):
        valid_mask_raw &= ~np.isclose(band, nodata, rtol=0, atol=abs(nodata) * 1e-6 + 1e-6)
    valid_values = band[valid_mask_raw]
    facts = readiness.RasterFacts(
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
        bounds=bounds,
        nodata_value=nodata,
        vertical_datum=declaration.vertical_datum if declaration else None,
        survey_epoch=declaration.survey_epoch if declaration else None,
        data_min=float(valid_values.min()) if valid_values.size else None,
        data_max=float(valid_values.max()) if valid_values.size else None,
        data_std=float(valid_values.std()) if valid_values.size else None,
        valid_cell_fraction=float(valid_mask_raw.mean()) if valid_mask_raw.size else None,
    )
    return facts, band, crs, transform, nodata


def _candidate_bathymetry_assets(context: PlanningContext) -> tuple[RecognizedAsset, ...]:
    return context.assets_with_role(project_categories.BATHYMETRY_RASTER)


def canonicalize_bathymetry_readiness_adapter(
    context: PlanningContext,
) -> tuple[BathymetryAssetReadiness, ...] | None:
    """`None` when no input recognized the required `BATHYMETRY_RASTER` role at all -- mirroring
    the earthquake-triggering capability's own "no facts supplied" contract."""

    candidates = _candidate_bathymetry_assets(context)
    if not candidates:
        return None
    results = []
    for asset in candidates:
        facts, _band, _crs, _transform, _nodata = _read_raster_facts(asset.path, asset.declaration)
        intrinsic = readiness.assess_bathymetry_readiness(facts)
        results.append(
            BathymetryAssetReadiness(
                asset=asset,
                declaration=asset.declaration,
                raster_facts=facts,
                intrinsic_readiness=intrinsic,
            )
        )
    return tuple(results)


def canonicalize_bathymetry_planner_adapter(
    context: PlanningContext, readiness_facts: tuple[BathymetryAssetReadiness, ...] | None
) -> CapabilityPlan:
    if readiness_facts is None:
        return CapabilityPlan(
            CANONICALIZE_BATHYMETRY,
            NOT_APPLICABLE,
            (f"no input recognized the required {project_categories.BATHYMETRY_RASTER} role",),
            (
                f"a {project_categories.BATHYMETRY_RASTER} asset, recognized via an explicit "
                "AssetDeclaration",
            ),
        )

    reasons: list[str] = []
    any_ready = False
    for entry in readiness_facts:
        if entry.effectively_ready:
            any_ready = True
            continue
        if not entry.sign_convention_declared:
            reasons.append(
                f"{SOURCE_SIGN_CONVENTION_REQUIRED}: asset {entry.asset.asset_id!r} has no "
                "declared source_sign_convention"
            )
        if entry.intrinsic_readiness.status == readiness.NOT_READY:
            reasons.extend(
                f"asset {entry.asset.asset_id!r}: {reason}"
                for reason in entry.intrinsic_readiness.reasons()
            )
    if any_ready:
        return CapabilityPlan(CANONICALIZE_BATHYMETRY, AVAILABLE, tuple(reasons), ())
    return CapabilityPlan(
        CANONICALIZE_BATHYMETRY,
        BLOCKED_MISSING_INPUT,
        tuple(reasons),
        (SOURCE_SIGN_CONVENTION_REQUIRED,),
    )


def canonicalize_bathymetry_executor(
    context: PlanningContext, plan: CapabilityPlan
) -> CapabilityExecutionOutcome:
    manifests: list[AnalysisProductManifest] = []
    for asset in _candidate_bathymetry_assets(context):
        declaration = asset.declaration
        if declaration is None or not declaration.source_sign_convention:
            continue  # not ready -- exact reason already reported by the plan, skip silently
        facts, band, crs, transform, nodata = _read_raster_facts(asset.path, declaration)
        intrinsic = readiness.assess_bathymetry_readiness(facts)
        if intrinsic.status == readiness.NOT_READY:
            continue

        canonical_raster = canonical.build_canonical_bed_elevation(
            band,
            nodata_value=nodata,
            source_sign_convention=declaration.source_sign_convention,
            source_vertical_datum=declaration.vertical_datum,
        )
        out_dir = Path("data/processed") / asset.asset_id / "terrain"
        tags = {
            "product": "MAR-034 generic canonicalize_bathymetry capability",
            "scientific_role": slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED,
            "layer": slope_stability_contract.SOURCE_TERRAIN_LAYER_REQUIRED,
            "units": "m",
            "convention": "higher=shallower",
            "source_sign_convention": canonical_raster.source_sign_convention,
            "source_vertical_datum": str(canonical_raster.source_vertical_datum),
            "source_survey_epoch": str(declaration.survey_epoch),
            "source_asset_id": asset.asset_id,
        }
        raster_path = raster_io.write_terrain_raster(
            canonical_raster.bed_elevation_m,
            transform,
            str(crs),
            out_dir / "canonical_bed_elevation.tif",
            tags,
        )
        manifest = AnalysisProductManifest(
            product_id=f"{asset.asset_id}__canonical_bed_elevation",
            capability_id=CANONICALIZE_BATHYMETRY,
            scientific_role=slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED,
            evidence_role="DERIVED",
            support_type="AREA_SURFACE",
            source_asset_ids=(asset.asset_id,),
            scenario_id=None,
            method_id=_METHOD_ID,
            units={"bed_elevation_m": "m"},
            primary_value_fields=("bed_elevation_m",),
            geometry_or_raster_path=str(raster_path),
            readiness=intrinsic.status,
            limitations=tuple(intrinsic.reasons()),
            display_name=f"Canonical Bed Elevation -- {asset.asset_id}",
        )
        write_product_manifest(manifest, out_dir / f"{manifest.product_id}.product_manifest.json")
        manifests.append(manifest)

    return CapabilityExecutionOutcome(
        capability_id=CANONICALIZE_BATHYMETRY, plan=plan, manifests=tuple(manifests), details=None
    )


def register_canonicalize_bathymetry_runtime() -> None:
    register_capability_runtime(
        CapabilityRuntime(
            definition=CAPABILITY_REGISTRY[CANONICALIZE_BATHYMETRY],
            readiness_adapter=canonicalize_bathymetry_readiness_adapter,
            planner_adapter=canonicalize_bathymetry_planner_adapter,
            executor=canonicalize_bathymetry_executor,
        )
    )

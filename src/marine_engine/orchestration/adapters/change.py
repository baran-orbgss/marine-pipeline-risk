"""Generic `observed_multi_epoch_seabed_change` capability runtime (MAR-034 Sections 22-28).

Registers the accepted MAR-021 `change` package (epoch_compatibility, alignment, common_support,
dod) behind the generic capability-orchestration runtime -- the SAME gates and functions the
standalone `build-seabed-change-poc` command already calls, no new science.

Epoch pairing/order comes ONLY from each canonical terrain source's own DECLARED `survey_epoch`
(never a filename, never a filesystem timestamp -- Section 23): a year is parsed out of the
declared text; if fewer than two sources carry a distinct, parseable year, order is reported as
`SURVEY_EPOCH_ORDER_NOT_ESTABLISHED` and change analysis stays blocked. Vertical-datum
compatibility is the accepted MAR-021 hard gate (Section 24): two epochs with an unknown or
undemonstrated-compatible datum are never change-analysis-ready, no assumption, no silent
"probably the same datum".

Change semantics are preserved exactly (Section 27): the product is the plain signed
`delta_bed_elevation_m` observation. Nothing here labels a cell "erosion"/"deposition", infers
causality, or produces a future prediction or risk class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from marine_engine.change import alignment, common_support, epoch_compatibility
from marine_engine.change import dod as change_dod
from marine_engine.orchestration.capability import (
    CAPABILITY_REGISTRY,
    OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
)
from marine_engine.orchestration.context import PlanningContext
from marine_engine.orchestration.planner import (
    AVAILABLE,
    BLOCKED_AMBIGUOUS_SEMANTICS,
    BLOCKED_INVALID_INPUT,
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
    "SURVEY_EPOCH_ORDER_NOT_ESTABLISHED",
    "ChangeEpochSource",
    "observed_multi_epoch_seabed_change_readiness_adapter",
    "observed_multi_epoch_seabed_change_planner_adapter",
    "observed_multi_epoch_seabed_change_executor",
    "register_observed_multi_epoch_seabed_change_runtime",
]

# Section 23: the exact, named blocker when fewer than two canonical terrain products carry a
# distinct, DECLARED survey epoch -- never resolved from a filename or filesystem timestamp.
SURVEY_EPOCH_ORDER_NOT_ESTABLISHED = "SURVEY_EPOCH_ORDER_NOT_ESTABLISHED"

_METHOD_ID = "MAR-021_MULTI_EPOCH_SEABED_CHANGE"
_YEAR_PATTERN = re.compile(r"(\d{4})")


def _extract_year(epoch_text: str | None) -> int | None:
    """A year parsed out of an operator-DECLARED survey_epoch string (e.g. "2020" or "November
    2020 - December 2020") -- narrow, deterministic parsing of an already-declared fact, never an
    inference from raster values, filename, or file timestamp."""

    if not epoch_text:
        return None
    match = _YEAR_PATTERN.search(epoch_text)
    return int(match.group(1)) if match else None


def _read_tags(path: Path) -> dict[str, str]:
    try:
        with rasterio.open(path) as dataset:
            return dict(dataset.tags())
    except Exception:  # noqa: BLE001 -- unreadable raster tags, treat as no tags at all
        return {}


@dataclass(frozen=True)
class ChangeEpochSource:
    """One candidate canonical bed-elevation epoch. `vertical_datum`/`survey_epoch` are the
    DECLARED facts for this asset -- from its `AssetDeclaration` when it is a recognized input, or
    recovered from the canonical raster's own embedded tags when it was produced earlier in this
    same run (Section 7: still declared provenance, never re-derived from the pixel values)."""

    source_asset_ids: tuple[str, ...]
    raster_path: Path
    vertical_datum: str | None
    survey_epoch: str | None
    epoch_year: int | None


def _canonical_terrain_epoch_sources(context: PlanningContext) -> tuple[ChangeEpochSource, ...]:
    sources: list[ChangeEpochSource] = []
    seen_paths: set[Path] = set()

    for asset in context.assets_with_role(slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED):
        declaration = asset.declaration
        vertical_datum = declaration.vertical_datum if declaration else None
        survey_epoch = declaration.survey_epoch if declaration else None
        sources.append(
            ChangeEpochSource(
                (asset.asset_id,),
                asset.path,
                vertical_datum,
                survey_epoch,
                _extract_year(survey_epoch),
            )
        )
        seen_paths.add(asset.path.resolve())

    for manifest in context.manifests_with_role(
        slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED
    ):
        if not manifest.geometry_or_raster_path:
            continue
        raster_path = Path(manifest.geometry_or_raster_path)
        if raster_path.resolve() in seen_paths:
            continue
        tags = _read_tags(raster_path)
        survey_epoch = tags.get("source_survey_epoch")
        if survey_epoch == "None":
            survey_epoch = None
        sources.append(
            ChangeEpochSource(
                manifest.source_asset_ids,
                raster_path,
                tags.get("source_vertical_datum"),
                survey_epoch,
                _extract_year(survey_epoch),
            )
        )
        seen_paths.add(raster_path.resolve())

    return tuple(sources)


def _select_epoch_pair(
    sources: tuple[ChangeEpochSource, ...],
) -> tuple[ChangeEpochSource | None, ChangeEpochSource | None, CapabilityPlan | None]:
    """Section 22-23/32: deterministic pairing from DECLARED epoch years only. Returns
    `(epoch1, epoch2, None)` on a clean pair, or `(None, None, blocking_plan)` otherwise."""

    dated = [s for s in sources if s.epoch_year is not None]
    distinct_years = sorted({s.epoch_year for s in dated})

    if len(distinct_years) < 2:
        return (
            None,
            None,
            CapabilityPlan(
                OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
                BLOCKED_MISSING_INPUT,
                (
                    f"{SURVEY_EPOCH_ORDER_NOT_ESTABLISHED}: fewer than two canonical terrain "
                    "products carry a distinct, declared survey_epoch year",
                ),
                (SURVEY_EPOCH_ORDER_NOT_ESTABLISHED,),
            ),
        )
    if len(distinct_years) > 2:
        return (
            None,
            None,
            CapabilityPlan(
                OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
                BLOCKED_AMBIGUOUS_SEMANTICS,
                (
                    f"more than two distinct declared survey epochs are available "
                    f"{distinct_years} -- exactly two are required and no automatic pair is "
                    "selected",
                ),
                (),
            ),
        )

    epoch1_candidates = [s for s in dated if s.epoch_year == distinct_years[0]]
    epoch2_candidates = [s for s in dated if s.epoch_year == distinct_years[1]]
    if len(epoch1_candidates) != 1 or len(epoch2_candidates) != 1:
        return (
            None,
            None,
            CapabilityPlan(
                OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
                BLOCKED_AMBIGUOUS_SEMANTICS,
                (
                    "more than one canonical terrain product shares the same declared "
                    "survey_epoch year -- exact pairing is ambiguous, no candidate is chosen",
                ),
                (),
            ),
        )
    return epoch1_candidates[0], epoch2_candidates[0], None


def observed_multi_epoch_seabed_change_readiness_adapter(
    context: PlanningContext,
) -> tuple[ChangeEpochSource, ...] | None:
    sources = _canonical_terrain_epoch_sources(context)
    return sources or None


def observed_multi_epoch_seabed_change_planner_adapter(
    context: PlanningContext, readiness_facts: tuple[ChangeEpochSource, ...] | None
) -> CapabilityPlan:
    if readiness_facts is None or len(readiness_facts) < 2:
        return CapabilityPlan(
            OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
            NOT_APPLICABLE,
            ("fewer than two canonical terrain products are available",),
            (
                "a second compatible canonical "
                f"{slope_stability_contract.SOURCE_TERRAIN_LAYER_REQUIRED} epoch",
            ),
        )

    epoch1, epoch2, blocked_plan = _select_epoch_pair(readiness_facts)
    if blocked_plan is not None:
        return blocked_plan

    datum_result = epoch_compatibility.assess_vertical_datum_compatibility(
        epoch1.vertical_datum, epoch2.vertical_datum
    )
    if datum_result.status != epoch_compatibility.VERTICAL_DATUM_HARMONIZED:
        return CapabilityPlan(
            OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
            BLOCKED_MISSING_INPUT,
            (f"{datum_result.status}: {datum_result.reason}",),
            ("demonstrated vertical-datum compatibility between the two epochs",),
        )

    try:
        with rasterio.open(epoch1.raster_path) as src1:
            crs1, transform1 = src1.crs, src1.transform
        with rasterio.open(epoch2.raster_path) as src2:
            crs2, transform2 = src2.crs, src2.transform
    except Exception as exc:  # noqa: BLE001 -- an unreadable raster is a controlled blocker
        return CapabilityPlan(
            OBSERVED_MULTI_EPOCH_SEABED_CHANGE, BLOCKED_INVALID_INPUT, (str(exc),), ()
        )

    grid_result = epoch_compatibility.classify_grid_alignment(
        crs1=str(crs1), transform1=transform1, crs2=str(crs2), transform2=transform2
    )
    if grid_result.status == epoch_compatibility.INCOMPATIBLE_HORIZONTAL_REFERENCE:
        return CapabilityPlan(
            OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
            BLOCKED_MISSING_INPUT,
            (f"{grid_result.status}: {grid_result.reason}",),
            (),
        )

    return CapabilityPlan(OBSERVED_MULTI_EPOCH_SEABED_CHANGE, AVAILABLE, (), ())


def observed_multi_epoch_seabed_change_executor(
    context: PlanningContext, plan: CapabilityPlan
) -> CapabilityExecutionOutcome:
    sources = _canonical_terrain_epoch_sources(context)
    epoch1, epoch2, blocked_plan = _select_epoch_pair(sources)
    if blocked_plan is not None or epoch1 is None or epoch2 is None:
        return CapabilityExecutionOutcome(
            capability_id=OBSERVED_MULTI_EPOCH_SEABED_CHANGE, plan=plan, manifests=(), details=None
        )

    with rasterio.open(epoch1.raster_path) as src1:
        band1, crs1, transform1 = src1.read(1), src1.crs, src1.transform
    with rasterio.open(epoch2.raster_path) as src2:
        band2, crs2, transform2 = src2.read(1), src2.crs, src2.transform
    valid1, valid2 = np.isfinite(band1), np.isfinite(band2)

    grid_result = epoch_compatibility.classify_grid_alignment(
        crs1=str(crs1), transform1=transform1, crs2=str(crs2), transform2=transform2
    )
    if grid_result.status == epoch_compatibility.INCOMPATIBLE_HORIZONTAL_REFERENCE:
        return CapabilityExecutionOutcome(
            capability_id=OBSERVED_MULTI_EPOCH_SEABED_CHANGE, plan=plan, manifests=(), details=None
        )

    aligned1, aligned2 = alignment.align_to_common_grid(
        classification=grid_result,
        elevation1=band1,
        valid1=valid1,
        transform1=transform1,
        elevation2=band2,
        valid2=valid2,
        transform2=transform2,
        crs=str(crs2),
    )
    support = common_support.build_common_valid_support(aligned1.valid_mask, aligned2.valid_mask)
    dod_result = change_dod.compute_delta_bed_elevation(
        aligned1.elevation, aligned2.elevation, support.common_valid_mask
    )

    epoch1_id, epoch2_id = epoch1.source_asset_ids[0], epoch2.source_asset_ids[0]
    out_dir = Path("data/processed") / f"{epoch1_id}__{epoch2_id}" / "change"
    tags = {
        "product": "MAR-034 generic observed_multi_epoch_seabed_change capability",
        "scientific_role": change_dod.MULTI_EPOCH_SEABED_CHANGE_POC,
        "layer": "delta_bed_elevation_m",
        "units": "m",
        "epoch1_asset_id": epoch1_id,
        "epoch2_asset_id": epoch2_id,
    }
    from marine_engine.terrain import raster_io

    raster_path = raster_io.write_terrain_raster(
        dod_result.delta_bed_elevation_m,
        aligned2.transform,
        str(crs2),
        out_dir / "delta_bed_elevation_m.tif",
        tags,
    )
    manifest = AnalysisProductManifest(
        product_id=f"{epoch1_id}__{epoch2_id}__observed_seabed_change",
        capability_id=OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
        scientific_role=change_dod.MULTI_EPOCH_SEABED_CHANGE_POC,
        evidence_role="DERIVED",
        support_type="AREA_SURFACE",
        source_asset_ids=tuple(epoch1.source_asset_ids) + tuple(epoch2.source_asset_ids),
        scenario_id=None,
        method_id=_METHOD_ID,
        units={"delta_bed_elevation_m": "m"},
        primary_value_fields=("delta_bed_elevation_m",),
        geometry_or_raster_path=str(raster_path),
        readiness=AVAILABLE,
        limitations=(
            f"definition: {dod_result.definition}",
            f"earlier_epoch={epoch1.survey_epoch!r} ({epoch1_id})",
            f"later_epoch={epoch2.survey_epoch!r} ({epoch2_id})",
            f"vertical_datum_evidence: epoch1={epoch1.vertical_datum!r}, "
            f"epoch2={epoch2.vertical_datum!r}",
            f"grid_alignment_method: {grid_result.status}",
            f"common_valid_cell_count={support.common_valid_cell_count}",
            f"{change_dod.OBSERVED_SEABED_ELEVATION_CHANGE} only -- no erosion/deposition or "
            "other causal label, no risk class, no future prediction.",
        ),
        display_name=f"Observed Multi-Epoch Seabed Change -- {epoch1_id} to {epoch2_id}",
    )
    write_product_manifest(manifest, out_dir / f"{manifest.product_id}.product_manifest.json")

    fresh_plan = CapabilityPlan(OBSERVED_MULTI_EPOCH_SEABED_CHANGE, AVAILABLE, (), ())
    return CapabilityExecutionOutcome(
        capability_id=OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
        plan=fresh_plan,
        manifests=(manifest,),
        details=dod_result,
    )


def register_observed_multi_epoch_seabed_change_runtime() -> None:
    register_capability_runtime(
        CapabilityRuntime(
            definition=CAPABILITY_REGISTRY[OBSERVED_MULTI_EPOCH_SEABED_CHANGE],
            readiness_adapter=observed_multi_epoch_seabed_change_readiness_adapter,
            planner_adapter=observed_multi_epoch_seabed_change_planner_adapter,
            executor=observed_multi_epoch_seabed_change_executor,
        )
    )

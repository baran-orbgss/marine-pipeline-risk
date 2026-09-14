"""Capability execution (MAR-033 Sections 25, 33-39; MAR-033A Part B).

`execute_earthquake_cpt_liquefaction_triggering` is the ONE place that actually runs the
capability: recognize -> build readiness facts -> plan -> (only when the plan says `AVAILABLE`)
compute the full Section 21 profile for every CPT test in the evidence -> Section 22 point
summary -> product manifests. No free-form command execution occurs anywhere in this module
(Section 33): every step is a direct Python function call into `geotechnical`/`liquefaction`,
never a shell-out, and a blocked/not-applicable plan performs no scientific write at all.

MAR-033A registers this as the FIRST capability behind the generic runtime registry
(`orchestration.runtime`): `earthquake_triggering_readiness_adapter` /
`earthquake_triggering_planner_adapter` / `earthquake_triggering_executor` are thin,
capability-specific adapters wrapping the functions above around the generic `PlanningContext` --
`auto-process` and every other generic caller dispatch through those adapters (via
`orchestration.runtime.CAPABILITY_RUNTIMES`), never by calling
`execute_earthquake_cpt_liquefaction_triggering` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from marine_engine.geotechnical import cpt_contract
from marine_engine.geotechnical.cpt_readiness import CptEvidenceFacts
from marine_engine.intake.recognition import UNCLASSIFIED, RecognitionDecision
from marine_engine.intake.registry import inspect_and_recognize
from marine_engine.liquefaction import contract as liq_contract
from marine_engine.liquefaction import manifest as liq_manifest
from marine_engine.liquefaction import products
from marine_engine.liquefaction.earthquake_triggering import (
    GENERAL_CORRELATION_SENSITIVITY_VARIANT_TAGS as _VARIANT_TAGS,
)
from marine_engine.liquefaction.earthquake_triggering import (
    TRIGGERING_PROFILE_COLUMNS,
    FinesDeclaration,
    SoilApplicabilityDeclaration,
    TipResistanceDeclaration,
    evaluate_triggering_profile,
    expand_general_correlation_sensitivity_fines,
)
from marine_engine.liquefaction.readiness import (
    EarthquakeTriggeringReadinessFacts,
)
from marine_engine.liquefaction.scenario import EarthquakeScenario
from marine_engine.liquefaction.stress import StressModel
from marine_engine.orchestration.capability import (
    CAPABILITY_REGISTRY,
    EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING,
)
from marine_engine.orchestration.context import PlanningContext, RecognizedAsset
from marine_engine.orchestration.declaration import DeclarationRequest
from marine_engine.orchestration.planner import (
    AVAILABLE,
    CapabilityPlan,
    plan_earthquake_cpt_liquefaction_triggering,
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
from marine_engine.project.categories import MEASURED

__all__ = [
    "CptEvidenceInspection",
    "inspect_cpt_evidence",
    "build_readiness_facts",
    "ExecutionResult",
    "execute_earthquake_cpt_liquefaction_triggering",
    "EarthquakeTriggeringDeclaration",
    "earthquake_triggering_declaration_adapter",
    "earthquake_triggering_readiness_adapter",
    "earthquake_triggering_planner_adapter",
    "earthquake_triggering_executor",
    "register_earthquake_triggering_runtime",
]


@dataclass(frozen=True)
class CptEvidenceInspection:
    """What Sections 24/33 need before anything can be planned: the recognition decision plus the
    same `CptEvidenceFacts` `geotechnical.cpt_readiness` already knows how to build from real
    bytes -- reused via `project.cpt_adapter.inspect_cpt_asset`, never re-derived."""

    path: Path
    recognition: RecognitionDecision
    cpt_evidence_facts: CptEvidenceFacts
    dataframe: pd.DataFrame | None
    marker_evidence_id: str | None

    @property
    def evidence_id(self) -> str:
        """The canonical product's OWN declared evidence id (from its verified marker) when one
        exists; the file stem is a last-resort fallback only for an unrecognized/unmarked input,
        never used as identity for anything that actually verified (Section 14.4: content
        identity is content-based, never filename-based)."""

        return self.marker_evidence_id or self.path.stem


def inspect_cpt_evidence(path: str | Path) -> CptEvidenceInspection:
    """Fingerprints and recognizes `path`, then assembles the MAR-032 readiness facts through the
    existing, unmodified project adapter. `declared_evidence_role=MEASURED` is passed
    unconditionally: it only takes effect once the file's OWN canonical marker and lineage
    independently verify inside `inspect_cpt_asset` -- it can never manufacture measured-evidence
    status for a file that does not already carry it (Section 24: "do not require a user to
    manually tell the engine 'this is CPT' if canonical identity already establishes it")."""

    from marine_engine.project.cpt_adapter import inspect_cpt_asset

    resolved = Path(path)
    _fingerprint, recognition = inspect_and_recognize(resolved)
    facts, observed = inspect_cpt_asset(resolved, declared_evidence_role=MEASURED)
    dataframe = pd.read_parquet(resolved) if facts.canonical_profile_created else None
    marker_evidence_id = None
    if facts.canonical_product_identity_verified:
        marker_evidence_id = observed.get("canonical_product_marker", {}).get("evidence_id")
    return CptEvidenceInspection(
        path=resolved,
        recognition=recognition,
        cpt_evidence_facts=facts,
        dataframe=dataframe,
        marker_evidence_id=marker_evidence_id,
    )


def build_readiness_facts(
    inspection: CptEvidenceInspection,
    *,
    tip_resistance: TipResistanceDeclaration | None,
    stress_model: StressModel | None,
    stress_model_problem: str | None,
    fines: FinesDeclaration | None,
    soil_applicability: SoilApplicabilityDeclaration | None,
    scenario: EarthquakeScenario | None,
    static_shear_material: bool = False,
) -> EarthquakeTriggeringReadinessFacts:
    return EarthquakeTriggeringReadinessFacts(
        cpt_evidence_facts=inspection.cpt_evidence_facts,
        tip_resistance_declared=tip_resistance,
        stress_model_declared=stress_model is not None or stress_model_problem is not None,
        stress_model_valid=stress_model is not None,
        stress_model_problem=stress_model_problem,
        fines_declared=fines,
        soil_applicability_declared=soil_applicability,
        scenario_declared=scenario,
        static_shear_material=static_shear_material,
    )


@dataclass(frozen=True)
class ExecutionResult:
    plan: CapabilityPlan
    profile_df: pd.DataFrame | None
    point_summary_df: pd.DataFrame | None
    point_analysis_gdf: gpd.GeoDataFrame | None
    point_analysis_findings: tuple[str, ...]
    manifests: tuple[AnalysisProductManifest, ...]


def _compute_profile_and_point_products(
    inspection: CptEvidenceInspection,
    *,
    evidence_id: str,
    scenario_id: str,
    scenario: EarthquakeScenario,
    tip_resistance: TipResistanceDeclaration | None,
    stress_model: StressModel | None,
    fines: FinesDeclaration | None,
    soil_applicability: SoilApplicabilityDeclaration | None,
    static_shear_material: bool,
    locations_gdf: gpd.GeoDataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame | None, tuple[str, ...]]:
    """Sections 21-22: the row-level profile plus its point summary/GIS layer for ONE `scenario_id`
    -- which may be a GENERAL_CORRELATION_SENSITIVITY variant's own distinct id (see
    `_execute_general_correlation_sensitivity`), never the SAME computation reused across
    variants."""

    assert inspection.dataframe is not None
    profiles = [
        evaluate_triggering_profile(
            group,
            evidence_id=evidence_id,
            scenario_id=scenario_id,
            scenario=scenario,
            tip_resistance=tip_resistance,
            stress_model=stress_model,
            fines=fines,
            soil_applicability=soil_applicability,
            static_shear_material=static_shear_material,
        )
        for _test_id, group in inspection.dataframe.groupby(cpt_contract.TEST_ID, sort=True)
    ]
    profile_df = (
        pd.concat(profiles, ignore_index=True)
        if profiles
        else pd.DataFrame(columns=list(TRIGGERING_PROFILE_COLUMNS))
    )
    point_summary_df = products.build_point_summary_df(profile_df)

    point_analysis_gdf: gpd.GeoDataFrame | None = None
    point_analysis_findings: tuple[str, ...] = ()
    if locations_gdf is not None:
        point_analysis_gdf, findings = products.build_point_analysis_gdf(
            point_summary_df, locations_gdf
        )
        point_analysis_findings = tuple(findings)
    return profile_df, point_summary_df, point_analysis_gdf, point_analysis_findings


def _write_profile_and_point_manifests(
    *,
    out_dir: Path,
    evidence_id: str,
    scenario_id: str,
    profile_df: pd.DataFrame,
    point_summary_df: pd.DataFrame,
    point_analysis_gdf: gpd.GeoDataFrame | None,
    readiness: str,
    file_prefix: str,
    display_suffix: str = "",
) -> list[AnalysisProductManifest]:
    """Sections 36-37: writes the profile parquet, point-summary parquet, optional point
    GeoPackage, and their two `AnalysisProductManifest`s under `out_dir`. `file_prefix` keeps a
    GENERAL_CORRELATION_SENSITIVITY variant's files from ever colliding with a sibling variant's
    (each variant gets its own traceable, independently-inspectable set of files)."""

    profile_path = out_dir / f"{file_prefix}_profile.parquet"
    products.profile_df_for_parquet(profile_df).to_parquet(profile_path, index=False)
    point_summary_path = out_dir / f"{file_prefix}_point_summary.parquet"
    point_summary_df.to_parquet(point_summary_path, index=False)
    point_gpkg_path: Path | None = None
    if point_analysis_gdf is not None and not point_analysis_gdf.empty:
        point_gpkg_path = out_dir / f"{file_prefix}_points.gpkg"
        point_analysis_gdf.to_file(point_gpkg_path, driver="GPKG", layer=f"{file_prefix}_points")

    common = {
        "capability_id": EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING,
        "source_asset_ids": (evidence_id,),
        "scenario_id": scenario_id,
        "method_id": liq_contract.METHOD_ID,
        "readiness": readiness,
        "limitations": (),
    }
    manifests = [
        AnalysisProductManifest(
            product_id=f"{evidence_id}__{scenario_id}__triggering_profile",
            scientific_role=liq_contract.EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING_SCREENING_PROFILE,
            evidence_role="DERIVED",
            support_type="PROFILE",
            units={"FS_liq": "dimensionless", "depth_bsf_m": "m"},
            primary_value_fields=("FS_liq", "evaluation_state"),
            geometry_or_raster_path=str(profile_path),
            display_name=f"Earthquake Liquefaction Triggering -- CPT Profile{display_suffix}",
            semantic_warning="Model factor-of-safety screening; not a risk class or probability.",
            **common,
        ),
        AnalysisProductManifest(
            product_id=f"{evidence_id}__{scenario_id}__triggering_points",
            scientific_role=liq_contract.POINT_ANALYSIS,
            evidence_role="DERIVED",
            support_type=liq_contract.POINT_ANALYSIS,
            units={"minimum_model_fs_liq": "dimensionless"},
            primary_value_fields=("minimum_model_fs_liq",),
            geometry_or_raster_path=(
                str(point_gpkg_path) if point_gpkg_path else str(point_summary_path)
            ),
            display_name=f"Earthquake Liquefaction Triggering -- CPT Points{display_suffix}",
            semantic_warning=products.SPATIAL_SUPPORT_STATEMENT,
            **common,
        ),
    ]
    for manifest in manifests:
        write_product_manifest(manifest, out_dir / f"{manifest.product_id}.product_manifest.json")
    return manifests


def _execute_general_correlation_sensitivity(
    inspection: CptEvidenceInspection,
    *,
    evidence_id: str,
    scenario: EarthquakeScenario,
    tip_resistance: TipResistanceDeclaration | None,
    stress_model: StressModel | None,
    fines: FinesDeclaration,
    soil_applicability: SoilApplicabilityDeclaration | None,
    static_shear_material: bool,
    locations_gdf: gpd.GeoDataFrame | None,
    out_dir: Path | None,
    plan: CapabilityPlan,
) -> ExecutionResult:
    """MAR-033A Part A item 6: GENERAL_CORRELATION_SENSITIVITY fans out into THREE fully
    independent profile computations, one per literature C_FC value -- never averaged, never
    reduced to one value, never turned into a probability or an auto-selected "best" result. Each
    variant gets its own scenario/variant identity (Section 6), its own profile/point-summary/
    product manifest, all sharing the common parent `scenario.scenario_id` as a prefix.
    """

    variant_fines = expand_general_correlation_sensitivity_fines(fines)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    profile_frames: list[pd.DataFrame] = []
    point_summary_frames: list[pd.DataFrame] = []
    point_analysis_frames: list[gpd.GeoDataFrame] = []
    point_analysis_findings: list[str] = []
    manifests: list[AnalysisProductManifest] = []

    for variant in variant_fines:
        assert variant.c_fc is not None
        tag = _VARIANT_TAGS[variant.c_fc]
        variant_scenario_id = f"{scenario.scenario_id}__{tag}"
        profile_df, point_summary_df, point_analysis_gdf, findings = (
            _compute_profile_and_point_products(
                inspection,
                evidence_id=evidence_id,
                scenario_id=variant_scenario_id,
                scenario=scenario,
                tip_resistance=tip_resistance,
                stress_model=stress_model,
                fines=variant,
                soil_applicability=soil_applicability,
                static_shear_material=static_shear_material,
                locations_gdf=locations_gdf,
            )
        )
        profile_frames.append(profile_df)
        point_summary_frames.append(point_summary_df)
        if point_analysis_gdf is not None:
            point_analysis_frames.append(point_analysis_gdf)
        point_analysis_findings.extend(findings)

        if out_dir is not None:
            manifests.extend(
                _write_profile_and_point_manifests(
                    out_dir=out_dir,
                    evidence_id=evidence_id,
                    scenario_id=variant_scenario_id,
                    profile_df=profile_df,
                    point_summary_df=point_summary_df,
                    point_analysis_gdf=point_analysis_gdf,
                    readiness=plan.status,
                    file_prefix=f"earthquake_liquefaction_triggering__{tag}",
                    display_suffix=(
                        f" (GENERAL_CORRELATION_SENSITIVITY variant C_FC={variant.c_fc:+.2f}, "
                        f"parent scenario {scenario.scenario_id})"
                    ),
                )
            )

    combined_profile_df = pd.concat(profile_frames, ignore_index=True)
    combined_point_summary_df = pd.concat(point_summary_frames, ignore_index=True)
    combined_point_analysis_gdf: gpd.GeoDataFrame | None = None
    if point_analysis_frames:
        combined_point_analysis_gdf = gpd.GeoDataFrame(
            pd.concat(point_analysis_frames, ignore_index=True), crs=point_analysis_frames[0].crs
        )

    return ExecutionResult(
        plan=plan,
        profile_df=combined_profile_df,
        point_summary_df=combined_point_summary_df,
        point_analysis_gdf=combined_point_analysis_gdf,
        point_analysis_findings=tuple(point_analysis_findings),
        manifests=tuple(manifests),
    )


def execute_earthquake_cpt_liquefaction_triggering(
    inspection: CptEvidenceInspection,
    *,
    evidence_id: str,
    scenario: EarthquakeScenario | None,
    tip_resistance: TipResistanceDeclaration | None,
    stress_model: StressModel | None,
    stress_model_problem: str | None,
    fines: FinesDeclaration | None,
    soil_applicability: SoilApplicabilityDeclaration | None,
    static_shear_material: bool = False,
    locations_gdf: gpd.GeoDataFrame | None = None,
    out_dir: Path | None = None,
) -> ExecutionResult:
    """Section 33: plan first; execute the safe local computation ONLY when the plan says
    `AVAILABLE`. A blocked/not-applicable plan still returns cleanly, with no profile/point
    products and no files written -- a controlled STOP, never an improvised partial computation.
    """

    readiness_facts = build_readiness_facts(
        inspection,
        tip_resistance=tip_resistance,
        stress_model=stress_model,
        stress_model_problem=stress_model_problem,
        fines=fines,
        soil_applicability=soil_applicability,
        scenario=scenario,
        static_shear_material=static_shear_material,
    )
    plan = plan_earthquake_cpt_liquefaction_triggering(
        recognition=inspection.recognition, readiness_facts=readiness_facts
    )
    if plan.status != AVAILABLE:
        return ExecutionResult(
            plan=plan,
            profile_df=None,
            point_summary_df=None,
            point_analysis_gdf=None,
            point_analysis_findings=(),
            manifests=(),
        )

    assert inspection.dataframe is not None
    assert scenario is not None

    if fines is not None and fines.source == liq_contract.GENERAL_CORRELATION_SENSITIVITY:
        return _execute_general_correlation_sensitivity(
            inspection,
            evidence_id=evidence_id,
            scenario=scenario,
            tip_resistance=tip_resistance,
            stress_model=stress_model,
            fines=fines,
            soil_applicability=soil_applicability,
            static_shear_material=static_shear_material,
            locations_gdf=locations_gdf,
            out_dir=out_dir,
            plan=plan,
        )

    profile_df, point_summary_df, point_analysis_gdf, point_analysis_findings = (
        _compute_profile_and_point_products(
            inspection,
            evidence_id=evidence_id,
            scenario_id=scenario.scenario_id,
            scenario=scenario,
            tip_resistance=tip_resistance,
            stress_model=stress_model,
            fines=fines,
            soil_applicability=soil_applicability,
            static_shear_material=static_shear_material,
            locations_gdf=locations_gdf,
        )
    )

    manifests: list[AnalysisProductManifest] = []
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifests = _write_profile_and_point_manifests(
            out_dir=out_dir,
            evidence_id=evidence_id,
            scenario_id=scenario.scenario_id,
            profile_df=profile_df,
            point_summary_df=point_summary_df,
            point_analysis_gdf=point_analysis_gdf,
            readiness=plan.status,
            file_prefix="earthquake_liquefaction_triggering",
        )

    return ExecutionResult(
        plan=plan,
        profile_df=profile_df,
        point_summary_df=point_summary_df,
        point_analysis_gdf=point_analysis_gdf,
        point_analysis_findings=point_analysis_findings,
        manifests=tuple(manifests),
    )


# --- MAR-033A Part B: generic capability-runtime registration -------------------------------------


@dataclass(frozen=True)
class EarthquakeTriggeringDeclaration:
    """The capability-specific declaration this capability's runtime adapters expect to find (or
    not find -- every field defaults to the honest "nothing declared" state) at
    `PlanningContext.capability_declarations[EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING]`. Opaque to
    the generic planner/executor (Section 9); only this module's own adapters below interpret it.
    """

    scenario: EarthquakeScenario | None = None
    tip_resistance: TipResistanceDeclaration | None = None
    stress_model: StressModel | None = None
    stress_model_problem: str | None = None
    fines: FinesDeclaration | None = None
    soil_applicability: SoilApplicabilityDeclaration | None = None
    static_shear_material: bool = False
    evidence_id: str | None = None
    locations_gdf: gpd.GeoDataFrame | None = None
    out_dir: Path | None = None


def earthquake_triggering_declaration_adapter(
    request: DeclarationRequest,
) -> EarthquakeTriggeringDeclaration | None:
    """MAR-034 Section 8: capability-specific declaration PARSING, moved here from `cli.py`'s
    former `_build_earthquake_triggering_declaration` (which the generic `_cmd_auto_process`/
    `_cmd_plan_processing` called directly). The generic CLI now hands every registered
    capability the SAME raw `DeclarationRequest`; only this capability's own adapter knows what
    `--scenario-manifest`/`--locations`/`--evidence-id`/`--out` mean for it. Returns `None`
    (never a fabricated declaration) when no scenario manifest was supplied."""

    if request.scenario_manifest_path is None:
        return None
    raw_manifest = liq_manifest.load_liquefaction_scenario_manifest(request.scenario_manifest_path)
    scenario, tip_resistance, stress_model, fines, soil_applicability = raw_manifest.to_core()
    locations_gdf = (
        gpd.read_file(request.locations_path) if request.locations_path is not None else None
    )
    return EarthquakeTriggeringDeclaration(
        scenario=scenario,
        tip_resistance=tip_resistance,
        stress_model=stress_model,
        fines=fines,
        soil_applicability=soil_applicability,
        static_shear_material=raw_manifest.static_shear_material,
        evidence_id=request.evidence_id,
        locations_gdf=locations_gdf,
        out_dir=request.out_dir,
    )


def _matched_cpt_asset(context: PlanningContext) -> RecognizedAsset | None:
    matched = context.assets_with_role(cpt_contract.MEASURED_CPT_CPTU_PROFILE)
    return matched[0] if matched else None


def earthquake_triggering_readiness_adapter(
    context: PlanningContext,
) -> EarthquakeTriggeringReadinessFacts | None:
    """Section 10: builds this capability's own typed readiness facts from the generic context, or
    `None` when no supplied input recognized the required CPT role at all -- mirroring
    `plan_earthquake_cpt_liquefaction_triggering`'s existing "no facts supplied" contract."""

    asset = _matched_cpt_asset(context)
    if asset is None:
        return None
    declaration = context.capability_declarations.get(
        EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING, EarthquakeTriggeringDeclaration()
    )
    inspection = inspect_cpt_evidence(asset.path)
    return build_readiness_facts(
        inspection,
        tip_resistance=declaration.tip_resistance,
        stress_model=declaration.stress_model,
        stress_model_problem=declaration.stress_model_problem,
        fines=declaration.fines,
        soil_applicability=declaration.soil_applicability,
        scenario=declaration.scenario,
        static_shear_material=declaration.static_shear_material,
    )


def earthquake_triggering_planner_adapter(
    context: PlanningContext, readiness_facts: EarthquakeTriggeringReadinessFacts | None
) -> CapabilityPlan:
    """Section 10: wraps the existing capability-specific planning function, resolving generically
    which supplied input's `RecognitionDecision` to plan against."""

    asset = _matched_cpt_asset(context)
    if asset is not None:
        recognition_decision = asset.recognition
    elif len(context.recognized_assets) == 1:
        # Exactly one input was supplied and it did not recognize as the required role -- reuse
        # ITS recognition decision so UNCLASSIFIED/AMBIGUOUS/INVALID is reported precisely
        # (matching the single-file `plan-processing`/`auto-process` behaviour MAR-033 shipped).
        recognition_decision = context.recognized_assets[0].recognition
    else:
        recognition_decision = RecognitionDecision(
            state=UNCLASSIFIED,
            candidates=(),
            recognized_role=None,
            reasons=(
                f"no supplied input recognized as {cpt_contract.MEASURED_CPT_CPTU_PROFILE!r}",
            ),
        )
    return plan_earthquake_cpt_liquefaction_triggering(
        recognition=recognition_decision, readiness_facts=readiness_facts
    )


def earthquake_triggering_executor(
    context: PlanningContext, plan: CapabilityPlan
) -> CapabilityExecutionOutcome:
    """Section 10: wraps the existing capability-specific executor. Reached by
    `orchestration.runtime.execute_plan` only when a FRESH plan already says AVAILABLE;
    `execute_earthquake_cpt_liquefaction_triggering` itself still independently refuses to compute
    on anything but its own AVAILABLE plan, rather than trusting the caller."""

    declaration = context.capability_declarations.get(
        EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING, EarthquakeTriggeringDeclaration()
    )
    asset = _matched_cpt_asset(context)
    if asset is None:
        return CapabilityExecutionOutcome(
            capability_id=EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING,
            plan=plan,
            manifests=(),
            details=None,
        )
    inspection = inspect_cpt_evidence(asset.path)
    evidence_id = declaration.evidence_id or inspection.evidence_id
    # Preserve MAR-033's established default output location (also used, independently, by the
    # `build-cpt-earthquake-liquefaction-poc` direct entry point) when the caller declared none --
    # this capability-specific default belongs in this capability's own adapter, not in the
    # generic `auto-process` CLI path.
    out_dir = declaration.out_dir or Path("data/processed") / evidence_id / "liquefaction"
    result = execute_earthquake_cpt_liquefaction_triggering(
        inspection,
        evidence_id=evidence_id,
        scenario=declaration.scenario,
        tip_resistance=declaration.tip_resistance,
        stress_model=declaration.stress_model,
        stress_model_problem=declaration.stress_model_problem,
        fines=declaration.fines,
        soil_applicability=declaration.soil_applicability,
        static_shear_material=declaration.static_shear_material,
        locations_gdf=declaration.locations_gdf,
        out_dir=out_dir,
    )
    return CapabilityExecutionOutcome(
        capability_id=EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING,
        plan=result.plan,
        manifests=result.manifests,
        details=result,
    )


def register_earthquake_triggering_runtime() -> None:
    """MAR-034 Section 10: registration is now an explicit call the engine bootstrap makes
    (`orchestration.bootstrap.register_builtin_runtimes`), never an import-time side effect --
    merely importing this module no longer registers anything. `register_builtin_runtimes` itself
    is idempotent; `register_capability_runtime` still refuses an outright duplicate
    registration, so this function must not be called more than once directly."""

    register_capability_runtime(
        CapabilityRuntime(
            definition=CAPABILITY_REGISTRY[EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING],
            readiness_adapter=earthquake_triggering_readiness_adapter,
            planner_adapter=earthquake_triggering_planner_adapter,
            executor=earthquake_triggering_executor,
            declaration_adapter=earthquake_triggering_declaration_adapter,
        )
    )

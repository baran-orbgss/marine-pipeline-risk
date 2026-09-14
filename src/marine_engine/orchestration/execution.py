"""Capability execution (MAR-033 Sections 25, 33-39).

`execute_earthquake_cpt_liquefaction_triggering` is the ONE place that actually runs the
capability: recognize -> build readiness facts -> plan -> (only when the plan says `AVAILABLE`)
compute the full Section 21 profile for every CPT test in the evidence -> Section 22 point
summary -> product manifests. No free-form command execution occurs anywhere in this module
(Section 33): every step is a direct Python function call into `geotechnical`/`liquefaction`,
never a shell-out, and a blocked/not-applicable plan performs no scientific write at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from marine_engine.geotechnical import cpt_contract
from marine_engine.geotechnical.cpt_readiness import CptEvidenceFacts
from marine_engine.intake.recognition import RecognitionDecision
from marine_engine.intake.registry import inspect_and_recognize
from marine_engine.liquefaction import contract as liq_contract
from marine_engine.liquefaction import products
from marine_engine.liquefaction.earthquake_triggering import (
    TRIGGERING_PROFILE_COLUMNS,
    FinesDeclaration,
    SoilApplicabilityDeclaration,
    TipResistanceDeclaration,
    evaluate_triggering_profile,
)
from marine_engine.liquefaction.readiness import (
    EarthquakeTriggeringReadinessFacts,
)
from marine_engine.liquefaction.scenario import EarthquakeScenario
from marine_engine.liquefaction.stress import StressModel
from marine_engine.orchestration.capability import EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING
from marine_engine.orchestration.planner import (
    AVAILABLE,
    CapabilityPlan,
    plan_earthquake_cpt_liquefaction_triggering,
)
from marine_engine.orchestration.product_manifest import (
    AnalysisProductManifest,
    write_product_manifest,
)
from marine_engine.project.categories import MEASURED

__all__ = [
    "CptEvidenceInspection",
    "inspect_cpt_evidence",
    "build_readiness_facts",
    "ExecutionResult",
    "execute_earthquake_cpt_liquefaction_triggering",
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

    profiles = [
        evaluate_triggering_profile(
            group,
            evidence_id=evidence_id,
            scenario_id=scenario.scenario_id,
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

    manifests: list[AnalysisProductManifest] = []
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        profile_path = out_dir / "earthquake_liquefaction_triggering_profile.parquet"
        products.profile_df_for_parquet(profile_df).to_parquet(profile_path, index=False)
        point_summary_path = out_dir / "earthquake_liquefaction_triggering_point_summary.parquet"
        point_summary_df.to_parquet(point_summary_path, index=False)
        point_gpkg_path: Path | None = None
        if point_analysis_gdf is not None and not point_analysis_gdf.empty:
            point_gpkg_path = out_dir / "earthquake_liquefaction_triggering_points.gpkg"
            point_analysis_gdf.to_file(
                point_gpkg_path, driver="GPKG", layer="earthquake_liquefaction_triggering_points"
            )

        common = {
            "capability_id": EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING,
            "source_asset_ids": (evidence_id,),
            "scenario_id": scenario.scenario_id,
            "method_id": liq_contract.METHOD_ID,
            "readiness": plan.status,
            "limitations": (),
        }
        manifests.append(
            AnalysisProductManifest(
                product_id=f"{evidence_id}__{scenario.scenario_id}__triggering_profile",
                scientific_role=liq_contract.EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING_SCREENING_PROFILE,
                evidence_role="DERIVED",
                support_type="PROFILE",
                units={"FS_liq": "dimensionless", "depth_bsf_m": "m"},
                primary_value_fields=("FS_liq", "evaluation_state"),
                geometry_or_raster_path=str(profile_path),
                display_name="Earthquake Liquefaction Triggering -- CPT Profile",
                semantic_warning=(
                    "Model factor-of-safety screening; not a risk class or probability."
                ),
                **common,
            )
        )
        manifests.append(
            AnalysisProductManifest(
                product_id=f"{evidence_id}__{scenario.scenario_id}__triggering_points",
                scientific_role=liq_contract.POINT_ANALYSIS,
                evidence_role="DERIVED",
                support_type=liq_contract.POINT_ANALYSIS,
                units={"minimum_model_fs_liq": "dimensionless"},
                primary_value_fields=("minimum_model_fs_liq",),
                geometry_or_raster_path=str(point_gpkg_path)
                if point_gpkg_path
                else str(point_summary_path),
                display_name="Earthquake Liquefaction Triggering -- CPT Points",
                semantic_warning=products.SPATIAL_SUPPORT_STATEMENT,
                **common,
            )
        )
        for manifest in manifests:
            write_product_manifest(
                manifest, out_dir / f"{manifest.product_id}.product_manifest.json"
            )

    return ExecutionResult(
        plan=plan,
        profile_df=profile_df,
        point_summary_df=point_summary_df,
        point_analysis_gdf=point_analysis_gdf,
        point_analysis_findings=point_analysis_findings,
        manifests=tuple(manifests),
    )

"""MAR-033 Section 43: adversarial tests for the generic intake/orchestration architecture
(fingerprint -> recognition -> planning -> execution -> product manifest).

Synthetic canonical CPT parquet products below are built through the REAL MAR-032 production
writer (`cpt_profile.build_canonical_measurements` + `write_canonical_cpt_measurements`), exactly
as `tests/test_cpt_evidence.py` does -- never hand-faked bytes for the positive case.
"""

from __future__ import annotations

import inspect
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from marine_engine.geotechnical import cpt_contract as contract
from marine_engine.geotechnical import cpt_profile
from marine_engine.intake import recognition, registry
from marine_engine.liquefaction import contract as liq_contract
from marine_engine.liquefaction.earthquake_triggering import (
    FinesDeclaration,
    SoilApplicabilityDeclaration,
    TipResistanceDeclaration,
)
from marine_engine.liquefaction.scenario import EarthquakeScenario
from marine_engine.liquefaction.stress import LayeredStressModel, SoilLayer
from marine_engine.orchestration import execution as orch_execution
from marine_engine.orchestration import planner as orch_planner
from marine_engine.orchestration.capability import EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING


def _write_canonical_cpt(path, *, evidence_id: str = "synthetic_evidence") -> None:
    observations = pd.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "depth": [0.0, 1.0, 2.0, 3.0],
            "qc": [1.0, 2.0, 3.0, 4.0],
            "fs": [10.0, 20.0, 30.0, 40.0],
            "u": [5.0, 10.0, 15.0, 20.0],
        }
    )
    build = cpt_profile.build_canonical_measurements(
        observations,
        source_id=evidence_id,
        test_id="CPT-1",
        depth=cpt_profile.DepthDeclaration("depth", "m", contract.DEPTH_BELOW_SEABED, "synthetic"),
        channels=[
            cpt_profile.ChannelDeclaration("qc", "MPa", contract.QC_MPA, "synthetic qc"),
            cpt_profile.ChannelDeclaration("fs", "kPa", contract.FS_KPA, "synthetic fs"),
            cpt_profile.ChannelDeclaration("u", "kPa", contract.U2_KPA, "synthetic u2"),
        ],
        observation_index_column="idx",
    )
    cpt_profile.write_canonical_cpt_measurements(build.measurements, path, evidence_id=evidence_id)


# --- "canonical CPT recognized without filename reliance" -----------------------------------------


def test_canonical_cpt_recognized_regardless_of_misleading_filename(tmp_path):
    path = tmp_path / "definitely_not_a_cpt_file.parquet"
    _write_canonical_cpt(path)
    fingerprint, decision = registry.inspect_and_recognize(path)
    assert fingerprint.container_type == "PARQUET"
    assert decision.state == recognition.RECOGNIZED
    assert decision.recognized_role == contract.MEASURED_CPT_CPTU_PROFILE


# --- "structural lookalike without canonical identity does not become measured CPT" ---------------


def test_structural_lookalike_without_marker_never_recognized(tmp_path):
    good_path = tmp_path / "good.parquet"
    _write_canonical_cpt(good_path)
    lookalike_df = pd.read_parquet(good_path)
    lookalike_path = tmp_path / "cpt_measurements.parquet"  # a CPT-suggestive filename on purpose
    lookalike_df.to_parquet(lookalike_path, index=False)  # no canonical marker stamped

    fingerprint, decision = registry.inspect_and_recognize(lookalike_path)
    assert fingerprint.container_type == "PARQUET"
    assert decision.state != recognition.RECOGNIZED
    assert decision.recognized_role is None
    assert len(decision.candidates) == 1
    candidate = decision.candidates[0]
    assert candidate.required_confirmation is True
    assert candidate.confidence == recognition.CONFIDENCE_INSUFFICIENT


def test_non_cpt_parquet_produces_no_candidate(tmp_path):
    path = tmp_path / "unrelated.parquet"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_parquet(path, index=False)
    fingerprint, decision = registry.inspect_and_recognize(path)
    assert decision.state == recognition.UNCLASSIFIED
    assert decision.candidates == ()


# --- "unknown raster stays unclassified" ----------------------------------------------------------


def test_unknown_geotiff_stays_unclassified(tmp_path):
    path = tmp_path / "elevation.tif"
    data = np.random.default_rng(0).normal(loc=-30.0, scale=5.0, size=(10, 10)).astype("float32")
    transform = from_origin(500000, 5900000, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    fingerprint, decision = registry.inspect_and_recognize(path)
    assert fingerprint.container_type == "GEOTIFF"
    assert fingerprint.crs is not None
    assert decision.state == recognition.UNCLASSIFIED
    assert decision.recognized_role is None
    plan = orch_planner.plan_earthquake_cpt_liquefaction_triggering(
        recognition=decision, readiness_facts=None
    )
    assert plan.status == orch_planner.NOT_APPLICABLE


# --- "a random line does not auto-become a pipeline" ----------------------------------------------


def test_random_line_does_not_become_pipeline(tmp_path):
    path = tmp_path / "random_line.gpkg"
    gdf = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[LineString([(0, 0), (100, 100), (200, 50)])], crs="EPSG:32631"
    )
    gdf.to_file(path, driver="GPKG", layer="random_line")

    fingerprint, decision = registry.inspect_and_recognize(path)
    assert fingerprint.container_type == "GEOPACKAGE"
    assert fingerprint.geometry_type == "LineString"
    assert decision.state == recognition.UNCLASSIFIED
    assert decision.recognized_role is None
    assert decision.recognized_role != "PIPELINE_ROUTE"


# --- "recognized inputs produce a deterministic capability plan" ----------------------------------


def test_recognized_input_produces_deterministic_plan(tmp_path):
    path = tmp_path / "cpt.parquet"
    _write_canonical_cpt(path)
    fingerprint, decision = registry.inspect_and_recognize(path)
    plan_a = orch_planner.plan_earthquake_cpt_liquefaction_triggering(
        recognition=decision, readiness_facts=None
    )
    plan_b = orch_planner.plan_earthquake_cpt_liquefaction_triggering(
        recognition=decision, readiness_facts=None
    )
    assert plan_a == plan_b
    assert plan_a.status == orch_planner.BLOCKED_MISSING_INPUT


# --- "ambiguous semantic role blocks auto-execution" ----------------------------------------------


def test_ambiguous_candidates_block_the_plan():
    candidate_a = recognition.SemanticCandidate(
        role="ROLE_A",
        confidence=recognition.CONFIDENCE_INSUFFICIENT,
        evidence_level=recognition.STRUCTURAL_CANDIDATE,
        evidence=(),
        contradictions=(),
        required_confirmation=True,
        recognizer_id="fake_a",
    )
    candidate_b = recognition.SemanticCandidate(
        role="ROLE_B",
        confidence=recognition.CONFIDENCE_INSUFFICIENT,
        evidence_level=recognition.STRUCTURAL_CANDIDATE,
        evidence=(),
        contradictions=(),
        required_confirmation=True,
        recognizer_id="fake_b",
    )
    from marine_engine.intake.fingerprint import DataFingerprint

    fp = DataFingerprint(
        path="x", byte_size=1, sha256="x" * 64, file_extension=".x", container_type="UNKNOWN"
    )
    decision = recognition.decide_recognition(fp, [candidate_a, candidate_b])
    assert decision.state == recognition.AMBIGUOUS
    plan = orch_planner.plan_earthquake_cpt_liquefaction_triggering(
        recognition=decision, readiness_facts=None
    )
    assert plan.status == orch_planner.BLOCKED_AMBIGUOUS_SEMANTICS


def test_contradicted_sufficient_candidates_block_the_plan():
    candidate_a = recognition.SemanticCandidate(
        role="ROLE_A",
        confidence=recognition.CONFIDENCE_SUFFICIENT,
        evidence_level=recognition.CANONICAL_VERIFIED,
        evidence=("x",),
        contradictions=(),
        required_confirmation=False,
        recognizer_id="fake_a",
    )
    candidate_b = recognition.SemanticCandidate(
        role="ROLE_B",
        confidence=recognition.CONFIDENCE_SUFFICIENT,
        evidence_level=recognition.CANONICAL_VERIFIED,
        evidence=("y",),
        contradictions=(),
        required_confirmation=False,
        recognizer_id="fake_b",
    )
    from marine_engine.intake.fingerprint import DataFingerprint

    fp = DataFingerprint(
        path="x", byte_size=1, sha256="x" * 64, file_extension=".x", container_type="UNKNOWN"
    )
    decision = recognition.decide_recognition(fp, [candidate_a, candidate_b])
    assert decision.state == recognition.CONTRADICTED
    plan = orch_planner.plan_earthquake_cpt_liquefaction_triggering(
        recognition=decision, readiness_facts=None
    )
    assert plan.status == orch_planner.BLOCKED_AMBIGUOUS_SEMANTICS


# --- "missing dependency produces exact blocker" --------------------------------------------------


def test_missing_dependency_produces_exact_named_blocker(tmp_path):
    path = tmp_path / "cpt.parquet"
    _write_canonical_cpt(path)
    inspection = orch_execution.inspect_cpt_evidence(path)
    facts = orch_execution.build_readiness_facts(
        inspection,
        tip_resistance=None,
        stress_model=None,
        stress_model_problem=None,
        fines=None,
        soil_applicability=None,
        scenario=None,
    )
    plan = orch_planner.plan_earthquake_cpt_liquefaction_triggering(
        recognition=inspection.recognition, readiness_facts=facts
    )
    assert plan.status == orch_planner.BLOCKED_MISSING_INPUT
    reasons_joined = " ".join(plan.reasons)
    assert "CORRECTED_TIP_RESISTANCE_NOT_ESTABLISHED" in reasons_joined
    assert "VERTICAL_STRESS_PROFILE_NOT_AVAILABLE" in reasons_joined
    assert "MISSING_EARTHQUAKE_SCENARIO" in reasons_joined


# --- "dependency DAG is acyclic" ------------------------------------------------------------------


def test_topological_order_sorts_an_acyclic_graph():
    order = orch_planner.topological_order(["c", "b", "a"], {"b": ["a"], "c": ["b"]})
    assert order.index("a") < order.index("b") < order.index("c")


def test_topological_order_rejects_a_cycle():
    with pytest.raises(orch_planner.CycleError):
        orch_planner.topological_order(["x", "y"], {"x": ["y"], "y": ["x"]})


def test_capability_registry_dependencies_are_acyclic():
    from marine_engine.orchestration.capability import CAPABILITY_REGISTRY

    nodes = list(CAPABILITY_REGISTRY)
    edges = {cap_id: list(defn.dependencies) for cap_id, defn in CAPABILITY_REGISTRY.items()}
    order = orch_planner.topological_order(nodes, edges)  # must not raise
    assert set(order) == set(nodes)


# --- "plan-only performs no scientific write" / "auto-process cannot execute arbitrary shell" -----


def test_blocked_plan_writes_nothing(tmp_path):
    path = tmp_path / "cpt.parquet"
    _write_canonical_cpt(path)
    inspection = orch_execution.inspect_cpt_evidence(path)
    out_dir = tmp_path / "out"
    result = orch_execution.execute_earthquake_cpt_liquefaction_triggering(
        inspection,
        evidence_id="synthetic_evidence",
        scenario=None,
        tip_resistance=None,
        stress_model=None,
        stress_model_problem=None,
        fines=None,
        soil_applicability=None,
        out_dir=out_dir,
    )
    assert result.plan.status != orch_planner.AVAILABLE
    assert result.manifests == ()
    assert not out_dir.exists()


def test_orchestration_modules_never_shell_out_or_eval():
    import marine_engine.intake.fingerprint as fp_mod
    import marine_engine.intake.recognition as rec_mod
    import marine_engine.intake.registry as reg_mod
    import marine_engine.orchestration.execution as exec_mod
    import marine_engine.orchestration.planner as plan_mod

    forbidden = ("subprocess", "os.system", "os.popen", "eval(", "exec(", "shell=True")
    for module in (fp_mod, rec_mod, reg_mod, exec_mod, plan_mod):
        source = inspect.getsource(module)
        for token in forbidden:
            assert token not in source, f"{module.__name__} unexpectedly references {token!r}"


# --- "new analysis product writes product manifest" / "manifest lineage points to source assets" --


def test_available_execution_writes_product_manifests_with_source_lineage(tmp_path):
    path = tmp_path / "cpt.parquet"
    _write_canonical_cpt(path)
    inspection = orch_execution.inspect_cpt_evidence(path)
    scenario = EarthquakeScenario(
        scenario_id="demo",
        moment_magnitude_mw=7.5,
        pga_g=0.2,
        pga_reference=liq_contract.FREE_FIELD_SEABED_SURFACE_PGA,
    )
    tip = TipResistanceDeclaration(
        mode="QC_PLUS_U2_AND_DECLARED_AREA_RATIO", basis="x", area_ratio=0.75
    )
    stress = LayeredStressModel(
        layers=(SoilLayer(0.0, 50.0, 18.0, "x"),), water_unit_weight_kn_m3=10.0
    )
    fines = FinesDeclaration(source="USER_DECLARED_FC_SCENARIO", basis="x", fc_percent=10.0)
    applicability = SoilApplicabilityDeclaration(
        established=True, basis_kind="SOURCE_ESTABLISHED", basis="x"
    )

    out_dir = tmp_path / "out"
    result = orch_execution.execute_earthquake_cpt_liquefaction_triggering(
        inspection,
        evidence_id="synthetic_evidence",
        scenario=scenario,
        tip_resistance=tip,
        stress_model=stress,
        stress_model_problem=None,
        fines=fines,
        soil_applicability=applicability,
        out_dir=out_dir,
    )
    assert result.plan.status == orch_planner.AVAILABLE
    assert len(result.manifests) == 2
    for manifest in result.manifests:
        assert manifest.capability_id == EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING
        assert "synthetic_evidence" in manifest.source_asset_ids
        manifest_path = out_dir / f"{manifest.product_id}.product_manifest.json"
        assert manifest_path.is_file()
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert on_disk["source_asset_ids"] == ["synthetic_evidence"]
    assert (out_dir / "earthquake_liquefaction_triggering_profile.parquet").is_file()


# --- MAR-033A: GENERAL_CORRELATION_SENSITIVITY executes exactly three traceable variants --------


def _sensitivity_execution_kwargs(inspection, out_dir):
    scenario = EarthquakeScenario(
        scenario_id="sensitivity_demo",
        moment_magnitude_mw=7.5,
        pga_g=0.2,
        pga_reference=liq_contract.FREE_FIELD_SEABED_SURFACE_PGA,
    )
    tip = TipResistanceDeclaration(
        mode="QC_PLUS_U2_AND_DECLARED_AREA_RATIO", basis="x", area_ratio=0.75
    )
    stress = LayeredStressModel(
        layers=(SoilLayer(0.0, 50.0, 18.0, "x"),), water_unit_weight_kn_m3=10.0
    )
    sensitivity_fines = FinesDeclaration(
        source=liq_contract.GENERAL_CORRELATION_SENSITIVITY, basis="automatic sensitivity"
    )
    applicability = SoilApplicabilityDeclaration(
        established=True, basis_kind="SOURCE_ESTABLISHED", basis="x"
    )
    return {
        "inspection": inspection,
        "evidence_id": "synthetic_evidence",
        "scenario": scenario,
        "tip_resistance": tip,
        "stress_model": stress,
        "stress_model_problem": None,
        "fines": sensitivity_fines,
        "soil_applicability": applicability,
        "out_dir": out_dir,
    }


def test_general_correlation_sensitivity_executes_exactly_three_traceable_variants(tmp_path):
    path = tmp_path / "cpt.parquet"
    _write_canonical_cpt(path)
    inspection = orch_execution.inspect_cpt_evidence(path)
    out_dir = tmp_path / "out"

    result = orch_execution.execute_earthquake_cpt_liquefaction_triggering(
        **_sensitivity_execution_kwargs(inspection, out_dir)
    )

    assert result.plan.status == orch_planner.AVAILABLE
    # Three variants x (profile manifest + points manifest) = six, never merged into one, never
    # averaged, never a "best" pick.
    assert len(result.manifests) == 6
    variant_scenario_ids = {manifest.scenario_id for manifest in result.manifests}
    assert len(variant_scenario_ids) == 3
    for scenario_id in variant_scenario_ids:
        assert scenario_id.startswith("sensitivity_demo__C_FC_")

    product_ids = [manifest.product_id for manifest in result.manifests]
    assert len(set(product_ids)) == len(product_ids)  # every manifest is uniquely identified

    for manifest in result.manifests:
        assert manifest.capability_id == EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING
        assert "synthetic_evidence" in manifest.source_asset_ids
        manifest_path = out_dir / f"{manifest.product_id}.product_manifest.json"
        assert manifest_path.is_file()
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert on_disk["scenario_id"] == manifest.scenario_id

    # Each variant's profile parquet is its own distinct, traceable file -- never a shared file two
    # variants would both need to be filtered out of.
    profile_files = sorted(
        out_dir.glob("earthquake_liquefaction_triggering__C_FC_*_profile.parquet")
    )
    assert len(profile_files) == 3

    # The combined in-memory profile carries all three variants, distinguishable by scenario_id --
    # an "explicitly indexed combined result", never a blended/averaged one.
    assert set(result.profile_df["scenario_id"]) == variant_scenario_ids


def test_general_correlation_sensitivity_blocked_plan_writes_nothing(tmp_path):
    path = tmp_path / "cpt.parquet"
    _write_canonical_cpt(path)
    inspection = orch_execution.inspect_cpt_evidence(path)
    out_dir = tmp_path / "out"
    kwargs = _sensitivity_execution_kwargs(inspection, out_dir)
    kwargs["scenario"] = None  # withhold the scenario -> plan cannot be AVAILABLE

    result = orch_execution.execute_earthquake_cpt_liquefaction_triggering(**kwargs)

    assert result.plan.status != orch_planner.AVAILABLE
    assert result.manifests == ()
    assert not out_dir.exists()

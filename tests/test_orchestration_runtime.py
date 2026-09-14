"""MAR-033A Part B: adversarial tests for the generic capability-runtime architecture.

These tests prove the architecture itself -- registration, dependency-ordered planning/execution,
role-based unlocking, closed-failure on a cycle or an unknown dependency -- using SYNTHETIC
capabilities (`A -> produces ROLE_A`, `B depends on A -> produces ROLE_B`, `C depends on B ->
produces ROLE_C`) registered only for the duration of each test. This is the acceptance proof that
`orchestration.runtime`/`orchestration.planner` are genuinely generic: the same `plan_all` /
`execute_plan` mechanics that run the one real registered capability
(`EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING`, exercised end-to-end in `test_intake_orchestration.py`)
also run these synthetic ones, unmodified.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from marine_engine.orchestration import bootstrap as orch_bootstrap
from marine_engine.orchestration import capability as orch_capability
from marine_engine.orchestration import context as orch_context
from marine_engine.orchestration import planner as orch_planner
from marine_engine.orchestration import runtime as orch_runtime
from marine_engine.orchestration.capability import EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING
from marine_engine.orchestration.product_manifest import AnalysisProductManifest

# MAR-034 Section 10: registration is now an explicit, idempotent bootstrap call -- never an
# import-time side effect. This module-level call (not incidental) is what populates the REAL
# registries before `test_new_capability_registers_alongside_the_real_one_unmodified` below runs;
# without it, this test module run in isolation would see an empty real registry regardless of
# the fix it tests for. Calling it again inside any individual test is always safe (idempotent).
orch_bootstrap.register_builtin_runtimes()


def _synthetic_definition(
    capability_id: str, *, dependencies: tuple[str, ...], produced_role: str
) -> orch_capability.CapabilityDefinition:
    return orch_capability.CapabilityDefinition(
        capability_id=capability_id,
        required_roles=(),
        optional_roles=(),
        scenario_requirements=(),
        dependencies=dependencies,
        produced_roles=(produced_role,),
        execution_adapter=capability_id,
    )


def _synthetic_runtime(
    definition: orch_capability.CapabilityDefinition,
    *,
    required_role: str | None,
    calls: list[str],
) -> orch_runtime.CapabilityRuntime:
    """A minimal synthetic runtime: AVAILABLE once `required_role` (if any) is present in
    `context.available_roles`, else BLOCKED_MISSING_INPUT. Its executor records its own
    capability_id in `calls` (proving execution order) and returns one manifest carrying its own
    `produced_role` (proving downstream role-unlocking) -- no capability-specific facts needed."""

    def readiness_adapter(context: orch_context.PlanningContext) -> None:
        return None

    def planner_adapter(
        context: orch_context.PlanningContext, readiness_facts: None
    ) -> orch_planner.CapabilityPlan:
        if required_role is not None and required_role not in context.available_roles:
            return orch_planner.CapabilityPlan(
                definition.capability_id,
                orch_planner.BLOCKED_MISSING_INPUT,
                (f"requires role {required_role!r}",),
                (required_role,),
            )
        return orch_planner.CapabilityPlan(definition.capability_id, orch_planner.AVAILABLE, (), ())

    def executor(
        context: orch_context.PlanningContext, plan: orch_planner.CapabilityPlan
    ) -> orch_runtime.CapabilityExecutionOutcome:
        calls.append(definition.capability_id)
        produced_role = definition.produced_roles[0]
        manifest = AnalysisProductManifest(
            product_id=f"{definition.capability_id}_synthetic_product",
            capability_id=definition.capability_id,
            scientific_role=produced_role,
            evidence_role="DERIVED",
            support_type="SYNTHETIC",
            source_asset_ids=(),
            scenario_id=None,
            method_id="SYNTHETIC_TEST_METHOD",
            units={},
            primary_value_fields=(),
            geometry_or_raster_path=None,
            readiness=plan.status,
            limitations=(),
        )
        return orch_runtime.CapabilityExecutionOutcome(
            capability_id=definition.capability_id, plan=plan, manifests=(manifest,), details=None
        )

    return orch_runtime.CapabilityRuntime(
        definition=definition,
        readiness_adapter=readiness_adapter,
        planner_adapter=planner_adapter,
        executor=executor,
    )


@pytest.fixture
def synthetic_registry(monkeypatch):
    """Isolates BOTH the declarative registry (`orchestration.capability.CAPABILITY_REGISTRY`)
    and the runtime registry (`orchestration.runtime.CAPABILITY_RUNTIMES`) to fresh, empty dicts
    for the duration of one test -- proving the generic mechanics with ZERO capability-specific
    (including the real earthquake-triggering) registrations present, and restoring the real
    registries afterward via monkeypatch's own teardown."""

    fresh_registry: dict[str, orch_capability.CapabilityDefinition] = {}
    fresh_runtimes: dict[str, orch_runtime.CapabilityRuntime] = {}
    # `orchestration.runtime` imported `CAPABILITY_REGISTRY` via `from ... import`, which bound a
    # SEPARATE name in its own module namespace -- patch that name too, not just
    # `orch_capability`'s, or `register_capability_runtime`'s validation would still see the real
    # (unpatched) registry.
    monkeypatch.setattr(orch_capability, "CAPABILITY_REGISTRY", fresh_registry)
    monkeypatch.setattr(orch_runtime, "CAPABILITY_REGISTRY", fresh_registry)
    monkeypatch.setattr(orch_runtime, "CAPABILITY_RUNTIMES", fresh_runtimes)
    return fresh_registry, fresh_runtimes


def _register_chain(registry, calls, *, required_roles: dict[str, str | None]) -> None:
    definitions = {
        "A": _synthetic_definition("A", dependencies=(), produced_role="ROLE_A"),
        "B": _synthetic_definition("B", dependencies=("A",), produced_role="ROLE_B"),
        "C": _synthetic_definition("C", dependencies=("B",), produced_role="ROLE_C"),
    }
    for capability_id, definition in definitions.items():
        registry[capability_id] = definition
        orch_runtime.register_capability_runtime(
            _synthetic_runtime(definition, required_role=required_roles[capability_id], calls=calls)
        )


# --- "synthetic A->B->C capabilities plan in dependency order" -----------------------------------


def test_synthetic_chain_executes_in_dependency_order(synthetic_registry):
    registry, _runtimes = synthetic_registry
    calls: list[str] = []
    _register_chain(registry, calls, required_roles={"A": None, "B": "ROLE_A", "C": "ROLE_B"})

    context = orch_context.PlanningContext(recognized_assets=())
    final_context, outcomes = orch_runtime.execute_plan(context)

    assert calls == ["A", "B", "C"]
    assert [o.capability_id for o in outcomes] == ["A", "B", "C"]
    for outcome in outcomes:
        assert outcome.plan.status == orch_planner.AVAILABLE
        assert len(outcome.manifests) == 1
    assert {m.scientific_role for m in final_context.produced_manifests} == {
        "ROLE_A",
        "ROLE_B",
        "ROLE_C",
    }


def test_plan_all_snapshot_shows_dependents_blocked_before_any_execution(synthetic_registry):
    # "generated products unlock downstream capabilities": B/C are BLOCKED against the INITIAL
    # (pre-execution) context, proving the unlocking in the previous test is real, not incidental.
    registry, _runtimes = synthetic_registry
    calls: list[str] = []
    _register_chain(registry, calls, required_roles={"A": None, "B": "ROLE_A", "C": "ROLE_B"})

    context = orch_context.PlanningContext(recognized_assets=())
    plans = {plan.capability_id: plan for plan in orch_runtime.plan_all(context)}

    assert plans["A"].status == orch_planner.AVAILABLE
    assert plans["B"].status == orch_planner.BLOCKED_MISSING_INPUT
    assert plans["C"].status == orch_planner.BLOCKED_MISSING_INPUT
    assert calls == []  # plan_all is read-only: no executor was ever invoked


# --- "cyclic dependencies fail before execution" --------------------------------------------------


def test_cyclic_synthetic_dependency_fails_closed_before_any_execution(synthetic_registry):
    registry, _runtimes = synthetic_registry
    calls: list[str] = []
    def_a = _synthetic_definition("A", dependencies=("B",), produced_role="ROLE_A")
    def_b = _synthetic_definition("B", dependencies=("A",), produced_role="ROLE_B")
    for definition in (def_a, def_b):
        registry[definition.capability_id] = definition
        orch_runtime.register_capability_runtime(
            _synthetic_runtime(definition, required_role=None, calls=calls)
        )

    context = orch_context.PlanningContext(recognized_assets=())
    with pytest.raises(orch_planner.CycleError):
        orch_runtime.plan_all(context)
    with pytest.raises(orch_planner.CycleError):
        orch_runtime.execute_plan(context)
    assert calls == []  # no executor was ever invoked, even partially


# --- "unknown dependency fails closed" ------------------------------------------------------------


def test_unknown_dependency_fails_closed_before_any_execution(synthetic_registry):
    registry, _runtimes = synthetic_registry
    calls: list[str] = []
    definition = _synthetic_definition("A", dependencies=("GHOST",), produced_role="ROLE_A")
    registry["A"] = definition
    orch_runtime.register_capability_runtime(
        _synthetic_runtime(definition, required_role=None, calls=calls)
    )

    context = orch_context.PlanningContext(recognized_assets=())
    with pytest.raises(ValueError, match="GHOST"):
        orch_runtime.execute_plan(context)
    assert calls == []


def test_register_capability_runtime_requires_a_prior_definition(synthetic_registry):
    registry, _runtimes = synthetic_registry
    orphan = _synthetic_definition("ORPHAN", dependencies=(), produced_role="ROLE_ORPHAN")
    # Deliberately NOT added to `registry` -- the runtime registry must refuse to register a
    # runtime for a capability_id with no declared CapabilityDefinition.
    with pytest.raises(ValueError, match="ORPHAN"):
        orch_runtime.register_capability_runtime(
            _synthetic_runtime(orphan, required_role=None, calls=[])
        )


# --- "blocked capability writes no product" ---------------------------------------------------


def test_blocked_synthetic_capability_writes_no_product(synthetic_registry):
    registry, _runtimes = synthetic_registry
    calls: list[str] = []
    definition = _synthetic_definition("D", dependencies=(), produced_role="ROLE_D")
    registry["D"] = definition
    orch_runtime.register_capability_runtime(
        _synthetic_runtime(definition, required_role="ROLE_NEVER_SUPPLIED", calls=calls)
    )

    context = orch_context.PlanningContext(recognized_assets=())
    final_context, outcomes = orch_runtime.execute_plan(context)

    assert calls == []
    assert outcomes[0].plan.status == orch_planner.BLOCKED_MISSING_INPUT
    assert outcomes[0].manifests == ()
    assert final_context.produced_manifests == ()


# --- "one capability can register without changing generic planner mechanics" --------------------


def test_new_capability_registers_alongside_the_real_one_unmodified(monkeypatch):
    # Uses `setitem`/`setdefault`-style patching on the REAL registries (never wholesale-replaced)
    # so the already-registered EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING runtime stays present --
    # proving a brand new capability plugs in ALONGSIDE it with zero change to generic mechanics.
    calls: list[str] = []
    definition = _synthetic_definition("SYNTHETIC_EXTRA", dependencies=(), produced_role="ROLE_X")
    monkeypatch.setitem(orch_capability.CAPABILITY_REGISTRY, definition.capability_id, definition)
    monkeypatch.setitem(
        orch_runtime.CAPABILITY_RUNTIMES,
        definition.capability_id,
        _synthetic_runtime(definition, required_role=None, calls=calls),
    )

    context = orch_context.PlanningContext(recognized_assets=())
    plan_ids = {plan.capability_id for plan in orch_runtime.plan_all(context)}
    assert EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING in plan_ids
    assert "SYNTHETIC_EXTRA" in plan_ids

    _final_context, outcomes = orch_runtime.execute_plan(context)
    outcome_ids = {o.capability_id for o in outcomes}
    assert EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING in outcome_ids
    assert "SYNTHETIC_EXTRA" in outcome_ids
    assert calls == ["SYNTHETIC_EXTRA"]


# --- "auto-process itself contains no CPT/liquefaction-specific execution branch" -----------------


def test_auto_process_cli_command_has_no_capability_specific_execution_branch():
    from marine_engine import cli as cli_module

    source = inspect.getsource(cli_module._cmd_auto_process)
    forbidden = (
        "plan_earthquake_cpt_liquefaction_triggering(",
        "execute_earthquake_cpt_liquefaction_triggering(",
        "evaluate_triggering_profile",
        "iterate_cn_qc1ncs",
        "iterate_ic_and_n",
        "assess_earthquake_triggering_readiness",
    )
    for token in forbidden:
        assert token not in source, f"_cmd_auto_process unexpectedly references {token!r}"
    # Positive check: it DOES dispatch through the generic registry.
    assert "orch_runtime.plan_all" in source
    assert "orch_runtime.execute_plan" in source


def test_auto_process_accepts_multiple_input_paths():
    from marine_engine.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["auto-process", "one.parquet", "two.tif", "--plan-only"])
    assert args.paths == [Path("one.parquet"), Path("two.tif")]
    assert args.plan_only is True

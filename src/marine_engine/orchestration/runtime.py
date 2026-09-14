"""Capability runtime registration and generic dispatch (MAR-033A Part B, Sections 10-14, 19).

A `CapabilityRuntime` binds one already-declared `CapabilityDefinition` (`orchestration.capability`)
to the actual code that plans and executes it. `CAPABILITY_RUNTIMES` is the registry the generic
mechanics below dispatch through -- adding capability #2 means registering one more
`CapabilityRuntime`, never adding another `if capability == ...` branch to `auto-process` or to
any function in this module.

`plan_all` and `execute_plan` are the two generic operations Section 12/14 describe:

* `plan_all` -- a read-only snapshot: what is AVAILABLE / BLOCKED_* / NOT_APPLICABLE right now,
  for every registered capability, in dependency order. Performs no write.
* `execute_plan` -- walks the SAME dependency order, re-planning each capability immediately
  before its turn (so a capability that depends on another sees that capability's just-produced
  roles -- Section 19) and executing only the ones whose fresh plan says AVAILABLE. A blocked or
  not-applicable capability is skipped entirely: no executor call, no write (Section 14/18).

`build_dependency_order` is what makes the Section 32 `topological_order` sort actually load-
bearing (MAR-033A Section 13): edges are built from every registered capability's OWN declared
`CapabilityDefinition.dependencies`, an unknown dependency fails closed (`ValueError`, before any
execution), and a cycle fails closed (`CycleError`, before any execution).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from marine_engine.orchestration.capability import CAPABILITY_REGISTRY, CapabilityDefinition
from marine_engine.orchestration.context import PlanningContext
from marine_engine.orchestration.planner import AVAILABLE, CapabilityPlan, topological_order
from marine_engine.orchestration.product_manifest import AnalysisProductManifest

__all__ = [
    "CapabilityExecutionOutcome",
    "CapabilityRuntime",
    "CAPABILITY_RUNTIMES",
    "register_capability_runtime",
    "build_dependency_order",
    "plan_all",
    "execute_plan",
]


@dataclass(frozen=True)
class CapabilityExecutionOutcome:
    """The generic result every registered executor returns (Section 18-19): explicit product
    manifests only -- the generic execution layer never discovers outputs by filename, and never
    needs to understand capability-specific detail to know which roles just became available
    (that comes from `manifest.scientific_role`, read generically). `details` is an opaque,
    capability-specific payload (e.g. row-level DataFrames) for a capability-specific caller (a
    dedicated CLI command, a test) -- the generic planner/executor never read it."""

    capability_id: str
    plan: CapabilityPlan
    manifests: tuple[AnalysisProductManifest, ...]
    details: Any = None


@dataclass(frozen=True)
class CapabilityRuntime:
    """Section 10: one capability's full runtime registration.

    `readiness_adapter(context) -> Any` builds the capability's own typed readiness facts (or
    `None`) from the generic context -- capability-specific, opaque to everything else.
    `planner_adapter(context, readiness_facts) -> CapabilityPlan` turns those facts into a
    `CapabilityPlan` (typically a thin wrapper around an existing capability-specific planning
    function). `executor(context, plan) -> CapabilityExecutionOutcome` performs the capability's
    safe local computation ONLY when `plan.status == AVAILABLE` (enforced by the executor itself,
    mirroring every existing capability-specific executor's own contract).
    """

    definition: CapabilityDefinition
    readiness_adapter: Callable[[PlanningContext], Any]
    planner_adapter: Callable[[PlanningContext, Any], CapabilityPlan]
    executor: Callable[[PlanningContext, CapabilityPlan], CapabilityExecutionOutcome]


CAPABILITY_RUNTIMES: dict[str, CapabilityRuntime] = {}


def register_capability_runtime(runtime: CapabilityRuntime) -> None:
    """Section 10-11: the registry owns capability dispatch. Registration requires the
    capability_id to already be declared in `orchestration.capability.CAPABILITY_REGISTRY` --
    the declarative "what" (`CapabilityDefinition`) and the executable "how" (`CapabilityRuntime`)
    stay two separate registries that must agree on identity, never silently diverge."""

    capability_id = runtime.definition.capability_id
    if capability_id not in CAPABILITY_REGISTRY:
        raise ValueError(
            f"cannot register a runtime for unregistered capability_id {capability_id!r}; "
            "declare a CapabilityDefinition in orchestration.capability.CAPABILITY_REGISTRY first"
        )
    CAPABILITY_RUNTIMES[capability_id] = runtime


def build_dependency_order() -> list[str]:
    """MAR-033A Section 13: build dependency edges from every REGISTERED runtime's own declared
    `CapabilityDefinition.dependencies` and topologically sort them. A dependency naming a
    capability_id with no registered runtime fails closed (`ValueError`) rather than being
    silently ignored; a cyclic graph fails closed (`CycleError`, from `topological_order`) --
    both BEFORE any capability is planned or executed in dependency order."""

    nodes = list(CAPABILITY_RUNTIMES)
    edges: dict[str, list[str]] = {}
    for capability_id, runtime in CAPABILITY_RUNTIMES.items():
        deps = list(runtime.definition.dependencies)
        unknown = [dep for dep in deps if dep not in CAPABILITY_RUNTIMES]
        if unknown:
            raise ValueError(
                f"capability {capability_id!r} declares dependencies with no registered runtime: "
                f"{unknown!r} -- fails closed, never silently ignored"
            )
        edges[capability_id] = deps
    return topological_order(nodes, edges)


def plan_all(context: PlanningContext) -> list[CapabilityPlan]:
    """Section 12: generic capability planning across every registered `CapabilityRuntime`, in
    dependency order. Performs no write. Dispatch is entirely through the registry -- this
    function has no knowledge of any specific capability's declaration/readiness-fact type."""

    order = build_dependency_order()
    plans: list[CapabilityPlan] = []
    for capability_id in order:
        runtime = CAPABILITY_RUNTIMES[capability_id]
        readiness_facts = runtime.readiness_adapter(context)
        plans.append(runtime.planner_adapter(context, readiness_facts))
    return plans


def execute_plan(
    context: PlanningContext,
) -> tuple[PlanningContext, list[CapabilityExecutionOutcome]]:
    """Sections 14, 19: execute every registered capability in dependency order. Each capability
    is RE-PLANNED immediately before its own turn (never reusing a stale up-front snapshot), so a
    capability that depends on another sees the dependency's just-produced roles in the SAME run.
    A capability whose fresh plan is not AVAILABLE is skipped: no executor call, no product, no
    write. Returns the (possibly updated) context alongside one outcome per registered capability,
    in the same dependency order.
    """

    order = build_dependency_order()
    outcomes: list[CapabilityExecutionOutcome] = []
    for capability_id in order:
        runtime = CAPABILITY_RUNTIMES[capability_id]
        readiness_facts = runtime.readiness_adapter(context)
        plan = runtime.planner_adapter(context, readiness_facts)
        if plan.status != AVAILABLE:
            outcomes.append(
                CapabilityExecutionOutcome(
                    capability_id=capability_id, plan=plan, manifests=(), details=None
                )
            )
            continue
        outcome = runtime.executor(context, plan)
        outcomes.append(outcome)
        if outcome.manifests:
            context = context.with_produced_manifests(outcome.manifests)
    return context, outcomes

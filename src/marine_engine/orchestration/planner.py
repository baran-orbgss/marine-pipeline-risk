"""Capability planner (MAR-033 Sections 31-32).

Given what has been recognized (Section 29) and what has been explicitly declared (a scenario
manifest, if any), decides for each registered capability:

    AVAILABLE | BLOCKED_MISSING_INPUT | BLOCKED_AMBIGUOUS_SEMANTICS | BLOCKED_INVALID_INPUT |
    NOT_APPLICABLE

with exact, named reasons -- what can run now, what cannot, why, and what input would unlock it
(Section 31). `topological_order` is a fully generic dependency-DAG sort (Section 32): it makes
no assumption about any particular capability graph shape, so a future capability's own
`dependencies` list plugs in without this function changing.

Dispatch to each capability's own planning logic goes through `_CAPABILITY_PLANNERS`, a registry
keyed by `capability_id` -- adding capability #2 means registering one more entry, never editing
this module's shared mechanics (`CapabilityPlan`, the five plan states, `topological_order`) or
any other capability's planning function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from marine_engine.geotechnical import cpt_contract
from marine_engine.intake.recognition import (
    AMBIGUOUS,
    CONTRADICTED,
    INVALID,
    RECOGNIZED,
    UNCLASSIFIED,
    RecognitionDecision,
)
from marine_engine.liquefaction import contract as liq_contract
from marine_engine.liquefaction.readiness import (
    EarthquakeTriggeringReadinessFacts,
    assess_earthquake_triggering_readiness,
)
from marine_engine.orchestration.capability import (
    CAPABILITY_REGISTRY,
    EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING,
)

__all__ = [
    "AVAILABLE",
    "BLOCKED_MISSING_INPUT",
    "BLOCKED_AMBIGUOUS_SEMANTICS",
    "BLOCKED_INVALID_INPUT",
    "NOT_APPLICABLE",
    "PLAN_STATES",
    "CapabilityPlan",
    "CycleError",
    "topological_order",
    "plan_earthquake_cpt_liquefaction_triggering",
    "plan_capabilities",
]

AVAILABLE = "AVAILABLE"
BLOCKED_MISSING_INPUT = "BLOCKED_MISSING_INPUT"
BLOCKED_AMBIGUOUS_SEMANTICS = "BLOCKED_AMBIGUOUS_SEMANTICS"
BLOCKED_INVALID_INPUT = "BLOCKED_INVALID_INPUT"
NOT_APPLICABLE = "NOT_APPLICABLE"

PLAN_STATES = frozenset(
    {
        AVAILABLE,
        BLOCKED_MISSING_INPUT,
        BLOCKED_AMBIGUOUS_SEMANTICS,
        BLOCKED_INVALID_INPUT,
        NOT_APPLICABLE,
    }
)


class CycleError(ValueError):
    """The declared capability dependency graph contains a cycle -- refused, never executed."""


def topological_order(nodes: list[str], edges: dict[str, list[str]]) -> list[str]:
    """Generic Kahn's-algorithm topological sort. `edges[node]` lists the nodes `node` DEPENDS ON
    (must run before it). Raises `CycleError` rather than silently truncating, guessing, or
    executing a partial order (Section 32: no cyclic execution, no hidden dependency)."""

    remaining = {n: list(edges.get(n, [])) for n in nodes}
    ordered: list[str] = []
    frontier = [n for n, deps in remaining.items() if not deps]
    while frontier:
        node = frontier.pop()
        ordered.append(node)
        for other, deps in remaining.items():
            if node in deps:
                deps.remove(node)
                if not deps and other not in ordered and other not in frontier:
                    frontier.append(other)
    if len(ordered) != len(nodes):
        unresolved = [n for n in nodes if n not in ordered]
        raise CycleError(f"capability dependency graph is cyclic; could not order: {unresolved}")
    return ordered


@dataclass(frozen=True)
class CapabilityPlan:
    capability_id: str
    status: str
    reasons: tuple[str, ...]
    unlocking_inputs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "unlocking_inputs": list(self.unlocking_inputs),
        }


def plan_earthquake_cpt_liquefaction_triggering(
    *,
    recognition: RecognitionDecision,
    readiness_facts: EarthquakeTriggeringReadinessFacts | None,
) -> CapabilityPlan:
    """Section 31, applied to the one MAR-033 capability. `readiness_facts` is `None` when no
    scenario/declaration has been supplied at all (e.g. an `auto-process --plan-only` call with
    no `--scenario-manifest`) -- that is reported as a specific, named missing-input blocker,
    listing exactly what a scenario manifest would need to declare to unlock the capability."""

    capability_id = EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING
    if recognition.state == INVALID:
        return CapabilityPlan(capability_id, BLOCKED_INVALID_INPUT, tuple(recognition.reasons), ())
    if recognition.state == CONTRADICTED:
        return CapabilityPlan(
            capability_id, BLOCKED_AMBIGUOUS_SEMANTICS, tuple(recognition.reasons), ()
        )
    if recognition.state == AMBIGUOUS:
        return CapabilityPlan(
            capability_id,
            BLOCKED_AMBIGUOUS_SEMANTICS,
            tuple(recognition.reasons),
            ("explicit semantic confirmation of the CPT role",),
        )
    if recognition.state == UNCLASSIFIED:
        return CapabilityPlan(
            capability_id,
            NOT_APPLICABLE,
            ("input was not recognized as a canonical MEASURED_CPT_CPTU_PROFILE",),
            ("a canonical MAR-032B CPT parquet product",),
        )
    assert recognition.state == RECOGNIZED
    if recognition.recognized_role != cpt_contract.MEASURED_CPT_CPTU_PROFILE:
        return CapabilityPlan(
            capability_id,
            NOT_APPLICABLE,
            (f"recognized role {recognition.recognized_role!r} is not MEASURED_CPT_CPTU_PROFILE",),
            (),
        )

    unlocking = CAPABILITY_REGISTRY[capability_id].scenario_requirements
    if readiness_facts is None:
        return CapabilityPlan(
            capability_id,
            BLOCKED_MISSING_INPUT,
            ("no scenario/declaration facts were supplied to the planner",),
            unlocking,
        )
    result = assess_earthquake_triggering_readiness(readiness_facts)
    if result.status == liq_contract.NOT_READY:
        blocking = tuple(r for a in result.axes for r in a.blocking_reasons)
        return CapabilityPlan(capability_id, BLOCKED_MISSING_INPUT, blocking, blocking)
    return CapabilityPlan(capability_id, AVAILABLE, (), ())


_CAPABILITY_PLANNERS: dict[str, Callable[..., CapabilityPlan]] = {
    EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING: plan_earthquake_cpt_liquefaction_triggering,
}


def plan_capabilities(
    *,
    recognition: RecognitionDecision,
    readiness_facts: EarthquakeTriggeringReadinessFacts | None,
) -> list[CapabilityPlan]:
    """Dispatches to each registered capability's own planning function via
    `_CAPABILITY_PLANNERS`. Adding capability #2 means registering one more entry there, never
    editing this function's body or any other capability's planning logic."""

    return [
        planner(recognition=recognition, readiness_facts=readiness_facts)
        for planner in _CAPABILITY_PLANNERS.values()
    ]

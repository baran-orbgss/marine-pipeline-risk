"""Generic capability declaration model (MAR-033 Sections 30, 47).

A `CapabilityDefinition` names what a capability requires, produces and depends on. The planner
(`planner.py`) and the dependency DAG (`topological_order`) consume this declaration generically
-- neither hard-codes a capability-specific dependency decision (Section 30). A future capability
registers one more `CapabilityDefinition` (and its own recognizer/planning function); nothing
here changes shape to accommodate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marine_engine.geotechnical import cpt_contract
from marine_engine.liquefaction import contract as liq_contract

__all__ = [
    "EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING",
    "CapabilityDefinition",
    "CAPABILITY_REGISTRY",
]

EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING = "earthquake_cpt_liquefaction_triggering"


@dataclass(frozen=True)
class CapabilityDefinition:
    capability_id: str
    required_roles: tuple[str, ...]
    optional_roles: tuple[str, ...]
    scenario_requirements: tuple[str, ...]
    dependencies: tuple[str, ...]
    produced_roles: tuple[str, ...]
    execution_adapter: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "required_roles": list(self.required_roles),
            "optional_roles": list(self.optional_roles),
            "scenario_requirements": list(self.scenario_requirements),
            "dependencies": list(self.dependencies),
            "produced_roles": list(self.produced_roles),
            "execution_adapter": self.execution_adapter,
        }


CAPABILITY_REGISTRY: dict[str, CapabilityDefinition] = {
    EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING: CapabilityDefinition(
        capability_id=EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING,
        required_roles=(cpt_contract.MEASURED_CPT_CPTU_PROFILE,),
        optional_roles=(),
        scenario_requirements=(
            "moment_magnitude_mw",
            "pga_g",
            "corrected_tip_resistance_declaration",
            "vertical_stress_model",
            "fines_content_source",
            "cohesionless_soil_applicability",
        ),
        dependencies=(),
        produced_roles=(
            liq_contract.EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING_SCREENING_PROFILE,
            liq_contract.POINT_ANALYSIS,
        ),
        execution_adapter=EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING,
    ),
}

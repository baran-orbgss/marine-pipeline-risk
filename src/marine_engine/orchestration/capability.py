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

from marine_engine.bedforms import contract as bedforms_contract
from marine_engine.change import dod as change_dod
from marine_engine.geotechnical import cpt_contract
from marine_engine.liquefaction import contract as liq_contract
from marine_engine.project import categories as project_categories
from marine_engine.slope_stability import contract as slope_stability_contract
from marine_engine.terrain import product_roles as terrain_product_roles

__all__ = [
    "EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING",
    "CANONICALIZE_BATHYMETRY",
    "TERRAIN_DERIVATIVES",
    "BEDFORM_MORPHODYNAMICS",
    "OBSERVED_MULTI_EPOCH_SEABED_CHANGE",
    "CapabilityDefinition",
    "CAPABILITY_REGISTRY",
]

EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING = "earthquake_cpt_liquefaction_triggering"

# MAR-034 Section 2: the generic terrain/bedform/change auto-processing capability family. Each
# registers existing accepted `terrain`/`bedforms`/`change` science behind this same generic
# runtime -- no new equation, threshold, or hazard interpretation is introduced here.
CANONICALIZE_BATHYMETRY = "canonicalize_bathymetry"
TERRAIN_DERIVATIVES = "terrain_derivatives"
BEDFORM_MORPHODYNAMICS = "bedform_morphodynamics"
OBSERVED_MULTI_EPOCH_SEABED_CHANGE = "observed_multi_epoch_seabed_change"


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
    CANONICALIZE_BATHYMETRY: CapabilityDefinition(
        capability_id=CANONICALIZE_BATHYMETRY,
        required_roles=(project_categories.BATHYMETRY_RASTER,),
        optional_roles=(),
        scenario_requirements=(),
        dependencies=(),
        produced_roles=(slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED,),
        execution_adapter=CANONICALIZE_BATHYMETRY,
    ),
    TERRAIN_DERIVATIVES: CapabilityDefinition(
        capability_id=TERRAIN_DERIVATIVES,
        required_roles=(slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED,),
        optional_roles=(),
        scenario_requirements=(),
        dependencies=(CANONICALIZE_BATHYMETRY,),
        produced_roles=(terrain_product_roles.TERRAIN_DERIVATIVE_PRODUCT,),
        execution_adapter=TERRAIN_DERIVATIVES,
    ),
    BEDFORM_MORPHODYNAMICS: CapabilityDefinition(
        capability_id=BEDFORM_MORPHODYNAMICS,
        required_roles=(slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED,),
        optional_roles=(),
        scenario_requirements=(),
        dependencies=(CANONICALIZE_BATHYMETRY,),
        produced_roles=(bedforms_contract.SANDBED_BEDFORM_MORPHOLOGY_AND_OBSERVED_CHANGE,),
        execution_adapter=BEDFORM_MORPHODYNAMICS,
    ),
    OBSERVED_MULTI_EPOCH_SEABED_CHANGE: CapabilityDefinition(
        capability_id=OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
        required_roles=(slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED,),
        optional_roles=(),
        scenario_requirements=(),
        dependencies=(CANONICALIZE_BATHYMETRY,),
        produced_roles=(change_dod.MULTI_EPOCH_SEABED_CHANGE_POC,),
        execution_adapter=OBSERVED_MULTI_EPOCH_SEABED_CHANGE,
    ),
}

"""Typed, hazard-specific slope-stability scenario manifest (MAR-031 Sections 11-13).

Kept deliberately OUTSIDE the generic `project.manifest.ProjectManifest` so the generic project
layer never accumulates hazard-specific parameters. Mirrors the repository's Pydantic v2
conventions (`ConfigDict(extra="forbid")` on every model).

There is NO default scenario: a manifest exists only when a user explicitly writes one, and every
scenario must declare `material_model` and `parameter_basis` explicitly (no defaults for those
fields either). MAR-031 accepts only `USER_DECLARED_HYPOTHETICAL_SCENARIO`; a scenario is never
treated as measured site geotechnical truth.

The manifest carries no terrain block: the terrain is the accepted canonical product of the study
config the CLI is invoked with, so there is exactly one source of terrain truth.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from marine_engine.slope_stability import core


class GeotechnicalScenarioDeclaration(BaseModel):
    """One explicit hypothetical scenario as written by the user (Section 12). Validation (Section
    13) is enforced twice on purpose: here at parse time (clear YAML error) and again in
    `core.UndrainedInfiniteSlopeScenario` (so programmatic callers get the same guarantees)."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    material_model: str
    parameter_basis: str
    undrained_shear_strength_kpa: float
    submerged_unit_weight_kn_m3: float
    slip_surface_depth_m: float

    @field_validator(
        "undrained_shear_strength_kpa", "submerged_unit_weight_kn_m3", "slip_surface_depth_m"
    )
    @classmethod
    def _finite_positive(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "must be a finite number > 0 (zero, negative, NaN and infinity are rejected; "
                "nothing is clipped, defaulted or inferred)"
            )
        return value

    @field_validator("material_model")
    @classmethod
    def _material_model_accepted(cls, value: str) -> str:
        if value != core.MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE:
            raise ValueError(
                f"must be {core.MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE!r}"
                " (the only material model implemented by MAR-031)"
            )
        return value

    @field_validator("parameter_basis")
    @classmethod
    def _parameter_basis_accepted(cls, value: str) -> str:
        if value != core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO:
            raise ValueError(
                f"must be {core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO!r} (MAR-031 "
                "does not accept a scenario as measured site geotechnical truth)"
            )
        return value

    @field_validator("scenario_id")
    @classmethod
    def _scenario_id_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scenario_id must be a non-empty string")
        return value

    def to_core(self) -> core.UndrainedInfiniteSlopeScenario:
        return core.UndrainedInfiniteSlopeScenario(
            scenario_id=self.scenario_id,
            material_model=self.material_model,
            parameter_basis=self.parameter_basis,
            undrained_shear_strength_kpa=self.undrained_shear_strength_kpa,
            submerged_unit_weight_kn_m3=self.submerged_unit_weight_kn_m3,
            slip_surface_depth_m=self.slip_surface_depth_m,
        )


class SlopeInstabilityScenarioManifest(BaseModel):
    """Root schema for an explicit slope-stability scenario manifest YAML file."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    geotechnical_scenarios: list[GeotechnicalScenarioDeclaration]
    slope_scales_m: list[float] | None = None

    @field_validator("analysis_id")
    @classmethod
    def _analysis_id_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("analysis_id must be a non-empty string")
        return value

    @field_validator("slope_scales_m")
    @classmethod
    def _scales_finite_positive_unique(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("slope_scales_m must list at least one scale when given")
        for scale in value:
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"slope scale {scale!r} must be a finite number > 0")
        if len(set(value)) != len(value):
            raise ValueError("slope_scales_m must not repeat a scale")
        return value

    @model_validator(mode="after")
    def _scenarios_present_and_unique(self) -> SlopeInstabilityScenarioManifest:
        if not self.geotechnical_scenarios:
            raise ValueError(
                "geotechnical_scenarios must contain at least one explicit scenario -- a manifest "
                "with no scenario is meaningless because no default scenario exists"
            )
        ids = [s.scenario_id for s in self.geotechnical_scenarios]
        if len(set(ids)) != len(ids):
            raise ValueError(f"scenario_id values must be unique, got {ids}")
        return self

    def core_scenarios(self) -> list[core.UndrainedInfiniteSlopeScenario]:
        return [s.to_core() for s in self.geotechnical_scenarios]


def load_slope_instability_scenario_manifest(path: str | Path) -> SlopeInstabilityScenarioManifest:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh) or {}
    return SlopeInstabilityScenarioManifest.model_validate(raw)

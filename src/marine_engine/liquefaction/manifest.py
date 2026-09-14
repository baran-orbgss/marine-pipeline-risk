"""Typed, hazard-specific earthquake-liquefaction scenario manifest (MAR-033 Sections 6, 9, 12,
15, 33).

Kept deliberately OUTSIDE the generic `project.manifest.ProjectManifest`, mirroring
`slope_stability.manifest`'s established precedent, so the generic project layer never
accumulates hazard-specific parameters. Also kept separate from WHICH CPT evidence file to run
against -- that stays a CLI argument (mirroring how `build-slope-instability-screening` keeps its
terrain `config` positional argument separate from its `--scenario-manifest`), so one scenario
manifest is reusable across different CPT datasets.

There is no default scenario: a manifest exists only when a user explicitly writes one, and every
field a chosen mode requires must be declared explicitly (no default Mw, PGA, tip-resistance
mode/area ratio, stress model, fines source/value, or soil applicability).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from marine_engine.liquefaction import contract
from marine_engine.liquefaction.earthquake_triggering import (
    FinesDeclaration,
    SoilApplicabilityDeclaration,
    TipResistanceDeclaration,
)
from marine_engine.liquefaction.scenario import EarthquakeScenario, LiquefactionInputError
from marine_engine.liquefaction.stress import (
    ExplicitStressProfile,
    LayeredStressModel,
    SoilLayer,
    StressModel,
)

__all__ = [
    "LiquefactionTriggeringScenarioManifest",
    "load_liquefaction_scenario_manifest",
]


class EarthquakeDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    moment_magnitude_mw: float
    pga_g: float
    pga_reference: str = contract.FREE_FIELD_SEABED_SURFACE_PGA

    def to_core(self, scenario_id: str) -> EarthquakeScenario:
        return EarthquakeScenario(
            scenario_id=scenario_id,
            moment_magnitude_mw=self.moment_magnitude_mw,
            pga_g=self.pga_g,
            pga_reference=self.pga_reference,
        )


class TipResistanceDeclarationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    basis: str
    area_ratio: float | None = None

    def to_core(self) -> TipResistanceDeclaration:
        return TipResistanceDeclaration(
            mode=self.mode, basis=self.basis, area_ratio=self.area_ratio
        )


class SoilLayerDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_depth_m: float
    bottom_depth_m: float
    total_unit_weight_kn_m3: float
    basis: str

    def to_core(self) -> SoilLayer:
        return SoilLayer(
            top_depth_m=self.top_depth_m,
            bottom_depth_m=self.bottom_depth_m,
            total_unit_weight_kn_m3=self.total_unit_weight_kn_m3,
            basis=self.basis,
        )


class ExplicitStressPointDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    depth_m: float
    sigma_v0_kpa: float
    sigma_v0_effective_kpa: float


class StressDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    basis: str = ""
    water_unit_weight_kn_m3: float | None = None
    layers: list[SoilLayerDeclaration] | None = None
    points: list[ExplicitStressPointDeclaration] | None = None

    @model_validator(mode="after")
    def _mode_known(self) -> StressDeclaration:
        if self.mode not in contract.STRESS_MODES:
            raise ValueError(
                f"stress mode must be one of {sorted(contract.STRESS_MODES)}, got {self.mode!r}"
            )
        return self

    def to_core(self) -> StressModel:
        if self.mode == contract.SEABED_RELATIVE_LAYERED_STRESS:
            if not self.layers or self.water_unit_weight_kn_m3 is None:
                raise LiquefactionInputError(
                    "SEABED_RELATIVE_LAYERED_STRESS requires at least one declared layer and an "
                    "explicit water_unit_weight_kn_m3 -- no default soil or seawater unit weight "
                    "is ever assumed"
                )
            return LayeredStressModel(
                layers=tuple(layer.to_core() for layer in self.layers),
                water_unit_weight_kn_m3=self.water_unit_weight_kn_m3,
            )
        if not self.points:
            raise LiquefactionInputError(
                "EXPLICIT_STRESS_PROFILE requires at least one declared depth/stress point"
            )
        return ExplicitStressProfile(
            depth_stress_kpa=tuple(
                (p.depth_m, p.sigma_v0_kpa, p.sigma_v0_effective_kpa) for p in self.points
            ),
            basis=self.basis,
        )


class FinesDeclarationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    basis: str
    fc_percent: float | None = None
    c_fc: float | None = None

    def to_core(self) -> FinesDeclaration:
        return FinesDeclaration(
            source=self.source, basis=self.basis, fc_percent=self.fc_percent, c_fc=self.c_fc
        )


class SoilApplicabilityDeclarationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    established: bool
    basis_kind: str | None = None
    basis: str = ""
    ic_cutoff: float | None = None

    def to_core(self) -> SoilApplicabilityDeclaration:
        return SoilApplicabilityDeclaration(
            established=self.established,
            basis_kind=self.basis_kind,
            basis=self.basis,
            ic_cutoff=self.ic_cutoff,
        )


class LiquefactionTriggeringScenarioManifest(BaseModel):
    """Root schema for an explicit MAR-033 earthquake-liquefaction scenario manifest YAML file."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    earthquake: EarthquakeDeclaration
    tip_resistance: TipResistanceDeclarationModel
    stress: StressDeclaration
    fines: FinesDeclarationModel
    soil_applicability: SoilApplicabilityDeclarationModel
    static_shear_material: bool = False

    @model_validator(mode="after")
    def _scenario_id_non_empty(self) -> LiquefactionTriggeringScenarioManifest:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string")
        return self

    def to_core(
        self,
    ) -> tuple[
        EarthquakeScenario,
        TipResistanceDeclaration,
        StressModel,
        FinesDeclaration,
        SoilApplicabilityDeclaration,
    ]:
        """Section 13's dual-validation pattern (also used by `slope_stability.manifest`): clear
        pydantic errors at parse time, then the same domain dataclasses' own guarantees again
        here, so a programmatic caller building these objects directly gets identical checks."""

        try:
            return (
                self.earthquake.to_core(self.scenario_id),
                self.tip_resistance.to_core(),
                self.stress.to_core(),
                self.fines.to_core(),
                self.soil_applicability.to_core(),
            )
        except LiquefactionInputError as exc:
            raise ValueError(str(exc)) from exc


def load_liquefaction_scenario_manifest(path: str | Path) -> LiquefactionTriggeringScenarioManifest:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh) or {}
    return LiquefactionTriggeringScenarioManifest.model_validate(raw)

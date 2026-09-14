"""Vertical total/effective stress models (MAR-033 Section 9).

Two explicit modes. Neither ever defaults a soil or seawater unit weight, and neither adds the
offshore water column above the seabed into the soil-overburden calculation:

* `ExplicitStressProfile` -- EXPLICIT_STRESS_PROFILE (Mode A). The caller declares
  sigma_v0_kpa/sigma_v0_effective_kpa at specific depths. Matching to a CPT row's own depth is
  EXACT (rounded to the millimetre) only -- a row whose depth has no declared match is reported
  missing, never interpolated or approximated from neighbouring declared depths.
* `LayeredStressModel` -- SEABED_RELATIVE_LAYERED_STRESS (Mode B). Explicit layer boundaries plus
  a declared total/saturated unit weight per layer and one declared water unit weight. Stress is
  integrated analytically downward from the seabed (z = 0); this is a closed-form evaluation of a
  declared step-wise-linear profile, never an interpolation between sparse samples. Pore pressure
  is the hydrostatic column from the seabed to depth z (no excess/pre-earthquake pore pressure is
  assumed away from hydrostatic without a source stating otherwise).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from marine_engine.liquefaction.scenario import (
    LiquefactionInputError,
    require_finite,
    require_finite_positive,
)

__all__ = [
    "LiquefactionInputError",
    "SoilLayer",
    "LayeredStressModel",
    "ExplicitStressProfile",
    "StressModel",
    "stress_at_depths",
]


@dataclass(frozen=True)
class SoilLayer:
    """One explicitly declared layer of the seabed-relative stress model (Section 9)."""

    top_depth_m: float
    bottom_depth_m: float
    total_unit_weight_kn_m3: float
    basis: str

    def __post_init__(self) -> None:
        top = require_finite("top_depth_m", self.top_depth_m)
        bottom = require_finite("bottom_depth_m", self.bottom_depth_m)
        if top < 0.0:
            raise LiquefactionInputError("top_depth_m must be >= 0 (depth below seabed)")
        if bottom <= top:
            raise LiquefactionInputError(
                f"bottom_depth_m ({bottom}) must be greater than top_depth_m ({top})"
            )
        object.__setattr__(self, "top_depth_m", top)
        object.__setattr__(self, "bottom_depth_m", bottom)
        object.__setattr__(
            self,
            "total_unit_weight_kn_m3",
            require_finite_positive("total_unit_weight_kn_m3", self.total_unit_weight_kn_m3),
        )
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise LiquefactionInputError(
                "SoilLayer.basis must state the declared source of the unit weight -- no default "
                "soil unit weight is ever assumed"
            )


@dataclass(frozen=True)
class LayeredStressModel:
    """Section 9 Mode B: SEABED_RELATIVE_LAYERED_STRESS. Layers must be contiguous from the
    seabed (z = 0) with no gap or overlap -- checked once at construction, never silently
    tolerated."""

    layers: tuple[SoilLayer, ...]
    water_unit_weight_kn_m3: float

    def __post_init__(self) -> None:
        if not self.layers:
            raise LiquefactionInputError(
                "SEABED_RELATIVE_LAYERED_STRESS requires at least one declared layer"
            )
        ordered = tuple(sorted(self.layers, key=lambda layer: layer.top_depth_m))
        object.__setattr__(self, "layers", ordered)
        if not math.isclose(ordered[0].top_depth_m, 0.0, rel_tol=0.0, abs_tol=1e-9):
            raise LiquefactionInputError(
                "declared layers must start at the seabed (top_depth_m = 0), got "
                f"{ordered[0].top_depth_m}"
            )
        for upper, lower in zip(ordered, ordered[1:], strict=False):
            if not math.isclose(upper.bottom_depth_m, lower.top_depth_m, rel_tol=0.0, abs_tol=1e-9):
                raise LiquefactionInputError(
                    f"declared layers are not contiguous: a layer ending at {upper.bottom_depth_m}"
                    f" m is followed by one starting at {lower.top_depth_m} m -- no gap or overlap "
                    "is permitted"
                )
        object.__setattr__(
            self,
            "water_unit_weight_kn_m3",
            require_finite_positive("water_unit_weight_kn_m3", self.water_unit_weight_kn_m3),
        )

    @property
    def max_depth_m(self) -> float:
        return self.layers[-1].bottom_depth_m

    def stress_at_depths(self, depths_m: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Analytical (not interpolated) total/effective vertical stress at each depth. NaN
        wherever the depth falls outside the declared layer coverage."""

        depths = np.asarray(depths_m, dtype=np.float64)
        covered = (depths >= 0.0) & (depths <= self.max_depth_m)
        total = np.zeros_like(depths)
        for layer in self.layers:
            full = depths >= layer.bottom_depth_m
            partial = (depths > layer.top_depth_m) & ~full
            total = np.where(
                full,
                total + layer.total_unit_weight_kn_m3 * (layer.bottom_depth_m - layer.top_depth_m),
                total,
            )
            total = np.where(
                partial, total + layer.total_unit_weight_kn_m3 * (depths - layer.top_depth_m), total
            )
        total = np.where(covered, total, np.nan)
        pore_pressure = self.water_unit_weight_kn_m3 * depths
        effective = np.where(covered, total - pore_pressure, np.nan)
        return total, effective


@dataclass(frozen=True)
class ExplicitStressProfile:
    """Section 9 Mode A: EXPLICIT_STRESS_PROFILE. `depth_stress_kpa` is a tuple of
    `(depth_m, sigma_v0_kpa, sigma_v0_effective_kpa)` triples declared by the source/caller."""

    depth_stress_kpa: tuple[tuple[float, float, float], ...]
    basis: str

    def __post_init__(self) -> None:
        if not self.depth_stress_kpa:
            raise LiquefactionInputError(
                "EXPLICIT_STRESS_PROFILE requires at least one declared depth/stress point"
            )
        seen: set[float] = set()
        for depth, sigma_v0, sigma_eff in self.depth_stress_kpa:
            require_finite("depth_m", depth)
            require_finite("sigma_v0_kpa", sigma_v0)
            require_finite("sigma_v0_effective_kpa", sigma_eff)
            key = round(float(depth), 3)
            if key in seen:
                raise LiquefactionInputError(
                    f"duplicate declared stress depth {depth} m (rounded to {key} m)"
                )
            seen.add(key)
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise LiquefactionInputError(
                "ExplicitStressProfile.basis must state the declared source of the stress values"
            )

    def _lookup(self) -> dict[float, tuple[float, float]]:
        return {
            round(float(depth), 3): (float(sigma_v0), float(sigma_eff))
            for depth, sigma_v0, sigma_eff in self.depth_stress_kpa
        }

    def stress_at_depths(self, depths_m: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Exact-depth (never interpolated) lookup of declared stress values."""

        table = self._lookup()
        depths = np.asarray(depths_m, dtype=np.float64)
        total = np.full(depths.shape, np.nan, dtype=np.float64)
        effective = np.full(depths.shape, np.nan, dtype=np.float64)
        flat_total = total.reshape(-1)
        flat_effective = effective.reshape(-1)
        for i, depth in enumerate(np.round(depths, 3).reshape(-1)):
            hit = table.get(float(depth))
            if hit is not None:
                flat_total[i], flat_effective[i] = hit
        return total, effective


StressModel = LayeredStressModel | ExplicitStressProfile


def stress_at_depths(
    model: StressModel, depths_m: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to whichever explicit stress mode `model` is. Returns
    `(sigma_v0_kpa, sigma_v0_effective_kpa)` arrays, NaN wherever unavailable."""

    return model.stress_at_depths(depths_m)

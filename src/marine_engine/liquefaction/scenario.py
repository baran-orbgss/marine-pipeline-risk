"""Explicit earthquake scenario declaration (MAR-033 Section 6).

No earthquake scenario defaults exist anywhere in this package: `moment_magnitude_mw`, `pga_g`
and `pga_reference` are all required, explicit, user/source-declared fields. Nothing here queries
a hazard map, infers PGA from location, or invents a magnitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from marine_engine.liquefaction import contract


class LiquefactionInputError(ValueError):
    """A scenario/stress/fines declaration is outside the supported domain or internally
    inconsistent. Never clipped, defaulted or inferred -- the caller must supply a valid,
    explicit value."""


def require_finite(name: str, value: object) -> float:
    """Shared validation helper (also used by `stress.py`): reject non-numeric, NaN and
    infinite values rather than silently coercing or defaulting them."""

    if isinstance(value, bool):
        raise LiquefactionInputError(f"{name} must be a finite number, got {value!r}")
    try:
        as_float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise LiquefactionInputError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(as_float):
        raise LiquefactionInputError(f"{name} must be a finite number, got {value!r}")
    return as_float


def require_finite_positive(name: str, value: object) -> float:
    as_float = require_finite(name, value)
    if as_float <= 0.0:
        raise LiquefactionInputError(
            f"{name} must be a finite number > 0 (zero, negative, NaN and infinity are rejected; "
            f"nothing is clipped or defaulted), got {value!r}"
        )
    return as_float


@dataclass(frozen=True)
class EarthquakeScenario:
    """One explicit, source/user-declared earthquake scenario (Section 6). Every field is
    REQUIRED -- there is no default scenario anywhere in MAR-033."""

    scenario_id: str
    moment_magnitude_mw: float
    pga_g: float
    pga_reference: str = contract.FREE_FIELD_SEABED_SURFACE_PGA

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise LiquefactionInputError("scenario_id must be a non-empty string")
        object.__setattr__(
            self,
            "moment_magnitude_mw",
            require_finite_positive("moment_magnitude_mw", self.moment_magnitude_mw),
        )
        object.__setattr__(self, "pga_g", require_finite_positive("pga_g", self.pga_g))
        if self.pga_reference not in contract.PGA_REFERENCES:
            raise LiquefactionInputError(
                f"pga_reference must be one of {sorted(contract.PGA_REFERENCES)}, got "
                f"{self.pga_reference!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "moment_magnitude_mw": self.moment_magnitude_mw,
            "pga_g": self.pga_g,
            "pga_reference": self.pga_reference,
        }

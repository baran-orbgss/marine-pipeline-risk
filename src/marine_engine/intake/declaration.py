"""Generic, engine-owned asset declaration (MAR-034 Section 6).

Spatial files often do not carry every scientific fact the engine needs (sign convention,
vertical datum, survey epoch, ...) in a form that can be observed structurally, and structural
compatibility alone must never be treated as sufficient evidence for one (Section 4). An
`AssetDeclaration` is the ONE generic, capability-agnostic place an operator/user states such
facts for a single asset -- loaded from the `assets:` section of a run manifest
(`intake.manifest.load_run_manifest`), never guessed.

Every field here is DECLARED (Section 7): kept structurally separate from OBSERVED facts
(`intake.fingerprint.DataFingerprint`) and from DERIVED products (e.g.
`terrain.canonical.CanonicalTerrainRaster`). None of these three is ever merged into another.

Only genuinely generic metadata belongs here. Hazard-specific scenario parameters (e.g. an
earthquake scenario's magnitude/PGA) remain in their own capability-specific declaration (e.g.
`orchestration.execution.EarthquakeTriggeringDeclaration`), never here (Section 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["AssetDeclaration"]


@dataclass(frozen=True)
class AssetDeclaration:
    """One asset's declared facts. Every field defaults to the honest "nothing declared" state --
    an `AssetDeclaration` existing at all never implies every field on it is populated."""

    asset_id: str
    semantic_role: str | None = None
    source_sign_convention: str | None = None
    vertical_datum: str | None = None
    survey_epoch: str | None = None
    evidence_role: str | None = None
    source_description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "semantic_role": self.semantic_role,
            "source_sign_convention": self.source_sign_convention,
            "vertical_datum": self.vertical_datum,
            "survey_epoch": self.survey_epoch,
            "evidence_role": self.evidence_role,
            "source_description": self.source_description,
        }

"""Generic planning/execution context (MAR-033A Part B, Section 9).

`PlanningContext` is the ONE input the generic planner and executor consume. It carries what has
been recognized (Section 27-29), which scientific roles are currently available (from recognition
AND from products already produced earlier in the same `auto-process` run -- Section 19), and an
opaque per-capability declaration bag. It never contains a capability-specific dataclass in its
own shape (e.g. `EarthquakeTriggeringReadinessFacts`): a capability's own readiness/planner
adapter is responsible for pulling its own typed declaration out of `capability_declarations` and
interpreting it -- the generic context, planner and executor never do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marine_engine.intake.fingerprint import DataFingerprint
from marine_engine.intake.recognition import RECOGNIZED, RecognitionDecision
from marine_engine.orchestration.product_manifest import AnalysisProductManifest

__all__ = ["RecognizedAsset", "PlanningContext"]


@dataclass(frozen=True)
class RecognizedAsset:
    """One inspected input file: its observed structure plus its semantic recognition decision
    (Sections 26-29), unmodified."""

    path: Path
    fingerprint: DataFingerprint
    recognition: RecognitionDecision


@dataclass(frozen=True)
class PlanningContext:
    """Section 9's generic planning context.

    `capability_declarations` is an opaque mapping keyed by `capability_id`: each capability's own
    adapter interprets the value it finds there (or its absence, meaning "nothing declared"). The
    generic planner/executor never inspect its contents.
    """

    recognized_assets: tuple[RecognizedAsset, ...]
    capability_declarations: dict[str, Any] = field(default_factory=dict)
    produced_manifests: tuple[AnalysisProductManifest, ...] = ()
    project_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available_roles(self) -> frozenset[str]:
        """Section 19: roles recognized among the supplied inputs UNION roles already produced by
        capabilities executed earlier in this same run -- the mechanism that lets a generated
        product unlock a downstream capability."""

        from_assets = {
            asset.recognition.recognized_role
            for asset in self.recognized_assets
            if asset.recognition.state == RECOGNIZED and asset.recognition.recognized_role
        }
        from_products = {manifest.scientific_role for manifest in self.produced_manifests}
        return frozenset(from_assets | from_products)

    def assets_with_role(self, role: str) -> tuple[RecognizedAsset, ...]:
        """Every supplied input whose recognition decision settled on exactly `role`."""

        return tuple(
            asset
            for asset in self.recognized_assets
            if asset.recognition.state == RECOGNIZED and asset.recognition.recognized_role == role
        )

    def manifests_with_role(self, role: str) -> tuple[AnalysisProductManifest, ...]:
        """Every already-produced manifest (this run) carrying scientific_role `role`."""

        return tuple(m for m in self.produced_manifests if m.scientific_role == role)

    def with_produced_manifests(
        self, manifests: tuple[AnalysisProductManifest, ...]
    ) -> PlanningContext:
        """Section 19: returns a NEW context (frozen dataclasses stay immutable) with `manifests`
        appended -- generated products becoming available to subsequent planning/execution."""

        if not manifests:
            return self
        return PlanningContext(
            recognized_assets=self.recognized_assets,
            capability_declarations=self.capability_declarations,
            produced_manifests=self.produced_manifests + tuple(manifests),
            project_metadata=self.project_metadata,
        )

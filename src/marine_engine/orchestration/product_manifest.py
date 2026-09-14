"""Generic analysis product manifest (MAR-033 Section 36).

Every successfully completed analysis writes one of these -- the canonical handoff to the GIS
layer system (Section 37), so the API/UI does not need to rediscover scientific semantics by
globbing output filenames. Cartographic styling (colour, risk-class breakpoints) is deliberately
absent: that remains presentation authority, never science (Section 37).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["AnalysisProductManifest", "write_product_manifest"]


@dataclass(frozen=True)
class AnalysisProductManifest:
    product_id: str
    capability_id: str
    scientific_role: str
    evidence_role: str
    support_type: str
    source_asset_ids: tuple[str, ...]
    scenario_id: str | None
    method_id: str
    units: dict[str, str]
    primary_value_fields: tuple[str, ...]
    geometry_or_raster_path: str | None
    readiness: str
    limitations: tuple[str, ...]
    display_name: str = ""
    semantic_warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "capability_id": self.capability_id,
            "scientific_role": self.scientific_role,
            "evidence_role": self.evidence_role,
            "support_type": self.support_type,
            "source_asset_ids": list(self.source_asset_ids),
            "scenario_id": self.scenario_id,
            "method_id": self.method_id,
            "units": dict(self.units),
            "primary_value_fields": list(self.primary_value_fields),
            "geometry_or_raster_path": self.geometry_or_raster_path,
            "readiness": self.readiness,
            "limitations": list(self.limitations),
            "display_name": self.display_name,
            "semantic_warning": self.semantic_warning,
        }


def write_product_manifest(manifest: AnalysisProductManifest, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")
    return path

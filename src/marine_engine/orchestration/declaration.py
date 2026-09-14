"""Generic capability-declaration loading request (MAR-034 Section 8).

`DeclarationRequest` is the ONE generic bag of raw, uninterpreted material the CLI hands every
registered capability's own `CapabilityRuntime.declaration_adapter`, in place of the CLI itself
constructing a capability-specific declaration (Section 8's architectural debt: `_cmd_auto_process`
must not know `EarthquakeTriggeringDeclaration`, `BedformDeclaration`, `ChangeDeclaration`, or any
other capability's declaration type -- Section 48).

The generic CLI does not interpret ANY field below for scientific meaning; it only loads whatever
raw material was supplied (a legacy singular `--scenario-manifest` path, an optional generic run
manifest's per-capability YAML section, ...) and lets each capability's own adapter decide what,
if anything, it needs from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DeclarationRequest"]


@dataclass(frozen=True)
class DeclarationRequest:
    capability_id: str
    raw_capability_section: dict[str, Any] | None = None
    scenario_manifest_path: Path | None = None
    locations_path: Path | None = None
    evidence_id: str | None = None
    out_dir: Path | None = None
    manifest_dir: Path | None = None
    asset_declarations: dict[str, Any] = field(default_factory=dict)

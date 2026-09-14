"""Generic optional run/intake manifest (MAR-034 Section 9).

A YAML file with two top-level sections:

    assets:
      <asset_id>:
        path: ...                  # resolved relative to the manifest's OWN directory
        semantic_role: ...
        source_sign_convention: ...
        vertical_datum: ...
        survey_epoch: ...
        evidence_role: ...
        source_description: ...
    capabilities:
      <capability_id>:
        ...                        # entirely capability-specific; opaque to this loader

`assets:` is parsed into generic, engine-owned `AssetDeclaration`s (Section 6/7) -- the ONLY
thing this module interprets. `capabilities:` is deliberately kept as RAW, unvalidated mappings
(Section 48): the generic loader must never understand any capability's own scientific fields;
only that capability's own `orchestration.runtime.CapabilityRuntime.declaration_adapter` may
interpret its section, given the raw dict verbatim.

Users are never required to author this manifest merely to unlock something a canonical marker
or embedded source metadata already establishes (Section 9) -- it exists only to supply facts
genuinely missing from the data itself. The manifest itself is therefore entirely OPTIONAL:
`auto-process`/`plan-processing` run with no declared facts at all, just with more blockers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from marine_engine.intake.declaration import AssetDeclaration

__all__ = ["RunManifest", "load_run_manifest"]


@dataclass(frozen=True)
class RunManifest:
    manifest_path: Path
    manifest_dir: Path
    asset_declarations: dict[str, AssetDeclaration] = field(default_factory=dict)
    asset_paths: dict[str, Path] = field(default_factory=dict)
    raw_capability_sections: dict[str, dict[str, Any]] = field(default_factory=dict)

    def asset_declaration_for_path(self, path: str | Path) -> AssetDeclaration | None:
        """The declared asset (if any) whose manifest `path:` resolves to the SAME real file as
        `path` -- matched by resolved absolute path, never by filename text alone."""

        resolved = Path(path).resolve()
        for asset_id, declared_path in self.asset_paths.items():
            if declared_path == resolved:
                return self.asset_declarations[asset_id]
        return None


def _require_mapping(value: Any, *, what: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"run manifest {what} must be a mapping, got {type(value).__name__}")
    return value


def load_run_manifest(path: str | Path) -> RunManifest:
    """Loads a generic run manifest. Raises `ValueError`/`OSError` on a structurally invalid or
    unreadable file -- never silently drops a malformed `assets`/`capabilities` section."""

    resolved_manifest_path = Path(path).resolve()
    manifest_dir = resolved_manifest_path.parent
    with resolved_manifest_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw = _require_mapping(raw, what="root")

    raw_assets = _require_mapping(raw.get("assets"), what="'assets' section")
    asset_declarations: dict[str, AssetDeclaration] = {}
    asset_paths: dict[str, Path] = {}
    for asset_id, entry in raw_assets.items():
        entry = _require_mapping(entry, what=f"asset {asset_id!r}")
        raw_path = entry.get("path")
        if not raw_path:
            raise ValueError(f"run manifest asset {asset_id!r} is missing a required 'path'")
        asset_paths[asset_id] = (manifest_dir / raw_path).resolve()
        asset_declarations[asset_id] = AssetDeclaration(
            asset_id=asset_id,
            semantic_role=entry.get("semantic_role"),
            source_sign_convention=entry.get("source_sign_convention"),
            vertical_datum=entry.get("vertical_datum"),
            survey_epoch=entry.get("survey_epoch"),
            evidence_role=entry.get("evidence_role"),
            source_description=entry.get("source_description"),
        )

    raw_capabilities = _require_mapping(raw.get("capabilities"), what="'capabilities' section")
    for capability_id, section in raw_capabilities.items():
        raw_capabilities[capability_id] = _require_mapping(
            section, what=f"capability section {capability_id!r}"
        )

    return RunManifest(
        manifest_path=resolved_manifest_path,
        manifest_dir=manifest_dir,
        asset_declarations=asset_declarations,
        asset_paths=asset_paths,
        raw_capability_sections=raw_capabilities,
    )

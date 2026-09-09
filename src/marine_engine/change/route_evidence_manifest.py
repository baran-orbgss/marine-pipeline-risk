"""Explicit route <-> DoD linkage manifest for route-referenced observed seabed change (MAR-029
Section 8).

A DoD raster never becomes associated with a project route merely because it overlaps
spatially, shares a CRS, sits in the same directory, carries the project name in its filename,
or is the only raster around. The association is declared HERE, explicitly, in a small typed
analysis manifest that is deliberately kept OUTSIDE the generic `ProjectManifest` schema, so the
generic project layer never accumulates hazard-specific analysis configuration.

Mirrors `project.manifest`'s Pydantic v2 conventions exactly (`ConfigDict(extra="forbid")` on
every model). Every path resolves relative to THIS manifest file's own directory, never the
process's current working directory -- `load_route_change_evidence_manifest` returns that
directory alongside the parsed manifest so callers cannot resolve paths any other way.

Declared facts recorded here (route asset ID, source change-study identity, declared DoD
definition, declared source scientific role, declared CRS, declared epochs, provenance
artefact) are kept structurally
separate from what is later OBSERVED from the DoD raster itself
(`route_evidence.inspect_dod_source`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class DoDSourceDeclaration(BaseModel):
    """Section 9 (declared/linked side): a DoD source raster to consume READ-ONLY, plus the
    operator/source-DECLARED facts about it. `path` is the DoD GeoTIFF; `provenance_path`, when
    given, is a machine-readable provenance/validation artefact of the change study that produced
    the DoD (e.g. MAR-021's `seabed_change_poc_validation.json`) -- it is identified by content
    hash only and copied nowhere. Nothing here is ever inferred from the raster.

    MAR-029A: `dod_definition_declared` and `source_scientific_role_declared` are REQUIRED and
    non-empty. Raster bytes and structure prove nothing about what the values MEAN; the operator
    must state the mathematical definition explicitly, and only a declaration that represents the
    accepted canonical definition (`change.dod.DoDResult.definition`) can be interpreted. The
    software never invents, flips, or guesses a definition from the values."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    source_change_study_id: str
    dod_definition_declared: str
    source_scientific_role_declared: str
    provenance_path: Path | None = None
    horizontal_crs_declared: str | None = None
    epoch1_survey_epoch_declared: str | None = None
    epoch2_survey_epoch_declared: str | None = None
    source_name: str | None = None
    licence_note: str | None = None

    @model_validator(mode="after")
    def _require_non_empty_semantics(self) -> DoDSourceDeclaration:
        for name in (
            "dod_definition_declared",
            "source_scientific_role_declared",
            "source_change_study_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"dod.{name} must be a non-empty declaration")
        return self


class RouteChangeEvidenceManifest(BaseModel):
    """Root schema for a route-change evidence manifest YAML file (Section 8). Identifies
    explicitly: the project manifest (MAR-026/027), the route asset the change evidence is to be
    referenced to, and the DoD raster + its provenance artefact."""

    model_config = ConfigDict(extra="forbid")

    project_manifest: Path
    route_asset_id: str
    dod: DoDSourceDeclaration


def load_route_change_evidence_manifest(
    path: str | Path,
) -> tuple[RouteChangeEvidenceManifest, Path]:
    """Parse and validate a route-change evidence manifest. Returns `(manifest, manifest_dir)`;
    `manifest_dir` is the ONLY base any path in the manifest is ever resolved against."""

    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh) or {}
    manifest = RouteChangeEvidenceManifest.model_validate(raw)
    return manifest, manifest_path.parent


def resolve_manifest_path(declared: Path, manifest_dir: Path) -> Path:
    """A declared path resolves relative to the manifest's own directory; an already-absolute
    path is returned unchanged."""

    if declared.is_absolute():
        return declared
    return (manifest_dir / declared).resolve()

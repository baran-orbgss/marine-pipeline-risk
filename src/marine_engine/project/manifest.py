"""Canonical operator project manifest schema and loader (MAR-026 Section 6).

Mirrors `marine_engine.config`'s established Pydantic v2 pattern exactly (`ConfigDict(extra=
"forbid")` on every model, so unknown free-form keys never silently pass a schema that is
supposed to be canonical). Asset paths are always resolved relative to the MANIFEST file's own
directory, never the process's current working directory (Section 6) -- `load_project_manifest`
returns that directory alongside the parsed manifest so callers cannot resolve paths any other
way by accident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from marine_engine.project.categories import ASSET_CATEGORIES, EVIDENCE_ROLES


class ProjectIdentity(BaseModel):
    """Identity and working CRS of an operator project (Section 6)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str = ""
    working_crs: str


class DeclaredProvenance(BaseModel):
    """Section 5.A: operator/source-DECLARED metadata. Kept structurally separate from the
    observed facts extracted from the actual file (Section 5.B, see `project.registry`) --
    never merged into one ambiguous field."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    source_uri: str | None = None
    supplier: str | None = None
    survey_epoch: str | None = None
    horizontal_crs_declared: str | None = None
    vertical_datum_declared: str | None = None
    units_declared: str | None = None
    measurement_reference_declared: str | None = None
    sign_convention_declared: str | None = None
    licence_note: str | None = None


class BurialColumnMapping(BaseModel):
    """Section 11: explicit column/semantic mapping for a generic tabular burial input --
    critical engineering semantics (which column is chainage, which is the measured value) are
    NEVER inferred from column names alone."""

    model_config = ConfigDict(extra="forbid")

    chainage_or_kp_column: str
    measured_value_column: str
    x_column: str | None = None
    y_column: str | None = None
    record_id_column: str | None = None
    uncertainty_column: str | None = None


class AssetEntry(BaseModel):
    """One registered operator asset (Section 6)."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    category: str
    evidence_role: str
    path: Path
    layer: str | None = None
    burial_columns: BurialColumnMapping | None = None
    provenance: DeclaredProvenance

    @model_validator(mode="after")
    def _validate_vocabulary(self) -> AssetEntry:
        if self.category not in ASSET_CATEGORIES:
            raise ValueError(
                f"asset {self.asset_id!r}: unknown category {self.category!r} -- must be one "
                f"of {sorted(ASSET_CATEGORIES)}"
            )
        if self.evidence_role not in EVIDENCE_ROLES:
            raise ValueError(
                f"asset {self.asset_id!r}: unknown evidence_role {self.evidence_role!r} -- must "
                f"be one of {sorted(EVIDENCE_ROLES)}"
            )
        return self


class ProjectManifest(BaseModel):
    """Root schema for an operator project manifest YAML file (Section 6)."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectIdentity
    primary_route_asset_id: str | None = None
    assets: list[AssetEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_asset_ids(self) -> ProjectManifest:
        ids = [a.asset_id for a in self.assets]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for asset_id in ids:
            (duplicates if asset_id in seen else seen).add(asset_id)
        if duplicates:
            raise ValueError(f"duplicate asset_id value(s) in manifest: {sorted(duplicates)}")
        if self.primary_route_asset_id is not None and self.primary_route_asset_id not in seen:
            raise ValueError(
                f"primary_route_asset_id {self.primary_route_asset_id!r} does not match any "
                "registered asset_id"
            )
        return self


def load_project_manifest(path: str | Path) -> tuple[ProjectManifest, Path]:
    """Section 6: parse and validate a manifest YAML file. Returns `(manifest, manifest_dir)` --
    `manifest_dir` is the ONLY base every asset path is ever resolved against (see
    `resolve_asset_path`), never the shell's current working directory."""

    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh) or {}
    manifest = ProjectManifest.model_validate(raw)
    return manifest, manifest_path.parent


def resolve_asset_path(asset: AssetEntry, manifest_dir: Path) -> Path:
    """Section 6: an asset's `path` resolves relative to the manifest's own directory -- an
    already-absolute path is returned unchanged."""

    if asset.path.is_absolute():
        return asset.path
    return (manifest_dir / asset.path).resolve()

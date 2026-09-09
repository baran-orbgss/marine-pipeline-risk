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

from marine_engine.project.categories import (
    ASSET_CATEGORIES,
    EVIDENCE_ROLES,
    LINEAR_REFERENCE_BASES,
    LINEAR_REFERENCE_UNITS,
    PIPELINE_ROUTE,
    ROUTE_RELATIONSHIP_TYPES,
)


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


class RouteReferenceConfig(BaseModel):
    """MAR-027 Section 7: optional project-level canonical route-reference grid configuration.
    `interval_m` is an INDEXING resolution along the canonical route (metres), never a survey
    accuracy or engineering resolution -- and there is no hidden default: when this section is
    absent, no grid interval is invented."""

    model_config = ConfigDict(extra="forbid")

    interval_m: float = Field(gt=0, allow_inf_nan=False)


class LinearReferenceDeclaration(BaseModel):
    """MAR-027 Section 16: an explicit, manifest-declared statement of what an asset's numeric
    chainage/KP values MEAN. The only supported `basis` means exactly: distances along the
    referenced canonical project route measured from that route's geometry-start origin. Never
    inferred from a column name or from `units_declared` (which describes the measured burial
    value, not the chainage column)."""

    model_config = ConfigDict(extra="forbid")

    basis: str
    units: str

    @model_validator(mode="after")
    def _validate_vocabulary(self) -> LinearReferenceDeclaration:
        if self.basis not in LINEAR_REFERENCE_BASES:
            raise ValueError(
                f"unknown linear_reference basis {self.basis!r} -- must be one of "
                f"{sorted(LINEAR_REFERENCE_BASES)}"
            )
        if self.units not in LINEAR_REFERENCE_UNITS:
            raise ValueError(
                f"unsupported linear_reference units {self.units!r} -- must be one of "
                f"{sorted(LINEAR_REFERENCE_UNITS)}"
            )
        return self


class RouteRelationship(BaseModel):
    """MAR-027 Section 8: an explicit, manifest-declared asset -> route relationship. Declaring
    it states only that the asset is referenced to that route; it never proves the asset covers
    the whole route, is representative everywhere along it, or is temporally compatible with
    any other asset (Section 5)."""

    model_config = ConfigDict(extra="forbid")

    route_asset_id: str
    relationship_type: str
    linear_reference: LinearReferenceDeclaration | None = None

    @model_validator(mode="after")
    def _validate_vocabulary(self) -> RouteRelationship:
        if self.relationship_type not in ROUTE_RELATIONSHIP_TYPES:
            raise ValueError(
                f"unknown relationship_type {self.relationship_type!r} -- must be one of "
                f"{sorted(ROUTE_RELATIONSHIP_TYPES)}"
            )
        return self


class AssetEntry(BaseModel):
    """One registered operator asset (Section 6)."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    category: str
    evidence_role: str
    path: Path
    layer: str | None = None
    burial_columns: BurialColumnMapping | None = None
    route_relationship: RouteRelationship | None = None
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
        if (
            self.route_relationship is not None
            and self.route_relationship.route_asset_id == self.asset_id
        ):
            raise ValueError(
                f"asset {self.asset_id!r}: route_relationship.route_asset_id may not reference "
                "the asset itself"
            )
        return self


class ProjectManifest(BaseModel):
    """Root schema for an operator project manifest YAML file (Section 6)."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectIdentity
    primary_route_asset_id: str | None = None
    route_reference: RouteReferenceConfig | None = None
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
        # MAR-027 Section 8: structurally impossible route relationships are caught here --
        # the referenced route must exist and must actually be a PIPELINE_ROUTE asset.
        by_id = {a.asset_id: a for a in self.assets}
        for asset in self.assets:
            rel = asset.route_relationship
            if rel is None:
                continue
            target = by_id.get(rel.route_asset_id)
            if target is None:
                raise ValueError(
                    f"asset {asset.asset_id!r}: route_relationship.route_asset_id "
                    f"{rel.route_asset_id!r} does not match any registered asset_id"
                )
            if target.category != PIPELINE_ROUTE:
                raise ValueError(
                    f"asset {asset.asset_id!r}: route_relationship.route_asset_id "
                    f"{rel.route_asset_id!r} has category {target.category!r}, not "
                    f"{PIPELINE_ROUTE!r}"
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

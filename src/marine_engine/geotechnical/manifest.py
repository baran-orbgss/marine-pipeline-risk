"""Typed, source-specific CPT evidence manifest (MAR-032 Section 33).

Kept deliberately OUTSIDE `marine_engine.config.StudyConfig` and outside the generic
`project.manifest.ProjectManifest`, so neither generic model accumulates geotechnical source
parameters. Mirrors the repository's Pydantic v2 conventions (`ConfigDict(extra="forbid")`).

Paths follow the study-config convention (`config.DataPaths`): relative paths resolve against the
process working directory, i.e. the repository root when the CLI is run from there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    "CptSourceBlock",
    "CptPathsBlock",
    "CptDeclaredBlock",
    "CptEvidenceManifest",
    "load_cpt_evidence_manifest",
]


class CptSourceBlock(BaseModel):
    """Which registered provider resolves the official source, and whether the associated
    factual/report package is acquired too (only when needed to interpret provenance)."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    acquire_reports_package: bool = False

    @field_validator("provider")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source.provider must be a non-empty provider id")
        return value


class CptPathsBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path


class CptDeclaredBlock(BaseModel):
    """USER/operator-declared metadata (Section 14.3). Structurally separate from anything observed
    in the source. `coordinate_crs` is compared SEMANTICALLY against the source's own coordinate
    statement by the provider -- it is never used to overwrite or reproject source coordinates."""

    model_config = ConfigDict(extra="forbid")

    coordinate_crs: str | None = None


class CptEvidenceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source: CptSourceBlock
    paths: CptPathsBlock
    declared: CptDeclaredBlock = CptDeclaredBlock()

    @field_validator("evidence_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence_id must be a non-empty string")
        return value


def load_cpt_evidence_manifest(path: str | Path) -> CptEvidenceManifest:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh) or {}
    return CptEvidenceManifest.model_validate(raw)

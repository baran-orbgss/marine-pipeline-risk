"""The ONLY commands `api/jobs.py` may launch, keyed by a stable `capability_key`.

A small, deliberately conservative subset of the engine's 39 CLI subcommands: only ones whose CLI
help text calls them "generic" AND whose output-directory convention is per-project (so the
result -> layer diffing in `api/jobs.py` can find what they produced). A few commands that sound
like they should belong here were excluded after checking their actual behaviour, not their name:

* `build-noncohesive-mobility` ("Sediment Mobility") requires a project-specific upstream
  metocean / current-normalization / wave-orbital-forcing / combined-bed-shear chain that
  predates the "generic POC" command family and has not been generalized to an arbitrary project.
* `build-free-span-poc` ("Free Span") writes to a shared, non-per-project directory
  (`<processed_dir>/freespan_poc/`), so a run cannot be attributed back to the calling project by
  the result-to-layer diff.
* `build-bathymetry` is coupled to a pipeline/chainage ingestion workflow not verified to
  generalize to a project without a pipeline route.

Where any of the three above already has built output on disk (e.g. PL854's cached evidence),
it still appears as an ordinary map layer via `api/layers.py` -- it just cannot be freshly
re-run generically from this launcher in this release. That is reported to the user as an
explicit `NOT_APPLICABLE` reason (see `_NOT_GENERALIZED` below), not silently hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from api import settings
from api.models import (
    AnalysisCapabilityDescriptor,
    DeclaredSourceRef,
    RequiredInputCheck,
    SupportType,
)
from api.projects import get_project_summary
from marine_engine.project import categories as cat
from marine_engine.project.manifest import load_project_manifest, resolve_asset_path

InputKind = Literal["study_config", "project_manifest", "cpt_evidence_manifest"]


@dataclass(frozen=True)
class AnalysisCapabilityDef:
    capability_key: str
    title: str
    cli_command: str
    input_kind: InputKind
    support_type: SupportType
    requires_bathymetry_asset: bool = False
    requires_route_asset: bool = False


TIER_A: list[AnalysisCapabilityDef] = [
    AnalysisCapabilityDef(
        "terrain",
        "Terrain",
        "build-highres-terrain-poc",
        "study_config",
        "AREA_SURFACE",
        requires_bathymetry_asset=True,
    ),
    AnalysisCapabilityDef(
        "erosion_deposition",
        "Seabed Change",
        "build-seabed-change-poc",
        "study_config",
        "AREA_SURFACE",
        requires_bathymetry_asset=True,
    ),
    AnalysisCapabilityDef(
        "bedforms",
        "Bedforms",
        "build-bedform-morphodynamics-poc",
        "study_config",
        "AREA_VECTOR",
        requires_bathymetry_asset=True,
    ),
    AnalysisCapabilityDef(
        "scour",
        "Scour Susceptibility",
        "build-scour-susceptibility-poc",
        "study_config",
        "LINEAR_ANALYSIS",
        requires_route_asset=True,
    ),
    AnalysisCapabilityDef(
        "burial_exposure",
        "Burial / Exposure",
        "build-burial-exposure-poc",
        "study_config",
        "LINEAR_ANALYSIS",
        requires_route_asset=True,
    ),
    AnalysisCapabilityDef(
        "slope_instability",
        "Slope Screening",
        "build-slope-instability-screening",
        "study_config",
        "AREA_SURFACE",
        requires_bathymetry_asset=True,
    ),
    AnalysisCapabilityDef(
        "cpt_evidence",
        "CPT Evidence",
        "build-cpt-evidence-poc",
        "cpt_evidence_manifest",
        "POINT_EVIDENCE",
    ),
    AnalysisCapabilityDef(
        "project_readiness",
        "Project Readiness",
        "build-project-readiness",
        "project_manifest",
        "SUPPORT_NODE",
    ),
    AnalysisCapabilityDef(
        "project_model",
        "Project Model",
        "build-project-model",
        "project_manifest",
        "SUPPORT_NODE",
    ),
]

CLI_COMMANDS_BY_KEY: dict[str, str] = {c.capability_key: c.cli_command for c in TIER_A}
DEF_BY_KEY: dict[str, AnalysisCapabilityDef] = {c.capability_key: c for c in TIER_A}

_NOT_GENERALIZED: list[tuple[str, str, str, SupportType]] = [
    (
        "sediment_mobility",
        "Sediment Mobility",
        "build-noncohesive-mobility requires a project-specific upstream metocean chain not "
        "generalized in this release; already-built results are shown as a layer where present.",
        "LINEAR_ANALYSIS",
    ),
    (
        "free_span",
        "Free Span",
        "no generic per-project free-span analysis exists in this release (the engine's "
        "free-span command writes to a shared, non-per-project location); already-built "
        "evidence is shown as a layer where present.",
        "LINEAR_ANALYSIS",
    ),
    (
        "bathymetry",
        "Bathymetry",
        "canonical bathymetry ingestion in this engine version is coupled to a pipeline/"
        "chainage workflow not verified to generalize; already-registered bathymetry is shown "
        "as a layer.",
        "AREA_SURFACE",
    ),
]


def _source_path(sources: list[DeclaredSourceRef], kind: str) -> str | None:
    for source in sources:
        if source.kind == kind or (kind == "project_manifest" and source.kind == "ad_hoc_manifest"):
            return source.path
    return None


def _manifest_has_category(manifest_path: str, category: str) -> bool:
    manifest, manifest_dir = load_project_manifest(settings.REPO_ROOT / manifest_path)
    for asset in manifest.assets:
        if asset.category != category:
            continue
        try:
            if resolve_asset_path(asset, manifest_dir).is_file():
                return True
        except Exception:
            continue
    return False


def describe_capabilities(project_id: str) -> list[AnalysisCapabilityDescriptor]:
    summary = get_project_summary(project_id)
    descriptors: list[AnalysisCapabilityDescriptor] = []
    sources = summary.declared_sources if summary else []
    study_config_path = _source_path(sources, "study_config")
    project_manifest_path = _source_path(sources, "project_manifest")
    cpt_manifest_path = _source_path(sources, "cpt_evidence_manifest")

    for definition in TIER_A:
        checks: list[RequiredInputCheck] = []
        input_path: str | None = None
        if definition.input_kind == "study_config":
            input_path = study_config_path
            checks.append(
                RequiredInputCheck(
                    description="study configuration declared for this project",
                    satisfied=input_path is not None,
                )
            )
        elif definition.input_kind == "project_manifest":
            input_path = project_manifest_path
            checks.append(
                RequiredInputCheck(
                    description="project manifest declared for this project",
                    satisfied=input_path is not None,
                )
            )
        else:
            input_path = cpt_manifest_path
            checks.append(
                RequiredInputCheck(
                    description="CPT evidence manifest declared for this project",
                    satisfied=input_path is not None,
                )
            )

        if input_path is None:
            descriptors.append(
                AnalysisCapabilityDescriptor(
                    capability_key=definition.capability_key,
                    title=definition.title,
                    cli_command=definition.cli_command,
                    availability="NOT_APPLICABLE",
                    required_inputs=checks,
                    reasons=[
                        f"this project has no declared {definition.input_kind.replace('_', ' ')}"
                    ],
                    support_type=definition.support_type,
                    disabled=True,
                    disabled_reason="required input source not declared for this project",
                )
            )
            continue

        reasons: list[str] = []
        if definition.requires_bathymetry_asset and project_manifest_path:
            has_bathy = _manifest_has_category(project_manifest_path, cat.BATHYMETRY_RASTER)
            checks.append(
                RequiredInputCheck(
                    description="bathymetry raster registered and present on disk",
                    satisfied=has_bathy,
                )
            )
            if not has_bathy:
                reasons.append("needs a registered bathymetry raster asset")
        if definition.requires_route_asset and project_manifest_path:
            has_route = _manifest_has_category(project_manifest_path, cat.PIPELINE_ROUTE)
            checks.append(
                RequiredInputCheck(
                    description="pipeline/cable route registered and present on disk",
                    satisfied=has_route,
                )
            )
            if not has_route:
                reasons.append("needs a registered pipeline/cable route asset")

        available = all(c.satisfied for c in checks)
        descriptors.append(
            AnalysisCapabilityDescriptor(
                capability_key=definition.capability_key,
                title=definition.title,
                cli_command=definition.cli_command,
                availability="AVAILABLE" if available else "MISSING_INPUTS",
                required_inputs=checks,
                reasons=reasons,
                support_type=definition.support_type,
            )
        )

    for key, title, reason, support_type in _NOT_GENERALIZED:
        descriptors.append(
            AnalysisCapabilityDescriptor(
                capability_key=key,
                title=title,
                cli_command="(not launchable in this release)",
                availability="NOT_APPLICABLE",
                reasons=[reason],
                support_type=support_type,
                disabled=True,
                disabled_reason=reason,
            )
        )
    return descriptors


def input_path_for(project_id: str, capability_key: str) -> str | None:
    definition = DEF_BY_KEY.get(capability_key)
    if definition is None:
        return None
    summary = get_project_summary(project_id)
    if summary is None:
        return None
    kind = definition.input_kind
    return _source_path(summary.declared_sources, kind)

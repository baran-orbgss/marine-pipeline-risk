"""Generic project catalog: discover projects from declared config/manifest sources and
correlate them against real `data/processed/*` output directories.

Deliberately generic -- the four example projects (PL854, Sheringham Shoal 2020, Sheringham Shoal
2008 CPTU, Barrow 2016) are not special-cased anywhere in this module. There are, in fact, THREE
distinct declared-source schemas in this repo (`StudyConfig`, `ProjectManifest`,
`CptEvidenceManifest`) plus a fourth location for ad hoc projects created through this API
(`data/processed/<id>/project/manifest.yaml`, itself `ProjectManifest`-shaped) -- this module scans
all four locations rather than assuming one uniform pattern.

Per-project output-directory naming is NOT declared anywhere in the engine's own schemas; `cli.py`
uses at least four different derivations across command families (`study.id.lower()`,
`pipeline.pipeline_id.lower()`, `project.id` verbatim, or an explicit `paths.processed_dir` for the
CPT schema). Rather than recompute any one of those conventions, this module unions every declared
candidate id/hint against the real directory names actually present under `data/processed/`,
correlating case-insensitively (a real casing mismatch already exists between PL854's study config
and its project manifest).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from api import settings
from api.models import DeclaredSourceRef, ProjectCatalog, ProjectSummary, UnmatchedOutputDir
from api.paths import repo_relative
from marine_engine.config import load_study_config
from marine_engine.geotechnical.manifest import load_cpt_evidence_manifest
from marine_engine.project.manifest import load_project_manifest


def _normalize(identifier: str) -> str:
    return identifier.strip().lower()


@dataclass
class _DeclaredProject:
    project_id: str
    display_name: str
    description: str | None = None
    working_crs: str | None = None
    is_ad_hoc: bool = False
    sources: list[DeclaredSourceRef] = field(default_factory=list)
    output_dir_hints: set[str] = field(default_factory=set)
    explicit_output_root: Path | None = None


def _repo_relative(path: Path) -> str:
    return repo_relative(path, settings.REPO_ROOT)


def _merge(registry: dict[str, _DeclaredProject], project: _DeclaredProject) -> None:
    existing = registry.get(project.project_id)
    if existing is None:
        registry[project.project_id] = project
        return
    existing.sources.extend(project.sources)
    existing.output_dir_hints |= project.output_dir_hints
    if project.explicit_output_root is not None:
        existing.explicit_output_root = project.explicit_output_root
    if project.working_crs and not existing.working_crs:
        existing.working_crs = project.working_crs
    if project.description and not existing.description:
        existing.description = project.description


def _scan_study_configs(registry: dict[str, _DeclaredProject]) -> None:
    if not settings.CONFIGS_DIR.is_dir():
        return
    for cfg_path in sorted(settings.CONFIGS_DIR.glob("*.yaml")):
        try:
            cfg = load_study_config(cfg_path)
        except Exception:
            continue
        project_id = _normalize(cfg.study.id)
        hints = {project_id}
        pipeline_id = cfg.pipeline.get("pipeline_id") if isinstance(cfg.pipeline, dict) else None
        if pipeline_id:
            hints.add(_normalize(str(pipeline_id)))
        _merge(
            registry,
            _DeclaredProject(
                project_id=project_id,
                display_name=cfg.study.name or cfg.study.id,
                description=cfg.study.description or None,
                working_crs=cfg.crs.horizontal,
                sources=[
                    DeclaredSourceRef(
                        kind="study_config", path=_repo_relative(cfg_path), declared_id=cfg.study.id
                    )
                ],
                output_dir_hints=hints,
            ),
        )


def _scan_project_manifests(
    registry: dict[str, _DeclaredProject], directory: Path, *, is_ad_hoc: bool
) -> None:
    if not directory.is_dir():
        return
    for manifest_path in sorted(directory.glob("*.yaml")):
        try:
            manifest, _manifest_dir = load_project_manifest(manifest_path)
        except Exception:
            continue
        project_id = _normalize(manifest.project.id)
        _merge(
            registry,
            _DeclaredProject(
                project_id=project_id,
                display_name=manifest.project.name or manifest.project.id,
                description=manifest.project.description or None,
                working_crs=manifest.project.working_crs,
                is_ad_hoc=is_ad_hoc,
                sources=[
                    DeclaredSourceRef(
                        kind="ad_hoc_manifest" if is_ad_hoc else "project_manifest",
                        path=_repo_relative(manifest_path),
                        declared_id=manifest.project.id,
                    )
                ],
                output_dir_hints={project_id},
                explicit_output_root=manifest_path.parent.parent if is_ad_hoc else None,
            ),
        )


def _scan_ad_hoc_manifests(registry: dict[str, _DeclaredProject]) -> None:
    if not settings.DATA_PROCESSED_DIR.is_dir():
        return
    for project_dir in sorted(settings.DATA_PROCESSED_DIR.iterdir()):
        manifest_path = project_dir / "project" / settings.AD_HOC_MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        try:
            manifest, _manifest_dir = load_project_manifest(manifest_path)
        except Exception:
            continue
        project_id = _normalize(manifest.project.id)
        _merge(
            registry,
            _DeclaredProject(
                project_id=project_id,
                display_name=manifest.project.name or manifest.project.id,
                description=manifest.project.description or None,
                working_crs=manifest.project.working_crs,
                is_ad_hoc=True,
                sources=[
                    DeclaredSourceRef(
                        kind="ad_hoc_manifest",
                        path=_repo_relative(manifest_path),
                        declared_id=manifest.project.id,
                    )
                ],
                output_dir_hints={project_id},
                explicit_output_root=project_dir,
            ),
        )


def _scan_cpt_evidence_manifests(registry: dict[str, _DeclaredProject]) -> None:
    if not settings.GEOTECHNICAL_CONFIGS_DIR.is_dir():
        return
    for manifest_path in sorted(settings.GEOTECHNICAL_CONFIGS_DIR.glob("*.yaml")):
        try:
            manifest = load_cpt_evidence_manifest(manifest_path)
        except Exception:
            continue
        project_id = _normalize(manifest.evidence_id)
        processed_dir = manifest.paths.processed_dir
        if not processed_dir.is_absolute():
            processed_dir = settings.REPO_ROOT / processed_dir
        _merge(
            registry,
            _DeclaredProject(
                project_id=project_id,
                display_name=manifest.evidence_id,
                working_crs=manifest.declared.coordinate_crs,
                sources=[
                    DeclaredSourceRef(
                        kind="cpt_evidence_manifest",
                        path=_repo_relative(manifest_path),
                        declared_id=manifest.evidence_id,
                    )
                ],
                output_dir_hints={project_id},
                explicit_output_root=processed_dir,
            ),
        )


def _has_registered_project(output_root: Path) -> bool:
    project_dir = output_root / "project"
    return (project_dir / "project_readiness.json").is_file() or (
        project_dir / "canonical_project_model.json"
    ).is_file()


def discover_projects() -> ProjectCatalog:
    registry: dict[str, _DeclaredProject] = {}
    _scan_study_configs(registry)
    _scan_project_manifests(registry, settings.PROJECT_MANIFESTS_DIR, is_ad_hoc=False)
    _scan_cpt_evidence_manifests(registry)
    _scan_ad_hoc_manifests(registry)

    real_dirs: dict[str, Path] = {}
    if settings.DATA_PROCESSED_DIR.is_dir():
        for entry in settings.DATA_PROCESSED_DIR.iterdir():
            if entry.is_dir():
                real_dirs[_normalize(entry.name)] = entry

    summaries: list[ProjectSummary] = []
    matched_dir_names: set[str] = set()
    for project in registry.values():
        output_root: Path | None = project.explicit_output_root
        if output_root is None:
            for hint in project.output_dir_hints:
                candidate = real_dirs.get(hint)
                if candidate is not None:
                    output_root = candidate
                    break
        if output_root is not None and output_root.is_dir():
            matched_dir_names.add(_normalize(output_root.name))
        else:
            output_root = None
        summaries.append(
            ProjectSummary(
                project_id=project.project_id,
                display_name=project.display_name,
                description=project.description,
                declared_sources=project.sources,
                working_crs=project.working_crs,
                output_root=_repo_relative(output_root) if output_root else None,
                has_registered_project=bool(output_root and _has_registered_project(output_root)),
                is_ad_hoc=project.is_ad_hoc,
            )
        )

    unmatched = [
        UnmatchedOutputDir(
            directory=_repo_relative(path),
            reason=(
                "no declared study config, project manifest, or CPT evidence manifest points here"
            ),
        )
        for name, path in sorted(real_dirs.items())
        if name not in matched_dir_names
    ]

    summaries.sort(key=lambda p: p.display_name.lower())
    return ProjectCatalog(projects=summaries, unmatched_output_dirs=unmatched)


def get_project_summary(project_id: str) -> ProjectSummary | None:
    catalog = discover_projects()
    normalized = _normalize(project_id)
    for summary in catalog.projects:
        if summary.project_id == normalized:
            return summary
    return None

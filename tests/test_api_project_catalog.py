"""Tests for api.projects: generic discovery across the three declared-source schemas plus the
ad hoc manifest location, correlated case-insensitively against real output directories."""

from __future__ import annotations

from pathlib import Path

import yaml
from api.projects import discover_projects, get_project_summary


def _write_study_config(path: Path, *, study_id: str, name: str, crs: str = "EPSG:32631") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "study": {"id": study_id, "name": name},
                "crs": {"horizontal": crs},
            }
        ),
        encoding="utf-8",
    )


def _write_project_manifest(
    path: Path, *, project_id: str, name: str, crs: str = "EPSG:32631"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "project": {"id": project_id, "name": name, "working_crs": crs},
                "assets": [],
            }
        ),
        encoding="utf-8",
    )


def _write_cpt_manifest(path: Path, *, evidence_id: str, processed_dir: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "evidence_id": evidence_id,
                "source": {"provider": "test_provider"},
                "paths": {
                    "raw_dir": "data/raw",
                    "interim_dir": "data/interim",
                    "processed_dir": processed_dir,
                },
            }
        ),
        encoding="utf-8",
    )


def test_study_config_and_project_manifest_merge_case_insensitively(api_sandbox: Path) -> None:
    _write_study_config(
        api_sandbox / "configs" / "pl854.yaml", study_id="PL854", name="Anglia A -> LOGGS"
    )
    _write_project_manifest(
        api_sandbox / "configs" / "project_manifests" / "pl854.yaml",
        project_id="pl854",
        name="Anglia A -> LOGGS",
    )
    (api_sandbox / "data" / "processed" / "pl854").mkdir(parents=True)

    catalog = discover_projects()

    assert len(catalog.projects) == 1
    project = catalog.projects[0]
    assert project.project_id == "pl854"
    assert {s.kind for s in project.declared_sources} == {"study_config", "project_manifest"}
    assert project.output_root == "data/processed/pl854"


def test_cpt_evidence_manifest_discovered_without_a_project_manifest(api_sandbox: Path) -> None:
    _write_cpt_manifest(
        api_sandbox / "configs" / "geotechnical" / "sheringham_shoal_2008_cptu.yaml",
        evidence_id="sheringham_shoal_2008_cptu",
        processed_dir="data/processed/sheringham_shoal_2008_cptu",
    )
    (api_sandbox / "data" / "processed" / "sheringham_shoal_2008_cptu").mkdir(parents=True)

    catalog = discover_projects()

    assert len(catalog.projects) == 1
    project = catalog.projects[0]
    assert project.project_id == "sheringham_shoal_2008_cptu"
    assert project.declared_sources[0].kind == "cpt_evidence_manifest"
    assert project.output_root == "data/processed/sheringham_shoal_2008_cptu"


def test_ad_hoc_manifest_is_discovered_and_flagged(api_sandbox: Path) -> None:
    _write_project_manifest(
        api_sandbox / "data" / "processed" / "my_new_project" / "project" / "manifest.yaml",
        project_id="my_new_project",
        name="My New Project",
    )

    catalog = discover_projects()

    assert len(catalog.projects) == 1
    project = catalog.projects[0]
    assert project.is_ad_hoc is True
    assert project.output_root == "data/processed/my_new_project"


def test_declared_project_with_no_output_yet_still_appears(api_sandbox: Path) -> None:
    _write_study_config(
        api_sandbox / "configs" / "brand_new.yaml", study_id="BrandNew", name="Brand New Study"
    )

    catalog = discover_projects()

    assert len(catalog.projects) == 1
    project = catalog.projects[0]
    assert project.output_root is None
    assert project.layer_count is None


def test_undeclared_output_directory_is_never_a_primary_project(api_sandbox: Path) -> None:
    _write_study_config(api_sandbox / "configs" / "pl854.yaml", study_id="PL854", name="PL854")
    (api_sandbox / "data" / "processed" / "pl854").mkdir(parents=True)
    # A generic/shared POC output directory with no declared source pointing at it (mirrors real
    # repo directories like `freespan_poc/`, `analogs/`, `scour_poc/`).
    (api_sandbox / "data" / "processed" / "freespan_poc").mkdir(parents=True)

    catalog = discover_projects()

    project_ids = {p.project_id for p in catalog.projects}
    assert "pl854" in project_ids
    assert "freespan_poc" not in project_ids
    unmatched_dirs = {u.directory for u in catalog.unmatched_output_dirs}
    assert "data/processed/freespan_poc" in unmatched_dirs


def test_get_project_summary_is_case_insensitive(api_sandbox: Path) -> None:
    _write_study_config(api_sandbox / "configs" / "pl854.yaml", study_id="PL854", name="PL854")
    (api_sandbox / "data" / "processed" / "pl854").mkdir(parents=True)

    assert get_project_summary("PL854") is not None
    assert get_project_summary("pl854") is not None
    assert get_project_summary("does-not-exist") is None

"""Shared pytest fixtures."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def pl854_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "pl854.yaml"


@pytest.fixture
def api_sandbox(tmp_path, monkeypatch):
    """UI-003: an isolated, empty repo-shaped tree for `api/**` tests. Points every path constant
    in `api.settings` at `tmp_path` instead of the real repository -- `api/**` tests must never
    depend on the real (gitignored, CI-absent) `data/processed/**` tree, only on fixtures they
    build themselves."""

    from api import settings

    configs_dir = tmp_path / "configs"
    project_manifests_dir = configs_dir / "project_manifests"
    geotechnical_dir = configs_dir / "geotechnical"
    data_dir = tmp_path / "data"
    data_processed_dir = data_dir / "processed"
    data_staging_dir = data_dir / "staging"
    tile_cache_dir = tmp_path / "runtime" / "tiles"
    job_runs_dir = tmp_path / "runtime" / "runs"
    for directory in (
        project_manifests_dir,
        geotechnical_dir,
        data_processed_dir,
        data_staging_dir,
        tile_cache_dir,
        job_runs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(settings, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(settings, "PROJECT_MANIFESTS_DIR", project_manifests_dir)
    monkeypatch.setattr(settings, "GEOTECHNICAL_CONFIGS_DIR", geotechnical_dir)
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "DATA_PROCESSED_DIR", data_processed_dir)
    monkeypatch.setattr(settings, "DATA_STAGING_DIR", data_staging_dir)
    monkeypatch.setattr(settings, "TILE_CACHE_DIR", tile_cache_dir)
    monkeypatch.setattr(settings, "JOB_RUNS_DIR", job_runs_dir)
    return tmp_path

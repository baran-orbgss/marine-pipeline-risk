"""Repo-relative paths and local-only runtime locations.

This is a local, single-user tool: no secrets, no environment-driven credentials. Runtime scratch
state (tile cache, job logs) lives under the OS temp directory, never under `data/` or `configs/`,
so it can never be mistaken for a canonical or reviewed artifact.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
PROJECT_MANIFESTS_DIR = CONFIGS_DIR / "project_manifests"
GEOTECHNICAL_CONFIGS_DIR = CONFIGS_DIR / "geotechnical"
DATA_DIR = REPO_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_STAGING_DIR = DATA_DIR / "staging"
WEB_DIST_DIR = REPO_ROOT / "web" / "dist"

_RUNTIME_ROOT = Path(tempfile.gettempdir()) / "marine_engine_api"
TILE_CACHE_DIR = _RUNTIME_ROOT / "tiles"
JOB_RUNS_DIR = _RUNTIME_ROOT / "runs"

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

AD_HOC_MANIFEST_FILENAME = "manifest.yaml"


def ensure_runtime_dirs() -> None:
    for directory in (DATA_STAGING_DIR, TILE_CACHE_DIR, JOB_RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    keep = DATA_STAGING_DIR / ".gitkeep"
    if not keep.exists():
        keep.touch()

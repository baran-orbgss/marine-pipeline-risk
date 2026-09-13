"""Upload/staging: session-scoped file drops, immediate spatial inspection, explicit CSV
coordinate declaration, and promotion to a real (ad hoc) project manifest.

Every path built from a client-supplied session id, file id, or filename goes through
`api/paths.safe_join`/`safe_filename`. Uploaded bytes are never modified in place; "promoting" a
session only ever WRITES a new manifest file that POINTS AT the staged file's existing location --
it never copies, moves, resamples, or rewrites the uploaded bytes (source immutability).

A staging session is browsable as a lightweight ad hoc project immediately after upload (see
`api/main.py`'s staging-preview routes), with zero manual CRS/AOI entry -- promotion is a
separate, explicit "Save Project" action, not a prerequisite for seeing the data on the map.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from api import settings
from api.discovery import (
    SUPPORTED_EXTENSIONS,
    bounds_to_wgs84,
    build_csv_point_geometry,
    inspect_file,
)
from api.errors import NotFoundError, UnsafePathRequestError
from api.models import AssetDeclaration, BoundingBox, StagedFile, StagingSessionRecord
from api.paths import safe_filename, safe_join
from marine_engine.project import categories as cat

settings.ensure_runtime_dirs()

_SESSIONS: dict[str, StagingSessionRecord] = {}
_FILES: dict[str, dict[str, StagedFile]] = {}


def create_session() -> StagingSessionRecord:
    session_id = uuid.uuid4().hex
    record = StagingSessionRecord(session_id=session_id, created_at=datetime.now(UTC))
    _SESSIONS[session_id] = record
    _FILES[session_id] = {}
    safe_join(settings.DATA_STAGING_DIR, session_id).mkdir(parents=True, exist_ok=True)
    return record


def _session_dir(session_id: str) -> Path:
    if session_id not in _SESSIONS:
        raise NotFoundError(f"unknown staging session {session_id!r}")
    return safe_join(settings.DATA_STAGING_DIR, session_id)


def save_upload(session_id: str, filename: str, content: bytes) -> StagedFile:
    session_dir = _session_dir(session_id)
    clean_name = safe_filename(filename)
    suffix = Path(clean_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsafePathRequestError(f"unsupported file type: {suffix}")

    file_id = uuid.uuid4().hex
    dest = safe_join(session_dir, f"{file_id}_{clean_name}")
    dest.write_bytes(content)
    try:
        inspection = inspect_file(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise UnsafePathRequestError(f"could not inspect uploaded file: {exc}") from exc

    staged = StagedFile(
        file_id=file_id,
        filename=clean_name,
        relative_path=dest.relative_to(settings.REPO_ROOT).as_posix(),
        inspection=inspection,
    )
    _FILES[session_id][file_id] = staged
    return staged


def declare_coordinate_columns(
    session_id: str, file_id: str, *, x_column: str, y_column: str, crs: str
) -> StagedFile:
    files = _FILES.get(session_id)
    if files is None:
        raise NotFoundError(f"unknown staging session {session_id!r}")
    staged = files.get(file_id)
    if staged is None:
        raise NotFoundError(f"unknown staged file {file_id!r}")

    absolute = settings.REPO_ROOT / staged.relative_path
    gdf = build_csv_point_geometry(absolute, x_column=x_column, y_column=y_column, crs=crs)
    minx, miny, maxx, maxy = gdf.total_bounds
    bounds_native = BoundingBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=crs)
    bounds_wgs84 = bounds_to_wgs84(minx, miny, maxx, maxy, crs)
    updated_inspection = staged.inspection.model_copy(
        update={
            "kind": "vector",
            "observed_crs": crs,
            "bounds_native": bounds_native,
            "bounds_wgs84": bounds_wgs84,
            "geometry_type": "Point",
            "feature_count": len(gdf),
            "coordinate_columns_declared": True,
            "warnings": [],
        }
    )
    staged = staged.model_copy(update={"inspection": updated_inspection})
    files[file_id] = staged
    return staged


def list_staged_files(session_id: str) -> list[StagedFile]:
    if session_id not in _SESSIONS:
        raise NotFoundError(f"unknown staging session {session_id!r}")
    return list(_FILES.get(session_id, {}).values())


def get_staged_file(session_id: str, file_id: str) -> StagedFile | None:
    return _FILES.get(session_id, {}).get(file_id)


def promote_session(
    session_id: str,
    *,
    project_id: str,
    display_name: str,
    working_crs: str,
    asset_declarations: dict[str, AssetDeclaration],
) -> Path:
    files = list_staged_files(session_id)
    if not files:
        raise UnsafePathRequestError("staging session has no files to promote")

    normalized_id = project_id.strip().lower()
    project_dir = settings.DATA_PROCESSED_DIR / normalized_id / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project_dir / settings.AD_HOC_MANIFEST_FILENAME

    assets: list[dict] = []
    for staged in files:
        declaration = asset_declarations.get(staged.file_id, AssetDeclaration())
        category = declaration.category or cat.OTHER
        evidence_role = declaration.evidence_role or cat.DERIVED
        source_name = declaration.source_name or staged.filename
        source_absolute = settings.REPO_ROOT / staged.relative_path
        relative_to_manifest = Path(os.path.relpath(source_absolute, project_dir)).as_posix()
        assets.append(
            {
                "asset_id": staged.file_id,
                "category": category,
                "evidence_role": evidence_role,
                "path": relative_to_manifest,
                "provenance": {"source_name": source_name},
            }
        )

    manifest_dict = {
        "project": {
            "id": normalized_id,
            "name": display_name,
            "description": "",
            "working_crs": working_crs,
        },
        "assets": assets,
    }
    manifest_path.write_text(yaml.safe_dump(manifest_dict, sort_keys=False), encoding="utf-8")
    return manifest_path

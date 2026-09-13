"""Analysis job execution: allowlisted commands only, argument arrays, never a shell.

A job snapshots the project's output directory before launch and diffs it after a successful
exit, registering only the new/changed files as layers -- no per-capability special-casing is
needed to satisfy "a successful analysis appears as a map layer automatically" (ticket S32). The
snapshot/diff is recomputed from a fresh project lookup both before and after the run so a
project's very first run (which creates its output directory) is handled the same way as a rerun.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from api import allowlist, settings
from api.errors import CapabilityNotAllowlistedError, CapabilityNotAvailableError
from api.layers import build_layer_catalog
from api.models import JobRecord, JobState, SimpleJobStatus
from api.projects import get_project_summary

settings.ensure_runtime_dirs()

_JOBS: dict[str, JobRecord] = {}
_JOB_TASKS: dict[str, asyncio.Task[None]] = {}

_SIMPLE_STATUS_FOR: dict[JobState, SimpleJobStatus] = {
    "QUEUED": "Preparing data",
    "RUNNING": "Running analysis",
    "SUCCEEDED": "Ready",
    "FAILED": "Failed",
}


def _output_root_for(project_id: str) -> Path | None:
    summary = get_project_summary(project_id)
    if summary is None or summary.output_root is None:
        return None
    return settings.REPO_ROOT / summary.output_root


def _snapshot(output_root: Path | None) -> dict[Path, tuple[int, int]]:
    if output_root is None or not output_root.is_dir():
        return {}
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in output_root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            snapshot[path.resolve()] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


async def _run_job(job_id: str) -> None:
    record = _JOBS[job_id]
    record.status = "RUNNING"
    record.simple_status = _SIMPLE_STATUS_FOR["RUNNING"]
    record.started_at = datetime.now(UTC)

    before = _snapshot(_output_root_for(record.project_id))
    log_dir = settings.JOB_RUNS_DIR / job_id
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
        process = await asyncio.create_subprocess_exec(
            *record.argv, stdout=out_f, stderr=err_f, cwd=str(settings.REPO_ROOT)
        )
        exit_code = await process.wait()

    record.exit_code = exit_code
    record.finished_at = datetime.now(UTC)
    if exit_code != 0:
        record.status = "FAILED"
        record.simple_status = _SIMPLE_STATUS_FOR["FAILED"]
        record.error_summary = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        return

    record.simple_status = "Building map layer"
    after_root = _output_root_for(record.project_id)
    after = _snapshot(after_root)
    changed = {path for path, value in after.items() if before.get(path) != value}
    produced: list[str] = []
    if changed:
        catalog = build_layer_catalog(record.project_id)
        for layer in catalog.layers:
            absolute = (settings.REPO_ROOT / layer.relative_path).resolve()
            if absolute in changed:
                produced.append(layer.layer_id)
    record.produced_layers = produced
    record.status = "SUCCEEDED"
    record.simple_status = _SIMPLE_STATUS_FOR["SUCCEEDED"]


def submit_job(
    project_id: str,
    capability_key: str,
    *,
    allow_network: bool = False,
    extra_args: list[str] | None = None,
) -> JobRecord:
    definition = allowlist.DEF_BY_KEY.get(capability_key)
    if definition is None:
        raise CapabilityNotAllowlistedError(f"capability {capability_key!r} is not allowlisted")

    descriptors = allowlist.describe_capabilities(project_id)
    descriptor = next((d for d in descriptors if d.capability_key == capability_key), None)
    if descriptor is None or descriptor.availability != "AVAILABLE":
        raise CapabilityNotAvailableError(
            f"capability {capability_key!r} is not available for project {project_id!r}"
        )
    if descriptor.requires_network and not allow_network:
        raise CapabilityNotAvailableError(
            f"capability {capability_key!r} requires network access; not authorized for this run"
        )

    input_path = allowlist.input_path_for(project_id, capability_key)
    if input_path is None:
        raise CapabilityNotAvailableError(f"no input path resolved for {capability_key!r}")

    argv = [
        sys.executable,
        "-m",
        "marine_engine.cli",
        definition.cli_command,
        str(settings.REPO_ROOT / input_path),
        *(extra_args or []),
    ]

    job_id = uuid.uuid4().hex
    record = JobRecord(
        job_id=job_id,
        project_id=project_id,
        capability_key=capability_key,
        cli_command=definition.cli_command,
        argv=argv,
        status="QUEUED",
        simple_status=_SIMPLE_STATUS_FOR["QUEUED"],
    )
    _JOBS[job_id] = record
    _JOB_TASKS[job_id] = asyncio.create_task(_run_job(job_id))
    return record


def get_job(job_id: str) -> JobRecord | None:
    return _JOBS.get(job_id)


def get_job_log(job_id: str) -> str:
    log_dir = settings.JOB_RUNS_DIR / job_id
    parts: list[str] = []
    for name in ("stdout.log", "stderr.log"):
        path = log_dir / name
        if path.is_file():
            parts.append(f"--- {name} ---\n{path.read_text(encoding='utf-8', errors='replace')}")
    return "\n".join(parts)


def list_jobs(project_id: str | None = None) -> list[JobRecord]:
    jobs = list(_JOBS.values())
    if project_id:
        jobs = [j for j in jobs if j.project_id == project_id]
    epoch = datetime.min.replace(tzinfo=UTC)
    return sorted(jobs, key=lambda j: j.started_at or epoch, reverse=True)

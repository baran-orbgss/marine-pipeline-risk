"""Tests for api.jobs: only allowlisted, available capabilities may be launched, invocation is
always an argument-array subprocess (never a shell string), and a successful run's new output
files -- and only those -- become registered layers, with no per-capability special-casing."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from api import jobs
from api.errors import CapabilityNotAllowlistedError, CapabilityNotAvailableError
from api.jobs import _snapshot, submit_job
from rasterio.transform import from_origin


def _write_pl854_with_bathymetry_asset(api_sandbox: Path) -> None:
    (api_sandbox / "configs" / "pl854.yaml").write_text(
        yaml.safe_dump(
            {"study": {"id": "pl854", "name": "PL854"}, "crs": {"horizontal": "EPSG:32631"}}
        ),
        encoding="utf-8",
    )
    raster_path = api_sandbox / "data" / "raw" / "bathy.tif"
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500000, 5900000, 1, 1),
    ) as dst:
        dst.write(np.zeros((4, 4), dtype="float32"), 1)
    manifest_path = api_sandbox / "configs" / "project_manifests" / "pl854.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "project": {"id": "pl854", "name": "PL854", "working_crs": "EPSG:32631"},
                "assets": [
                    {
                        "asset_id": "bathy",
                        "category": "BATHYMETRY_RASTER",
                        "evidence_role": "DERIVED",
                        "path": "../../data/raw/bathy.tif",
                        "provenance": {"source_name": "test"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (api_sandbox / "data" / "processed" / "pl854").mkdir(parents=True)


def test_snapshot_detects_new_and_changed_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "a.txt").write_text("1")
    before = _snapshot(output_dir)

    (output_dir / "b.txt").write_text("2")
    (output_dir / "a.txt").write_text("1-changed")
    after = _snapshot(output_dir)

    changed = {path for path, value in after.items() if before.get(path) != value}
    assert (output_dir / "b.txt").resolve() in changed
    assert (output_dir / "a.txt").resolve() in changed


def test_submit_job_rejects_non_allowlisted_capability(api_sandbox: Path) -> None:
    _write_pl854_with_bathymetry_asset(api_sandbox)
    with pytest.raises(CapabilityNotAllowlistedError):
        asyncio.run(_submit(api_sandbox, "pl854", "not_a_real_capability"))


def test_submit_job_rejects_capability_without_required_inputs(api_sandbox: Path) -> None:
    (api_sandbox / "configs" / "pl854.yaml").write_text(
        yaml.safe_dump(
            {"study": {"id": "pl854", "name": "PL854"}, "crs": {"horizontal": "EPSG:32631"}}
        ),
        encoding="utf-8",
    )
    (api_sandbox / "configs" / "project_manifests" / "pl854.yaml").write_text(
        yaml.safe_dump(
            {"project": {"id": "pl854", "name": "PL854", "working_crs": "EPSG:32631"}, "assets": []}
        ),
        encoding="utf-8",
    )
    (api_sandbox / "data" / "processed" / "pl854").mkdir(parents=True)
    # "terrain" requires a registered bathymetry asset, which this project does not have.
    with pytest.raises(CapabilityNotAvailableError):
        asyncio.run(_submit(api_sandbox, "pl854", "terrain"))


async def _submit(api_sandbox: Path, project_id: str, capability_key: str):
    return submit_job(project_id, capability_key)


def test_submitted_job_runs_an_argument_array_never_a_shell_string_and_registers_new_layer(
    api_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pl854_with_bathymetry_asset(api_sandbox)
    captured: dict = {}

    class _FakeProcess:
        async def wait(self) -> int:
            # Simulate the real CLI command's effect: it would have written a new terrain
            # raster into the project's own output directory.
            terrain_dir = api_sandbox / "data" / "processed" / "pl854" / "terrain"
            terrain_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                terrain_dir / "slope.tif",
                "w",
                driver="GTiff",
                height=4,
                width=4,
                count=1,
                dtype="float32",
                crs="EPSG:32631",
                transform=from_origin(500000, 5900000, 1, 1),
            ) as dst:
                dst.write(np.ones((4, 4), dtype="float32"), 1)
            return 0

    async def _fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    async def scenario():
        record = submit_job("pl854", "terrain")
        await jobs._JOB_TASKS[record.job_id]
        return jobs.get_job(record.job_id)

    record = asyncio.run(scenario())

    assert isinstance(captured["args"], tuple)
    assert all(isinstance(arg, str) for arg in captured["args"])
    assert "shell" not in captured["kwargs"]
    assert captured["args"][2] == "marine_engine.cli"
    assert captured["args"][3] == "build-highres-terrain-poc"

    assert record is not None
    assert record.status == "SUCCEEDED"
    assert record.simple_status == "Ready"
    assert any("terrain/slope.tif" in layer_id for layer_id in record.produced_layers)

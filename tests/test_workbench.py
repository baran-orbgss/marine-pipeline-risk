"""UI-001 engineering workbench: pure helper tests (no browser, no Streamlit session).

Proves the presentation registry is complete and honest, the Test Lab cannot run anything outside
the allowlist or any live test, the output inspector stays inside the local data roots, reads
metadata lazily and never fabricates a missing file, and the UI never imports or mutates the
scientific engine."""

from __future__ import annotations

import ast
import inspect
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ui import REPO_ROOT, repo_state
from ui import capability_registry as registry
from ui import guided_story as gs
from ui import map_view as mv
from ui import output_inspector as oi
from ui import test_runner as tr

UI_MODULES = (registry, tr, oi, repo_state, gs, mv)
LIVE_FILES = {
    "tests/test_bathymetry_live.py",
    "tests/test_metocean_live.py",
    "tests/test_nsta_freespan_live.py",
    "tests/test_nsta_live.py",
    "tests/test_sediment_live.py",
}


# --- capability registry -----------------------------------------------------------------------


def test_registry_has_exactly_fifteen_hazards_in_canonical_order():
    assert len(registry.HAZARDS) == 15
    assert [c.hazard_number for c in registry.HAZARDS] == list(range(1, 16))
    assert len({c.id for c in registry.HAZARDS}) == 15
    assert [c.title for c in registry.HAZARDS] == [
        "Terrain",
        "Bedforms",
        "Sediment Mobility",
        "Scour",
        "Burial / Exposure",
        "Free Span",
        "Erosion / Deposition",
        "Sediment Transport Intensity",
        "Slope Instability",
        "Liquefaction",
        "Shallow Gas",
        "Fault / Structural Crossing",
        "Boulder / Hardground",
        "Infrastructure / Crossings",
        "Route Constraint / Suitability",
    ]


def test_maturity_vocabulary_and_initial_state():
    assert registry.MATURITY_VOCABULARY == (
        "QUALIFIED_POC",
        "FOUNDATION_READY",
        "PARTIAL",
        "UNDER_CONSTRUCTION",
    )
    for cap in registry.HAZARDS:
        assert cap.maturity in registry.MATURITY_VOCABULARY, cap.id
    for sup in registry.SUPPORTING:
        assert sup.maturity in registry.MATURITY_VOCABULARY, sup.id
    by_id = {c.id: c.maturity for c in registry.HAZARDS}
    assert by_id["liquefaction"] == registry.FOUNDATION_READY
    assert by_id["liquefaction"] != registry.QUALIFIED_POC
    assert {k for k, v in by_id.items() if v == registry.UNDER_CONSTRUCTION} == {
        "shallow_gas",
        "fault_structural_crossing",
        "route_suitability",
    }
    assert {k for k, v in by_id.items() if v == registry.PARTIAL} == {
        "boulder_hardground",
        "infrastructure_crossings",
    }
    assert sum(1 for v in by_id.values() if v == registry.QUALIFIED_POC) == 9
    # Summary strip is computed from the registry, never hard-coded.
    assert registry.maturity_counts() == {
        registry.QUALIFIED_POC: 9,
        registry.FOUNDATION_READY: 1,
        registry.PARTIAL: 2,
        registry.UNDER_CONSTRUCTION: 3,
    }
    assert sum(registry.maturity_counts().values()) == 15


def test_status_semantics_are_explicitly_not_risk():
    notice = registry.STATUS_SEMANTICS_NOTICE
    for phrase in ("NOT hazard severity", "NOT a risk level", "NOT engineering acceptance"):
        assert phrase in notice
    for meaning in registry.MATURITY_MEANING.values():
        assert "risk" not in meaning.lower() or "not" in meaning.lower()


def test_registry_paths_exist_and_limits_panel_is_mandatory():
    for cap in registry.HAZARDS:
        assert cap.limitations, cap.id
        for t in cap.tests:
            assert (REPO_ROOT / t).is_file(), t
            assert t not in LIVE_FILES, t
        for cfg in cap.configs:
            assert (REPO_ROOT / cfg).is_file(), cfg
        for pattern in cap.output_patterns:
            assert ".." not in pattern.glob and not pattern.glob.startswith("/")
            assert pattern.evidence_kind in registry.EVIDENCE_KINDS
    for sup in registry.SUPPORTING:
        for t in sup.tests:
            assert (REPO_ROOT / t).is_file(), t
            assert t not in LIVE_FILES, t
    for project in registry.PROJECTS:
        for cfg in project.configs:
            assert (REPO_ROOT / cfg).is_file(), cfg
        for cid in project.capability_ids:
            registry.hazard_by_id(cid)


def test_under_construction_capabilities_claim_nothing():
    for cap in registry.HAZARDS:
        if cap.maturity == registry.UNDER_CONSTRUCTION:
            assert cap.tests == () and cap.configs == () and cap.output_patterns == ()
            assert cap.outputs == () and cap.cli_commands == ()
            assert "Under construction" in " ".join(cap.limitations)


def test_supporting_and_project_registries():
    assert len(registry.SUPPORTING) == 16
    assert [s.title for s in registry.SUPPORTING] == [
        "Route ingestion",
        "AOI",
        "Chainage / KP",
        "Bathymetry discovery",
        "Bathymetry acquisition",
        "Canonical bathymetry",
        "Metocean currents",
        "Wave orbital forcing",
        "Combined wave-current bed shear",
        "Sediment evidence",
        "Generic project ingestion",
        "CRS integrity",
        "Cross-asset route linkage",
        "Engineering Evidence Atlas",
        "CI / audit",
        "CPT/CPTU evidence",
    ]
    assert [p.id for p in registry.PROJECTS] == [
        "pl854",
        "sheringham_shoal_2020",
        "sheringham_shoal_2008_cptu",
        "barrow_2016",
    ]
    assert all(p.evidence_kind == registry.REAL_SOURCE for p in registry.PROJECTS)
    with pytest.raises(KeyError):
        registry.project_by_id("nope")
    with pytest.raises(KeyError):
        registry.hazard_by_id("nope")


def test_liquefaction_card_matches_ticket():
    card = registry.LIQUEFACTION_CARD
    assert card["maturity"] == registry.FOUNDATION_READY
    assert "138,514 canonical measurement rows" in card["available"]
    assert "100 parsed CPT tests" in card["available"]
    assert {"qc", "fs", "u2", "schema + lineage integrity"} <= set(card["available"])
    assert {"CRR", "CSR", "earthquake triggering FS"} <= set(card["under_construction"])
    assert {"LPI", "settlement", "lateral spreading"} <= set(card["not_implemented"])


def test_registry_validation_rejects_bad_metadata():
    with pytest.raises(ValueError):
        registry.OutputPattern("x", "../escape/*.json")
    with pytest.raises(ValueError):
        registry.OutputPattern("x", "processed/a.json", evidence_kind="SCORE_95")
    base = registry.hazard_by_id("terrain")
    with pytest.raises(ValueError):
        registry.Capability(**{**base.__dict__, "maturity": "PRODUCTION_READY"})
    with pytest.raises(ValueError):
        registry.Capability(**{**base.__dict__, "limitations": ()})
    with pytest.raises(ValueError):
        registry.Capability(**{**base.__dict__, "tests": ("src/marine_engine/cli.py",)})


# --- test runner: allowlist, traversal, live rejection, commands --------------------------------


def test_allowlist_is_the_registry_and_excludes_live_files():
    allowlisted = set(tr.allowlisted_test_files())
    assert allowlisted == set(registry.all_registered_test_files())
    assert allowlisted.isdisjoint(LIVE_FILES)
    assert "tests/test_slope_stability.py" in allowlisted
    assert "tests/test_workbench.py" not in allowlisted  # not registered to a capability


@pytest.mark.parametrize(
    "bad",
    [
        "tests/test_bathymetry_live.py",  # exists, live, unregistered
        "tests/test_workbench.py",  # exists, unregistered
        "tests/../pyproject.toml",
        "../tests/test_aoi.py",
        "tests/test_aoi.py; rm -rf /",
        "tests/test_aoi.py && echo pwned",
        "tests/test_aoi.py|cat",
        "tests\\test_aoi.py",
        "/etc/passwd",
        "C:\\Windows\\System32\\cmd.exe",
        "src/marine_engine/cli.py",
        "",
        None,
        123,
    ],
)
def test_resolve_test_path_rejects_unregistered_traversal_and_metacharacters(bad):
    with pytest.raises(tr.TestRunnerError):
        tr.resolve_test_path(bad)  # type: ignore[arg-type]
    with pytest.raises(tr.TestRunnerError):
        tr.validate_node_ids([bad])  # type: ignore[list-item]


def test_resolve_test_path_accepts_registered_file_inside_tests_dir():
    path = tr.resolve_test_path("tests/test_slope_stability.py")
    assert path.is_file()
    assert (REPO_ROOT / "tests").resolve() in path.parents


def test_ast_discovery_lists_test_functions_and_selected_source():
    functions = tr.discover_test_functions("tests/test_slope_stability.py")
    assert functions and all(f.name.startswith("test_") for f in functions)
    assert not any(f.live for f in functions)
    names = [f.name for f in functions]
    assert len(names) == len(set(names))
    first = functions[0]
    assert first.node_id == f"tests/test_slope_stability.py::{first.name}"
    source = tr.extract_function_source("tests/test_slope_stability.py", first.name)
    assert re.search(rf"^def {first.name}\(", source, re.M)
    whole = (REPO_ROOT / "tests/test_slope_stability.py").read_text(encoding="utf-8")
    assert len(source) < len(whole) / 4  # only the selected function, not the file dump
    with pytest.raises(tr.TestRunnerError):
        tr.extract_function_source("tests/test_slope_stability.py", "not_a_test")
    with pytest.raises(tr.TestRunnerError):
        tr.extract_function_source("tests/test_slope_stability.py", "_helper")


def test_ast_live_detection_module_and_decorator_level():
    module_live = ast.parse(
        "import pytest\npytestmark = pytest.mark.live\ndef test_a():\n    pass\n"
    )
    assert tr._module_is_live(module_live) is True
    module_list = ast.parse("import pytest\npytestmark = [pytest.mark.slow, pytest.mark.live]\n")
    assert tr._module_is_live(module_list) is True
    plain = ast.parse("import pytest\npytestmark = pytest.mark.slow\ndef test_a():\n    pass\n")
    assert tr._module_is_live(plain) is False
    deco = ast.parse("@pytest.mark.live\ndef test_b():\n    pass\n").body[0]
    assert any(tr._is_live_marker(d) for d in deco.decorator_list)
    deco_call = ast.parse("@pytest.mark.live(reason='x')\ndef test_c():\n    pass\n").body[0]
    assert any(tr._is_live_marker(d) for d in deco_call.decorator_list)


def test_live_tests_are_refused_even_if_registered(monkeypatch: pytest.MonkeyPatch):
    live_file = "tests/test_nsta_live.py"
    original = registry.all_registered_test_files
    monkeypatch.setattr(
        tr.registry,
        "all_registered_test_files",
        lambda: (*original(), live_file),
    )
    functions = tr.discover_test_functions(live_file)
    assert functions and all(f.live for f in functions)
    with pytest.raises(tr.TestRunnerError, match="live"):
        tr.validate_node_ids([live_file])
    with pytest.raises(tr.TestRunnerError, match="live"):
        tr.validate_node_ids([functions[0].node_id])
    with pytest.raises(tr.TestRunnerError, match="live"):
        tr.build_pytest_command([functions[0].node_id])


def test_selected_pytest_command_construction():
    functions = tr.discover_test_functions("tests/test_slope_stability.py")
    node_id = functions[0].node_id
    command = tr.build_pytest_command([node_id])
    launcher = tr.pytest_launcher()
    assert command[: len(launcher)] == launcher
    assert launcher in (["uv", "run", "--frozen", "pytest"], [sys.executable, "-m", "pytest"])
    assert command[-1] == node_id
    assert "-m" in command and command[command.index("-m") + 1] == "not live"
    assert "-q" in command
    assert all(isinstance(part, str) for part in command)
    # Capability suite = every registered file of one capability, no function selection.
    cap = registry.hazard_by_id("slope_instability")
    suite = tr.build_capability_command(cap.tests)
    assert suite[-len(cap.tests) :] == list(cap.tests)
    # Full suite carries no node id and is a separate, explicit builder.
    full = tr.build_full_suite_command()
    assert not any(part.startswith("tests/") for part in full)
    assert "not live" in full
    with pytest.raises(tr.TestRunnerError):
        tr.build_pytest_command([])
    with pytest.raises(tr.TestRunnerError):
        tr.build_capability_command([])
    with pytest.raises(tr.TestRunnerError):
        tr.build_pytest_command(["tests/test_slope_stability.py::test_does_not_exist"])


def test_runner_never_uses_a_shell_and_refuses_foreign_commands():
    source = inspect.getsource(tr)
    assert "shell=True" not in source
    assert "os.system" not in source and "os.popen" not in source
    assert "shell=False" in source
    with pytest.raises(tr.TestRunnerError):
        tr.run_command(["git", "status"])
    with pytest.raises(tr.TestRunnerError):
        tr.run_command([])
    with pytest.raises(tr.TestRunnerError):
        tr.run_command(["cmd", "/c", "echo", "pwned"])


def test_run_command_executes_one_registered_test_with_captured_output():
    functions = tr.discover_test_functions("tests/test_imports.py")
    command = tr.build_pytest_command([functions[0].node_id])
    if command[0] == "uv" and shutil.which("uv") is None:  # pragma: no cover - environment guard
        pytest.skip("uv not on PATH")
    result = tr.run_command(command, timeout_s=600)
    assert result.passed, result.stderr or result.stdout
    assert result.returncode == 0 and result.timed_out is False
    assert "9 passed" in result.stdout and "failed" not in result.stdout
    assert result.elapsed_s > 0
    assert result.status_label == "PASSED"
    assert result.to_dict()["passed"] is True


def test_run_result_labels():
    failed = tr.RunResult(command=("x",), returncode=1, stdout="", stderr="", elapsed_s=0.1)
    assert failed.status_label == "FAILED (exit 1)" and failed.passed is False
    timed = tr.RunResult(
        command=("x",), returncode=None, stdout="", stderr="", elapsed_s=1.0, timed_out=True
    )
    assert timed.status_label == "TIMED_OUT" and timed.passed is False


# --- output inspector ----------------------------------------------------------------------------


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    (root / "processed").mkdir(parents=True)
    (root / "interim").mkdir(parents=True)
    monkeypatch.setattr(oi, "DATA_ROOT", root.resolve())
    monkeypatch.setattr(
        oi, "_ALLOWED_ROOTS", (root.resolve() / "processed", root.resolve() / "interim")
    )
    return root.resolve()


@pytest.mark.parametrize(
    "bad",
    [
        "../src/marine_engine/cli.py",
        "processed/../../pyproject.toml",
        "raw/anything.tif",
        "/etc/passwd",
        "C:\\Windows\\System32",
    ],
)
def test_output_path_containment_rejects_escapes(data_root: Path, bad: str):
    with pytest.raises(oi.OutputInspectorError):
        oi.contain_path(bad)
    with pytest.raises(oi.OutputInspectorError):
        oi.inspect_json(bad)
    with pytest.raises(oi.OutputInspectorError):
        oi.list_directory_outputs(bad)


def test_output_path_containment_accepts_data_roots(data_root: Path):
    assert oi.contain_path("processed/x/y.json") == data_root / "processed" / "x" / "y.json"
    assert oi.contain_path(data_root / "interim" / "a.parquet").name == "a.parquet"
    assert oi.contain_path("processed") == data_root / "processed"
    with pytest.raises(oi.OutputInspectorError):
        oi.contain_path(data_root)  # data/ itself is not an output root


def test_missing_outputs_are_reported_not_fabricated(data_root: Path):
    info = oi.inspect_json("processed/pl854/nothing.json")
    assert info["status"] == oi.NOT_PRESENT_LOCALLY and "data" not in info
    assert oi.inspect_parquet("processed/none.parquet")["status"] == oi.NOT_PRESENT_LOCALLY
    assert oi.inspect_gpkg("processed/none.gpkg")["status"] == oi.NOT_PRESENT_LOCALLY
    assert oi.inspect_geotiff("processed/none.tif")["status"] == oi.NOT_PRESENT_LOCALLY
    with pytest.raises(oi.OutputInspectorError):
        oi.parquet_head("processed/none.parquet")
    with pytest.raises(oi.OutputInspectorError):
        oi.geotiff_preview("processed/none.tif")
    entries = oi.discover_outputs(
        [registry.OutputPattern("Slope rasters", "processed/ghost/slope_stability/*.tif")]
    )
    assert len(entries) == 1
    assert entries[0].exists is False
    assert entries[0].status == oi.NOT_PRESENT_LOCALLY
    assert entries[0].evidence_kind == registry.NOT_AVAILABLE
    assert entries[0].byte_size is None
    assert oi.list_directory_outputs("processed/ghost") == []


def test_json_inspection_and_readiness_summary(data_root: Path):
    payload = {
        "status": "READY_WITH_LIMITATIONS",
        "cpt_evidence_readiness": {"status": "READY_WITH_LIMITATIONS"},
        "blocking_reasons": [],
        "limitation_reasons": ["qt not provided by source"],
        "nested": {"value": 1},
    }
    path = data_root / "processed" / "demo" / "readiness.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    info = oi.inspect_json(path)
    assert info["status"] == "PRESENT" and info["top_level_type"] == "dict"
    assert info["data"] == payload
    assert "status" in info["top_level_keys"]
    summary = oi.summarize_readiness(info["data"])
    assert summary["status"] == "READY_WITH_LIMITATIONS"
    assert summary["cpt_evidence_readiness.status"] == "READY_WITH_LIMITATIONS"
    assert summary["limitation_reasons"] == ["qt not provided by source"]
    assert "nested" not in summary and "score" not in json.dumps(summary).lower()
    assert oi.summarize_readiness([1, 2]) == {}
    with pytest.raises(oi.OutputInspectorError):
        oi.inspect_json(path, max_bytes=10)  # size guard, not loaded


def test_parquet_metadata_inspection_reads_no_rows(data_root: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(
        {"source_id": ["A"] * 6, "qc_mpa": [1.0, 2.0, None, 4.0, None, 6.0], "qt_mpa": [None] * 6}
    )
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"marine_engine_cpt_contract"] = b"CPT_CANONICAL_PROFILE_V1"
    path = data_root / "processed" / "demo" / "m.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(table.replace_schema_metadata(meta), path, row_group_size=3)
    info = oi.inspect_parquet(path)
    assert info["rows"] == 6 and info["row_groups"] == 2
    assert info["columns"] == ["source_id", "qc_mpa", "qt_mpa"]
    assert info["null_counts"] == {"source_id": 0, "qc_mpa": 2, "qt_mpa": 6}
    assert info["null_count_basis"]["qc_mpa"] == "row_group_statistics"
    assert info["null_count_basis"]["qt_mpa"] in ("row_group_statistics", "column_scan")
    assert info["marine_engine_metadata"] == {
        "marine_engine_cpt_contract": "CPT_CANONICAL_PROFILE_V1"
    }
    assert any(s.startswith("qc_mpa: double") for s in info["schema"])
    # Lazy: the metadata inspector never materializes a table or a DataFrame.
    source = inspect.getsource(oi.inspect_parquet)
    assert "to_pandas" not in source and "read_parquet" not in source
    assert "columns=" in source  # any fallback scan is column-restricted, never the whole table
    head = oi.parquet_head(path, rows=2)
    assert len(head) == 2 and list(head.columns) == ["source_id", "qc_mpa", "qt_mpa"]
    assert len(oi.parquet_head(path, rows=100_000)) == 6  # capped read, not the whole cap value
    # The file on disk is untouched by inspection.
    assert pq.read_metadata(path).num_rows == 6


def test_gpkg_metadata_inspection(data_root: Path):
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"test_id": ["T1", "T2", "T3"]},
        geometry=[Point(500000, 5900000), Point(500100, 5900100), Point(500200, 5900200)],
        crs="EPSG:32631",
    )
    path = data_root / "processed" / "demo" / "pts.gpkg"
    path.parent.mkdir(parents=True)
    gdf.to_file(path, layer="cpt_locations", driver="GPKG")
    before = path.read_bytes()
    info = oi.inspect_gpkg(path)
    assert info["status"] == "PRESENT" and len(info["layers"]) == 1
    layer = info["layers"][0]
    assert layer["layer"] == "cpt_locations" and layer["feature_count"] == 3
    assert "32631" in str(layer["crs"])
    assert "Point" in layer["geometry_type"]
    assert layer["bounds"] == pytest.approx([500000, 5900000, 500200, 5900200])
    assert "test_id" in layer["fields"]
    assert path.read_bytes() == before  # not rewritten


def test_geotiff_metadata_inspection_and_labelled_bounded_preview(data_root: Path):
    import rasterio
    from rasterio.transform import from_origin

    path = data_root / "processed" / "demo" / "slope.tif"
    path.parent.mkdir(parents=True)
    data = np.arange(2000 * 1500, dtype="float32").reshape(2000, 1500)
    data[:60, :60] = -9999.0  # a nodata block wide enough to survive 20x decimation
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2000,
        width=1500,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(400000, 5900000, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    before = path.read_bytes()
    info = oi.inspect_geotiff(path)
    assert info["crs"] == "EPSG:32631"
    assert (info["width"], info["height"], info["band_count"]) == (1500, 2000, 1)
    assert info["cell_count"] == 3_000_000
    assert info["resolution"] == [1.0, 1.0] and info["nodata"] == -9999.0
    assert info["transform"][2] == 400000 and info["transform"][5] == 5900000
    assert info["bounds"] == pytest.approx([400000, 5898000, 401500, 5900000])
    assert info["dtypes"] == ["float32"]
    # Header only: no pixel read in the metadata inspector.
    assert ".read(" not in inspect.getsource(oi.inspect_geotiff)
    preview = oi.geotiff_preview(path, max_pixels=100)
    assert preview["label"] == "DISPLAY PREVIEW -- NOT CANONICAL DATA"
    assert max(preview["preview_shape"]) <= 100
    assert preview["source_shape"] == [2000, 1500]
    assert preview["decimation_factor"] == pytest.approx(20.0)
    assert np.isnan(preview["array"][0, 0])  # nodata masked, never shown as a value
    assert preview["vmin"] is not None and preview["vmax"] is not None
    assert path.read_bytes() == before  # canonical raster untouched


def test_discover_outputs_and_evidence_kinds(data_root: Path):
    real = data_root / "processed" / "freespan_poc" / "maps" / "real.png"
    synthetic = (
        data_root / "processed" / "freespan_poc" / "synthetic" / "synthetic_free_span_map.png"
    )
    for p in (real, synthetic):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
    patterns = [
        registry.OutputPattern("maps", "processed/freespan_poc/maps/*.png"),
        registry.OutputPattern(
            "synthetic", "processed/freespan_poc/synthetic/*", registry.SYNTHETIC_TEST_FIXTURE
        ),
        registry.OutputPattern("absent", "processed/freespan_poc/none/*.gpkg"),
    ]
    entries = oi.discover_outputs(patterns)
    by_label = {e.label: e for e in entries}
    assert by_label["maps"].exists and by_label["maps"].evidence_kind == registry.CACHED_REAL_OUTPUT
    assert by_label["synthetic"].evidence_kind == registry.SYNTHETIC_TEST_FIXTURE
    assert (
        by_label["absent"].exists is False and by_label["absent"].status == oi.NOT_PRESENT_LOCALLY
    )
    assert by_label["maps"].kind == "image" and by_label["maps"].byte_size == 8
    # Directory listing + registry-declared evidence kinds (synthetic pattern from the registry).
    listed = [
        oi.with_registry_evidence_kind(e)
        for e in oi.list_directory_outputs("processed/freespan_poc")
    ]
    kinds = {e.relative_path: e.evidence_kind for e in listed}
    assert kinds["processed/freespan_poc/synthetic/synthetic_free_span_map.png"] == (
        registry.SYNTHETIC_TEST_FIXTURE
    )
    assert kinds["processed/freespan_poc/maps/real.png"] == registry.CACHED_REAL_OUTPUT
    assert oi.classify_kind("x.TIF") == "geotiff" and oi.classify_kind("x.bin") == "other"


def test_large_file_flag_threshold():
    entry = oi.OutputEntry(
        "big", "processed/x.parquet", True, "parquet", oi.LARGE_FILE_BYTES + 1, "CACHED_REAL_OUTPUT"
    )
    small = oi.OutputEntry(
        "small", "processed/y.parquet", True, "parquet", 10, "CACHED_REAL_OUTPUT"
    )
    assert entry.is_large is True and small.is_large is False
    assert entry.to_dict()["status"] == "PRESENT"


def test_registered_output_patterns_resolve_against_real_local_data_without_error():
    # Uses the real data/ tree if present; absent outputs must yield NOT PRESENT LOCALLY,
    # never raise.
    for cap in registry.HAZARDS:
        entries = oi.discover_outputs(cap.output_patterns)
        assert len(entries) >= len(cap.output_patterns)
        for e in entries:
            assert e.status in ("PRESENT", oi.NOT_PRESENT_LOCALLY)
            assert e.relative_path.startswith(("processed/", "interim/"))
    for project in registry.PROJECTS:
        for d in project.processed_dirs:
            oi.list_directory_outputs(f"processed/{d}")


# --- repository state and engine isolation ------------------------------------------------------


def test_repo_state_is_read_only_git():
    source = inspect.getsource(repo_state)
    for forbidden in (
        '"push"',
        '"fetch"',
        '"pull"',
        '"commit"',
        '"checkout"',
        '"reset"',
    ):
        assert forbidden not in source
    assert "shell=True," not in source and "shell=False" in source
    state = repo_state.read_repository_state()
    if state.error:  # pragma: no cover - environment without git
        pytest.skip(state.error)
    assert re.fullmatch(r"[0-9a-f]{40}", state.head_sha or "")
    assert state.branch
    assert state.status_label in ("CLEAN", "DIRTY")
    assert state.to_dict()["head_sha"] == state.head_sha


def test_ui_never_imports_or_mutates_the_scientific_engine():
    ui_dir = REPO_ROOT / "ui"
    for py in sorted(ui_dir.glob("*.py")):
        source = py.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "marine_engine" for a in node.names), py.name
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "marine_engine", py.name
        for token in (
            ".to_parquet(",
            ".to_file(",
            "pq.write",
            ".write_text(",
            ".write_bytes(",
            "shutil.rmtree",
            "os.remove",
            ".unlink(",
        ):
            assert token not in source, (py.name, token)
        assert "shell=True," not in source, py.name
        assert "requests." not in source and "urllib" not in source, py.name  # no downloads
    # Pure helper modules do not depend on Streamlit.
    for module in UI_MODULES:
        assert "streamlit" not in inspect.getsource(module)


# --- UI-002 guided view: story data (ui/guided_story.py) ----------------------------------------


def test_every_project_has_exactly_one_guided_story_in_registry_order():
    assert [story.project_id for story in gs.STORIES] == [p.id for p in registry.PROJECTS]


def test_pl854_story_stage_order():
    story = gs.story_for_project("pl854")
    assert [stage.label for stage in story.stages] == [
        "Route & Area",
        "Bathymetry",
        "Currents & Waves",
        "Bed Shear",
        "Sediment Mobility",
        "Scour",
        "Burial / Exposure",
        "Free Span",
        "Transport Intensity",
        "Evidence Summary",
    ]


def test_sheringham_2020_story_stage_order():
    story = gs.story_for_project("sheringham_shoal_2020")
    assert [stage.label for stage in story.stages] == [
        "Survey Bathymetry",
        "Terrain",
        "Bedforms",
        "Multi-epoch Change",
        "Slope Screening",
    ]


def test_cptu_story_stage_order():
    story = gs.story_for_project("sheringham_shoal_2008_cptu")
    assert [stage.label for stage in story.stages] == [
        "Investigation Area",
        "CPT Locations",
        "CPT Evidence",
        "Profile Availability",
        "Liquefaction Foundation",
    ]


def test_barrow_story_stage_order():
    story = gs.story_for_project("barrow_2016")
    assert [stage.label for stage in story.stages] == [
        "Route / Asset",
        "Burial Profile",
        "Observed Burial / Exposure State",
    ]


_JARGON_PATTERN = re.compile(
    r"MAR-\d|QUALIFIED_POC|FOUNDATION_READY|UNDER_CONSTRUCTION|\.png|\.gpkg|\.parquet|\.tif"
)


def test_guided_text_has_no_ticket_numbers_filenames_or_registry_constants():
    for story in gs.STORIES:
        for stage in story.stages:
            for text in (
                stage.what_you_are_looking_at,
                stage.what_it_means,
                stage.what_it_does_not_mean,
            ):
                assert not _JARGON_PATTERN.search(text), (story.project_id, stage.key, text)


def test_guided_text_blocks_stay_short():
    for story in gs.STORIES:
        for stage in story.stages:
            for text in (
                stage.what_you_are_looking_at,
                stage.what_it_means,
                stage.what_it_does_not_mean,
            ):
                assert text.count(". ") <= 2, (story.project_id, stage.key, text)
                assert len(text) <= 260, (story.project_id, stage.key, text)


def test_visual_candidate_rejects_traversal_globs():
    with pytest.raises(ValueError):
        gs.VisualCandidate(kind="png", label="x", glob="../escape.png")
    with pytest.raises(ValueError):
        gs.VisualCandidate(kind="png", label="x", glob="/etc/passwd")
    with pytest.raises(ValueError):
        gs.VisualCandidate(kind="vector", label="x", layers=(("../escape.gpkg", None, "x"),))
    with pytest.raises(ValueError):
        gs.VisualCandidate(kind="raster", label="x", raster_glob="../escape.tif")


def test_visual_candidate_rejects_unknown_kind():
    with pytest.raises(ValueError):
        gs.VisualCandidate(kind="video", label="x")


def test_stage_badge_maturity_is_a_strict_mapping_of_the_four_real_values():
    for story in gs.STORIES:
        for stage in story.stages:
            badge = gs.stage_badge_maturity(stage)
            if stage.capability_id is None:
                assert badge is None
            else:
                assert badge in gs.BADGE_VOCABULARY
    assert set(gs._BADGE_BY_MATURITY) == set(registry.MATURITY_VOCABULARY)


def test_pl854_scour_stage_never_uses_the_not_available_sentinel_layer():
    story = gs.story_for_project("pl854")
    scour = next(stage for stage in story.stages if stage.key == "scour")
    layer_names = [
        layer[1]
        for candidate in (*scour.visuals, *scour.optional_layers)
        for layer in candidate.layers
    ]
    assert "site_specific_susceptibility_screening" not in layer_names
    assert "tested_embedment_scenario_envelope" in layer_names
    assert any(
        "SITE_SPECIFIC_SCOUR_SUSCEPTIBILITY_NOT_AVAILABLE" in limitation
        for limitation in scour.limitations
    )


def test_pl854_burial_stage_is_honestly_empty_not_fabricated():
    story = gs.story_for_project("pl854")
    burial = next(stage for stage in story.stages if stage.key == "burial_exposure")
    assert burial.visuals == ()
    assert burial.not_applicable_note


# --- UI-002 guided view: rendering (ui/map_view.py) ---------------------------------------------


def test_map_view_never_writes_or_reprojects_in_place():
    source = inspect.getsource(mv)
    assert ".to_file(" not in source
    assert ".to_crs(" in source  # copy-reprojection only, never in place


def test_render_raster_preview_delegates_to_bounded_geotiff_preview(data_root: Path):
    import rasterio
    from rasterio.transform import from_origin

    path = data_root / "processed" / "demo" / "big.tif"
    path.parent.mkdir(parents=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2000,
        width=1500,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(400000, 5900000, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(np.zeros((2000, 1500), dtype="float32"), 1)
    fig, meta = mv.render_raster_preview("processed/demo/big.tif")
    assert fig is not None
    assert meta["label"] == oi.PREVIEW_LABEL
    assert max(meta["preview_shape"]) <= 512


def test_render_raster_preview_missing_file_is_reported_not_fabricated(data_root: Path):
    fig, meta = mv.render_raster_preview("processed/demo/nope.tif")
    assert fig is None
    assert meta["present"] is False


def test_render_vector_scene_reports_missing_layers_not_fabricated(data_root: Path):
    fig, meta = mv.render_vector_scene([("processed/demo/nope.gpkg", None, "Ghost")])
    assert fig is None
    assert meta["missing_labels"] == ["Ghost"]
    assert meta["present_labels"] == []


def test_render_vector_scene_reprojects_a_copy_never_the_source(data_root: Path):
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    a = gpd.GeoDataFrame({"id": [1]}, geometry=[LineString([(0, 0), (100, 100)])], crs="EPSG:32631")
    b = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(500000, 5900000)], crs="EPSG:32632")
    path_a = data_root / "processed" / "demo" / "a.gpkg"
    path_b = data_root / "processed" / "demo" / "b.gpkg"
    path_a.parent.mkdir(parents=True)
    a.to_file(path_a, driver="GPKG")
    b.to_file(path_b, driver="GPKG")
    before_a, before_b = path_a.read_bytes(), path_b.read_bytes()
    fig, meta = mv.render_vector_scene(
        [("processed/demo/a.gpkg", None, "A"), ("processed/demo/b.gpkg", None, "B")]
    )
    assert fig is not None
    assert meta["present_labels"] == ["A", "B"]
    assert meta["crs"] == "EPSG:32631"
    assert path_a.read_bytes() == before_a
    assert path_b.read_bytes() == before_b


def test_render_vector_scene_thins_large_point_layers_without_fabricating_points(data_root: Path):
    import geopandas as gpd
    from shapely.geometry import Point

    points = gpd.GeoDataFrame(
        {"id": list(range(500))}, geometry=[Point(x, 0) for x in range(500)], crs="EPSG:32631"
    )
    path = data_root / "processed" / "demo" / "pts.gpkg"
    path.parent.mkdir(parents=True)
    points.to_file(path, driver="GPKG")
    fig, meta = mv.render_vector_scene(
        [("processed/demo/pts.gpkg", None, "Points")], max_points_per_layer=50
    )
    assert fig is not None
    assert meta["present_labels"] == ["Points"]


def test_cpt_channel_profile_uses_measured_channels_only_and_qt_stays_unavailable():
    profile = mv.cpt_channel_profile("CPT-A2")
    if not profile["present"]:
        pytest.skip("real CPT parquet not present locally")
    assert profile["channels"]["qt_mpa"]["available"] is False
    assert profile["channels"]["qt_mpa"]["values"] is None
    for channel in ("qc_mpa", "fs_kpa", "u2_kpa"):
        assert profile["channels"][channel]["available"] is True
        assert profile["channels"][channel]["values"] is not None


def test_cpt_profile_unknown_test_id_yields_no_fabricated_plot():
    real = mv.resolve_first_existing([mv._CPT_MEASUREMENTS_GLOB])
    if real is None:
        pytest.skip("real CPT parquet not present locally")
    fig, _meta = mv.render_cpt_profile("NOT-A-REAL-TEST-ID")
    assert fig is None


def test_render_cpt_profile_never_plots_an_unavailable_channel():
    fig, meta = mv.render_cpt_profile("CPT-A2")
    if fig is None:
        pytest.skip("real CPT parquet not present locally")
    assert meta["unavailable_channels"] == ["qt_mpa"]
    axis_labels = [ax.get_xlabel() for ax in fig.axes]
    assert not any(label.startswith("qt") for label in axis_labels)
    assert len(fig.axes) == 3  # qc, fs, u2 only -- never a shared MPa/kPa axis


def test_no_liquefaction_calculation_in_guided_or_map_view_modules():
    for module in (gs, mv):
        source = inspect.getsource(module)
        for banned in ("CRR =", "CSR =", "= crr", "= csr", "liquefaction_factor_of_safety ="):
            assert banned not in source, (module.__name__, banned)


def test_render_burial_profile_never_asserts_a_burial_depth_label():
    assert "source_burial_semantics" in inspect.getsource(mv.render_burial_profile)
    fig, meta = mv.render_burial_profile()
    if fig is None:
        pytest.skip("real Barrow burial parquet not present locally")
    ylabel = fig.axes[0].get_ylabel()
    assert "burial depth" not in ylabel.lower()  # the actual rendered axis label, not prose
    assert isinstance(meta["caveat"], str) and meta["caveat"]


def test_cpt_locations_frame_uses_geometry_not_raw_declared_fields():
    gdf = mv.cpt_locations_frame()
    if gdf is None:
        pytest.skip("real CPT locations gpkg not present locally")
    assert {"easting_m", "northing_m", "test_id"} <= set(gdf.columns)
    assert len(gdf) == 100
    # Easting/northing are derived from the canonical geometry -- the source's own declared
    # position_x_raw/position_y_raw columns pass through untouched alongside them (kept
    # separate, never merged into or overwritten by the computed canonical value).
    assert gdf["easting_m"].iloc[0] == pytest.approx(gdf.geometry.iloc[0].x)
    assert gdf["northing_m"].iloc[0] == pytest.approx(gdf.geometry.iloc[0].y)


def test_resolve_layer_state_never_allows_an_unavailable_layer_to_be_checked():
    assert mv.resolve_layer_state(available=False, default_on=True, requested=True) is False
    assert mv.resolve_layer_state(available=False, default_on=False, requested=None) is False
    assert mv.resolve_layer_state(available=True, default_on=True, requested=None) is True
    assert mv.resolve_layer_state(available=True, default_on=False, requested=True) is True
    assert mv.resolve_layer_state(available=True, default_on=True, requested=False) is False


# --- UI-002 guided view: app.py (source-text only -- top-level Streamlit calls execute on
# import, so this file is never imported by tests, matching the app.py/components.py convention)


def test_app_py_defaults_to_guided_mode():
    source = (REPO_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert 'st.session_state.setdefault("ui_mode", "guided")' in source
    assert "mode = c.mode_toggle()" in source


def test_app_py_preserves_all_five_engineering_pages_and_both_view_functions():
    source = (REPO_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    for page in ("Overview", "Capability Explorer", "Test Lab", "Data & Outputs", "Roadmap"):
        assert f'"{page}"' in source
    assert "def render_engineering_view" in source
    assert "def render_guided_view" in source

"""Allowlisted pytest discovery and safe execution for the Test Lab (UI-001 Sections 13-16).

Safety rules enforced here, not left to the UI:

* only test files explicitly registered to a capability (`capability_registry`) may be inspected or
  run; the allowlist is the registry, never the filesystem;
* node ids are validated against a strict pattern and against the functions actually discovered
  with `ast`, so no user-typed string ever reaches a command line;
* commands are argument arrays executed with `shell=False`; there is no shell, no string
  concatenation and no free-form command box;
* live / network tests are refused twice: a live-marked function or module is never selectable, and
  every command carries `-m "not live"`;
* the full suite is never run implicitly -- a separate, explicitly named builder exists for it.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ui import REPO_ROOT
from ui import capability_registry as registry

__all__ = [
    "TestRunnerError",
    "TestFunction",
    "RunResult",
    "LIVE_MARKER",
    "NODE_ID_PATTERN",
    "DEFAULT_TIMEOUT_S",
    "allowlisted_test_files",
    "resolve_test_path",
    "discover_test_functions",
    "extract_function_source",
    "validate_node_ids",
    "pytest_launcher",
    "build_pytest_command",
    "build_capability_command",
    "build_full_suite_command",
    "run_command",
]

LIVE_MARKER = "live"
DEFAULT_TIMEOUT_S = 900
FULL_SUITE_TIMEOUT_S = 3600
NODE_ID_PATTERN = re.compile(r"^tests/[A-Za-z0-9_]+\.py(::[A-Za-z_][A-Za-z0-9_]*)?$")
_TESTS_DIR = (REPO_ROOT / "tests").resolve()


class TestRunnerError(ValueError):
    """A test path / node id is not allowlisted, does not exist, is live-marked or is malformed."""


@dataclass(frozen=True)
class TestFunction:
    file: str  # repo-relative posix path, e.g. tests/test_slope_stability.py
    name: str
    lineno: int
    end_lineno: int
    live: bool
    parametrized: bool = False

    @property
    def node_id(self) -> str:
        return f"{self.file}::{self.name}"


@dataclass(frozen=True)
class RunResult:
    command: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False
    started_at: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def status_label(self) -> str:
        if self.timed_out:
            return "TIMED_OUT"
        if self.returncode == 0:
            return "PASSED"
        return f"FAILED (exit {self.returncode})"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status_label": self.status_label, "passed": self.passed}


def allowlisted_test_files() -> tuple[str, ...]:
    """The allowlist IS the capability registry (hazards + supporting capabilities)."""

    return registry.all_registered_test_files()


def resolve_test_path(test_file: str) -> Path:
    """Return the absolute path of an allowlisted test file, or raise `TestRunnerError`.

    Rejects anything not in the allowlist (which also rejects `..`, absolute paths, backslashes,
    shell metacharacters and files outside `tests/`), then re-checks containment of the resolved
    path inside the repository `tests/` directory."""

    if not isinstance(test_file, str) or test_file not in allowlisted_test_files():
        raise TestRunnerError(f"test file is not registered to any capability: {test_file!r}")
    if not NODE_ID_PATTERN.match(test_file):
        raise TestRunnerError(f"malformed test path: {test_file!r}")
    resolved = (REPO_ROOT / test_file).resolve()
    if _TESTS_DIR not in resolved.parents:
        raise TestRunnerError(f"test path escapes the repository tests directory: {test_file!r}")
    if not resolved.is_file():
        raise TestRunnerError(f"registered test file is missing locally: {test_file}")
    return resolved


def _is_live_marker(node: ast.AST) -> bool:
    """True for `pytest.mark.live` or `pytest.mark.live(...)` expressions."""

    target = node.func if isinstance(node, ast.Call) else node
    return (
        isinstance(target, ast.Attribute)
        and target.attr == LIVE_MARKER
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "mark"
    )


def _module_is_live(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            value = node.value
            candidates = value.elts if isinstance(value, ast.List | ast.Tuple) else [value]
            if any(_is_live_marker(c) for c in candidates):
                return True
    return False


def _is_parametrized(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr == "parametrize":
            return True
    return False


def discover_test_functions(test_file: str) -> tuple[TestFunction, ...]:
    """Parse the allowlisted file with `ast` and list module-level `test_*` functions."""

    path = resolve_test_path(test_file)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_live = _module_is_live(tree)
    functions: list[TestFunction] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "test_"
        ):
            live = module_live or any(_is_live_marker(d) for d in node.decorator_list)
            functions.append(
                TestFunction(
                    file=test_file,
                    name=node.name,
                    lineno=node.lineno,
                    end_lineno=node.end_lineno or node.lineno,
                    live=live,
                    parametrized=_is_parametrized(node),
                )
            )
    return tuple(functions)


def extract_function_source(test_file: str, function_name: str) -> str:
    """Return ONLY the selected test function's source (decorators included), not the whole file."""

    path = resolve_test_path(test_file)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == function_name
            and node.name.startswith("test_")
        ):
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            end = node.end_lineno or node.lineno
            lines = source.splitlines()
            return "\n".join(lines[start - 1 : end])
    raise TestRunnerError(f"no test function {function_name!r} in {test_file}")


def validate_node_ids(node_ids: Sequence[str]) -> tuple[str, ...]:
    """Every node id must be `tests/<allowlisted>.py` or `tests/<allowlisted>.py::test_<name>` where
    the function was discovered by `ast` and is not live-marked."""

    if not node_ids:
        raise TestRunnerError("no test selected")
    validated: list[str] = []
    for node_id in node_ids:
        if not isinstance(node_id, str) or not NODE_ID_PATTERN.match(node_id):
            raise TestRunnerError(f"malformed pytest node id: {node_id!r}")
        file_part, _, function_part = node_id.partition("::")
        functions = discover_test_functions(file_part)  # validates + resolves the file
        if function_part:
            match = next((f for f in functions if f.name == function_part), None)
            if match is None:
                raise TestRunnerError(f"function not discovered in {file_part}: {function_part!r}")
            if match.live:
                raise TestRunnerError(f"live/network test refused: {node_id}")
        elif functions and all(f.live for f in functions):
            raise TestRunnerError(f"live/network test module refused: {file_part}")
        validated.append(node_id)
    return tuple(validated)


def pytest_launcher() -> list[str]:
    """`uv run --frozen pytest` when uv is on PATH, else the current interpreter's pytest. Both are
    fixed argument arrays; neither is influenced by user input."""

    if shutil.which("uv"):
        return ["uv", "run", "--frozen", "pytest"]
    return [sys.executable, "-m", "pytest"]


def _base_command() -> list[str]:
    return [
        *pytest_launcher(),
        "-q",
        "--color=no",
        "-m",
        f"not {LIVE_MARKER}",
        "-p",
        "no:cacheprovider",
    ]


def build_pytest_command(node_ids: Sequence[str]) -> list[str]:
    return [*_base_command(), *validate_node_ids(node_ids)]


def build_capability_command(test_files: Sequence[str]) -> list[str]:
    """Run every registered (non-live) test file of one capability."""

    if not test_files:
        raise TestRunnerError("capability has no registered tests")
    return build_pytest_command(list(test_files))


def build_full_suite_command() -> list[str]:
    """Advanced only: the whole offline suite. Never triggered implicitly by the UI."""

    return _base_command()


def run_command(command: Sequence[str], timeout_s: int = DEFAULT_TIMEOUT_S) -> RunResult:
    """Execute a command built by this module: argument array, no shell, bounded timeout."""

    argv = [str(part) for part in command]
    if not argv:
        raise TestRunnerError("empty command")
    if argv[: len(pytest_launcher())] != pytest_launcher():
        raise TestRunnerError("only pytest commands built by this module may be executed")
    started = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 -- argv built from the internal allowlist only
            argv,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            command=tuple(argv),
            returncode=None,
            stdout=(exc.stdout or b"").decode()
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or ""),
            stderr=(exc.stderr or b"").decode()
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or ""),
            elapsed_s=time.monotonic() - started,
            timed_out=True,
        )
    return RunResult(
        command=tuple(argv),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_s=time.monotonic() - started,
    )

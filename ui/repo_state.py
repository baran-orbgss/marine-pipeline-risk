"""Read-only local repository facts for the workbench sidebar (UI-001 Section 22).

Only `git` read commands are ever executed, always as argument arrays (never through a shell), and
never fetch / pull / push or any GitHub write action. Any failure degrades to `None` fields so the
UI keeps rendering without inventing a SHA."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ui import REPO_ROOT

__all__ = ["RepositoryState", "read_repository_state"]

_GIT_TIMEOUT_S = 15


@dataclass(frozen=True)
class RepositoryState:
    branch: str | None
    head_sha: str | None
    dirty: bool | None
    changed_files: tuple[str, ...]
    error: str | None = None

    @property
    def status_label(self) -> str:
        if self.dirty is None:
            return "UNKNOWN"
        return "DIRTY" if self.dirty else "CLEAN"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status_label": self.status_label}


def _git(args: list[str], repo_root: Path) -> str:
    """Run one read-only git command as an argument array and return stripped stdout."""

    completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell, read-only git subcommands
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
        check=True,
        shell=False,
    )
    return completed.stdout.strip()


def read_repository_state(repo_root: Path = REPO_ROOT) -> RepositoryState:
    try:
        branch = _git(["branch", "--show-current"], repo_root) or None
        head = _git(["rev-parse", "HEAD"], repo_root) or None
        porcelain = _git(["status", "--porcelain"], repo_root)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return RepositoryState(
            branch=None, head_sha=None, dirty=None, changed_files=(), error=str(exc)
        )
    changed = tuple(line[3:] for line in porcelain.splitlines() if line.strip())
    return RepositoryState(branch=branch, head_sha=head, dirty=bool(changed), changed_files=changed)

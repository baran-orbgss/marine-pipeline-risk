"""Path-containment helpers.

Every filesystem path built from a client-supplied identifier (project id, layer id, staging
session id, filename) must be constructed through `safe_join`, never by naive string
concatenation. This is the single mechanism that keeps upload/staging/layer-serving endpoints from
ever reading or writing outside their intended base directory.
"""

from __future__ import annotations

from pathlib import Path


class UnsafePathError(ValueError):
    """Raised when a client-supplied path segment would escape its base directory."""


def _validate_segment(part: str) -> None:
    if not part:
        raise UnsafePathError("empty path segment")
    if "\x00" in part:
        raise UnsafePathError("null byte in path segment")
    if part in (".", ".."):
        raise UnsafePathError(f"disallowed path segment: {part!r}")
    normalized = part.replace("\\", "/")
    if "/" in normalized:
        raise UnsafePathError(f"path segment must not contain a separator: {part!r}")
    if normalized.startswith("~"):
        raise UnsafePathError(f"disallowed path segment: {part!r}")
    if len(normalized) >= 2 and normalized[1] == ":":
        raise UnsafePathError(f"disallowed path segment: {part!r}")


def safe_join(base: Path, *parts: str) -> Path:
    """Join `parts` onto `base`, raising `UnsafePathError` if the result would escape it."""
    resolved_base = base.resolve()
    candidate = resolved_base
    for part in parts:
        _validate_segment(part)
        candidate = candidate / part
    candidate = candidate.resolve()
    if candidate != resolved_base and not candidate.is_relative_to(resolved_base):
        raise UnsafePathError(f"path escapes base: {candidate} not under {resolved_base}")
    return candidate


def safe_filename(filename: str) -> str:
    """Strip any directory component and validate what remains is a bare filename."""
    name = Path(filename).name
    _validate_segment(name)
    return name


def repo_relative(path: Path, repo_root: Path) -> str:
    """POSIX-style path relative to the repo root, for provenance display only (never used to
    resolve a path back on the client side)."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)

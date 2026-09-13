"""Tests for api.paths: every client-supplied path segment must be contained."""

from __future__ import annotations

from pathlib import Path

import pytest
from api.paths import UnsafePathError, safe_filename, safe_join


def test_safe_join_allows_normal_segments(tmp_path: Path) -> None:
    result = safe_join(tmp_path, "session-123", "file.tif")
    assert result == (tmp_path / "session-123" / "file.tif").resolve()


@pytest.mark.parametrize(
    "segment",
    [
        "..",
        ".",
        "../escape",
        "a/b",
        "a\\b",
        "C:\\Windows",
        "~root",
        "\x00null",
        "",
    ],
)
def test_safe_join_rejects_traversal_and_absolute_segments(tmp_path: Path, segment: str) -> None:
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, segment)


def test_safe_join_rejects_multi_segment_traversal(tmp_path: Path) -> None:
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "session", "..", "..", "etc", "passwd")


def test_safe_filename_strips_directory_components() -> None:
    assert safe_filename("bathymetry.tif") == "bathymetry.tif"
    assert safe_filename("../../etc/passwd") == "passwd"


def test_backslash_path_never_escapes_the_base_directory(tmp_path: Path) -> None:
    # `Path(...).name` splits on "/" identically on every platform, but only splits on "\\" on
    # Windows -- on POSIX (the CI runner) a backslash is just an ordinary filename character. So
    # this asserts the actual safety property (never escapes `tmp_path`), not one specific
    # platform's stripped filename: `safe_join` must either resolve safely inside `tmp_path`, or
    # raise, and never produce a path outside it.
    candidate = "C:\\Windows\\evil.tif"
    try:
        name = safe_filename(candidate)
        resolved = safe_join(tmp_path, name)
    except UnsafePathError:
        return
    assert resolved.is_relative_to(tmp_path.resolve())


def test_safe_filename_rejects_dot_only_names() -> None:
    with pytest.raises(UnsafePathError):
        safe_filename("..")

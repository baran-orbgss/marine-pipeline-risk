"""Immutable, content-based source-file identity (MAR-026 Section 8).

Zero dependency on any specific project or dataset. Source files are evidence: this module
only ever opens a file for READING (to hash it), never writes to, moves, or normalizes the
supplied source file. Identity is content-based (SHA-256), never filename-based -- two files
with the same name but different bytes get different identities, and a byte-for-byte-identical
file under a different name gets the same identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB read chunks -- avoids loading large rasters fully into memory


@dataclass(frozen=True)
class FileIdentity:
    resolved_path: Path
    filename: str
    byte_size: int
    sha256: str


def compute_file_identity(path: Path) -> FileIdentity:
    """Section 8: canonical resolved path, filename, byte size, and SHA-256 -- computed by
    reading the file exactly once, never modifying it. Raises `FileNotFoundError`/`OSError` if
    the path does not exist or cannot be read; callers decide how to report that as a
    registration failure (Section 16.A's "missing file" proof point)."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"asset source file not found or not a regular file: {resolved}")

    digest = hashlib.sha256()
    with resolved.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)

    return FileIdentity(
        resolved_path=resolved,
        filename=resolved.name,
        byte_size=resolved.stat().st_size,
        sha256=digest.hexdigest(),
    )

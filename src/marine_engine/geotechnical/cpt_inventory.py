"""Recursive, deterministic inventory of an acquired CPT source package (MAR-032 Section 9).

For every file: relative path, name, extension, byte size, SHA-256, a content type detected from
the real leading bytes (never from the extension alone) and a candidate role. Role assignment is
evidence-based:

* a file is `MACHINE_READABLE_CPT_PROFILE` ONLY if a caller-supplied probe confirms, from the
  file's own bytes, that it parses as a structured numeric CPT record (the probe is the
  source-specific reader's own strict format check -- never a filename pattern);
* `METADATA` requires the bytes to be an XML document carrying recognisable metadata markers;
* PDF / PNG / JPEG / TIFF content is DOCUMENTARY (`is_documentary = True`). It is never a numeric
  source (Section 10/30). Whether such a document is a factual report or a rendered CPT log is not
  inferred from the file name: the caller may state a role for documentary files when the source
  package itself declares one (e.g. the publisher labels the package "Reports"); otherwise it is
  `UNKNOWN` -- still documentary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from marine_engine.geotechnical import cpt_contract as contract

__all__ = [
    "INVENTORY_COLUMNS",
    "sniff_content_type",
    "inventory_package",
    "documentary_only",
    "machine_readable_files",
]

INVENTORY_COLUMNS = (
    "package",
    "relative_path",
    "file_name",
    "extension",
    "size_bytes",
    "sha256",
    "detected_content_type",
    "is_documentary",
    "candidate_role",
)

_HEAD_BYTES = 4096
_XML_METADATA_MARKERS = (b"MD_Metadata", b"xman_metadata", b"gmd:", b"<metadata")


def sniff_content_type(head: bytes) -> str:
    """Magic-byte content detection on the real leading bytes of a file."""

    if not head:
        return contract.CONTENT_EMPTY
    if head.startswith(b"%PDF"):
        return contract.CONTENT_PDF
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return contract.CONTENT_PNG
    if head.startswith(b"\xff\xd8\xff"):
        return contract.CONTENT_JPEG
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return contract.CONTENT_TIFF
    if head.startswith(b"PK\x03\x04"):
        return contract.CONTENT_ZIP
    stripped = head.lstrip(b"\xef\xbb\xbf").lstrip()
    if stripped.startswith(b"<?xml") or stripped.startswith(b"<"):
        return contract.CONTENT_XML
    if _looks_like_text(head):
        return contract.CONTENT_TEXT
    return contract.CONTENT_BINARY_UNKNOWN


def _looks_like_text(head: bytes) -> bool:
    if b"\x00" in head:
        return False
    # Latin-1 decodes any byte; require the bulk to be printable/whitespace so binary blobs that
    # merely lack NUL bytes are not mistaken for text.
    printable = sum(1 for b in head if 32 <= b < 127 or b in (9, 10, 13) or b >= 160)
    return printable / len(head) > 0.95


def _is_metadata_xml(head: bytes) -> bool:
    return any(marker in head for marker in _XML_METADATA_MARKERS)


def inventory_package(
    extracted_dir: Path,
    *,
    package: str,
    profile_probe: Callable[[Path, bytes], bool] | None = None,
    location_probe: Callable[[Path, bytes], bool] | None = None,
    documentary_role: str | None = None,
) -> pd.DataFrame:
    """Walk `extracted_dir` recursively and describe every regular file, sorted by POSIX
    relative path so the inventory is byte-for-byte deterministic for identical input.

    `profile_probe(path, head)` / `location_probe(path, head)` are source-specific, strict,
    content-based checks. `documentary_role`, when given, must be a `CANDIDATE_ROLES` member and
    is applied ONLY to files whose detected content is documentary (PDF/image)."""

    if documentary_role is not None and documentary_role not in contract.CANDIDATE_ROLES:
        raise ValueError(f"documentary_role {documentary_role!r} is not a known candidate role")
    if not extracted_dir.is_dir():
        raise FileNotFoundError(f"extracted package directory not found: {extracted_dir}")

    files = sorted(
        (p for p in extracted_dir.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(extracted_dir).as_posix(),
    )
    rows: list[dict[str, object]] = []
    for path in files:
        data = path.read_bytes()
        head = data[:_HEAD_BYTES]
        content_type = sniff_content_type(head)
        is_documentary = content_type in contract.DOCUMENTARY_CONTENT_TYPES
        role = contract.ROLE_UNKNOWN
        if is_documentary:
            if documentary_role is not None:
                role = documentary_role
        elif content_type == contract.CONTENT_XML and _is_metadata_xml(head):
            role = contract.ROLE_METADATA
        elif content_type == contract.CONTENT_TEXT:
            if profile_probe is not None and profile_probe(path, head):
                role = contract.ROLE_MACHINE_READABLE_CPT_PROFILE
            elif location_probe is not None and location_probe(path, head):
                role = contract.ROLE_CPT_LOCATION_DATA
        rows.append(
            {
                "package": package,
                "relative_path": path.relative_to(extracted_dir).as_posix(),
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "detected_content_type": content_type,
                "is_documentary": is_documentary,
                "candidate_role": role,
            }
        )
    return pd.DataFrame(rows, columns=list(INVENTORY_COLUMNS))


def documentary_only(inventory: pd.DataFrame) -> bool:
    """True when the package holds documentary evidence but NO machine-readable profile -- the
    honest `DOCUMENTARY_CPT_EVIDENCE_AVAILABLE = true / DIGITAL_NUMERIC_CPT_PROFILE_READY = false`
    outcome of Section 10."""

    if inventory.empty:
        return False
    has_documentary = bool(inventory["is_documentary"].any())
    has_profile = bool(
        (inventory["candidate_role"] == contract.ROLE_MACHINE_READABLE_CPT_PROFILE).any()
    )
    return has_documentary and not has_profile


def machine_readable_files(inventory: pd.DataFrame) -> list[str]:
    mask = inventory["candidate_role"] == contract.ROLE_MACHINE_READABLE_CPT_PROFILE
    return sorted(inventory.loc[mask, "relative_path"].tolist())

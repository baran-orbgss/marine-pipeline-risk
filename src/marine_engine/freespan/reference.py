"""Generic pipe vertical reference normalization (MAR-025 Section 3).

Zero dependency on any specific project or dataset. A raw operator pipe
elevation is meaningless on its own -- the caller must state which of the
four supported reference semantics applies (never inferred), and this
module normalizes it into exactly one canonical quantity,
`pipe_bottom_elevation_m`, exactly once.
"""

from __future__ import annotations

# --- Section 3: pipe vertical reference vocabulary -- never inferred -------------------------

PIPE_CENTRELINE_ELEVATION = "PIPE_CENTRELINE_ELEVATION"
PIPE_BOTTOM_ELEVATION = "PIPE_BOTTOM_ELEVATION"
PIPE_TOP_ELEVATION = "PIPE_TOP_ELEVATION"
OTHER_EXPLICIT_PIPE_REFERENCE = "OTHER_EXPLICIT_PIPE_REFERENCE"

PIPE_VERTICAL_REFERENCES = frozenset(
    {
        PIPE_CENTRELINE_ELEVATION,
        PIPE_BOTTOM_ELEVATION,
        PIPE_TOP_ELEVATION,
        OTHER_EXPLICIT_PIPE_REFERENCE,
    }
)


def normalize_pipe_bottom_elevation_m(
    raw_pipe_elevation_m: float | None,
    *,
    pipe_vertical_reference: str,
    reference_to_pipe_bottom_offset_m: float | None = None,
) -> float | None:
    """Normalize a raw operator pipe elevation to `pipe_bottom_elevation_m`, exactly once
    (Section 3). `PIPE_BOTTOM_ELEVATION` passes through unchanged. `PIPE_CENTRELINE_ELEVATION`,
    `PIPE_TOP_ELEVATION`, and `OTHER_EXPLICIT_PIPE_REFERENCE` all require an explicit
    `reference_to_pipe_bottom_offset_m` -- normally D/2 for a circular pipe centreline, or the
    full outer diameter for a top reference, but this is NEVER assumed here; without it, no
    canonical pipe-bottom elevation is produced (`None`)."""

    if pipe_vertical_reference not in PIPE_VERTICAL_REFERENCES:
        raise ValueError(f"unknown pipe_vertical_reference: {pipe_vertical_reference!r}")
    if raw_pipe_elevation_m is None:
        return None
    if pipe_vertical_reference == PIPE_BOTTOM_ELEVATION:
        return raw_pipe_elevation_m
    if reference_to_pipe_bottom_offset_m is None:
        return None
    return raw_pipe_elevation_m - reference_to_pipe_bottom_offset_m

"""Recognizer plugin registry (MAR-033 Sections 27, 43).

A future capability registers its own recognizer here; the planner (`orchestration.planner`)
never changes, because it only ever consumes a `RecognitionDecision` -- never a
recognizer-specific detail.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from marine_engine.intake.declaration import AssetDeclaration
from marine_engine.intake.fingerprint import DataFingerprint, fingerprint_path
from marine_engine.intake.recognition import (
    RecognitionDecision,
    SemanticCandidate,
    decide_recognition,
    recognize_canonical_cpt_profile,
    recognize_canonical_terrain_product,
    recognize_generated_change_product,
    recognize_terrain_derivative_product,
    recognize_verified_analytical_bathymetry,
)

__all__ = ["Recognizer", "REGISTERED_RECOGNIZERS", "run_recognizers", "inspect_and_recognize"]

Recognizer = Callable[[DataFingerprint, Path, "AssetDeclaration | None"], list[SemanticCandidate]]

# Section 27/29: each registered recognizer has the SAME generic signature
# `(fingerprint, path, asset_declaration) -> list[SemanticCandidate]` -- a future ticket appends
# its own here (pipeline/cable route, burial profile, ...) without this module's shape, or the
# planner's, changing. `asset_declaration` (Section 6) is `None` unless the caller supplied one
# for this exact path via a run manifest's `assets:` section; a recognizer that never needs a
# declaration (e.g. canonical CPT, identified entirely from its own bytes) simply ignores it.
REGISTERED_RECOGNIZERS: tuple[Recognizer, ...] = (
    recognize_canonical_cpt_profile,
    recognize_verified_analytical_bathymetry,
    recognize_canonical_terrain_product,
    recognize_terrain_derivative_product,
    recognize_generated_change_product,
)


def run_recognizers(
    fingerprint: DataFingerprint, path: Path, asset_declaration: AssetDeclaration | None = None
) -> list[SemanticCandidate]:
    candidates: list[SemanticCandidate] = []
    for recognizer in REGISTERED_RECOGNIZERS:
        candidates.extend(recognizer(fingerprint, path, asset_declaration))
    return candidates


def inspect_and_recognize(
    path: str | Path, *, asset_declaration: AssetDeclaration | None = None
) -> tuple[DataFingerprint, RecognitionDecision]:
    """The single entry point the CLI commands (Section 33) call: fingerprint the real bytes, run
    every registered recognizer, and fold the candidates into one deterministic decision.

    `asset_declaration` (MAR-034 Section 6) is the caller's own declared facts for this exact
    path, if any (e.g. resolved from a run manifest's `assets:` section) -- passed through to
    every recognizer unchanged; a recognizer that needs no declaration simply ignores it.
    """

    resolved = Path(path)
    fingerprint = fingerprint_path(resolved)
    candidates = run_recognizers(fingerprint, resolved, asset_declaration)
    decision = decide_recognition(fingerprint, candidates)
    return fingerprint, decision

"""Recognizer plugin registry (MAR-033 Sections 27, 43).

A future capability registers its own recognizer here; the planner (`orchestration.planner`)
never changes, because it only ever consumes a `RecognitionDecision` -- never a
recognizer-specific detail.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from marine_engine.intake.fingerprint import DataFingerprint, fingerprint_path
from marine_engine.intake.recognition import (
    RecognitionDecision,
    SemanticCandidate,
    decide_recognition,
    recognize_canonical_cpt_profile,
)

__all__ = ["Recognizer", "REGISTERED_RECOGNIZERS", "run_recognizers", "inspect_and_recognize"]

Recognizer = Callable[[DataFingerprint, Path], list[SemanticCandidate]]

# Section 27: MAR-033 registers exactly one recognizer. A future ticket appends its own here
# (bathymetry raster, pipeline/cable route, burial profile, ...) without this module's shape, or
# the planner's, changing.
REGISTERED_RECOGNIZERS: tuple[Recognizer, ...] = (recognize_canonical_cpt_profile,)


def run_recognizers(fingerprint: DataFingerprint, path: Path) -> list[SemanticCandidate]:
    candidates: list[SemanticCandidate] = []
    for recognizer in REGISTERED_RECOGNIZERS:
        candidates.extend(recognizer(fingerprint, path))
    return candidates


def inspect_and_recognize(path: str | Path) -> tuple[DataFingerprint, RecognitionDecision]:
    """The single entry point the CLI commands (Section 33) call: fingerprint the real bytes, run
    every registered recognizer, and fold the candidates into one deterministic decision."""

    resolved = Path(path)
    fingerprint = fingerprint_path(resolved)
    candidates = run_recognizers(fingerprint, resolved)
    decision = decide_recognition(fingerprint, candidates)
    return fingerprint, decision

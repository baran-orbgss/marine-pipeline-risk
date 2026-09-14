"""Semantic role recognition (MAR-033 Sections 27-29).

Structural recognition (`fingerprint.py`) may be automatic. Scientific semantic recognition may
only auto-accept when evidence is sufficiently specific (Section 28): a GeoTIFF with a CRS and
elevation-like values is NOT automatically bathymetry; a LineString is NOT automatically a
pipeline; a CSV with columns named `x`/`y` is NOT automatically spatial. Unknown/ambiguous data
becomes `UNCLASSIFIED` (or, when a lookalike candidate exists but is not confidently verified,
still `UNCLASSIFIED`/`AMBIGUOUS` with `required_confirmation=True`) -- never guessed for
convenience.

Only one recognizer is registered by MAR-033 (`recognize_canonical_cpt_profile`), and it
delegates ENTIRELY to the accepted MAR-032/032A/032B canonical-identity reader/verifier
(`geotechnical.cpt_profile`) -- it never re-derives that logic. This is deliberate: the
architecture (`registry.py`) must let future tickets register additional recognizers (bathymetry,
pipeline route, burial profile, ...) without touching the planner, but MAR-033 does not need to
implement them to prove the architecture (Section 27).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marine_engine.geotechnical import cpt_contract, cpt_profile
from marine_engine.intake.fingerprint import (
    CONTAINER_PARQUET,
    CONTAINER_UNREADABLE,
    DataFingerprint,
)

__all__ = [
    "CONFIDENCE_INSUFFICIENT",
    "CONFIDENCE_SUFFICIENT",
    "RECOGNIZED",
    "AMBIGUOUS",
    "UNCLASSIFIED",
    "CONTRADICTED",
    "INVALID",
    "DECISION_STATES",
    "SemanticCandidate",
    "RecognitionDecision",
    "recognize_canonical_cpt_profile",
    "decide_recognition",
]

CONFIDENCE_INSUFFICIENT = "INSUFFICIENT"
CONFIDENCE_SUFFICIENT = "SUFFICIENT"

RECOGNIZED = "RECOGNIZED"
AMBIGUOUS = "AMBIGUOUS"
UNCLASSIFIED = "UNCLASSIFIED"
CONTRADICTED = "CONTRADICTED"
INVALID = "INVALID"

DECISION_STATES = frozenset({RECOGNIZED, AMBIGUOUS, UNCLASSIFIED, CONTRADICTED, INVALID})


@dataclass(frozen=True)
class SemanticCandidate:
    """Section 27's example shape, verbatim."""

    role: str
    confidence: str
    evidence: tuple[str, ...]
    contradictions: tuple[str, ...]
    required_confirmation: bool
    recognizer_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "contradictions": list(self.contradictions),
            "required_confirmation": self.required_confirmation,
            "recognizer_id": self.recognizer_id,
        }


@dataclass(frozen=True)
class RecognitionDecision:
    state: str
    candidates: tuple[SemanticCandidate, ...]
    recognized_role: str | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "recognized_role": self.recognized_role,
            "candidates": [c.to_dict() for c in self.candidates],
            "reasons": list(self.reasons),
        }


def recognize_canonical_cpt_profile(
    fingerprint: DataFingerprint, path: Path
) -> list[SemanticCandidate]:
    """Delegates entirely to the accepted canonical-identity reader/verifier. A structurally
    CPT-lookalike parquet without a verified marker still produces a candidate, but marked
    `required_confirmation=True` / `CONFIDENCE_INSUFFICIENT` rather than silently becoming
    measured CPT evidence (Section 43: "structural lookalike without canonical identity does not
    become measured CPT")."""

    if fingerprint.container_type != CONTAINER_PARQUET:
        return []
    structural_missing = cpt_profile.structural_columns_missing(fingerprint.column_names)
    if structural_missing:
        return []  # not even structurally CPT-shaped; this recognizer has nothing to say

    try:
        marker = cpt_profile.read_canonical_cpt_product_marker(path)
    except Exception as exc:  # noqa: BLE001 -- an unreadable marker is a non-match, never a crash
        return [
            SemanticCandidate(
                role=cpt_contract.MEASURED_CPT_CPTU_PROFILE,
                confidence=CONFIDENCE_INSUFFICIENT,
                evidence=(
                    "structurally CPT-lookalike parquet (canonical structural columns present)",
                ),
                contradictions=(f"canonical product marker could not be read: {exc}",),
                required_confirmation=True,
                recognizer_id="canonical_cpt_profile",
            )
        ]

    import pandas as pd

    dataframe = pd.read_parquet(path)
    verification = cpt_profile.verify_canonical_cpt_product(dataframe, marker)
    if verification.verified and marker.product_role == cpt_contract.MEASURED_CPT_CPTU_PROFILE:
        return [
            SemanticCandidate(
                role=cpt_contract.MEASURED_CPT_CPTU_PROFILE,
                confidence=CONFIDENCE_SUFFICIENT,
                evidence=(
                    f"verified {cpt_contract.CPT_CANONICAL_PROFILE_CONTRACT} product marker "
                    f"(evidence_id={marker.evidence_id!r})",
                    "full required V1 column contract present",
                    "row-level source_id lineage verified against the marker evidence id",
                ),
                contradictions=(),
                required_confirmation=False,
                recognizer_id="canonical_cpt_profile",
            )
        ]
    return [
        SemanticCandidate(
            role=cpt_contract.MEASURED_CPT_CPTU_PROFILE,
            confidence=CONFIDENCE_INSUFFICIENT,
            evidence=("structurally CPT-lookalike parquet (canonical structural columns present)",),
            contradictions=tuple(verification.problems),
            required_confirmation=True,
            recognizer_id="canonical_cpt_profile",
        )
    ]


def decide_recognition(
    fingerprint: DataFingerprint, candidates: list[SemanticCandidate]
) -> RecognitionDecision:
    """Section 29: a deterministic decision from whatever candidates the registered recognizers
    produced. Auto-processing (Sections 27, 33) may occur only for `RECOGNIZED` inputs."""

    if fingerprint.container_type == CONTAINER_UNREADABLE:
        return RecognitionDecision(
            state=INVALID,
            candidates=tuple(candidates),
            recognized_role=None,
            reasons=tuple(fingerprint.read_problems) or ("file could not be read",),
        )

    sufficient = [
        c for c in candidates if c.confidence == CONFIDENCE_SUFFICIENT and not c.contradictions
    ]
    sufficient_roles = {c.role for c in sufficient}
    if len(sufficient_roles) > 1:
        return RecognitionDecision(
            state=CONTRADICTED,
            candidates=tuple(candidates),
            recognized_role=None,
            reasons=(
                "more than one recognizer produced a sufficient-confidence candidate with a "
                f"different role: {sorted(sufficient_roles)}",
            ),
        )
    if len(sufficient) >= 1:
        return RecognitionDecision(
            state=RECOGNIZED,
            candidates=tuple(candidates),
            recognized_role=sufficient[0].role,
            reasons=(),
        )
    if not candidates:
        return RecognitionDecision(
            state=UNCLASSIFIED,
            candidates=(),
            recognized_role=None,
            reasons=("no registered recognizer produced a candidate for this input",),
        )
    candidate_roles = {c.role for c in candidates}
    if len(candidate_roles) > 1:
        return RecognitionDecision(
            state=AMBIGUOUS,
            candidates=tuple(candidates),
            recognized_role=None,
            reasons=(
                f"multiple candidate roles proposed, none sufficiently confident: "
                f"{sorted(candidate_roles)}",
            ),
        )
    return RecognitionDecision(
        state=UNCLASSIFIED,
        candidates=tuple(candidates),
        recognized_role=None,
        reasons=("a candidate role exists but requires explicit semantic confirmation",),
    )

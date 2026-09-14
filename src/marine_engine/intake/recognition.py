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

from marine_engine.change import dod as change_dod
from marine_engine.geotechnical import cpt_contract, cpt_profile
from marine_engine.intake.declaration import AssetDeclaration
from marine_engine.intake.fingerprint import (
    CONTAINER_GEOTIFF,
    CONTAINER_PARQUET,
    CONTAINER_UNREADABLE,
    DataFingerprint,
)
from marine_engine.project import categories as project_categories
from marine_engine.slope_stability import contract as slope_stability_contract
from marine_engine.terrain import product_roles as terrain_product_roles

__all__ = [
    "CONFIDENCE_INSUFFICIENT",
    "CONFIDENCE_SUFFICIENT",
    "CANONICAL_VERIFIED",
    "SOURCE_METADATA_VERIFIED",
    "USER_DECLARED",
    "STRUCTURAL_CANDIDATE",
    "UNKNOWN",
    "EVIDENCE_LEVELS",
    "RECOGNIZED",
    "AMBIGUOUS",
    "UNCLASSIFIED",
    "NEEDS_SEMANTIC_CONFIRMATION",
    "CONTRADICTED",
    "INVALID",
    "DECISION_STATES",
    "SemanticCandidate",
    "RecognitionDecision",
    "recognize_canonical_cpt_profile",
    "recognize_verified_analytical_bathymetry",
    "recognize_canonical_terrain_product",
    "recognize_terrain_derivative_product",
    "recognize_generated_change_product",
    "decide_recognition",
]

CONFIDENCE_INSUFFICIENT = "INSUFFICIENT"
CONFIDENCE_SUFFICIENT = "SUFFICIENT"

# MAR-034 Section 5: WHERE a candidate's evidence came from -- orthogonal to `confidence` (how
# strong the recognizer judged that evidence). A `STRUCTURAL_CANDIDATE` must never be silently
# treated as `CANONICAL_VERIFIED`; nothing here upgrades one level into another.
CANONICAL_VERIFIED = "CANONICAL_VERIFIED"  # this engine's own verified canonical product marker
SOURCE_METADATA_VERIFIED = "SOURCE_METADATA_VERIFIED"  # embedded source-file metadata confirmed it
USER_DECLARED = "USER_DECLARED"  # an explicit AssetDeclaration, checked against real structure
STRUCTURAL_CANDIDATE = "STRUCTURAL_CANDIDATE"  # structurally plausible only, nothing confirms it
UNKNOWN = "UNKNOWN"  # no evidence at all (a candidate is not normally emitted at this level)

EVIDENCE_LEVELS = frozenset(
    {CANONICAL_VERIFIED, SOURCE_METADATA_VERIFIED, USER_DECLARED, STRUCTURAL_CANDIDATE, UNKNOWN}
)

RECOGNIZED = "RECOGNIZED"
AMBIGUOUS = "AMBIGUOUS"
UNCLASSIFIED = "UNCLASSIFIED"
NEEDS_SEMANTIC_CONFIRMATION = "NEEDS_SEMANTIC_CONFIRMATION"
CONTRADICTED = "CONTRADICTED"
INVALID = "INVALID"

DECISION_STATES = frozenset(
    {RECOGNIZED, AMBIGUOUS, UNCLASSIFIED, NEEDS_SEMANTIC_CONFIRMATION, CONTRADICTED, INVALID}
)


@dataclass(frozen=True)
class SemanticCandidate:
    """Section 27's example shape, plus Section 5's `evidence_level` (WHERE the evidence for
    `role` came from -- kept distinct from `confidence`, HOW strong the recognizer judged it)."""

    role: str
    confidence: str
    evidence_level: str
    evidence: tuple[str, ...]
    contradictions: tuple[str, ...]
    required_confirmation: bool
    recognizer_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "confidence": self.confidence,
            "evidence_level": self.evidence_level,
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
    fingerprint: DataFingerprint, path: Path, asset_declaration: AssetDeclaration | None = None
) -> list[SemanticCandidate]:
    """Delegates entirely to the accepted canonical-identity reader/verifier. A structurally
    CPT-lookalike parquet without a verified marker still produces a candidate, but marked
    `required_confirmation=True` / `CONFIDENCE_INSUFFICIENT` rather than silently becoming
    measured CPT evidence (Section 43: "structural lookalike without canonical identity does not
    become measured CPT"). `asset_declaration` is accepted for signature uniformity with every
    other registered recognizer (Section 27) but never consulted: canonical CPT identity is
    established entirely from the file's own verified marker, never from a declaration."""

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
                evidence_level=STRUCTURAL_CANDIDATE,
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
                evidence_level=CANONICAL_VERIFIED,
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
            evidence_level=STRUCTURAL_CANDIDATE,
            evidence=("structurally CPT-lookalike parquet (canonical structural columns present)",),
            contradictions=tuple(verification.problems),
            required_confirmation=True,
            recognizer_id="canonical_cpt_profile",
        )
    ]


def _read_raster_tags(path: Path) -> dict[str, str] | None:
    """Best-effort read of a GeoTIFF's own embedded GDAL tags -- the accepted MAR-020/MAR-021
    canonical-product identity mechanism (`terrain.raster_io.write_terrain_raster`'s `tags`
    argument, already consumed by `slope_stability.contract`'s canonical-terrain check). Returns
    `None` on any read failure rather than raising -- an unreadable/corrupt raster is simply not
    a match, never a crash."""

    try:
        import rasterio

        with rasterio.open(path) as dataset:
            return dict(dataset.tags())
    except Exception:  # noqa: BLE001 -- unreadable raster tags is a non-match, never a crash
        return None


def recognize_verified_analytical_bathymetry(
    fingerprint: DataFingerprint, path: Path, asset_declaration: AssetDeclaration | None = None
) -> list[SemanticCandidate]:
    """Section 11: recognition succeeds ONLY through an explicit `AssetDeclaration` claiming
    `semantic_role=BATHYMETRY_RASTER` (Section 6-7's generic, engine-owned declaration -- reusing
    the existing MAR-026 category name, never a competing one) combined with structural raster
    compatibility (single-band, georeferenced). A plain GeoTIFF with no declaration at all
    produces NO candidate -- it remains unclassified (Section 4/11), never guessed from CRS or
    values alone."""

    if fingerprint.container_type != CONTAINER_GEOTIFF:
        return []
    if (
        asset_declaration is None
        or asset_declaration.semantic_role != project_categories.BATHYMETRY_RASTER
    ):
        return []  # no declared claim at all -- this recognizer has nothing to say

    structurally_compatible = (
        fingerprint.band_count == 1
        and fingerprint.crs is not None
        and fingerprint.raster_width is not None
        and fingerprint.raster_height is not None
    )
    if not structurally_compatible:
        return [
            SemanticCandidate(
                role=project_categories.BATHYMETRY_RASTER,
                confidence=CONFIDENCE_INSUFFICIENT,
                evidence_level=STRUCTURAL_CANDIDATE,
                evidence=(
                    f"asset declared semantic_role={project_categories.BATHYMETRY_RASTER!r}",
                ),
                contradictions=(
                    f"raster is not structurally compatible with single-band analytical "
                    f"bathymetry (band_count={fingerprint.band_count!r}, "
                    f"crs={fingerprint.crs!r}, "
                    f"dimensions={fingerprint.raster_width!r}x{fingerprint.raster_height!r})",
                ),
                required_confirmation=True,
                recognizer_id="verified_analytical_bathymetry",
            )
        ]
    return [
        SemanticCandidate(
            role=project_categories.BATHYMETRY_RASTER,
            confidence=CONFIDENCE_SUFFICIENT,
            evidence_level=USER_DECLARED,
            evidence=(
                f"explicit AssetDeclaration semantic_role={project_categories.BATHYMETRY_RASTER!r}",
                "structurally compatible single-band georeferenced raster",
            ),
            contradictions=(),
            required_confirmation=False,
            recognizer_id="verified_analytical_bathymetry",
        )
    ]


def recognize_canonical_terrain_product(
    fingerprint: DataFingerprint, path: Path, asset_declaration: AssetDeclaration | None = None
) -> list[SemanticCandidate]:
    """Recognizes a PREVIOUSLY GENERATED canonical bed-elevation product fed back into the engine
    (e.g. from an earlier `auto-process` run, or from the standalone MAR-020
    `build-highres-terrain-poc` command) via its own embedded `scientific_role`/`layer` tags --
    the same accepted identity `slope_stability.contract` already checks, never re-derived."""

    if fingerprint.container_type != CONTAINER_GEOTIFF:
        return []
    tags = _read_raster_tags(path)
    if tags is None:
        return []
    role = tags.get("scientific_role")
    layer = tags.get("layer")
    if (
        role != slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED
        or layer != slope_stability_contract.SOURCE_TERRAIN_LAYER_REQUIRED
    ):
        return []
    return [
        SemanticCandidate(
            role=slope_stability_contract.SOURCE_TERRAIN_ROLE_REQUIRED,
            confidence=CONFIDENCE_SUFFICIENT,
            evidence_level=CANONICAL_VERIFIED,
            evidence=(f"embedded scientific_role={role!r}, layer={layer!r} tags verified",),
            contradictions=(),
            required_confirmation=False,
            recognizer_id="canonical_terrain_product",
        )
    ]


def recognize_terrain_derivative_product(
    fingerprint: DataFingerprint, path: Path, asset_declaration: AssetDeclaration | None = None
) -> list[SemanticCandidate]:
    """Recognizes a PREVIOUSLY GENERATED terrain-derivative raster (slope, aspect, curvature, ...)
    via its own embedded `scientific_role` tag -- the role MAR-034's own `terrain_derivatives`
    capability writes onto every layer it produces."""

    if fingerprint.container_type != CONTAINER_GEOTIFF:
        return []
    tags = _read_raster_tags(path)
    if tags is None:
        return []
    if tags.get("scientific_role") != terrain_product_roles.TERRAIN_DERIVATIVE_PRODUCT:
        return []
    return [
        SemanticCandidate(
            role=terrain_product_roles.TERRAIN_DERIVATIVE_PRODUCT,
            confidence=CONFIDENCE_SUFFICIENT,
            evidence_level=CANONICAL_VERIFIED,
            evidence=(
                f"embedded scientific_role={terrain_product_roles.TERRAIN_DERIVATIVE_PRODUCT!r} "
                "tag verified",
            ),
            contradictions=(),
            required_confirmation=False,
            recognizer_id="terrain_derivative_product",
        )
    ]


def recognize_generated_change_product(
    fingerprint: DataFingerprint, path: Path, asset_declaration: AssetDeclaration | None = None
) -> list[SemanticCandidate]:
    """Recognizes a PREVIOUSLY GENERATED multi-epoch change (DoD) raster via its own embedded
    `scientific_role` tag -- the same name already used by the standalone MAR-021
    `build-seabed-change-poc` command's own output tag."""

    if fingerprint.container_type != CONTAINER_GEOTIFF:
        return []
    tags = _read_raster_tags(path)
    if tags is None:
        return []
    if tags.get("scientific_role") != change_dod.MULTI_EPOCH_SEABED_CHANGE_POC:
        return []
    return [
        SemanticCandidate(
            role=change_dod.MULTI_EPOCH_SEABED_CHANGE_POC,
            confidence=CONFIDENCE_SUFFICIENT,
            evidence_level=CANONICAL_VERIFIED,
            evidence=(
                f"embedded scientific_role={change_dod.MULTI_EPOCH_SEABED_CHANGE_POC!r} tag "
                "verified",
            ),
            contradictions=(),
            required_confirmation=False,
            recognizer_id="generated_change_product",
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
        state=NEEDS_SEMANTIC_CONFIRMATION,
        candidates=tuple(candidates),
        recognized_role=None,
        reasons=("a candidate role exists but requires explicit semantic confirmation",),
    )

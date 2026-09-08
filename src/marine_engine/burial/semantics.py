"""Source burial-measurement semantics -- a hard gate before interpretation (MAR-024 Section 4).

Before any measured value is classified as buried/exposed/at-seabed, this
module forces an explicit answer to "what does the source's own number
actually mean". Resolving this is an evidence-gathering exercise the
caller performs against the real source documentation and data (this
module never guesses); this module's job is only to hold the resulting
vocabulary and assemble it into a stable, JSON-ready record so the
resolution is never silently skipped or later forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# --- Section 4: measurement reference vocabulary -- never assumed ---------------------------

TOP_OF_ASSET_BURIAL = "TOP_OF_ASSET_BURIAL"
CENTRELINE_BURIAL = "CENTRELINE_BURIAL"
OTHER_SOURCE_SPECIFIC_REFERENCE = "OTHER_SOURCE_SPECIFIC_REFERENCE"
# The escape hatch this section exists for: never silently relabelled as top-of-asset burial.
SOURCE_BURIAL_REFERENCE_UNRESOLVED = "SOURCE_BURIAL_REFERENCE_UNRESOLVED"

BURIAL_REFERENCE_TYPES = frozenset(
    {
        TOP_OF_ASSET_BURIAL,
        CENTRELINE_BURIAL,
        OTHER_SOURCE_SPECIFIC_REFERENCE,
        SOURCE_BURIAL_REFERENCE_UNRESOLVED,
    }
)

# --- MAR-024A Problem A: explicit sign-convention vocabulary -- never arbitrary free text ----
# for physical classification. Source free text describing sign behaviour may still be
# preserved separately (`BurialMeasurementSemantics.source_sign_convention_text`).

POSITIVE_VALUE_MEANS_DEEPER_BURIAL = "POSITIVE_VALUE_MEANS_DEEPER_BURIAL"
NEGATIVE_VALUE_MEANS_DEEPER_BURIAL = "NEGATIVE_VALUE_MEANS_DEEPER_BURIAL"
# The escape hatch this vocabulary exists for: never silently assumed to be either convention.
SIGN_CONVENTION_UNRESOLVED = "SIGN_CONVENTION_UNRESOLVED"

SIGN_CONVENTIONS = frozenset(
    {
        POSITIVE_VALUE_MEANS_DEEPER_BURIAL,
        NEGATIVE_VALUE_MEANS_DEEPER_BURIAL,
        SIGN_CONVENTION_UNRESOLVED,
    }
)


@dataclass(frozen=True)
class BurialMeasurementSemantics:
    """One resolved (or explicitly unresolved) statement of what a source's own burial/depth
    column means. Every field the ticket requires is present -- `unknown_fields` names
    anything the source simply does not state, rather than omitting it silently.

    `sign_convention` is always one of `SIGN_CONVENTIONS` -- never arbitrary free text -- since
    it drives physical classification (MAR-024A Problem A). `source_sign_convention_text` is
    an optional, separate free-text record of what the source itself says about sign, kept for
    provenance without ever being trusted for classification.
    """

    source_measurement_name: str
    measurement_reference_point: str
    sign_convention: str
    units: str
    source_stated_uncertainty_available: bool
    survey_technique: str | None
    unknown_fields: tuple[str, ...]
    resolution_evidence: str
    source_sign_convention_text: str | None = None

    def __post_init__(self) -> None:
        if self.measurement_reference_point not in BURIAL_REFERENCE_TYPES:
            raise ValueError(
                f"unknown measurement_reference_point: {self.measurement_reference_point!r}"
            )
        if self.sign_convention not in SIGN_CONVENTIONS:
            raise ValueError(
                f"unknown sign_convention: {self.sign_convention!r} -- must be one of "
                f"{sorted(SIGN_CONVENTIONS)}"
            )


def build_source_burial_semantics(semantics: BurialMeasurementSemantics) -> dict[str, Any]:
    """The JSON-ready `source_burial_semantics.json` content (Section 4)."""

    return {
        "scientific_role": "SOURCE_BURIAL_MEASUREMENT_SEMANTICS_RESOLUTION",
        "source_measurement_name": semantics.source_measurement_name,
        "measurement_reference_point": semantics.measurement_reference_point,
        "sign_convention": semantics.sign_convention,
        "source_sign_convention_text": semantics.source_sign_convention_text,
        "units": semantics.units,
        "source_stated_uncertainty_available": semantics.source_stated_uncertainty_available,
        "survey_technique": semantics.survey_technique,
        "unknown_fields": list(semantics.unknown_fields),
        "resolution_evidence": semantics.resolution_evidence,
        "reference_resolved": (
            semantics.measurement_reference_point != SOURCE_BURIAL_REFERENCE_UNRESOLVED
        ),
        "sign_convention_resolved": (semantics.sign_convention != SIGN_CONVENTION_UNRESOLVED),
    }

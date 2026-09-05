"""Official freespan comment / temporal-lineage provenance repair (MAR-014B).

Source-stated only, never inferred (Sections 4-7)
--------------------------------------------------------
Table B.1's own per-row comments and the surrounding document narrative
describe which spans the source ITSELF calls "the same span" across
surveys, which spans changed length/height, and which areas simply were
not surveyed in an earlier year. This module carries that evidence
forward -- it never adds a relationship the source does not state, and it
never resolves an ambiguous table layout by picking the spatially nearest
candidate event (`marine_engine.resources.ALLOWED_ATTRIBUTION_METHODS`
enforces this at the resource layer; this module never bypasses it).

Canonical role: `SOURCE_STATED_HISTORICAL_FREESPAN_EVOLUTION_EVIDENCE`.
Never a susceptibility score, probability, prediction model, or ML
validation metric.

Coverage semantics are honest about their own asymmetry (Section 8)
-------------------------------------------------------------------------------
2014's own survey narrative states it mapped only the Anglia West/Anglia A
NUI/LOGGS areas, not the full pipeline routes --
`COVERAGE_STATUS_2014 = PARTIAL_ROUTE_COVERAGE_SOURCE_STATED`. 2018 is
recorded only as `COVERAGE_STATUS_2018 = PRE_DECOMMISSIONING_SURVEY` --
this module NEVER promotes an absence of a 2018-listed event to a
route-wide "no freespan here" negative label
(`FORBIDDEN_2018_FULL_ROUTE_NEGATIVE_LABEL`) unless a future official
source explicitly states that stronger claim.
"""

import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCIENTIFIC_ROLE = "SOURCE_STATED_HISTORICAL_FREESPAN_EVOLUTION_EVIDENCE"

COVERAGE_STATUS_2014 = "PARTIAL_ROUTE_COVERAGE_SOURCE_STATED"
COVERAGE_STATUS_2018 = "PRE_DECOMMISSIONING_SURVEY"

# Documentary only -- this exact label must never be applied to 2018 unless a
# future official source explicitly supports it (Section 8).
FORBIDDEN_2018_FULL_ROUTE_NEGATIVE_LABEL = "2018_FULL_ROUTE_COMPLETE_NEGATIVE_LABEL_COVERAGE"

_COVERAGE_SOURCE_STATEMENT_2014 = (
    "The 2014 survey mapped only the Anglia West, Anglia A NUI, and LOGGS areas and "
    "did not cover the full pipeline routes."
)


def build_survey_coverage_metadata() -> dict[str, Any]:
    """The explicit, source-grounded coverage semantics for 2014 and 2018 (Section 8).

    2018 deliberately carries no source statement -- its status records
    only what kind of survey it was, never a claim about completeness.
    """

    return {
        "2014": {
            "coverage_status": COVERAGE_STATUS_2014,
            "source_statement": _COVERAGE_SOURCE_STATEMENT_2014,
        },
        "2018": {
            "coverage_status": COVERAGE_STATUS_2018,
            "source_statement": None,
        },
        "2018_full_route_negative_label_assumption_applied": False,
    }


TEMPORAL_EVIDENCE_COLUMNS = (
    "relationship_id",
    "survey_year_a",
    "event_id_a",
    "canonical_mid_chainage_a_m",
    "canonical_mid_kp_a",
    "survey_year_b",
    "event_id_b",
    "canonical_mid_chainage_b_m",
    "canonical_mid_kp_b",
    "relationship_type",
    "source_statement",
    "source_page",
    "source_table",
    "attribution_method",
    "scientific_role",
)


def _lookup_chainage_context(
    event_id: Any, spatial_evidence_by_id: dict[str, dict[str, Any]]
) -> tuple[float | None, str | None]:
    if event_id is None or (isinstance(event_id, float) and pd.isna(event_id)):
        return None, None
    record = spatial_evidence_by_id.get(event_id)
    if record is None:
        return None, None
    return record["canonical_mid_chainage_m"], record["canonical_mid_kp"]


def build_freespan_temporal_relationship_evidence(
    relationships_df: pd.DataFrame, spatial_evidence_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach canonical chainage/KP CONTEXT to each source-stated relationship.

    `spatial_evidence_df` (MAR-014A's own `anglia_freespan_spatial_evidence`
    output) supplies chainage/KP purely for human/plot context -- a null
    `event_id_a`/`event_id_b` (a group/narrative-only row, Section 6) simply
    gets a null context, never a fabricated position. The relationship
    itself -- every other field -- is copied through unchanged from
    `relationships_df`; this function never invents or drops a relationship.
    """

    spatial_evidence_by_id = {
        row["event_id"]: {
            "canonical_mid_chainage_m": row["canonical_mid_chainage_m"],
            "canonical_mid_kp": row["canonical_mid_kp"],
        }
        for _, row in spatial_evidence_df.iterrows()
    }

    records = []
    for _, row in relationships_df.iterrows():
        chainage_a, kp_a = _lookup_chainage_context(row["event_id_a"], spatial_evidence_by_id)
        chainage_b, kp_b = _lookup_chainage_context(row["event_id_b"], spatial_evidence_by_id)
        records.append(
            {
                "relationship_id": row["relationship_id"],
                "survey_year_a": row["survey_year_a"],
                "event_id_a": row["event_id_a"],
                "canonical_mid_chainage_a_m": chainage_a,
                "canonical_mid_kp_a": kp_a,
                "survey_year_b": row["survey_year_b"],
                "event_id_b": row["event_id_b"],
                "canonical_mid_chainage_b_m": chainage_b,
                "canonical_mid_kp_b": kp_b,
                "relationship_type": row["relationship_type"],
                "source_statement": row["source_statement"],
                "source_page": row["source_page"],
                "source_table": row["source_table"],
                "attribution_method": row["attribution_method"],
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )
    return pd.DataFrame(records, columns=list(TEMPORAL_EVIDENCE_COLUMNS))


def write_freespan_temporal_relationship_evidence(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def print_freespan_temporal_provenance_report(
    *,
    freespans_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    temporal_evidence_df: pd.DataFrame,
    coverage_metadata: dict[str, Any],
    temporal_evidence_path: Path,
    figure_path: Path,
    figure_dimensions: tuple[int, int],
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854/PL855 Official Freespan Temporal-Lineage Provenance (MAR-014B) ===", ""]

    event_comment_count = int(freespans_df["source_comment"].notna().sum())
    lines.append(f"Event-specific source comments preserved: {event_comment_count}")
    for _, row in freespans_df[freespans_df["source_comment"].notna()].iterrows():
        lines.append(f"  {row['event_id']}: {row['source_comment']}")
    lines.append("")

    event_level = relationships_df[
        relationships_df["event_id_a"].notna() | relationships_df["event_id_b"].notna()
    ]
    narrative_only = relationships_df[
        relationships_df["event_id_a"].isna() & relationships_df["event_id_b"].isna()
    ]
    lines.append(f"Source-stated event-level relationships: {len(event_level)}")
    lines.append(
        f"Group/narrative statements (never forced into an event pair): {len(narrative_only)}"
    )
    lines.append("")

    lines.append("## Same-span relationships retained")
    same_span = relationships_df[relationships_df["relationship_type"] == "SOURCE_STATED_SAME_SPAN"]
    for _, row in same_span.iterrows():
        event_b = row["event_id_b"] if pd.notna(row["event_id_b"]) else "(unresolved)"
        lines.append(f"  {row['event_id_a']} <-> {event_b}: {row['source_statement']}")
    lines.append("")

    lines.append("## Coverage semantics")
    lines.append(
        f"  2014: {coverage_metadata['2014']['coverage_status']} "
        f"({coverage_metadata['2014']['source_statement']})"
    )
    lines.append(f"  2018: {coverage_metadata['2018']['coverage_status']}")
    lines.append(
        "  2018_full_route_negative_label_assumption_applied: "
        f"{coverage_metadata['2018_full_route_negative_label_assumption_applied']}"
    )
    lines.append("")

    lines.append(
        f"Temporal evidence output: {temporal_evidence_path} ({len(temporal_evidence_df)} rows)"
    )
    lines.append(
        f"Temporal evolution figure: {figure_path} "
        f"({figure_dimensions[0]}x{figure_dimensions[1]} px)"
    )
    lines.append("")

    lines.append("NO CROSS-SURVEY RELATIONSHIP WAS CREATED BY SPATIAL PROXIMITY ALONE.")
    lines.append(
        "SOURCE-STATED TEMPORAL EVOLUTION EVIDENCE IS PRESERVED SEPARATELY FROM MODEL OUTPUTS."
    )

    print("\n".join(lines), file=file)

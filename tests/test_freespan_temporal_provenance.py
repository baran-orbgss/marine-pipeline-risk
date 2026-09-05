"""Offline unit tests for marine_engine.scour.freespan_temporal_provenance (MAR-014B).

Small hand-built synthetic relationship/spatial-evidence tables only --
never the real PL854 route, never network access. Lettered comments map to
MAR-014B Section 13's required test list.
"""

import pandas as pd

from marine_engine.scour import freespan_temporal_provenance as ftp


def _relationships_df() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "relationship_id": "REL-01",
                "survey_year_a": 2018,
                "event_id_a": "2018-01",
                "survey_year_b": None,
                "event_id_b": None,
                "relationship_type": "SOURCE_STATED_AREA_NOT_SURVEYED_PREVIOUSLY",
                "source_statement": (
                    "Span identified from 2018 survey in area not surveyed in 2014."
                ),
                "source_page": 44,
                "source_table": "Appendix B Table B.1",
                "attribution_method": "DIRECT_TABLE_COMMENT",
            },
            {
                "relationship_id": "REL-04",
                "survey_year_a": 2014,
                "event_id_a": "2014-05",
                "survey_year_b": 2018,
                "event_id_b": "2018-06",
                "relationship_type": "SOURCE_STATED_SAME_SPAN",
                "source_statement": "Same span 2014 and 2018.",
                "source_page": 44,
                "source_table": "Appendix B Table B.1",
                "attribution_method": "DIRECT_TABLE_COMMENT",
            },
            {
                "relationship_id": "REL-09",
                "survey_year_a": None,
                "event_id_a": None,
                "survey_year_b": None,
                "event_id_b": None,
                "relationship_type": "SOURCE_NARRATIVE_CORRESPONDENCE",
                "source_statement": "Freespans changed over time in length, height, and location.",
                "source_page": None,
                "source_table": None,
                "attribution_method": "DIRECT_DOCUMENT_NARRATIVE",
            },
        ]
    )


def _spatial_evidence_df() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "event_id": "2018-01",
                "canonical_mid_chainage_m": 22896.9,
                "canonical_mid_kp": "KP 22+897",
            },
            {
                "event_id": "2014-05",
                "canonical_mid_chainage_m": 128.5,
                "canonical_mid_kp": "KP 0+129",
            },
            {
                "event_id": "2018-06",
                "canonical_mid_chainage_m": 132.0,
                "canonical_mid_kp": "KP 0+132",
            },
        ]
    )


# --- G: 2014 coverage is explicitly partial and source-stated --------------------------


def test_G_2014_coverage_is_partial_and_source_stated():
    metadata = ftp.build_survey_coverage_metadata()
    assert metadata["2014"]["coverage_status"] == "PARTIAL_ROUTE_COVERAGE_SOURCE_STATED"
    assert metadata["2014"]["source_statement"] is not None
    assert "Anglia West" in metadata["2014"]["source_statement"]


# --- H: 2018 is never silently promoted to a complete route-wide negative label --------


def test_H_2018_coverage_never_promoted_to_full_route_negative_label():
    metadata = ftp.build_survey_coverage_metadata()
    assert metadata["2018"]["coverage_status"] == "PRE_DECOMMISSIONING_SURVEY"
    assert metadata["2018"]["coverage_status"] != ftp.FORBIDDEN_2018_FULL_ROUTE_NEGATIVE_LABEL
    assert metadata["2018_full_route_negative_label_assumption_applied"] is False


def test_H_2018_carries_no_source_statement_that_would_imply_completeness():
    metadata = ftp.build_survey_coverage_metadata()
    # 2018 deliberately has no source statement backing a completeness claim --
    # unlike 2014, which cites the Anglia West/A NUI/LOGGS-only statement.
    assert metadata["2018"]["source_statement"] is None


# --- I: temporal evidence output has no score/probability/prediction fields ------------


def test_I_temporal_evidence_has_no_score_or_prediction_fields():
    evidence_df = ftp.build_freespan_temporal_relationship_evidence(
        _relationships_df(), _spatial_evidence_df()
    )
    assert len(evidence_df) == 3
    forbidden = ("score", "probability", "prediction", "rank", "confidence")
    for column in evidence_df.columns:
        for token in forbidden:
            assert token not in column.lower(), column
    assert (evidence_df["scientific_role"] == ftp.SCIENTIFIC_ROLE).all()
    assert ftp.SCIENTIFIC_ROLE == "SOURCE_STATED_HISTORICAL_FREESPAN_EVOLUTION_EVIDENCE"


def test_I_narrative_row_gets_null_chainage_context_never_fabricated():
    evidence_df = ftp.build_freespan_temporal_relationship_evidence(
        _relationships_df(), _spatial_evidence_df()
    )
    narrative_row = evidence_df[evidence_df["relationship_id"] == "REL-09"].iloc[0]
    assert pd.isna(narrative_row["canonical_mid_chainage_a_m"])
    assert pd.isna(narrative_row["canonical_mid_chainage_b_m"])


def test_I_paired_row_gets_real_chainage_context_for_both_sides():
    evidence_df = ftp.build_freespan_temporal_relationship_evidence(
        _relationships_df(), _spatial_evidence_df()
    )
    paired_row = evidence_df[evidence_df["relationship_id"] == "REL-04"].iloc[0]
    assert paired_row["canonical_mid_chainage_a_m"] == 128.5
    assert paired_row["canonical_mid_chainage_b_m"] == 132.0
    assert paired_row["canonical_mid_kp_a"] == "KP 0+129"

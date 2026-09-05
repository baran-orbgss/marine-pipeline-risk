"""Offline unit tests for marine_engine.resources (MAR-014A/B tracked CSV resources).

The real tracked CSVs are tiny, version-controlled, and offline -- reading
them directly IS the subject under test here, never a network dependency.
Lettered comments map to MAR-014B Section 13's required test list.
"""

from pathlib import Path

import pandas as pd
import pytest

from marine_engine import resources


def test_loads_all_17_rows_with_required_columns():
    df = resources.load_anglia_table_b1_freespans()
    assert len(df) == 17
    assert list(df.columns) == list(resources.ANGLIA_TABLE_B1_COLUMNS)


def test_checksum_verification_passes_for_the_real_tracked_file():
    # Exercises the same checksum guard `load_anglia_table_b1_freespans` runs
    # internally -- if this ever fails, the tracked CSV itself was edited
    # incorrectly, not a test bug.
    df = resources.load_anglia_table_b1_freespans()
    for year, expected in resources._EXPECTED_CHECKSUMS.items():
        rows = df[df["survey_year"] == year]
        assert len(rows) == expected["count"]
        assert rows["source_length_m"].sum() == pytest.approx(expected["sum_length_m"], abs=0.005)


def test_wrong_event_count_raises_checksum_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real_df = resources.load_anglia_table_b1_freespans()
    tampered_df = real_df[real_df["event_id"] != "2018-08"]  # drop one 2018 row
    broken_csv = tmp_path / "broken.csv"
    tampered_df.to_csv(broken_csv, index=False)

    monkeypatch.setattr(resources, "ANGLIA_TABLE_B1_FREESPANS_CSV", broken_csv)
    with pytest.raises(resources.AngliaTableB1ChecksumError, match="2018 event count"):
        resources.load_anglia_table_b1_freespans()


def test_wrong_sum_length_raises_checksum_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real_df = resources.load_anglia_table_b1_freespans()
    tampered_df = real_df.copy()
    tampered_df.loc[tampered_df["event_id"] == "2018-08", "source_length_m"] = 99.0
    broken_csv = tmp_path / "broken.csv"
    tampered_df.to_csv(broken_csv, index=False)

    monkeypatch.setattr(resources, "ANGLIA_TABLE_B1_FREESPANS_CSV", broken_csv)
    with pytest.raises(resources.AngliaTableB1ChecksumError, match="sum"):
        resources.load_anglia_table_b1_freespans()


def test_missing_required_column_raises_checksum_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    broken_df = pd.DataFrame({"survey_year": [2018], "event_id": ["2018-01"]})
    broken_csv = tmp_path / "broken.csv"
    broken_df.to_csv(broken_csv, index=False)

    monkeypatch.setattr(resources, "ANGLIA_TABLE_B1_FREESPANS_CSV", broken_csv)
    with pytest.raises(resources.AngliaTableB1ChecksumError, match="missing required column"):
        resources.load_anglia_table_b1_freespans()


# --- A: all 17 event rows carry the correct (non-zero-based) source page ---------------


def test_A_all_17_rows_have_source_page_44():
    df = resources.load_anglia_table_b1_freespans()
    assert len(df) == 17
    assert (df["source_page"] == 44).all()


# --- B: 2018-01/02/03 preserve the source-stated not-surveyed-in-2014 comment ----------


def test_B_2018_01_02_03_preserve_not_surveyed_in_2014_comment():
    df = resources.load_anglia_table_b1_freespans()
    for event_id in ("2018-01", "2018-02", "2018-03"):
        comment = df.loc[df["event_id"] == event_id, "source_comment"].iloc[0]
        assert "not surveyed in 2014" in comment


# --- C: relationship resource contains only allowed SOURCE_STATED types ----------------


def test_C_relationships_contain_only_allowed_source_stated_types():
    rel = resources.load_anglia_table_b1_freespan_relationships()
    assert len(rel) > 0
    assert set(rel["relationship_type"]).issubset(set(resources.ALLOWED_RELATIONSHIP_TYPES))


def test_C_unallowed_relationship_type_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real_df = resources.load_anglia_table_b1_freespan_relationships()
    tampered_df = real_df.copy()
    tampered_df.loc[0, "relationship_type"] = "SPATIAL_NEAREST_NEIGHBOUR_MATCH"
    broken_csv = tmp_path / "broken_relationships.csv"
    tampered_df.to_csv(broken_csv, index=False)

    monkeypatch.setattr(resources, "ANGLIA_TABLE_B1_FREESPAN_RELATIONSHIPS_CSV", broken_csv)
    with pytest.raises(
        resources.AngliaFreespanRelationshipValidationError, match="relationship_type"
    ):
        resources.load_anglia_table_b1_freespan_relationships()


# --- D: no relationship row has attribution_method = inferred/nearest ------------------


def test_D_no_relationship_uses_an_inferred_or_nearest_neighbour_attribution():
    rel = resources.load_anglia_table_b1_freespan_relationships()
    assert set(rel["attribution_method"]).issubset(set(resources.ALLOWED_ATTRIBUTION_METHODS))
    forbidden = {"SPATIAL_NEAREST_NEIGHBOUR", "INFERRED_MATCH"}
    assert not forbidden & set(rel["attribution_method"])


def test_D_unallowed_attribution_method_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real_df = resources.load_anglia_table_b1_freespan_relationships()
    tampered_df = real_df.copy()
    tampered_df.loc[0, "attribution_method"] = "INFERRED_MATCH"
    broken_csv = tmp_path / "broken_relationships.csv"
    tampered_df.to_csv(broken_csv, index=False)

    monkeypatch.setattr(resources, "ANGLIA_TABLE_B1_FREESPAN_RELATIONSHIPS_CSV", broken_csv)
    with pytest.raises(
        resources.AngliaFreespanRelationshipValidationError, match="attribution_method"
    ):
        resources.load_anglia_table_b1_freespan_relationships()


# --- E: known 2014<->2018 same-span source statement is preserved ----------------------


def test_E_2014_to_2018_same_span_statements_preserved():
    rel = resources.load_anglia_table_b1_freespan_relationships()

    pair_1 = rel[
        (rel["event_id_a"] == "2014-05")
        & (rel["event_id_b"] == "2018-06")
        & (rel["relationship_type"] == "SOURCE_STATED_SAME_SPAN")
    ]
    assert len(pair_1) == 1
    assert pair_1.iloc[0]["attribution_method"] == "DIRECT_TABLE_COMMENT"

    pair_2 = rel[
        (rel["event_id_a"] == "2014-06")
        & (rel["event_id_b"] == "2018-07")
        & (rel["relationship_type"] == "SOURCE_STATED_SAME_SPAN")
    ]
    assert len(pair_2) == 1

    length_change = rel[
        (rel["event_id_a"] == "2014-05")
        & (rel["event_id_b"] == "2018-06")
        & (rel["relationship_type"] == "SOURCE_STATED_LENGTH_CHANGE")
    ]
    assert len(length_change) == 1
    assert "8.14" in length_change.iloc[0]["source_statement"]


# --- F: known 2012<->2014 same-span source statement is preserved, unresolved side ------


def test_F_2012_to_2014_same_span_statement_preserved_with_ambiguous_side_null():
    rel = resources.load_anglia_table_b1_freespan_relationships()

    pair = rel[
        (rel["event_id_a"] == "2012-02")
        & (rel["relationship_type"] == "SOURCE_STATED_SAME_SPAN")
        & (rel["survey_year_b"] == 2014)
    ]
    assert len(pair) == 1
    # The specific 2014 counterpart is genuinely ambiguous from this transcription --
    # Section 6 requires a null event_id_b here, never a fabricated pair.
    assert pd.isna(pair.iloc[0]["event_id_b"])
    assert "2012" in pair.iloc[0]["source_statement"] and "2014" in pair.iloc[0]["source_statement"]

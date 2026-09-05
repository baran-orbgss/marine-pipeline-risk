"""Offline unit tests for marine_engine.resources (MAR-014A tracked CSV resource).

The real tracked `anglia_table_b1_freespans.csv` is tiny, version-controlled,
and offline -- reading it directly IS the subject under test here, never a
network dependency.
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

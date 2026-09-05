"""Offline unit tests for marine_engine.providers.nsta_freespan (MAR-014C).

Pure functions only -- no network access. Lettered comments map to MAR-014C
Section 21's required test list. See `test_nsta_freespan_live.py` for the
real-service checks (layer id, URL, schema) excluded from the default run.
"""

from pathlib import Path

from marine_engine.providers import nsta_freespan

# --- C: query uses exact NSTAPIPNO for PL854/PL855 --------------------------------------


def test_C_where_clause_uses_exact_nstapipno_in_list():
    clause = nsta_freespan.build_pipeline_number_where_clause(("PL854", "PL855"))
    assert clause == "NSTAPIPNO IN ('PL854','PL855')"


def test_C_where_clause_never_uses_pipe_name_or_like():
    clause = nsta_freespan.build_pipeline_number_where_clause(("PL854", "PL855"))
    assert "LIKE" not in clause
    assert "PIPE_NAME" not in clause


def test_manifest_entry_records_every_required_field(tmp_path: Path):
    raw_path = tmp_path / "current_pipeline_freespans.geojson"
    raw_path.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")

    from datetime import UTC, datetime

    acquisition = nsta_freespan.FreespanLayerAcquisition(
        registry_layer=nsta_freespan.CURRENT_REGISTRY_LAYER,
        source_title=nsta_freespan.CURRENT_FREESPAN_SOURCE_TITLE,
        service_url=nsta_freespan.CURRENT_FREESPAN_SERVICE_URL,
        query_where_clause="NSTAPIPNO IN ('PL854','PL855')",
        feature_count=0,
        raw_cache_path=raw_path,
        response_crs=None,
        schema_field_names=("FEATURE_ID", "NSTAPIPNO"),
        retrieved_at=datetime.now(UTC),
    )
    entry = nsta_freespan.build_acquisition_manifest_entry(acquisition)

    for required_key in (
        "source_agency",
        "registry_layer",
        "service_url",
        "query_where_clause",
        "retrieved_at_utc",
        "returned_feature_count",
        "raw_file_path",
        "sha256",
        "response_crs",
        "schema_field_names",
    ):
        assert required_key in entry, required_key
    assert entry["source_agency"] == "North Sea Transition Authority (NSTA)"
    assert len(entry["sha256"]) == 64  # a real sha256 hex digest
    assert entry["raw_unmodified"] is True

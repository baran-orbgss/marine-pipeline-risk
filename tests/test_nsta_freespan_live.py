"""Live smoke test against the real NSTA Pipeline Freespans ArcGIS services.

Excluded from the default `pytest` run (see `-m "not live"` in
pyproject.toml) so the normal suite never depends on NSTA availability. Run
explicitly with:

    uv run pytest -m live

Tests A and B (MAR-014C Section 21) verify the REAL, empirically-discovered
layer id -- both freespan services expose their data at layer id 1 (matching
`nsta.py`'s own pipeline-linear services), never at the ids 3/9 the ticket's
own (non-resolving) `data.nstauthority.co.uk` URL assumed. See
`providers/nsta_freespan.py`'s module docstring for the full discovery
narrative.
"""

from pathlib import Path

import pytest

from marine_engine.providers import nsta_freespan

pytestmark = pytest.mark.live


# --- A: current layer URL (the real, verified layer id -- not the ticket's assumed 3) --


def test_A_current_layer_is_reachable_at_the_real_layer_id():
    assert nsta_freespan.CURRENT_FREESPAN_SERVICE_URL.endswith("/FeatureServer/1")
    schema = nsta_freespan.fetch_layer_schema(nsta_freespan.CURRENT_FREESPAN_SERVICE_URL)
    field_names = {f["name"] for f in schema.get("fields", [])}
    assert nsta_freespan.PIPELINE_NUMBER_FIELD in field_names
    assert schema.get("geometryType") == "esriGeometryPolyline"


# --- B: removed layer URL (the real, verified layer id -- not the ticket's assumed 9) --


def test_B_removed_layer_is_reachable_at_the_real_layer_id():
    assert nsta_freespan.REMOVED_FREESPAN_SERVICE_URL.endswith("/FeatureServer/1")
    schema = nsta_freespan.fetch_layer_schema(nsta_freespan.REMOVED_FREESPAN_SERVICE_URL)
    field_names = {f["name"] for f in schema.get("fields", [])}
    assert nsta_freespan.PIPELINE_NUMBER_FIELD in field_names


def test_pl854_pl855_freespan_query_is_reproducibly_zero(tmp_path: Path):
    """Documents the real, current acquisition outcome: as of this ticket, NSTA's
    Pipeline Freespans registry has zero PL854/PL855 records in either layer --
    confirmed genuine (not a matching bug) via the distinct-NSTAPIPNO/LEGACY_ID/
    PIPE_NAME diagnostics in the module docstring. A future NSTA update could
    change this; this test documents today's real state, not a permanent
    assumption baked into the reconciliation logic itself.
    """

    report = nsta_freespan.ingest_freespan_registry(
        cache_dir=tmp_path / "raw", manifest_path=tmp_path / "manifest.json"
    )
    for acquisition in report.acquisitions:
        assert acquisition.feature_count == 0
        assert acquisition.raw_cache_path.exists()

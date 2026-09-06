"""Offline unit tests for marine_engine.validation.freespan_context_audit_map (MAR-015).

Small synthetic routes/sections only -- never the real PL854 route, never
network access. Lettered comments map to MAR-015 Section 27's required test
list (L/M/N/O here; A-K/P live in test_freespan_context_audit.py alongside
the core module they exercise).
"""

import inspect
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from marine_engine.scour import freespan_evidence
from marine_engine.validation import freespan_context_audit as fca
from marine_engine.validation import freespan_context_audit_map as fcam

WORKING_CRS = "EPSG:32631"

_BASE_SECTION_DEFAULTS = {
    "kp_start": "KP 0+000",
    "kp_end": "KP 1+000",
    "section_length_m": 1000.0,
    "observed_2018_event_count": 0,
    "observed_2018_total_length_m": 0.0,
    "observed_2018_max_length_m": 0.0,
    "observed_2018_max_height_m": 0.0,
    "observation_status": fca.NO_EVENT_STATUS,
    "tau_max_p95_sensitivity_min_pa": 0.2,
    "tau_max_p95_sensitivity_width_pa": 0.3,
    "current_reference_speed_p95_m_s": 0.3,
    "orbital_rms_p95_m_s": 0.2,
    "tau_max_p95_sensitivity_max_pa": 0.5,
    "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": 0.5,
    "p95_required_embedment_upper_class": "0.03",
    "local_relief_1000m_median_m": 0.5,
    "slope_500m_median_deg": 1.0,
    "mapped_250k_folk_class": "SAND",
    "nearest_valid_psa_d50_mm": 0.25,
}


def _make_section_df(rows: list[dict]) -> pd.DataFrame:
    records = []
    for i, overrides in enumerate(rows):
        record = dict(_BASE_SECTION_DEFAULTS)
        record["hydro_pair_id"] = f"pair_{chr(65 + i)}"
        record["segment_id"] = f"seg_{i}"
        record.update(overrides)
        records.append(record)
    return pd.DataFrame(records)


def _events_2018_df_for_small_multiple() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "2018-01",
                "canonical_chainage_min_m": 100.0,
                "canonical_chainage_max_m": 105.0,
            }
        ]
    )


def _audit_pieces_for_table(section_df: pd.DataFrame):
    contrasts_by_key = {
        key: fca.compute_descriptive_contrast(section_df, key)
        for key in fca.AUDIT_TABLE_FEATURE_KEYS
        if not fca.FEATURE_BY_KEY[key].is_categorical
    }
    categorical_audits_by_key = {
        key: fca.compute_categorical_audit(section_df, key)
        for key in fca.AUDIT_TABLE_FEATURE_KEYS
        if fca.FEATURE_BY_KEY[key].is_categorical or key == "mobility_capacity"
    }
    readiness = fca.build_evidence_readiness(categorical_audits_by_key)
    return contrasts_by_key, categorical_audits_by_key, readiness


# --- L: the observed span geometry is plotted at its TRUE length, never widened/buffered --


def test_L_render_never_buffers_or_widens_the_observed_span_geometry():
    source = inspect.getsource(fcam.render_freespan_evidence_resolution_audit_map)
    assert ".buffer(" not in source


def test_L_event_geometry_length_is_unchanged_after_rendering(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (500100.0, 5900000.0)])  # 100 m route
    events_df = pd.DataFrame([{"canonical_chainage_min_m": 40.0, "canonical_chainage_max_m": 45.0}])
    geometries = freespan_evidence.build_event_geometries(events_df, route)
    events_gdf = gpd.GeoDataFrame({"event_id": ["2018-01"]}, geometry=geometries, crs=WORKING_CRS)

    original_length_m = events_gdf.geometry.iloc[0].length
    assert original_length_m == pytest.approx(5.0)

    section_df = _make_section_df(
        [
            {
                "start_chainage_m": 0.0,
                "end_chainage_m": 100.0,
                "observation_status": fca.EVENT_PRESENT_STATUS,
                "observed_2018_event_count": 1,
            }
        ]
    )
    fcam.render_freespan_evidence_resolution_audit_map(
        events_2018_gdf=events_gdf,
        section_df=section_df,
        route=route,
        independent_section_count=1,
        output_path=tmp_path / "map1.png",
    )
    assert events_gdf.geometry.iloc[0].length == pytest.approx(original_length_m)


# --- M: the primary resolution-audit map renders a non-empty PNG --------------------------


def test_M_primary_resolution_audit_map_renders_a_nonempty_png(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (506000.0, 5900500.0)])
    events_df = pd.DataFrame([{"canonical_chainage_min_m": 40.0, "canonical_chainage_max_m": 45.0}])
    geometries = freespan_evidence.build_event_geometries(events_df, route)
    events_gdf = gpd.GeoDataFrame({"event_id": ["2018-01"]}, geometry=geometries, crs=WORKING_CRS)

    section_df = _make_section_df(
        [
            {
                "start_chainage_m": 0.0,
                "end_chainage_m": route.length,
                "observation_status": fca.EVENT_PRESENT_STATUS,
                "observed_2018_event_count": 1,
            }
        ]
    )
    output_path = tmp_path / "map1.png"

    result_path = fcam.render_freespan_evidence_resolution_audit_map(
        events_2018_gdf=events_gdf,
        section_df=section_df,
        route=route,
        independent_section_count=1,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    width_px, height_px = fcam.read_png_dimensions(output_path)
    assert width_px > 0
    assert height_px > 0


def test_render_map_handles_missing_background_raster_gracefully(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (501000.0, 5900000.0)])
    events_df = pd.DataFrame([{"canonical_chainage_min_m": 10.0, "canonical_chainage_max_m": 15.0}])
    geometries = freespan_evidence.build_event_geometries(events_df, route)
    events_gdf = gpd.GeoDataFrame({"event_id": ["2018-01"]}, geometry=geometries, crs=WORKING_CRS)
    section_df = _make_section_df([{"start_chainage_m": 0.0, "end_chainage_m": route.length}])

    result_path = fcam.render_freespan_evidence_resolution_audit_map(
        events_2018_gdf=events_gdf,
        section_df=section_df,
        route=route,
        independent_section_count=0,
        output_path=tmp_path / "map1.png",
        background_raster_path=tmp_path / "does_not_exist.tif",
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0


# --- N: the feature-context small-multiple renders a non-empty PNG (Panels A-D only) ------


def test_N_feature_context_small_multiple_renders_a_nonempty_png(tmp_path: Path):
    section_df = _make_section_df(
        [
            {"start_chainage_m": 0.0, "end_chainage_m": 500.0},
            {"start_chainage_m": 500.0, "end_chainage_m": 1000.0},
        ]
    )
    output_path = tmp_path / "map2.png"

    result_path = fcam.render_feature_context_small_multiple(
        section_df=section_df,
        events_2018_df=_events_2018_df_for_small_multiple(),
        total_length_m=1000.0,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    width_px, height_px = fcam.read_png_dimensions(output_path)
    assert width_px > 0
    assert height_px > 0


def test_N_small_multiple_never_draws_a_fitted_trend_or_correlation_line():
    source = inspect.getsource(fcam.render_feature_context_small_multiple)
    forbidden = ("polyfit", "linregress", "trendline", "corrcoef", "np.corr")
    for token in forbidden:
        assert token not in source


# --- O: the descriptive feature-audit table renders a non-empty PNG -----------------------


def test_O_feature_audit_table_renders_a_nonempty_png(tmp_path: Path):
    section_df = _make_section_df(
        [
            {"observation_status": fca.EVENT_PRESENT_STATUS, "observed_2018_event_count": 1},
            {"observation_status": fca.NO_EVENT_STATUS},
        ]
    )
    contrasts_by_key, categorical_audits_by_key, readiness = _audit_pieces_for_table(section_df)
    output_path = tmp_path / "map3.png"

    result_path = fcam.render_feature_audit_table(
        contrasts_by_key=contrasts_by_key,
        categorical_audits_by_key=categorical_audits_by_key,
        readiness=readiness,
        feature_keys=fca.AUDIT_TABLE_FEATURE_KEYS,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    width_px, height_px = fcam.read_png_dimensions(output_path)
    assert width_px > 0
    assert height_px > 0


def test_O_audit_table_uses_neutral_text_formatting_never_a_traffic_light_colour():
    source = inspect.getsource(fcam.render_feature_audit_table)
    _, _, body = source.partition('"""')
    _, _, body = body.partition('"""')  # drop the docstring, which itself names the anti-pattern
    forbidden = ("'green'", '"green"', "'red'", '"red"', "traffic")
    for token in forbidden:
        assert token not in body


# --- Shared: every rendered figure carries the explicit no-score statement ----------------


def test_no_score_statement_is_present_in_source_of_all_three_renderers():
    for func in (
        fcam.render_freespan_evidence_resolution_audit_map,
        fcam.render_feature_context_small_multiple,
        fcam.render_feature_audit_table,
    ):
        assert "NO_SCORE_STATEMENT" in inspect.getsource(func)

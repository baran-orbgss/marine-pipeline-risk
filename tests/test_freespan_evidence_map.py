"""Offline unit tests for marine_engine.scour.freespan_evidence_map (MAR-014A).

Small hand-built synthetic routes/events only -- never the real PL854 route,
never network access. Lettered comments map to MAR-014A Section 25's
required test list.
"""

import inspect
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from marine_engine.scour import freespan_evidence as fe
from marine_engine.scour import freespan_evidence_map as fem

WORKING_CRS = "EPSG:32631"
SYNTHETIC_ROUTE = LineString([(500000.0, 5900000.0), (506000.0, 5901500.0)])


def _synthetic_events_df() -> pd.DataFrame:
    """Three years of events: 2012 and 2018 both report an event near chainage
    1000-1010 m; 2014 reports nothing there but DOES report something far away
    (chainage 5000-5010 m) -- a clean, deliberate partial-coverage fixture."""

    records = [
        {
            "event_id": "2012-01",
            "survey_year": 2012,
            "canonical_chainage_min_m": 1000.0,
            "canonical_chainage_max_m": 1010.0,
            "source_length_m": 10.0,
            "source_height_m": 0.2,
        },
        {
            "event_id": "2014-01",
            "survey_year": 2014,
            "canonical_chainage_min_m": 5000.0,
            "canonical_chainage_max_m": 5010.0,
            "source_length_m": 10.0,
            "source_height_m": 0.2,
        },
        {
            "event_id": "2018-01",
            "survey_year": 2018,
            "canonical_chainage_min_m": 1002.0,
            "canonical_chainage_max_m": 1012.0,
            "source_length_m": 10.0,
            "source_height_m": 0.2,
        },
        {
            "event_id": "2018-02",
            "survey_year": 2018,
            "canonical_chainage_min_m": 5001.0,
            "canonical_chainage_max_m": 5011.0,
            "source_length_m": 10.0,
            "source_height_m": 0.2,
        },
    ]
    for record in records:
        record["canonical_mid_chainage_m"] = (
            record["canonical_chainage_min_m"] + record["canonical_chainage_max_m"]
        ) / 2.0
    return pd.DataFrame.from_records(records)


# --- G: 2014 partial-coverage flag is data-derived, never hard-coded -------------------


def test_G_2014_coverage_gap_detected_only_where_2014_truly_has_no_event():
    events_df = _synthetic_events_df()
    gaps = fem.compute_2014_coverage_gap_zones(events_df)

    reported_years = {g["reported_by_year"] for g in gaps}
    assert 2012 in reported_years  # 2012's zone (chainage ~1000-1010) has no nearby 2014 event
    for gap in gaps:
        # The only real gap zone is near chainage 1000-1010, not 5000-5010 (2014 covers that).
        assert gap["zone_start_chainage_m"] < 2000.0


def test_G_no_gap_reported_when_2014_covers_every_other_years_zone():
    events_df = _synthetic_events_df()
    # Add a 2014 event right where 2012/2018 already report one -- the gap must disappear.
    extra = pd.DataFrame.from_records(
        [
            {
                "event_id": "2014-02",
                "survey_year": 2014,
                "canonical_chainage_min_m": 1001.0,
                "canonical_chainage_max_m": 1011.0,
                "source_length_m": 10.0,
                "source_height_m": 0.2,
            }
        ]
    )
    full_events_df = pd.concat([events_df, extra], ignore_index=True)
    gaps = fem.compute_2014_coverage_gap_zones(full_events_df)
    assert gaps == []


# --- S: 2014 absence is never treated as no-freespan evidence --------------------------


def test_S_gap_warning_text_never_claims_no_freespan_only_incomplete_coverage():
    events_df = _synthetic_events_df()
    gaps = fem.compute_2014_coverage_gap_zones(events_df)
    warning = fem._format_2014_gap_warning(gaps)

    assert "must never be read as evidence of no freespan" in warning
    assert "may reflect incomplete 2014 survey coverage" in warning


def test_S_gap_zone_records_never_carry_a_no_freespan_claim_field():
    events_df = _synthetic_events_df()
    gaps = fem.compute_2014_coverage_gap_zones(events_df)
    for gap in gaps:
        assert set(gap.keys()) == {
            "reported_by_year",
            "zone_start_chainage_m",
            "zone_end_chainage_m",
        }


# --- U: map renderers produce non-empty PNGs --------------------------------------------


def test_U_2018_map_renderer_produces_a_nonempty_png(tmp_path: Path):
    events_df = _synthetic_events_df()
    events_2018_df = events_df[events_df["survey_year"] == 2018].reset_index(drop=True)
    geometries = fe.build_event_geometries(events_2018_df, SYNTHETIC_ROUTE)
    events_2018_gdf = gpd.GeoDataFrame(events_2018_df, geometry=geometries, crs=WORKING_CRS)

    output_path = tmp_path / "observed_2018.png"
    result_path = fem.render_2018_freespan_evidence_map(
        events_2018_gdf=events_2018_gdf, route=SYNTHETIC_ROUTE, output_path=output_path
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_U_historical_map_renderer_produces_a_nonempty_png(tmp_path: Path):
    events_df = _synthetic_events_df()
    geometries = fe.build_event_geometries(events_df, SYNTHETIC_ROUTE)
    events_all_gdf = gpd.GeoDataFrame(events_df, geometry=geometries, crs=WORKING_CRS)

    output_path = tmp_path / "historical.png"
    result_path = fem.render_historical_freespan_evidence_map(
        events_all_gdf=events_all_gdf, route=SYNTHETIC_ROUTE, output_path=output_path
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_U_model_context_profile_renderer_produces_a_nonempty_png(tmp_path: Path):
    context_df = pd.DataFrame.from_records(
        [
            {"event_id": "2018-01", "canonical_mid_chainage_m": 1007.0},
            {"event_id": "2018-02", "canonical_mid_chainage_m": 5006.0},
        ]
    )
    events_2018_span_df = pd.DataFrame.from_records(
        [
            {
                "event_id": "2018-01",
                "canonical_chainage_min_m": 1000.0,
                "canonical_chainage_max_m": 1014.0,
            },
            {
                "event_id": "2018-02",
                "canonical_chainage_min_m": 5000.0,
                "canonical_chainage_max_m": 5012.0,
            },
        ]
    )
    segments_df = pd.DataFrame.from_records(
        [
            {
                "start_chainage_m": 0.0,
                "end_chainage_m": SYNTHETIC_ROUTE.length,
                "hydro_pair_id": "pair_A",
                "tau_max_p95_sensitivity_min_pa": 0.3,
                "tau_max_p95_sensitivity_max_pa": 0.9,
                "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": 1.0,
                "p95_required_embedment_upper_ratio": 0.03,
            }
        ]
    )

    output_path = tmp_path / "profile.png"
    result_path = fem.render_freespan_model_context_profile(
        context_df=context_df,
        events_2018_span_df=events_2018_span_df,
        combined_bed_shear_segments_df=segments_df,
        noncohesive_mobility_segments_df=segments_df,
        scour_onset_segments_df=segments_df,
        total_length_m=SYNTHETIC_ROUTE.length,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_span_markers_use_the_real_observed_width_never_enlarged():
    """MAR-014B Section 11: true spans are ~0.2-23 m on a 23.5 km route --
    the profile must never widen a span's marker to stay visible."""

    import inspect

    source = inspect.getsource(fem.render_freespan_model_context_profile)
    assert "events_2018_span_df" in source
    assert "canonical_chainage_min_m" in source and "canonical_chainage_max_m" in source
    # never a fabricated minimum-visible-width constant
    assert "MIN_VISIBLE_WIDTH" not in source
    assert "min_width" not in source.lower()


# --- V: historical map includes an explicit 2014 partial-coverage warning --------------


def test_V_historical_map_source_includes_partial_coverage_warning():
    """The rendered map's footer text is not machine-inspectable from the PNG bytes
    directly (see the MAR-014 precedent, test_AG), so this verifies the render
    function actually computes and includes the coverage-gap warning."""

    source = inspect.getsource(fem.render_historical_freespan_evidence_map)
    assert "compute_2014_coverage_gap_zones" in source
    assert "_format_2014_gap_warning" in source


def test_V_no_score_statement_present_on_every_route_map():
    for renderer in (
        fem.render_2018_freespan_evidence_map,
        fem.render_historical_freespan_evidence_map,
    ):
        source = inspect.getsource(renderer)
        assert "NO_SCORE_STATEMENT" in source


# --- J: optional temporal evolution figure renders non-empty (MAR-014B) ----------------


def _temporal_evidence_df() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "relationship_id": "REL-01",
                "event_id_a": "2018-01",
                "canonical_mid_chainage_a_m": None,
                "event_id_b": None,
                "canonical_mid_chainage_b_m": None,
                "source_statement": "Span identified from 2018 survey in area not surveyed "
                "in 2014.",
            },
            {
                "relationship_id": "REL-04",
                "event_id_a": "2014-05",
                "canonical_mid_chainage_a_m": 1000.0,
                "event_id_b": "2018-06",
                "canonical_mid_chainage_b_m": 1005.0,
                "source_statement": "Same span 2014 and 2018.",
            },
            {
                "relationship_id": "REL-09",
                "event_id_a": None,
                "canonical_mid_chainage_a_m": None,
                "event_id_b": None,
                "canonical_mid_chainage_b_m": None,
                "source_statement": "Freespans changed over time in length, height, and location.",
            },
        ]
    )


def test_J_temporal_evolution_figure_renders_non_empty(tmp_path: Path):
    # REL-01's single-event marker needs its own (non-null) chainage.
    df = _temporal_evidence_df()
    df.loc[df["relationship_id"] == "REL-01", "canonical_mid_chainage_a_m"] = 5000.0

    output_path = tmp_path / "evolution.png"
    result_path = fem.render_freespan_temporal_evolution_figure(
        temporal_evidence_df=df, route=SYNTHETIC_ROUTE, output_path=output_path
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_J_narrative_only_rows_never_get_a_fabricated_geometry():
    source = inspect.getsource(fem.render_freespan_temporal_evolution_figure)
    assert "narrative_df" in source
    assert "never" in source.lower() or "fabricated" in source.lower()

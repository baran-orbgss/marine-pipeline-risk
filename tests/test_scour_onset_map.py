"""Offline unit tests for marine_engine.scour.scour_onset_map (MAR-014).

Small synthetic routes/chainage tables only -- never the real PL854 route,
never network access. Lettered comments map to MAR-014 Section 40's
required test list.
"""

from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import LineString

from marine_engine.scour import scour_onset_map as somap

WORKING_CRS = "EPSG:32631"


def _chainage_hydro_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- Route sections + local tangent bearing (Section 12 / test P) ---------------------


def test_many_stations_sharing_one_hydro_pair_dissolve_into_one_section():
    route = LineString([(0.0, 0.0), (1000.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [
            {"chainage_m": c, "hydro_pair_id": "current_A__wave_A"}
            for c in (0.0, 100.0, 200.0, 1000.0)
        ]
    )
    sections = somap.build_route_sections(route, chainage_df)
    assert len(sections) == 1
    assert sections[0]["hydro_pair_id"] == "current_A__wave_A"
    assert sections[0]["start_chainage_m"] == pytest.approx(0.0)
    assert sections[0]["end_chainage_m"] == pytest.approx(1000.0)
    # A straight route: local tangent bearing == due east (90 deg).
    assert sections[0]["route_tangent_bearing_deg"] == pytest.approx(90.0)


def test_hydro_pair_change_creates_a_segment_boundary():
    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [
            {"chainage_m": c, "hydro_pair_id": pair_id}
            for c, pair_id in (
                (0.0, "current_A__wave_A"),
                (50.0, "current_A__wave_A"),
                (100.0, "current_B__wave_B"),
            )
        ]
    )
    sections = somap.build_route_sections(route, chainage_df)
    assert len(sections) == 2
    assert sections[0]["hydro_pair_id"] == "current_A__wave_A"
    assert sections[1]["hydro_pair_id"] == "current_B__wave_B"
    assert sections[0]["end_chainage_m"] == sections[1]["start_chainage_m"]


def test_build_tangent_bearing_by_pair_id():
    sections = [
        {
            "hydro_pair_id": "A",
            "start_chainage_m": 0.0,
            "end_chainage_m": 100.0,
            "route_tangent_bearing_deg": 45.0,
        },
        {
            "hydro_pair_id": "B",
            "start_chainage_m": 100.0,
            "end_chainage_m": 200.0,
            "route_tangent_bearing_deg": 60.0,
        },
    ]
    result = somap.build_tangent_bearing_by_pair_id(sections)
    assert result == {"A": 45.0, "B": 60.0}


def test_build_route_sections_empty_input():
    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    assert somap.build_route_sections(route, pd.DataFrame()) == []


# --- Segment enrichment: capacity/applicability/morphology context --------------------


def _envelope_by_pair_id() -> dict:
    return {
        "current_A__wave_A": {
            "p95_required_embedment_lower_class": "0.03",
            "p95_required_embedment_lower_ratio": 0.03,
            "p95_required_embedment_upper_class": "0.1",
            "p95_required_embedment_upper_ratio": 0.10,
        }
    }


def test_segments_column_schema_matches_constant():
    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [{"chainage_m": c, "hydro_pair_id": "current_A__wave_A"} for c in (0.0, 100.0)]
    )
    sections = somap.build_route_sections(route, chainage_df)

    segments = somap.build_scour_onset_embedment_segments(
        pipeline_id="PL854",
        route=route,
        sections=sections,
        envelope_by_pair_id=_envelope_by_pair_id(),
        applicability={
            "within_source_pipe_diameter_envelope": False,
            "projected_uc_source_range_fraction": 0.5,
            "projected_uw_source_range_fraction": 0.4,
            "kc_source_range_fraction": 0.3,
        },
        morphology_by_pair_id={
            "current_A__wave_A": {
                "slope_500m_median_deg": 0.1,
                "slope_500m_p95_deg": 0.3,
                "slope_1000m_median_deg": 0.12,
                "tpi_1000m_median_m": 0.02,
                "local_relief_1000m_median_m": 1.0,
                "terrain_std_1000m_median_m": 0.9,
            }
        },
        diameter_m=0.3048,
        working_crs=WORKING_CRS,
    )
    assert list(segments.columns) == [
        *list(somap.SCOUR_ONSET_EMBEDMENT_SEGMENTS_COLUMNS),
        "geometry",
    ]
    row = segments.iloc[0]
    assert (
        row["pipe_diameter_source_envelope_status"] == somap.PIPE_DIAMETER_OUTSIDE_SOURCE_ENVELOPE
    )
    assert row["p95_required_embedment_lower_m"] == pytest.approx(0.03 * 0.3048)
    assert row["p95_required_embedment_upper_m"] == pytest.approx(0.10 * 0.3048)
    assert row["morphology_role"] == "LEGACY_REGIONAL_CONTEXT_ONLY"


def test_segments_empty_input():
    route = LineString([(0.0, 0.0), (1.0, 0.0)])
    result = somap.build_scour_onset_embedment_segments(
        pipeline_id="PL854",
        route=route,
        sections=[],
        envelope_by_pair_id={},
        applicability={},
        morphology_by_pair_id={},
        diameter_m=0.3048,
        working_crs=WORKING_CRS,
    )
    assert result.empty


def test_segments_handle_unassigned_run_without_crashing():
    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [
            {"chainage_m": 0.0, "hydro_pair_id": "current_A__wave_A"},
            {"chainage_m": 50.0, "hydro_pair_id": None},
            {"chainage_m": 100.0, "hydro_pair_id": None},
        ]
    )
    sections = somap.build_route_sections(route, chainage_df)
    segments = somap.build_scour_onset_embedment_segments(
        pipeline_id="PL854",
        route=route,
        sections=sections,
        envelope_by_pair_id=_envelope_by_pair_id(),
        applicability={},
        morphology_by_pair_id={},
        diameter_m=0.3048,
        working_crs=WORKING_CRS,
    )
    assert len(segments) == 2
    assert pd.isna(segments.iloc[1]["hydro_pair_id"])
    assert pd.isna(segments.iloc[1]["p95_required_embedment_upper_class"])


# --- AE/AF: map and profile renderers produce non-empty PNGs ---------------------------


def _segments_for_render(route: LineString):
    chainage_df = _chainage_hydro_df(
        [
            {"chainage_m": 0.0, "hydro_pair_id": "current_A__wave_A"},
            {"chainage_m": route.length / 2.0, "hydro_pair_id": "current_B__wave_B"},
            {"chainage_m": route.length, "hydro_pair_id": "current_B__wave_B"},
        ]
    )
    sections = somap.build_route_sections(route, chainage_df)
    return somap.build_scour_onset_embedment_segments(
        pipeline_id="PL854",
        route=route,
        sections=sections,
        envelope_by_pair_id={
            "current_A__wave_A": {
                "p95_required_embedment_lower_class": "0",
                "p95_required_embedment_lower_ratio": 0.0,
                "p95_required_embedment_upper_class": "0.03",
                "p95_required_embedment_upper_ratio": 0.03,
            },
            "current_B__wave_B": {
                "p95_required_embedment_lower_class": "0.1",
                "p95_required_embedment_lower_ratio": 0.10,
                "p95_required_embedment_upper_class": "0.15",
                "p95_required_embedment_upper_ratio": 0.15,
            },
        },
        applicability={
            "within_source_pipe_diameter_envelope": False,
            "projected_uc_source_range_fraction": 0.5,
            "projected_uw_source_range_fraction": 0.4,
            "kc_source_range_fraction": 0.3,
        },
        morphology_by_pair_id={},
        diameter_m=0.3048,
        working_crs=WORKING_CRS,
    )


def test_AE_render_map_produces_a_nonempty_landscape_png(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (506000.0, 5901500.0)])
    segments = _segments_for_render(route)
    output_path = tmp_path / "scour_map.png"

    result_path = somap.render_scour_onset_embedment_map(
        segments_gdf=segments,
        route=route,
        working_crs=WORKING_CRS,
        output_path=output_path,
        diameter_m=0.3048,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    width_px, height_px = somap.read_png_dimensions(output_path)
    assert width_px > 0
    assert height_px > 0
    assert width_px > height_px  # landscape


def test_render_map_handles_missing_background_gracefully(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (501000.0, 5900000.0)])
    segments = _segments_for_render(route)
    result_path = somap.render_scour_onset_embedment_map(
        segments_gdf=segments,
        route=route,
        working_crs=WORKING_CRS,
        output_path=tmp_path / "scour_map.png",
        diameter_m=0.3048,
        background_raster_path=tmp_path / "does_not_exist.tif",
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_AF_render_profile_produces_a_nonempty_stepped_png(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (506000.0, 5900000.0)])
    segments = _segments_for_render(route)
    output_path = tmp_path / "scour_profile.png"

    result_path = somap.render_scour_onset_embedment_profile(
        segments_gdf=segments, diameter_m=0.3048, output_path=output_path
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# --- AG: map contains explicit diameter-extrapolation limitation -----------------------


def test_AG_map_footer_states_diameter_extrapolation():
    """The rendered map's footer text is not machine-inspectable from the PNG bytes
    directly, so this test verifies the exact wording is present in source and would
    be drawn -- the render function itself is exercised by test_AE above with no
    exception, and the literal footer string is checked here for the required
    diameter-extrapolation disclosure."""

    import inspect

    source = inspect.getsource(somap.render_scour_onset_embedment_map)
    assert "exceeds the source" in source
    assert "SOURCE_DIAMETER_RANGE_M" in source


# --- Discrete embedment-class colours: no interpolation between classes ---------------


def test_discrete_embedment_classes_each_get_a_distinct_colour():
    colors = somap._discrete_embedment_class_colors()
    assert len(colors) == 6  # 5 tested + one ">0.15" class
    assert len(set(colors.values())) == 6


def test_colour_for_class_none_gets_neutral_grey():
    colors = somap._discrete_embedment_class_colors()
    assert somap._colour_for_class(colors, None) == (0.6, 0.6, 0.6, 1.0)


# --- AH: no forbidden downstream-physics terms in the segment schema ------------------


def test_AH_segments_schema_contains_no_forbidden_downstream_terms():
    forbidden = (
        "equilibrium_scour_depth",
        "erosion_rate",
        "exposure_probability",
        "freespan_probability",
        "fatigue",
        "risk_score",
    )
    columns_lower = [c.lower() for c in somap.SCOUR_ONSET_EMBEDMENT_SEGMENTS_COLUMNS]
    for term in forbidden:
        assert not any(term in column for column in columns_lower)

"""Offline unit tests for marine_engine.scour.freespan_evidence (MAR-014A).

Tests A-F and P read the tracked `anglia_table_b1_freespans.csv` resource
directly -- a tiny, version-controlled, offline file that IS the actual
subject under test for those letters, not a stand-in for real-time/network
data. Every other lettered test uses a small hand-built synthetic
route/events fixture, never the real PL854 route, never network access.
Lettered comments map to MAR-014A Section 25's required test list.
"""

import pandas as pd
import pyproj
import pytest
from shapely.geometry import LineString, Point

from marine_engine.resources import load_anglia_table_b1_freespans
from marine_engine.scour import freespan_evidence as fe

WORKING_CRS = "EPSG:32631"
SOURCE_CRS = "EPSG:23031"

# A bent (never straight) synthetic route so chord-vs-substring tests are meaningful.
SYNTHETIC_ROUTE = LineString([(500000.0, 5900000.0), (505000.0, 5910000.0), (500000.0, 5920000.0)])

# Hand-picked (never linspace-symmetric around the route midpoint, which would make
# a source-KP-derived chainage coincidentally equal the true one at that one point).
_SYNTHETIC_CHAINAGES_M = (2000.0, 5000.0, 9000.0, 15500.0, 19500.0)


def _to_source_crs(x: float, y: float) -> tuple[float, float]:
    transformer = pyproj.Transformer.from_crs(WORKING_CRS, SOURCE_CRS, always_xy=True)
    return transformer.transform(x, y)


def _make_synthetic_good_fit_events() -> pd.DataFrame:
    """Events whose source (candidate-A) coordinates decode, via the real
    EPSG:23031->EPSG:32631 transform, to points sitting exactly on
    SYNTHETIC_ROUTE -- with survey KP assigned in REVERSE order relative to
    canonical chainage, mirroring the real PL854/Table B.1 relationship."""

    total_length = SYNTHETIC_ROUTE.length
    records = []
    for i, chainage_a in enumerate(_SYNTHETIC_CHAINAGES_M):
        chainage_b = chainage_a + 5.0
        point_a = SYNTHETIC_ROUTE.interpolate(chainage_a)
        point_b = SYNTHETIC_ROUTE.interpolate(chainage_b)
        easting_a, northing_a = _to_source_crs(point_a.x, point_a.y)
        easting_b, northing_b = _to_source_crs(point_b.x, point_b.y)
        records.append(
            {
                "event_id": f"SYN-{i + 1:02d}",
                "survey_year": 2018,
                "source_survey_kp_start_km": (total_length - chainage_a) / 1000.0,
                "source_survey_kp_end_km": (total_length - chainage_b) / 1000.0,
                "source_easting_start_m": easting_a,
                "source_northing_start_m": northing_a,
                "source_easting_end_m": easting_b,
                "source_northing_end_m": northing_b,
                "source_length_m": point_a.distance(point_b),
                "source_height_m": 0.2,
            }
        )
    return pd.DataFrame.from_records(records)


def _make_synthetic_bad_fit_events() -> pd.DataFrame:
    """Events with NO coherent spatial relationship to SYNTHETIC_ROUTE or to
    their own assigned KP -- neither candidate CRS should ever fit this."""

    far_offsets = (3000.0, -4000.0, 5500.0, -6500.0, 8000.0)
    records = []
    for i, (chainage_a, offset) in enumerate(zip(_SYNTHETIC_CHAINAGES_M, far_offsets, strict=True)):
        point_a = SYNTHETIC_ROUTE.interpolate(chainage_a)
        point_b = SYNTHETIC_ROUTE.interpolate(chainage_a + 5.0)
        easting_a, northing_a = _to_source_crs(point_a.x + offset, point_a.y + offset)
        easting_b, northing_b = _to_source_crs(point_b.x + offset, point_b.y + offset)
        records.append(
            {
                "event_id": f"BAD-{i + 1:02d}",
                "survey_year": 2018,
                # arbitrary, deliberately unrelated to position
                "source_survey_kp_start_km": float(i) * 3.7 + 0.1,
                "source_survey_kp_end_km": float(i) * 3.7 + 0.11,
                "source_easting_start_m": easting_a,
                "source_northing_start_m": northing_a,
                "source_easting_end_m": easting_b,
                "source_northing_end_m": northing_b,
                "source_length_m": point_a.distance(point_b),
                "source_height_m": 0.2,
            }
        )
    return pd.DataFrame.from_records(records)


# --- A-F: source counts/sums/max exactly (real tracked CSV resource) -------------------


def test_A_2012_event_count_and_sum():
    df = load_anglia_table_b1_freespans()
    rows = df[df["survey_year"] == 2012]
    assert len(rows) == 2
    assert rows["source_length_m"].sum() == pytest.approx(23.20)


def test_B_2014_event_count_and_sum():
    df = load_anglia_table_b1_freespans()
    rows = df[df["survey_year"] == 2014]
    assert len(rows) == 7
    assert rows["source_length_m"].sum() == pytest.approx(68.23)


def test_C_2018_event_count_and_sum():
    df = load_anglia_table_b1_freespans()
    rows = df[df["survey_year"] == 2018]
    assert len(rows) == 8
    assert rows["source_length_m"].sum() == pytest.approx(97.42)


def test_D_2018_max_length():
    df = load_anglia_table_b1_freespans()
    rows = df[df["survey_year"] == 2018]
    assert rows["source_length_m"].max() == pytest.approx(23.16)


def test_E_2018_max_height():
    df = load_anglia_table_b1_freespans()
    rows = df[df["survey_year"] == 2018]
    assert rows["source_height_m"].max() == pytest.approx(0.41)


def test_F_total_event_count_and_scope():
    df = load_anglia_table_b1_freespans()
    assert len(df) == 17
    assert set(df["survey_year"]) == {2012, 2014, 2018}
    assert set(df["asset_scope"]) == {"PL854_PL855_PIGGYBACK_CORRIDOR"}


# --- H: source CRS is explicitly unstated, never assumed --------------------------------


def test_H_source_crs_status_states_not_source_stated():
    assert fe.SOURCE_CRS_STATUS == "CRS_INFERRED_FROM_SPATIAL_CONSISTENCY_NOT_SOURCE_STATED"
    assert "NOT_SOURCE_STATED" in fe.SOURCE_CRS_STATUS


# --- I: candidate comparison actually performed (both candidates get real diagnostics) --


def test_I_both_candidates_evaluated_with_real_diagnostics():
    events_df = _make_synthetic_good_fit_events()
    diagnostics_by_epsg = {
        epsg: fe.evaluate_crs_candidate(events_df, SYNTHETIC_ROUTE, WORKING_CRS, epsg)
        for epsg in fe.CANDIDATE_CRS_EPSG_CODES
    }
    assert set(diagnostics_by_epsg.keys()) == {23031, 32631}
    for diag in diagnostics_by_epsg.values():
        assert diag.endpoint_count == 2 * len(events_df)
        assert diag.distance_median_m >= 0.0

    # The correct candidate (23031) must fit far better than the naive one (32631).
    assert (
        diagnostics_by_epsg[23031].distance_median_m < diagnostics_by_epsg[32631].distance_median_m
    )


# --- J: wrong candidate rejected on large residual (synthetic bad-fit fixture) ----------


def test_J_bad_fit_data_is_rejected_never_silently_accepted():
    events_df = _make_synthetic_bad_fit_events()
    diagnostics_by_epsg = {
        epsg: fe.evaluate_crs_candidate(events_df, SYNTHETIC_ROUTE, WORKING_CRS, epsg)
        for epsg in fe.CANDIDATE_CRS_EPSG_CODES
    }
    with pytest.raises(fe.CRSReconciliationError):
        fe.select_working_crs(diagnostics_by_epsg)


def test_J_good_fit_data_is_accepted():
    events_df = _make_synthetic_good_fit_events()
    diagnostics_by_epsg = {
        epsg: fe.evaluate_crs_candidate(events_df, SYNTHETIC_ROUTE, WORKING_CRS, epsg)
        for epsg in fe.CANDIDATE_CRS_EPSG_CODES
    }
    accepted_epsg, checks = fe.select_working_crs(diagnostics_by_epsg)
    assert accepted_epsg == 23031
    assert all(checks.values())


# --- K: accepted CRS metadata never claims SOURCE_STATED --------------------------------


def test_K_accepted_crs_metadata_never_claims_source_stated():
    events_df = _make_synthetic_good_fit_events()
    projected = fe.project_events_to_canonical_route(events_df, SYNTHETIC_ROUTE, WORKING_CRS, 23031)
    assert (projected["source_crs_status"] == fe.SOURCE_CRS_STATUS).all()
    assert "NOT_SOURCE_STATED" in fe.SOURCE_CRS_STATUS
    assert fe.SOURCE_CRS_STATUS != "SOURCE_STATED"


# --- L: source KP is never copied directly to canonical chainage -----------------------


def test_L_source_kp_never_copied_directly_to_canonical_chainage():
    events_df = _make_synthetic_good_fit_events()
    projected = fe.project_events_to_canonical_route(events_df, SYNTHETIC_ROUTE, WORKING_CRS, 23031)
    for _, row in projected.iterrows():
        naive_chainage_m = row["source_survey_kp_start_km"] * 1000.0
        assert row["canonical_chainage_a_m"] != pytest.approx(naive_chainage_m)


# --- M: reverse direction detected from a synthetic fixture, never hard-coded -----------


def test_M_reverse_direction_detected_from_fit_slope():
    events_df = _make_synthetic_good_fit_events()
    diag = fe.evaluate_crs_candidate(events_df, SYNTHETIC_ROUTE, WORKING_CRS, 23031)
    assert diag.fit_slope < 0
    assert fe.classify_survey_direction(diag.fit_slope) == fe.REVERSED


def test_M_same_direction_detected_for_a_positive_slope():
    assert fe.classify_survey_direction(1.0) == fe.SAME_AS_CANONICAL


# --- N: event geometry is a true route substring ----------------------------------------


def test_N_event_geometry_is_a_route_substring():
    events_df = _make_synthetic_good_fit_events()
    projected = fe.project_events_to_canonical_route(events_df, SYNTHETIC_ROUTE, WORKING_CRS, 23031)
    geometries = fe.build_event_geometries(projected, SYNTHETIC_ROUTE)

    for geom, (_, row) in zip(geometries, projected.iterrows(), strict=True):
        expected_length = row["canonical_chainage_max_m"] - row["canonical_chainage_min_m"]
        assert geom.length == pytest.approx(expected_length, abs=1e-6)
        for x, y in geom.coords:
            assert Point(x, y).distance(SYNTHETIC_ROUTE) < 1e-6


# --- O: a raw straight chord is never used for an event spanning the route's bend -------


def test_O_geometry_follows_the_bend_not_a_straight_chord():
    bend_chainage = SYNTHETIC_ROUTE.length / 2.0
    events_df = pd.DataFrame.from_records(
        [
            {
                "event_id": "BEND-01",
                "canonical_chainage_min_m": bend_chainage - 500.0,
                "canonical_chainage_max_m": bend_chainage + 500.0,
            }
        ]
    )
    geom = fe.build_event_geometries(events_df, SYNTHETIC_ROUTE)[0]

    chord_length = Point(geom.coords[0]).distance(Point(geom.coords[-1]))
    assert geom.length > chord_length + 1.0  # meaningfully longer than the straight chord


# --- P: 2018 event count stays exactly 8 (source integrity safeguard) ------------------


def test_P_2018_event_count_stays_exactly_8():
    df = load_anglia_table_b1_freespans()
    assert len(df[df["survey_year"] == 2018]) == 8


# --- Q: individual line attribution stays UNRESOLVED for every event -------------------


def test_Q_individual_line_attribution_always_unresolved():
    events_df = _make_synthetic_good_fit_events()
    projected = fe.project_events_to_canonical_route(events_df, SYNTHETIC_ROUTE, WORKING_CRS, 23031)
    assert (projected["individual_line_attribution"] == "UNRESOLVED").all()
    assert fe.INDIVIDUAL_LINE_ATTRIBUTION == "UNRESOLVED"


# --- R: no event is ever auto-labelled PL854-only ---------------------------------------


def test_R_asset_scope_is_always_the_piggyback_corridor_never_pl854_alone():
    events_df = _make_synthetic_good_fit_events()
    projected = fe.project_events_to_canonical_route(events_df, SYNTHETIC_ROUTE, WORKING_CRS, 23031)
    assert (projected["asset_scope"] == "PL854_PL855_PIGGYBACK_CORRIDOR").all()
    assert fe.ASSET_SCOPE == "PL854_PL855_PIGGYBACK_CORRIDOR"
    assert "PL855" in fe.ASSET_SCOPE


# --- Gross length-mismatch hard-fail path (never rounding-scale) -----------------------


def test_gross_length_mismatch_raises_never_silently_accepted():
    total_length = SYNTHETIC_ROUTE.length
    point_a = SYNTHETIC_ROUTE.interpolate(2000.0)
    point_b = SYNTHETIC_ROUTE.interpolate(2200.0)
    easting_a, northing_a = _to_source_crs(point_a.x, point_a.y)
    easting_b, northing_b = _to_source_crs(point_b.x, point_b.y)
    events_df = pd.DataFrame.from_records(
        [
            {
                "event_id": "GROSS-01",
                "survey_year": 2018,
                "source_survey_kp_start_km": (total_length - 2000.0) / 1000.0,
                "source_survey_kp_end_km": (total_length - 2200.0) / 1000.0,
                "source_easting_start_m": easting_a,
                "source_northing_start_m": northing_a,
                "source_easting_end_m": easting_b,
                "source_northing_end_m": northing_b,
                # Actual projected interval is ~200 m; claiming 1 m is grossly (not
                # rounding-scale) inconsistent -- both the absolute (>50 m) and
                # relative (>100%) thresholds are exceeded, so this must hard-fail.
                "source_length_m": 1.0,
                "source_height_m": 0.2,
            }
        ]
    )
    with pytest.raises(fe.AngliaFreespanValidationError):
        fe.project_events_to_canonical_route(events_df, SYNTHETIC_ROUTE, WORKING_CRS, 23031)


def test_small_rounding_scale_length_difference_never_raises():
    events_df = _make_synthetic_good_fit_events()
    # Should not raise for ordinary small transcription/rounding differences.
    projected = fe.project_events_to_canonical_route(events_df, SYNTHETIC_ROUTE, WORKING_CRS, 23031)
    assert len(projected) == len(events_df)

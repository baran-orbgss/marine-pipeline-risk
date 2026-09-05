"""Offline unit tests for marine_engine.scour.nsta_freespan_reconciliation (MAR-014C).

Small hand-built synthetic routes/features/events only -- never the real
PL854 route or real NSTA data, never network access. Lettered comments map
to MAR-014C Section 21's required test list.
"""

import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from marine_engine.scour import nsta_freespan_reconciliation as nfr

WORKING_CRS = "EPSG:32631"
SYNTHETIC_ROUTE = LineString([(500000.0, 5900000.0), (505000.0, 5910000.0), (500000.0, 5920000.0)])


def _raw_feature(
    *,
    feature_id: str,
    nstapipno: str,
    registry_layer: str,
    point_working_crs,
    length_m=None,
    mxheight_m=None,
    survey_id=None,
    comments=None,
) -> dict:
    """A `parse_raw_freespan_features`-shaped record, built directly (bypassing
    GeoJSON parsing) with a geometry placed at a known working-CRS point,
    transformed back to WGS84 for the `_geometry_wgs84` field."""

    import geopandas as gpd

    point_wgs84 = gpd.GeoSeries([point_working_crs], crs=WORKING_CRS).to_crs("EPSG:4326").iloc[0]
    return {
        "registry_layer": registry_layer,
        "feature_id": feature_id,
        "nsta_pipeline_number": nstapipno,
        "label": None,
        "pipe_name": None,
        "rep_group": None,
        "freespanno": 1,
        "length_m": length_m,
        "mxheight_m": mxheight_m,
        "comments": comments,
        "crs_code": None,
        "crs_name": None,
        "start_date": None,
        "end_date": None,
        "end_reas": None,
        "upd_date": None,
        "upd_type": None,
        "upd_reas": None,
        "survey_id": survey_id,
        "_geometry_wgs84": point_wgs84,
    }


def _b1_event(event_id, survey_year, chainage_min, chainage_max, length_m, height_m) -> pd.Series:
    mid = (chainage_min + chainage_max) / 2.0
    return pd.Series(
        {
            "event_id": event_id,
            "survey_year": survey_year,
            "canonical_chainage_min_m": chainage_min,
            "canonical_chainage_max_m": chainage_max,
            "canonical_mid_chainage_m": mid,
            "canonical_mid_kp": f"KP {mid:.0f}",
            "source_length_m": length_m,
            "source_height_m": height_m,
            "asset_scope": "PL854_PL855_PIGGYBACK_CORRIDOR",
            "individual_line_attribution": "UNRESOLVED",
        }
    )


# --- D: raw NSTA source attributes preserved --------------------------------------------


def test_D_raw_source_attributes_preserved_verbatim():
    payload = {
        "features": [
            {
                "properties": {
                    "FEATURE_ID": "abc-123",
                    "NSTAPIPNO": "PL854",
                    "LABEL": "FS-1",
                    "PIPE_NAME": "ANGLIA A TO LOGGS",
                    "REP_GROUP": "OPERATOR X",
                    "FREESPANNO": 3,
                    "LENGTH_M": 12.5,
                    "MXHEIGHT_M": 0.3,
                    "COMMENTS": "some remark",
                    "CRS_CODE": 23031,
                    "CRS_NAME": "ED50 / UTM zone 31N",
                    "START_DATE": 1500000000000,
                    "END_DATE": None,
                    "END_REAS": None,
                    "UPD_DATE": 1500000000000,
                    "UPD_TYPE": "ADD",
                    "UPD_REAS": "NEW SURVEY",
                    "SURVEY_ID": "SURVEY-2020-01",
                },
                "geometry": {"type": "Point", "coordinates": [1.7, 53.37]},
            }
        ]
    }
    records = nfr.parse_raw_freespan_features(payload, nfr.CURRENT_REGISTRY_LAYER)
    assert len(records) == 1
    record = records[0]
    assert record["feature_id"] == "abc-123"
    assert record["nsta_pipeline_number"] == "PL854"
    assert record["freespanno"] == 3
    assert record["length_m"] == 12.5
    assert record["mxheight_m"] == 0.3
    assert record["comments"] == "some remark"
    assert record["survey_id"] == "SURVEY-2020-01"
    assert record["upd_type"] == "ADD"


# --- E: current vs removed registry status preserved ------------------------------------


def test_E_registry_layer_tag_preserved_current_vs_removed():
    current_payload = {
        "features": [
            {
                "properties": {"FEATURE_ID": "cur-1", "NSTAPIPNO": "PL854"},
                "geometry": {"type": "Point", "coordinates": [1.7, 53.37]},
            }
        ]
    }
    removed_payload = {
        "features": [
            {
                "properties": {"FEATURE_ID": "rem-1", "NSTAPIPNO": "PL854"},
                "geometry": {"type": "Point", "coordinates": [1.7, 53.37]},
            }
        ]
    }
    current_records = nfr.parse_raw_freespan_features(current_payload, nfr.CURRENT_REGISTRY_LAYER)
    removed_records = nfr.parse_raw_freespan_features(removed_payload, nfr.REMOVED_REGISTRY_LAYER)

    assert current_records[0]["registry_layer"] == "CURRENT_PIPELINE_FREESPANS"
    assert removed_records[0]["registry_layer"] == "REMOVED_PIPELINE_FREESPANS"

    gdf = nfr.build_nsta_freespan_registry_gdf(
        current_records + removed_records, SYNTHETIC_ROUTE, WORKING_CRS
    )
    assert set(gdf["registry_layer"]) == {
        "CURRENT_PIPELINE_FREESPANS",
        "REMOVED_PIPELINE_FREESPANS",
    }
    # Never a derived "is_disappeared"/"is_active" style column -- registry_layer
    # is lifecycle provenance only (Section 5).
    assert not any("disappear" in c.lower() or "is_active" in c.lower() for c in gdf.columns)


# --- F: route reconciliation checks actual geometry --------------------------------------


def test_F_on_route_feature_gets_near_zero_distance_and_correct_chainage():
    on_route_point = SYNTHETIC_ROUTE.interpolate(2000.0)
    record = _raw_feature(
        feature_id="f1",
        nstapipno="PL854",
        registry_layer=nfr.CURRENT_REGISTRY_LAYER,
        point_working_crs=on_route_point,
        length_m=10.0,
    )
    gdf = nfr.build_nsta_freespan_registry_gdf([record], SYNTHETIC_ROUTE, WORKING_CRS)

    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row["source_to_route_distance_m"] == pytest.approx(0.0, abs=1e-6)
    assert row["canonical_mid_chainage_m"] == pytest.approx(2000.0, abs=1e-6)
    assert bool(row["gross_spatial_inconsistency_flag"]) is False


def test_F_far_from_route_feature_is_hard_flagged():
    far_point = Point(
        SYNTHETIC_ROUTE.interpolate(2000.0).x + 20000.0, SYNTHETIC_ROUTE.interpolate(2000.0).y
    )
    record = _raw_feature(
        feature_id="f2",
        nstapipno="PL854",
        registry_layer=nfr.CURRENT_REGISTRY_LAYER,
        point_working_crs=far_point,
        length_m=10.0,
    )
    gdf = nfr.build_nsta_freespan_registry_gdf([record], SYNTHETIC_ROUTE, WORKING_CRS)

    row = gdf.iloc[0]
    assert row["source_to_route_distance_m"] > nfr.GROSS_SPATIAL_INCONSISTENCY_THRESHOLD_M
    assert bool(row["gross_spatial_inconsistency_flag"]) is True


def test_F_empty_records_gives_correctly_shaped_empty_gdf():
    gdf = nfr.build_nsta_freespan_registry_gdf([], SYNTHETIC_ROUTE, WORKING_CRS)
    assert len(gdf) == 0
    assert set(nfr.NSTA_FREESPAN_REGISTRY_COLUMNS).issubset(set(gdf.columns))
    assert gdf.crs == WORKING_CRS


# --- G: Table B.1 matching never uses nearest distance alone ---------------------------


def test_G_clear_spatial_match_with_no_attribute_data_is_never_strong():
    diagnostics = {
        "midpoint_separation_m": 1.0,  # extremely close
        "interval_overlap_fraction": 1.0,
        "length_difference_m": None,
        "length_difference_pct": None,
        "height_difference_m": None,
    }
    status = nfr.classify_candidate_match_status(diagnostics)
    assert status != nfr.STRONG_CROSS_SOURCE_MATCH
    assert status == nfr.INSUFFICIENT_ATTRIBUTES_FOR_MATCH


# --- H: strong match requires spatial + independent attribute support ------------------


def test_H_clear_spatial_plus_length_agreement_is_strong():
    diagnostics = {
        "midpoint_separation_m": 1.0,
        "interval_overlap_fraction": 1.0,
        "length_difference_m": 0.5,
        "length_difference_pct": 3.0,  # well within the agreement threshold
        "height_difference_m": None,
    }
    status = nfr.classify_candidate_match_status(diagnostics)
    assert status == nfr.STRONG_CROSS_SOURCE_MATCH


def test_H_far_spatial_with_attribute_agreement_is_never_strong():
    diagnostics = {
        "midpoint_separation_m": 5000.0,
        "interval_overlap_fraction": 0.0,
        "length_difference_m": 0.1,
        "length_difference_pct": 1.0,
        "height_difference_m": 0.01,
    }
    status = nfr.classify_candidate_match_status(diagnostics)
    assert status != nfr.STRONG_CROSS_SOURCE_MATCH


# --- I: multiple plausible NSTA records -> AMBIGUOUS, not an arbitrary winner ----------


def test_I_two_distinct_candidates_yield_ambiguous_not_a_winner():
    candidates = pd.DataFrame(
        [
            {"nsta_feature_id": "f1", "cross_source_match_status": nfr.POSSIBLE_CROSS_SOURCE_MATCH},
            {"nsta_feature_id": "f2", "cross_source_match_status": nfr.POSSIBLE_CROSS_SOURCE_MATCH},
        ]
    )
    status = nfr.summarize_event_match_status(candidates)
    assert status == nfr.AMBIGUOUS_MULTIPLE_NSTA_CANDIDATES


def test_I_single_candidate_is_never_ambiguous():
    candidates = pd.DataFrame(
        [{"nsta_feature_id": "f1", "cross_source_match_status": nfr.STRONG_CROSS_SOURCE_MATCH}]
    )
    status = nfr.summarize_event_match_status(candidates)
    assert status == nfr.STRONG_CROSS_SOURCE_MATCH


def test_I_no_candidates_is_no_match():
    status = nfr.summarize_event_match_status(
        pd.DataFrame(columns=list(nfr.MATCH_DIAGNOSTICS_COLUMNS))
    )
    assert status == nfr.NO_NSTA_MATCH


# --- J: coincident PL854/PL855 records detected explicitly -----------------------------


def test_J_coincident_pl854_pl855_records_detected():
    point = SYNTHETIC_ROUTE.interpolate(3000.0)
    record_854 = _raw_feature(
        feature_id="f854",
        nstapipno="PL854",
        registry_layer=nfr.CURRENT_REGISTRY_LAYER,
        point_working_crs=point,
        length_m=10.0,
        mxheight_m=0.3,
        survey_id="SURVEY-A",
    )
    record_855 = _raw_feature(
        feature_id="f855",
        nstapipno="PL855",
        registry_layer=nfr.CURRENT_REGISTRY_LAYER,
        point_working_crs=point,
        length_m=10.2,
        mxheight_m=0.32,
        survey_id="SURVEY-A",
    )
    gdf = nfr.build_nsta_freespan_registry_gdf(
        [record_854, record_855], SYNTHETIC_ROUTE, WORKING_CRS
    )

    piggyback_df = nfr.detect_piggyback_coincident_records(gdf)
    assert len(piggyback_df) == 1
    row = piggyback_df.iloc[0]
    assert row["classification"] == nfr.PIGGYBACK_COINCIDENT_FREESPAN_RECORDS
    assert bool(row["length_agreement"]) is True
    assert bool(row["survey_id_agreement"]) is True


def test_J_distant_pl854_pl855_records_are_never_coincident():
    point_a = SYNTHETIC_ROUTE.interpolate(1000.0)
    point_b = SYNTHETIC_ROUTE.interpolate(15000.0)
    record_854 = _raw_feature(
        feature_id="f854",
        nstapipno="PL854",
        registry_layer=nfr.CURRENT_REGISTRY_LAYER,
        point_working_crs=point_a,
        length_m=10.0,
    )
    record_855 = _raw_feature(
        feature_id="f855",
        nstapipno="PL855",
        registry_layer=nfr.CURRENT_REGISTRY_LAYER,
        point_working_crs=point_b,
        length_m=10.0,
    )
    gdf = nfr.build_nsta_freespan_registry_gdf(
        [record_854, record_855], SYNTHETIC_ROUTE, WORKING_CRS
    )
    piggyback_df = nfr.detect_piggyback_coincident_records(gdf)
    assert len(piggyback_df) == 0


# --- K: original Table B.1 corridor scope is never overwritten -------------------------


def test_K_original_asset_scope_and_attribution_preserved_in_output():
    event = _b1_event("2018-01", 2018, 1000.0, 1010.0, 10.0, 0.2)
    events_df = pd.DataFrame([event])
    evidence_df = nfr.build_2018_attribution_evidence(
        events_df, pd.DataFrame(columns=list(nfr.MATCH_DIAGNOSTICS_COLUMNS)), pd.DataFrame()
    )
    assert evidence_df.iloc[0]["original_asset_scope"] == "PL854_PL855_PIGGYBACK_CORRIDOR"
    assert evidence_df.iloc[0]["original_individual_line_attribution"] == "UNRESOLVED"


# --- L: no unmatched Table B.1 event becomes PL854 by default --------------------------


def test_L_no_candidates_never_defaults_to_pl854():
    status = nfr.derive_attribution_evidence_status(
        pd.DataFrame(columns=list(nfr.MATCH_DIAGNOSTICS_COLUMNS)), pd.DataFrame()
    )
    assert status == nfr.NO_NSTA_CROSS_SOURCE_MATCH
    assert status != nfr.PL854_SUPPORTED_BY_NSTA


# --- M: NSTA dates are descriptive, not assumed to be survey year ----------------------


def test_M_temporal_summary_never_infers_a_survey_year():
    gdf_columns = list(nfr.NSTA_FREESPAN_REGISTRY_COLUMNS)
    empty_gdf = __import__("geopandas").GeoDataFrame(
        columns=gdf_columns, geometry=[], crs=WORKING_CRS
    )
    summary = nfr.summarize_registry_temporal_context(empty_gdf)
    keys_lower = [k.lower() for k in summary]
    assert not any("survey_year" in k or "is_2018" in k for k in keys_lower)
    assert "start_date_range" in summary
    assert "end_date_range" in summary
    assert "upd_date_range" in summary


# --- N: crosswalk contains no score/probability field -----------------------------------


def test_N_match_diagnostics_columns_have_no_score_or_probability_field():
    forbidden = ("score", "probability", "confidence", "accuracy", "auc", "roc")
    for column in nfr.MATCH_DIAGNOSTICS_COLUMNS:
        for token in forbidden:
            assert token not in column.lower(), column


# --- O: 2018 attribution table always preserves original UNRESOLVED source field ------


def test_O_unresolved_preserved_even_when_a_strong_match_is_found():
    event = _b1_event("2018-01", 2018, 1000.0, 1010.0, 10.0, 0.2)
    events_df = pd.DataFrame([event])
    candidates = pd.DataFrame(
        [
            {
                "table_b1_event_id": "2018-01",
                "nsta_feature_id": "f1",
                "nsta_pipeline_number": "PL854",
                "cross_source_match_status": nfr.STRONG_CROSS_SOURCE_MATCH,
            }
        ]
    )
    evidence_df = nfr.build_2018_attribution_evidence(events_df, candidates, pd.DataFrame())

    row = evidence_df.iloc[0]
    assert row["attribution_evidence_status"] == nfr.PL854_SUPPORTED_BY_NSTA
    # The ORIGINAL Table B.1 field is never overwritten, even though a strong
    # match was found -- attribution evidence is additive, never a rewrite.
    assert row["original_individual_line_attribution"] == "UNRESOLVED"

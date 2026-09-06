"""Offline unit tests for marine_engine.validation.highres_seabed_survey_inventory
(MAR-016).

Small hand-built synthetic route/AOI/BGS-feature fixtures only -- never a
live network call, never the real PL854 route. Lettered comments map to
MAR-016 Section 24's required test list (A-J, M here; K/L live in
test_highres_seabed_survey_inventory_map.py alongside the map renderers).
"""

import inspect

import pandas as pd
import pyproj
from shapely.geometry import LineString, Point

from marine_engine.validation import highres_seabed_survey_inventory as hsi

WORKING_CRS = "EPSG:32631"
ROUTE = LineString([(420000.0, 5915000.0), (435000.0, 5916500.0)])
AOI_GEOMETRY = ROUTE.buffer(5000.0)

_TO_WGS84 = pyproj.Transformer.from_crs(WORKING_CRS, "EPSG:4326", always_xy=True).transform

ON_ROUTE_POINT = ROUTE.interpolate(0.5, normalized=True)
AOI_ONLY_POINT = Point(427500.0, 5918750.0)  # ~3 km off the route, inside the 5 km AOI buffer
FAR_AWAY_POINT = Point(427500.0, 5965750.0)  # ~50 km off the route, well outside the AOI


def _square_geom_from_working_xy(point: Point, half_size_m: float = 200.0) -> dict:
    x, y = point.x, point.y
    corners_working = [
        (x - half_size_m, y - half_size_m),
        (x - half_size_m, y + half_size_m),
        (x + half_size_m, y + half_size_m),
        (x + half_size_m, y - half_size_m),
        (x - half_size_m, y - half_size_m),
    ]
    return {
        "type": "Polygon",
        "coordinates": [[list(_TO_WGS84(cx, cy)) for cx, cy in corners_working]],
    }


def _make_feature(
    *,
    bgs_ref_no: str,
    decc_ref_no: str | None = None,
    site_svy_name: str = "Test Survey",
    originator: str = "Test Operator",
    svy_start_date: str = "2002-08-29T00:00:00",
    svy_end_date: str = "2002-09-03T00:00:00",
    additional_info: str = (
        "Data was collected using multibeam echo sounder, side scan sonar. Information about "
        "the survey was provided to the British Geological Survey in a legacy site survey report."
    ),
    custodian: str = "Test Operator",
    custodian_name: str = "Not Available",
    custodian_email: str = "Not Available",
    geometry: dict,
) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "site_svy_id": abs(hash(bgs_ref_no)) % 100000,
            "mdfileid_nerc_guid": "test-guid-0000",
            "bgs_ref_no": bgs_ref_no,
            "decc_ref_no": decc_ref_no,
            "site_svy_name": site_svy_name,
            "originator": originator,
            "contractor": "Test Contractor Ltd",
            "svy_start_date": svy_start_date,
            "svy_end_date": svy_end_date,
            "abstract": f"An oil and gas industry site survey. {additional_info}",
            "additional_info": additional_info,
            "custodian": custodian,
            "custodian_name": custodian_name,
            "custodian_email": custodian_email,
            "custodian_tel": "Not Available",
        },
        "geometry": geometry,
    }


def _row(feature: dict) -> dict:
    return hsi.build_survey_candidate_row(
        feature, route=ROUTE, aoi_geometry=AOI_GEOMETRY, working_crs=WORKING_CRS
    )


# --- A: acquisition year is never confused with a metadata retrieval/update date ----------


def test_A_acquisition_year_is_derived_only_from_svy_start_date():
    row_2002 = _row(
        _make_feature(
            bgs_ref_no="TEST0001",
            svy_start_date="2002-08-29T00:00:00",
            svy_end_date="2002-09-03T00:00:00",
            geometry=_square_geom_from_working_xy(ON_ROUTE_POINT),
        )
    )
    row_2018 = _row(
        _make_feature(
            bgs_ref_no="TEST0002",
            svy_start_date="2018-11-26T00:00:00",
            svy_end_date="2018-12-07T00:00:00",
            geometry=_square_geom_from_working_xy(ON_ROUTE_POINT),
        )
    )
    assert row_2002["acquisition_year"] == 2002
    assert row_2018["acquisition_year"] == 2018
    # This collection exposes no distinct "metadata last-updated" field -- never fabricated.
    assert row_2002["publication_or_update_date"] is None

    source = inspect.getsource(hsi.build_survey_candidate_row)
    assert "datetime.now" not in source
    assert "utcnow" not in source


# --- B: a shared block-number reference in the text never implies route overlap -----------


def test_B_shared_block_reference_text_alone_never_implies_route_overlap():
    shared_text = (
        "An oil and gas industry rig site survey. The block number traversed was 48/18. Data "
        "was collected using multibeam echo sounder. Information about the survey was provided "
        "to the British Geological Survey in a legacy site survey report."
    )
    on_route_row = _row(
        _make_feature(
            bgs_ref_no="ONROUTE01",
            additional_info=shared_text,
            geometry=_square_geom_from_working_xy(ON_ROUTE_POINT),
        )
    )
    far_away_row = _row(
        _make_feature(
            bgs_ref_no="FARAWAY01",
            decc_ref_no="GS_TEST",
            additional_info=shared_text,  # the SAME block-number text
            geometry=_square_geom_from_working_xy(FAR_AWAY_POINT),
        )
    )
    assert on_route_row["route_overlap_class"] == hsi.ROUTE_INTERSECTING
    assert far_away_row["route_overlap_class"] == hsi.NEARBY_NOT_AOI


# --- C: route intersection is computed from the real returned geometry --------------------


def test_C_route_overlap_reflects_real_geometry_not_bounding_box_proximity():
    # AOI_ONLY_POINT's bounding box sits well within the route's own overall bounding-box
    # envelope, yet the small square drawn there never actually touches the route LineString.
    row = _row(
        _make_feature(
            bgs_ref_no="NEARBUTOFF", geometry=_square_geom_from_working_xy(AOI_ONLY_POINT)
        )
    )
    assert row["route_distance_m"] > 0
    assert row["route_overlap_class"] != hsi.ROUTE_INTERSECTING


# --- D: AOI-only and route-intersecting classifications are real and distinct -------------


def test_D_route_intersecting_aoi_only_and_nearby_are_all_distinct_outcomes():
    route_row = _row(
        _make_feature(bgs_ref_no="R1", geometry=_square_geom_from_working_xy(ON_ROUTE_POINT))
    )
    aoi_only_row = _row(
        _make_feature(bgs_ref_no="A1", geometry=_square_geom_from_working_xy(AOI_ONLY_POINT))
    )
    nearby_row = _row(
        _make_feature(bgs_ref_no="N1", geometry=_square_geom_from_working_xy(FAR_AWAY_POINT))
    )
    assert route_row["route_overlap_class"] == hsi.ROUTE_INTERSECTING
    assert aoi_only_row["route_overlap_class"] == hsi.AOI_INTERSECTING_NOT_ROUTE
    assert nearby_row["route_overlap_class"] == hsi.NEARBY_NOT_AOI
    assert route_row["route_coverage_fraction"] is not None
    assert aoi_only_row["route_coverage_fraction"] is None
    assert nearby_row["route_coverage_fraction"] is None


# --- E: a metadata-only survey never becomes OPEN_DIRECT_DOWNLOAD/OPEN_API_DATA ------------


def test_E_metadata_only_survey_never_becomes_open_access():
    for custodian_name, custodian_email in (
        ("Not Available", "Not Available"),
        ("Jane Smith", "jane.smith@example.com"),
    ):
        row = _row(
            _make_feature(
                bgs_ref_no="ACCESSTEST",
                custodian_name=custodian_name,
                custodian_email=custodian_email,
                geometry=_square_geom_from_working_xy(ON_ROUTE_POINT),
            )
        )
        assert row["access_class"] == hsi.METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED
        assert row["access_class"] not in (hsi.OPEN_DIRECT_DOWNLOAD, hsi.OPEN_API_DATA)


# --- F: MBES equipment metadata never becomes an assumed downloadable data type ------------


def test_F_equipment_present_never_implies_actual_downloadable_data():
    row = _row(
        _make_feature(
            bgs_ref_no="EQUIPTEST",
            additional_info=(
                "Data was collected using multibeam echo sounder, side scan sonar, sub bottom "
                "profiler. Information about the survey was provided to the British Geological "
                "Survey in a legacy site survey report."
            ),
            geometry=_square_geom_from_working_xy(ON_ROUTE_POINT),
        )
    )
    assert "MBES" in row["equipment"]
    assert "SIDESCAN_SONAR" in row["equipment"]
    assert row["actual_available_data_types"] == ()


# --- G: endpoint-only / partial coverage never becomes a full pipeline-scale gap close -----


def test_G_route_intersecting_metadata_only_is_never_classified_as_closing_the_gap():
    small_coverage_row = _row(
        _make_feature(bgs_ref_no="SMALLCOV", geometry=_square_geom_from_working_xy(ON_ROUTE_POINT))
    )
    assert small_coverage_row["route_overlap_class"] == hsi.ROUTE_INTERSECTING
    assert small_coverage_row["morphology_gap_relevance_class"] == hsi.PARTIAL_ENDPOINT_CONTEXT_ONLY

    # Even a LARGE nominal coverage fraction must not upgrade the classification while
    # access remains metadata-only (Section 11's explicit warning).
    assert (
        hsi.classify_morphology_gap_relevance(
            route_overlap_class=hsi.ROUTE_INTERSECTING,
            access_class=hsi.METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED,
            route_coverage_fraction=0.95,
        )
        == hsi.PARTIAL_ENDPOINT_CONTEXT_ONLY
    )
    assert (
        hsi.classify_morphology_gap_relevance(
            route_overlap_class=hsi.ROUTE_INTERSECTING,
            access_class=hsi.OPEN_DIRECT_DOWNLOAD,
            route_coverage_fraction=0.95,
        )
        == hsi.CAN_ADDRESS_PIPELINE_SCALE_MORPHOLOGY_GAP
    )


# --- H: an unknown vertical datum is preserved as unknown, never silently defaulted --------


def test_H_vertical_datum_status_is_always_explicitly_unknown():
    row = _row(
        _make_feature(bgs_ref_no="DATUMTEST", geometry=_square_geom_from_working_xy(ON_ROUTE_POINT))
    )
    assert row["vertical_datum_status"] == "unknown"


# --- I: the Fugro 2018 dossier never claims open raw data without an actual file found -----


def test_I_fugro_dossier_never_claims_open_data_when_none_was_found():
    empty_survey_df = pd.DataFrame(columns=list(hsi.SURVEY_INVENTORY_COLUMNS))
    dossier = hsi.build_fugro_2018_recovery_dossier(
        survey_inventory_df=empty_survey_df, retrieved_at_utc="2026-09-06T00:00:00+00:00"
    )
    assert dossier["final_status"] != hsi.OPEN_DATA_FOUND
    assert dossier["final_status"] in (
        hsi.OPEN_DATA_FOUND,
        hsi.METADATA_FOUND_RAW_DATA_NOT_OPEN,
        hsi.REPORT_EVIDENCE_ONLY,
        hsi.ACCESS_PATH_REQUIRES_CUSTODIAN_REQUEST,
    )
    lowered = dossier["raw_data_availability_status"].lower()
    assert "no mbes grid, xyz soundings, geotiff, or shapefile" in lowered


# --- J: regional/analog datasets never become PL854 canonical evidence --------------------


def test_J_analog_datasets_are_structurally_separate_from_pl854_survey_evidence():
    real_survey_ids = {"GB02SS0001", "GB02SS0003", "GB03SS0002", "CS03SS0003", "ZE18SS0001"}
    for analog in hsi.ANALOG_DATASET_REGISTRY:
        assert analog["classification"] == hsi.METHOD_DEVELOPMENT_ANALOG_ONLY
        assert analog["name"] not in real_survey_ids
        # An analog registry entry has no `survey_id`/geometry -- it cannot be merged into
        # `build_survey_inventory_table`'s output, which only ever consumes real BGS features.
        assert "survey_id" not in analog
        assert "geometry" not in analog


# --- M: no output anywhere contains a forbidden risk/probability/susceptibility term -------


def test_M_no_output_contains_a_forbidden_score_or_probability_term():
    row = _row(
        _make_feature(bgs_ref_no="SCORETEST", geometry=_square_geom_from_working_xy(ON_ROUTE_POINT))
    )
    survey_df = pd.DataFrame([row])[list(hsi.SURVEY_INVENTORY_COLUMNS)]
    dossier = hsi.build_fugro_2018_recovery_dossier(
        survey_inventory_df=survey_df, retrieved_at_utc="2026-09-06T00:00:00+00:00"
    )
    access_gap = hsi.build_access_gap_report(survey_inventory_df=survey_df, fugro_dossier=dossier)

    forbidden = ("risk_score", "probability", "susceptibility", "prediction_accuracy")
    for column in survey_df.columns:
        for token in forbidden:
            assert token not in column.lower(), column
    import json

    dumped = json.dumps({"dossier": dossier, "access_gap": access_gap}, default=str).lower()
    for token in ("risk_score", "susceptibility", "prediction_accuracy"):
        assert token not in dumped, token

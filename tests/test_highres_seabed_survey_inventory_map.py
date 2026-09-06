"""Offline unit tests for marine_engine.validation.highres_seabed_survey_inventory_map
(MAR-016).

Small synthetic routes/footprints only -- never the real PL854 route, never
network access. Lettered comments map to MAR-016 Section 24's required test
list (K/L here; A-J/M live in test_highres_seabed_survey_inventory.py
alongside the core module they exercise).
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from marine_engine.validation import highres_seabed_survey_inventory as hsi
from marine_engine.validation import highres_seabed_survey_inventory_map as hsim

WORKING_CRS = "EPSG:32631"
ROUTE = LineString([(420000.0, 5915000.0), (435000.0, 5916500.0)])
AOI_GEOMETRY = ROUTE.buffer(5000.0)


def _survey_row(survey_id: str, *, route_overlap_class: str, access_class: str, year: int) -> dict:
    return {
        "survey_id": survey_id,
        "decc_ref_no": None,
        "title": f"Test survey {survey_id}",
        "operator": "Test Operator",
        "custodian": "Test Operator",
        "acquisition_start": pd.Timestamp(f"{year}-01-01"),
        "acquisition_end": pd.Timestamp(f"{year}-01-10"),
        "acquisition_year": year,
        "publication_or_update_date": None,
        "source_catalogue": hsi.SOURCE_CATALOGUE_BGS,
        "footprint_source": "test",
        "route_overlap_class": route_overlap_class,
        "route_distance_m": 0.0 if route_overlap_class == hsi.ROUTE_INTERSECTING else 1000.0,
        "route_coverage_fraction": 0.1 if route_overlap_class == hsi.ROUTE_INTERSECTING else None,
        "footprint_rectangularity": 0.97,
        "footprint_likely_licensed_block_extent": True,
        "equipment": ("MBES", "SIDESCAN_SONAR"),
        "actual_available_data_types": (),
        "access_class": access_class,
        "download_or_enquiry_reference": "Contact Test Operator.",
        "crs_status": "Published WGS84; reprojected for intersection only.",
        "vertical_datum_status": "unknown",
        "stated_resolution": None,
        "morphology_gap_relevance_class": hsi.PARTIAL_ENDPOINT_CONTEXT_ONLY,
        "limitations": "test limitation",
        "survey_footprint_metadata_does_not_imply_data_custody": True,
    }


def _survey_inventory_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _survey_row(
                "R1",
                route_overlap_class=hsi.ROUTE_INTERSECTING,
                access_class=hsi.METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED,
                year=2002,
            ),
            _survey_row(
                "A1",
                route_overlap_class=hsi.AOI_INTERSECTING_NOT_ROUTE,
                access_class=hsi.METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED,
                year=2003,
            ),
            _survey_row(
                "N1",
                route_overlap_class=hsi.NEARBY_NOT_AOI,
                access_class=hsi.METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED,
                year=2018,
            ),
        ]
    )[list(hsi.SURVEY_INVENTORY_COLUMNS)]


def _footprints_gdf() -> gpd.GeoDataFrame:
    on_route = ROUTE.interpolate(0.5, normalized=True).buffer(500.0)
    aoi_only = Point(427500.0, 5918750.0).buffer(500.0)
    nearby = Point(427500.0, 5965750.0).buffer(500.0)
    return gpd.GeoDataFrame(
        {"survey_id": ["R1", "A1", "N1"]}, geometry=[on_route, aoi_only, nearby], crs=WORKING_CRS
    )


# --- K: the coverage map renders a non-empty PNG -------------------------------------------


def test_K_coverage_map_renders_a_nonempty_png(tmp_path: Path):
    output_path = tmp_path / "coverage_map.png"

    result_path = hsim.render_survey_inventory_coverage_map(
        survey_inventory_df=_survey_inventory_df(),
        footprints_gdf=_footprints_gdf(),
        route=ROUTE,
        aoi_geometry=AOI_GEOMETRY,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    width_px, height_px = hsim.read_png_dimensions(output_path)
    assert width_px > 0
    assert height_px > 0


def test_coverage_map_handles_missing_background_raster_gracefully(tmp_path: Path):
    result_path = hsim.render_survey_inventory_coverage_map(
        survey_inventory_df=_survey_inventory_df(),
        footprints_gdf=_footprints_gdf(),
        route=ROUTE,
        aoi_geometry=AOI_GEOMETRY,
        output_path=tmp_path / "coverage_map.png",
        background_raster_path=tmp_path / "does_not_exist.tif",
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0


# --- L: the timeline renders a non-empty PNG -----------------------------------------------


def test_L_timeline_renders_a_nonempty_png(tmp_path: Path):
    output_path = tmp_path / "timeline.png"

    result_path = hsim.render_seabed_data_timeline(
        survey_inventory_df=_survey_inventory_df(), output_path=output_path
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    width_px, height_px = hsim.read_png_dimensions(output_path)
    assert width_px > 0
    assert height_px > 0


def test_L_timeline_renders_with_a_single_candidate(tmp_path: Path):
    single_df = _survey_inventory_df().iloc[[0]]
    result_path = hsim.render_seabed_data_timeline(
        survey_inventory_df=single_df, output_path=tmp_path / "timeline_single.png"
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0


# --- Shared: never colour by risk/morphology, never a fitted trend ------------------------


def test_coverage_map_never_colours_by_risk_or_morphology():
    import inspect

    source = inspect.getsource(hsim.render_survey_inventory_coverage_map)
    _, _, body = source.partition('"""')
    _, _, body = body.partition('"""')  # drop the docstring, which itself names the anti-pattern
    forbidden = ("risk_score", "susceptibility_score", "morphology_score", "cmap=")
    for token in forbidden:
        assert token not in body.lower()
    # Colour is only ever looked up from the fixed, categorical overlap-class dict.
    assert "_OVERLAP_CLASS_COLORS.get(" in body

"""Offline unit tests for marine_engine.scour.nsta_freespan_reconciliation_map (MAR-014C).

Small hand-built synthetic routes/events only -- never the real PL854 route,
never network access. Lettered comment maps to MAR-014C Section 21's
required test list.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from marine_engine.scour import nsta_freespan_reconciliation as nfr
from marine_engine.scour import nsta_freespan_reconciliation_map as nfrm

WORKING_CRS = "EPSG:32631"
SYNTHETIC_ROUTE = LineString([(500000.0, 5900000.0), (506000.0, 5901500.0)])


def _events_2018_gdf() -> gpd.GeoDataFrame:
    from shapely.ops import substring

    df = pd.DataFrame.from_records(
        [
            {
                "event_id": "2018-01",
                "canonical_chainage_min_m": 1000.0,
                "canonical_chainage_max_m": 1010.0,
            },
            {
                "event_id": "2018-02",
                "canonical_chainage_min_m": 3000.0,
                "canonical_chainage_max_m": 3020.0,
            },
        ]
    )
    geometries = [
        substring(
            SYNTHETIC_ROUTE,
            row["canonical_chainage_min_m"],
            row["canonical_chainage_max_m"],
            normalized=False,
        )
        for _, row in df.iterrows()
    ]
    return gpd.GeoDataFrame(df, geometry=geometries, crs=WORKING_CRS)


# --- P: map renders non-empty ------------------------------------------------------------


def test_P_reconciliation_map_renders_non_empty_with_no_nsta_records(tmp_path: Path):
    events_2018_gdf = _events_2018_gdf()
    empty_registry_gdf = gpd.GeoDataFrame(
        columns=list(nfr.NSTA_FREESPAN_REGISTRY_COLUMNS), geometry=[], crs=WORKING_CRS
    )

    output_path = tmp_path / "reconciliation.png"
    result_path = nfrm.render_nsta_table_b1_reconciliation_map(
        events_2018_gdf=events_2018_gdf,
        nsta_registry_gdf=empty_registry_gdf,
        piggyback_df=pd.DataFrame(),
        route=SYNTHETIC_ROUTE,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_P_reconciliation_map_renders_non_empty_with_nsta_records(tmp_path: Path):
    events_2018_gdf = _events_2018_gdf()
    point = SYNTHETIC_ROUTE.interpolate(1500.0)
    registry_gdf = gpd.GeoDataFrame(
        [
            {
                "registry_layer": "CURRENT_PIPELINE_FREESPANS",
                "feature_id": "f1",
                "nsta_pipeline_number": "PL854",
                "canonical_mid_chainage_m": 1500.0,
            }
        ],
        geometry=[point],
        crs=WORKING_CRS,
    )

    output_path = tmp_path / "reconciliation_with_data.png"
    result_path = nfrm.render_nsta_table_b1_reconciliation_map(
        events_2018_gdf=events_2018_gdf,
        nsta_registry_gdf=registry_gdf,
        piggyback_df=pd.DataFrame(),
        route=SYNTHETIC_ROUTE,
        output_path=output_path,
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_P_attribution_crosswalk_figure_renders_non_empty(tmp_path: Path):
    attribution_evidence_df = pd.DataFrame.from_records(
        [
            {
                "event_id": "2018-01",
                "canonical_mid_kp": "KP 1+005",
                "source_length_m": 10.0,
                "source_height_m": 0.2,
                "cross_source_match_status": nfr.NO_NSTA_MATCH,
                "matched_nsta_pipeline_numbers": [],
            },
            {
                "event_id": "2018-02",
                "canonical_mid_kp": "KP 3+010",
                "source_length_m": 12.0,
                "source_height_m": 0.3,
                "cross_source_match_status": nfr.STRONG_CROSS_SOURCE_MATCH,
                "matched_nsta_pipeline_numbers": ["PL854"],
            },
        ]
    )
    match_diagnostics_df = pd.DataFrame(columns=list(nfr.MATCH_DIAGNOSTICS_COLUMNS))

    output_path = tmp_path / "crosswalk.png"
    result_path = nfrm.render_2018_attribution_crosswalk_figure(
        attribution_evidence_df=attribution_evidence_df,
        match_diagnostics_df=match_diagnostics_df,
        output_path=output_path,
    )
    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

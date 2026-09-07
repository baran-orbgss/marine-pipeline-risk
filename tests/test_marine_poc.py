"""Offline unit tests for marine_engine.evidence_atlas.poc (MAR-019).

Small hand-built synthetic tables and tiny synthetic GeoDataFrames only --
never the real PL854 outputs, never network access. Lettered test names map
to MAR-019 Section 27's required test list (A-O).
"""

from __future__ import annotations

import inspect
import socket

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from marine_engine.evidence_atlas import maps, poc, report

CRS = "EPSG:32631"


def _demonstrated_items(blocks: list[dict]) -> list[str]:
    for block in blocks:
        if block["type"] == "heading" and block["text"] == "What this POC demonstrates today":
            idx = blocks.index(block)
            return blocks[idx + 1]["items"]
    raise AssertionError("'What this POC demonstrates today' section not found")


# --- A: no scientific values change from MAR-018 --------------------------------------------


def test_A_poc_module_never_imports_the_scientific_core():
    """`poc.py` has zero dependency on `core.py` (the module that builds
    section-evidence scientific values) -- by construction it can only READ
    already-persisted MAR-018 outputs, never recompute them."""

    source = inspect.getsource(poc)
    assert "evidence_atlas import core" not in source
    assert "evidence_atlas.core" not in source


def test_A2_reading_persisted_section_evidence_is_a_pure_round_trip(tmp_path):
    original = pd.DataFrame({"segment_id": [0, 1], "combined_tau_max_p95_upper_pa": [0.5, 0.8]})
    path = tmp_path / "pl854_section_evidence.parquet"
    original.to_parquet(path, index=False)

    reloaded = pd.read_parquet(path)

    pd.testing.assert_frame_equal(original, reloaded)


# --- B: no new hazard score exists ------------------------------------------------------------


def test_B_no_fused_score_in_poc_content():
    # Note: bare "susceptibility"/"screening" nouns are legitimate here -- the ticket itself
    # requires naming FUTURE hazard classes like "freespan susceptibility" (Sections 17-18).
    # What must never appear is an actual computed metric/score pattern.
    forbidden_patterns = (
        "susceptibility score",
        "susceptibility index",
        "risk score",
        "confidence score",
        "priority score",
        "probability of",
        "hotspot",
    )
    overview_text = report.render_blocks_markdown(poc.build_poc_overview_blocks())
    guide_text = poc.build_review_guide_markdown()
    for text in (overview_text, guide_text):
        for fragment in forbidden_patterns:
            assert fragment not in text.lower(), fragment


# --- C: atlas route title does not assert an unverified physical direction ------------------


def test_C_route_corridor_label_is_non_directional():
    assert "->" not in maps._ROUTE_CORRIDOR_LABEL
    assert "→" not in maps._ROUTE_CORRIDOR_LABEL
    assert "Anglia A -> LOGGS" not in maps._ROUTE_CORRIDOR_LABEL
    assert "corridor" in maps._ROUTE_CORRIDOR_LABEL.lower()


# --- D: true freespan widths remain unchanged ------------------------------------------------


def test_D_evidence_strip_band1_still_uses_true_chainage_columns():
    source = inspect.getsource(maps.render_evidence_strip)
    assert '"canonical_chainage_min_m"' in source
    assert '"canonical_chainage_max_m"' in source


# --- E: visibility markers do not alter canonical GIS geometry ------------------------------


def test_E_visibility_markers_never_mutate_freespan_geometry(tmp_path):
    freespans_gdf = gpd.GeoDataFrame(
        {
            "source_length_m": [5.0],
            "source_height_m": [0.2],
            "canonical_chainage_min_m": [100.0],
            "canonical_chainage_max_m": [105.0],
        },
        geometry=[LineString([(100.0, 0.0), (105.0, 0.0)])],
        crs=CRS,
    )
    original_wkt = freespans_gdf.geometry.iloc[0].wkt

    section_df = pd.DataFrame(
        {
            "segment_id": [0],
            "start_chainage_m": [0.0],
            "end_chainage_m": [200.0],
            "kp_start": ["KP 0+000"],
            "kp_end": ["KP 0+200"],
            "combined_tau_max_p95_lower_pa": [0.2],
            "combined_tau_max_p95_upper_pa": [0.5],
            "mobility_capacity_p95_d50_mm": [0.5],
            "p95_required_embedment_upper_class": ["0.03"],
            "local_relief_1000m_median_m": [3.0],
            "mapped_250k_folk_class": ["gS"],
        }
    )
    maps.render_evidence_strip(
        section_df=section_df,
        freespans_2018_gdf=freespans_gdf,
        output_path=tmp_path / "strip.png",
    )

    assert freespans_gdf.geometry.iloc[0].wkt == original_wkt
    assert freespans_gdf["canonical_chainage_min_m"].iloc[0] == 100.0
    assert freespans_gdf["canonical_chainage_max_m"].iloc[0] == 105.0


# --- F: report separates freespan spatial evidence from exposure aggregate evidence ---------


def test_F_report_separates_freespan_and_exposure_evidence():
    section_df = pd.DataFrame(
        {
            "segment_id": [0, 1],
            "kp_start": ["KP 0+000", "KP 0+500"],
            "kp_end": ["KP 0+500", "KP 1+000"],
            "observed_2018_freespan_count": [1, 0],
            "observed_2018_freespan_total_length_m": [5.0, 0.0],
            "observed_2018_freespan_max_length_m": [5.0, None],
            "observed_2018_freespan_max_height_m": [0.2, None],
            "mobility_capacity_p95_d50_mm": [0.5, 1.0],
            "p95_required_embedment_upper_class": ["0.03", "0.03"],
        }
    )
    freespans_gdf = gpd.GeoDataFrame(
        {"source_length_m": [5.0], "source_height_m": [0.2]},
        geometry=[LineString([(0.0, 0.0), (5.0, 0.0)])],
        crs=CRS,
    )
    historical_gdf = gpd.GeoDataFrame(
        {"survey_year": [2014, 2018]},
        geometry=[LineString([(0.0, 0.0), (3.0, 0.0)]), LineString([(0.0, 0.0), (5.0, 0.0)])],
        crs=CRS,
    )
    blocks = report.build_report_blocks(
        section_df=section_df,
        freespans_2018_gdf=freespans_gdf,
        historical_freespans_gdf=historical_gdf,
        condition_benchmark={"exposed_section_count": 19, "total_exposed_length_m": 519},
        route_length_km=23.48,
        depth_stats={"min_m": 10.0, "median_m": 15.0, "max_m": 20.0},
        sediment_metadata={
            "coverage_diagnostics": {"sample_year_min": 1980, "sample_year_max": 1990}
        },
        morphology_metadata={},
        highres_survey_df=pd.DataFrame(),
        seabed_data_access_gap={"final_status": "REPORT_EVIDENCE_ONLY"},
        analog_family_status={
            "real_analog_validation_attempts": 3,
            "analog_search_status": "NO_FURTHER_OPEN_ANALOG_SEARCH_PLANNED",
        },
        key_limitations=("a limitation",),
    )
    all_items = [item for b in blocks if b["type"] == "list" for item in b["items"]]
    combined = " ".join(all_items)

    assert "freespan/exposure events" not in combined
    assert any("freespan events are spatially documented" in item for item in all_items)
    assert any("aggregate corridor evidence" in item for item in all_items)


# --- G: POC overview states the operator-supplied data model --------------------------------


def test_G_poc_overview_states_operator_supplied_data_model():
    md = report.render_blocks_markdown(poc.build_poc_overview_blocks())
    assert "operator-supplied" in md.lower()


# --- H: POC overview does not describe OrbGSS as a survey contractor ------------------------


def test_H_poc_overview_does_not_describe_orbgss_as_a_survey_contractor():
    md = report.render_blocks_markdown(poc.build_poc_overview_blocks())
    assert "a survey acquisition company" in md
    assert "a geophysical contractor" in md
    assert "orbgss is a survey" not in md.lower()
    assert "orbgss is a geophysical contractor" not in md.lower()
    assert "orbgss is a drilling" not in md.lower()


# --- I: measured/interpreted/derived categories remain distinct -----------------------------


def test_I_measured_interpreted_derived_categories_remain_distinct():
    blocks = poc.build_poc_overview_blocks()
    for block in blocks:
        if (
            block["type"] == "heading"
            and block["text"] == "Measured / Interpreted / Derived Data Model"
        ):
            idx = blocks.index(block)
            items = blocks[idx + 1]["items"]
            break
    else:
        raise AssertionError("data model section not found")

    assert len(items) == 3
    assert items[0].startswith("MEASURED DATA")
    assert items[1].startswith("INTERPRETED DATA")
    assert items[2].startswith("DERIVED ENGINEERING LAYERS")


# --- J: planned hazard layers are not described as implemented ------------------------------


def test_J_planned_hazard_layers_not_claimed_as_implemented():
    blocks = poc.build_poc_overview_blocks()
    demonstrated_items = " ".join(_demonstrated_items(blocks)).lower()

    planned_hazard_names = [
        name for name, status in poc.HAZARD_MAP_STATUS_ROWS if status == "planned"
    ]
    assert planned_hazard_names, "expected at least one planned hazard class"
    for name in planned_hazard_names:
        assert name.lower() not in demonstrated_items


# --- K: offline POC index contains all required deliverables --------------------------------


def test_K_poc_index_contains_all_required_deliverables():
    html = poc.build_poc_index_html(
        deliverables=[
            {
                "title": "Marine POC Overview",
                "href": "orbgss_marine_poc_overview.html",
                "description": "d",
            },
            {"title": "Engineering Evidence Atlas", "href": "../maps/a.png", "description": "d"},
            {"title": "KP Evidence Strip", "href": "../maps/s.png", "description": "d"},
            {
                "title": "Engineering Evidence Report",
                "href": "../report/r.html",
                "description": "d",
            },
            {
                "title": "External Reviewer Guide",
                "href": "external_georisk_review_guide.md",
                "description": "d",
            },
        ]
    )
    for expected in (
        "Marine POC Overview",
        "Engineering Evidence Atlas",
        "KP Evidence Strip",
        "Engineering Evidence Report",
        "External Reviewer Guide",
    ):
        assert expected in html


# --- L: review guide contains data-authorization/privacy caveat -----------------------------


def test_L_review_guide_contains_data_authorization_caveat():
    guide = poc.build_review_guide_markdown()
    assert "owned or authorized" in guide
    assert "non-confidential" in guide
    assert "anonymized" in guide
    assert "PL854 public data are sufficient" in guide


# --- M: package manifest hashes all review deliverables --------------------------------------


def test_M_package_manifest_hashes_all_review_deliverables(tmp_path):
    names = [
        "poc_index_html",
        "poc_overview_html",
        "poc_overview_md",
        "external_review_guide_md",
        "polished_atlas_png",
        "polished_evidence_strip_png",
        "engineering_report_html",
    ]
    deliverables = {}
    for name in names:
        path = tmp_path / f"{name}.txt"
        path.write_text(f"content for {name}", encoding="utf-8")
        deliverables[name] = path

    manifest = report.build_package_manifest(
        deliverables=deliverables,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        project_root=tmp_path,
    )

    assert manifest["file_count"] == len(names)
    manifest_names = {f["name"] for f in manifest["files"]}
    assert manifest_names == set(names)
    for entry in manifest["files"]:
        full_path = tmp_path / entry["relative_path"]
        assert entry["sha256"] == report.compute_sha256(full_path)
        assert entry["file_size_bytes"] == full_path.stat().st_size


# --- N: all HTML works with network disabled --------------------------------------------------


def test_N_poc_html_builds_with_network_disabled(monkeypatch):
    def _blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted during POC content build")

    monkeypatch.setattr(socket, "socket", _blocked)

    overview_html = report.render_blocks_html(poc.build_poc_overview_blocks(), title=poc.POC_TITLE)
    index_html = poc.build_poc_index_html(
        deliverables=[{"title": "X", "href": "x.html", "description": "d"}]
    )
    guide_md = poc.build_review_guide_markdown()

    assert "<html" in overview_html
    assert "<html" in index_html
    assert len(guide_md) > 0


# --- O: primary PNG outputs render non-empty --------------------------------------------------


def test_O_polished_atlas_and_strip_render_non_empty(tmp_path):
    route = LineString([(0.0, 0.0), (1000.0, 0.0)])
    sections_gdf = gpd.GeoDataFrame(
        {
            "segment_id": [0, 1],
            "combined_tau_max_p95_upper_pa": [0.5, 0.8],
            "mobility_capacity_p95_d50_mm": [0.5, 1.0],
            "observed_2018_freespan_count": [1, 0],
            "observed_2018_freespan_total_length_m": [5.0, 0.0],
        },
        geometry=[
            LineString([(0.0, 0.0), (500.0, 0.0)]),
            LineString([(500.0, 0.0), (1000.0, 0.0)]),
        ],
        crs=CRS,
    )
    freespans_gdf = gpd.GeoDataFrame(
        {
            "source_length_m": [5.0],
            "source_height_m": [0.2],
            "canonical_chainage_min_m": [100.0],
            "canonical_chainage_max_m": [105.0],
        },
        geometry=[LineString([(100.0, 0.0), (105.0, 0.0)])],
        crs=CRS,
    )
    historical_gdf = gpd.GeoDataFrame(
        {"survey_year": [2014, 2018]},
        geometry=[LineString([(10.0, 0.0), (15.0, 0.0)]), LineString([(100.0, 0.0), (105.0, 0.0)])],
        crs=CRS,
    )
    empty_gdf = gpd.GeoDataFrame(geometry=[], crs=CRS)
    chainage_reference_gdf = gpd.GeoDataFrame(
        {"reference_label": ["KP 0", "KP end"]},
        geometry=gpd.points_from_xy([0.0, 1000.0], [0.0, 0.0]),
        crs=CRS,
    )
    section_df = pd.DataFrame(
        {
            "segment_id": [0, 1],
            "start_chainage_m": [0.0, 500.0],
            "end_chainage_m": [500.0, 1000.0],
            "kp_start": ["KP 0+000", "KP 0+500"],
            "kp_end": ["KP 0+500", "KP 1+000"],
            "combined_tau_max_p95_lower_pa": [0.2, 0.3],
            "combined_tau_max_p95_upper_pa": [0.5, 0.8],
            "mobility_capacity_p95_d50_mm": [0.5, 1.0],
            "p95_required_embedment_upper_class": ["0.03", "0.03"],
            "local_relief_1000m_median_m": [3.0, 5.0],
            "mapped_250k_folk_class": ["gS", "S"],
        }
    )

    atlas_path = maps.render_engineering_evidence_atlas(
        route=route,
        sections_gdf=sections_gdf,
        freespans_2018_gdf=freespans_gdf,
        historical_freespans_gdf=historical_gdf,
        psa_points_gdf=empty_gdf,
        highres_survey_gdf=empty_gdf,
        chainage_reference_gdf=chainage_reference_gdf,
        observed_condition_summary={
            "event_count": 1,
            "total_length_m": 5.0,
            "max_length_m": 5.0,
            "max_height_m": 0.2,
            "exposed_section_count": 1,
            "total_exposed_length_m": 10.0,
        },
        key_limitations=("a test limitation",),
        output_path=tmp_path / "atlas.png",
    )
    strip_path = maps.render_evidence_strip(
        section_df=section_df,
        freespans_2018_gdf=freespans_gdf,
        folk_class_descriptions={"gS": "GRAVELLY SAND", "S": "SAND"},
        output_path=tmp_path / "strip.png",
    )

    for path in (atlas_path, strip_path):
        assert path.exists()
        assert path.stat().st_size > 0

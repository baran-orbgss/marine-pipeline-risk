"""Offline unit tests for marine_engine.evidence_atlas (MAR-018).

Small hand-built synthetic segment/event tables and tiny synthetic
GeoDataFrames only -- never the real PL854 outputs, never network access.
Lettered test names map to MAR-018 Section 28's required test list (A-T).
"""

from __future__ import annotations

import inspect
import json
import socket
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
from shapely.geometry import LineString, Point

from marine_engine.evidence_atlas import core, maps, report

CRS = "EPSG:32631"


# --- Shared synthetic fixture builders (never the real 14-section PL854 data) --------------


def _segment_geoms(n: int) -> list[LineString]:
    return [LineString([(i * 100.0, 0.0), ((i + 1) * 100.0, 0.0)]) for i in range(n)]


def _write_combined_segments(path: Path, n: int) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "pipeline_id": ["TESTPIPE"] * n,
            "segment_id": list(range(n)),
            "hydro_pair_id": [f"pair_{i}" for i in range(n)],
            "start_chainage_m": [i * 100.0 for i in range(n)],
            "end_chainage_m": [(i + 1) * 100.0 for i in range(n)],
            "kp_start": [f"KP 0+{i * 100:03d}" for i in range(n)],
            "kp_end": [f"KP 0+{(i + 1) * 100:03d}" for i in range(n)],
            "tau_max_p95_sensitivity_min_pa": [0.20 + 0.01 * i for i in range(n)],
            "tau_max_p95_sensitivity_max_pa": [0.50 + 0.01 * i for i in range(n)],
            "tau_max_p95_sensitivity_width_pa": [0.30] * n,
        },
        geometry=_segment_geoms(n),
        crs=CRS,
    )
    gdf.to_file(path, driver="GPKG", layer="combined_bed_shear_segments")


def _write_current_segments(path: Path, n: int) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "segment_id": list(range(n)),
            "current_reference_speed_p95_m_s": [0.30 + 0.01 * i for i in range(n)],
        },
        geometry=_segment_geoms(n),
        crs=CRS,
    )
    gdf.to_file(path, driver="GPKG", layer="current_reference_segments")


def _write_wave_segments(path: Path, n: int) -> None:
    gdf = gpd.GeoDataFrame(
        {"segment_id": list(range(n)), "orbital_rms_p95_m_s": [0.10 + 0.01 * i for i in range(n)]},
        geometry=_segment_geoms(n),
        crs=CRS,
    )
    gdf.to_file(path, driver="GPKG", layer="wave_orbital_reference_segments")


def _write_mobility_segments(path: Path, n: int) -> None:
    ladder = [0.5, 1.0, 1.0, 0.5, 1.0][:n]
    gdf = gpd.GeoDataFrame(
        {
            "segment_id": list(range(n)),
            "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": ladder,
            "mobility_ratio_p95_d50_500um": [1.2] * n,
            "mobility_ratio_p95_d50_1000um": [0.8] * n,
            "mapped_250k_folk_class": ["gS"] * n,
            "nearest_valid_psa_id": list(range(100, 100 + n)),
            "nearest_valid_psa_d50_mm": [0.25] * n,
            "nearest_valid_psa_distance_m": [500.0] * n,
            "nearest_valid_psa_sample_year": [1980] * n,
        },
        geometry=_segment_geoms(n),
        crs=CRS,
    )
    gdf.to_file(path, driver="GPKG", layer="noncohesive_mobility_capacity_segments")


def _write_scour_segments(path: Path, n: int) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "segment_id": list(range(n)),
            "p95_required_embedment_lower_class": ["0.03"] * n,
            "p95_required_embedment_upper_class": ["0.03"] * n,
            "slope_500m_median_deg": [1.0] * n,
            "slope_1000m_median_deg": [1.0] * n,
            "tpi_1000m_median_m": [0.1] * n,
            "local_relief_1000m_median_m": [5.0] * n,
            "terrain_std_1000m_median_m": [1.0] * n,
        },
        geometry=_segment_geoms(n),
        crs=CRS,
    )
    gdf.to_file(path, driver="GPKG", layer="scour_onset_embedment_segments")


def _write_freespan_counts(path: Path, n: int, event_hydro_pair_ids: set[str]) -> None:
    df = pd.DataFrame(
        {
            "hydro_pair_id": [f"pair_{i}" for i in range(n)],
            "freespan_2018_count": [
                2 if f"pair_{i}" in event_hydro_pair_ids else 0 for i in range(n)
            ],
            "freespan_2018_total_length_m": [
                10.0 if f"pair_{i}" in event_hydro_pair_ids else 0.0 for i in range(n)
            ],
            "freespan_2018_max_length_m": [
                6.0 if f"pair_{i}" in event_hydro_pair_ids else None for i in range(n)
            ],
            "freespan_2018_max_height_m": [
                0.2 if f"pair_{i}" in event_hydro_pair_ids else None for i in range(n)
            ],
            "any_2018_freespan": [f"pair_{i}" in event_hydro_pair_ids for i in range(n)],
        }
    )
    df.to_parquet(path, index=False)


def _build_minimal_study_dir(
    base_dir: Path, n: int = 3, event_hydro_pair_ids: set[str] | None = None
) -> Path:
    event_hydro_pair_ids = event_hydro_pair_ids or {"pair_1"}
    study_dir = base_dir / "pl854"
    (study_dir / "metocean").mkdir(parents=True)
    (study_dir / "sediment").mkdir(parents=True)
    (study_dir / "scour").mkdir(parents=True)
    (study_dir / "freespan_evidence").mkdir(parents=True)

    _write_combined_segments(study_dir / "metocean" / "combined_bed_shear_segments.gpkg", n)
    _write_current_segments(study_dir / "metocean" / "current_reference_segments.gpkg", n)
    _write_wave_segments(study_dir / "metocean" / "wave_orbital_reference_segments.gpkg", n)
    _write_mobility_segments(
        study_dir / "sediment" / "noncohesive_mobility_capacity_segments.gpkg", n
    )
    _write_scour_segments(study_dir / "scour" / "scour_onset_embedment_segments.gpkg", n)
    _write_freespan_counts(
        study_dir / "freespan_evidence" / "freespan_segment_event_counts_2018.parquet",
        n,
        event_hydro_pair_ids,
    )
    return study_dir


def _write_historical_freespans(path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "event_id": ["2014-01", "2018-01", "2018-02"],
            "survey_year": [2014, 2018, 2018],
            "asset_scope": ["PL854_PL855_PIGGYBACK_CORRIDOR"] * 3,
            "individual_line_attribution": ["UNRESOLVED"] * 3,
            "source_length_m": [8.0, 10.0, 6.0],
            "source_height_m": [0.2, 0.3, 0.15],
            "projected_route_interval_length_m": [8.3, 10.4, 6.2],
        },
        geometry=[
            LineString([(0.0, 0.0), (8.3, 0.0)]),
            LineString([(50.0, 0.0), (60.4, 0.0)]),
            LineString([(150.0, 0.0), (156.2, 0.0)]),
        ],
        crs=CRS,
    )
    gdf.to_file(path, driver="GPKG", layer="historical_freespans")


def _build_minimal_report_blocks() -> list[dict]:
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
    return report.build_report_blocks(
        section_df=section_df,
        freespans_2018_gdf=freespans_gdf,
        historical_freespans_gdf=historical_gdf,
        condition_benchmark={"exposed_section_count": 1, "total_exposed_length_m": 10.0},
        route_length_km=1.0,
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
        key_limitations=(
            "Hydrodynamic forcing (2024-2026) postdates the 2018 observations",
            "No verified open 2018 Fugro bathymetric grid",
        ),
    )


# --- A: section count is derived, never hard-coded to 14 -----------------------------------


def test_A_section_count_is_derived_not_hardcoded(tmp_path):
    for n in (2, 5):
        study_dir = _build_minimal_study_dir(tmp_path / f"n{n}", n=n)
        section_df = core.build_section_evidence_table(study_dir)
        assert len(section_df) == n


# --- B: 2018 events remain corridor-level and unresolved by line ---------------------------


def test_B_2018_events_remain_corridor_level_unresolved(tmp_path):
    study_dir = tmp_path / "pl854"
    (study_dir / "freespan_evidence").mkdir(parents=True)
    _write_historical_freespans(
        study_dir / "freespan_evidence" / "anglia_freespan_spatial_evidence.gpkg"
    )

    layer = core.build_observed_freespans_2018_layer(study_dir)

    assert len(layer) == 2
    assert set(layer["asset_scope"]) == {"PL854_PL855_PIGGYBACK_CORRIDOR"}
    assert set(layer["individual_line_attribution"]) == {"UNRESOLVED"}


# --- C: no event is enlarged in canonical GIS geometry --------------------------------------


def test_C_event_geometry_never_enlarged(tmp_path):
    study_dir = tmp_path / "pl854"
    (study_dir / "freespan_evidence").mkdir(parents=True)
    _write_historical_freespans(
        study_dir / "freespan_evidence" / "anglia_freespan_spatial_evidence.gpkg"
    )

    layer = core.build_observed_freespans_2018_layer(study_dir)

    for _, row in layer.iterrows():
        assert row.geometry.length == pytest.approx(
            row["projected_route_interval_length_m"], abs=1e-6
        )


# --- D: exposure aggregate is never spatially fabricated ------------------------------------


def test_D_no_spatial_exposure_column_exists():
    assert not any("expos" in column.lower() for column in core.SECTION_EVIDENCE_COLUMNS)


# --- E: observed/modelled/screening evidence types remain distinct --------------------------


def test_E_evidence_types_remain_distinct_per_scientific_role():
    manifest = report.build_evidence_variable_manifest(generated_at_utc="2026-01-01T00:00:00+00:00")
    types_by_role: dict[str, set[str]] = {}
    for variable in manifest["variables"]:
        types_by_role.setdefault(variable["scientific_role"], set()).add(variable["evidence_type"])
    for role, types in types_by_role.items():
        assert len(types) == 1, f"{role} mixes evidence types: {types}"
    # and observed/modelled/screening are not all collapsed into one type across the manifest
    all_types = {v["evidence_type"] for v in manifest["variables"]}
    assert core.EVIDENCE_TYPE_OFFICIAL_OBSERVED_CONDITION in all_types
    assert core.EVIDENCE_TYPE_PHYSICS_BASED_MODEL_OUTPUT in all_types
    assert core.EVIDENCE_TYPE_EMPIRICAL_ENGINEERING_SCREENING in all_types


# --- F: no fused score column exists ---------------------------------------------------------


def test_F_no_fused_score_column():
    for column in core.SECTION_EVIDENCE_COLUMNS:
        for fragment in core.FORBIDDEN_TERM_FRAGMENTS:
            assert fragment not in column.lower(), column


# --- G: no probability/risk field exists ------------------------------------------------------


def test_G_no_probability_or_risk_value_anywhere():
    assert "PROBABILITY" not in core.OBSERVED_EVENT_PRESENT_STATUS
    assert "RISK" not in core.OBSERVED_EVENT_PRESENT_STATUS
    assert "PROBABILITY" not in core.NO_OBSERVED_EVENT_STATUS
    assert "RISK" not in core.NO_OBSERVED_EVENT_STATUS

    manifest = report.build_evidence_variable_manifest(generated_at_utc="2026-01-01T00:00:00+00:00")
    manifest_text = json.dumps(manifest).lower()
    assert "probability" not in manifest_text
    assert "risk_score" not in manifest_text


# --- H: combined shear map uses the MAR-012 physical variable directly ----------------------


def test_H_combined_shear_value_passed_through_unmodified(tmp_path):
    study_dir = _build_minimal_study_dir(tmp_path, n=3)
    section_df = core.build_section_evidence_table(study_dir)
    combined_gdf = gpd.read_file(
        study_dir / "metocean" / "combined_bed_shear_segments.gpkg",
        layer="combined_bed_shear_segments",
    )
    merged = section_df.merge(pd.DataFrame(combined_gdf.drop(columns="geometry")), on="segment_id")

    assert (
        merged["combined_tau_max_p95_upper_pa"] == merged["tau_max_p95_sensitivity_max_pa"]
    ).all()
    assert (
        merged["combined_tau_max_p95_lower_pa"] == merged["tau_max_p95_sensitivity_min_pa"]
    ).all()


# --- I: mobility map uses the MAR-013 discrete capacity directly ----------------------------


def test_I_mobility_capacity_value_passed_through_unmodified(tmp_path):
    study_dir = _build_minimal_study_dir(tmp_path, n=3)
    section_df = core.build_section_evidence_table(study_dir)
    mobility_gdf = gpd.read_file(
        study_dir / "sediment" / "noncohesive_mobility_capacity_segments.gpkg",
        layer="noncohesive_mobility_capacity_segments",
    )
    merged = section_df.merge(pd.DataFrame(mobility_gdf.drop(columns="geometry")), on="segment_id")

    assert (
        merged["mobility_capacity_p95_d50_mm"]
        == merged["largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"]
    ).all()


# --- J: MAR-014 0.03D remains screening, not a design burial requirement --------------------


def test_J_embedment_class_remains_screening_not_design():
    blocks = _build_minimal_report_blocks()
    all_text = " ".join(
        item for block in blocks if block["type"] == "list" for item in block["items"]
    )
    assert "never an engineering burial design requirement" in all_text
    assert "SCREENING" in all_text.upper()


# --- K: morphology retains its 1991-1992 legacy label ----------------------------------------


def test_K_morphology_legacy_label_present():
    manifest = report.build_evidence_variable_manifest(generated_at_utc="2026-01-01T00:00:00+00:00")
    morphology_entries = [
        v
        for v in manifest["variables"]
        if v["evidence_type"] == core.EVIDENCE_TYPE_LEGACY_REGIONAL_MORPHOLOGY_CONTEXT
    ]
    assert morphology_entries
    assert all("1991-1992" in v["acquisition_epoch"] for v in morphology_entries)


# --- L: hydrodynamic support is never represented as 25 m resolution -----------------------


def test_L_hydrodynamic_support_not_25m():
    manifest = report.build_evidence_variable_manifest(generated_at_utc="2026-01-01T00:00:00+00:00")
    hydro_entries = [
        v
        for v in manifest["variables"]
        if v["evidence_type"] == core.EVIDENCE_TYPE_PHYSICS_BASED_MODEL_OUTPUT
    ]
    assert hydro_entries
    for variable in hydro_entries:
        assert "25 m" not in variable["spatial_support"]
        assert "km" in variable["spatial_support"]


# --- M: GeoPackage opens and expected layers exist -------------------------------------------


def test_M_gpkg_opens_with_expected_layers(tmp_path):
    layers = {
        "pipeline_route": gpd.GeoDataFrame(
            {"pipeline_id": ["X"]}, geometry=[LineString([(0.0, 0.0), (10.0, 0.0)])], crs=CRS
        ),
        "engineering_support_sections": gpd.GeoDataFrame(
            {"segment_id": [0]}, geometry=[LineString([(0.0, 0.0), (10.0, 0.0)])], crs=CRS
        ),
    }
    out_path = tmp_path / "atlas.gpkg"

    core.write_evidence_atlas_gpkg(out_path, layers)

    found = {name for name, _geom_type in pyogrio.list_layers(str(out_path))}
    assert found == set(layers.keys())


# --- N: all projected GIS layers are EPSG:32631 ----------------------------------------------


def test_N_crs_mismatch_is_detected():
    good = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0.0, 0.0)], crs=CRS)
    bad = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0.0, 0.0)], crs="EPSG:4326")

    mismatched = core.verify_projected_layers_crs({"good": good, "bad": bad})

    assert mismatched == ["bad"]


# --- O: HTML report builds without network ---------------------------------------------------


def test_O_html_report_builds_without_network(monkeypatch):
    def _blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted during report build")

    monkeypatch.setattr(socket, "socket", _blocked)

    blocks = _build_minimal_report_blocks()
    html_text = report.render_blocks_html(blocks, title="Test Report")

    assert "<html" in html_text
    assert "PL854 Engineering Evidence Report" in html_text


# --- P: report contains the key limitations section ------------------------------------------


def test_P_report_contains_key_limitations():
    blocks = _build_minimal_report_blocks()
    callout_texts = [block["text"] for block in blocks if block["type"] == "callout"]
    all_list_items = [
        item for block in blocks if block["type"] == "list" for item in block["items"]
    ]

    assert any("Key evidence limits" in text for text in callout_texts)
    assert any("postdates the 2018 observations" in item for item in all_list_items)


# --- Q: report states no high-resolution PL854 bathymetry is available ---------------------


def test_Q_report_states_no_highres_bathymetry():
    blocks = _build_minimal_report_blocks()
    all_list_items = [
        item for block in blocks if block["type"] == "list" for item in block["items"]
    ]
    assert any(
        "No verified open PL854-specific high-resolution bathymetric grid" in item
        for item in all_list_items
    )


# --- R: MAR-017 analog data never enters PL854 scientific section values -------------------


def test_R_section_columns_never_reference_analogs():
    for column in core.SECTION_EVIDENCE_COLUMNS:
        assert "analog" not in column.lower()
        assert "sandwave" not in column.lower()
        assert "morphometry" not in column.lower()


def test_R2_core_module_never_reads_analog_outputs():
    # The module docstring legitimately explains that analog outputs are excluded (and says so
    # using the word "analog") -- what must never appear is an actual path reference to them.
    source = inspect.getsource(core).lower()
    assert "processed/analogs" not in source
    assert '"analogs"' not in source


# --- S: package manifest hashes every deliverable correctly ---------------------------------


def test_S_package_manifest_hashes_correctly(tmp_path):
    file_a = tmp_path / "a.txt"
    file_a.write_text("hello world", encoding="utf-8")
    file_b = tmp_path / "b.txt"
    file_b.write_text("second file", encoding="utf-8")

    manifest = report.build_package_manifest(
        deliverables={"a": file_a, "b": file_b},
        generated_at_utc="2026-01-01T00:00:00+00:00",
        project_root=tmp_path,
    )

    assert manifest["file_count"] == 2
    for entry in manifest["files"]:
        full_path = tmp_path / entry["relative_path"]
        assert entry["file_size_bytes"] == full_path.stat().st_size
        assert entry["sha256"] == report.compute_sha256(full_path)


# --- T: all primary PNGs render non-empty ----------------------------------------------------


def test_T_all_primary_pngs_render_non_empty(tmp_path):
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
        geometry=[Point(0.0, 0.0), Point(1000.0, 0.0)],
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
            "observed_2018_freespan_count": [1, 0],
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
        section_df=section_df, freespans_2018_gdf=freespans_gdf, output_path=tmp_path / "strip.png"
    )
    table_path = maps.render_section_summary_table(
        section_df=section_df, output_path=tmp_path / "table.png"
    )
    timeline_path = maps.render_provenance_timeline(
        epochs=[
            {
                "label": "Test epoch (2000-2001)",
                "start_year": 2000.0,
                "end_year": 2001.0,
                "kind": "OBSERVATION_PERIOD",
                "color": "0.5",
            }
        ],
        output_path=tmp_path / "timeline.png",
    )

    for path in (atlas_path, strip_path, table_path, timeline_path):
        assert path.exists()
        assert path.stat().st_size > 0

"""Explicitly synthetic engineering validation case for the generic free-span / support-loss
engine (MAR-025 Section 17).

Because no verified open project-grade pipeline vertical-profile + seabed support-profile
dataset is available in this repo, this module builds a small, entirely synthetic, ANALYTICALLY
KNOWN operator dataset -- every expected answer (measured span count/lengths/max clearance,
scenario new/extended span counts) is computed by hand and recorded as a module constant, then
asserted exactly by `tests/test_free_span_poc.py`. It is run through the SAME generic engine
functions (`freespan.reference`, `freespan.clearance`, `freespan.support_state`,
`freespan.scenario`) a real operator dataset would use -- there is no special-cased shortcut.

Scientific role: `SYNTHETIC_EXACT_ENGINE_VALIDATION`. Never presented as field validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from marine_engine.burial import route as generic_route
from marine_engine.freespan import clearance as freespan_clearance
from marine_engine.freespan import reference as freespan_reference
from marine_engine.freespan import scenario as freespan_scenario
from marine_engine.freespan import support_state as freespan_support_state

SCIENTIFIC_ROLE = "SYNTHETIC_EXACT_ENGINE_VALIDATION"
PROMINENT_DISCLAIMER = "SYNTHETIC EXACT VALIDATION -- NOT FIELD DATA"

SYNTHETIC_ASSET_ID = "SYNTHETIC_FREESPAN_ASSET"
SYNTHETIC_CRS = "EPSG:32631"
SYNTHETIC_SURVEY_EPOCH = "SYNTHETIC_ENGINEERING_CASE"
SYNTHETIC_EVIDENCE_TYPE = "SYNTHETIC_EXACT_ENGINE_VALIDATION_MEASURED_PROFILE"

_ROUTE_ORIGIN_X = 500000.0
_ROUTE_ORIGIN_Y = 6000000.0
_ROUTE_LENGTH_M = 650.0

# --- Section 3: pipe vertical reference --------------------------------------------------------
PIPE_VERTICAL_REFERENCE = freespan_reference.PIPE_CENTRELINE_ELEVATION
RAW_PIPE_CENTRELINE_ELEVATION_M = -9.8
CENTRELINE_TO_PIPE_BOTTOM_OFFSET_M = 0.2
EXPECTED_PIPE_BOTTOM_ELEVATION_M = -10.0

# --- Section 4: seabed support reference --------------------------------------------------------
SEABED_SIGN_CONVENTION = freespan_clearance.ALREADY_ELEVATION_STYLE

# --- Section 6-7: classification threshold -------------------------------------------------------
CLASSIFICATION_THRESHOLD_M = 0.10
CLASSIFICATION_THRESHOLD_PROVENANCE = (
    freespan_clearance.SOURCE_OR_OPERATOR_CLEARANCE_CLASSIFICATION_THRESHOLD
)

# --- Section 10: gap governance ------------------------------------------------------------------
MAX_MEASUREMENT_GAP_M = 50.0

# --- Section 12-13: support-loss scenario ---------------------------------------------------------
SCENARIO_LOWERING_M = 0.30
SCENARIO_EVIDENCE_TYPE = freespan_scenario.ENGINEERING_SCENARIO

# --- Section 17: hand-authored, analytically-known sample table ----------------------------------
# (nominal_chainage_m_for_placement, raw_seabed_elevation_m, source_interpreted_free_span_present,
# note). raw_pipe_centreline_elevation_m is constant (RAW_PIPE_CENTRELINE_ELEVATION_M) for every
# sample -- a flat synthetic pipe, since Section 13 fixes pipe vertical position for the
# screening scenario anyway. chainage_m itself is never hand-assigned: every sample's (x, y) is
# placed exactly on the route and chainage is DERIVED by
# `generic_route.project_measurements_to_chainage`, exactly as a real dataset would be processed.
#
# MAR-025A: the source-interpreted flag is carried on TWO samples, deliberately proving
# independence from geometric classification in both directions (Section 13):
#   - chainage 200 sits INSIDE the already-geometrically-unsupported 150-250 run -- the flag
#     must neither change its measured_support_state (still MEASURED_UNSUPPORTED) nor split the
#     measured interval (still one continuous 150-250 span, AGREES_UNSUPPORTED).
#   - chainage 320 is geometrically supported -- the flag must neither promote it to
#     MEASURED_UNSUPPORTED nor create a new measured interval there (SOURCE_ONLY_FREESPAN_EVIDENCE,
#     a real evidence disagreement, preserved rather than silently resolved).
_SAMPLES: tuple[tuple[float, float, bool, str], ...] = (
    (0.0, -9.5, False, "supported region"),
    (25.0, -9.5, False, "supported region"),
    (50.0, -9.5, False, "supported region"),
    (75.0, -9.5, False, "supported region"),
    (100.0, -9.5, False, "supported region"),
    (125.0, -10.05, False, "support transition"),
    (150.0, -10.6, False, "known existing free span -- start"),
    (175.0, -10.6, False, "known existing free span"),
    (
        200.0,
        -10.6,
        True,
        "known existing free span -- also source-interpreted (AGREES_UNSUPPORTED)",
    ),
    (225.0, -10.6, False, "known existing free span"),
    (250.0, -10.6, False, "known existing free span -- end"),
    (260.0, -9.85, False, "supported at baseline -- extends the existing span under the scenario"),
    (275.0, -9.4, False, "supported at baseline -- remains supported under the scenario"),
    (
        320.0,
        -9.3,
        True,
        "geometrically supported, also source-interpreted -- a real evidence "
        "disagreement (SOURCE_ONLY_FREESPAN_EVIDENCE), never silently resolved",
    ),
    (390.0, -10.6, False, "unsurveyed-gap span A -- start"),
    (400.0, -10.6, False, "unsurveyed-gap span A -- end"),
    (460.0, -10.6, False, "unsurveyed-gap span B -- start"),
    (470.0, -10.6, False, "unsurveyed-gap span B -- end"),
    (600.0, -9.85, False, "supported at baseline -- new support loss under the scenario"),
    (610.0, -9.85, False, "supported at baseline -- new support loss under the scenario"),
)

# --- Hand-verified expected answers (Section 17: "all geometry and expected answers must be
# analytically known") -----------------------------------------------------------------------
EXPECTED_SAMPLE_COUNT = len(_SAMPLES)
EXPECTED_MEASURED_SPAN_COUNT = 3
EXPECTED_MEASURED_SPAN_LENGTHS_M = (100.0, 10.0, 10.0)
EXPECTED_MAX_CLEARANCE_M = 0.6
EXPECTED_GAP_SPLIT_MEASURED_SPAN_COUNT = 2  # the two 390-400 / 460-470 halves
EXPECTED_NEW_SPAN_COUNT = 1
EXPECTED_EXTENDED_SPAN_COUNT = 1

# MAR-025A Section 12: every sample's chainage was hand-placed and every expected boundary is
# known exactly by construction -- an honest, explicit label, never the generic engine's default
# (`support_state.UNSUPPORTED_SAMPLE_RUN_EXTENT`) and never silently assigned
# `support_state.EXPLICIT_BOUNDARY_SAMPLES` without this note explaining why it applies here.
SYNTHETIC_BOUNDARY_SEMANTICS_NOTE = "EXPLICIT_BOUNDARY_SAMPLES_BY_SYNTHETIC_CONSTRUCTION"


@dataclass(frozen=True)
class SyntheticFreeSpanCase:
    route: LineString
    route_gdf: gpd.GeoDataFrame
    profile_df: pd.DataFrame
    measured_intervals_df: pd.DataFrame
    scenario_result: dict[str, Any]
    lowering_input: freespan_scenario.SeabedLoweringScenarioInput
    vertical_reference_semantics: str


def build_synthetic_free_span_case() -> SyntheticFreeSpanCase:
    """Section 17: build the full synthetic engineering validation case end to end, through the
    same generic engine functions a real operator dataset would use."""

    route = LineString(
        [(_ROUTE_ORIGIN_X, _ROUTE_ORIGIN_Y), (_ROUTE_ORIGIN_X + _ROUTE_LENGTH_M, _ROUTE_ORIGIN_Y)]
    )
    route_gdf = generic_route.build_canonical_asset_route(
        route=route,
        asset_id=SYNTHETIC_ASSET_ID,
        source_route_name="SYNTHETIC_STRAIGHT_ROUTE",
        source_crs=SYNTHETIC_CRS,
        geometry_direction_semantics="SYNTHETIC_ROUTE_INCREASING_CHAINAGE_BY_CONSTRUCTION",
    )

    nominal_chainage = [s[0] for s in _SAMPLES]
    x = [_ROUTE_ORIGIN_X + c for c in nominal_chainage]
    y = [_ROUTE_ORIGIN_Y for _ in _SAMPLES]
    chainage_m = generic_route.project_measurements_to_chainage(route, x, y)

    records_df = pd.DataFrame(
        {
            "asset_id": SYNTHETIC_ASSET_ID,
            "chainage_m": chainage_m,
            "x_m": x,
            "y_m": y,
            "raw_pipe_centreline_elevation_m": [RAW_PIPE_CENTRELINE_ELEVATION_M for _ in _SAMPLES],
            "raw_seabed_elevation_m": [s[1] for s in _SAMPLES],
            "source_interpreted_free_span_present": [s[2] for s in _SAMPLES],
            "note": [s[3] for s in _SAMPLES],
        }
    )

    pipe_bottom_elevation_m = [
        freespan_reference.normalize_pipe_bottom_elevation_m(
            v,
            pipe_vertical_reference=PIPE_VERTICAL_REFERENCE,
            reference_to_pipe_bottom_offset_m=CENTRELINE_TO_PIPE_BOTTOM_OFFSET_M,
        )
        for v in records_df["raw_pipe_centreline_elevation_m"]
    ]
    seabed_support_elevation_m = freespan_clearance.canonicalize_seabed_support_elevation_m(
        records_df["raw_seabed_elevation_m"].to_numpy(),
        source_sign_convention=SEABED_SIGN_CONVENTION,
    )
    clearance_m = [
        freespan_clearance.compute_pipe_underside_clearance_m(pb, float(ss))
        for pb, ss in zip(pipe_bottom_elevation_m, seabed_support_elevation_m, strict=True)
    ]
    # MAR-025A Section 3: measured geometry depends ONLY on clearance + threshold -- the source
    # flag is never passed in here at all.
    measured_support_state = [
        freespan_support_state.classify_measured_support_state(
            c, classification_threshold_m=CLASSIFICATION_THRESHOLD_M
        )
        for c in clearance_m
    ]
    geometry_vs_source_interpretation_status = [
        freespan_support_state.compute_geometry_vs_source_interpretation_status(state, bool(flag))
        for state, flag in zip(
            measured_support_state, records_df["source_interpreted_free_span_present"], strict=True
        )
    ]

    profile_df = records_df.copy()
    profile_df["pipe_vertical_reference"] = PIPE_VERTICAL_REFERENCE
    profile_df["pipe_bottom_elevation_m"] = pipe_bottom_elevation_m
    profile_df["seabed_support_elevation_m"] = seabed_support_elevation_m
    profile_df["clearance_m"] = clearance_m
    profile_df["measured_support_state"] = measured_support_state
    profile_df["geometry_vs_source_interpretation_status"] = (
        geometry_vs_source_interpretation_status
    )
    profile_df["survey_epoch"] = SYNTHETIC_SURVEY_EPOCH
    profile_df["classification_threshold_m"] = CLASSIFICATION_THRESHOLD_M
    profile_df["evidence_type"] = SYNTHETIC_EVIDENCE_TYPE
    profile_df["scientific_role"] = SCIENTIFIC_ROLE

    vertical_reference_semantics = (
        f"{PIPE_VERTICAL_REFERENCE} (offset {CENTRELINE_TO_PIPE_BOTTOM_OFFSET_M} m) -> "
        f"pipe_bottom_elevation_m; seabed_support_elevation_m already {SEABED_SIGN_CONVENTION}"
    )
    measured_intervals_df = freespan_support_state.extract_measured_free_span_intervals(
        profile_df,
        asset_id=SYNTHETIC_ASSET_ID,
        max_measurement_gap_m=MAX_MEASUREMENT_GAP_M,
        survey_epoch=SYNTHETIC_SURVEY_EPOCH,
        classification_threshold_m=CLASSIFICATION_THRESHOLD_M,
        vertical_reference_semantics=vertical_reference_semantics,
        interval_boundary_semantics=freespan_support_state.EXPLICIT_BOUNDARY_SAMPLES,
    )

    lowering_input = freespan_scenario.SeabedLoweringScenarioInput(
        SCENARIO_LOWERING_M, SCENARIO_EVIDENCE_TYPE
    )
    scenario_result = freespan_scenario.build_support_loss_scenario_profile(
        profile_df,
        asset_id=SYNTHETIC_ASSET_ID,
        lowering_input=lowering_input,
        classification_threshold_m=CLASSIFICATION_THRESHOLD_M,
        max_measurement_gap_m=MAX_MEASUREMENT_GAP_M,
        survey_epoch=SYNTHETIC_SURVEY_EPOCH,
        vertical_reference_semantics=vertical_reference_semantics,
        measured_intervals_df=measured_intervals_df,
        interval_boundary_semantics=freespan_support_state.EXPLICIT_BOUNDARY_SAMPLES,
    )

    return SyntheticFreeSpanCase(
        route=route,
        route_gdf=route_gdf,
        profile_df=profile_df,
        measured_intervals_df=measured_intervals_df,
        scenario_result=scenario_result,
        lowering_input=lowering_input,
        vertical_reference_semantics=vertical_reference_semantics,
    )


def summarize_synthetic_case(case: SyntheticFreeSpanCase) -> dict[str, Any]:
    """Section 34's required synthetic-validation report facts: expected vs. recovered."""

    intervals = case.measured_intervals_df
    source_flagged = case.profile_df[case.profile_df["source_interpreted_free_span_present"]]
    return {
        "scientific_role": SCIENTIFIC_ROLE,
        "disclaimer": PROMINENT_DISCLAIMER,
        "boundary_semantics_note": SYNTHETIC_BOUNDARY_SEMANTICS_NOTE,
        "sample_count": int(len(case.profile_df)),
        "expected_measured_span_count": EXPECTED_MEASURED_SPAN_COUNT,
        "recovered_measured_span_count": int(len(intervals)),
        "expected_measured_span_lengths_m": list(EXPECTED_MEASURED_SPAN_LENGTHS_M),
        "recovered_measured_span_lengths_m": (
            sorted(float(v) for v in intervals["unsupported_sample_run_extent_m"])
            if not intervals.empty
            else []
        ),
        "expected_max_clearance_m": EXPECTED_MAX_CLEARANCE_M,
        "recovered_max_clearance_m": (
            float(intervals["maximum_clearance_m"].max()) if not intervals.empty else None
        ),
        "gap_split_interval_count": int(
            intervals["limitations"].notna().sum() if not intervals.empty else 0
        ),
        "interval_boundary_semantics": (
            sorted(intervals["interval_boundary_semantics"].unique().tolist())
            if not intervals.empty
            else []
        ),
        "expected_scenario_new_span_count": EXPECTED_NEW_SPAN_COUNT,
        "recovered_scenario_new_span_count": case.scenario_result["new_span_count"],
        "expected_scenario_extended_span_count": EXPECTED_EXTENDED_SPAN_COUNT,
        "recovered_scenario_extended_span_count": case.scenario_result["extended_span_count"],
        "source_interpreted_free_span_sample_count": int(len(source_flagged)),
        "source_interpreted_geometry_vs_source_statuses": sorted(
            source_flagged["geometry_vs_source_interpretation_status"].unique().tolist()
        ),
        "seabed_lowering_m": case.lowering_input.seabed_lowering_m,
        "seabed_lowering_evidence_type": case.lowering_input.evidence_type,
        "classification_threshold_m": CLASSIFICATION_THRESHOLD_M,
        "classification_threshold_provenance": CLASSIFICATION_THRESHOLD_PROVENANCE,
        "max_measurement_gap_m": MAX_MEASUREMENT_GAP_M,
    }

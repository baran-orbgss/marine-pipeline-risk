"""Offline unit tests for marine_engine.analogs.idr_bnr_cend1111 (MAR-017B).

Never a live network call, never the real downloaded ZIP -- these tests
exercise the pure, offline parts of the second-analog orchestration.
Lettered comments map to MAR-017B Section 26's required test list.
"""

import inspect
import json

import pandas as pd
import pytest

from marine_engine.analogs import idr_bnr_cend1111 as idrbnr
from marine_engine.morphology import sandwave_morphometry as swm

_FORBIDDEN_OUTPUT_TERMS = (
    "freespan_probability",
    "susceptibility",
    "risk",
    "migration_rate",
    "scour_prediction",
    "ml_",
)


def _preflight_row(
    raster_candidate_id: str,
    *,
    best_2000: float | None = None,
    qualifying_2000: int = 0,
    best_1000: float | None = None,
    qualifying_1000: int = 0,
    excluded: bool = False,
    exclusion_reason: str | None = None,
) -> dict:
    return {
        "raster_candidate_id": raster_candidate_id,
        "storage_format": "ESRI_GRID",
        "crs": "EPSG:32631",
        "crs_is_geographic": False,
        "pixel_scale_plausible": True,
        "width": 1000,
        "height": 1000,
        "native_pixel_size_m": 1.0,
        "best_valid_fraction_2000m": best_2000,
        "qualifying_tile_count_2000m": qualifying_2000,
        "best_valid_fraction_1000m": best_1000,
        "qualifying_tile_count_1000m": qualifying_1000,
        "excluded_from_preflight": excluded,
        "exclusion_reason": exclusion_reason,
    }


# --- A: support preflight runs before morphometry -------------------------------------------


def test_A_cli_runs_the_canonical_support_preflight_before_any_tile_search():
    from marine_engine import cli

    source = inspect.getsource(cli._cmd_build_idrbnr_sandwave_validation)
    preflight_pos = source.index("run_canonical_support_preflight")
    tile_search_pos = source.index("build_tile_candidates")
    assert preflight_pos < tile_search_pos


def test_A_build_tile_candidates_is_never_called_when_early_stopped():
    """The CLI must not even attempt a tile search once the preflight
    already answered the question -- `build_tile_candidates` may only
    appear inside the `else` branch of the early-stop check."""

    from marine_engine import cli

    source = inspect.getsource(cli._cmd_build_idrbnr_sandwave_validation)
    early_stop_branch, _, rest = source.partition("if early_stop_status is not None:")
    early_stop_body, _, else_body = rest.partition("else:")
    assert "build_tile_candidates" not in early_stop_body
    assert "build_tile_candidates" in else_body


# --- B: zero >=1000 m support causes early stop ----------------------------------------------


def test_B_zero_qualifying_tiles_anywhere_triggers_early_stop():
    preflight_df = pd.DataFrame(
        [
            _preflight_row("a", best_2000=0.5, best_1000=0.7),
            _preflight_row("b", best_2000=0.3, best_1000=0.6),
        ]
    )
    assert (
        idrbnr.derive_early_stop_status(preflight_df)
        == idrbnr.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT
    )


def test_B_a_single_qualifying_tile_anywhere_prevents_early_stop():
    preflight_df = pd.DataFrame(
        [
            _preflight_row("a", best_2000=0.5, best_1000=0.7),
            _preflight_row("b", best_2000=0.3, best_1000=0.95, qualifying_1000=2),
        ]
    )
    assert idrbnr.derive_early_stop_status(preflight_df) is None


def test_B_all_candidates_excluded_also_triggers_early_stop():
    preflight_df = pd.DataFrame(
        [
            _preflight_row(
                "a", excluded=True, exclusion_reason="CRS_INCONSISTENT_WITH_NATIVE_PIXEL_SCALE"
            )
        ]
    )
    assert (
        idrbnr.derive_early_stop_status(preflight_df)
        == idrbnr.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT
    )


# --- C: no sub-1000 m canonical tile is ever generated -------------------------------------


def test_C_build_tile_candidates_delegates_to_the_canonical_only_engine_search():
    """`build_tile_candidates` must call the engine's `find_valid_tiles`
    (2000 m -> 1000 m only, never lower) -- never a custom cascade, and
    never the exploratory below-floor search."""

    source = inspect.getsource(idrbnr.build_tile_candidates)
    assert "swm.find_valid_tiles" in source
    assert "exploratory" not in source.lower()


def test_C_engine_tile_search_never_produces_a_sub_1000m_tile():
    import numpy as np
    import rasterio.transform

    rng = np.random.default_rng(3)
    valid = rng.random((3000, 3000)) < 0.95
    transform = rasterio.transform.from_origin(400000.0, 5900000.0, 1.0, 1.0)
    _tiles, meta = swm.find_valid_tiles(valid, 1.0, transform=transform)
    sizes = [c["tile_size_m"] for c in meta["cascade_log"]]
    assert all(size >= swm.MIN_TILE_SIZE_M for size in sizes)


# --- D: grid selection only considers canonical-support-capable grids -----------------------


def test_D_selection_never_picks_a_non_qualifying_candidate_despite_a_high_valid_fraction():
    preflight_df = pd.DataFrame(
        [
            _preflight_row("almost_but_not_quite", best_1000=0.85, qualifying_1000=0),
            _preflight_row("qualifies", best_1000=0.91, qualifying_1000=1),
        ]
    )
    selection = idrbnr.select_primary_grid_from_preflight(preflight_df)
    assert selection["selected_raster_candidate_id"] == "qualifies"


def test_D_selection_prefers_largest_tile_size_then_most_qualifying_then_highest_fraction():
    preflight_df = pd.DataFrame(
        [
            _preflight_row("only_1000m", best_1000=0.95, qualifying_1000=10),
            _preflight_row(
                "has_2000m_too",
                best_2000=0.91,
                qualifying_2000=1,
                best_1000=0.95,
                qualifying_1000=10,
            ),
        ]
    )
    selection = idrbnr.select_primary_grid_from_preflight(preflight_df)
    # A qualifying 2000 m tile beats any number of qualifying 1000 m tiles (Section 8:
    # "largest supported tile size" is preference #1).
    assert selection["selected_raster_candidate_id"] == "has_2000m_too"


def test_D_no_qualifying_candidate_returns_no_selection():
    preflight_df = pd.DataFrame(
        [_preflight_row("a", best_1000=0.5), _preflight_row("b", best_1000=0.6)]
    )
    selection = idrbnr.select_primary_grid_from_preflight(preflight_df)
    assert selection["selected_raster_candidate_id"] is None


# --- I: canonical real validation cannot pass with fewer than 3 canonical bedforms ----------


@pytest.mark.parametrize("bedform_count", [0, 1, 2])
def test_I_fewer_than_3_bedforms_never_yields_a_passing_status(bedform_count):
    status, _reason = idrbnr.derive_canonical_real_validation_status(
        canonical_tile_count=1,
        any_meets_3_wavelengths=True,
        successful_transect_count=3,
        canonical_bedform_count=bedform_count,
    )
    assert status == idrbnr.INSUFFICIENT_CANONICAL_BEDFORMS


def test_I_exactly_3_bedforms_is_sufficient():
    status, _reason = idrbnr.derive_canonical_real_validation_status(
        canonical_tile_count=1,
        any_meets_3_wavelengths=True,
        successful_transect_count=3,
        canonical_bedform_count=3,
    )
    assert status == idrbnr.CANONICAL_REAL_DATA_VALIDATED


# --- J: canonical validation status is derived from real criteria, never a fixed value -----


@pytest.mark.parametrize(
    ("kwargs", "expected_status"),
    [
        (
            {
                "canonical_tile_count": 0,
                "any_meets_3_wavelengths": False,
                "successful_transect_count": 0,
                "canonical_bedform_count": 0,
            },
            idrbnr.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT,
        ),
        (
            {
                "canonical_tile_count": 2,
                "any_meets_3_wavelengths": False,
                "successful_transect_count": 0,
                "canonical_bedform_count": 0,
            },
            idrbnr.INSUFFICIENT_WAVELENGTH_SUPPORT,
        ),
        (
            {
                "canonical_tile_count": 2,
                "any_meets_3_wavelengths": True,
                "successful_transect_count": 0,
                "canonical_bedform_count": 0,
            },
            idrbnr.INSUFFICIENT_VALID_TRANSECTS,
        ),
        (
            {
                "canonical_tile_count": 2,
                "any_meets_3_wavelengths": True,
                "successful_transect_count": 3,
                "canonical_bedform_count": 1,
            },
            idrbnr.INSUFFICIENT_CANONICAL_BEDFORMS,
        ),
        (
            {
                "canonical_tile_count": 2,
                "any_meets_3_wavelengths": True,
                "successful_transect_count": 3,
                "canonical_bedform_count": 5,
            },
            idrbnr.CANONICAL_REAL_DATA_VALIDATED,
        ),
    ],
)
def test_J_status_tracks_exactly_which_criterion_actually_failed(kwargs, expected_status):
    status, reason = idrbnr.derive_canonical_real_validation_status(**kwargs)
    assert status == expected_status
    assert reason  # a non-empty, real explanation is always returned, never a bare code


# --- K: IDRBNR analog never enters PL854 outputs ---------------------------------------------


def test_K_analog_only_flags_are_fixed_and_correct():
    assert idrbnr.ANALOG_ONLY_FLAGS == {
        "pl854_evidence": False,
        "pl854_feature_input": False,
        "pl854_validation_input": False,
        "method_development_analog_only": True,
    }
    assert idrbnr.SCIENTIFIC_ROLE == "HIGH_RESOLUTION_SANDBED_MORPHOMETRY_METHOD_DEVELOPMENT_ANALOG"


def test_K_every_canonical_table_schema_carries_the_analog_only_flags():
    for columns in (
        idrbnr.TILE_SPECTRAL_COLUMNS,
        idrbnr.TRANSECT_COLUMNS,
        idrbnr.INDIVIDUAL_BEDFORM_COLUMNS,
    ):
        for flag_name in idrbnr.ANALOG_ONLY_FLAGS:
            assert flag_name in columns
        assert "scientific_role" in columns


def test_K_transfer_contract_states_pl854_flags_false_regardless_of_outcome():
    for status in (
        idrbnr.CANONICAL_REAL_DATA_VALIDATED,
        idrbnr.INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT,
    ):
        contract = idrbnr.build_pipeline_transfer_contract(
            canonical_real_validation_status=status,
            canonical_real_validation_reason="test reason",
        )
        assert contract["pl854_evidence"] is False
        assert contract["PL854_real_data_validation_passed"] is False
        assert contract["method_development_analog_only"] is True
        json.dumps(contract, default=str)  # must be JSON-serialisable as the CLI writes it verbatim


def test_K_idrbnr_outputs_are_never_written_under_a_pl854_study_directory():
    from marine_engine import cli

    source = inspect.getsource(cli._cmd_build_idrbnr_sandwave_validation)
    assert '"analogs"' in source
    assert '"idr_bnr_cend1111"' in source
    assert "pipeline_id.lower()" not in source


# --- L: HHW exploratory rows do not enter IDRBNR canonical tables ---------------------------


def test_L_hhw_cross_analog_row_never_reads_hhw_exploratory_outputs():
    """The cross-analog summary (Section 23) must build HHW's row from
    HHW's CANONICAL tables only -- HHW's real exploratory 250 m tables
    have genuine non-zero tile/bedform counts (MAR-017A), so accidentally
    reading them here would silently misrepresent HHW's canonical
    (legitimately zero) support as if it were real canonical validation."""

    from marine_engine import cli

    source = inspect.getsource(cli._build_hhw_cross_analog_summary_row)
    assert "exploratory" not in source.lower()


def test_L_idrbnr_module_never_imports_the_hhw_exploratory_search_function():
    """The module docstring legitimately explains IN PROSE that there is
    no exploratory pipeline here (Section 9) -- that is documentation,
    not a dependency. What must never appear is an actual reference to
    HHW's exploratory function or module."""

    source = inspect.getsource(idrbnr)
    assert "find_exploratory_small_support_tiles" not in source
    assert "hhw_cend1111" not in source.lower()


# --- M: cross-analog summary contains validation support only, no risk fields ---------------


def test_M_cross_analog_summary_columns_contain_no_forbidden_term():
    from marine_engine import cli

    for column in cli.CROSS_ANALOG_SUMMARY_COLUMNS:
        for token in _FORBIDDEN_OUTPUT_TERMS:
            assert token not in column.lower(), column


def test_M_no_canonical_column_schema_contains_a_forbidden_term():
    all_columns = (
        list(idrbnr.CANONICAL_SUPPORT_PREFLIGHT_COLUMNS)
        + list(idrbnr.TILE_SPECTRAL_COLUMNS)
        + list(idrbnr.TRANSECT_COLUMNS)
        + list(idrbnr.INDIVIDUAL_BEDFORM_COLUMNS)
    )
    for column in all_columns:
        for token in _FORBIDDEN_OUTPUT_TERMS:
            assert token not in column.lower(), column


def test_M_reusable_engine_source_contains_no_forbidden_term():
    source = inspect.getsource(swm).lower()
    for token in _FORBIDDEN_OUTPUT_TERMS:
        assert token not in source, token

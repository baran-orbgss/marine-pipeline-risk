"""Offline unit tests for marine_engine.morphology.sandwave_morphometry (MAR-017).

Small synthetic sinusoidal DEMs/profiles only -- never the real HHW
dataset, never network access. Lettered comments map to MAR-017 Section
29's required test list (A-L here; M/N/O live in
test_hhw_cend1111_analog.py alongside the analog-specific module).
"""

import numpy as np
import pytest
import rasterio.transform

from marine_engine.morphology import sandwave_morphometry as swm

PIXEL_SIZE_M = 1.0


def _make_sinusoidal_tile(size_px, wavelength_m, amplitude_m, azimuth_deg, pixel_size_m=1.0):
    """A synthetic sand-wave field with known wavelength/amplitude/crest
    orientation, plus a linear regional trend."""

    y, x = np.indices((size_px, size_px)).astype(np.float64) * pixel_size_m
    crest_rad = np.radians(azimuth_deg)
    kx = np.sin(crest_rad + np.pi / 2) * 2 * np.pi / wavelength_m
    ky = np.cos(crest_rad + np.pi / 2) * 2 * np.pi / wavelength_m
    bedforms = amplitude_m * np.sin(kx * x + ky * y)
    trend = 0.01 * x + 0.02 * y
    return bedforms + trend


def _make_profile(
    n, wavelength_m, amplitude_m, extra_ripple_wavelength_m=None, extra_ripple_amp=0.0
):
    d = np.arange(n, dtype=np.float64) * PIXEL_SIZE_M
    profile = amplitude_m * np.sin(2 * np.pi * d / wavelength_m) + 0.02 * d
    if extra_ripple_wavelength_m:
        profile = profile + extra_ripple_amp * np.sin(2 * np.pi * d / extra_ripple_wavelength_m)
    return d, profile


# --- A: a sinusoidal DEM of known wavelength recovers wavelength within tolerance ----------


def test_A_2d_tile_recovers_known_wavelength():
    tile = _make_sinusoidal_tile(1000, wavelength_m=150.0, amplitude_m=2.0, azimuth_deg=30.0)
    diagnostics = swm.analyze_tile(tile, np.ones_like(tile, dtype=bool), PIXEL_SIZE_M)
    assert diagnostics is not None
    assert abs(diagnostics["dominant_wavelength_m"] - 150.0) < 20.0
    assert diagnostics["directional_concentration"] > 0.8


# --- B: known sinusoidal amplitude produces correct trough-baseline wave height -----------


def test_B_known_amplitude_yields_correct_trough_baseline_height():
    d, profile = _make_profile(2000, wavelength_m=200.0, amplitude_m=3.0)
    detrended = swm.remove_linear_trend_1d(profile, d)
    filtered = swm.low_pass_filter_profile(detrended, PIXEL_SIZE_M, 30.0)
    bedforms = swm.compute_bedform_morphometrics(detrended, filtered, d)

    assert len(bedforms) >= 5
    heights = [b["wave_height_m"] for b in bedforms]
    wavelengths = [b["wavelength_m"] for b in bedforms]
    # Peak-to-trough of a pure sinusoid of amplitude 3 m is 6 m.
    assert abs(np.median(heights) - 6.0) < 1.0
    assert abs(np.median(wavelengths) - 200.0) < 15.0


# --- C: crest orientation is perpendicular to wavevector orientation -----------------------


def test_C_crest_orientation_is_perpendicular_to_wavevector():
    tile = _make_sinusoidal_tile(800, wavelength_m=120.0, amplitude_m=1.5, azimuth_deg=40.0)
    diagnostics = swm.analyze_tile(tile, np.ones_like(tile, dtype=bool), PIXEL_SIZE_M)
    assert diagnostics is not None
    wavevector = diagnostics["dominant_wavevector_azimuth_deg"]
    crest = diagnostics["dominant_crest_azimuth_deg"]
    # The crest must be exactly 90 degrees from the wavevector, modulo 180 (a crest
    # orientation is undirected, so +90 and -90 are the same value mod 180).
    expected_crest = (wavevector + 90.0) % 180.0
    diff = min(
        abs(crest - expected_crest),
        abs(crest - expected_crest + 180.0),
        abs(crest - expected_crest - 180.0),
    )
    assert diff < 1e-6


# --- D: positive-down depth sign does not invert crest/trough semantics -------------------


def test_D_positive_down_sign_convention_does_not_invert_semantics():
    d, profile = _make_profile(2000, 200.0, 3.0)
    detrended = swm.remove_linear_trend_1d(profile, d)
    filtered = swm.low_pass_filter_profile(detrended, PIXEL_SIZE_M, 30.0)
    bedforms_elevation = swm.compute_bedform_morphometrics(detrended, filtered, d)

    positive_down_profile = -profile
    detrended_pd = swm.remove_linear_trend_1d(positive_down_profile, d)
    corrected = -detrended_pd  # the caller must negate BEFORE detection
    filtered_corrected = swm.low_pass_filter_profile(corrected, PIXEL_SIZE_M, 30.0)
    bedforms_corrected = swm.compute_bedform_morphometrics(corrected, filtered_corrected, d)

    assert len(bedforms_elevation) == len(bedforms_corrected)
    assert (
        abs(
            np.median([b["wave_height_m"] for b in bedforms_elevation])
            - np.median([b["wave_height_m"] for b in bedforms_corrected])
        )
        < 0.5
    )


# --- E: rotated synthetic bedforms recover the correct orientation ------------------------


@pytest.mark.parametrize("azimuth_deg", [0.0, 45.0, 90.0, 135.0])
def test_E_rotated_bedforms_recover_orientation(azimuth_deg):
    tile = _make_sinusoidal_tile(800, wavelength_m=120.0, amplitude_m=1.5, azimuth_deg=azimuth_deg)
    diagnostics = swm.analyze_tile(tile, np.ones_like(tile, dtype=bool), PIXEL_SIZE_M)
    assert diagnostics is not None
    recovered = diagnostics["dominant_crest_azimuth_deg"]
    diff = min(
        abs(recovered - azimuth_deg),
        abs(recovered - (azimuth_deg + 180) % 180),
        abs(recovered - (azimuth_deg - 180)),
    )
    assert diff < 5.0


# --- F: planar trend removal preserves bedform wavelength ----------------------------------


def test_F_planar_trend_removal_preserves_wavelength():
    with_trend = _make_sinusoidal_tile(1000, 150.0, 2.0, 30.0)
    no_trend = with_trend - (
        0.01 * np.indices((1000, 1000))[1] + 0.02 * np.indices((1000, 1000))[0]
    )
    valid = np.ones((1000, 1000), dtype=bool)
    diag_with_trend = swm.analyze_tile(with_trend, valid, PIXEL_SIZE_M)
    diag_no_trend = swm.analyze_tile(no_trend, valid, PIXEL_SIZE_M)
    assert diag_with_trend is not None and diag_no_trend is not None
    assert (
        abs(diag_with_trend["dominant_wavelength_m"] - diag_no_trend["dominant_wavelength_m"]) < 5.0
    )


# --- G: a <30 m synthetic ripple is strongly suppressed by the canonical filter ------------


def test_G_short_wavelength_ripple_is_suppressed_by_canonical_filter():
    d, profile = _make_profile(
        2000,
        wavelength_m=200.0,
        amplitude_m=3.0,
        extra_ripple_wavelength_m=8.0,
        extra_ripple_amp=1.5,
    )
    detrended = swm.remove_linear_trend_1d(profile, d)
    filtered = swm.low_pass_filter_profile(detrended, PIXEL_SIZE_M, 30.0)
    bedforms = swm.compute_bedform_morphometrics(detrended, filtered, d)
    # Crest detection on the filtered profile should track the ~200 m sand wave, not be
    # swamped into many spurious extra bedforms by the 8 m ripple.
    assert 6 <= len(bedforms) <= 12


# --- H: a 100-300 m synthetic sand-wave wavelength is preserved ----------------------------


@pytest.mark.parametrize("wavelength_m", [100.0, 200.0, 300.0])
def test_H_sandwave_scale_wavelength_is_preserved(wavelength_m):
    d, profile = _make_profile(3000, wavelength_m=wavelength_m, amplitude_m=2.0)
    detrended = swm.remove_linear_trend_1d(profile, d)
    filtered = swm.low_pass_filter_profile(detrended, PIXEL_SIZE_M, 30.0)
    bedforms = swm.compute_bedform_morphometrics(detrended, filtered, d)
    median_wavelength = np.median([b["wavelength_m"] for b in bedforms])
    assert abs(median_wavelength - wavelength_m) < wavelength_m * 0.15


# --- I: crest/trough positions do not phase-shift materially under symmetric filtering -----


def test_I_crest_positions_do_not_phase_shift_under_symmetric_filtering():
    d, profile = _make_profile(2000, 200.0, 3.0)
    detrended = swm.remove_linear_trend_1d(profile, d)
    filtered = swm.low_pass_filter_profile(detrended, PIXEL_SIZE_M, 30.0)
    crest_idx, _trough_idx = swm.detect_extrema(filtered)
    crest_positions = d[crest_idx]
    # True crest positions of this pure sine (period 200, after linear detrend) are at
    # d = 50, 250, 450, ...
    expected = np.array([50.0, 250.0, 450.0, 650.0])
    max_displacement = max(np.min(np.abs(crest_positions - e)) for e in expected)
    assert max_displacement < 5.0


# --- J: trough-crest-trough wavelength and crest-to-crest spacing stay separate fields -----


def test_J_wavelength_and_crest_spacing_are_computed_and_reported_separately():
    d, profile = _make_profile(2000, 200.0, 3.0)
    detrended = swm.remove_linear_trend_1d(profile, d)
    filtered = swm.low_pass_filter_profile(detrended, PIXEL_SIZE_M, 30.0)
    crest_spacing = swm.compute_crest_to_crest_spacing(filtered, d)
    bedforms = swm.compute_bedform_morphometrics(detrended, filtered, d)
    assert abs(np.median(crest_spacing) - 200.0) < 15.0
    assert abs(np.median([b["wavelength_m"] for b in bedforms]) - 200.0) < 15.0


# --- K: asymmetry calculation against hand geometry is exact -------------------------------


def test_K_asymmetry_matches_hand_computed_geometry_exactly():
    x_l, x_c, x_r = 0.0, 60.0, 200.0
    crest_elev, left_trough_elev, right_trough_elev = 5.0, 0.0, 1.0
    wavelength_m = x_r - x_l
    frac = (x_c - x_l) / (x_r - x_l)
    trough_baseline = left_trough_elev + frac * (right_trough_elev - left_trough_elev)
    wave_height = crest_elev - trough_baseline
    left_half, right_half = x_c - x_l, x_r - x_c
    asymmetry = (right_half - left_half) / wavelength_m

    assert asymmetry == pytest.approx(0.4)
    assert wave_height == pytest.approx(5.0 - 0.3)


# --- L: a nodata-heavy raster is excluded from the tile search -----------------------------


def test_L_uniformly_sparse_raster_yields_no_valid_tiles():
    rng = np.random.default_rng(42)
    valid_sparse = rng.random((3000, 3000)) < 0.15  # no large dense cluster anywhere
    transform = rasterio.transform.from_origin(500000.0, 5900000.0, PIXEL_SIZE_M, PIXEL_SIZE_M)
    candidates, meta = swm.find_valid_tiles(
        valid_sparse, PIXEL_SIZE_M, transform=transform, tile_size_m=2000.0
    )
    assert candidates == []
    assert meta["tile_size_used_m"] is None


def test_L_a_dense_but_small_cluster_is_canonically_rejected_but_exploratory_found():
    """MAR-017A: a genuinely dense patch too small for the canonical
    >=1000 m floor must never be picked up by the CANONICAL search
    (`find_valid_tiles` no longer cascades below 1000 m) -- only the
    separate, explicitly-labelled exploratory search may find it."""

    valid = np.zeros((3000, 3000), dtype=bool)
    valid[:300, :300] = True  # a genuinely dense 300x300 patch -- well below 1000 m
    transform = rasterio.transform.from_origin(500000.0, 5900000.0, PIXEL_SIZE_M, PIXEL_SIZE_M)

    canonical_candidates, canonical_meta = swm.find_valid_tiles(
        valid, PIXEL_SIZE_M, transform=transform, tile_size_m=2000.0
    )
    assert canonical_candidates == []
    assert canonical_meta["tile_size_used_m"] is None

    exploratory_candidates, exploratory_meta = swm.find_exploratory_small_support_tiles(
        valid, PIXEL_SIZE_M, transform=transform, starting_size_m=500.0, absolute_floor_m=100.0
    )
    assert len(exploratory_candidates) > 0
    assert exploratory_meta["tile_size_used_m"] <= 300.0


# --- MAR-017A Section 15's own required test list (A-L) -- prefixed `mar017a_` to avoid ----
# --- colliding with MAR-017 Section 29's unrelated A-L lettering above ---------------------


def test_mar017a_A_canonical_search_cascades_to_1000m_but_never_further():
    rng = np.random.default_rng(11)
    valid = rng.random((3000, 3000)) < 0.30  # sparse everywhere...
    valid[500:1500, 500:1500] = (
        rng.random((1000, 1000)) < 0.96
    )  # ...except one dense 1000x1000 block
    transform = rasterio.transform.from_origin(500000.0, 5900000.0, PIXEL_SIZE_M, PIXEL_SIZE_M)
    _candidates, meta = swm.find_valid_tiles(valid, PIXEL_SIZE_M, transform=transform)
    sizes_tried = [entry["tile_size_m"] for entry in meta["cascade_log"]]
    assert all(size >= swm.MIN_TILE_SIZE_M for size in sizes_tried)
    assert meta["tile_size_used_m"] == swm.MIN_TILE_SIZE_M  # found only at the 1000 m floor


def test_mar017a_B_zero_qualifying_tiles_yields_a_real_empty_list_not_fake_data():
    valid_sparse = np.zeros((3000, 3000), dtype=bool)  # nothing valid anywhere
    transform = rasterio.transform.from_origin(500000.0, 5900000.0, PIXEL_SIZE_M, PIXEL_SIZE_M)
    candidates, meta = swm.find_valid_tiles(valid_sparse, PIXEL_SIZE_M, transform=transform)
    assert candidates == []
    assert meta["tile_size_used_m"] is None
    assert all(entry["outcome"] != "OK" for entry in meta["cascade_log"])


def test_mar017a_C_tile_at_2point9_wavelengths_is_not_eligible():
    assert (
        swm.meets_wavelengths_across_tile(dominant_wavelength_m=1000.0, tile_size_m=2900.0) is False
    )


def test_mar017a_D_tile_at_exactly_3point0_wavelengths_is_eligible():
    assert (
        swm.meets_wavelengths_across_tile(dominant_wavelength_m=1000.0, tile_size_m=3000.0) is True
    )


def test_mar017a_E_no_fallback_selects_an_ineligible_tile():
    pool = [
        {
            "tile_id": "only_2.9x",
            "tile_size_m": 2900.0,
            "diagnostics": {
                "dominant_wavelength_m": 1000.0,
                "directional_concentration": 0.99,
                "spectral_peak_to_median_power_ratio": 50.0,
            },
        },
        {
            "tile_id": "only_2.0x",
            "tile_size_m": 2000.0,
            "diagnostics": {
                "dominant_wavelength_m": 1000.0,
                "directional_concentration": 0.95,
                "spectral_peak_to_median_power_ratio": 40.0,
            },
        },
    ]
    selected = swm.select_canonical_eligible_tiles(pool, top_n=3)
    assert selected == []  # never forces a top_n pick from an all-ineligible pool


def test_mar017a_F_a_21m_detected_feature_is_rejected_from_canonical_output():
    bedforms = [
        {"wavelength_m": 21.1, "wave_height_m": 0.3},
        {"wavelength_m": 45.0, "wave_height_m": 1.2},
    ]
    canonical, rejected_count = swm.apply_canonical_wavelength_gate(bedforms)
    assert [b["wavelength_m"] for b in canonical] == [45.0]
    assert rejected_count == 1


def test_mar017a_G_a_30m_feature_remains_eligible_at_the_exact_boundary():
    bedforms = [{"wavelength_m": 30.0, "wave_height_m": 0.5}]
    canonical, rejected_count = swm.apply_canonical_wavelength_gate(bedforms)
    assert len(canonical) == 1
    assert rejected_count == 0


def test_mar017a_H_rejected_sub_cutoff_count_is_reported_not_dropped_silently():
    bedforms = [
        {"wavelength_m": 12.0, "wave_height_m": 0.1},
        {"wavelength_m": 21.1, "wave_height_m": 0.3},
        {"wavelength_m": 60.0, "wave_height_m": 1.5},
    ]
    canonical, rejected_count = swm.apply_canonical_wavelength_gate(bedforms)
    assert len(canonical) == 1
    assert rejected_count == 2


# --- MAR-017B Section 26's own required test list -- E/F/G land here since they exercise ---
# --- `select_spatially_independent_eligible_tiles`/`apply_canonical_wavelength_gate`, the ---
# --- same reusable-engine functions the IDRBNR analog module calls unchanged ----------------


def test_mar017b_E_spatially_independent_selection_keeps_the_3_wavelength_rule_strict():
    pool = [
        {
            "tile_id": "ineligible_2.9x",
            "tile_size_m": 2900.0,
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "diagnostics": {
                "dominant_wavelength_m": 1000.0,
                "directional_concentration": 0.99,
                "spectral_peak_to_median_power_ratio": 50.0,
            },
        }
    ]
    selected = swm.select_spatially_independent_eligible_tiles(pool, max_tiles=5)
    assert selected == []  # ranked #1 by every descriptive metric, still excluded


def test_mar017b_F_zero_eligible_tile_does_not_trigger_a_fallback_selection():
    pool = [
        {
            "tile_id": "t1",
            "tile_size_m": 2000.0,
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "diagnostics": {
                "dominant_wavelength_m": 900.0,  # 2000/900 = 2.22x -- ineligible
                "directional_concentration": 0.9,
                "spectral_peak_to_median_power_ratio": 5.0,
            },
        },
        {
            "tile_id": "t2",
            "tile_size_m": 2000.0,
            "center_x_m": 5000.0,
            "center_y_m": 5000.0,
            "diagnostics": {
                "dominant_wavelength_m": 800.0,  # 2000/800 = 2.5x -- also ineligible
                "directional_concentration": 0.5,
                "spectral_peak_to_median_power_ratio": 2.0,
            },
        },
    ]
    selected = swm.select_spatially_independent_eligible_tiles(pool, max_tiles=5)
    assert selected == []  # never forces max_tiles from an all-ineligible pool


def test_mar017b_E_a_second_overlapping_eligible_tile_is_rejected_for_independence():
    """Two tiles at the SAME location (a real, direct MAR-017B Section 13
    scenario -- e.g. a 2000 m candidate and a 1000 m candidate covering
    the same ground) must never both be selected as independent
    validation samples."""

    pool = [
        {
            "tile_id": "best",
            "tile_size_m": 2000.0,
            "center_x_m": 500000.0,
            "center_y_m": 5900000.0,
            "diagnostics": {
                "dominant_wavelength_m": 500.0,  # 2000/500 = 4.0x -- eligible
                "directional_concentration": 0.99,
                "spectral_peak_to_median_power_ratio": 50.0,
            },
        },
        {
            "tile_id": "overlapping_but_worse_ranked",
            "tile_size_m": 2000.0,
            "center_x_m": 500500.0,  # 500 m offset -- well within a 2000 m tile's footprint
            "center_y_m": 5900000.0,
            "diagnostics": {
                "dominant_wavelength_m": 500.0,
                "directional_concentration": 0.5,
                "spectral_peak_to_median_power_ratio": 2.0,
            },
        },
        {
            "tile_id": "far_away_independent",
            "tile_size_m": 2000.0,
            "center_x_m": 600000.0,  # 100 km away -- genuinely independent
            "center_y_m": 5900000.0,
            "diagnostics": {
                "dominant_wavelength_m": 500.0,
                "directional_concentration": 0.4,
                "spectral_peak_to_median_power_ratio": 1.5,
            },
        },
    ]
    selected_ids = [
        d["tile_id"] for d in swm.select_spatially_independent_eligible_tiles(pool, max_tiles=5)
    ]
    assert selected_ids == ["best", "far_away_independent"]


def test_mar017b_G_the_30m_wavelength_gate_remains_active_for_a_second_analog():
    bedforms = [
        {"wavelength_m": 18.5, "wave_height_m": 0.2},
        {"wavelength_m": 29.9, "wave_height_m": 0.4},
        {"wavelength_m": 30.0, "wave_height_m": 0.5},
        {"wavelength_m": 85.0, "wave_height_m": 2.0},
    ]
    canonical, rejected_count = swm.apply_canonical_wavelength_gate(bedforms)
    assert [b["wavelength_m"] for b in canonical] == [30.0, 85.0]
    assert rejected_count == 2


# --- MAR-022A Section 5: band-restricted spectral diagnostics ------------------------------


def test_mar022a_A_max_wavelength_none_preserves_original_behaviour_exactly():
    tile = _make_sinusoidal_tile(1000, wavelength_m=150.0, amplitude_m=2.0, azimuth_deg=30.0)
    valid = np.ones_like(tile, dtype=bool)
    residual, _trend, _coeffs = swm.remove_planar_trend(tile, valid, PIXEL_SIZE_M)
    windowed = swm.apply_hann_window_2d(residual)
    power, freq_x, freq_y = swm.compute_2d_power_spectrum(windowed, PIXEL_SIZE_M)

    baseline = swm.compute_spectral_diagnostics(power, freq_x, freq_y, 30.0)
    explicit_none = swm.compute_spectral_diagnostics(
        power, freq_x, freq_y, 30.0, max_wavelength_m=None
    )
    assert baseline == explicit_none


def test_mar022a_B_band_restriction_finds_the_true_peak_under_a_dominant_long_wavelength_trend():
    """A synthetic tile with a WEAK ~150 m sand wave riding on top of a
    much STRONGER ~1900 m (near-tile-scale) undulation the linear
    detrend cannot remove -- the global (unbounded-above) diagnostic
    must lock onto the long-wavelength artefact, while the canonical-
    band-restricted diagnostic (30-667 m, i.e. tile_size/3 for a 2000 m
    tile) must recover the real ~150 m sand wave instead."""

    size_px = 2000
    y, x = np.indices((size_px, size_px)).astype(np.float64) * PIXEL_SIZE_M
    long_wavelength_m = 1900.0
    sand_wave = 1.0 * np.sin(2 * np.pi * x / 150.0)
    long_trend = 20.0 * np.sin(2 * np.pi * x / long_wavelength_m)
    tile = sand_wave + long_trend
    valid = np.ones_like(tile, dtype=bool)

    global_diag, band_diag = swm.analyze_tile_dual_band(
        tile, valid, PIXEL_SIZE_M, max_wavelength_m=2000.0 / 3.0
    )
    assert global_diag is not None
    assert band_diag is not None
    assert global_diag["dominant_wavelength_m"] > 667.0
    assert abs(band_diag["dominant_wavelength_m"] - 150.0) < 20.0


def test_mar022a_C_empty_band_returns_none_never_the_global_peak():
    tile = _make_sinusoidal_tile(1000, wavelength_m=800.0, amplitude_m=2.0, azimuth_deg=0.0)
    valid = np.ones_like(tile, dtype=bool)
    # A band with max_wavelength_m below cutoff_wavelength_m is deliberately empty.
    global_diag, band_diag = swm.analyze_tile_dual_band(
        tile, valid, PIXEL_SIZE_M, max_wavelength_m=10.0
    )
    assert global_diag is not None
    assert band_diag is None


def test_mar022a_D_dual_band_matches_two_separate_calls():
    tile = _make_sinusoidal_tile(600, wavelength_m=90.0, amplitude_m=1.2, azimuth_deg=15.0)
    valid = np.ones_like(tile, dtype=bool)
    global_diag, band_diag = swm.analyze_tile_dual_band(
        tile, valid, PIXEL_SIZE_M, max_wavelength_m=200.0
    )
    separate_global = swm.analyze_tile(tile, valid, PIXEL_SIZE_M)
    separate_band = swm.analyze_tile(tile, valid, PIXEL_SIZE_M, max_wavelength_m=200.0)
    assert global_diag == separate_global
    assert band_diag == separate_band


# --- Shared: no forbidden downstream-modelling terms in this module's own vocabulary -------


def test_module_never_computes_a_migration_rate_or_score():
    import inspect

    source = inspect.getsource(swm)
    forbidden = ("migration_rate", "susceptibility", "risk_score", "freespan_probability")
    for token in forbidden:
        assert token not in source.lower()

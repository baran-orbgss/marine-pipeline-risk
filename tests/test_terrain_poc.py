"""Offline unit tests for marine_engine.terrain (MAR-020).

Small synthetic arrays/facts only -- never the real Sheringham Shoal
raster, never network access. Lettered test names map to MAR-020 Section
20's required test list (A-M); a few numbered follow-ups (e.g. H2) add
coverage the lettered list implies but doesn't spell out.
"""

from __future__ import annotations

import inspect
import json
import socket

import numpy as np
import pytest
import requests

from marine_engine import cli
from marine_engine.providers.bathymetry import sheringham_shoal_2020 as sheringham_provider
from marine_engine.terrain import canonical, contract, derivatives, raster_io, readiness
from marine_engine.terrain import maps as terrain_maps
from marine_engine.terrain import report as terrain_report

CELL_M = 1.0

TERRAIN_MODULES = (
    canonical,
    contract,
    derivatives,
    raster_io,
    readiness,
    terrain_maps,
    terrain_report,
)


def _code_only_source(module) -> str:
    """Module source with its own leading module-docstring stripped. Every
    terrain module's docstring explicitly explains what it does NOT depend
    on / is NOT specific to (genericity is the whole point of this
    package), which necessarily names the very things a naive substring
    search is trying to rule out -- e.g. derivatives.py's docstring
    explains it mirrors-but-never-imports `morphology.regional`, and
    disclaims PL854/Sheringham-specificity in prose. Only actual CODE
    should be scanned for a real dependency or hardcoded identifier."""

    source = inspect.getsource(module)
    stripped = source.lstrip()
    if stripped.startswith('"""'):
        end = stripped.find('"""', 3)
        if end != -1:
            return stripped[end + 3 :]
    return source


# --- A: no PL854 dependency in the generic terrain engine ------------------------------------


def test_A_terrain_package_has_no_pl854_dependency():
    """Every module under marine_engine.terrain is generic -- none of them
    IMPORT PL854-scoped code (evidence_atlas, morphology.regional, or the
    pl854 provider), by construction, not just by convention. Scoped to
    actual `import`/`from` statements (not prose) so a docstring that
    explains this module deliberately does NOT import something doesn't
    trip its own independence check."""

    forbidden_imports = ("evidence_atlas", "morphology.regional", "providers.pl854", "pl854_")
    for module in TERRAIN_MODULES:
        code = _code_only_source(module)
        import_lines = "\n".join(
            line for line in code.splitlines() if line.strip().startswith(("import ", "from "))
        )
        for fragment in forbidden_imports:
            assert fragment not in import_lines, f"{module.__name__} imports {fragment!r}"


# --- B: no hard-coded Sheringham coordinates in the generic engine ---------------------------


def test_B_terrain_package_has_no_hardcoded_sheringham_identifiers():
    """The generic terrain/ package must stay usable for ANY future project
    -- Sheringham-specific names, series IDs, or coordinates belong only in
    the provider module, never in derivatives/canonical/contract/maps/
    readiness/raster_io/report CODE (docstrings legitimately discuss the
    real benchmark used to validate this module -- see `_code_only_source`)."""

    forbidden_fragments = ("sheringham", "TCE-1986", "G201193", "32631")
    for module in TERRAIN_MODULES:
        code = _code_only_source(module).lower()
        for fragment in forbidden_fragments:
            assert fragment.lower() not in code, f"{module.__name__} references {fragment!r}"


# --- C: no risk/susceptibility/hazard-score fields anywhere in terrain outputs ----------------


def test_C_no_risk_or_susceptibility_fields_in_terrain_outputs():
    forbidden_patterns = (
        "risk score",
        "risk_score",
        "susceptibility",
        "scour",
        "freespan",
        "route suitability",
        "hazard map",
    )
    contract_text = json.dumps(contract.build_terrain_input_contract()).lower()
    for module in (canonical, contract, derivatives, readiness):
        source = inspect.getsource(module).lower()
        for fragment in forbidden_patterns:
            assert fragment not in source, f"{module.__name__} references {fragment!r}"
    for fragment in forbidden_patterns:
        assert fragment not in contract_text, fragment


def test_C2_cli_terrain_command_writes_no_risk_field_into_validation_json():
    """Checks for actual dict/JSON-key-shaped literals, never a bare noun --
    the function's own `limitations` list legitimately contains disclaiming
    prose like 'no ... susceptibility interpretation is made', which is the
    point of this ticket, not a violation of it."""

    source = inspect.getsource(cli._cmd_build_highres_terrain_poc)
    forbidden_field_literals = (
        '"risk_score"',
        '"susceptibility_score"',
        '"susceptibility_index"',
        '"scour_score"',
        '"freespan_score"',
        '"hazard_score"',
        '"route_suitability"',
    )
    for forbidden in forbidden_field_literals:
        assert forbidden not in source


# --- D: CRS QA works ---------------------------------------------------------------------------


def _base_facts(**overrides) -> readiness.RasterFacts:
    defaults = {
        "band_count": 1,
        "dtype": "float32",
        "color_interpretations": ("gray",),
        "crs_is_present": True,
        "crs_is_geographic": False,
        "crs_linear_units": "metre",
        "width": 100,
        "height": 100,
        "pixel_size_x_m": 1.0,
        "pixel_size_y_m": 1.0,
        "bounds": (500000.0, 5900000.0, 500100.0, 5900100.0),
        "nodata_value": -9999.0,
        "vertical_datum": "LAT (Lowest Astronomical Tide)",
        "survey_epoch": "2020",
        "data_min": -24.0,
        "data_max": -3.0,
        "data_std": 2.5,
        "valid_cell_fraction": 0.5,
    }
    defaults.update(overrides)
    return readiness.RasterFacts(**defaults)


def test_D_crs_qa_rejects_geographic_crs():
    result = readiness.assess_bathymetry_readiness(_base_facts(crs_is_geographic=True))
    assert result.status == readiness.NOT_READY
    assert any("crs_projected_metric" in r for r in result.reasons())


def test_D2_crs_qa_rejects_missing_crs():
    result = readiness.assess_bathymetry_readiness(
        _base_facts(crs_is_present=False, crs_is_geographic=None, crs_linear_units=None)
    )
    assert result.status == readiness.NOT_READY
    assert any("crs_present" in r for r in result.reasons())


def test_D3_crs_qa_passes_a_valid_projected_metric_raster():
    result = readiness.assess_bathymetry_readiness(_base_facts())
    assert result.status == readiness.READY


def test_D4_passing_crs_present_check_never_carries_the_failure_detail_text():
    """Regression: `crs_present`'s detail string used to be hard-coded to
    the FAILURE message regardless of pass/fail -- a real run against the
    actual Sheringham raster produced the self-contradictory
    `passed: true, detail: "no CRS found on the raster"`."""

    result = readiness.assess_bathymetry_readiness(_base_facts())
    crs_present_check = next(c for c in result.checks if c.check_id == "crs_present")
    assert crs_present_check.passed is True
    assert "no CRS found" not in crs_present_check.detail


def test_D5_every_check_passed_value_is_a_native_bool_not_a_numpy_bool():
    """Regression: `resolution_known`/`range_plausible` ended an `and`
    chain with `np.isfinite(...)`, so `passed` was `numpy.bool_` for those
    two checks. `json.dumps(..., default=str)` then serialized them as the
    STRING "True"/"False" instead of a JSON boolean -- confirmed in a real
    run's `bathymetry_readiness.json` (a "machine-readable reasons"
    document per the ticket, where that distinction matters)."""

    result = readiness.assess_bathymetry_readiness(_base_facts())
    for check in result.checks:
        assert type(check.passed) is bool, f"{check.check_id}: passed is {type(check.passed)!r}"
    serialized = json.dumps(result.to_dict())
    assert '"True"' not in serialized
    assert '"False"' not in serialized


# --- E: vertical datum metadata preserved end-to-end ------------------------------------------


def test_E_vertical_datum_flows_through_canonical_conversion_unchanged():
    raw = np.array([[-5.0, -10.0], [-15.0, -9999.0]], dtype=np.float32)
    result = canonical.build_canonical_bed_elevation(
        raw,
        nodata_value=-9999.0,
        source_sign_convention=canonical.ALREADY_ELEVATION_STYLE,
        source_vertical_datum="LAT (Lowest Astronomical Tide)",
    )
    assert result.source_vertical_datum == "LAT (Lowest Astronomical Tide)"


def test_E2_missing_vertical_datum_is_a_limitation_not_a_blocker():
    result = readiness.assess_bathymetry_readiness(_base_facts(vertical_datum=None))
    assert result.status == readiness.READY_WITH_LIMITATIONS
    assert any("vertical_datum_known" in r for r in result.reasons())


# --- F: sign-convention conversion is correct in BOTH directions ------------------------------


def test_F_positive_down_depth_is_negated_to_canonical_elevation():
    raw = np.array([[5.0, 10.0], [15.0, -9999.0]], dtype=np.float64)
    result = canonical.build_canonical_bed_elevation(
        raw,
        nodata_value=-9999.0,
        source_sign_convention=canonical.POSITIVE_DOWN_DEPTH,
        source_vertical_datum="MSL",
    )
    assert result.bed_elevation_m[0, 0] == -5.0
    assert result.bed_elevation_m[0, 1] == -10.0
    assert result.bed_elevation_m[1, 0] == -15.0
    assert np.isnan(result.bed_elevation_m[1, 1])
    # Deeper source depth (larger positive) must map to a LOWER canonical elevation.
    assert result.bed_elevation_m[1, 0] < result.bed_elevation_m[0, 0]


def test_F2_already_elevation_style_source_is_never_flipped():
    """The REAL Sheringham raster turned out to already be elevation-style
    (verified by direct rasterio inspection, not the filename) -- this is
    the zero-sign-flip branch that case actually exercises."""

    raw = np.array([[-5.0, -10.0], [-15.0, -9999.0]], dtype=np.float64)
    result = canonical.build_canonical_bed_elevation(
        raw,
        nodata_value=-9999.0,
        source_sign_convention=canonical.ALREADY_ELEVATION_STYLE,
        source_vertical_datum="LAT (Lowest Astronomical Tide)",
    )
    assert result.bed_elevation_m[0, 0] == -5.0
    assert result.bed_elevation_m[0, 1] == -10.0
    assert result.bed_elevation_m[1, 0] == -15.0
    assert np.isnan(result.bed_elevation_m[1, 1])


def test_F3_sign_convention_plausibility_evidence_flags_a_mismatched_declaration():
    evidence = canonical.infer_source_sign_convention_evidence(
        raw_min=-24.4,
        raw_max=-3.2,
        raw_mean=-18.0,
        declared_convention=canonical.POSITIVE_DOWN_DEPTH,
    )
    assert evidence.plausible is False


def test_F4_sign_convention_plausibility_evidence_accepts_the_real_sheringham_case():
    evidence = canonical.infer_source_sign_convention_evidence(
        raw_min=-24.397,
        raw_max=-3.189,
        raw_mean=-18.03,
        declared_convention=canonical.ALREADY_ELEVATION_STYLE,
    )
    assert evidence.plausible is True


# --- G: no resolution inflation (native resolution used as-is, never upsampled) ---------------


def test_G_terrain_package_never_calls_a_resampling_or_reprojection_function():
    forbidden_calls = ("scipy.ndimage.zoom", ".zoom(", "cv2.resize", "rasterio.warp", ".reproject(")
    for module in TERRAIN_MODULES:
        source = inspect.getsource(module)
        for fragment in forbidden_calls:
            assert fragment not in source, f"{module.__name__} calls {fragment!r}"


def test_G2_write_terrain_raster_preserves_the_input_array_shape_exactly(tmp_path):
    import rasterio

    array = np.random.default_rng(0).normal(size=(37, 53)).astype(np.float32)
    transform = rasterio.Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 5900000.0)
    output_path = raster_io.write_terrain_raster(
        array, transform, "EPSG:32631", tmp_path / "out.tif", {"layer": "test"}
    )
    with rasterio.open(output_path) as ds:
        assert (ds.height, ds.width) == array.shape
        assert ds.transform == transform


# --- H: physical-metre scale parameters (no hidden pixel counts) ------------------------------


def test_H_relief_window_is_physically_invariant_across_native_pixel_sizes():
    """A fixed PHYSICAL radius must give the same relief answer regardless
    of native pixel size -- formalizes the scratchpad validation case."""

    def make_bowl(cell_size, extent_m=120.0):
        n_px = int(extent_m / cell_size)
        yy, xx = np.mgrid[0:n_px, 0:n_px]
        x = (xx - n_px // 2).astype(float) * cell_size
        y = -(yy - n_px // 2).astype(float) * cell_size
        return 0.002 * (x**2 + y**2), n_px // 2

    radius_m = 30.0
    bowl_1m, c_1m = make_bowl(1.0)
    bowl_2m, c_2m = make_bowl(2.0)
    relief_1m, _ = derivatives.compute_local_relief(
        bowl_1m, np.ones_like(bowl_1m, dtype=bool), radius_m, 1.0
    )
    relief_2m, _ = derivatives.compute_local_relief(
        bowl_2m, np.ones_like(bowl_2m, dtype=bool), radius_m, 2.0
    )
    rel_diff = abs(relief_1m[c_1m, c_1m] - relief_2m[c_2m, c_2m]) / relief_1m[c_1m, c_1m]
    assert rel_diff < 0.05


def test_H2_checkerboard_tri_matches_the_exact_finite_window_closed_form():
    """Every derivative in this module is parameterized by physical radius,
    never a fixed pixel window -- and the square-window engine's exact
    checkerboard TRI closed form (A*sqrt(2*(1-1/window_size^2))) is a
    strong, radius-independent correctness signal for the whole windowed-
    moment machinery."""

    amplitude = 3.0
    n = 81
    yy, xx = np.mgrid[0:n, 0:n]
    checker = np.where((yy + xx) % 2 == 0, amplitude, -amplitude).astype(float)
    valid = np.ones_like(checker, dtype=bool)
    for radius_m in (5.0, 10.0, 15.0):
        radius_px = max(1, round(radius_m / CELL_M))
        window_size = 2 * radius_px + 1
        expected = amplitude * np.sqrt(2.0 * (1.0 - 1.0 / window_size**2))
        tri, _ = derivatives.compute_ruggedness(
            checker, valid, radius_m=radius_m, cell_size_m=CELL_M
        )
        assert abs(tri[n // 2, n // 2] - expected) < 1e-9


# --- I: curvature synthetic case (bowl vs dome sign) -------------------------------------------


def test_I_paraboloid_curvature_sign_distinguishes_valley_from_hill():
    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - n // 2).astype(float) * CELL_M
    y = -(yy - n // 2).astype(float) * CELL_M
    c = n // 2 + 8  # off the exact apex, where the gradient (and hence the profile/plan
    # decomposition) is genuinely nonzero and well-defined

    bowl = 0.01 * (x**2 + y**2)
    dome = -0.01 * (x**2 + y**2)
    valid = np.ones((n, n), dtype=bool)

    profile_bowl, plan_bowl, _ = derivatives.compute_profile_plan_curvature(
        bowl, valid, step_m=5.0, cell_size_m=CELL_M
    )
    profile_dome, plan_dome, _ = derivatives.compute_profile_plan_curvature(
        dome, valid, step_m=5.0, cell_size_m=CELL_M
    )

    assert np.sign(profile_bowl[c, c]) != np.sign(profile_dome[c, c])
    assert np.sign(plan_bowl[c, c]) != np.sign(plan_dome[c, c])


def test_I2_curvature_is_undefined_zero_at_a_perfectly_symmetric_apex():
    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - n // 2).astype(float) * CELL_M
    y = -(yy - n // 2).astype(float) * CELL_M
    c = n // 2
    bowl = 0.01 * (x**2 + y**2)
    valid = np.ones((n, n), dtype=bool)
    profile, plan, _ = derivatives.compute_profile_plan_curvature(
        bowl, valid, step_m=5.0, cell_size_m=CELL_M
    )
    assert profile[c, c] == 0.0
    assert plan[c, c] == 0.0


# --- J: slope/aspect synthetic case -------------------------------------------------------------


def test_J_planar_slope_and_aspect_are_recovered_exactly():
    n = 61
    a_true, b_true = 0.05, 0.03
    yy, xx = np.mgrid[0:n, 0:n]
    x_m = xx.astype(float) * CELL_M
    y_m = -yy.astype(float) * CELL_M
    elevation = a_true * x_m + b_true * y_m
    valid = np.ones_like(elevation, dtype=bool)

    slope_deg, aspect_deg, _ = derivatives.compute_slope_aspect_deg(
        elevation, valid, radius_m=10.0, cell_size_m=CELL_M
    )
    center = n // 2
    expected_slope = np.degrees(np.arctan(np.sqrt(a_true**2 + b_true**2)))
    expected_aspect = np.degrees(np.arctan2(-a_true, -b_true)) % 360.0
    assert abs(slope_deg[center, center] - expected_slope) < 1e-6
    assert abs(aspect_deg[center, center] - expected_aspect) < 1e-6


@pytest.mark.parametrize(
    "a,b,expected_bearing",
    [
        (-0.05, 0.0, 90.0),
        (0.0, -0.05, 0.0),
        (0.0, 0.05, 180.0),
        (0.05, 0.0, 270.0),
    ],
)
def test_J2_cardinal_aspect_cases(a, b, expected_bearing):
    n = 61
    yy, xx = np.mgrid[0:n, 0:n]
    x_m = xx.astype(float) * CELL_M
    y_m = -yy.astype(float) * CELL_M
    elevation = a * x_m + b * y_m
    valid = np.ones_like(elevation, dtype=bool)
    _, aspect_deg, _ = derivatives.compute_slope_aspect_deg(
        elevation, valid, radius_m=10.0, cell_size_m=CELL_M
    )
    center = n // 2
    got = aspect_deg[center, center] % 360.0
    diff = min(abs(got - expected_bearing), 360 - abs(got - expected_bearing))
    assert diff < 1e-6


def test_J3_flat_surface_has_zero_slope_and_undefined_aspect():
    flat = np.full((41, 41), 5.0)
    valid = np.ones_like(flat, dtype=bool)
    slope, aspect, _ = derivatives.compute_slope_aspect_deg(
        flat, valid, radius_m=10.0, cell_size_m=CELL_M
    )
    center = 20
    assert abs(slope[center, center]) < 1e-9
    assert np.isnan(aspect[center, center])


# --- K: nodata does not contaminate derivatives -------------------------------------------------


def _half_nodata_flat_scene():
    n = 61
    elevation = np.full((n, n), 10.0)
    valid = np.ones((n, n), dtype=bool)
    valid[:, n // 2 :] = False
    elevation[~valid] = -9999.0  # sentinel garbage that must never leak into any statistic
    return elevation, valid, n


def test_K_slope_is_not_contaminated_by_nodata_sentinel_values():
    elevation, valid, n = _half_nodata_flat_scene()
    slope, _, vf = derivatives.compute_slope_aspect_deg(
        elevation, valid, radius_m=10.0, cell_size_m=CELL_M
    )
    far_left = (n // 2, 15)
    assert vf[far_left] == 1.0
    assert abs(slope[far_left]) < 1e-9


def test_K2_low_validity_cells_near_a_nodata_boundary_become_nan_not_biased():
    elevation, valid, n = _half_nodata_flat_scene()
    slope, _, vf = derivatives.compute_slope_aspect_deg(
        elevation, valid, radius_m=10.0, cell_size_m=CELL_M
    )
    boundary = (n // 2, n // 2 - 1)
    assert vf[boundary] < derivatives.MIN_VALID_NEIGHBORHOOD_FRACTION
    assert np.isnan(slope[boundary])


def test_K3_std_and_tri_are_not_contaminated_by_nodata_sentinel_values():
    elevation, valid, n = _half_nodata_flat_scene()
    std, _ = derivatives.compute_terrain_std(elevation, valid, radius_m=10.0, cell_size_m=CELL_M)
    tri, _ = derivatives.compute_ruggedness(elevation, valid, radius_m=10.0, cell_size_m=CELL_M)
    far_left = (n // 2, 15)
    assert abs(std[far_left]) < 1e-9
    assert abs(tri[far_left]) < 1e-9


# --- L: route/KP view is never fabricated when no authoritative route is supplied -------------


def test_L_route_kp_status_is_always_the_fixed_not_applicable_literal():
    source = inspect.getsource(cli._cmd_build_highres_terrain_poc)
    assert '"NOT_APPLICABLE_NO_AUTHORITATIVE_ROUTE_SUPPLIED"' in source
    # `route_kp_status` must be assigned this literal exactly once and never reassigned to
    # anything derived from a geometry/centerline -- i.e. it is a constant, not a computed value.
    assignments = [
        line.strip() for line in source.splitlines() if line.strip().startswith("route_kp_status")
    ]
    assert len(assignments) == 1
    assert assignments[0] == 'route_kp_status = "NOT_APPLICABLE_NO_AUTHORITATIVE_ROUTE_SUPPLIED"'


def test_L2_question_c_route_availability_is_never_assigned_true():
    source = inspect.getsource(cli._cmd_build_highres_terrain_poc)
    assert '"question_c_route_kp_view_available": False' in source
    assert '"question_c_route_kp_view_available": True' not in source


def test_L3_terrain_engine_never_constructs_a_centerline_geometry():
    forbidden = ("LineString(", "centerline", "route_geometry")
    for module in (derivatives, canonical, readiness, contract):
        source = inspect.getsource(module)
        for fragment in forbidden:
            assert fragment not in source


# --- M: outputs remain offline on rerun (no network I/O on a cache hit) -----------------------


def test_M_cached_acquisition_performs_zero_network_io(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    target_bytes = b"fake geotiff bytes for a cache-hit test"
    local_path = raw_dir / sheringham_provider.TARGET_ENTRY_NAME
    local_path.write_bytes(target_bytes)
    sidecar_path = sheringham_provider._acquisition_sidecar_path(local_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "bundle_total_bytes": 1_007_697_908,
                "target_entry_compressed_bytes": 153_567_376,
                "target_entry_bytes": len(target_bytes),
                "retrieved_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    def _blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted on a cached-acquisition rerun")

    monkeypatch.setattr(socket, "socket", _blocked)

    result = sheringham_provider.download_sheringham_shoal_bathymetry(raw_dir)

    assert result.already_cached is True
    assert result.target_entry_bytes == len(target_bytes)
    assert result.bundle_total_bytes == 1_007_697_908


def test_M2_a_stale_or_missing_sidecar_falls_back_to_a_real_acquisition_attempt(
    tmp_path, monkeypatch
):
    """If the cached raster exists but its sidecar metadata is missing, the
    provider must NOT silently fabricate bundle metadata -- it must attempt
    a real (network) acquisition, which this test observes indirectly by
    confirming network access WAS attempted (and fails fast).

    Patches `requests.Session.head` (what `RemoteZipReader.__init__`
    actually calls), NOT `socket.socket` -- a `socket.socket` patch only
    intercepts AFTER DNS resolution, so in an environment with flaky or
    absent real DNS/network reachability, `getaddrinfo` can raise its own
    (unrelated) exception first and never reach the patched constructor,
    making the test genuinely flaky (observed: passed in isolation, failed
    inside the full suite). Patching at the `requests` call boundary is
    deterministic regardless of the sandbox's real network posture."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / sheringham_provider.TARGET_ENTRY_NAME).write_bytes(b"stale file, no sidecar")

    def _blocked(*_args, **_kwargs):
        raise AssertionError("network attempted")

    monkeypatch.setattr(requests.Session, "head", _blocked)

    with pytest.raises(AssertionError, match="network attempted"):
        sheringham_provider.download_sheringham_shoal_bathymetry(raw_dir)

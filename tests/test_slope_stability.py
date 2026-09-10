"""Offline tests for marine_engine.slope_stability (MAR-031).

Small synthetic arrays / rasters generated inside the tests -- never the real Sheringham Shoal
raster, never network access. Numbered test names map to the MAR-031 Section 34 test matrix
(1-43); lettered follow-ups add invariants the matrix implies.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import re
import tokenize
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from pydantic import ValidationError

from marine_engine import cli
from marine_engine.slope_stability import contract, core, manifest, maps, report, screening
from marine_engine.terrain import raster_io as terrain_raster_io

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_MODULES = (core, contract, manifest, maps, report, screening)

CRS = "EPSG:32631"
CELL_M = 1.0
ORIGIN_X, ORIGIN_Y = 371143.0, 5894017.0

DEG30 = 30.0
GAMMA_KN = 8.0
DEPTH_M = 2.0
# s_u chosen so FS == 1 exactly at 30 deg for gamma' = 8 kN/m3, z = 2 m (Section 24) -- computed
# at full float precision, never the rounded 6.928203 text value.
SU_AT_FS1_KPA = (
    GAMMA_KN
    * 1000.0
    * DEPTH_M
    * math.sin(math.radians(DEG30))
    * math.cos(math.radians(DEG30))
    / 1000.0
)


def _scenario(
    scenario_id: str = "synthetic_case",
    su_kpa: float = SU_AT_FS1_KPA,
    gamma_kn_m3: float = GAMMA_KN,
    depth_m: float = DEPTH_M,
) -> core.UndrainedInfiniteSlopeScenario:
    return core.UndrainedInfiniteSlopeScenario(
        scenario_id=scenario_id,
        material_model=core.MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE,
        parameter_basis=core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO,
        undrained_shear_strength_kpa=su_kpa,
        submerged_unit_weight_kn_m3=gamma_kn_m3,
        slip_surface_depth_m=depth_m,
    )


def _code_only_source(module) -> str:
    source = inspect.getsource(module)
    stripped = source.lstrip()
    if stripped.startswith('"""'):
        end = stripped.find('"""', 3)
        if end != -1:
            return stripped[end + 3 :]
    return source


def _code_identifiers(module) -> set[str]:
    """Every NAME token in the module's CODE -- string literals, docstrings and comments are
    excluded, so disclaiming prose (e.g. a limitation stating that BGS Folk class is NOT used)
    can never trip a test that is looking for an actual implementation."""

    source = inspect.getsource(module)
    names: set[str] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME:
            names.add(token.string)
    return names


# --- Synthetic canonical terrain fixture ----------------------------------------------------------


def _synthetic_bed_elevation(height: int = 260, width: int = 240) -> np.ndarray:
    """A plane tilted ~10 deg in the west half, perfectly flat in the east half, with a NaN hole.
    Elevation-style (higher = shallower), all negative like a real submerged site."""

    rows, cols = np.mgrid[0:height, 0:width]
    x_m = cols * CELL_M
    bed = np.full((height, width), -20.0, dtype=np.float64)
    west = cols < width // 2
    bed[west] = -25.0 + math.tan(math.radians(10.0)) * x_m[west]
    bed[10:16, 5:12] = np.nan  # nodata hole inside the sloping region
    return bed


def _write_canonical(study_dir: Path, bed: np.ndarray, *, tags: dict | None = None) -> Path:
    transform = rasterio.Affine(CELL_M, 0.0, ORIGIN_X, 0.0, -CELL_M, ORIGIN_Y)
    default_tags = {
        "product": "MAR-020 generic high-resolution seabed terrain POC",
        "scientific_role": contract.SOURCE_TERRAIN_ROLE_REQUIRED,
        "layer": contract.SOURCE_TERRAIN_LAYER_REQUIRED,
        "units": "m",
        "convention": "higher=shallower",
        "source_dataset": "synthetic test terrain",
        "source_sha256": "0" * 64,
        "source_sign_convention": "ALREADY_ELEVATION_STYLE",
        "source_vertical_datum": "TEST DATUM",
    }
    return terrain_raster_io.write_terrain_raster(
        bed,
        transform,
        CRS,
        study_dir / screening.TERRAIN_SUBDIR / contract.CANONICAL_TERRAIN_FILENAME,
        tags if tags is not None else default_tags,
    )


@pytest.fixture
def synthetic_study(tmp_path: Path) -> Path:
    study_dir = tmp_path / "processed" / "synthetic_site"
    _write_canonical(study_dir, _synthetic_bed_elevation())
    return study_dir


def _run(study_dir: Path, **kwargs) -> screening.SlopeInstabilityScreeningResult:
    kwargs.setdefault("render_figure", False)
    kwargs.setdefault("log", lambda _m: None)
    return screening.run_slope_instability_screening(
        project_id="synthetic_site", study_dir=study_dir, **kwargs
    )


# --- 1-5: normalized strength demand --------------------------------------------------------------


def test_01_normalized_demand_is_exactly_sin_cos():
    slopes = np.linspace(0.0, 90.0, 181)
    demand = core.compute_normalized_strength_demand(slopes)
    alpha = np.radians(slopes)
    np.testing.assert_array_equal(demand, np.sin(alpha) * np.cos(alpha))


def test_02_zero_degrees_gives_zero_demand():
    assert float(core.compute_normalized_strength_demand(0.0)) == 0.0


def test_03_forty_five_degrees_gives_half():
    assert float(core.compute_normalized_strength_demand(45.0)) == pytest.approx(0.5, abs=1e-15)


def test_03b_thirty_degrees_gives_sin30_cos30():
    expected = math.sin(math.radians(30.0)) * math.cos(math.radians(30.0))
    assert float(core.compute_normalized_strength_demand(30.0)) == pytest.approx(
        expected, abs=1e-15
    )


def test_04_nan_propagates():
    demand = core.compute_normalized_strength_demand(np.array([np.nan, 10.0, np.nan]))
    assert np.isnan(demand[0]) and np.isnan(demand[2]) and np.isfinite(demand[1])


def test_05_demand_is_never_negative_and_never_above_half():
    demand = core.compute_normalized_strength_demand(np.linspace(0.0, 90.0, 9001))
    assert np.all(demand >= 0.0)
    assert np.all(demand <= 0.5 + 1e-15)


def test_05b_out_of_range_or_infinite_slope_fails_explicitly():
    for bad in ([-1.0], [90.5], [np.inf], [-np.inf]):
        with pytest.raises(core.SlopeStabilityInputError):
            core.compute_normalized_strength_demand(np.array(bad))


# --- 6, 25, 30: forbidden vocabulary / no classes -------------------------------------------------


def test_06_no_slope_hazard_classes_anywhere():
    forbidden_literals = (
        '"LOW"',
        '"MODERATE"',
        '"MEDIUM"',
        '"HIGH"',
        '"VERY_HIGH"',
        '"VERY HIGH"',
        '"CRITICAL"',
        "critical_slope",
        "susceptibility_class",
        "hazard_class",
    )
    for module in PACKAGE_MODULES:
        source = inspect.getsource(module)
        for literal in forbidden_literals:
            assert literal not in source, f"{module.__name__} contains {literal!r}"
    assert set(core.MODEL_STATES) == {
        "MODEL_FS_BELOW_1",
        "MODEL_FS_AT_1",
        "MODEL_FS_ABOVE_1",
        "NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR",
        "NOT_EVALUABLE",
    }


def test_25_no_safe_unsafe_or_risk_verdict_vocabulary(synthetic_study: Path):
    pattern = re.compile(r"\b(SAFE|UNSAFE|LOW RISK|HIGH RISK|LOW_RISK|HIGH_RISK)\b")
    for module in PACKAGE_MODULES:
        assert not pattern.search(inspect.getsource(module)), module.__name__
    result = _run(synthetic_study, scenarios=[_scenario()])
    for path in (result.metadata_path, result.readiness_path, result.contract_path):
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name


def test_30_no_weighted_overlay_quantile_or_statistical_susceptibility():
    identifiers = {n.lower() for n in set().union(*(_code_identifiers(m) for m in PACKAGE_MODULES))}
    for fragment in ("overlay", "quantile", "sklearn", "susceptib", "logistic", "weight_factor"):
        offenders = [n for n in identifiers if fragment in n]
        assert offenders == [], (fragment, offenders)


# --- 7, 43: scales stay separate; determinism -----------------------------------------------------


def test_07_ten_and_fifty_metre_scales_remain_separate(synthetic_study: Path):
    result = _run(synthetic_study, slope_scales_m=[10.0, 50.0])
    assert [r["scale_m"] for r in result.scale_results] == [10.0, 50.0]
    p10 = Path(result.outputs["normalized_strength_demand_10m"])
    p50 = Path(result.outputs["normalized_strength_demand_50m"])
    assert p10.exists() and p50.exists() and p10 != p50
    with rasterio.open(p10) as a, rasterio.open(p50) as b:
        d10, d50 = a.read(1), b.read(1)
    assert not np.array_equal(d10, d50, equal_nan=True)
    metadata_keys = json.dumps(result.metadata).lower()
    for forbidden in (
        "combined_demand",
        "max_demand",
        "mean_demand_across_scales",
        "canonical_scale",
    ):
        assert forbidden not in metadata_keys
    assert result.metadata["scale_semantics"] == "SLOPE_SCALE_SENSITIVITY_NOT_FAILURE_SURFACE_SCALE"


def test_43_identical_input_gives_identical_output():
    bed = _synthetic_bed_elevation()
    valid = np.isfinite(bed)
    s1, d1 = screening.compute_scale_slope_and_demand(bed, valid, 10.0, CELL_M)
    s2, d2 = screening.compute_scale_slope_and_demand(bed, valid, 10.0, CELL_M)
    assert np.array_equal(s1, s2, equal_nan=True)
    assert np.array_equal(d1, d2, equal_nan=True)


def test_43b_row_tiled_slope_is_identical_to_untiled_slope():
    bed = _synthetic_bed_elevation(height=260, width=60)
    valid = np.isfinite(bed)
    untiled, _ = screening.compute_scale_slope_and_demand(
        bed, valid, 10.0, CELL_M, max_tile_cells=None
    )
    tiled, _ = screening.compute_scale_slope_and_demand(
        bed, valid, 10.0, CELL_M, max_tile_cells=60 * 37
    )
    assert np.array_equal(untiled, tiled, equal_nan=True)


# --- 8-12: array immutability, raster integrity ---------------------------------------------------


def test_08_terrain_arrays_are_not_mutated():
    bed = _synthetic_bed_elevation()
    valid = np.isfinite(bed)
    bed_copy, valid_copy = bed.copy(), valid.copy()
    slope, _ = screening.compute_scale_slope_and_demand(bed, valid, 10.0, CELL_M)
    slope_copy = slope.copy()
    core.compute_normalized_strength_demand(slope)
    core.compute_scenario_factor_of_safety(slope, _scenario())
    assert np.array_equal(bed, bed_copy, equal_nan=True)
    assert np.array_equal(valid, valid_copy)
    assert np.array_equal(slope, slope_copy, equal_nan=True)


def test_09b_canonical_nodata_cells_never_receive_a_slope_or_demand_value():
    bed = _synthetic_bed_elevation()
    valid = np.isfinite(bed)
    slope, demand = screening.compute_scale_slope_and_demand(bed, valid, 10.0, CELL_M)
    assert np.all(np.isnan(slope[~valid]))
    assert np.all(np.isnan(demand[~valid]))
    assert np.isfinite(slope[valid]).sum() > 0


def test_09_10_11_12_outputs_preserve_nodata_crs_transform_and_dimensions(synthetic_study: Path):
    result = _run(synthetic_study, scenarios=[_scenario()])
    canonical = synthetic_study / screening.TERRAIN_SUBDIR / contract.CANONICAL_TERRAIN_FILENAME
    with rasterio.open(canonical) as src:
        src_crs, src_transform, src_shape = src.crs, src.transform, (src.height, src.width)
        src_nan = ~np.isfinite(src.read(1))
    for key, path in result.outputs.items():
        if not str(path).endswith(".tif"):
            continue
        with rasterio.open(path) as out:
            assert out.crs == src_crs, key
            assert out.transform == src_transform, key
            assert (out.height, out.width) == src_shape, key
            band = out.read(1)
            if out.dtypes[0] != "int8":
                assert np.isnan(out.nodata), key
                assert np.all(~np.isfinite(band[src_nan])), f"{key}: nodata footprint filled"
                assert not np.any(np.isinf(band)), f"{key}: infinity written"
    integrity = result.metadata["raster_integrity"]
    assert integrity["reprojection"] == "none" and integrity["resampling"] == "none"


def test_11_12b_package_never_calls_resampling_or_reprojection():
    identifiers = set().union(*(_code_identifiers(m) for m in PACKAGE_MODULES))
    for name in (
        "reproject",
        "Resampling",
        "resample",
        "warp",
        "zoom",
        "interp",
        "griddata",
        "fillna",
        "interpolate",
        "transform_bounds",
        "calculate_default_transform",
    ):
        assert name not in identifiers, name


# --- 13-16: analytic factor of safety -------------------------------------------------------------


def test_13_analytic_case_gives_fs_exactly_one():
    result = core.compute_scenario_factor_of_safety(np.array([DEG30]), _scenario())
    tau_expected = 8000.0 * 2.0 * math.sin(math.radians(30.0)) * math.cos(math.radians(30.0))
    assert float(result.tau_driving_pa[0]) == pytest.approx(tau_expected, rel=1e-12)
    assert float(result.tau_driving_pa[0]) == pytest.approx(6928.203, abs=1e-3)
    assert float(result.factor_of_safety[0]) == pytest.approx(1.0, rel=1e-12)
    assert result.model_state[0] == core.MODEL_FS_AT_1


def test_14_doubling_su_doubles_fs():
    base = core.compute_scenario_factor_of_safety(np.array([DEG30]), _scenario()).factor_of_safety
    doubled = core.compute_scenario_factor_of_safety(
        np.array([DEG30]), _scenario(su_kpa=2 * SU_AT_FS1_KPA)
    ).factor_of_safety
    assert float(doubled[0] / base[0]) == pytest.approx(2.0, rel=1e-12)


def test_15_doubling_depth_halves_fs():
    base = core.compute_scenario_factor_of_safety(np.array([DEG30]), _scenario()).factor_of_safety
    halved = core.compute_scenario_factor_of_safety(
        np.array([DEG30]), _scenario(depth_m=2 * DEPTH_M)
    ).factor_of_safety
    assert float(halved[0] / base[0]) == pytest.approx(0.5, rel=1e-12)


def test_16_doubling_submerged_unit_weight_halves_fs():
    base = core.compute_scenario_factor_of_safety(np.array([DEG30]), _scenario()).factor_of_safety
    halved = core.compute_scenario_factor_of_safety(
        np.array([DEG30]), _scenario(gamma_kn_m3=2 * GAMMA_KN)
    ).factor_of_safety
    assert float(halved[0] / base[0]) == pytest.approx(0.5, rel=1e-12)


def test_16b_units_are_converted_explicitly_to_si():
    scenario = _scenario(su_kpa=5.0, gamma_kn_m3=8.0)
    assert scenario.undrained_shear_strength_pa == 5000.0
    assert scenario.submerged_unit_weight_n_m3 == 8000.0


# --- 17-19: parameter validation ------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_17_invalid_su_rejected(bad):
    with pytest.raises(core.SlopeStabilityInputError):
        _scenario(su_kpa=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_18_invalid_gamma_rejected(bad):
    with pytest.raises(core.SlopeStabilityInputError):
        _scenario(gamma_kn_m3=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_19_invalid_slip_depth_rejected(bad):
    with pytest.raises(core.SlopeStabilityInputError):
        _scenario(depth_m=bad)


def test_19b_wrong_material_model_or_parameter_basis_rejected():
    with pytest.raises(core.SlopeStabilityInputError):
        core.UndrainedInfiniteSlopeScenario(
            scenario_id="x",
            material_model="DRAINED_GRANULAR",
            parameter_basis=core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO,
            undrained_shear_strength_kpa=5.0,
            submerged_unit_weight_kn_m3=8.0,
            slip_surface_depth_m=2.0,
        )
    with pytest.raises(core.SlopeStabilityInputError):
        core.UndrainedInfiniteSlopeScenario(
            scenario_id="x",
            material_model=core.MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE,
            parameter_basis="MEASURED_SITE_GEOTECHNICAL_TRUTH",
            undrained_shear_strength_kpa=5.0,
            submerged_unit_weight_kn_m3=8.0,
            slip_surface_depth_m=2.0,
        )


@pytest.mark.parametrize(
    "field, bad",
    [
        ("undrained_shear_strength_kpa", 0.0),
        ("undrained_shear_strength_kpa", -3.0),
        ("undrained_shear_strength_kpa", float("nan")),
        ("submerged_unit_weight_kn_m3", float("inf")),
        ("slip_surface_depth_m", 0.0),
    ],
)
def test_19c_manifest_rejects_invalid_parameters(field, bad):
    raw = {
        "scenario_id": "s",
        "material_model": core.MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE,
        "parameter_basis": core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO,
        "undrained_shear_strength_kpa": 5.0,
        "submerged_unit_weight_kn_m3": 8.0,
        "slip_surface_depth_m": 2.0,
    }
    raw[field] = bad
    with pytest.raises(ValidationError):
        manifest.GeotechnicalScenarioDeclaration.model_validate(raw)


def test_19d_manifest_requires_explicit_model_and_basis_and_unique_ids():
    base = {
        "scenario_id": "s",
        "undrained_shear_strength_kpa": 5.0,
        "submerged_unit_weight_kn_m3": 8.0,
        "slip_surface_depth_m": 2.0,
    }
    with pytest.raises(ValidationError):  # material_model / parameter_basis have NO default
        manifest.GeotechnicalScenarioDeclaration.model_validate(base)
    full = {
        **base,
        "material_model": core.MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE,
        "parameter_basis": core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO,
    }
    with pytest.raises(ValidationError):
        manifest.SlopeInstabilityScenarioManifest.model_validate(
            {"analysis_id": "a", "geotechnical_scenarios": [full, full]}
        )
    with pytest.raises(ValidationError):
        manifest.SlopeInstabilityScenarioManifest.model_validate(
            {"analysis_id": "a", "geotechnical_scenarios": []}
        )
    with pytest.raises(ValidationError):  # extra keys forbidden
        manifest.SlopeInstabilityScenarioManifest.model_validate(
            {"analysis_id": "a", "geotechnical_scenarios": [full], "bgs_folk_class": "S"}
        )


# --- 20-24: flat cells, small slopes, model states ------------------------------------------------


def test_20_zero_slope_produces_null_not_infinite_fs():
    result = core.compute_scenario_factor_of_safety(np.array([0.0, DEG30]), _scenario())
    assert result.tau_driving_pa[0] == 0.0
    assert np.isnan(result.factor_of_safety[0])
    assert not np.isinf(result.factor_of_safety).any()
    assert result.model_state[0] == core.NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR


def test_21_very_small_finite_slope_gives_finite_large_fs_no_minimum_threshold():
    result = core.compute_scenario_factor_of_safety(np.array([1e-6]), _scenario())
    fs = float(result.factor_of_safety[0])
    assert np.isfinite(fs) and fs > 1e6
    assert result.model_state[0] == core.MODEL_FS_ABOVE_1


def test_22_exact_fs_one_is_model_fs_at_1():
    assert core.classify_model_state(1.0, 100.0) == core.MODEL_FS_AT_1


def test_23_fs_below_one_is_model_fs_below_1():
    assert core.classify_model_state(0.999, 100.0) == core.MODEL_FS_BELOW_1


def test_24_fs_above_one_is_model_fs_above_1():
    assert core.classify_model_state(1.001, 100.0) == core.MODEL_FS_ABOVE_1


def test_24b_nan_slope_is_not_evaluable_and_codes_round_trip():
    result = core.compute_scenario_factor_of_safety(np.array([np.nan, 0.0, 30.0]), _scenario())
    assert list(result.model_state) == [
        core.NOT_EVALUABLE,
        core.NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR,
        core.MODEL_FS_AT_1,
    ]
    codes = core.encode_model_states(result.model_state)
    assert codes.dtype == np.int8
    assert list(codes) == [0, 1, 3]


# --- 26-29: no default / inferred soil ------------------------------------------------------------


def test_26_no_default_geotechnical_scenario(synthetic_study: Path):
    signature = inspect.signature(screening.run_slope_instability_screening)
    assert signature.parameters["scenarios"].default is None
    result = _run(synthetic_study)
    assert result.scenario_results == []
    assert result.metadata["geotechnical_factor_of_safety_computed"] is False
    assert result.metadata["default_geotechnical_scenario_exists"] is False
    assert result.metadata["geotechnical_parameter_basis"] is None
    assert not any("factor_of_safety" in k for k in result.outputs)
    with pytest.raises(core.SlopeStabilityInputError):
        core.compute_scenario_factor_of_safety(np.array([10.0]), None)  # type: ignore[arg-type]


def test_27_no_literature_soil_parameters_are_hard_coded():
    pattern = re.compile(r"(shear_strength|unit_weight|slip_surface_depth)\w*\s*[:=]\s*[0-9]")
    for module in PACKAGE_MODULES:
        for line in _code_only_source(module).splitlines():
            assert not pattern.search(line), f"{module.__name__}: {line.strip()}"
    contract_doc = contract.build_slope_instability_contract()
    assert contract_doc["scenario_contract"]["default_scenario_exists"] is False


def test_28_29_no_bgs_folk_or_psa_to_strength_conversion():
    code = "\n".join(_code_only_source(m) for m in PACKAGE_MODULES)
    import_lines = "\n".join(
        line for line in code.splitlines() if line.strip().startswith(("import ", "from "))
    )
    for fragment in ("sediment", "bgs", "providers", "grain_size", "psa"):
        assert fragment not in import_lines.lower(), fragment
    identifiers = {n.lower() for n in set().union(*(_code_identifiers(m) for m in PACKAGE_MODULES))}
    for fragment in ("folk", "d50", "grain", "sand", "mud", "gravel", "psa", "sediment"):
        offenders = [n for n in identifiers if fragment in n]
        assert offenders == [], (fragment, offenders)


# --- 30-36: nothing beyond static screening is computed -------------------------------------------


def test_31_to_36_not_modelled_flags_are_false_in_every_output(synthetic_study: Path):
    result = _run(synthetic_study, scenarios=[_scenario()])
    for document in (result.metadata, result.readiness):
        for flag, value in contract.NOT_MODELLED_FLAGS.items():
            assert document[flag] is False, flag
            assert value is False
    assert result.readiness["landslide_probability_available"] is False
    assert result.readiness["risk_score_available"] is False
    assert result.readiness["site_specific_factor_of_safety_available"] is False
    for key in (
        "excess_pore_pressure_modelled",
        "earthquake_trigger_modelled",
        "liquefaction_modelled",
        "runout_modelled",
        "pipeline_impact_modelled",
        "landslide_probability_computed",
        "risk_score_computed",
        "progressive_failure_modelled",
        "retrogressive_failure_modelled",
    ):
        assert result.metadata[key] is False


def test_30_to_36b_no_function_implements_forbidden_physics():
    code = "\n".join(_code_only_source(m) for m in PACKAGE_MODULES)
    definitions = re.findall(r"^\s*def\s+(\w+)", code, flags=re.MULTILINE)
    forbidden = re.compile(
        r"pore|earthquake|seismic|pga|liquef|runout|velocity|volume|impact|probab|risk|tsunami|"
        r"debris|trigger|retrogress|progressive",
        flags=re.IGNORECASE,
    )
    offenders = [d for d in definitions if forbidden.search(d)]
    assert offenders == [], offenders
    identifiers = set().union(*(_code_identifiers(m) for m in PACKAGE_MODULES))
    assert not (
        identifiers
        & {"r_u", "pore_pressure_ratio", "k_h", "pseudostatic", "runout_distance", "impact_force"}
    )


# --- 37-40: real-terrain mode and scenario provenance ---------------------------------------------


def test_37_terrain_mode_works_without_geotechnical_inputs(synthetic_study: Path):
    result = _run(synthetic_study)
    assert result.terrain_screening["status"] == contract.TERRAIN_SCREENING_READY
    assert len(result.scale_results) == 2
    for scale_result in result.scale_results:
        demand = scale_result["normalized_strength_demand"]
        assert demand["valid_cells"] > 0
        assert 0.0 <= demand["min"] <= demand["max"] <= 0.5
        assert Path(scale_result["normalized_strength_demand_path"]).exists()
    assert result.metadata["scientific_role"] == (
        "UNDRAINED_INFINITE_SLOPE_NORMALIZED_STRENGTH_DEMAND"
    )
    assert result.metadata["terrain_source_sha256"] == "0" * 64
    assert result.metadata["canonical_terrain_sha256"] is not None
    with rasterio.open(result.outputs["normalized_strength_demand_10m"]) as src:
        tags = src.tags()
    assert tags["scientific_role"] == contract.SCIENTIFIC_ROLE_NORMALIZED_STRENGTH_DEMAND
    assert tags["not_a_factor_of_safety"] == "True"


def test_37b_synthetic_plane_demand_matches_analytic_sin_cos_of_ten_degrees(synthetic_study: Path):
    result = _run(synthetic_study, slope_scales_m=[10.0])
    with rasterio.open(result.outputs["normalized_strength_demand_10m"]) as src:
        demand = src.read(1)
        slope = rasterio.open(result.outputs["slope_10m_deg"]).read(1)
    # Interior of the west (tilted) half, away from the hole, the east flat half and the edges.
    interior = demand[60:200, 15:100]
    expected = math.sin(math.radians(10.0)) * math.cos(math.radians(10.0))
    assert np.allclose(interior, expected, atol=1e-5)
    assert np.allclose(slope[60:200, 15:100], 10.0, atol=1e-3)
    # Interior of the flat half: exactly zero demand (a=b=0 recovered by the plane fit).
    assert np.allclose(demand[60:200, 135:225], 0.0, atol=1e-9)


def test_38_terrain_mode_reports_fos_not_evaluable(synthetic_study: Path):
    result = _run(synthetic_study)
    readiness = result.readiness
    assert readiness["terrain_screening_status"] == contract.TERRAIN_SCREENING_READY
    assert readiness["geotechnical_status"] == contract.GEOTECHNICAL_STABILITY_NOT_EVALUABLE
    assert contract.NO_GEOTECHNICAL_SCENARIO_SUPPLIED in readiness["geotechnical_reasons"]
    assert readiness["trigger_status"] == contract.TRIGGER_RESPONSE_NOT_MODELLED
    assert readiness["local_slope_stability_status"] == contract.LOCAL_SLOPE_STABILITY_NOT_EVALUABLE
    assert readiness["questions"]["factor_of_safety_evaluable"] is False
    assert readiness["questions"]["trigger_modelling_available"] is False
    assert "readiness_percentage" not in json.dumps(readiness)
    assert not re.search(r"\"\w*score\w*\":\s*[0-9]", json.dumps(readiness))


def test_39_40_scenario_outputs_retain_scenario_id_and_hypothetical_basis(synthetic_study: Path):
    result = _run(synthetic_study, scenarios=[_scenario("illustrative_a")], slope_scales_m=[10.0])
    assert len(result.scenario_results) == 1
    scenario_result = result.scenario_results[0]
    assert scenario_result["scenario_id"] == "illustrative_a"
    assert scenario_result["parameter_basis"] == "USER_DECLARED_HYPOTHETICAL_SCENARIO"
    assert scenario_result["material_model"] == ("COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE")
    assert "NOT_SITE_SPECIFIC_MEASUREMENT" in scenario_result["disclaimer"]
    fos_path = Path(result.outputs["factor_of_safety_illustrative_a_10m"])
    state_path = Path(result.outputs["model_state_illustrative_a_10m"])
    assert "illustrative_a" in fos_path.name and "illustrative_a" in state_path.name
    with rasterio.open(fos_path) as src:
        tags = src.tags()
        fos = src.read(1)
    assert tags["scenario_id"] == "illustrative_a"
    assert tags["parameter_basis"] == "USER_DECLARED_HYPOTHETICAL_SCENARIO"
    assert tags["material_model"] == "COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE"
    assert "NOT_SITE_SPECIFIC_MEASUREMENT" in tags["disclaimer"]
    assert tags["scientific_role"] == contract.SCIENTIFIC_ROLE_FACTOR_OF_SAFETY_SCENARIO
    assert not np.isinf(fos).any()
    with rasterio.open(state_path) as src:
        assert json.loads(src.tags()["legend"]) == core.MODEL_STATE_CODES
        assert src.dtypes[0] == "int8"
    counts = scenario_result["model_state_counts"]
    assert counts[core.NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR] > 0  # the flat half
    assert counts[core.MODEL_FS_ABOVE_1] > 0  # 10 deg plane with the FS=1-at-30-deg scenario
    assert result.readiness["geotechnical_status"] == (
        contract.GEOTECHNICAL_STABILITY_SCENARIO_AVAILABLE
    )
    assert result.readiness["local_slope_stability_status"] == (
        contract.HYPOTHETICAL_SCENARIO_FOS_AVAILABLE_NOT_SITE_SPECIFIC
    )
    assert result.readiness["site_specific_factor_of_safety_available"] is False
    assert result.metadata["geotechnical_scenarios"][0]["site_specific_measurement"] is False


# --- 41-42: protected accepted modules unchanged --------------------------------------------------


PROTECTED_MODULE_SHA256 = {
    # Content hashes at the MAR-031 canonical base (main @ d69efa5). MAR-031 must not change the
    # accepted MAR-020 terrain derivative / readiness science or the MAR-007 regional morphology.
    "src/marine_engine/terrain/derivatives.py": (
        "891e782754c71764a88845155d0bbd882f7aecbcd7f76d4d0e4a88c7ef225a24"
    ),
    "src/marine_engine/terrain/readiness.py": (
        "b50b7a8864af78972d1ead13a3556f6bf92742e6306ff24f5582e6a9baff278a"
    ),
    "src/marine_engine/morphology/regional.py": (
        "9ed8fb770fe85c9a921ca79a6320dde2dcd6a8a61fcb5fd3f163d7b46933b3ce"
    ),
}


@pytest.mark.parametrize("relative_path, expected_sha256", sorted(PROTECTED_MODULE_SHA256.items()))
def test_41_42_accepted_mar020_and_mar007_modules_are_byte_identical(
    relative_path, expected_sha256
):
    content = (REPO_ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(content).hexdigest() == expected_sha256, relative_path


def test_41b_slope_is_delegated_to_the_accepted_mar020_derivative_not_reimplemented():
    code = _code_only_source(screening)
    assert "terrain_derivatives.compute_slope_aspect_deg(" in code
    for module in PACKAGE_MODULES:
        for name in ("def compute_slope", "def _plane_fit", "np.linalg.solve", "correlate1d"):
            assert name not in _code_only_source(module), f"{module.__name__} reimplements {name}"
    package_code = "\n".join(_code_only_source(m) for m in PACKAGE_MODULES)
    import_lines = "\n".join(
        line for line in package_code.splitlines() if line.strip().startswith(("import ", "from "))
    )
    assert "morphology" not in import_lines
    assert "evidence_atlas" not in import_lines


def test_41c_intrinsic_readiness_is_delegated_to_mar020(synthetic_study: Path):
    result = _run(synthetic_study)
    intrinsic = result.terrain_screening["intrinsic_bathymetry_readiness"]
    assert intrinsic is not None
    check_ids = {c["check_id"] for c in intrinsic["checks"]}
    assert {"crs_projected_metric", "analytical_not_render", "data_range_plausible"} <= check_ids
    assert result.readiness["intrinsic_bathymetry_readiness"]["status"] in (
        "READY",
        "READY_WITH_LIMITATIONS",
    )


# --- PL854-style boundary: no high-resolution terrain ---------------------------------------------


def _write_fake_regional_metadata(study_dir: Path) -> Path:
    morph_dir = study_dir / "morphology"
    morph_dir.mkdir(parents=True)
    transform = rasterio.Affine(100.0, 0.0, 400000.0, 0.0, -100.0, 5920000.0)
    slope = np.full((12, 20), 0.2, dtype=np.float32)
    slope[0, 0] = np.nan
    terrain_raster_io.write_terrain_raster(
        slope, transform, CRS, morph_dir / "slope_500m_deg.tif", {"layer": "slope_500m_deg"}
    )
    metadata = {
        "source_product": "EMODnet Digital Bathymetry (DTM 2024)",
        "source_nominal_resolution_m": 115.0,
        "analysis_grid_spacing_m": 100.0,
        "underlying_acquisition_years": [1991, 1992],
        "features": {"slope_500m_deg": {"radius_m": 500.0, "unit": "degrees"}},
        "raster_outputs": {
            "slope_500m_deg": "data\\processed\\x\\morphology\\slope_500m_deg.tif",
            "tpi_1000m_m": "data\\processed\\x\\morphology\\tpi_1000m_m.tif",
        },
    }
    path = morph_dir / "morphology_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_P1_missing_canonical_terrain_is_controlled_not_ready_with_regional_context_only(
    tmp_path: Path,
):
    study_dir = tmp_path / "processed" / "route_like"
    study_dir.mkdir(parents=True)
    _write_fake_regional_metadata(study_dir)
    result = screening.run_slope_instability_screening(
        project_id="route_like", study_dir=study_dir, render_figure=False, log=lambda _m: None
    )
    readiness = result.readiness
    assert readiness["terrain_screening_status"] == contract.TERRAIN_SCREENING_NOT_READY
    assert readiness["pipeline_scale_slope_stability_terrain_readiness"] == "NOT_READY"
    assert readiness["pipeline_scale_slope_stability_terrain_reason"] == (
        "HIGH_RESOLUTION_CURRENT_SEABED_GEOMETRY_NOT_AVAILABLE"
    )
    assert readiness["local_slope_stability_status"] == "LOCAL_SLOPE_STABILITY_NOT_EVALUABLE"
    assert readiness["geotechnical_status"] == contract.GEOTECHNICAL_STABILITY_NOT_EVALUABLE
    assert contract.SITE_GEOTECHNICAL_PROFILE_UNAVAILABLE in readiness["geotechnical_reasons"]
    regional = readiness["regional_slope_context"]
    assert regional["status"] == screening.REGIONAL_CONTEXT_AVAILABLE
    assert regional["role"] == "REGIONAL_CONTEXT_ONLY"
    assert regional["promoted_to_pipeline_scale_slope_stability_input"] is False
    assert regional["normalized_strength_demand_derived_from_regional_slope"] is False
    assert [layer["layer"] for layer in regional["slope_layers"]] == ["slope_500m_deg"]
    assert regional["slope_layers"][0]["slope_deg_summary"]["valid_cells"] == 239
    assert result.scale_results == [] and result.scenario_results == []
    assert not any(str(p).endswith(".tif") for p in result.outputs.values())
    assert (study_dir / "slope_stability").is_dir()
    assert result.metadata_path.exists() and result.readiness_path.exists()


def test_P2_scenarios_without_ready_terrain_compute_no_fos(tmp_path: Path):
    study_dir = tmp_path / "processed" / "route_like"
    study_dir.mkdir(parents=True)
    result = screening.run_slope_instability_screening(
        project_id="route_like",
        study_dir=study_dir,
        scenarios=[_scenario()],
        render_figure=False,
        log=lambda _m: None,
    )
    assert result.scenario_results == []
    assert result.readiness["geotechnical_status"] == contract.GEOTECHNICAL_STABILITY_NOT_EVALUABLE
    assert result.readiness["questions"]["geotechnical_parameters_supplied"] is True
    assert result.readiness["questions"]["factor_of_safety_evaluable"] is False
    assert result.readiness["regional_slope_context"]["status"] == (
        screening.REGIONAL_CONTEXT_NOT_AVAILABLE
    )


def test_P3_non_canonical_raster_role_is_rejected_not_reinterpreted(tmp_path: Path):
    study_dir = tmp_path / "processed" / "wrong_role"
    _write_canonical(
        study_dir,
        _synthetic_bed_elevation(),
        tags={"scientific_role": "SOMETHING_ELSE", "layer": "depth_positive_down"},
    )
    result = screening.run_slope_instability_screening(
        project_id="wrong_role", study_dir=study_dir, render_figure=False, log=lambda _m: None
    )
    assert result.terrain_screening["status"] == contract.TERRAIN_SCREENING_NOT_READY
    assert any(
        r.startswith(contract.CANONICAL_TERRAIN_ROLE_MISMATCH)
        for r in result.terrain_screening["reasons"]
    )
    assert result.scale_results == []


def test_P4_invalid_scales_rejected(synthetic_study: Path):
    for bad in ([], [10.0, 10.0], [0.0], [-5.0], [float("nan")]):
        with pytest.raises(core.SlopeStabilityInputError):
            _run(synthetic_study, slope_scales_m=bad)


# --- Figure ---------------------------------------------------------------------------------------


def test_F1_figure_renders_four_panels_with_sequential_colormaps_only(synthetic_study: Path):
    result = _run(synthetic_study, render_figure=True)
    assert result.figure_path is not None and result.figure_path.exists()
    width_px, height_px = maps.read_png_dimensions(result.figure_path)
    assert width_px > 500 and height_px > 300
    code = inspect.getsource(screening)
    for forbidden in ("RdYlGn", "traffic", "jet", "RdYlBu", "Reds"):
        assert forbidden not in code, forbidden


def test_F2_figure_rejects_non_sequential_colormap(tmp_path: Path):
    transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    with pytest.raises(ValueError):
        maps.render_slope_instability_screening_figure(
            layers={"x": (np.zeros((10, 10)), "RdYlGn", "")},
            transform=transform,
            full_shape=(10, 10),
            output_path=tmp_path / "x.png",
            title="",
            subtitle="",
            footer_note="",
        )


def test_F3_display_decimation_is_a_pure_subsample():
    array = np.arange(100.0).reshape(10, 10)
    assert maps.display_stride_for((10, 10), max_display_side_px=4) == 3
    view = maps.decimate_for_display(array, 3)
    assert np.array_equal(view, array[::3, ::3])
    assert maps.decimate_for_display(array, 1) is array


# --- Manifest YAML + CLI --------------------------------------------------------------------------


def _write_manifest(path: Path, *, scales: list[float] | None = None) -> Path:
    raw = {
        "analysis_id": "synthetic_slope_screening",
        "geotechnical_scenarios": [
            {
                "scenario_id": "synthetic_case",
                "material_model": (
                    core.MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE
                ),
                "parameter_basis": core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO,
                "undrained_shear_strength_kpa": SU_AT_FS1_KPA,
                "submerged_unit_weight_kn_m3": GAMMA_KN,
                "slip_surface_depth_m": DEPTH_M,
            }
        ],
    }
    if scales is not None:
        raw["slope_scales_m"] = scales
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _write_study_config(tmp_path: Path, study_id: str) -> Path:
    config = {
        "study": {"id": study_id, "name": f"{study_id} synthetic"},
        "crs": {"horizontal": CRS, "vertical": None},
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "interim_dir": str(tmp_path / "interim"),
            "processed_dir": str(tmp_path / "processed"),
        },
    }
    path = tmp_path / f"{study_id}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_M1_manifest_yaml_round_trip(tmp_path: Path):
    path = _write_manifest(tmp_path / "scenarios.yaml", scales=[10.0])
    loaded = manifest.load_slope_instability_scenario_manifest(path)
    assert loaded.analysis_id == "synthetic_slope_screening"
    assert loaded.slope_scales_m == [10.0]
    [scenario] = loaded.core_scenarios()
    assert isinstance(scenario, core.UndrainedInfiniteSlopeScenario)
    assert scenario.parameter_basis == "USER_DECLARED_HYPOTHETICAL_SCENARIO"


def test_C1_cli_runs_terrain_mode_without_scenario(tmp_path: Path, capsys):
    config_path = _write_study_config(tmp_path, "SYNTHETIC_SITE")
    _write_canonical(tmp_path / "processed" / "synthetic_site", _synthetic_bed_elevation())
    assert cli.main(["build-slope-instability-screening", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "IS SLOPE ANGLE ALONE TREATED AS LANDSLIDE STABILITY? NO" in out
    assert "REAL TERRAIN-DERIVED NORMALIZED STRENGTH DEMAND AVAILABLE? YES" in out
    assert "SITE-SPECIFIC GEOTECHNICAL FACTOR OF SAFETY AVAILABLE? NO" in out
    assert "LANDSLIDE PROBABILITY OR RISK SCORE COMPUTED? NO" in out
    assert "GEOTECHNICAL_STABILITY_NOT_EVALUABLE" in out
    out_dir = tmp_path / "processed" / "synthetic_site" / "slope_stability"
    assert (out_dir / "normalized_strength_demand_10m.tif").exists()
    assert (out_dir / "normalized_strength_demand_50m.tif").exists()
    assert (out_dir / "slope_instability_screening_metadata.json").exists()
    assert (out_dir / "slope_instability_readiness.json").exists()
    assert (
        tmp_path
        / "processed"
        / "synthetic_site"
        / "maps"
        / "synthetic_site_slope_instability_screening.png"
    ).exists()
    assert not list(out_dir.glob("factor_of_safety_*.tif"))


def test_C2_cli_runs_explicit_scenario_manifest(tmp_path: Path, capsys):
    config_path = _write_study_config(tmp_path, "SYNTHETIC_SITE")
    _write_canonical(tmp_path / "processed" / "synthetic_site", _synthetic_bed_elevation())
    manifest_path = _write_manifest(tmp_path / "scenarios.yaml", scales=[10.0])
    assert (
        cli.main(
            [
                "build-slope-instability-screening",
                str(config_path),
                "--scenario-manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "USER_DECLARED_HYPOTHETICAL_SCENARIO" in out
    assert "NOT_SITE_SPECIFIC_MEASUREMENT" in out
    out_dir = tmp_path / "processed" / "synthetic_site" / "slope_stability"
    assert (out_dir / "factor_of_safety_synthetic_case_10m.tif").exists()
    assert not (out_dir / "normalized_strength_demand_50m.tif").exists()  # manifest scales honoured


def test_C3_cli_rejects_invalid_manifest_with_controlled_exit(tmp_path: Path, capsys):
    config_path = _write_study_config(tmp_path, "SYNTHETIC_SITE")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "analysis_id": "bad",
                "geotechnical_scenarios": [
                    {
                        "scenario_id": "s",
                        "material_model": (
                            core.MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE
                        ),
                        "parameter_basis": core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO,
                        "undrained_shear_strength_kpa": -5.0,
                        "submerged_unit_weight_kn_m3": 8.0,
                        "slip_surface_depth_m": 2.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    code = cli.main(
        ["build-slope-instability-screening", str(config_path), "--scenario-manifest", str(bad)]
    )
    assert code == 1
    assert "rejected" in capsys.readouterr().out


def test_C4_cli_pl854_like_study_without_terrain_reports_not_evaluable(tmp_path: Path, capsys):
    config_path = _write_study_config(tmp_path, "ROUTE_LIKE")
    (tmp_path / "processed" / "route_like").mkdir(parents=True)
    assert cli.main(["build-slope-instability-screening", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "TERRAIN_SCREENING_NOT_READY" in out
    assert "HIGH_RESOLUTION_CURRENT_SEABED_GEOMETRY_NOT_AVAILABLE" in out
    assert "LOCAL_SLOPE_STABILITY_NOT_EVALUABLE" in out
    assert "REAL TERRAIN-DERIVED NORMALIZED STRENGTH DEMAND AVAILABLE? NO" in out


def test_C5_cli_command_performs_no_network_io(monkeypatch, tmp_path: Path):
    import socket

    def _no_network(*_args, **_kwargs):
        raise AssertionError("network access attempted during slope-instability screening")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    config_path = _write_study_config(tmp_path, "SYNTHETIC_SITE")
    _write_canonical(tmp_path / "processed" / "synthetic_site", _synthetic_bed_elevation())
    assert cli.main(["build-slope-instability-screening", str(config_path)]) == 0
    identifiers = _code_identifiers(cli._cmd_build_slope_instability_screening)
    assert not any("download" in n or "fetch" in n for n in identifiers)
    assert "requests" not in identifiers and "sheringham_provider" not in identifiers


# --- Contract / metadata content ------------------------------------------------------------------


def test_K1_contract_records_equations_references_and_applicability(synthetic_study: Path):
    doc = contract.build_slope_instability_contract()
    assert "sin(alpha) * cos(alpha)" in doc["normalized_strength_demand_definition"]
    assert (
        "FS = s_u / (gamma_prime * z * sin(alpha) * cos(alpha))" in doc["infinite_slope_equation"]
    )
    dois = {r["doi"] for r in doc["references"]}
    assert dois == {
        "10.1098/rsta.2006.1810",
        "10.1139/t01-089",
        "10.1016/j.marpetgeo.2004.10.019",
        "10.1002/2013JF003068",
    }
    assert "runout" in next(r for r in doc["references"] if "Baeten" in r["key"])["used_for"]
    assert "thin translational slab" in doc["model_applicability"]["applicable_to"]
    assert "liquefaction" in doc["model_applicability"]["not_automatically_applicable_to"]
    assert "NOT MODELLED" in doc["model_applicability"]["pore_pressure_semantics"]
    assert doc["units"]["normalized_undrained_strength_demand"] == "dimensionless"
    result = _run(synthetic_study)
    for key in (
        "scientific_role",
        "source_terrain_role",
        "infinite_slope_equation",
        "normalized_strength_demand_definition",
        "units",
        "slope_scales_m",
        "terrain_source_identity",
        "terrain_source_sha256",
        "terrain_crs",
        "terrain_resolution_m",
        "geotechnical_factor_of_safety_computed",
        "geotechnical_parameter_basis",
        "model_applicability",
        "references",
        "limitations",
    ):
        assert key in result.metadata, key
    assert result.metadata["terrain_crs"] == CRS
    assert result.metadata["terrain_resolution_m"] == CELL_M

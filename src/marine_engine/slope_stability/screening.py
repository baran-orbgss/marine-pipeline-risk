"""Slope-instability screening orchestration (MAR-031 Sections 8-10, 22, 26-29, 32).

Consumes the ACCEPTED MAR-020 canonical terrain product (`terrain/canonical_bed_elevation.tif`,
`bed_elevation_m`, higher = shallower) of a study, read-only, and:

1. answers terrain-screening readiness by DELEGATING intrinsic raster readiness to the accepted
   `terrain.readiness.assess_bathymetry_readiness` (never re-implemented) and adding only the
   MAR-031-specific integration conditions (canonical role tag, square metric pixels, no rotation);
2. for each INDEPENDENT slope scale, calls the accepted
   `terrain.derivatives.compute_slope_aspect_deg` unchanged and derives
   `normalized_undrained_strength_demand = sin(alpha) * cos(alpha)`;
3. only when explicit user-declared hypothetical scenarios are supplied, evaluates the undrained
   infinite-slope factor of safety per scale and per scenario;
4. reports broad regional morphology slope (if a MAR-007 metadata file exists) strictly as
   REGIONAL_CONTEXT_ONLY, never as a pipeline-scale slope-stability input.

Raster integrity (Section 26): every output grid is written on the canonical terrain's own CRS,
transform, width and height; no reprojection, resampling, interpolation or nodata filling occurs
anywhere in this module.

Zero dependency on any specific project or dataset: paths, IDs and scales are parameters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.errors import RasterioError, RasterioIOError

from marine_engine.slope_stability import contract, core, maps, report
from marine_engine.terrain import derivatives as terrain_derivatives
from marine_engine.terrain import raster_io as terrain_raster_io
from marine_engine.terrain import readiness as terrain_readiness

SLOPE_STABILITY_SUBDIR = "slope_stability"
MAPS_SUBDIR = "maps"
TERRAIN_SUBDIR = "terrain"
MORPHOLOGY_METADATA_RELATIVE = Path("morphology") / "morphology_metadata.json"
METADATA_FILENAME = "slope_instability_screening_metadata.json"
READINESS_FILENAME = "slope_instability_readiness.json"
CONTRACT_FILENAME = "slope_instability_contract.json"

REGIONAL_CONTEXT_AVAILABLE = "AVAILABLE"
REGIONAL_CONTEXT_NOT_AVAILABLE = "NOT_AVAILABLE"


# --- Canonical terrain facts (observed from the file, never declared) -----------------------------


@dataclass(frozen=True)
class CanonicalTerrainFacts:
    path: Path
    readable: bool
    read_error: str | None
    crs: str | None
    crs_is_geographic: bool | None
    crs_linear_units: str | None
    transform: tuple[float, ...] | None
    width: int
    height: int
    pixel_size_x_m: float | None
    pixel_size_y_m: float | None
    bounds: tuple[float, float, float, float] | None
    nodata: float | None
    dtype: str | None
    band_count: int
    color_interpretations: tuple[str, ...]
    tags: dict[str, str] = field(default_factory=dict)
    canonical_sha256: str | None = None

    @property
    def affine(self) -> rasterio.Affine | None:
        return rasterio.Affine(*self.transform[:6]) if self.transform else None

    @property
    def is_rotated(self) -> bool:
        return bool(self.transform) and (self.transform[1] != 0.0 or self.transform[3] != 0.0)

    @property
    def pixels_square_metric(self) -> bool:
        return (
            self.pixel_size_x_m is not None
            and self.pixel_size_y_m is not None
            and np.isfinite(self.pixel_size_x_m)
            and np.isfinite(self.pixel_size_y_m)
            and self.pixel_size_x_m > 0
            and abs(self.pixel_size_x_m - self.pixel_size_y_m) <= 1e-9 * self.pixel_size_x_m
            and self.crs_is_geographic is False
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "readable": self.readable,
            "read_error": self.read_error,
            "crs": self.crs,
            "crs_is_geographic": self.crs_is_geographic,
            "crs_linear_units": self.crs_linear_units,
            "transform": list(self.transform) if self.transform else None,
            "dimensions": {"width": self.width, "height": self.height},
            "pixel_size_m": self.pixel_size_x_m,
            "pixel_size_x_m": self.pixel_size_x_m,
            "pixel_size_y_m": self.pixel_size_y_m,
            "bounds": list(self.bounds) if self.bounds else None,
            "nodata": None if self.nodata is None or np.isnan(self.nodata) else self.nodata,
            "nodata_is_nan": bool(self.nodata is not None and np.isnan(self.nodata)),
            "dtype": self.dtype,
            "band_count": self.band_count,
            "embedded_tags": dict(self.tags),
            "source_identity": self.tags.get("source_dataset"),
            "source_sha256": self.tags.get("source_sha256"),
            "source_vertical_datum": self.tags.get("source_vertical_datum"),
            "source_sign_convention": self.tags.get("source_sign_convention"),
            "survey_epoch": self.tags.get("survey_epoch"),
            "canonical_sha256": self.canonical_sha256,
        }


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_canonical_terrain(
    path: Path, *, compute_sha256: bool = True
) -> CanonicalTerrainFacts | None:
    """Observed facts about the canonical terrain raster, or None when the file does not exist
    (missing means missing; nothing is substituted)."""

    if not path.exists():
        return None
    try:
        with rasterio.open(path) as src:
            crs = src.crs
            transform = src.transform
            facts = CanonicalTerrainFacts(
                path=path,
                readable=True,
                read_error=None,
                crs=crs.to_string() if crs is not None else None,
                crs_is_geographic=crs.is_geographic if crs is not None else None,
                crs_linear_units=crs.linear_units if crs is not None else None,
                transform=tuple(float(v) for v in tuple(transform)[:6]),
                width=src.width,
                height=src.height,
                pixel_size_x_m=float(transform.a),
                pixel_size_y_m=float(-transform.e),
                bounds=tuple(float(v) for v in src.bounds),
                nodata=src.nodata,
                dtype=src.dtypes[0],
                band_count=src.count,
                color_interpretations=tuple(str(c) for c in src.colorinterp),
                tags={k: str(v) for k, v in src.tags().items()},
                canonical_sha256=_sha256_of(path) if compute_sha256 else None,
            )
    except (RasterioIOError, RasterioError) as exc:
        return CanonicalTerrainFacts(
            path=path,
            readable=False,
            read_error=str(exc),
            crs=None,
            crs_is_geographic=None,
            crs_linear_units=None,
            transform=None,
            width=0,
            height=0,
            pixel_size_x_m=None,
            pixel_size_y_m=None,
            bounds=None,
            nodata=None,
            dtype=None,
            band_count=0,
            color_interpretations=(),
        )
    return facts


def read_canonical_band(path: Path) -> np.ndarray:
    """Band 1 as float32, exactly as stored (the accepted MAR-020 convention is float32 with NaN
    nodata). Never rescaled, never filled."""

    with rasterio.open(path) as src:
        return src.read(1)


# --- Section 22: terrain-screening readiness (delegating intrinsic checks) ------------------------


def _raster_facts_for_intrinsic_readiness(
    facts: CanonicalTerrainFacts, band: np.ndarray
) -> terrain_readiness.RasterFacts:
    valid = np.isfinite(band)
    values = band[valid]
    return terrain_readiness.RasterFacts(
        band_count=facts.band_count,
        dtype=str(facts.dtype),
        color_interpretations=facts.color_interpretations,
        crs_is_present=facts.crs is not None,
        crs_is_geographic=facts.crs_is_geographic,
        crs_linear_units=facts.crs_linear_units,
        width=facts.width,
        height=facts.height,
        pixel_size_x_m=facts.pixel_size_x_m,
        pixel_size_y_m=facts.pixel_size_y_m,
        bounds=facts.bounds,
        nodata_value=facts.nodata,
        vertical_datum=facts.tags.get("source_vertical_datum"),
        survey_epoch=facts.tags.get("survey_epoch"),
        data_min=float(values.min()) if values.size else None,
        data_max=float(values.max()) if values.size else None,
        data_std=float(values.std()) if values.size else None,
        valid_cell_fraction=float(valid.mean()) if valid.size else None,
    )


def assess_terrain_screening_readiness(
    facts: CanonicalTerrainFacts | None, band: np.ndarray | None
) -> dict[str, Any]:
    """TERRAIN_SCREENING_READY / TERRAIN_SCREENING_NOT_READY with controlled reasons. Intrinsic
    raster readiness is the accepted MAR-020 function's verdict, reported separately and never
    mutated; MAR-031 adds only its own integration conditions on top."""

    reasons: list[str] = []
    intrinsic: dict[str, Any] | None = None

    if facts is None:
        reasons.append(contract.HIGH_RESOLUTION_CURRENT_SEABED_GEOMETRY_NOT_AVAILABLE)
        reasons.append(
            f"{contract.CANONICAL_TERRAIN_NOT_AVAILABLE}: no accepted canonical terrain product "
            f"({contract.CANONICAL_TERRAIN_FILENAME}) exists for this study"
        )
        return {
            "status": contract.TERRAIN_SCREENING_NOT_READY,
            "reasons": reasons,
            "intrinsic_bathymetry_readiness": None,
        }

    if not facts.readable:
        reasons.append(f"{contract.CANONICAL_TERRAIN_NOT_READABLE}: {facts.read_error}")
        return {
            "status": contract.TERRAIN_SCREENING_NOT_READY,
            "reasons": reasons,
            "intrinsic_bathymetry_readiness": None,
        }

    role = facts.tags.get("scientific_role")
    layer = facts.tags.get("layer")
    if (
        role != contract.SOURCE_TERRAIN_ROLE_REQUIRED
        or layer != contract.SOURCE_TERRAIN_LAYER_REQUIRED
    ):
        reasons.append(
            f"{contract.CANONICAL_TERRAIN_ROLE_MISMATCH}: embedded scientific_role={role!r}, "
            f"layer={layer!r}; required {contract.SOURCE_TERRAIN_ROLE_REQUIRED!r} / "
            f"{contract.SOURCE_TERRAIN_LAYER_REQUIRED!r} (the raster is not the accepted canonical "
            "bed_elevation_m product; nothing is inferred from its values)"
        )
    if facts.is_rotated:
        reasons.append(f"{contract.TERRAIN_ROTATED_GRID_UNSUPPORTED}: transform={facts.transform}")
    if not facts.pixels_square_metric:
        reasons.append(
            f"{contract.TERRAIN_PIXELS_NOT_SQUARE_METRIC}: pixel size "
            f"{facts.pixel_size_x_m} x {facts.pixel_size_y_m}, "
            f"crs_is_geographic={facts.crs_is_geographic}"
        )

    if band is not None:
        intrinsic_result = terrain_readiness.assess_bathymetry_readiness(
            _raster_facts_for_intrinsic_readiness(facts, band)
        )
        intrinsic = intrinsic_result.to_dict()
        if intrinsic_result.status == terrain_readiness.NOT_READY:
            reasons.append(
                f"{contract.CANONICAL_TERRAIN_INTRINSIC_NOT_READY}: "
                + "; ".join(intrinsic_result.reasons())
            )

    status = contract.TERRAIN_SCREENING_NOT_READY if reasons else contract.TERRAIN_SCREENING_READY
    if status == contract.TERRAIN_SCREENING_NOT_READY:
        reasons.insert(0, contract.HIGH_RESOLUTION_CURRENT_SEABED_GEOMETRY_NOT_AVAILABLE)
    elif intrinsic is not None:
        # Informational only: intrinsic LIMITATION reasons stay visible but do not block.
        reasons.extend(f"intrinsic limitation: {r}" for r in intrinsic["limitation_reasons"])
    return {
        "status": status,
        "reasons": reasons,
        "intrinsic_bathymetry_readiness": intrinsic,
    }


# --- Section 10 / 32: regional slope context (context only, never an input) -----------------------


def inspect_regional_slope_context(metadata_path: Path | None) -> dict[str, Any]:
    """Report a broad regional morphology product (e.g. MAR-007) as REGIONAL_CONTEXT_ONLY. Reads
    the slope layers' descriptive statistics for visibility; derives nothing from them."""

    base: dict[str, Any] = {
        "role": contract.REGIONAL_CONTEXT_ONLY,
        "promoted_to_pipeline_scale_slope_stability_input": False,
        "normalized_strength_demand_derived_from_regional_slope": False,
        "note": contract.REGIONAL_CONTEXT_NOTE,
    }
    if metadata_path is None or not metadata_path.exists():
        return {**base, "status": REGIONAL_CONTEXT_NOT_AVAILABLE, "metadata_path": None}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": REGIONAL_CONTEXT_NOT_AVAILABLE,
            "metadata_path": str(metadata_path),
            "error": str(exc),
        }

    slope_layers: list[dict[str, Any]] = []
    for name, raster_ref in (metadata.get("raster_outputs") or {}).items():
        if "slope" not in name:
            continue
        raster_path = metadata_path.parent / Path(str(raster_ref).replace("\\", "/")).name
        layer: dict[str, Any] = {
            "layer": name,
            "path": str(raster_path),
            "exists": raster_path.exists(),
            "radius_m": (metadata.get("features") or {}).get(name, {}).get("radius_m"),
        }
        if raster_path.exists():
            try:
                with rasterio.open(raster_path) as src:
                    band = src.read(1).astype(np.float64)
                    if src.nodata is not None and not np.isnan(src.nodata):
                        band = np.where(band == src.nodata, np.nan, band)
                    layer["crs"] = src.crs.to_string() if src.crs else None
                    layer["pixel_size_m"] = float(src.transform.a)
                    layer["dimensions"] = {"width": src.width, "height": src.height}
                layer["slope_deg_summary"] = report.summarize_values(band)
            except (RasterioIOError, RasterioError) as exc:
                layer["error"] = str(exc)
        slope_layers.append(layer)

    return {
        **base,
        "status": REGIONAL_CONTEXT_AVAILABLE,
        "metadata_path": str(metadata_path),
        "source_product": metadata.get("source_product"),
        "source_nominal_resolution_m": metadata.get("source_nominal_resolution_m"),
        "analysis_grid_spacing_m": metadata.get("analysis_grid_spacing_m"),
        "underlying_acquisition_years": metadata.get("underlying_acquisition_years"),
        "slope_layers": slope_layers,
    }


# --- Sections 8-9, 14: per-scale terrain computation ----------------------------------------------


# Row-band tiling keeps peak memory bounded on very large rasters (the accepted MAR-020 slope
# function materializes ~10 float64 moment grids plus a batched 3x3 system per solvable cell). A
# band is processed with a halo of >= the window half-width on each side, so every windowed sum
# inside the band's own rows sees exactly the same neighbourhood as the untiled computation --
# the tiled result is IDENTICAL, not an approximation (proved by test, not asserted).
DEFAULT_MAX_TILE_CELLS = 10_000_000


def compute_scale_slope_and_demand(
    bed_elevation_m: np.ndarray,
    valid_mask: np.ndarray,
    scale_m: float,
    cell_size_m: float,
    *,
    max_tile_cells: int | None = DEFAULT_MAX_TILE_CELLS,
) -> tuple[np.ndarray, np.ndarray]:
    """One independent scale: accepted MAR-020 slope (called unchanged) -> normalized strength
    demand. Inputs are never modified. `max_tile_cells=None` disables tiling."""

    height, width = bed_elevation_m.shape
    slope_deg = _compute_slope_tiled(
        bed_elevation_m, valid_mask, scale_m, cell_size_m, max_tile_cells=max_tile_cells
    )
    assert slope_deg.shape == (height, width)
    # Section 26: the canonical nodata footprint is preserved. The accepted MAR-020 plane fit can
    # return a (window-supported) slope value AT a nodata cell when its neighbourhood is >= 90 %
    # valid; MAR-031 does not present a value where the canonical terrain itself has no
    # measurement, so the output support is restricted to the canonical valid mask. The MAR-020
    # function is called unchanged; only MAR-031's own product support is masked.
    slope_deg = np.where(valid_mask, slope_deg, np.nan)
    demand = core.compute_normalized_strength_demand(slope_deg)
    return slope_deg, demand


def _compute_slope_tiled(
    bed_elevation_m: np.ndarray,
    valid_mask: np.ndarray,
    scale_m: float,
    cell_size_m: float,
    *,
    max_tile_cells: int | None,
) -> np.ndarray:
    height, width = bed_elevation_m.shape
    halo_px = max(1, round(scale_m / cell_size_m))
    if max_tile_cells is None or height * width <= max_tile_cells:
        slope_deg, _aspect, _vf = terrain_derivatives.compute_slope_aspect_deg(
            bed_elevation_m, valid_mask, scale_m, cell_size_m
        )
        return slope_deg

    rows_per_tile = max(1, max_tile_cells // max(width, 1))
    slope_deg = np.full((height, width), np.nan, dtype=np.float64)
    for r0 in range(0, height, rows_per_tile):
        r1 = min(height, r0 + rows_per_tile)
        h0 = max(0, r0 - halo_px)
        h1 = min(height, r1 + halo_px)
        tile_slope, _aspect, _vf = terrain_derivatives.compute_slope_aspect_deg(
            bed_elevation_m[h0:h1], valid_mask[h0:h1], scale_m, cell_size_m
        )
        slope_deg[r0:r1] = tile_slope[r0 - h0 : r1 - h0]
        del tile_slope, _aspect, _vf
    return slope_deg


def _write_int8_raster(
    array: np.ndarray, transform: rasterio.Affine, crs: str, output_path: Path, tags: dict[str, Any]
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "int8",
        "crs": crs,
        "transform": transform,
        "nodata": None,
    }
    with rasterio.open(output_path, "w", **profile) as dataset:
        dataset.write(array.astype("int8"), 1)
        dataset.update_tags(**{k: str(v) for k, v in tags.items()})
    return output_path


# --- Orchestration --------------------------------------------------------------------------------


@dataclass
class SlopeInstabilityScreeningResult:
    project_id: str
    output_dir: Path
    terrain_facts: CanonicalTerrainFacts | None
    terrain_screening: dict[str, Any]
    regional_context: dict[str, Any]
    slope_scales_m: list[float]
    scale_results: list[dict[str, Any]]
    scenario_results: list[dict[str, Any]]
    outputs: dict[str, Any]
    metadata: dict[str, Any]
    readiness: dict[str, Any]
    metadata_path: Path
    readiness_path: Path
    contract_path: Path
    figure_path: Path | None


def run_slope_instability_screening(
    *,
    project_id: str,
    study_dir: Path,
    slope_scales_m: list[float] | tuple[float, ...] = contract.DEFAULT_SLOPE_SCALES_M,
    scenarios: list[core.UndrainedInfiniteSlopeScenario] | None = None,
    analysis_id: str | None = None,
    figure_title: str | None = None,
    regional_morphology_metadata_path: Path | None = None,
    render_figure: bool = True,
    log=print,
) -> SlopeInstabilityScreeningResult:
    """End-to-end screening for one study directory. Offline. Writes under
    `<study_dir>/slope_stability/` and `<study_dir>/maps/`; never touches `<study_dir>/terrain/`.

    `scenarios` defaults to NO scenario -- a factor of safety is computed only when the caller
    passes explicit `UndrainedInfiniteSlopeScenario` objects."""

    scenarios = list(scenarios or [])
    scales = [float(s) for s in slope_scales_m]
    if (
        not scales
        or len(set(scales)) != len(scales)
        or any(not np.isfinite(s) or s <= 0 for s in scales)
    ):
        raise core.SlopeStabilityInputError(
            f"slope_scales_m must be non-empty, unique, finite and > 0, got {scales}"
        )

    output_dir = study_dir / SLOPE_STABILITY_SUBDIR
    maps_dir = study_dir / MAPS_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = study_dir / TERRAIN_SUBDIR / contract.CANONICAL_TERRAIN_FILENAME
    if regional_morphology_metadata_path is None:
        regional_morphology_metadata_path = study_dir / MORPHOLOGY_METADATA_RELATIVE

    log(f"Inspecting canonical terrain {canonical_path} ...")
    facts = inspect_canonical_terrain(canonical_path)
    band: np.ndarray | None = None
    if facts is not None and facts.readable:
        band = read_canonical_band(canonical_path)
    terrain_screening = assess_terrain_screening_readiness(facts, band)
    log(f"  terrain_screening_status: {terrain_screening['status']}")
    for reason in terrain_screening["reasons"]:
        log(f"    - {reason}")

    regional_context = inspect_regional_slope_context(regional_morphology_metadata_path)
    log(
        f"  {contract.REGIONAL_SLOPE_CONTEXT}: {regional_context['status']} "
        f"({regional_context['role']})"
    )

    scale_results: list[dict[str, Any]] = []
    scenario_results: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    figure_path: Path | None = None

    if terrain_screening["status"] == contract.TERRAIN_SCREENING_READY:
        assert facts is not None and band is not None
        transform = facts.affine
        crs = facts.crs
        cell_size_m = float(facts.pixel_size_x_m)
        valid_mask = np.isfinite(band)
        full_shape = band.shape
        stride = maps.display_stride_for(full_shape)
        display_layers: dict[str, tuple[np.ndarray, str, str]] = {
            "A. Bathymetry (bed_elevation_m)": (
                maps.decimate_for_display(band, stride).astype(np.float64),
                "viridis",
                "m (higher = shallower)",
            )
        }

        tags_base = {
            "product": "MAR-031 generic submarine slope-instability screening POC",
            "product_role": contract.PRODUCT_ROLE,
            "source_terrain_role": contract.SOURCE_TERRAIN_ROLE_REQUIRED,
            "source_terrain_path": str(canonical_path),
            "source_terrain_sha256": facts.tags.get("source_sha256", ""),
            "canonical_terrain_sha256": facts.canonical_sha256 or "",
            "scale_semantics": contract.SCALE_SEMANTICS,
            "raster_integrity": "same CRS/transform/dimensions as canonical terrain; no "
            "reprojection, resampling, interpolation or nodata filling",
        }

        for scale_m in scales:
            log(f"Computing slope + normalized strength demand at {scale_m:g} m ...")
            slope_deg, demand = compute_scale_slope_and_demand(
                band, valid_mask, scale_m, cell_size_m
            )

            slope_path = terrain_raster_io.write_terrain_raster(
                slope_deg,
                transform,
                crs,
                output_dir / f"slope_{scale_m:g}m_deg.tif",
                {
                    **tags_base,
                    "layer": f"slope_{scale_m:g}m_deg",
                    "units": "deg",
                    "scale_m": scale_m,
                    "scientific_role": "HIGH_RESOLUTION_TERRAIN_SLOPE_MAGNITUDE",
                    "method": "accepted MAR-020 terrain.derivatives.compute_slope_aspect_deg",
                },
            )
            demand_path = terrain_raster_io.write_terrain_raster(
                demand,
                transform,
                crs,
                output_dir / f"normalized_strength_demand_{scale_m:g}m.tif",
                {
                    **tags_base,
                    "layer": f"normalized_strength_demand_{scale_m:g}m",
                    "units": "dimensionless",
                    "scale_m": scale_m,
                    "scientific_role": contract.SCIENTIFIC_ROLE_NORMALIZED_STRENGTH_DEMAND,
                    "definition": "sin(alpha) * cos(alpha) = s_u/(gamma_prime z) required for FS=1",
                    "not_a_factor_of_safety": True,
                    "not_a_hazard_or_susceptibility_product": True,
                },
            )
            slope_summary = report.summarize_values(slope_deg)
            demand_summary = report.summarize_values(demand)
            scale_results.append(
                {
                    "scale_m": scale_m,
                    "window_half_width_px": round(scale_m / cell_size_m),
                    "slope_deg": slope_summary,
                    "normalized_strength_demand": demand_summary,
                    "slope_path": str(slope_path),
                    "normalized_strength_demand_path": str(demand_path),
                }
            )
            outputs[f"slope_{scale_m:g}m_deg"] = str(slope_path)
            outputs[f"normalized_strength_demand_{scale_m:g}m"] = str(demand_path)
            log(
                f"  slope {slope_summary['min']}..{slope_summary['max']} deg; demand "
                f"{demand_summary['min']}..{demand_summary['max']}; valid "
                f"{demand_summary['valid_cells']:,}"
            )

            if scale_m == scales[0]:
                display_layers[f"B. Slope ({scale_m:g} m)"] = (
                    maps.decimate_for_display(slope_deg, stride),
                    "inferno",
                    "deg",
                )
            panel_letter = chr(ord("C") + scales.index(scale_m))
            display_layers[f"{panel_letter}. Norm. strength demand ({scale_m:g} m)"] = (
                maps.decimate_for_display(demand, stride),
                "magma",
                "sin(a)cos(a), dimensionless (required s_u/(gamma' z) for FS=1)",
            )

            for scenario in scenarios:
                log(f"  scenario {scenario.scenario_id} @ {scale_m:g} m ...")
                fos = core.compute_scenario_factor_of_safety(slope_deg, scenario)
                scenario_tags = {
                    **tags_base,
                    "scientific_role": contract.SCIENTIFIC_ROLE_FACTOR_OF_SAFETY_SCENARIO,
                    "scenario_id": scenario.scenario_id,
                    "parameter_basis": scenario.parameter_basis,
                    "material_model": scenario.material_model,
                    "undrained_shear_strength_kpa": scenario.undrained_shear_strength_kpa,
                    "submerged_unit_weight_kn_m3": scenario.submerged_unit_weight_kn_m3,
                    "slip_surface_depth_m": scenario.slip_surface_depth_m,
                    "scale_m": scale_m,
                    "disclaimer": "USER_DECLARED_HYPOTHETICAL_SCENARIO -- "
                    "NOT_SITE_SPECIFIC_MEASUREMENT",
                    "flat_cells": "factor_of_safety null where tau_driving_pa == 0 "
                    "(NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR); infinity never written",
                }
                fos_path = terrain_raster_io.write_terrain_raster(
                    fos.factor_of_safety,
                    transform,
                    crs,
                    output_dir / f"factor_of_safety_{scenario.scenario_id}_{scale_m:g}m.tif",
                    {**scenario_tags, "layer": "factor_of_safety", "units": "dimensionless"},
                )
                state_codes = core.encode_model_states(fos.model_state)
                state_path = _write_int8_raster(
                    state_codes,
                    transform,
                    crs,
                    output_dir / f"model_state_{scenario.scenario_id}_{scale_m:g}m.tif",
                    {
                        **scenario_tags,
                        "layer": "model_state_code",
                        "units": "code",
                        "legend": json.dumps(core.MODEL_STATE_CODES),
                    },
                )
                state_counts = report.summarize_model_states(fos.model_state)
                scenario_results.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "parameter_basis": scenario.parameter_basis,
                        "material_model": scenario.material_model,
                        "scale_m": scale_m,
                        "factor_of_safety": report.summarize_values(fos.factor_of_safety),
                        "tau_driving_pa": report.summarize_values(fos.tau_driving_pa),
                        "model_state_counts": state_counts,
                        "factor_of_safety_path": str(fos_path),
                        "model_state_path": str(state_path),
                        "disclaimer": "USER_DECLARED_HYPOTHETICAL_SCENARIO -- "
                        "NOT_SITE_SPECIFIC_MEASUREMENT",
                    }
                )
                outputs[f"factor_of_safety_{scenario.scenario_id}_{scale_m:g}m"] = str(fos_path)
                outputs[f"model_state_{scenario.scenario_id}_{scale_m:g}m"] = str(state_path)
                del fos, state_codes

            del slope_deg, demand

        if render_figure:
            log("Rendering screening figure ...")
            figure_path = maps.render_slope_instability_screening_figure(
                layers=display_layers,
                transform=transform,
                full_shape=full_shape,
                output_path=maps_dir / f"{project_id}_slope_instability_screening.png",
                title=figure_title or f"{project_id} -- Submarine Slope-Instability Screening POC",
                subtitle=(
                    "Terrain predisposition only: normalized strength demand sin(a)cos(a) is the "
                    "required s_u/(gamma' z) for FS=1, NOT a factor of safety, NOT a probability, "
                    "NOT a hazard class"
                ),
                footer_note=(
                    "OBSERVED bathymetry (A); MODELLED generic slope (B) and derived normalized "
                    "strength demand at independent 10 m / 50 m scales "
                    f"({contract.SCALE_SEMANTICS}). No soil parameters, no trigger, no runout, no "
                    f"risk score. Display stride {stride} (visual only; GeoTIFFs are native)."
                ),
            )
            outputs["figure"] = str(figure_path)
    else:
        log("Terrain not ready for local slope screening -- no slope/demand/FoS product computed.")

    del band

    contract_doc = contract.build_slope_instability_contract()
    contract_path = output_dir / CONTRACT_FILENAME
    contract_path.write_text(json.dumps(contract_doc, indent=2, default=str), encoding="utf-8")
    outputs["contract"] = str(contract_path)

    readiness = report.build_screening_readiness(
        project_id=project_id,
        terrain_screening=terrain_screening,
        scenarios_supplied=len(scenarios),
        scenario_results_written=len(scenario_results),
        regional_context=regional_context,
    )
    readiness_path = output_dir / READINESS_FILENAME
    readiness_path.write_text(json.dumps(readiness, indent=2, default=str), encoding="utf-8")
    outputs["readiness"] = str(readiness_path)

    metadata = report.build_screening_metadata(
        project_id=project_id,
        analysis_id=analysis_id,
        terrain_facts=facts.to_dict() if facts is not None else None,
        slope_scales_m=scales,
        scale_results=scale_results,
        scenarios=[s.to_dict() for s in scenarios],
        scenario_results=scenario_results,
        regional_context=regional_context,
        outputs=outputs,
    )
    metadata_path = output_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    outputs["metadata"] = str(metadata_path)

    return SlopeInstabilityScreeningResult(
        project_id=project_id,
        output_dir=output_dir,
        terrain_facts=facts,
        terrain_screening=terrain_screening,
        regional_context=regional_context,
        slope_scales_m=scales,
        scale_results=scale_results,
        scenario_results=scenario_results,
        outputs=outputs,
        metadata=metadata,
        readiness=readiness,
        metadata_path=metadata_path,
        readiness_path=readiness_path,
        contract_path=contract_path,
        figure_path=figure_path,
    )

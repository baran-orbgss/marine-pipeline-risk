"""Machine- and human-readable summaries for slope-instability screening (MAR-031 Section 29).

Pure builders over caller-supplied facts. Nothing here computes science, opens a raster, or
invents a number: every value is passed in from `screening`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

from marine_engine import __version__
from marine_engine.slope_stability import contract, core

SOFTWARE_VERSION = f"marine-engine {__version__} (MAR-031)"


def summarize_values(array: np.ndarray) -> dict[str, Any]:
    """Descriptive statistics of the FINITE cells of a raster array. `nodata_count` counts every
    non-finite cell (NaN nodata footprint preserved from the canonical terrain plus cells whose
    neighbourhood failed the accepted MAR-020 validity threshold). Nothing is filled."""

    finite = np.isfinite(array)
    valid_count = int(finite.sum())
    total = int(array.size)
    summary: dict[str, Any] = {
        "total_cells": total,
        "valid_cells": valid_count,
        "nodata_cells": total - valid_count,
        "valid_fraction": (valid_count / total) if total else None,
    }
    if valid_count == 0:
        summary.update(dict.fromkeys(("min", "max", "mean", "std", "p50", "p95", "p99")))
        return summary
    values = array[finite].astype(np.float64, copy=False)
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    summary.update(
        {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p50": float(p50),
            "p95": float(p95),
            "p99": float(p99),
        }
    )
    return summary


def summarize_model_states(model_state: np.ndarray) -> dict[str, int]:
    return {state: int(np.count_nonzero(model_state == state)) for state in core.MODEL_STATES}


def build_screening_metadata(
    *,
    project_id: str,
    analysis_id: str | None,
    terrain_facts: dict[str, Any] | None,
    slope_scales_m: list[float],
    scale_results: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    scenario_results: list[dict[str, Any]],
    regional_context: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    """Section 29 machine-readable metadata. Every not-modelled flag is asserted from the contract,
    never computed, so it cannot drift to `true` by accident."""

    return {
        "product_role": contract.PRODUCT_ROLE,
        "software_version": SOFTWARE_VERSION,
        "processing_timestamp_utc": datetime.now(UTC).isoformat(),
        "project_id": project_id,
        "analysis_id": analysis_id,
        "scientific_role": contract.SCIENTIFIC_ROLE_NORMALIZED_STRENGTH_DEMAND,
        "scientific_role_scenario": contract.SCIENTIFIC_ROLE_FACTOR_OF_SAFETY_SCENARIO,
        "source_terrain_role": contract.SOURCE_TERRAIN_ROLE_REQUIRED,
        "infinite_slope_equation": contract.INFINITE_SLOPE_EQUATION,
        "normalized_strength_demand_definition": contract.NORMALIZED_STRENGTH_DEMAND_DEFINITION,
        "units": dict(contract.UNITS),
        "slope_scales_m": list(slope_scales_m),
        "scale_semantics": contract.SCALE_SEMANTICS,
        "scale_note": contract.SCALE_NOTE,
        "slope_method": (
            "accepted MAR-020 marine_engine.terrain.derivatives.compute_slope_aspect_deg (local "
            "planar fit z = ax + by + c over a square window of the given physical half-width; "
            "slope = atan(sqrt(a^2 + b^2))), reused unchanged"
        ),
        "terrain_source_identity": (terrain_facts or {}).get("source_identity"),
        "terrain_source_sha256": (terrain_facts or {}).get("source_sha256"),
        "canonical_terrain_path": (terrain_facts or {}).get("path"),
        "canonical_terrain_sha256": (terrain_facts or {}).get("canonical_sha256"),
        "terrain_crs": (terrain_facts or {}).get("crs"),
        # Metric resolution is null unless the horizontal CRS unit is verified metre (MAR-031A).
        "terrain_resolution_m": (terrain_facts or {}).get("pixel_size_m"),
        "terrain_horizontal_crs_units": {
            key: (terrain_facts or {}).get(key)
            for key in (
                "crs",
                "crs_is_geographic",
                "crs_is_projected",
                "crs_linear_unit_name",
                "crs_linear_unit_to_m_factor",
                "pixel_size_x_crs_units",
                "pixel_size_y_crs_units",
                "pixel_size_x_m",
                "pixel_size_y_m",
                "horizontal_linear_unit_verified_metres",
            )
        }
        if terrain_facts
        else None,
        "terrain_dimensions": (terrain_facts or {}).get("dimensions"),
        "terrain_transform": (terrain_facts or {}).get("transform"),
        "terrain_vertical_datum": (terrain_facts or {}).get("source_vertical_datum"),
        "terrain_survey_epoch": (terrain_facts or {}).get("survey_epoch"),
        "raster_integrity": {
            "crs_preserved": True,
            "transform_preserved": True,
            "dimensions_preserved": True,
            "nodata_footprint_preserved": True,
            "reprojection": "none",
            "resampling": "none",
            "interpolation": "none",
            "nodata_filling": "none",
        },
        "scale_results": scale_results,
        "geotechnical_factor_of_safety_computed": bool(scenario_results),
        "geotechnical_parameter_basis": (
            core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO if scenarios else None
        ),
        "default_geotechnical_scenario_exists": False,
        "geotechnical_scenarios": scenarios,
        "scenario_results": scenario_results,
        "scenario_output_disclaimer": (
            "USER_DECLARED_HYPOTHETICAL_SCENARIO -- NOT_SITE_SPECIFIC_MEASUREMENT"
            if scenarios
            else None
        ),
        "model_states": list(core.MODEL_STATES),
        "model_state_codes": dict(core.MODEL_STATE_CODES),
        "model_applicability": contract.MODEL_APPLICABILITY,
        **dict(contract.NOT_MODELLED_FLAGS),
        "trigger_note": contract.TRIGGER_NOTE,
        "regional_slope_context": regional_context,
        "outputs": outputs,
        "references": [dict(r) for r in contract.REFERENCES],
        "limitations": list(contract.LIMITATIONS),
    }


def build_screening_readiness(
    *,
    project_id: str,
    terrain_screening: dict[str, Any],
    scenarios_supplied: int,
    scenario_results_written: int,
    regional_context: dict[str, Any],
) -> dict[str, Any]:
    """Section 22 readiness: four separate questions, four separate answers, no percentage."""

    terrain_ready = terrain_screening["status"] == contract.TERRAIN_SCREENING_READY

    if scenarios_supplied == 0:
        geotechnical_status = contract.GEOTECHNICAL_STABILITY_NOT_EVALUABLE
        geotechnical_reasons = [
            contract.NO_GEOTECHNICAL_SCENARIO_SUPPLIED,
            contract.SITE_GEOTECHNICAL_PROFILE_UNAVAILABLE,
        ]
    elif not terrain_ready:
        geotechnical_status = contract.GEOTECHNICAL_STABILITY_NOT_EVALUABLE
        geotechnical_reasons = [
            "explicit hypothetical scenarios were supplied but the terrain is not ready for local "
            "slope screening; no factor of safety was computed",
            contract.SITE_GEOTECHNICAL_PROFILE_UNAVAILABLE,
        ]
    else:
        geotechnical_status = contract.GEOTECHNICAL_STABILITY_SCENARIO_AVAILABLE
        geotechnical_reasons = [
            f"{scenarios_supplied} explicit USER_DECLARED_HYPOTHETICAL_SCENARIO(s) evaluated; "
            "NOT_SITE_SPECIFIC_MEASUREMENT",
            contract.SITE_GEOTECHNICAL_PROFILE_UNAVAILABLE,
        ]

    if terrain_ready and geotechnical_status == contract.GEOTECHNICAL_STABILITY_SCENARIO_AVAILABLE:
        local_status = contract.HYPOTHETICAL_SCENARIO_FOS_AVAILABLE_NOT_SITE_SPECIFIC
    else:
        local_status = contract.LOCAL_SLOPE_STABILITY_NOT_EVALUABLE

    return {
        "product_role": contract.PRODUCT_ROLE,
        "project_id": project_id,
        "software_version": SOFTWARE_VERSION,
        "questions": {
            "terrain_suitable_for_local_slope_screening": terrain_ready,
            "geotechnical_parameters_supplied": scenarios_supplied > 0,
            "factor_of_safety_evaluable": scenario_results_written > 0,
            "trigger_modelling_available": False,
        },
        "terrain_screening_status": terrain_screening["status"],
        "terrain_screening_reasons": list(terrain_screening.get("reasons", [])),
        "pipeline_scale_slope_stability_terrain_readiness": (
            contract.PIPELINE_SCALE_TERRAIN_READY
            if terrain_ready
            else contract.PIPELINE_SCALE_TERRAIN_NOT_READY
        ),
        "pipeline_scale_slope_stability_terrain_reason": (
            None
            if terrain_ready
            else contract.HIGH_RESOLUTION_CURRENT_SEABED_GEOMETRY_NOT_AVAILABLE
        ),
        "intrinsic_bathymetry_readiness": terrain_screening.get("intrinsic_bathymetry_readiness"),
        # MAR-031A: observed horizontal CRS unit evidence (raw CRS-unit spacing vs metric spacing).
        "terrain_horizontal_crs_units": terrain_screening.get("terrain_horizontal_crs_units"),
        "geotechnical_status": geotechnical_status,
        "geotechnical_reasons": geotechnical_reasons,
        "geotechnical_parameter_basis": (
            core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO if scenarios_supplied else None
        ),
        "trigger_status": contract.TRIGGER_RESPONSE_NOT_MODELLED,
        "trigger_note": contract.TRIGGER_NOTE,
        "local_slope_stability_status": local_status,
        "regional_slope_context": regional_context,
        "site_specific_factor_of_safety_available": False,
        "landslide_probability_available": False,
        "risk_score_available": False,
        **dict(contract.NOT_MODELLED_FLAGS),
    }


def format_summary_lines(
    *,
    project_id: str,
    readiness: dict[str, Any],
    scale_results: list[dict[str, Any]],
    scenario_results: list[dict[str, Any]],
    outputs: dict[str, Any],
) -> list[str]:
    """Console summary. Every line is derived from the run's own facts."""

    lines: list[str] = []
    lines.append(f"=== Submarine Slope-Instability Screening POC (MAR-031) -- {project_id} ===")
    lines.append("")
    lines.append("## Readiness")
    lines.append(f"  terrain_screening_status: {readiness['terrain_screening_status']}")
    for reason in readiness["terrain_screening_reasons"]:
        lines.append(f"    - {reason}")
    units = readiness.get("terrain_horizontal_crs_units")
    if units:
        lines.append(
            f"  terrain horizontal CRS unit: {units.get('crs')} / "
            f"{units.get('crs_linear_unit_name')!r} (to-metre factor "
            f"{units.get('crs_linear_unit_to_m_factor')!r}); raw spacing "
            f"{units.get('pixel_size_x_crs_units')} x {units.get('pixel_size_y_crs_units')} "
            f"CRS units; horizontal_linear_unit_verified_metres="
            f"{str(units.get('horizontal_linear_unit_verified_metres')).lower()}; "
            f"pixel_size_m={units.get('pixel_size_x_m')}"
        )
    lines.append(
        "  pipeline_scale_slope_stability_terrain_readiness: "
        f"{readiness['pipeline_scale_slope_stability_terrain_readiness']}"
        + (
            f" ({readiness['pipeline_scale_slope_stability_terrain_reason']})"
            if readiness["pipeline_scale_slope_stability_terrain_reason"]
            else ""
        )
    )
    lines.append(f"  geotechnical_status: {readiness['geotechnical_status']}")
    for reason in readiness["geotechnical_reasons"]:
        lines.append(f"    - {reason}")
    lines.append(f"  trigger_status: {readiness['trigger_status']}")
    lines.append(f"  local_slope_stability_status: {readiness['local_slope_stability_status']}")
    regional = readiness["regional_slope_context"]
    lines.append(f"  {contract.REGIONAL_SLOPE_CONTEXT}: {regional['status']} ({regional['role']})")
    lines.append("")
    lines.append("## Terrain scales (independent; never averaged or maximized)")
    if not scale_results:
        lines.append("  none computed")
    for result in scale_results:
        slope = result["slope_deg"]
        demand = result["normalized_strength_demand"]
        lines.append(
            f"  {result['scale_m']:g} m: slope {slope['min']:.4f}..{slope['max']:.4f} deg "
            f"(mean {slope['mean']:.4f}, p99 {slope['p99']:.4f}); demand "
            f"{demand['min']:.6f}..{demand['max']:.6f} (mean {demand['mean']:.6f}); "
            f"valid {demand['valid_cells']:,} / nodata {demand['nodata_cells']:,}"
            if slope["valid_cells"]
            else f"  {result['scale_m']:g} m: no valid cells"
        )
    lines.append("")
    lines.append("## Geotechnical scenarios")
    if not scenario_results:
        lines.append("  none supplied -- no factor of safety computed (no default scenario exists)")
    for result in scenario_results:
        lines.append(
            f"  {result['scenario_id']} @ {result['scale_m']:g} m "
            f"[{result['parameter_basis']}; {result['material_model']}]: "
            f"states {result['model_state_counts']}"
        )
    lines.append("")
    lines.append("## Outputs")
    for key, value in outputs.items():
        lines.append(f"  {key}: {value}")
    return lines

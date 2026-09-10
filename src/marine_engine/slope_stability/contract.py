"""Slope-instability screening contract: science roles, readiness vocabulary, model
applicability, not-modelled flags, references, limitations (MAR-031 Sections 5-7, 17-22, 29).

Documentation-grade constants only -- no per-run data, no computation. Every string here is the
single accepted spelling used by `core`, `screening`, `report`, the CLI and the tests.
"""

from __future__ import annotations

from typing import Any

from marine_engine.slope_stability import core

# --- Scientific roles -----------------------------------------------------------------------------

SCIENTIFIC_ROLE_NORMALIZED_STRENGTH_DEMAND = "UNDRAINED_INFINITE_SLOPE_NORMALIZED_STRENGTH_DEMAND"
SCIENTIFIC_ROLE_FACTOR_OF_SAFETY_SCENARIO = "UNDRAINED_INFINITE_SLOPE_FACTOR_OF_SAFETY_SCENARIO"
PRODUCT_ROLE = "GENERIC_SUBMARINE_SLOPE_INSTABILITY_SCREENING_POC"

# The accepted MAR-020 canonical terrain product this screening consumes (its own embedded tag).
SOURCE_TERRAIN_ROLE_REQUIRED = "HIGH_RESOLUTION_SEABED_TERRAIN_POC"
SOURCE_TERRAIN_LAYER_REQUIRED = "bed_elevation_m"
CANONICAL_TERRAIN_FILENAME = "canonical_bed_elevation.tif"

# --- Scale semantics (Section 8-9) ----------------------------------------------------------------

SCALE_SEMANTICS = "SLOPE_SCALE_SENSITIVITY_NOT_FAILURE_SURFACE_SCALE"
DEFAULT_SLOPE_SCALES_M: tuple[float, ...] = (10.0, 50.0)
SCALE_NOTE = (
    "Each slope scale is an independent terrain product from the same canonical terrain. The 10 m "
    "and 50 m scales are never averaged, never combined by maximum, and neither is selected as the "
    "correct landslide scale: the failure-surface geometry is unknown, so scale differences "
    "express slope-estimate sensitivity to analysis window, not failure-surface size."
)

# --- PL854 boundary (Section 10, 32) --------------------------------------------------------------

REGIONAL_SLOPE_CONTEXT = "REGIONAL_SLOPE_CONTEXT"
REGIONAL_CONTEXT_ONLY = "REGIONAL_CONTEXT_ONLY"
PIPELINE_SCALE_SLOPE_STABILITY_INPUT = "PIPELINE_SCALE_SLOPE_STABILITY_INPUT"
REGIONAL_CONTEXT_NOTE = (
    "Broad regional morphology slope (e.g. MAR-007 EMODnet DTM 2024, ~115 m-class source, "
    "500/1000 m windows, underlying surveys mainly 1991-1992) is REGIONAL_SLOPE_CONTEXT only. It "
    "is never promoted to PIPELINE_SCALE_SLOPE_STABILITY_INPUT and no normalized strength demand "
    "or factor of safety is derived from it."
)

# --- Readiness vocabulary (Section 22) ------------------------------------------------------------

TERRAIN_SCREENING_READY = "TERRAIN_SCREENING_READY"
TERRAIN_SCREENING_NOT_READY = "TERRAIN_SCREENING_NOT_READY"

GEOTECHNICAL_STABILITY_SCENARIO_AVAILABLE = "GEOTECHNICAL_STABILITY_SCENARIO_AVAILABLE"
GEOTECHNICAL_STABILITY_NOT_EVALUABLE = "GEOTECHNICAL_STABILITY_NOT_EVALUABLE"

TRIGGER_RESPONSE_NOT_MODELLED = "TRIGGER_RESPONSE_NOT_MODELLED"

LOCAL_SLOPE_STABILITY_NOT_EVALUABLE = "LOCAL_SLOPE_STABILITY_NOT_EVALUABLE"
HYPOTHETICAL_SCENARIO_FOS_AVAILABLE_NOT_SITE_SPECIFIC = (
    "HYPOTHETICAL_SCENARIO_FOS_AVAILABLE_NOT_SITE_SPECIFIC"
)

PIPELINE_SCALE_TERRAIN_READY = "READY"
PIPELINE_SCALE_TERRAIN_NOT_READY = "NOT_READY"

# Controlled reason codes.
HIGH_RESOLUTION_CURRENT_SEABED_GEOMETRY_NOT_AVAILABLE = (
    "HIGH_RESOLUTION_CURRENT_SEABED_GEOMETRY_NOT_AVAILABLE"
)
CANONICAL_TERRAIN_NOT_AVAILABLE = "CANONICAL_TERRAIN_NOT_AVAILABLE"
CANONICAL_TERRAIN_NOT_READABLE = "CANONICAL_TERRAIN_NOT_READABLE"
CANONICAL_TERRAIN_ROLE_MISMATCH = "CANONICAL_TERRAIN_ROLE_MISMATCH"
CANONICAL_TERRAIN_INTRINSIC_NOT_READY = "CANONICAL_TERRAIN_INTRINSIC_NOT_READY"
TERRAIN_PIXELS_NOT_SQUARE_METRIC = "TERRAIN_PIXELS_NOT_SQUARE_METRIC"
TERRAIN_ROTATED_GRID_UNSUPPORTED = "TERRAIN_ROTATED_GRID_UNSUPPORTED"
SITE_GEOTECHNICAL_PROFILE_UNAVAILABLE = "SITE_GEOTECHNICAL_PROFILE_UNAVAILABLE"
NO_GEOTECHNICAL_SCENARIO_SUPPLIED = "NO_GEOTECHNICAL_SCENARIO_SUPPLIED"

# --- Equations / definitions (Sections 4-5, 14) ---------------------------------------------------

INFINITE_SLOPE_EQUATION = (
    "FS = s_u / (gamma_prime * z * sin(alpha) * cos(alpha)); "
    "tau_d = gamma_prime * z * sin(alpha) * cos(alpha); "
    "alpha = seabed / slip-plane inclination, s_u = undrained shear strength [Pa], "
    "gamma_prime = submerged unit weight [N/m3], z = vertical depth below seabed to the assumed "
    "slip surface [m]; classical undrained translational infinite-slope idealization "
    "(Baeten et al. 2014: shallow slab, basal rupture approximately parallel to the slope)"
)
NORMALIZED_STRENGTH_DEMAND_DEFINITION = (
    "normalized_undrained_strength_demand = sin(alpha) * cos(alpha) = N_u_required, the "
    "normalized undrained shear-strength ratio s_u / (gamma_prime * z) required to reach FS = 1 "
    "under the simplified translational infinite-slope geometry. NOT actual soil strength, NOT a "
    "factor of safety, NOT a probability of failure, NOT a landslide susceptibility score, NOT a "
    "landslide risk."
)
UNITS: dict[str, str] = {
    "slope_deg": "degrees",
    "normalized_undrained_strength_demand": "dimensionless",
    "undrained_shear_strength": "Pa (input convenience kPa, converted explicitly x1000)",
    "submerged_unit_weight": "N/m3 (input convenience kN/m3, converted explicitly x1000)",
    "slip_surface_depth": "m",
    "tau_driving": "Pa",
    "factor_of_safety": "dimensionless",
}

PORE_PRESSURE_SEMANTICS = (
    "Hydrostatic water-pressure effects are represented through submerged unit weight in this "
    "simplified geometry. EXCESS PORE PRESSURE GENERATION / DISSIPATION IS NOT MODELLED (rapid "
    "sedimentation, earthquakes, cyclic loading, gas, strain contraction). The absence of an "
    "excess-pore-pressure model is NOT a measurement that excess pore pressure equals zero; it is "
    "NOT MODELLED."
)

# --- Model applicability (Section 18) -------------------------------------------------------------

MODEL_APPLICABILITY: dict[str, Any] = {
    "material_model": core.MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE,
    "applicable_to": [
        "thin translational slab",
        "failure surface approximately parallel to seabed",
        "cohesive sediment",
        "undrained total-stress idealization",
        "static gravitational loading",
    ],
    "not_automatically_applicable_to": [
        "granular drained sand slopes",
        "rock slopes",
        "rotational failures",
        "deep-seated curved slip surfaces",
        "progressive weak-layer propagation",
        "retrogressive slides",
        "sensitive-clay post-failure evolution",
        "liquefaction",
        "debris flows",
        "turbidity currents",
    ],
    "pore_pressure_semantics": PORE_PRESSURE_SEMANTICS,
}

# --- Not-modelled flags (Sections 17, 19, 20, 29) -------------------------------------------------

NOT_MODELLED_FLAGS: dict[str, bool] = {
    "excess_pore_pressure_modelled": False,
    "earthquake_trigger_modelled": False,
    "wave_trigger_modelled": False,
    "toe_erosion_trigger_modelled": False,
    "rapid_sedimentation_trigger_modelled": False,
    "gas_trigger_modelled": False,
    "progressive_failure_modelled": False,
    "retrogressive_failure_modelled": False,
    "liquefaction_modelled": False,
    "runout_modelled": False,
    "post_failure_dynamics_modelled": False,
    "pipeline_impact_modelled": False,
    "landslide_probability_computed": False,
    "risk_score_computed": False,
}
TRIGGER_NOTE = (
    "A missing trigger model is not evidence that the trigger is absent; trigger response is "
    "TRIGGER_RESPONSE_NOT_MODELLED."
)

# --- Section 7: source literature -----------------------------------------------------------------

REFERENCES: tuple[dict[str, str], ...] = (
    {
        "key": "Masson et al. 2006",
        "citation": "D.G. Masson, C.B. Harbitz, R.B. Wynn, G. Pedersen, F. Lovholt (2006). "
        "Submarine landslides: processes, triggers and hazard prediction. Philosophical "
        "Transactions of the Royal Society A, 364, 2009-2039.",
        "doi": "10.1098/rsta.2006.1810",
        "used_for": "weak layers; elevated pore pressure; earthquake triggering; limitations of "
        "simple terrain-only hazard inference",
    },
    {
        "key": "Locat & Lee 2002",
        "citation": "J. Locat, H.J. Lee (2002). Submarine landslides: advances and challenges. "
        "Canadian Geotechnical Journal, 39, 193-212.",
        "doi": "10.1139/t01-089",
        "used_for": "general submarine mass-movement hazard context",
    },
    {
        "key": "Kvalstad et al. 2005",
        "citation": "T.J. Kvalstad et al. (2005). The Storegga slide: evaluation of triggering "
        "sources and slide mechanics. Marine and Petroleum Geology, 22, 245-256.",
        "doi": "10.1016/j.marpetgeo.2004.10.019",
        "used_for": "large failures can occur at very low slope angles; excess pore pressure; "
        "strain softening; progressive / retrogressive mechanisms. Storegga soil parameters are "
        "NOT copied to any other site.",
    },
    {
        "key": "Baeten et al. 2014",
        "citation": "N.J. Baeten et al. (2014). Origin of shallow submarine mass movements and "
        "their glide planes -- Sedimentological and geotechnical analyses from the continental "
        "slope off northern Norway. Journal of Geophysical Research: Earth Surface, 119, "
        "2335-2360.",
        "doi": "10.1002/2013JF003068",
        "used_for": "principal basis for the simplified undrained infinite-slope equation. Baeten "
        "et al. state the simple infinite-slope model does not determine runout, spreading, "
        "mass-flow dynamics, or disintegration.",
    },
)

# --- Limitations ----------------------------------------------------------------------------------

LIMITATIONS: tuple[str, ...] = (
    "Slope alone is not stability: submarine landslides occur on gradients below 2 degrees where "
    "weak layers, excess pore pressure, strain softening, rapid deposition, earthquakes, toe "
    "erosion or other mechanisms are present (Masson et al. 2006; Kvalstad et al. 2005).",
    "normalized_undrained_strength_demand is a terrain-derived required-strength ratio, not "
    "actual instability, not a factor of safety, not a probability, not a susceptibility class.",
    "No slope-angle hazard class (LOW / MODERATE / HIGH / CRITICAL) and no universal critical "
    "slope angle exist in this product; no threshold is borrowed from another region.",
    "A factor of safety is computed ONLY for an explicit USER_DECLARED_HYPOTHETICAL_SCENARIO; "
    "there is no default soil scenario and no parameter is inferred from literature, BGS Folk "
    "class, sand/mud/gravel percentage, nearest PSA sample, terrain morphology, or water depth.",
    "The scenario factor of safety is a static, hydrostatic-only, undrained total-stress, "
    "translational infinite-slope idealization; it is not validated site-specific slope stability.",
    "Model states (MODEL_FS_BELOW_1 / AT_1 / ABOVE_1) are descriptive positions relative to the "
    "limit-equilibrium boundary FS = 1; they are not safety verdicts (no safe/unsafe "
    "labelling) and no design factor (e.g. FS >= 1.5) is applied.",
    "At exactly zero slope the idealized driving shear is zero and the factor of safety is null "
    "(NO_DOWNSLOPE_GRAVITATIONAL_DRIVING_SHEAR), never infinity; no minimum slope threshold is "
    "introduced, so a very small finite slope legitimately yields a very large finite scenario FS.",
    "Excess pore pressure, earthquake / wave / toe-erosion / sedimentation / gas / human "
    "triggers, progressive and retrogressive failure, liquefaction, runout, slide volume / "
    "velocity, pipeline impact force / displacement and tsunami are NOT modelled.",
    "10 m and 50 m slope scales are independent terrain products; neither is the failure-surface "
    "scale (SLOPE_SCALE_SENSITIVITY_NOT_FAILURE_SURFACE_SCALE).",
    "Broad regional slope context (~115 m-class EMODnet, 500/1000 m windows) is "
    "REGIONAL_CONTEXT_ONLY and is never promoted to a pipeline-scale slope-stability input.",
)


def build_slope_instability_contract() -> dict[str, Any]:
    """The static contract document written alongside every run."""

    return {
        "product_role": PRODUCT_ROLE,
        "scientific_roles": {
            "normalized_strength_demand": SCIENTIFIC_ROLE_NORMALIZED_STRENGTH_DEMAND,
            "factor_of_safety_scenario": SCIENTIFIC_ROLE_FACTOR_OF_SAFETY_SCENARIO,
        },
        "source_terrain_role_required": SOURCE_TERRAIN_ROLE_REQUIRED,
        "infinite_slope_equation": INFINITE_SLOPE_EQUATION,
        "normalized_strength_demand_definition": NORMALIZED_STRENGTH_DEMAND_DEFINITION,
        "units": dict(UNITS),
        "scale_semantics": SCALE_SEMANTICS,
        "scale_note": SCALE_NOTE,
        "regional_context_note": REGIONAL_CONTEXT_NOTE,
        "scenario_contract": {
            "material_model_required": (
                core.MATERIAL_MODEL_COHESIVE_UNDRAINED_TRANSLATIONAL_INFINITE_SLOPE
            ),
            "parameter_basis_required": core.PARAMETER_BASIS_USER_DECLARED_HYPOTHETICAL_SCENARIO,
            "parameters": [
                "scenario_id",
                "material_model",
                "parameter_basis",
                "undrained_shear_strength_kpa",
                "submerged_unit_weight_kn_m3",
                "slip_surface_depth_m",
            ],
            "validation": "finite and > 0 for each numeric parameter; zero, negative, NaN and "
            "infinity rejected; no clipping, no defaults, no inferred values",
            "default_scenario_exists": False,
        },
        "model_states": list(core.MODEL_STATES),
        "model_state_codes": dict(core.MODEL_STATE_CODES),
        "limit_equilibrium_boundary": core.FS_LIMIT_EQUILIBRIUM,
        "model_applicability": MODEL_APPLICABILITY,
        "not_modelled": dict(NOT_MODELLED_FLAGS),
        "trigger_note": TRIGGER_NOTE,
        "readiness_vocabulary": [
            TERRAIN_SCREENING_READY,
            TERRAIN_SCREENING_NOT_READY,
            GEOTECHNICAL_STABILITY_SCENARIO_AVAILABLE,
            GEOTECHNICAL_STABILITY_NOT_EVALUABLE,
            TRIGGER_RESPONSE_NOT_MODELLED,
            LOCAL_SLOPE_STABILITY_NOT_EVALUABLE,
            HYPOTHETICAL_SCENARIO_FOS_AVAILABLE_NOT_SITE_SPECIFIC,
        ],
        "references": [dict(r) for r in REFERENCES],
        "limitations": list(LIMITATIONS),
    }

"""Generic submarine slope-instability screening POC (MAR-031).

Two strictly separated levels, never conflated:

    A. TERRAIN / GEOMETRIC PREDISPOSITION (real bathymetry only)
       slope_deg (accepted MAR-020 `terrain.derivatives.compute_slope_aspect_deg`, reused, never
       reimplemented)  ->  normalized_undrained_strength_demand = sin(alpha) * cos(alpha)

    B. OPTIONAL GEOTECHNICAL SCENARIO (explicit user-declared hypothetical parameters only)
       undrained infinite-slope factor of safety
       FS = s_u / (gamma_prime * z * sin(alpha) * cos(alpha))

No built-in soil scenario exists. No slope-angle hazard class exists. Nothing here is a
landslide probability, a landslide risk, a validated site-specific stability assessment, a
runout / debris-flow / pipeline-impact model, a trigger (earthquake / wave / toe-erosion /
sedimentation / gas) model, or an excess-pore-pressure model.

Zero dependency on any specific project or dataset: every function takes plain arrays and
explicit parameters.
"""

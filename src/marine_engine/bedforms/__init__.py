"""Generic sand-wave/bedform morphodynamics engine (MAR-022).

Zero dependency on any specific project or dataset -- no Sheringham,
PL854, or other site coordinates/identifiers appear anywhere in this
package. The single-epoch spectral/transect/crest-trough scientific core
(planar detrending, filtering, spectral diagnostics, morphometrics) is NOT
duplicated here -- it already exists in `marine_engine.morphology.
sandwave_morphometry` (MAR-017) and is reused unchanged.

This package adds exactly the NEW generic capability MAR-017 did not
cover:

- `interpretation`: classifying an operator's own GIS interpretation
  layers into natural-bedform / anthropogenic-disturbance / unclassified
  categories, from a caller-supplied vocabulary mapping (never a
  hardcoded source-specific descriptor list).
- `natural_context`: per-tile natural-vs-anthropogenic validation status,
  using exact geometric intersection only -- never an invented buffer.
- `extraction`: per-tile, per-epoch transect sampling and bedform
  extraction from an in-memory elevation array (the array-based
  counterpart of the zip/rasterio-based helpers in `marine_engine.
  analogs`).
- `matching`: independent multi-epoch crest matching (spatial proximity +
  orientation + wavelength-scale consistency + local cross-crest-normal
  displacement), never seeded from source interpretations, never scored
  with a numeric confidence value, never using DoD to force a match.
- `contract` / `maps` / `report`: generic input-contract, figure, and
  report-block builders in this project's established per-package
  self-containment convention.

STATIC bedform geometry, OBSERVED multi-epoch bedform change, and FUTURE
migration prediction are three distinct things (Section 1) -- this
package implements only the first two. No prediction, no susceptibility,
no hazard/risk score, no ML anywhere in this package.
"""

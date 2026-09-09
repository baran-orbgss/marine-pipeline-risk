"""Canonical asset category and evidence-role vocabulary (MAR-026 Sections 4, 7).

Zero dependency on any specific project or dataset. Evidence roles are orthogonal axes:
`SOURCE_INTERPRETED` data can never be automatically re-labelled `MEASURED`, and `DERIVED`
output can never be automatically re-labelled source evidence -- both are declared explicitly
by the manifest, never inferred from filename text or asset category.
"""

from __future__ import annotations

# --- Section 4: evidence roles -- orthogonal, never interchangeable, never inferred ------------

PROJECT_GEOMETRY = "PROJECT_GEOMETRY"
MEASURED = "MEASURED"
SOURCE_INTERPRETED = "SOURCE_INTERPRETED"
DERIVED = "DERIVED"

EVIDENCE_ROLES = frozenset({PROJECT_GEOMETRY, MEASURED, SOURCE_INTERPRETED, DERIVED})

# --- Section 7: asset category vocabulary -------------------------------------------------------
# Only these three have a real readiness adapter in MAR-026.

PIPELINE_ROUTE = "PIPELINE_ROUTE"
BATHYMETRY_RASTER = "BATHYMETRY_RASTER"
BURIAL_PROFILE = "BURIAL_PROFILE"

CATEGORIES_WITH_READINESS_ADAPTERS = frozenset({PIPELINE_ROUTE, BATHYMETRY_RASTER, BURIAL_PROFILE})

# Forward-compatible vocabulary: registrable (identity + provenance recorded) but with no
# scientific interpretation implemented in MAR-026 -- see REGISTERED_READINESS_NOT_IMPLEMENTED.
METOCEAN = "METOCEAN"
SEDIMENT_SAMPLE = "SEDIMENT_SAMPLE"
CPT = "CPT"
BOREHOLE = "BOREHOLE"
FREESPAN_OBSERVATION = "FREESPAN_OBSERVATION"
SHALLOW_GAS_INTERPRETATION = "SHALLOW_GAS_INTERPRETATION"
FAULT_INTERPRETATION = "FAULT_INTERPRETATION"
BURIED_CHANNEL_INTERPRETATION = "BURIED_CHANNEL_INTERPRETATION"
BOULDER_CATALOGUE = "BOULDER_CATALOGUE"
EXISTING_INFRASTRUCTURE = "EXISTING_INFRASTRUCTURE"
OTHER = "OTHER"

FUTURE_CATEGORIES_WITHOUT_ADAPTERS = frozenset(
    {
        METOCEAN,
        SEDIMENT_SAMPLE,
        CPT,
        BOREHOLE,
        FREESPAN_OBSERVATION,
        SHALLOW_GAS_INTERPRETATION,
        FAULT_INTERPRETATION,
        BURIED_CHANNEL_INTERPRETATION,
        BOULDER_CATALOGUE,
        EXISTING_INFRASTRUCTURE,
        OTHER,
    }
)

ASSET_CATEGORIES = CATEGORIES_WITH_READINESS_ADAPTERS | FUTURE_CATEGORIES_WITHOUT_ADAPTERS

# Section 7: a future category present in a manifest is registered honestly under this status --
# it must NEVER be reported as READY (or any real readiness status) merely because a file exists.
REGISTERED_READINESS_NOT_IMPLEMENTED = "REGISTERED_READINESS_NOT_IMPLEMENTED"

# --- MAR-027 Section 8: explicit asset -> route relationship vocabulary ---------------------------
# A relationship exists ONLY when the manifest declares it -- never inferred from a filename, a
# shared directory, a matching CRS, or an asset category. Deliberately narrow: one type for now.

ROUTE_REFERENCED = "ROUTE_REFERENCED"

ROUTE_RELATIONSHIP_TYPES = frozenset({ROUTE_REFERENCED})

# --- MAR-027 Section 16: explicit linear-reference declaration vocabulary -----------------------
# The ONLY supported basis means exactly: numeric chainage values are distances along the
# referenced canonical project route, measured from that route's own geometry-start origin. A
# column merely named "chainage"/"kp" never implies this.

CANONICAL_ROUTE_FROM_GEOMETRY_START = "CANONICAL_ROUTE_FROM_GEOMETRY_START"

LINEAR_REFERENCE_BASES = frozenset({CANONICAL_ROUTE_FROM_GEOMETRY_START})

LINEAR_REFERENCE_UNITS = frozenset({"m"})

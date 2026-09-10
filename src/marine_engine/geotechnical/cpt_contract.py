"""Canonical CPT/CPTU evidence vocabulary (MAR-032 Sections 9, 11, 13, 20-22, 27, 28).

Every status, role, axis, field name and unit-conversion factor used by the generic CPT layer is
defined here once, explicitly. Nothing in this module is numeric-score-like: readiness is an
explicit state vocabulary with named reasons (no 0-100 scores, no percentages, no weights).
"""

from __future__ import annotations

# --- Section 20: readiness status vocabulary ----------------------------------------------------

READY = "READY"
READY_WITH_LIMITATIONS = "READY_WITH_LIMITATIONS"
NOT_READY = "NOT_READY"
NOT_AVAILABLE = "NOT_AVAILABLE"
NOT_EVALUABLE = "NOT_EVALUABLE"

READINESS_STATUSES = frozenset(
    {READY, READY_WITH_LIMITATIONS, NOT_READY, NOT_AVAILABLE, NOT_EVALUABLE}
)

BLOCKING = "BLOCKING"
LIMITATION = "LIMITATION"

# --- Section 20: readiness axes -------------------------------------------------------------------

SOURCE_PACKAGE = "SOURCE_PACKAGE"
DIGITAL_PROFILE = "DIGITAL_PROFILE"
IDENTITY = "IDENTITY"
DEPTH_REFERENCE = "DEPTH_REFERENCE"
MEASUREMENT_SEMANTICS = "MEASUREMENT_SEMANTICS"
UNITS = "UNITS"
SPATIAL_REFERENCE = "SPATIAL_REFERENCE"
LIQUEFACTION_RESISTANCE_INPUT = "LIQUEFACTION_RESISTANCE_INPUT"
EARTHQUAKE_LOADING_INPUT = "EARTHQUAKE_LOADING_INPUT"
WAVE_LIQUEFACTION_INPUT = "WAVE_LIQUEFACTION_INPUT"

# The first seven axes describe the CPT EVIDENCE itself; the last three describe whether a
# FUTURE liquefaction calculation could be fed -- they are reported separately and never merged.
CPT_PROFILE_AXES = (
    SOURCE_PACKAGE,
    DIGITAL_PROFILE,
    IDENTITY,
    DEPTH_REFERENCE,
    MEASUREMENT_SEMANTICS,
    UNITS,
    SPATIAL_REFERENCE,
)
LIQUEFACTION_INPUT_AXES = (
    LIQUEFACTION_RESISTANCE_INPUT,
    EARTHQUAKE_LOADING_INPUT,
    WAVE_LIQUEFACTION_INPUT,
)
READINESS_AXES = CPT_PROFILE_AXES + LIQUEFACTION_INPUT_AXES

# --- Section 9: package-inventory candidate roles (assigned from inspected bytes only) -----------

ROLE_MACHINE_READABLE_CPT_PROFILE = "MACHINE_READABLE_CPT_PROFILE"
ROLE_CPT_LOCATION_DATA = "CPT_LOCATION_DATA"
ROLE_CPT_LOG_DOCUMENT = "CPT_LOG_DOCUMENT"
ROLE_FACTUAL_REPORT = "FACTUAL_REPORT"
ROLE_METADATA = "METADATA"
ROLE_UNKNOWN = "UNKNOWN"

CANDIDATE_ROLES = frozenset(
    {
        ROLE_MACHINE_READABLE_CPT_PROFILE,
        ROLE_CPT_LOCATION_DATA,
        ROLE_CPT_LOG_DOCUMENT,
        ROLE_FACTUAL_REPORT,
        ROLE_METADATA,
        ROLE_UNKNOWN,
    }
)

# Detected content types (magic-byte sniffing of the real file head, never the extension).
CONTENT_PDF = "application/pdf"
CONTENT_PNG = "image/png"
CONTENT_JPEG = "image/jpeg"
CONTENT_TIFF = "image/tiff"
CONTENT_ZIP = "application/zip"
CONTENT_XML = "application/xml"
CONTENT_TEXT = "text/plain"
CONTENT_EMPTY = "application/x-empty"
CONTENT_BINARY_UNKNOWN = "application/octet-stream"

# Section 10/30: these are DOCUMENTARY. They are never parsed as numeric CPT data -- no OCR, no
# chart digitization, no computer-vision trace reconstruction, no manual transcription.
DOCUMENTARY_CONTENT_TYPES = frozenset({CONTENT_PDF, CONTENT_PNG, CONTENT_JPEG, CONTENT_TIFF})
DOCUMENTARY_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"})

# --- Section 27: scientific roles of MAR-032 products ---------------------------------------------

OFFSHORE_CPT_CPTU_SOURCE_EVIDENCE = "OFFSHORE_CPT_CPTU_SOURCE_EVIDENCE"
MEASURED_CPT_CPTU_PROFILE = "MEASURED_CPT_CPTU_PROFILE"
LIQUEFACTION_INPUT_READINESS_ASSESSMENT = "LIQUEFACTION_INPUT_READINESS_ASSESSMENT"
SOURCE_INTERPRETED_LITERATURE_CONTEXT = "SOURCE_INTERPRETED_LITERATURE_CONTEXT"
SOURCE_DECLARED_COLLECTION_METADATA = "SOURCE_DECLARED_COLLECTION_METADATA"

# Labels that NO MAR-032 product may carry.
PROHIBITED_PRODUCT_LABELS = (
    "LIQUEFACTION_SUSCEPTIBILITY",
    "LIQUEFACTION_HAZARD",
    "LIQUEFACTION_RISK",
)

# --- Section 16: depth reference vocabulary -----------------------------------------------------

DEPTH_BELOW_SEABED = "DEPTH_BELOW_SEABED"
DEPTH_REFERENCE_UNRESOLVED = "DEPTH_REFERENCE_UNRESOLVED"
DEPTH_REFERENCES = frozenset({DEPTH_BELOW_SEABED, DEPTH_REFERENCE_UNRESOLVED})

# --- Section 17: spatial vocabulary --------------------------------------------------------------

SPATIAL_LOCATION_UNRESOLVED = "SPATIAL_LOCATION_UNRESOLVED"
CRS_UNRESOLVED = "CRS_UNRESOLVED"

# --- Section 13: canonical identity and measurement fields ----------------------------------------

SOURCE_ID = "source_id"
TEST_ID = "test_id"
LOCATION_ID = "location_id"
OBSERVATION_INDEX = "observation_index"
DEPTH_SOURCE_VALUE = "depth_source_value"
DEPTH_SOURCE_UNIT = "depth_source_unit"
DEPTH_REFERENCE_FIELD = "depth_reference"
DEPTH_BSF_M = "depth_bsf_m"
QC_MPA = "qc_mpa"
QT_MPA = "qt_mpa"
FS_KPA = "fs_kpa"
U2_KPA = "u2_kpa"

IDENTITY_FIELDS = (SOURCE_ID, TEST_ID, LOCATION_ID, OBSERVATION_INDEX)
CANONICAL_MEASUREMENT_FIELDS = (QC_MPA, QT_MPA, FS_KPA, U2_KPA)
CANONICAL_FIELD_UNITS = {
    DEPTH_BSF_M: "m",
    QC_MPA: "MPa",
    QT_MPA: "MPa",
    FS_KPA: "kPa",
    U2_KPA: "kPa",
}
RAW_CHANNEL_PREFIX = "raw__"

# --- Section 15: deterministic unit conversions (source unit MUST be explicit; never inferred) ---
# Keyed (from_unit, to_unit). Anything absent is "not deterministic here" -> value stays null.

UNIT_CONVERSION_FACTORS: dict[tuple[str, str], float] = {
    ("m", "m"): 1.0,
    ("cm", "m"): 0.01,
    ("mm", "m"): 0.001,
    ("MPa", "MPa"): 1.0,
    ("kPa", "MPa"): 0.001,
    ("Pa", "MPa"): 1.0e-6,
    ("kPa", "kPa"): 1.0,
    ("MPa", "kPa"): 1000.0,
    ("Pa", "kPa"): 0.001,
    ("Pa", "Pa"): 1.0,
    ("kPa", "Pa"): 1000.0,
    ("MPa", "Pa"): 1.0e6,
    ("deg", "deg"): 1.0,
}

# --- Section 6: marine liquefaction is not one mechanism -----------------------------------------

EARTHQUAKE_INDUCED_LIQUEFACTION = "EARTHQUAKE_INDUCED_LIQUEFACTION"
WAVE_CURRENT_INDUCED_SEABED_LIQUEFACTION = "WAVE_CURRENT_INDUCED_SEABED_LIQUEFACTION"
LIQUEFACTION_MECHANISMS = (
    EARTHQUAKE_INDUCED_LIQUEFACTION,
    WAVE_CURRENT_INDUCED_SEABED_LIQUEFACTION,
)

EARTHQUAKE_LIQUEFACTION_TRIGGERING_NOT_EVALUABLE = (
    "EARTHQUAKE_LIQUEFACTION_TRIGGERING_NOT_EVALUABLE"
)
WAVE_INDUCED_LIQUEFACTION_NOT_EVALUABLE = "WAVE_INDUCED_LIQUEFACTION_NOT_EVALUABLE"
CPT_GEOTECHNICAL_PROFILE_NOT_AVAILABLE = "CPT_GEOTECHNICAL_PROFILE_NOT_AVAILABLE"

# Section 21: evidence the FUTURE CPT-based earthquake triggering framework is expected to need.
# Listed as data dependencies only -- no equation from this list is implemented in MAR-032.
EARTHQUAKE_REQUIRED_EVIDENCE = (
    "machine_readable_cpt_profile",
    "depth_below_seabed",
    "qc_or_qt",
    "sleeve_friction_fs",
    "pore_pressure_u2",
    "cone_area_ratio",
    "soil_unit_weight_or_stress_state",
    "vertical_total_stress_basis",
    "vertical_effective_stress_basis",
    "earthquake_pga_a_max",
    "earthquake_magnitude",
    "stress_reduction_factor_method",
)

# Section 22: evidence categories a FUTURE hydro-geotechnical wave-liquefaction model would need.
WAVE_REQUIRED_EVIDENCE = (
    "wave_forcing",
    "water_depth",
    "soil_profile",
    "soil_hydraulic_properties",
    "soil_compressibility_stiffness_properties",
    "initial_effective_stress_state",
    "pore_pressure_response_parameters",
)

AVAILABLE = "AVAILABLE"
NOT_AUTHORIZED = "NOT_AUTHORIZED"

# Section 28: every liquefaction-model output that MAR-032 does NOT compute, stated explicitly.
NOT_COMPUTED_FLAGS: dict[str, bool] = {
    "earthquake_liquefaction_triggering_computed": False,
    "wave_induced_liquefaction_computed": False,
    "WAVE_INDUCED_LIQUEFACTION_MODELLED": False,
    "CRR_computed": False,
    "CSR_computed": False,
    "liquefaction_factor_of_safety_computed": False,
    "liquefaction_probability_computed": False,
    "LPI_computed": False,
    "settlement_computed": False,
    "lateral_spreading_computed": False,
    "pipeline_response_computed": False,
    "qt_unequal_area_correction_applied": False,
    "pore_pressure_correction_applied": False,
}

# Section 19: profile QA reports, never repairs. None of these operations exist in this package.
PROHIBITED_PROFILE_OPERATIONS = (
    "smoothing",
    "interpolation",
    "gap filling",
    "spike removal / despiking",
    "depth resampling",
    "replicate averaging",
    "OCR of PDF or image logs",
    "chart / pixel digitization",
)

# --- Section 5, 6, 24: references recorded as FUTURE-METHOD or LITERATURE CONTEXT only -----------

REFERENCES = (
    {
        "key": "youd_2001",
        "citation": (
            "Youd, T.L. et al. (2001). Liquefaction Resistance of Soils: Summary Report from the "
            "1996 NCEER and 1998 NCEER/NSF Workshops on Evaluation of Liquefaction Resistance of "
            "Soils. Journal of Geotechnical and Geoenvironmental Engineering, 127(10), 817-833."
        ),
        "doi": "10.1061/(ASCE)1090-0241(2001)127:10(817)",
        "role": "FUTURE_METHOD_CONTEXT",
        "use": "simplified liquefaction triggering framework; SPT/CPT/Vs resistance methods; "
        "earthquake magnitude and peak acceleration requirements",
        "implemented_in_mar_032": False,
    },
    {
        "key": "boulanger_idriss_2014",
        "citation": (
            "Boulanger, R.W. and Idriss, I.M. (2014). CPT and SPT Based Liquefaction Triggering "
            "Procedures. Report UCD/CGM-14/01, University of California, Davis."
        ),
        "doi": None,
        "role": "FUTURE_METHOD_CONTEXT",
        "use": "CPT/SPT based liquefaction triggering; CSR/CRR separation; penetration "
        "resistance corrections; stress normalization",
        "implemented_in_mar_032": False,
    },
    {
        "key": "robertson_wride_1998",
        "citation": (
            "Robertson, P.K. and Wride, C.E. (1998). Evaluating cyclic liquefaction potential "
            "using the cone penetration test. Canadian Geotechnical Journal 35, 442-459."
        ),
        "doi": "10.1139/t98-017",
        "role": "FUTURE_METHOD_CONTEXT",
        "use": "historical CPT-based liquefaction-method context",
        "implemented_in_mar_032": False,
    },
    {
        "key": "jeng_2001",
        "citation": (
            "Jeng, D.S. (2001). Mechanism of the wave-induced seabed instability in the vicinity "
            "of a breakwater: a review. Ocean Engineering."
        ),
        "doi": "10.1016/S0029-8018(00)00013-5",
        "role": "FUTURE_METHOD_CONTEXT",
        "use": "wave-induced seabed liquefaction: oscillatory/momentary response and residual "
        "pore-pressure buildup are distinct from earthquake-induced liquefaction",
        "implemented_in_mar_032": False,
    },
    {
        "key": "le_2014",
        "citation": (
            "Le, T.M.H. et al. (2014). Geological and geotechnical characterisation for offshore "
            "wind turbine foundations: A case study of the Sheringham Shoal wind farm. "
            "Engineering Geology 177, 40-53."
        ),
        "doi": "10.1016/j.enggeo.2014.05.005",
        "role": SOURCE_INTERPRETED_LITERATURE_CONTEXT,
        "use": "confirms substantial 2005-2008 field/laboratory geotechnical investigation exists "
        "at Sheringham Shoal; NO table value, mean property or interpreted soil unit from this "
        "paper is copied into, or assigned to, any canonical CPT measurement",
        "implemented_in_mar_032": False,
    },
)

# --- Section 23: the BGS surface-sediment boundary ------------------------------------------------

SURFACE_SEDIMENT_BOUNDARY_STATEMENT = (
    "MAR-008 BGS surface sediment evidence (Folk class, PSA, D50, sand fraction) and MAR-013 "
    "noncohesive mobility are CONTEXT ONLY. A surface sample is not a subsurface geotechnical "
    "profile. They are never converted into LIQUEFIABLE / NON_LIQUEFIABLE, CRR, relative density, "
    "CPT resistance, SPT N-value or a soil state parameter, and no sand-percentage, D50 or "
    "soil-type-only liquefaction verdict exists."
)

MAR_032_SCOPE_STATEMENT = (
    "MAR-032 acquires, inventories and normalizes offshore CPT/CPTU EVIDENCE and reports "
    "liquefaction-INPUT readiness. It computes no CSR, CRR, factor of safety, probability, LPI, "
    "settlement, lateral spreading, post-liquefaction strength, pipeline flotation or wave-induced "
    "pore pressure. Earthquake-induced and wave/current-induced seabed liquefaction are separate "
    "mechanisms and neither is modelled."
)

"""Canonical vocabulary for MAR-033 CPT-based earthquake liquefaction triggering screening.

Method authority (Sections 2, 46):

    Boulanger, R.W. and Idriss, I.M. (2014). CPT and SPT Based Liquefaction Triggering
    Procedures. Report UCD/CGM-14/01, University of California, Davis.

This ticket implements exactly ONE mechanism: deterministic CPT-based earthquake liquefaction
TRIGGERING screening for cohesionless soil under free-field conditions (Section 3). It never
computes wave-induced liquefaction, LPI, settlement, lateral spreading, runout, pipeline damage,
probabilistic hazard, site response, or cyclic softening of cohesive soil -- those remain
separate, unimplemented mechanisms (`geotechnical.cpt_readiness` still reports them
NOT_EVALUABLE for its own, broader evidence-readiness purpose).

Every status, mode and reason used by the `liquefaction` package is named here once. Nothing
here is a fabricated 0-100 score, risk percentage or safety class (Section 14.1 / Section 4):
`FS_liq` is a model factor-of-safety NUMBER, and its only allowed categorical companions are the
five explicit `EVALUATION_STATES` below.
"""

from __future__ import annotations

METHOD_ID = "BOULANGER_IDRISS_2014_DETERMINISTIC_CPT"
METHOD_AUTHORITY = "UCD/CGM-14/01"
METHOD_CITATION = (
    "Boulanger, R.W. and Idriss, I.M. (2014). CPT and SPT Based Liquefaction Triggering "
    "Procedures. Report UCD/CGM-14/01, University of California, Davis."
)
METHOD_SCOPE_STATEMENT = (
    "Deterministic CPT-based earthquake liquefaction TRIGGERING screening for cohesionless "
    "soil under free-field / nearly-level conditions (Boulanger & Idriss 2014). Does not "
    "compute wave-induced liquefaction, residual pore-pressure generation, LPI, settlement, "
    "lateral spreading, post-liquefaction deformation, runout, pipeline damage, probabilistic "
    "liquefaction hazard, site-response analysis, PSHA, soil-structure interaction or cyclic "
    "softening of cohesive soil."
)

# --- Section 4: model output states -- never a safety class, never a probability -----------------

MODEL_FS_BELOW_1 = "MODEL_FS_BELOW_1"
MODEL_FS_AT_1 = "MODEL_FS_AT_1"
MODEL_FS_ABOVE_1 = "MODEL_FS_ABOVE_1"
NOT_EVALUABLE = "NOT_EVALUABLE"
OUTSIDE_METHOD_SUPPORT = "OUTSIDE_METHOD_SUPPORT"

EVALUATION_STATES = frozenset(
    {MODEL_FS_BELOW_1, MODEL_FS_AT_1, MODEL_FS_ABOVE_1, NOT_EVALUABLE, OUTSIDE_METHOD_SUPPORT}
)

PROHIBITED_OUTPUT_LABELS = (
    "SAFE",
    "UNSAFE",
    "LIQUEFIED",
    "NOT_LIQUEFIED",
    "LOW RISK",
    "MEDIUM RISK",
    "HIGH RISK",
    "PROBABILITY OF LIQUEFACTION",
)

# --- Section 5: corrected tip-resistance input modes ----------------------------------------------

QT_MEASURED_OR_SOURCE_CORRECTED = "QT_MEASURED_OR_SOURCE_CORRECTED"
QC_PLUS_U2_AND_DECLARED_AREA_RATIO = "QC_PLUS_U2_AND_DECLARED_AREA_RATIO"
QC_EXPLICITLY_DECLARED_AREA_CORRECTED = "QC_EXPLICITLY_DECLARED_AREA_CORRECTED"

TIP_RESISTANCE_MODES = frozenset(
    {
        QT_MEASURED_OR_SOURCE_CORRECTED,
        QC_PLUS_U2_AND_DECLARED_AREA_RATIO,
        QC_EXPLICITLY_DECLARED_AREA_CORRECTED,
    }
)

CORRECTED_TIP_RESISTANCE_NOT_ESTABLISHED = "CORRECTED_TIP_RESISTANCE_NOT_ESTABLISHED"

# --- Section 6: seismic scenario ------------------------------------------------------------------

FREE_FIELD_SEABED_SURFACE_PGA = "FREE_FIELD_SEABED_SURFACE_PGA"
PGA_REFERENCES = frozenset({FREE_FIELD_SEABED_SURFACE_PGA})
MISSING_EARTHQUAKE_SCENARIO = "MISSING_EARTHQUAKE_SCENARIO"

# --- Section 8: stress-reduction (rd) deep-extrapolation limitation -------------------------------

RD_DEEP_EXTRAPOLATION_LIMITATION = "RD_DEEP_EXTRAPOLATION_LIMITATION"
RD_DEEP_EXTRAPOLATION_DEPTH_M = 10.0

# --- Section 9: vertical stress modes -------------------------------------------------------------

EXPLICIT_STRESS_PROFILE = "EXPLICIT_STRESS_PROFILE"
SEABED_RELATIVE_LAYERED_STRESS = "SEABED_RELATIVE_LAYERED_STRESS"
STRESS_MODES = frozenset({EXPLICIT_STRESS_PROFILE, SEABED_RELATIVE_LAYERED_STRESS})

VERTICAL_STRESS_PROFILE_NOT_AVAILABLE = "VERTICAL_STRESS_PROFILE_NOT_AVAILABLE"
STRESS_LAYER_MODEL_INVALID = "STRESS_LAYER_MODEL_INVALID"
MISSING_STRESS_AT_DEPTH = "MISSING_STRESS_AT_DEPTH"
NONPOSITIVE_EFFECTIVE_STRESS = "NONPOSITIVE_EFFECTIVE_STRESS"

# --- Section 10: atmospheric reference pressure ---------------------------------------------------

ATMOSPHERIC_PRESSURE_KPA = 101.3

# --- Section 11: overburden normalization iteration -----------------------------------------------

CN_MAX = 1.7
QC1NCS_M_BOUND_LOW = 21.0
QC1NCS_M_BOUND_HIGH = 254.0
CN_ITERATION_NOT_CONVERGED = "CN_ITERATION_NOT_CONVERGED"
CN_DEFAULT_TOLERANCE = 1.0e-6
CN_DEFAULT_MAX_ITERATIONS = 50

# --- Section 12: fines-content sources ------------------------------------------------------------

MEASURED_LAB_FC = "MEASURED_LAB_FC"
SOURCE_DECLARED_FC = "SOURCE_DECLARED_FC"
USER_DECLARED_FC_SCENARIO = "USER_DECLARED_FC_SCENARIO"
CPT_ESTIMATED_FC_GENERAL_CORRELATION = "CPT_ESTIMATED_FC_GENERAL_CORRELATION"

MISSING_FINES_OR_APPLICABILITY = "MISSING_FINES_OR_APPLICABILITY"

# --- Section 14: CPT-estimated fines (Robertson 2009 Ic / n, as adopted by Boulanger & Idriss
# 2014 for their Ic-based FC correlation) --------------------------------------------------------

IC_N_EXPONENT_MIN = 0.5
IC_N_EXPONENT_MAX = 1.0
IC_N_DEFAULT_TOLERANCE = 1.0e-6
IC_N_DEFAULT_MAX_ITERATIONS = 50
IC_ITERATION_NOT_CONVERGED = "IC_ITERATION_NOT_CONVERGED"
IC_NOT_EVALUABLE_NONPOSITIVE_TERM = "IC_NOT_EVALUABLE_NONPOSITIVE_TERM"

# GENERAL_CORRELATION_SENSITIVITY (MAR-033A Part A item 6) is deliberately a FIFTH fines source,
# semantically distinct from CPT_ESTIMATED_FC_GENERAL_CORRELATION: the latter is the single-C_FC
# "expert mode" (one explicit, user-chosen C_FC -- MAR-033A Part A item 7); the former is the
# automatic mode that fans out into all three literature-defined C_FC variants below and must
# never be reduced to one value, averaged, or turned into a probability by this engine.
GENERAL_CORRELATION_SENSITIVITY = "GENERAL_CORRELATION_SENSITIVITY"
GENERAL_CORRELATION_C_FC_VALUES: tuple[float, ...] = (-0.29, 0.0, 0.29)
FC_BOUND_LOW = 0.0
FC_BOUND_HIGH = 100.0

FINES_CONTENT_SOURCES = frozenset(
    {
        MEASURED_LAB_FC,
        SOURCE_DECLARED_FC,
        USER_DECLARED_FC_SCENARIO,
        CPT_ESTIMATED_FC_GENERAL_CORRELATION,
        GENERAL_CORRELATION_SENSITIVITY,
    }
)

# --- Section 15: cohesionless-soil applicability --------------------------------------------------

COHESIONLESS_SOIL_APPLICABILITY_NOT_ESTABLISHED = "COHESIONLESS_SOIL_APPLICABILITY_NOT_ESTABLISHED"
SOURCE_ESTABLISHED = "SOURCE_ESTABLISHED"
CPT_IC_SCREEN = "CPT_IC_SCREEN"
SOIL_APPLICABILITY_BASES = frozenset({SOURCE_ESTABLISHED, CPT_IC_SCREEN})

# --- Section 17: overburden CRR correction (Ksigma) -----------------------------------------------

QC1NCS_KSIGMA_BOUND_HIGH = 211.0
CSIGMA_MAX = 0.3
KSIGMA_MAX = 1.1

# --- Section 18: magnitude scaling factor ---------------------------------------------------------

MSFMAX_MAX = 2.2

# --- Section 19: static shear / model scope -------------------------------------------------------

STATIC_SHEAR_OUTSIDE_MODEL_SCOPE = "STATIC_SHEAR_OUTSIDE_MODEL_SCOPE"

# --- Section 20: factor-of-safety guards ----------------------------------------------------------

NONPOSITIVE_CSR = "NONPOSITIVE_CSR"
# Floating-point equality tolerance for "FS_liq is exactly at the model boundary". A numerical
# tolerance for round-off, not a physical band around FS = 1 (mirrors slope_stability.core).
FS_AT_1_RELATIVE_TOLERANCE = 1.0e-9
# A row whose declared inputs all passed every named gate above can still produce a non-finite
# intermediate/final term (e.g. CRR overflowing to +inf from a qc1Ncs far outside the method's
# calibrated range, typically from a near-zero effective stress a few millimetres below the
# seabed). Reported explicitly rather than silently landing on NOT_EVALUABLE with no stated
# reason.
NONFINITE_MODEL_RESULT = "NONFINITE_MODEL_RESULT"

# --- Section 21/22: product roles -----------------------------------------------------------------

EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING_SCREENING_PROFILE = (
    "EARTHQUAKE_CPT_LIQUEFACTION_TRIGGERING_SCREENING_PROFILE"
)
POINT_ANALYSIS = "POINT_ANALYSIS"

# --- Section 40: extended readiness axes (distinct from
# geotechnical.cpt_contract.READINESS_AXES, which describe the CPT EVIDENCE itself; these
# describe whether the earthquake-triggering CAPABILITY specifically can run) ---------------------

READY = "READY"
READY_WITH_LIMITATIONS = "READY_WITH_LIMITATIONS"
NOT_READY = "NOT_READY"
NOT_AVAILABLE = "NOT_AVAILABLE"

READINESS_STATUSES = frozenset(
    {READY, READY_WITH_LIMITATIONS, NOT_READY, NOT_AVAILABLE, NOT_EVALUABLE}
)

BLOCKING = "BLOCKING"
LIMITATION = "LIMITATION"

AXIS_CPT_IDENTITY = "CPT_IDENTITY"
AXIS_CORRECTED_TIP_RESISTANCE = "CORRECTED_TIP_RESISTANCE"
AXIS_DEPTH_REFERENCE = "DEPTH_REFERENCE"
AXIS_STRESS_PROFILE = "STRESS_PROFILE"
AXIS_FINES_OR_SOIL_APPLICABILITY = "FINES_OR_SOIL_APPLICABILITY"
AXIS_EARTHQUAKE_MAGNITUDE = "EARTHQUAKE_MAGNITUDE"
AXIS_SEABED_PGA = "SEABED_PGA"
AXIS_METHOD_DOMAIN = "METHOD_DOMAIN"
AXIS_EARTHQUAKE_TRIGGERING = "EARTHQUAKE_TRIGGERING"
AXIS_WAVE_LIQUEFACTION = "WAVE_LIQUEFACTION"

READINESS_AXES = (
    AXIS_CPT_IDENTITY,
    AXIS_CORRECTED_TIP_RESISTANCE,
    AXIS_DEPTH_REFERENCE,
    AXIS_STRESS_PROFILE,
    AXIS_FINES_OR_SOIL_APPLICABILITY,
    AXIS_EARTHQUAKE_MAGNITUDE,
    AXIS_SEABED_PGA,
    AXIS_METHOD_DOMAIN,
)

# --- Section 45: PL854 has no CPT evidence at all -------------------------------------------------

CPT_GEOTECHNICAL_PROFILE_NOT_AVAILABLE = "CPT_GEOTECHNICAL_PROFILE_NOT_AVAILABLE"

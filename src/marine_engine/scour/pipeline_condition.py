"""Official 2018 PL854/PL855 pipeline condition benchmark (MAR-014, Sections 28-30).

Public, official, ROUTE/CORRIDOR-LEVEL context only -- never spatially
distributed to PL854 segments and never used to train/calibrate any
segment-level MAR-014 result.

Source
------
Ithaca Energy (UK) Limited. Anglia Decommissioning Environmental
Appraisal. April 2020. Official UK Government publication, page 29,
Tables 3.4 and 3.5. Benchmark scope is the PL854/PL855 export/methanol
line -- NOT PL854 alone.

Internal source inconsistency, preserved verbatim (Section 29)
--------------------------------------------------------------------
The Environmental Appraisal's narrative prose around Table 3.4 states
that the identified free spans were no more than 10 m in length, while
Table 3.4 itself reports a maximum length of 23.2 m for the PL854/PL855
line. This module never silently "corrects" either source statement --
both are preserved, with the table-specific value retained as the
structured benchmark field.

No spatial distribution (Section 30)
------------------------------------------
Originally (MAR-014) neither free-span nor exposed-section KP positions
were available in canonical machine-readable form, so this benchmark was
never used as spatial validation for any MAR-014 segment-level result.

MAR-014A update -- freespan positions recovered, exposed sections still not
-------------------------------------------------------------------------------
A separate, later official source (Ithaca Energy "Pipelines and Umbilical
Comparative Assessment", April 2020, Appendix B Table B.1) tabulates
explicit 2018 free-span KP/Easting/Northing positions for the PL854/PL855
corridor -- see MAR-014A's `anglia_freespan_spatial_evidence` outputs.
`comparative_assessment_freespan_locations_tabulated` /
`2018_freespan_spatial_evidence_available` record this. This is still
NEVER used as a validation/accuracy metric for MAR-014 segment-level
results (MAR-014A Section 17 attaches model context purely for side-by-side
human review -- no score, probability, or rank). Exposed-section KP
positions remain unavailable in canonical machine-readable form
(`environmental_appraisal_exposure_section_locations_available = False`,
`2018_exposed_section_spatial_evidence_available = False`), and individual
PL854-vs-PL855 freespan attribution remains unresolved
(`individual_PL854_vs_PL855_freespan_attribution_available = False`). This
benchmark may still only support the qualitative statements that (a)
exposure/free spans existed historically on the corridor, (b) most of the
corridor was reported buried >=0.6 m, and (c) local exceptions existed.
"""

import json
from pathlib import Path
from typing import Any

EVIDENCE_ROLE = "HISTORICAL_AGGREGATE_CONDITION_BENCHMARK"

BENCHMARK_SCOPE = "PL854/PL855 export / methanol line"

SOURCE_INTERNAL_CONSISTENCY_NOTE = (
    "Narrative prose and Table 3.4 disagree on maximum freespan length; "
    "table-specific PL854/PL855 value 23.2 m retained as the structured "
    "benchmark value."
)

MAR014_0_15D_MM = 45.72  # PL854 0.15*D, D=0.3048 m
MAJORITY_BURIAL_APPROX_PIPE_DIAMETERS = 0.6 / 0.3048

BENCHMARK_INTERPRETATION_STATEMENT = (
    "MAR-014 IS MOST RELEVANT TO EXPOSED OR SHALLOWLY EMBEDDED LOCAL SECTIONS; IT IS NOT "
    "A MODEL OF THE DEEPLY BURIED MAJORITY OF THE 2018 CORRIDOR."
)


def build_2018_condition_benchmark() -> dict[str, Any]:
    """The fixed, official 2018 PL854/PL855 corridor condition benchmark record.

    Purely static reference data -- no computation, no PL854-segment
    inference, no KP assignment.
    """

    return {
        "evidence_role": EVIDENCE_ROLE,
        "benchmark_scope": BENCHMARK_SCOPE,
        "source": {
            "publisher": "Ithaca Energy (UK) Limited",
            "title": "Anglia Decommissioning Environmental Appraisal",
            "date": "April 2020",
            "classification": "official UK Government publication",
            "page": 29,
            "tables": ["3.4", "3.5"],
        },
        "survey_year": 2018,
        "reported_line_length_m": 24000,
        "majority_buried_at_least_m": 0.6,
        "majority_burial_statement_is_qualitative": True,
        "free_span_count": 8,
        "total_free_span_length_m": 97,
        "max_free_span_height_m": 0.4,
        "max_free_span_length_m": 23.2,
        "exposed_section_count": 19,
        "total_exposed_length_m": 519,
        "longest_exposed_section_m": 87,
        "environmental_appraisal_exposure_section_locations_available": False,
        "comparative_assessment_freespan_locations_tabulated": True,
        "comparative_assessment_freespan_table": "Appendix B Table B.1",
        "2018_freespan_spatial_evidence_available": True,
        "2018_exposed_section_spatial_evidence_available": False,
        "individual_PL854_vs_PL855_freespan_attribution_available": False,
        "source_internal_consistency_warning": True,
        "source_internal_consistency_note": SOURCE_INTERNAL_CONSISTENCY_NOTE,
        "interpretation": {
            "usable_statements": [
                "Actual exposure/free spans existed historically on the corridor.",
                "Most of the corridor was reported buried >= 0.6 m.",
                "Local exceptions to majority burial existed.",
                "As of MAR-014A, 2018 free-span KP/Easting/Northing positions are "
                "available (Ithaca Energy Comparative Assessment, April 2020, Appendix B "
                "Table B.1) -- see the anglia_freespan_spatial_evidence outputs.",
            ],
            "not_usable_as": [
                "Spatial validation of any MAR-014 segment-level result: free-span "
                "positions are now available (MAR-014A) but are deliberately NOT used as "
                "a validation/accuracy metric (MAR-014A Section 17 attaches model context "
                "for side-by-side human review only -- no score, probability, or rank). "
                "Exposed-section KP positions remain unavailable in canonical "
                "machine-readable form.",
                "Training or calibration data for MAR-014 segment-level results.",
            ],
            "mar014_0_15d_mm": MAR014_0_15D_MM,
            "majority_burial_approx_pipe_diameters": MAJORITY_BURIAL_APPROX_PIPE_DIAMETERS,
            "statement": BENCHMARK_INTERPRETATION_STATEMENT,
        },
        "historical_condition_used_for_spatial_calibration": False,
    }


def write_2018_condition_benchmark(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_2018_condition_benchmark(), indent=2, default=str), encoding="utf-8"
    )
    return output_path

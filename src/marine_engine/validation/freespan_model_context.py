"""2018 freespan event vs. existing model-output context join (MAR-014A Section 17).

No score, probability, rank, or accuracy metric (Section 17)
--------------------------------------------------------------------
This module juxtaposes each 2018 official freespan event (MAR-014A) against
the model outputs already computed at that location by MAR-010/011A (via
MAR-012's combined bed shear), MAR-013 (noncohesive mobility capacity), and
MAR-014 (scour-onset embedment screening, which already carries MAR-007
morphology as legacy context) -- side by side, for a human reader to look
at. It never fuses these into a composite score, a probability, a rank, or
any form of predictive-accuracy metric: an event's mere presence in Table
B.1 is not a labelled outcome this model was fitted or validated against,
and no such claim is made anywhere in this module's output.

Containing-segment lookup, not the finest possible resolution (Section 17)
--------------------------------------------------------------------------------
Each event is attached to the ONE segment (from each of the three source
segment tables, independently) whose own `[start_chainage_m,
end_chainage_m)` contains the event's `canonical_mid_chainage_m` -- never
distributed across multiple segments, so the output is exactly one row per
event. The three source segment tables are read independently (never
assumed to share identical chainage boundaries), each contributing its own
namespaced columns.
"""

from typing import Any

import pandas as pd

MODEL_CONTEXT_ROLE = "INDEPENDENT_MODEL_OUTPUTS_JUXTAPOSED_NO_SCORE_NO_VALIDATION_METRIC"

FREESPAN_MODEL_CONTEXT_COLUMNS = (
    "event_id",
    "survey_year",
    "canonical_mid_chainage_m",
    "canonical_mid_kp",
    "hydro_pair_id",
    "mar012_tau_max_p95_sensitivity_min_pa",
    "mar012_tau_max_p95_sensitivity_max_pa",
    "mar012_tau_max_p95_sensitivity_width_pa",
    "mar013_largest_tested_d50_with_p95_mobility_ratio_ge_1_mm",
    "mar013_largest_tested_d50_with_any_exceedance_mm",
    "mar013_mapped_250k_folk_class",
    "mar013_nearest_valid_psa_d50_mm",
    "mar014_p95_required_embedment_lower_class",
    "mar014_p95_required_embedment_upper_class",
    "mar014_pipe_diameter_source_envelope_status",
    "mar007_slope_500m_median_deg",
    "mar007_tpi_1000m_median_m",
    "mar007_local_relief_1000m_median_m",
    "model_context_role",
)


def _find_containing_segment(chainage_m: float, segments_df: pd.DataFrame) -> pd.Series | None:
    """The one segment whose `[start_chainage_m, end_chainage_m)` contains `chainage_m`.

    Falls back to the last segment when `chainage_m` lands exactly on (or
    numerically past) the final segment's own end -- the same route-terminus
    edge case `shapely` handles by clamping (never dropping the event).
    """

    if segments_df.empty:
        return None
    within = segments_df[
        (segments_df["start_chainage_m"] <= chainage_m)
        & (chainage_m < segments_df["end_chainage_m"])
    ]
    if not within.empty:
        return within.iloc[0]
    return segments_df.sort_values("end_chainage_m").iloc[-1]


def attach_model_context_to_events(
    events_2018_df: pd.DataFrame,
    *,
    combined_bed_shear_segments_df: pd.DataFrame,
    noncohesive_mobility_segments_df: pd.DataFrame,
    scour_onset_segments_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per 2018 event, with each source's own segment context alongside it.

    Every attached field keeps its own `mar0XX_` prefix -- a reader must
    never be able to mistake this table for a single fused model output.
    """

    records = []
    for _, event in events_2018_df.iterrows():
        mid_chainage_m = event["canonical_mid_chainage_m"]

        bed_shear_segment = _find_containing_segment(mid_chainage_m, combined_bed_shear_segments_df)
        mobility_segment = _find_containing_segment(
            mid_chainage_m, noncohesive_mobility_segments_df
        )
        scour_segment = _find_containing_segment(mid_chainage_m, scour_onset_segments_df)

        hydro_pair_id = None
        for segment in (scour_segment, mobility_segment, bed_shear_segment):
            if segment is not None and pd.notna(segment.get("hydro_pair_id")):
                hydro_pair_id = segment.get("hydro_pair_id")
                break

        records.append(
            {
                "event_id": event["event_id"],
                "survey_year": event["survey_year"],
                "canonical_mid_chainage_m": mid_chainage_m,
                "canonical_mid_kp": event["canonical_mid_kp"],
                "hydro_pair_id": hydro_pair_id,
                "mar012_tau_max_p95_sensitivity_min_pa": (
                    bed_shear_segment.get("tau_max_p95_sensitivity_min_pa")
                    if bed_shear_segment is not None
                    else None
                ),
                "mar012_tau_max_p95_sensitivity_max_pa": (
                    bed_shear_segment.get("tau_max_p95_sensitivity_max_pa")
                    if bed_shear_segment is not None
                    else None
                ),
                "mar012_tau_max_p95_sensitivity_width_pa": (
                    bed_shear_segment.get("tau_max_p95_sensitivity_width_pa")
                    if bed_shear_segment is not None
                    else None
                ),
                "mar013_largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": (
                    mobility_segment.get("largest_tested_d50_with_p95_mobility_ratio_ge_1_mm")
                    if mobility_segment is not None
                    else None
                ),
                "mar013_largest_tested_d50_with_any_exceedance_mm": (
                    mobility_segment.get("largest_tested_d50_with_any_exceedance_mm")
                    if mobility_segment is not None
                    else None
                ),
                "mar013_mapped_250k_folk_class": (
                    mobility_segment.get("mapped_250k_folk_class")
                    if mobility_segment is not None
                    else None
                ),
                "mar013_nearest_valid_psa_d50_mm": (
                    mobility_segment.get("nearest_valid_psa_d50_mm")
                    if mobility_segment is not None
                    else None
                ),
                "mar014_p95_required_embedment_lower_class": (
                    scour_segment.get("p95_required_embedment_lower_class")
                    if scour_segment is not None
                    else None
                ),
                "mar014_p95_required_embedment_upper_class": (
                    scour_segment.get("p95_required_embedment_upper_class")
                    if scour_segment is not None
                    else None
                ),
                "mar014_pipe_diameter_source_envelope_status": (
                    scour_segment.get("pipe_diameter_source_envelope_status")
                    if scour_segment is not None
                    else None
                ),
                "mar007_slope_500m_median_deg": (
                    scour_segment.get("slope_500m_median_deg")
                    if scour_segment is not None
                    else None
                ),
                "mar007_tpi_1000m_median_m": (
                    scour_segment.get("tpi_1000m_median_m") if scour_segment is not None else None
                ),
                "mar007_local_relief_1000m_median_m": (
                    scour_segment.get("local_relief_1000m_median_m")
                    if scour_segment is not None
                    else None
                ),
                "model_context_role": MODEL_CONTEXT_ROLE,
            }
        )

    return pd.DataFrame(records, columns=list(FREESPAN_MODEL_CONTEXT_COLUMNS))


def print_freespan_model_context_report(context_df: pd.DataFrame, *, file: Any = None) -> None:
    import sys

    file = file or sys.stdout
    lines = [
        "=== PL854 2018 Freespan Event vs. Model-Output Context (MAR-014A Section 17) ===",
        "",
        f"Events: {len(context_df)}",
        "",
    ]
    for _, row in context_df.iterrows():
        lines.append(f"  {row['event_id']} ({row['canonical_mid_kp']}):")
        lines.append(
            "    MAR-012 tau_max p95 sensitivity (Pa): "
            f"{row['mar012_tau_max_p95_sensitivity_min_pa']:.3f}-"
            f"{row['mar012_tau_max_p95_sensitivity_max_pa']:.3f}"
            if pd.notna(row["mar012_tau_max_p95_sensitivity_min_pa"])
            else "    MAR-012 tau_max p95 sensitivity (Pa): n/a"
        )
        lines.append(
            "    MAR-013 largest passing D50 (p95, mm): "
            f"{row['mar013_largest_tested_d50_with_p95_mobility_ratio_ge_1_mm']}"
        )
        lines.append(
            "    MAR-014 p95 required embedment class: "
            f"{row['mar014_p95_required_embedment_lower_class']}-"
            f"{row['mar014_p95_required_embedment_upper_class']}"
        )
    lines.append("")
    lines.append(f"Model context role: {MODEL_CONTEXT_ROLE}")
    lines.append(
        "NO SCORE, PROBABILITY, RANK, OR ACCURACY METRIC HAS BEEN COMPUTED. Values above are "
        "independent model outputs placed alongside each observed event for human review only."
    )
    print("\n".join(lines), file=file)

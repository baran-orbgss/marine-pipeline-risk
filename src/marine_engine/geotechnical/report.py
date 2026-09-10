"""Plain-text summary of a CPT evidence build (MAR-032 Sections 29, 41)."""

from __future__ import annotations

from marine_engine.geotechnical import cpt_contract as contract
from marine_engine.geotechnical.evidence_build import CptEvidenceBuildResult

__all__ = ["format_summary_lines", "format_acceptance_lines"]


def format_summary_lines(result: CptEvidenceBuildResult) -> list[str]:
    facts = result.facts
    qa = result.qa
    lines = [f"CPT evidence build: {result.evidence_id}", "", "Source packages:"]
    for acq in result.acquisitions:
        cached = "cache reuse" if acq["already_cached"] else "acquired"
        lines.append(
            f"  [{acq['package_id']}] {acq['package_url']} -> {acq['package_bytes']} B, "
            f"SHA-256 {acq['package_sha256'][:16]}... ({cached})"
        )
    inventory = result.inventory
    lines.append("")
    lines.append(
        f"Inventory: {len(inventory)} files; machine-readable CPT profiles: "
        f"{int((inventory['candidate_role'] == contract.ROLE_MACHINE_READABLE_CPT_PROFILE).sum())}"
        f"; documentary files: {int(inventory['is_documentary'].sum())}"
    )
    for content_type, count in sorted(inventory["detected_content_type"].value_counts().items()):
        lines.append(f"  {content_type}: {count}")
    lines.append("")
    if facts.canonical_profile_created:
        lines.append(
            f"Canonical numeric profile: {qa['row_count']} rows, {qa['test_count']} tests "
            f"(source-declared {facts.declared_test_count}); channels {qa['channels_present']}"
        )
        lines.append(
            f"  depth reference: {facts.depth_reference}; negative-depth rows: "
            f"{qa['negative_depth_row_count']}; depth-order violations: "
            f"{qa['depth_order_violation_count']}; duplicate identities: "
            f"{qa['duplicate_observation_identity_count']}"
        )
        lines.append(
            f"  coordinates: {'present' if facts.coordinates_available else 'absent'}; CRS "
            f"{'resolved' if facts.crs_resolved else 'unresolved'}"
            + (f"; CONFLICT: {facts.crs_conflict}" if facts.crs_conflict else "")
        )
    else:
        lines.append("Canonical numeric profile: NOT CREATED (no machine-readable numeric source)")
    lines.append("")
    lines.append(f"CPT evidence readiness: {result.cpt_readiness.cpt_profile_status}")
    for axis in result.cpt_readiness.axes:
        lines.append(f"  {axis.axis}: {axis.status}")
        for reason in axis.blocking_reasons:
            lines.append(f"    BLOCKING: {reason}")
        for reason in axis.limitation_reasons:
            lines.append(f"    LIMITATION: {reason}")
    liq = result.liquefaction_readiness
    lines.append("")
    lines.append(
        f"Earthquake-induced liquefaction triggering: {liq['earthquake_induced']['status_code']}"
    )
    lines.append(
        "  missing/unauthorized: " + ", ".join(liq["earthquake_induced"]["missing_or_unauthorized"])
    )
    lines.append(
        f"Wave/current-induced seabed liquefaction: {liq['wave_current_induced']['status_code']}"
    )
    lines.append("  missing: " + ", ".join(liq["wave_current_induced"]["missing"]))
    lines.append("")
    lines.append("Outputs:")
    for name, path in result.outputs.items():
        lines.append(f"  {name}: {path}")
    return lines


def format_acceptance_lines(result: CptEvidenceBuildResult) -> list[str]:
    facts = result.facts
    yes_no = lambda flag: "YES" if flag else "NO"  # noqa: E731
    return [
        "DOES CPT FILE EXISTENCE ALONE IMPLY LIQUEFACTION READINESS? NO",
        f"WAS THE OFFICIAL MDE CPT/CPTU SOURCE RESOLVED? {yes_no(facts.source_package_resolved)}",
        f"WAS THE ACQUIRED PACKAGE CHECKSUM-RECORDED? {yes_no(facts.source_checksum_recorded)}",
        "DOES THE PACKAGE CONTAIN MACHINE-READABLE NUMERIC CPT/CPTU PROFILES? "
        f"{yes_no(facts.machine_readable_profile_available)}",
        "WERE PDF/IMAGE LOGS OCR-DIGITIZED? NO",
        "CAN QC SILENTLY BECOME QT? NO",
        "ARE UNKNOWN CPT UNITS GUESSED FROM MAGNITUDE? NO",
        "CAN BGS SURFACE SEDIMENT CLASS PRODUCE A LIQUEFACTION VERDICT? NO",
        "IS EARTHQUAKE CSR COMPUTED IN MAR-032? NO",
        "IS CPT-BASED CRR COMPUTED IN MAR-032? NO",
        "IS LIQUEFACTION FACTOR OF SAFETY COMPUTED? NO",
        "IS WAVE-INDUCED EXCESS PORE PRESSURE COMPUTED? NO",
        "ARE EARTHQUAKE- AND WAVE-INDUCED LIQUEFACTION KEPT SEPARATE? YES",
        "CAN CURRENT PL854 DATA SUPPORT A SITE-SPECIFIC LIQUEFACTION CLAIM? NO",
    ]

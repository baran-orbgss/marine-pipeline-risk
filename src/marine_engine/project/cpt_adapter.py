"""Generic project-registered `CPT` asset inspection (MAR-032 Section 25).

Does NOT reimplement CPT readiness: `inspect_cpt_asset` only inspects the registered file's real
bytes and assembles `marine_engine.geotechnical.cpt_readiness.CptEvidenceFacts`; the registry
passes those facts to the existing, unmodified `assess_cpt_readiness` (imported there directly).

Three honest outcomes for a registered path:

* documentary content (PDF / PNG / JPEG / TIFF by magic bytes) -> facts say "documentary only,
  no machine-readable profile" -> DIGITAL_PROFILE is BLOCKING -> never READY;
* a canonical CPT measurements Parquet (the `MEASURED_CPT_CPTU_PROFILE` product of a
  source-specific provider build, recognised by its canonical columns) -> facts from its QA;
* anything else -> "no generic reader recognises this format" -> never READY. A raw operator CSV
  is not parsed here: its column semantics and units require an explicit, source-specific
  declaration (see `providers.geotechnical`), never a guess from column names.

Evidence roles are untouched: whatever role the manifest declared stays declared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from marine_engine.geotechnical import cpt_contract as contract
from marine_engine.geotechnical import cpt_inventory, cpt_profile
from marine_engine.geotechnical.cpt_readiness import CptEvidenceFacts

__all__ = ["CptAssetLoadError", "CANONICAL_REQUIRED_COLUMNS", "inspect_cpt_asset"]

CANONICAL_REQUIRED_COLUMNS = (
    contract.SOURCE_ID,
    contract.TEST_ID,
    contract.OBSERVATION_INDEX,
    contract.DEPTH_SOURCE_VALUE,
    contract.DEPTH_REFERENCE_FIELD,
    contract.DEPTH_BSF_M,
)


class CptAssetLoadError(RuntimeError):
    """The registered file cannot be read at all -- a registration-level failure."""


def inspect_cpt_asset(path: Path) -> tuple[CptEvidenceFacts, dict[str, Any]]:
    if not path.is_file():
        raise CptAssetLoadError(f"CPT asset not found: {path}")
    head = path.read_bytes()[:4096]
    content_type = cpt_inventory.sniff_content_type(head)
    observed: dict[str, Any] = {"detected_content_type": content_type}

    if content_type in contract.DOCUMENTARY_CONTENT_TYPES:
        observed["is_documentary"] = True
        facts = CptEvidenceFacts(
            source_package_resolved=True,
            source_checksum_recorded=True,
            documentary_evidence_available=True,
            machine_readable_profile_available=False,
            notes=("documentary CPT evidence; not digitized (no OCR, no chart digitization)",),
        )
        return facts, observed

    observed["is_documentary"] = False
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 -- any reader failure is a load failure
            raise CptAssetLoadError(f"CPT parquet could not be read: {path} ({exc})") from exc
        missing = [c for c in CANONICAL_REQUIRED_COLUMNS if c not in df.columns]
        observed["columns"] = list(df.columns)
        observed["record_count"] = int(len(df))
        if missing:
            observed["canonical_columns_missing"] = missing
            facts = CptEvidenceFacts(
                source_package_resolved=True,
                source_checksum_recorded=True,
                documentary_evidence_available=False,
                machine_readable_profile_available=False,
                notes=(
                    "parquet is not a canonical CPT measurements table (missing "
                    f"{missing}); no semantics are inferred from its columns",
                ),
            )
            return facts, observed
        qa = cpt_profile.compute_profile_qa(df)
        references = df[contract.DEPTH_REFERENCE_FIELD].dropna().unique().tolist()
        depth_reference = (
            references[0]
            if len(references) == 1 and references[0] in contract.DEPTH_REFERENCES
            else contract.DEPTH_REFERENCE_UNRESOLVED
        )
        observed.update(
            {
                "test_count": qa["test_count"],
                "channels_present": qa["channels_present"],
                "depth_reference_values": references,
                "duplicate_observation_identity_count": qa["duplicate_observation_identity_count"],
                "depth_order_violation_count": qa["depth_order_violation_count"],
            }
        )
        facts = CptEvidenceFacts(
            source_package_resolved=True,
            source_checksum_recorded=True,
            documentary_evidence_available=False,
            machine_readable_profile_available=True,
            canonical_profile_created=qa["row_count"] > 0,
            row_count=qa["row_count"],
            test_count=qa["test_count"],
            duplicate_observation_identity_count=qa["duplicate_observation_identity_count"],
            contradictory_duplicate_depth_row_count=qa["contradictory_duplicate_depth_row_count"],
            depth_order_violation_count=qa["depth_order_violation_count"],
            depth_reference=depth_reference,
            depth_bsf_available=qa["depth_bsf_available"],
            channels_present=tuple(qa["channels_present"]),
            coordinates_available=False,  # locations are a separate product, never inferred here
            crs_resolved=False,
            notes=(
                "canonical measurements table registered; test coordinates live in the separate "
                "locations product and are not inferred from this table",
            ),
        )
        return facts, observed

    facts = CptEvidenceFacts(
        source_package_resolved=True,
        source_checksum_recorded=True,
        documentary_evidence_available=False,
        machine_readable_profile_available=False,
        notes=(
            f"no generic reader recognises content type {content_type!r}; source-specific column "
            "semantics and units must be declared through a provider build",
        ),
    )
    return facts, observed

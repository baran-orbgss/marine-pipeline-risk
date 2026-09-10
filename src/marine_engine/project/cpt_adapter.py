"""Generic project-registered `CPT` asset inspection (MAR-032 Section 25; MAR-032A repair).

Does NOT reimplement CPT readiness: `inspect_cpt_asset` only inspects the registered file's real
bytes and assembles `marine_engine.geotechnical.cpt_readiness.CptEvidenceFacts`; the registry
passes those facts to the existing `assess_cpt_readiness` (imported there directly).

MAR-032A keeps three facts structurally distinct and never collapses them:

* the manifest's DECLARED `evidence_role` (passed in by the registry; never mutated here);
* the OBSERVED canonical-product marker read from the Parquet file's own schema metadata
  (`cpt_profile.read_canonical_cpt_product_marker`);
* the OBSERVED structural schema (columns present / missing).

Honest outcomes for a registered path:

* documentary content (PDF / PNG / JPEG / TIFF by magic bytes) -> "documentary only, no
  machine-readable profile" -> DIGITAL_PROFILE is BLOCKING -> never READY;
* a Parquet file (magic bytes `PAR1`, never the filename) that is structurally canonical-looking
  but carries no verified product marker -> `canonical_product_identity_verified = False` ->
  DIGITAL_PROFILE is BLOCKING (`CPT_CANONICAL_PRODUCT_IDENTITY_NOT_VERIFIED`) -> never READY;
* a genuine marked canonical product -> facts from its QA; it is advertised as MEASURED CPT
  evidence only when the observed product role is `MEASURED_CPT_CPTU_PROFILE` AND the declared
  evidence role is `MEASURED` (`measured_evidence_role_verified`); a SOURCE_INTERPRETED or
  DERIVED registration keeps its declared role exactly and is denied measured-CPT readiness with
  the controlled reason `CPT_MEASURED_EVIDENCE_ROLE_NOT_VERIFIED`;
* anything else -> "no generic reader recognises this format" -> never READY. A raw operator CSV
  is not parsed here: its column semantics and units require an explicit, source-specific
  declaration (see `providers.geotechnical`), never a guess from column names.

Checksum evidence is never manufactured: `source_checksum_recorded` is true only when the
registry hands over the SHA-256 it actually computed for the registered bytes
(`registered_asset_sha256`). A standalone call without one records no checksum evidence.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from marine_engine.geotechnical import cpt_contract as contract
from marine_engine.geotechnical import cpt_inventory, cpt_profile
from marine_engine.geotechnical.cpt_readiness import CptEvidenceFacts
from marine_engine.project.categories import MEASURED

__all__ = ["CptAssetLoadError", "CANONICAL_REQUIRED_COLUMNS", "inspect_cpt_asset"]

CANONICAL_REQUIRED_COLUMNS = contract.CANONICAL_STRUCTURAL_COLUMNS
_PARQUET_MAGIC = b"PAR1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CptAssetLoadError(RuntimeError):
    """The registered file cannot be read at all -- a registration-level failure."""


def _checksum_recorded(registered_sha256: str | None) -> bool:
    return isinstance(registered_sha256, str) and bool(_SHA256_PATTERN.match(registered_sha256))


def inspect_cpt_asset(
    path: Path,
    *,
    declared_evidence_role: str | None = None,
    registered_sha256: str | None = None,
) -> tuple[CptEvidenceFacts, dict[str, Any]]:
    """Inspect the bytes at `path` and assemble readiness facts plus OBSERVED facts.

    `declared_evidence_role` is the project manifest's declared role, used ONLY as one input of the
    measured-evidence-role gate; it is echoed back under `measured_evidence_role_gate` and never
    rewritten. `registered_sha256` is the content SHA-256 the registration layer actually computed;
    without it no checksum evidence is claimed."""

    if not path.is_file():
        raise CptAssetLoadError(f"CPT asset not found: {path}")
    head = path.read_bytes()[:4096]
    content_type = cpt_inventory.sniff_content_type(head)
    checksum_recorded = _checksum_recorded(registered_sha256)
    observed: dict[str, Any] = {
        "detected_content_type": content_type,
        "registered_asset_sha256": registered_sha256 if checksum_recorded else None,
        "canonical_cpt_product_identity_verified": False,
    }
    base = {
        "source_package_resolved": True,
        "source_checksum_recorded": checksum_recorded,
        "registered_asset_sha256": registered_sha256 if checksum_recorded else None,
    }

    if content_type in contract.DOCUMENTARY_CONTENT_TYPES:
        observed["is_documentary"] = True
        facts = CptEvidenceFacts(
            **base,
            documentary_evidence_available=True,
            machine_readable_profile_available=False,
            notes=("documentary CPT evidence; not digitized (no OCR, no chart digitization)",),
        )
        return facts, observed

    observed["is_documentary"] = False
    if head.startswith(_PARQUET_MAGIC):
        try:
            marker = cpt_profile.read_canonical_cpt_product_marker(path)
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 -- any reader failure is a load failure
            raise CptAssetLoadError(f"CPT parquet could not be read: {path} ({exc})") from exc
        missing = [c for c in CANONICAL_REQUIRED_COLUMNS if c not in df.columns]
        structural_present = not missing
        identity_verified = bool(marker.verified and structural_present)
        identity_problems = list(marker.problems)
        if missing:
            identity_problems.append(f"required structural column(s) missing: {missing}")
        role_verified = bool(
            identity_verified
            and marker.product_role == contract.MEASURED_CPT_CPTU_PROFILE
            and declared_evidence_role == MEASURED
        )
        observed.update(
            {
                "columns": list(df.columns),
                "record_count": int(len(df)),
                "structural_canonical_columns_present": structural_present,
                "canonical_columns_missing": missing,
                "canonical_product_marker": marker.to_dict(),
                "canonical_cpt_product_identity_verified": identity_verified,
                "canonical_product_identity_problems": identity_problems,
                "canonical_product_role_observed": marker.product_role,
                "measured_evidence_role_gate": {
                    "observed_product_role": marker.product_role,
                    "declared_evidence_role": declared_evidence_role,
                    "verified": role_verified,
                },
            }
        )
        marker_facts = {
            "canonical_product_identity_verified": identity_verified,
            "canonical_product_role_observed": marker.product_role,
            "canonical_product_contract_observed": marker.contract_version,
            "measured_evidence_role_verified": role_verified,
        }
        if missing:
            facts = CptEvidenceFacts(
                **base,
                **marker_facts,
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
        if identity_verified:
            note = (
                "canonical measurements product registered (product marker verified from file "
                "metadata); test coordinates live in the separate locations product and are not "
                "inferred from this table"
            )
        else:
            note = (
                "structurally canonical-looking parquet WITHOUT a verified canonical product "
                "marker; column names are not measurement semantics and no identity is inferred"
            )
        facts = CptEvidenceFacts(
            **base,
            **marker_facts,
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
            notes=(note,),
        )
        return facts, observed

    facts = CptEvidenceFacts(
        **base,
        documentary_evidence_available=False,
        machine_readable_profile_available=False,
        notes=(
            f"no generic reader recognises content type {content_type!r}; source-specific column "
            "semantics and units must be declared through a provider build",
        ),
    )
    return facts, observed

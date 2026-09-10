"""CPT evidence build orchestration (MAR-032 Sections 8-9, 26, 28-29).

    manifest -> provider acquisition (cache-first) -> recursive package inventory
             -> strict source-specific parse -> generic canonical normalization -> QA
             -> CPT readiness + liquefaction-INPUT readiness -> interim/processed outputs

Numeric products (`cpt_measurements.parquet`, `cpt_tests.parquet`, `cpt_locations.gpkg`) are
written ONLY when the inspected package actually yields machine-readable numeric profiles; a
documentary-only package still produces `cpt_readiness.json` / `liquefaction_readiness.json`
with the documented limitation and never a fabricated measurements table.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import geopandas as gpd
import pandas as pd

from marine_engine.geotechnical import cpt_contract as contract
from marine_engine.geotechnical import cpt_inventory, cpt_profile, cpt_readiness
from marine_engine.geotechnical.manifest import CptEvidenceManifest
from marine_engine.providers.geotechnical import sheringham_2008_cptu

__all__ = ["PROVIDERS", "CptEvidenceBuildResult", "run_cpt_evidence_build"]

PROVIDERS: dict[str, ModuleType] = {sheringham_2008_cptu.PROVIDER_ID: sheringham_2008_cptu}

Log = Callable[[str], None]


@dataclass(frozen=True)
class CptEvidenceBuildResult:
    evidence_id: str
    acquisitions: list[dict[str, Any]]
    inventory: pd.DataFrame
    measurements: pd.DataFrame | None
    tests: pd.DataFrame | None
    qa: dict[str, Any]
    facts: cpt_readiness.CptEvidenceFacts
    cpt_readiness: cpt_readiness.CptReadinessResult
    liquefaction_readiness: dict[str, Any]
    metadata: dict[str, Any]
    outputs: dict[str, Path] = field(default_factory=dict)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def run_cpt_evidence_build(
    manifest: CptEvidenceManifest, *, log: Log | None = None
) -> CptEvidenceBuildResult:
    say = log or (lambda _m: None)
    provider = PROVIDERS.get(manifest.source.provider)
    if provider is None:
        raise ValueError(
            f"unknown CPT evidence provider {manifest.source.provider!r}; registered: "
            f"{sorted(PROVIDERS)}"
        )
    evidence_id = manifest.evidence_id
    raw_dir = manifest.paths.raw_dir
    interim_dir = manifest.paths.interim_dir
    processed_dir = manifest.paths.processed_dir
    outputs: dict[str, Path] = {}

    # 1. acquisition (cache-first)
    acquisitions = provider.acquire(
        raw_dir, include_reports=manifest.source.acquire_reports_package, log=say
    )
    acquisition_dicts = [provider.acquisition_to_dict(a) for a in acquisitions]
    declaration = provider.source_declaration()

    # 2. inventory every acquired package recursively
    inventories = []
    for acq in acquisitions:
        probes = provider.inventory_probes(acq.package_id)
        inventories.append(
            cpt_inventory.inventory_package(
                acq.extracted_dir,
                package=acq.package_id,
                profile_probe=probes["profile_probe"],
                location_probe=probes["location_probe"],
                documentary_role=probes["documentary_role"],
            )
        )
    inventory = pd.concat(inventories, ignore_index=True)
    machine_readable = cpt_inventory.machine_readable_files(inventory)
    documentary_available = bool(inventory["is_documentary"].any())
    say(
        f"inventory: {len(inventory)} files; machine-readable CPT profiles: "
        f"{len(machine_readable)}; documentary files: {int(inventory['is_documentary'].sum())}"
    )

    interim_dir.mkdir(parents=True, exist_ok=True)
    outputs["acquisition_manifest"] = _write_json(
        interim_dir / "acquisition_manifest.json",
        {
            "evidence_id": evidence_id,
            "role": contract.OFFSHORE_CPT_CPTU_SOURCE_EVIDENCE,
            "source_declaration": declaration,
            "acquisitions": acquisition_dicts,
        },
    )
    inventory_path = interim_dir / "source_file_inventory.parquet"
    inventory.to_parquet(inventory_path, index=False)
    outputs["source_file_inventory"] = inventory_path

    # 3. parse + normalize (only the machine-readable package files pass the strict probe)
    measurements: pd.DataFrame | None = None
    tests: pd.DataFrame | None = None
    provenance: tuple[dict[str, Any], ...] = ()
    unresolved: tuple[str, ...] = ()
    parse_failures: tuple[str, ...] = ()
    semantics: dict[str, Any] = {}
    qa: dict[str, Any] = cpt_profile.compute_profile_qa(cpt_profile.concat_canonical([]))
    crs = None
    if machine_readable:
        cpt_acq = acquisitions[0]
        canonical = provider.build_canonical(cpt_acq.extracted_dir, source_id=evidence_id)
        measurements = canonical.measurements
        tests = canonical.tests
        provenance = canonical.field_provenance
        unresolved = canonical.unresolved
        parse_failures = canonical.parse_failures
        semantics = canonical.source_semantics
        qa = cpt_profile.compute_profile_qa(measurements)
        crs = provider.assess_crs(manifest.declared.coordinate_crs, tests)
        say(
            f"canonical profile: {qa['row_count']} rows, {qa['test_count']} tests, channels "
            f"{qa['channels_present']}"
        )

    unresolved_canonical = tuple(
        sorted(
            {
                u.split(":", 1)[0]
                for u in unresolved
                if not u.startswith(contract.RAW_CHANNEL_PREFIX)
            }
        )
    )
    unresolved_raw = tuple(
        sorted(
            {u.split(":", 1)[0] for u in unresolved if u.startswith(contract.RAW_CHANNEL_PREFIX)}
        )
    )
    depth_reference = (
        semantics.get("depth_reference", contract.DEPTH_REFERENCE_UNRESOLVED)
        if measurements is not None
        else contract.DEPTH_REFERENCE_UNRESOLVED
    )
    facts = cpt_readiness.CptEvidenceFacts(
        source_package_resolved=bool(acquisitions),
        source_checksum_recorded=all(a.package_sha256 for a in acquisitions),
        documentary_evidence_available=documentary_available,
        machine_readable_profile_available=bool(machine_readable),
        canonical_profile_created=measurements is not None and qa["row_count"] > 0,
        row_count=qa["row_count"],
        test_count=qa["test_count"],
        declared_test_count=declaration.get("source_declared_test_count"),
        duplicate_observation_identity_count=qa["duplicate_observation_identity_count"],
        contradictory_duplicate_depth_row_count=qa["contradictory_duplicate_depth_row_count"],
        depth_order_violation_count=qa["depth_order_violation_count"],
        depth_reference=depth_reference,
        depth_bsf_available=qa["depth_bsf_available"],
        channels_present=tuple(qa["channels_present"]),
        unit_unresolved_canonical_channels=unresolved_canonical,
        unit_unresolved_raw_channels=unresolved_raw,
        coordinates_available=bool(crs and crs.coordinates_available),
        crs_resolved=bool(crs and crs.crs_resolved),
        crs_conflict=crs.conflict if crs else None,
        cone_area_ratio_source_stated=semantics.get("cone_area_ratio_source_stated") is not None,
    )
    readiness = cpt_readiness.assess_cpt_readiness(facts)
    liquefaction = cpt_readiness.assess_liquefaction_input_readiness(facts)

    # 4. source-evidence summary (interim)
    outputs["cpt_source_evidence"] = _write_json(
        interim_dir / "cpt_source_evidence.json",
        {
            "evidence_id": evidence_id,
            "role": contract.OFFSHORE_CPT_CPTU_SOURCE_EVIDENCE,
            "official_source_resolved": bool(acquisitions),
            "package_checksums_recorded": facts.source_checksum_recorded,
            "documentary_cpt_evidence_available": documentary_available,
            "machine_readable_cpt_profile_available": bool(machine_readable),
            "machine_readable_files": machine_readable,
            "file_count": int(len(inventory)),
            "content_type_counts": inventory["detected_content_type"].value_counts().to_dict(),
            "candidate_role_counts": inventory["candidate_role"].value_counts().to_dict(),
            "parse_failures": list(parse_failures),
            "ocr_or_chart_digitization_performed": False,
        },
    )

    # 5. processed outputs
    processed_dir.mkdir(parents=True, exist_ok=True)
    canonical_fields: list[str] = []
    if measurements is not None and qa["row_count"] > 0:
        measurements_path = processed_dir / "cpt_measurements.parquet"
        measurements.to_parquet(measurements_path, index=False)
        outputs["cpt_measurements"] = measurements_path
        canonical_fields = list(measurements.columns)
        if tests is not None and not tests.empty:
            tests_path = processed_dir / "cpt_tests.parquet"
            tests.to_parquet(tests_path, index=False)
            outputs["cpt_tests"] = tests_path
            if crs is not None and crs.crs_resolved and crs.resolved_crs and not crs.conflict:
                gdf = gpd.GeoDataFrame(
                    tests,
                    geometry=gpd.points_from_xy(tests["position_x_raw"], tests["position_y_raw"]),
                    crs=crs.resolved_crs,
                )
                gpkg_path = processed_dir / "cpt_locations.gpkg"
                gdf.to_file(gpkg_path, layer="cpt_locations", driver="GPKG")
                outputs["cpt_locations"] = gpkg_path

    metadata = {
        "evidence_id": evidence_id,
        "roles": {
            "source_evidence": contract.OFFSHORE_CPT_CPTU_SOURCE_EVIDENCE,
            "measurements": (
                contract.MEASURED_CPT_CPTU_PROFILE if "cpt_measurements" in outputs else None
            ),
            "readiness": contract.LIQUEFACTION_INPUT_READINESS_ASSESSMENT,
        },
        "scope_statement": contract.MAR_032_SCOPE_STATEMENT,
        **{k: v for k, v in declaration.items() if k != "role"},
        "acquired_files": [
            {
                "package_id": a["package_id"],
                "package_url": a["package_url"],
                "local_archive_path": a["local_archive_path"],
                "package_bytes": a["package_bytes"],
                "sha256": a["package_sha256"],
                "archive_type": a["archive_type"],
                "already_cached": a["already_cached"],
            }
            for a in acquisition_dicts
        ],
        "file_inventory_path": str(inventory_path),
        "file_inventory_count": int(len(inventory)),
        "documentary_cpt_evidence_available": documentary_available,
        "machine_readable_cpt_profile_available": bool(machine_readable),
        "canonical_profile_created": facts.canonical_profile_created,
        "canonical_profile_fields": canonical_fields,
        "field_provenance": list(provenance),
        "source_units": semantics.get("source_unit_tokens", {}),
        "source_unit_token_aliases": semantics.get("source_unit_token_aliases", {}),
        "normalized_units": {
            k: v for k, v in contract.CANONICAL_FIELD_UNITS.items() if k in canonical_fields
        },
        "depth_reference": depth_reference,
        "depth_reference_source_definition": semantics.get("depth_reference_source_definition"),
        "depth_reference_limitation": semantics.get("depth_reference_limitation"),
        "coordinate_reference": {
            "coordinates_available": facts.coordinates_available,
            "declared_crs": crs.declared_crs if crs else None,
            "resolved_crs": crs.resolved_crs if crs else None,
            "crs_resolved": facts.crs_resolved,
            "conflict": facts.crs_conflict,
            "observed": crs.observed if crs else {},
            "note": crs.note if crs else contract.SPATIAL_LOCATION_UNRESOLVED,
        },
        "source_semantics": semantics,
        "unresolved": list(unresolved),
        "parse_failures": list(parse_failures),
        "profile_qa": qa,
        **contract.NOT_COMPUTED_FLAGS,
        "prohibited_profile_operations_performed": [],
        "prohibited_profile_operations": list(contract.PROHIBITED_PROFILE_OPERATIONS),
        "product_label_boundary": (
            "no MAR-032 product is a liquefaction susceptibility, hazard or risk product; "
            "see cpt_contract.PROHIBITED_PRODUCT_LABELS"
        ),
        "references": list(contract.REFERENCES),
        "surface_sediment_boundary": contract.SURFACE_SEDIMENT_BOUNDARY_STATEMENT,
        "limitations": readiness.reasons() + list(unresolved),
    }
    outputs["cpt_metadata"] = _write_json(processed_dir / "cpt_metadata.json", metadata)
    outputs["cpt_readiness"] = _write_json(
        processed_dir / "cpt_readiness.json",
        {
            "evidence_id": evidence_id,
            "role": contract.LIQUEFACTION_INPUT_READINESS_ASSESSMENT,
            "cpt_evidence_readiness": readiness.to_dict(),
            "facts": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(facts).items()},
            "DOCUMENTARY_CPT_EVIDENCE_AVAILABLE": documentary_available,
            "DIGITAL_NUMERIC_CPT_PROFILE_READY": facts.canonical_profile_created,
        },
    )
    outputs["liquefaction_readiness"] = _write_json(
        processed_dir / "liquefaction_readiness.json",
        {"evidence_id": evidence_id, **liquefaction, "pl854": cpt_readiness.pl854_cpt_status()},
    )

    return CptEvidenceBuildResult(
        evidence_id=evidence_id,
        acquisitions=acquisition_dicts,
        inventory=inventory,
        measurements=measurements,
        tests=tests,
        qa=qa,
        facts=facts,
        cpt_readiness=readiness,
        liquefaction_readiness=liquefaction,
        metadata=metadata,
        outputs=outputs,
    )

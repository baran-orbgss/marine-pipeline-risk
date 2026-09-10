"""MAR-032 -- generic offshore CPT/CPTU evidence & liquefaction-INPUT readiness (Section 32 matrix).

Everything here is offline. The real Sheringham Shoal 2008 package is NEVER fetched or required:
acquisition is exercised against a monkeypatched HTTP layer serving SYNTHETIC ZIP packages, and
every numeric value below is a SYNTHETIC_TEST_FIXTURE -- none is a Sheringham measurement.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from marine_engine import cli
from marine_engine.geotechnical import (
    cpt_contract as contract,
)
from marine_engine.geotechnical import (
    cpt_inventory,
    cpt_profile,
    cpt_readiness,
    evidence_build,
)
from marine_engine.geotechnical import manifest as cpt_manifest
from marine_engine.geotechnical import report as cpt_report
from marine_engine.project import categories, cpt_adapter
from marine_engine.project import manifest as project_manifest
from marine_engine.project import registry as project_registry
from marine_engine.providers.geotechnical import sheringham_2008_cptu as provider

SYNTHETIC_TAG = "SYNTHETIC_TEST_FIXTURE"

# --- synthetic fixtures ---------------------------------------------------------------------------


def _geo_csv_text(
    test_id: str, rows: list[tuple[Any, ...]], *, position: str = " 1000.5; 2000.5;WGS 84;UTM;31"
) -> str:
    """A SYNTHETIC file in the exact GEO structure inspected in the real package."""

    header = [
        f"Job ;99999;{SYNTHETIC_TAG}",
        f"CPT name;;{test_id}",
        "Date & Time ;2008-10-14 5:47:16 PM",
        "Depth ;15.70",
        "Clients ref. :;",
        f"Position:;{position}",
        "Dist. Tip to Sleeve center;10.50",
        provider.GEO_COLUMN_HEADER,
        provider.GEO_UNIT_ROW,
    ]
    body = [";".join(str(v) for v in row) + ";" for row in rows]
    return "\r\n".join(header + body) + "\r\n"


SYN_ROWS_A = [
    (1, -0.05, 0.0, 0.0, 0.0, 1.5, 39735.7411603),
    (2, 0.00, 0.10, 0.002, 0.001, 1.6, 39735.7411700),
    (3, 0.02, 1.20, 0.010, 0.005, 1.6, 39735.7411800),
    (4, 0.04, 2.50, 0.020, 0.012, 1.7, 39735.7411900),
]
SYN_ROWS_B = [
    (1, -0.03, 0.0, 0.0, 0.0, 2.0, 39736.1),
    (2, 0.01, 0.50, 0.004, 0.002, 2.1, 39736.2),
    (3, 0.03, 3.00, 0.030, 0.020, 2.1, 39736.3),
]


def _write_geo_csv(path: Path, test_id: str, rows: list[tuple[Any, ...]], **kw: Any) -> Path:
    path.write_bytes(_geo_csv_text(test_id, rows, **kw).encode("latin-1"))
    return path


def _fake_pdf_bytes() -> bytes:
    return b"%PDF-1.4\n% SYNTHETIC_TEST_FIXTURE documentary log -- no numeric content\n%%EOF\n"


def _fake_png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _fake_tiff_bytes() -> bytes:
    return b"II*\x00" + b"\x00" * 64


def _cpt_package_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("CPT-SYN1.csv", _geo_csv_text("CPT-SYN1", SYN_ROWS_A).encode("latin-1"))
        zf.writestr("CPT-SYN2.csv", _geo_csv_text("CPT-SYN2", SYN_ROWS_B).encode("latin-1"))
        zf.writestr(
            "_metadata/medin_metadata.xml",
            b'<?xml version="1.0"?><gmd:MD_Metadata xmlns:gmd="x">synthetic</gmd:MD_Metadata>',
        )
        zf.writestr("dataname.txt", b"CPT\n")
    return buf.getvalue()


def _reports_package_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Report Part B.pdf", _fake_pdf_bytes())
        zf.writestr("Report Part C.pdf", _fake_pdf_bytes())
        zf.writestr("rendered_log.png", _fake_png_bytes())
        zf.writestr("_metadata/reports_metadata.xml", b"<?xml version='1.0'?><xman_metadata/>")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes, url: str):
        self.content = content
        self.url = url
        self.status_code = 200
        self.headers = {
            "Content-Length": str(len(content)),
            "Content-Type": "application/octet-stream",
            "ETag": "synthetic",
            "Last-Modified": "Mon, 20 Nov 2023 08:13:35 GMT",
            "Accept-Ranges": "bytes",
        }

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def fake_http(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    packages = {
        provider.CPT_LOGS_PACKAGE_URL: _cpt_package_zip(),
        provider.REPORTS_PACKAGE_URL: _reports_package_zip(),
    }

    def fake_get(url: str, timeout: float = 0.0) -> _FakeResponse:
        calls.append(url)
        if url not in packages:
            raise AssertionError(f"unexpected (guessed?) URL requested: {url}")
        return _FakeResponse(packages[url], url)

    monkeypatch.setattr(provider.requests, "get", fake_get)
    return calls


def _manifest(
    tmp_path: Path, *, reports: bool = True, crs: str | None = "EPSG:32631"
) -> cpt_manifest.CptEvidenceManifest:
    return cpt_manifest.CptEvidenceManifest.model_validate(
        {
            "evidence_id": "syn_cptu",
            "source": {"provider": provider.PROVIDER_ID, "acquire_reports_package": reports},
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "interim_dir": str(tmp_path / "interim"),
                "processed_dir": str(tmp_path / "processed"),
            },
            "declared": {"coordinate_crs": crs},
        }
    )


def _simple_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "depth": [0.0, 0.1, 0.2, 0.3],
            "qc": [0.5, 1.0, np.nan, 2.0],
            "fs": [10.0, 20.0, 30.0, 40.0],
            "u": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _depth(
    reference: str = contract.DEPTH_BELOW_SEABED, unit: str | None = "m"
) -> cpt_profile.DepthDeclaration:
    return cpt_profile.DepthDeclaration("depth", unit, reference, "synthetic declaration")


def _channels(qc_unit: str | None = "MPa") -> list[cpt_profile.ChannelDeclaration]:
    return [
        cpt_profile.ChannelDeclaration("qc", qc_unit, contract.QC_MPA, "synthetic qc"),
        cpt_profile.ChannelDeclaration("fs", "kPa", contract.FS_KPA, "synthetic fs"),
        cpt_profile.ChannelDeclaration("u", "kPa", contract.U2_KPA, "synthetic u2"),
    ]


# --- 1-3: exact official source identity and source-declared counts -------------------------------


def test_01_exact_official_source_identity():
    decl = provider.source_declaration()
    assert decl["source_uri"] == "https://www.marinedataexchange.co.uk/details/1964/summary"
    assert decl["source_series_id"] == "TCE-1964"
    assert decl["source_publisher"] == "The Crown Estate"
    assert "Sheringham Shoal" in decl["source_name"]
    assert decl["source_dataset_package_label"] == "Cone PenetrationTests Logs"
    assert decl["role"] == contract.SOURCE_DECLARED_COLLECTION_METADATA


def test_02_03_declared_counts_recorded_as_declaration_not_enforced():
    decl = provider.source_declaration()
    assert decl["source_declared_test_count"] == 101
    assert decl["source_declared_location_count"] == 86
    assert decl["declared_counts_are_enforced_as_parsed_counts"] is False
    # A parsed count that differs is a LIMITATION on IDENTITY, never a failure or a repair.
    facts = cpt_readiness.CptEvidenceFacts(
        source_package_resolved=True,
        source_checksum_recorded=True,
        machine_readable_profile_available=True,
        canonical_profile_created=True,
        row_count=10,
        test_count=100,
        declared_test_count=101,
        depth_reference=contract.DEPTH_BELOW_SEABED,
        depth_bsf_available=True,
        channels_present=(contract.QC_MPA,),
    )
    result = cpt_readiness.assess_cpt_readiness(facts)
    identity = result.axis(contract.IDENTITY)
    assert identity.status == contract.READY_WITH_LIMITATIONS
    assert any("101" in r and "100" in r for r in identity.limitation_reasons)


# --- 4-7: acquisition SHA-256, deterministic inventory, cache reuse, no guessed URL ---------------


def test_04_acquisition_records_sha256_and_http_evidence(tmp_path: Path, fake_http: list[str]):
    acqs = provider.acquire(tmp_path / "raw", include_reports=True)
    assert [a.package_id for a in acqs] == ["cpt_logs", "reports"]
    for acq in acqs:
        assert acq.package_sha256 == hashlib.sha256(acq.local_archive_path.read_bytes()).hexdigest()
        assert acq.package_bytes == acq.local_archive_path.stat().st_size
        assert acq.archive_type == "application/zip"
        assert acq.http_evidence["status_code"] == 200
        assert acq.already_cached is False
        sidecar = json.loads((tmp_path / "raw" / f"{acq.package_id}.acquisition.json").read_text())
        assert sidecar["package_sha256"] == acq.package_sha256
    # The raw archive is saved unaltered under the URL's own (unquoted) file name.
    assert (tmp_path / "raw" / "122-1964-Cone PenetrationTests Logs.zip").exists()


def test_05_file_inventory_is_deterministic_and_content_based(tmp_path: Path, fake_http: list[str]):
    acqs = provider.acquire(tmp_path / "raw", include_reports=True)
    probes = provider.inventory_probes("cpt_logs")
    inv_a = cpt_inventory.inventory_package(acqs[0].extracted_dir, package="cpt_logs", **probes)
    inv_b = cpt_inventory.inventory_package(acqs[0].extracted_dir, package="cpt_logs", **probes)
    pd.testing.assert_frame_equal(inv_a, inv_b)
    assert list(inv_a.columns) == list(cpt_inventory.INVENTORY_COLUMNS)
    assert inv_a["relative_path"].tolist() == sorted(inv_a["relative_path"].tolist())
    roles = dict(zip(inv_a["relative_path"], inv_a["candidate_role"], strict=True))
    assert roles["CPT-SYN1.csv"] == contract.ROLE_MACHINE_READABLE_CPT_PROFILE
    assert roles["_metadata/medin_metadata.xml"] == contract.ROLE_METADATA
    assert roles["dataname.txt"] == contract.ROLE_UNKNOWN  # text, but not a CPT record
    for _, row in inv_a.iterrows():
        real = hashlib.sha256((acqs[0].extracted_dir / row.relative_path).read_bytes()).hexdigest()
        assert row.sha256 == real


def test_06_cache_reuse_makes_no_network_request(tmp_path: Path, fake_http: list[str]):
    provider.acquire(tmp_path / "raw", include_reports=True)
    assert len(fake_http) == 2
    again = provider.acquire(tmp_path / "raw", include_reports=True)
    assert len(fake_http) == 2  # no further HTTP call
    assert all(a.already_cached for a in again)


def test_06b_cache_with_tampered_bytes_is_refused(tmp_path: Path, fake_http: list[str]):
    acqs = provider.acquire(tmp_path / "raw", include_reports=False)
    archive = acqs[0].local_archive_path
    data = bytearray(archive.read_bytes())
    data[-1] ^= 0xFF  # same size, different content
    archive.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256"):
        provider.acquire(tmp_path / "raw", include_reports=False)


def test_07_only_the_intercepted_package_urls_are_requested(tmp_path: Path, fake_http: list[str]):
    provider.acquire(tmp_path / "raw", include_reports=True)
    assert fake_http == [provider.CPT_LOGS_PACKAGE_URL, provider.REPORTS_PACKAGE_URL]
    assert provider.CPT_LOGS_PACKAGE_URL == (
        "https://www.marinedataexchange.co.uk/pub/TCE/122-1964-Cone%20PenetrationTests%20Logs.zip"
    )
    assert "anchor-click" in provider.URL_RESOLUTION_METHOD
    assert "guess" in provider.URL_RESOLUTION_METHOD  # documented as NOT guessed


# --- 8-10: PDF / image are documentary; no OCR or chart digitization path -------------------------


def test_08_pdf_is_documentary_not_numeric(tmp_path: Path):
    pdf = tmp_path / "log.pdf"
    pdf.write_bytes(_fake_pdf_bytes())
    assert cpt_inventory.sniff_content_type(pdf.read_bytes()[:64]) == contract.CONTENT_PDF
    assert provider.probe_geo_cpt_csv(pdf, pdf.read_bytes()) is False
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "log.pdf").write_bytes(_fake_pdf_bytes())
    inv = cpt_inventory.inventory_package(
        tmp_path / "pkg", package="p", profile_probe=provider.probe_geo_cpt_csv
    )
    assert inv["is_documentary"].tolist() == [True]
    assert cpt_inventory.documentary_only(inv) is True
    assert cpt_inventory.machine_readable_files(inv) == []


@pytest.mark.parametrize(
    "payload", [_fake_png_bytes(), _fake_tiff_bytes(), b"\xff\xd8\xff\xe0jpeg"]
)
def test_09_image_log_is_documentary_not_numeric(tmp_path: Path, payload: bytes):
    (tmp_path / "pkg").mkdir()
    # Deliberately mis-labelled with a .csv extension: the bytes decide, not the name.
    (tmp_path / "pkg" / "CPT-X.csv").write_bytes(payload)
    inv = cpt_inventory.inventory_package(
        tmp_path / "pkg", package="p", profile_probe=provider.probe_geo_cpt_csv
    )
    assert inv["is_documentary"].tolist() == [True]
    assert inv["candidate_role"].tolist() == [contract.ROLE_UNKNOWN]


def test_10_no_ocr_or_chart_digitization_dependency():
    forbidden = re.compile(
        r"\b(pytesseract|tesseract|easyocr|PIL|Pillow|cv2|opencv|skimage|fitz|pymupdf|pdfplumber|"
        r"pypdf|PyPDF2|pdfminer|camelot|tabula|ocrmypdf)\b"
    )
    for module in (
        cpt_inventory,
        cpt_profile,
        cpt_readiness,
        evidence_build,
        cpt_report,
        cpt_adapter,
        provider,
    ):
        assert not forbidden.search(inspect.getsource(module)), module.__name__
    assert contract.NOT_COMPUTED_FLAGS["qt_unequal_area_correction_applied"] is False
    assert "OCR of PDF or image logs" in contract.PROHIBITED_PROFILE_OPERATIONS


# --- 11-19: structured fixture accepted; units; depth reference; qc/qt/fs/u2 distinct -------------


def test_11_structured_numeric_fixture_accepted():
    build = cpt_profile.build_canonical_measurements(
        _simple_observations(),
        source_id=SYNTHETIC_TAG,
        test_id="SYN-001",
        depth=_depth(),
        channels=_channels(),
        observation_index_column="idx",
    )
    df = build.measurements
    assert len(df) == 4
    assert df[contract.TEST_ID].unique().tolist() == ["SYN-001"]
    assert df[contract.OBSERVATION_INDEX].tolist() == [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_allclose(df[contract.DEPTH_BSF_M], [0.0, 0.1, 0.2, 0.3])
    assert build.unresolved == ()


def test_12_explicit_unit_conversion_is_deterministic():
    obs = _simple_observations()
    build = cpt_profile.build_canonical_measurements(
        obs,
        source_id=SYNTHETIC_TAG,
        test_id="SYN-001",
        depth=cpt_profile.DepthDeclaration("depth", "cm", contract.DEPTH_BELOW_SEABED, "cm depth"),
        channels=[
            cpt_profile.ChannelDeclaration("qc", "kPa", contract.QC_MPA, "qc in kPa"),
            cpt_profile.ChannelDeclaration("fs", "MPa", contract.FS_KPA, "fs in MPa"),
            cpt_profile.ChannelDeclaration("u", "Pa", contract.U2_KPA, "u in Pa"),
        ],
    )
    df = build.measurements
    np.testing.assert_allclose(df[contract.DEPTH_BSF_M], obs["depth"] * 0.01)
    np.testing.assert_allclose(df[contract.QC_MPA], obs["qc"] * 0.001)
    np.testing.assert_allclose(df[contract.FS_KPA], obs["fs"] * 1000.0)
    np.testing.assert_allclose(df[contract.U2_KPA], obs["u"] * 0.001)
    conversions = {p["normalized_field"]: p["conversion_applied"] for p in build.field_provenance}
    assert conversions[contract.QC_MPA] == "x0.001 (kPa -> MPa)"
    assert conversions[contract.FS_KPA] == "x1000 (MPa -> kPa)"
    assert cpt_profile.conversion_factor("MPa", "kPa") == 1000.0


def test_13_unknown_unit_is_not_guessed():
    for unit in (None, "psi", "bar", "tsf", "unknown"):
        build = cpt_profile.build_canonical_measurements(
            _simple_observations(),
            source_id=SYNTHETIC_TAG,
            test_id="SYN-001",
            depth=_depth(),
            channels=_channels(qc_unit=unit),
        )
        assert build.measurements[contract.QC_MPA].isna().all(), unit
        assert build.measurements["raw__qc"].notna().sum() == 3  # raw preserved
        assert any(u.startswith(contract.QC_MPA) for u in build.unresolved), unit
    assert cpt_profile.conversion_factor("psi", "MPa") is None
    assert cpt_profile.conversion_factor(None, "MPa") is None


def test_14_15_depth_reference_required_and_unresolved_stays_unresolved():
    build = cpt_profile.build_canonical_measurements(
        _simple_observations(),
        source_id=SYNTHETIC_TAG,
        test_id="SYN-001",
        depth=_depth(reference=contract.DEPTH_REFERENCE_UNRESOLVED),
        channels=_channels(),
    )
    df = build.measurements
    assert df[contract.DEPTH_BSF_M].isna().all()
    assert df[contract.DEPTH_SOURCE_VALUE].tolist() == [0.0, 0.1, 0.2, 0.3]  # raw preserved
    assert df[contract.DEPTH_REFERENCE_FIELD].unique().tolist() == [
        contract.DEPTH_REFERENCE_UNRESOLVED
    ]
    assert any(contract.DEPTH_REFERENCE_UNRESOLVED in u for u in build.unresolved)
    with pytest.raises(cpt_profile.CptProfileError):
        cpt_profile.DepthDeclaration("depth", "m", "DEPTH_BELOW_RIG_PLATE", "x")
    # Known unit but unknown reference -> still null; known reference but unknown unit -> null.
    build2 = cpt_profile.build_canonical_measurements(
        _simple_observations(),
        source_id=SYNTHETIC_TAG,
        test_id="SYN-001",
        depth=_depth(unit="ft"),
        channels=_channels(),
    )
    assert build2.measurements[contract.DEPTH_BSF_M].isna().all()


def test_16_17_qc_and_qt_remain_distinct_and_qc_is_never_copied_into_qt():
    build = cpt_profile.build_canonical_measurements(
        _simple_observations(),
        source_id=SYNTHETIC_TAG,
        test_id="SYN-001",
        depth=_depth(),
        channels=_channels(),
    )
    df = build.measurements
    assert df[contract.QC_MPA].notna().sum() == 3
    assert df[contract.QT_MPA].isna().all()
    # With an explicitly declared qt column both are kept separately, unequal.
    obs = _simple_observations().assign(qt=[0.6, 1.1, 1.6, 2.1])
    build2 = cpt_profile.build_canonical_measurements(
        obs,
        source_id=SYNTHETIC_TAG,
        test_id="SYN-001",
        depth=_depth(),
        channels=[
            *_channels(),
            cpt_profile.ChannelDeclaration("qt", "MPa", contract.QT_MPA, "explicit source qt"),
        ],
    )
    df2 = build2.measurements
    np.testing.assert_allclose(df2[contract.QT_MPA], obs["qt"])
    assert not np.allclose(df2[contract.QT_MPA], df2[contract.QC_MPA].fillna(-1))
    # Two channels claiming the same canonical field is an error, never a silent overwrite.
    with pytest.raises(cpt_profile.CptProfileError):
        cpt_profile.build_canonical_measurements(
            obs,
            source_id=SYNTHETIC_TAG,
            test_id="SYN-001",
            depth=_depth(),
            channels=[
                *_channels(),
                cpt_profile.ChannelDeclaration("qt", "MPa", contract.QC_MPA, "dup"),
            ],
        )
    # No area-ratio / pore-pressure correction identifiers exist in the generic layer.
    source = inspect.getsource(cpt_profile)
    assert "(1 - a)" not in source and "area_ratio" not in source.lower().replace(
        "_source_stated", ""
    )


def test_18_19_fs_and_u2_remain_distinct_channels():
    build = cpt_profile.build_canonical_measurements(
        _simple_observations(),
        source_id=SYNTHETIC_TAG,
        test_id="SYN-001",
        depth=_depth(),
        channels=_channels(),
    )
    df = build.measurements
    np.testing.assert_allclose(df[contract.FS_KPA], [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(df[contract.U2_KPA], [1.0, 2.0, 3.0, 4.0])
    assert contract.CANONICAL_FIELD_UNITS[contract.FS_KPA] == "kPa"
    assert contract.CANONICAL_FIELD_UNITS[contract.U2_KPA] == "kPa"


# --- 20-26: duplicates; no averaging, interpolation, smoothing, resampling, despiking ------------


def test_20_21_duplicate_identities_detected_and_contradictory_duplicates_not_averaged():
    obs = pd.DataFrame(
        {
            "idx": [1, 2, 2, 3],
            "depth": [0.0, 0.1, 0.1, 0.2],
            "qc": [1.0, 2.0, 4.0, 3.0],
            "fs": [1.0, 1.0, 1.0, 1.0],
            "u": [0.0, 0.0, 0.0, 0.0],
        }
    )
    build = cpt_profile.build_canonical_measurements(
        obs,
        source_id=SYNTHETIC_TAG,
        test_id="SYN-DUP",
        depth=_depth(),
        channels=_channels(),
        observation_index_column="idx",
    )
    qa = cpt_profile.compute_profile_qa(build.measurements)
    assert qa["row_count"] == 4  # nothing collapsed
    assert qa["duplicate_observation_identity_count"] == 2
    assert qa["duplicate_depth_row_count"] == 2
    assert qa["contradictory_duplicate_depth_row_count"] == 2
    assert build.measurements[contract.QC_MPA].tolist() == [1.0, 2.0, 4.0, 3.0]  # 2.0 & 4.0 kept
    facts = cpt_readiness.CptEvidenceFacts(
        source_package_resolved=True,
        source_checksum_recorded=True,
        machine_readable_profile_available=True,
        canonical_profile_created=True,
        row_count=4,
        test_count=1,
        duplicate_observation_identity_count=2,
        contradictory_duplicate_depth_row_count=2,
        depth_reference=contract.DEPTH_BELOW_SEABED,
        depth_bsf_available=True,
        channels_present=(contract.QC_MPA,),
    )
    result = cpt_readiness.assess_cpt_readiness(facts)
    assert result.axis(contract.IDENTITY).status == contract.NOT_READY  # duplicate IDENTITY blocks
    assert result.cpt_profile_status == contract.NOT_READY
    # Distinct observations sharing a depth value with different readings are kept, flagged as a
    # LIMITATION (never averaged/collapsed) -- they are not a duplicated identity.
    facts_depth_only = cpt_readiness.CptEvidenceFacts(
        source_package_resolved=True,
        source_checksum_recorded=True,
        machine_readable_profile_available=True,
        canonical_profile_created=True,
        row_count=4,
        test_count=1,
        duplicate_observation_identity_count=0,
        contradictory_duplicate_depth_row_count=2,
        depth_reference=contract.DEPTH_BELOW_SEABED,
        depth_bsf_available=True,
        channels_present=(contract.QC_MPA,),
    )
    result2 = cpt_readiness.assess_cpt_readiness(facts_depth_only)
    identity2 = result2.axis(contract.IDENTITY)
    assert identity2.status == contract.READY_WITH_LIMITATIONS
    assert identity2.blocking_reasons == ()
    assert any("not averaged" in r for r in identity2.limitation_reasons)


def test_22_25_no_interpolation_smoothing_resampling_or_despiking():
    obs = _simple_observations()  # qc has a NaN at row 3
    obs.loc[3, "qc"] = 500.0  # an obvious spike -- must survive untouched
    build = cpt_profile.build_canonical_measurements(
        obs, source_id=SYNTHETIC_TAG, test_id="SYN-001", depth=_depth(), channels=_channels()
    )
    df = build.measurements
    assert np.isnan(df[contract.QC_MPA].iloc[2])  # gap NOT filled
    assert df[contract.QC_MPA].iloc[3] == 500.0  # spike NOT removed
    assert df[contract.DEPTH_BSF_M].tolist() == obs["depth"].tolist()  # depth NOT resampled
    forbidden = re.compile(
        r"\.(interpolate|rolling|resample|ewm|fillna|ffill|bfill|savgol_filter|medfilt|"
        r"drop_duplicates\(\)\s*$)|np\.interp|scipy\.signal|scipy\.interpolate|\.sort_values\("
    )
    for module in (cpt_profile, evidence_build, provider):
        for line in inspect.getsource(module).splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"'):
                continue
            assert not forbidden.search(stripped), f"{module.__name__}: {stripped}"


def test_26_profile_statistics_are_deterministic():
    obs = _simple_observations()
    frames = [
        cpt_profile.build_canonical_measurements(
            obs, source_id=SYNTHETIC_TAG, test_id="SYN-001", depth=_depth(), channels=_channels()
        ).measurements
        for _ in range(2)
    ]
    pd.testing.assert_frame_equal(frames[0], frames[1])
    assert cpt_profile.compute_profile_qa(frames[0]) == cpt_profile.compute_profile_qa(frames[1])
    qa = cpt_profile.compute_profile_qa(frames[0])
    assert qa["channel_finite_fraction"][contract.QC_MPA] == 0.75
    assert qa["channel_missing_count"][contract.QC_MPA] == 1
    assert qa["tests"]["SYN-001"]["depth_source_min"] == 0.0
    assert qa["tests"]["SYN-001"]["depth_source_max"] == 0.3


# --- 27-29: coordinates / CRS ---------------------------------------------------------------------


def test_27_missing_coordinates_remain_missing():
    assessment = provider.assess_crs("EPSG:32631", pd.DataFrame())
    assert assessment.coordinates_available is False
    assert assessment.crs_resolved is False
    assert assessment.note == contract.SPATIAL_LOCATION_UNRESOLVED
    tests = pd.DataFrame(
        {
            "position_x_raw": [None],
            "position_y_raw": [None],
            "position_datum_raw": [None],
            "position_projection_raw": [None],
            "position_zone_raw": [None],
        }
    )
    assessment2 = provider.assess_crs("EPSG:32631", tests)
    assert assessment2.coordinates_available is False and assessment2.resolved_crs is None


def _tests_with_coords() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position_x_raw": [1000.5, 1100.5],
            "position_y_raw": [2000.5, 2100.5],
            "position_datum_raw": ["WGS 84", "WGS 84"],
            "position_projection_raw": ["UTM", "UTM"],
            "position_zone_raw": ["31", "31"],
        }
    )


def test_28_unknown_crs_remains_unknown_without_declaration():
    assessment = provider.assess_crs(None, _tests_with_coords())
    assert assessment.coordinates_available is True
    assert assessment.crs_resolved is False
    assert assessment.resolved_crs is None
    assert assessment.conflict is None
    assert contract.CRS_UNRESOLVED in assessment.note


def test_29_no_auto_utm_and_wrong_declared_crs_is_a_conflict():
    ok = provider.assess_crs("EPSG:32631", _tests_with_coords())
    assert ok.crs_resolved is True and ok.conflict is None
    wrong_zone = provider.assess_crs("EPSG:32632", _tests_with_coords())
    assert wrong_zone.crs_resolved is False and "UTM zone" in (wrong_zone.conflict or "")
    wrong_datum = provider.assess_crs("EPSG:23031", _tests_with_coords())  # ED50 / UTM 31N
    assert wrong_datum.crs_resolved is False and "WGS 84" in (wrong_datum.conflict or "")
    geographic = provider.assess_crs("EPSG:4326", _tests_with_coords())
    assert geographic.crs_resolved is False and "not projected" in (geographic.conflict or "")
    invalid = provider.assess_crs("NOT_A_CRS", _tests_with_coords())
    assert invalid.crs_resolved is False and "not a valid" in (invalid.conflict or "")
    # Nothing in the generic or provider code infers a UTM zone from coordinates or location.
    for module in (provider, evidence_build, cpt_profile, cpt_adapter):
        source = inspect.getsource(module)
        assert "estimate_utm_crs" not in source and "utm_from" not in source.lower()
        assert "reproject(" not in source and ".to_crs(" not in source
        assert "rasterio.warp" not in source and "Transformer" not in source
    # A CRS conflict blocks the spatial axis and the whole profile.
    facts = cpt_readiness.CptEvidenceFacts(
        source_package_resolved=True,
        source_checksum_recorded=True,
        machine_readable_profile_available=True,
        canonical_profile_created=True,
        row_count=1,
        test_count=1,
        depth_reference=contract.DEPTH_BELOW_SEABED,
        depth_bsf_available=True,
        channels_present=(contract.QC_MPA,),
        coordinates_available=True,
        crs_resolved=False,
        crs_conflict="declared CRS contradicts source",
    )
    result = cpt_readiness.assess_cpt_readiness(facts)
    assert result.axis(contract.SPATIAL_REFERENCE).status == contract.NOT_READY
    assert result.cpt_profile_status == contract.NOT_READY


# --- 30-32: BGS Folk / D50 / MAR-013 mobility cannot produce liquefaction evidence ----------------


def test_30_32_surface_sediment_and_mobility_cannot_feed_liquefaction():
    # The contract module legitimately NAMES the boundary it forbids; every mechanics module must
    # not mention surface-sediment descriptors or a liquefiability verdict at all.
    mechanics_modules = (
        cpt_inventory,
        cpt_profile,
        cpt_readiness,
        evidence_build,
        cpt_report,
        cpt_adapter,
        provider,
    )
    for module in (contract, *mechanics_modules):
        source = inspect.getsource(module)
        assert "marine_engine.sediment" not in source, module.__name__
        assert "marine_engine.metocean" not in source, module.__name__
    for module in mechanics_modules:
        lowered = inspect.getsource(module).lower()
        for token in ("folk", "d50", "liquefiable", "relative_density", "spt_n", "state_parameter"):
            assert token not in lowered, f"{module.__name__} mentions {token!r}"
    field_names = set(cpt_readiness.CptEvidenceFacts.__dataclass_fields__)
    assert not {f for f in field_names if "sediment" in f or "grain" in f or "mobility" in f}
    assert "never converted into LIQUEFIABLE" in contract.SURFACE_SEDIMENT_BOUNDARY_STATEMENT
    liq = cpt_readiness.assess_liquefaction_input_readiness(cpt_readiness.CptEvidenceFacts())
    assert liq["surface_sediment_boundary"] == contract.SURFACE_SEDIMENT_BOUNDARY_STATEMENT
    assert liq["status"] == contract.NOT_EVALUABLE


# --- 33-40: no CSR, CRR, FoS, LPI, probability, settlement, lateral spreading, pore pressure ----


def test_33_40_no_liquefaction_calculation_exists():
    flags = contract.NOT_COMPUTED_FLAGS
    for key in (
        "CSR_computed",
        "CRR_computed",
        "liquefaction_factor_of_safety_computed",
        "LPI_computed",
        "liquefaction_probability_computed",
        "settlement_computed",
        "lateral_spreading_computed",
        "wave_induced_liquefaction_computed",
        "earthquake_liquefaction_triggering_computed",
        "pipeline_response_computed",
    ):
        assert flags[key] is False, key
    forbidden_defs = re.compile(
        r"def\s+\w*(csr|crr|factor_of_safety|liquefaction_probab|lpi|settlement|lateral_spread|"
        r"pore_pressure_ratio|excess_pore|biot|magnitude_scaling|k_?sigma|r_d\b|stress_reduction)\w*\s*\(",
        re.IGNORECASE,
    )
    forbidden_numbers = re.compile(r"0\.65\s*\*|/\s*9\.81|a_max\s*/\s*g")
    for module in (cpt_profile, cpt_readiness, evidence_build, cpt_report, cpt_adapter, provider):
        source = inspect.getsource(module)
        assert not forbidden_defs.search(source), module.__name__
        assert not forbidden_numbers.search(source), module.__name__
    # The provider records the source's own area ratio but never applies it.
    canonical_source = inspect.getsource(provider.build_canonical)
    assert "SOURCE_CONE_AREA_RATIO *" not in canonical_source
    assert "* SOURCE_CONE_AREA_RATIO" not in canonical_source


# --- 41-44: earthquake readiness itemizes missing PGA, magnitude, stress state; wave separate -----


def test_41_43_earthquake_readiness_reports_missing_pga_magnitude_and_stress_state():
    facts = cpt_readiness.CptEvidenceFacts(
        source_package_resolved=True,
        source_checksum_recorded=True,
        machine_readable_profile_available=True,
        canonical_profile_created=True,
        row_count=5,
        test_count=1,
        depth_reference=contract.DEPTH_BELOW_SEABED,
        depth_bsf_available=True,
        channels_present=(contract.QC_MPA, contract.FS_KPA, contract.U2_KPA),
        cone_area_ratio_source_stated=True,
    )
    liq = cpt_readiness.assess_liquefaction_input_readiness(facts)
    eq = liq["earthquake_induced"]
    assert eq["status"] == contract.NOT_EVALUABLE
    assert eq["status_code"] == contract.EARTHQUAKE_LIQUEFACTION_TRIGGERING_NOT_EVALUABLE
    assert eq["required_evidence"]["earthquake_pga_a_max"] == contract.NOT_AVAILABLE
    assert eq["required_evidence"]["earthquake_magnitude"] == contract.NOT_AVAILABLE
    assert eq["required_evidence"]["soil_unit_weight_or_stress_state"] == contract.NOT_AVAILABLE
    assert eq["required_evidence"]["vertical_effective_stress_basis"] == contract.NOT_AVAILABLE
    assert eq["required_evidence"]["stress_reduction_factor_method"] == contract.NOT_AUTHORIZED
    assert eq["required_evidence"]["qc"] == contract.AVAILABLE
    assert eq["required_evidence"]["qt"] == contract.NOT_AVAILABLE
    assert eq["required_evidence"]["cone_area_ratio"] == "AVAILABLE_SOURCE_STATED_NOT_APPLIED"
    for item in ("earthquake_pga_a_max", "earthquake_magnitude", "vertical_total_stress_basis"):
        assert item in eq["missing_or_unauthorized"]
    assert eq["method_authority"] == "NOT_GRANTED_IN_MAR_032"


def test_44_wave_liquefaction_readiness_is_a_separate_block():
    liq = cpt_readiness.assess_liquefaction_input_readiness(cpt_readiness.CptEvidenceFacts())
    assert liq["mechanisms_kept_separate"] is True
    wave = liq["wave_current_induced"]
    assert wave["mechanism"] == contract.WAVE_CURRENT_INDUCED_SEABED_LIQUEFACTION
    assert liq["earthquake_induced"]["mechanism"] == contract.EARTHQUAKE_INDUCED_LIQUEFACTION
    assert wave["status_code"] == contract.WAVE_INDUCED_LIQUEFACTION_NOT_EVALUABLE
    assert wave["WAVE_INDUCED_LIQUEFACTION_MODELLED"] is False
    assert set(wave["required_evidence"]) == set(contract.WAVE_REQUIRED_EVIDENCE)
    assert "wave_forcing" in wave["missing"] and "soil_hydraulic_properties" in wave["missing"]
    # Even with every wave evidence flag present, MAR-032 still does not model it.
    full = cpt_readiness.CptEvidenceFacts(
        machine_readable_profile_available=True,
        canonical_profile_created=True,
        wave_forcing_available=True,
        water_depth_available=True,
        soil_hydraulic_properties_available=True,
        soil_compressibility_stiffness_available=True,
        initial_effective_stress_state_available=True,
        pore_pressure_response_parameters_available=True,
    )
    wave_full = cpt_readiness.assess_liquefaction_input_readiness(full)["wave_current_induced"]
    assert wave_full["missing"] == []
    assert wave_full["status"] == contract.NOT_EVALUABLE
    assert wave_full["WAVE_INDUCED_LIQUEFACTION_MODELLED"] is False


# --- 45-47: file existence != readiness; evidence role preserved; interpretation != measured ------


def _project_manifest(tmp_path: Path, filename: str, role: str = "MEASURED") -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""project:
  id: syn
  name: Synthetic
  working_crs: EPSG:32631
assets:
  - asset_id: cpt_001
    category: CPT
    evidence_role: {role}
    path: ./{filename}
    provenance:
      source_name: {SYNTHETIC_TAG}
""",
        encoding="utf-8",
    )
    return manifest_path


def _register(manifest_path: Path) -> project_registry.AssetRegistrationResult:
    m, mdir = project_manifest.load_project_manifest(manifest_path)
    return project_registry.register_asset(m.assets[0], manifest_dir=mdir, working_crs="EPSG:32631")


def test_45_cpt_file_existence_alone_does_not_imply_readiness(tmp_path: Path):
    (tmp_path / "cpt.txt").write_text("stub", encoding="utf-8")
    result = _register(_project_manifest(tmp_path, "cpt.txt"))
    reg = result.registration
    assert reg.registration_status == project_registry.REGISTERED
    assert reg.readiness_status_intrinsic == contract.NOT_READY
    assert reg.readiness_status_effective == contract.NOT_READY
    assert reg.readiness_status != categories.REGISTERED_READINESS_NOT_IMPLEMENTED
    assert reg.readiness_status != "READY"
    assert reg.readiness_result is not None
    axes = {a["axis"]: a for a in reg.readiness_result["axes"]}
    assert axes[contract.DIGITAL_PROFILE]["status"] == contract.NOT_AVAILABLE
    assert categories.CPT in categories.CATEGORIES_WITH_READINESS_ADAPTERS
    # A documentary PDF registered as CPT is also never READY.
    (tmp_path / "log.pdf").write_bytes(_fake_pdf_bytes())
    pdf_result = _register(_project_manifest(tmp_path, "log.pdf"))
    assert pdf_result.registration.readiness_status == contract.NOT_READY
    pdf_axes = {a["axis"]: a for a in pdf_result.registration.readiness_result["axes"]}
    assert pdf_axes[contract.DIGITAL_PROFILE]["status"] == contract.NOT_READY
    assert any("DOCUMENTARY" in r for r in pdf_axes[contract.DIGITAL_PROFILE]["blocking_reasons"])
    assert pdf_result.registration.observed_facts["is_documentary"] is True


def test_45b_canonical_parquet_asset_delegates_to_generic_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    build = cpt_profile.build_canonical_measurements(
        _simple_observations(),
        source_id=SYNTHETIC_TAG,
        test_id="SYN-001",
        depth=_depth(),
        channels=_channels(),
        observation_index_column="idx",
    )
    parquet = tmp_path / "cpt_measurements.parquet"
    build.measurements.to_parquet(parquet, index=False)
    result = _register(_project_manifest(tmp_path, "cpt_measurements.parquet"))
    reg = result.registration
    assert reg.readiness_status_intrinsic == contract.READY_WITH_LIMITATIONS
    axes = {a["axis"]: a for a in reg.readiness_result["axes"]}
    assert axes[contract.DIGITAL_PROFILE]["status"] == contract.READY
    assert axes[contract.DEPTH_REFERENCE]["status"] == contract.READY
    assert axes[contract.SPATIAL_REFERENCE]["status"] == contract.NOT_AVAILABLE
    assert reg.observed_facts["record_count"] == 4
    # Explicit delegation: the registry calls the generic readiness function, not a copy of it.
    facts_direct, _ = cpt_adapter.inspect_cpt_asset(parquet)
    direct = cpt_readiness.assess_cpt_readiness(facts_direct)
    assert direct.to_dict() == reg.readiness_result
    calls: list[cpt_readiness.CptEvidenceFacts] = []
    real = cpt_readiness.assess_cpt_readiness

    def spy(facts: cpt_readiness.CptEvidenceFacts) -> cpt_readiness.CptReadinessResult:
        calls.append(facts)
        return real(facts)

    monkeypatch.setattr(project_registry.cpt_readiness, "assess_cpt_readiness", spy)
    _register(_project_manifest(tmp_path, "cpt_measurements.parquet"))
    assert len(calls) == 1
    # A parquet that is not a canonical CPT table is never READY either.
    pd.DataFrame({"a": [1, 2]}).to_parquet(tmp_path / "other.parquet", index=False)
    other = _register(_project_manifest(tmp_path, "other.parquet"))
    assert other.registration.readiness_status == contract.NOT_READY
    assert "canonical_columns_missing" in other.registration.observed_facts


def test_46_evidence_role_preserved_and_never_inferred(tmp_path: Path):
    (tmp_path / "cpt.txt").write_text("stub", encoding="utf-8")
    for role in ("MEASURED", "SOURCE_INTERPRETED", "DERIVED"):
        reg = _register(_project_manifest(tmp_path, "cpt.txt", role=role)).registration
        assert reg.evidence_role == role
    inv_roles = set(contract.CANDIDATE_ROLES)
    assert inv_roles.isdisjoint(categories.EVIDENCE_ROLES)  # inventory roles != evidence roles


def test_47_source_interpretation_cannot_become_measured_data(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write_geo_csv(pkg / "CPT-SYN1.csv", "CPT-SYN1", SYN_ROWS_A)
    canonical = provider.build_canonical(pkg, source_id=SYNTHETIC_TAG)
    df = canonical.measurements
    # The source REPORT states qt = qc + (1 - a) u with a = 0.75 -- an interpretation applied in
    # the source's own plots. It is recorded, never applied: qt stays null, qc stays measured.
    assert df[contract.QT_MPA].isna().all()
    np.testing.assert_allclose(df[contract.QC_MPA], [r[2] for r in SYN_ROWS_A])
    assert canonical.source_semantics["cone_area_ratio_source_stated"] == 0.75
    assert canonical.source_semantics["cone_area_ratio_applied"] is False
    # Literature (Le et al. 2014) is context only, and no reference is 'implemented'.
    le = next(r for r in contract.REFERENCES if r["key"] == "le_2014")
    assert le["role"] == contract.SOURCE_INTERPRETED_LITERATURE_CONTEXT
    assert all(r["implemented_in_mar_032"] is False for r in contract.REFERENCES)
    # The header 'Depth' line is preserved raw with UNRESOLVED semantics, not turned into data.
    assert canonical.tests["header_depth_raw"].tolist() == ["15.70"]
    assert (
        canonical.tests["header_depth_semantics"]
        .iloc[0]
        .startswith("SOURCE_HEADER_FIELD_UNRESOLVED")
    )
    assert (
        canonical.tests[contract.LOCATION_ID].isna().all()
    )  # 86 locations declared, none inferred


# --- 48-49: no accepted protected science touched by this package ---------------------------------


def test_48_49_geotechnical_layer_is_decoupled_from_protected_engines():
    protected = (
        "marine_engine.terrain",
        "marine_engine.morphology",
        "marine_engine.metocean",
        "marine_engine.sediment",
        "marine_engine.change",
        "marine_engine.scour",
        "marine_engine.burial",
        "marine_engine.freespan",
        "marine_engine.slope_stability",
    )
    for module in (
        contract,
        cpt_inventory,
        cpt_profile,
        cpt_readiness,
        evidence_build,
        cpt_report,
        cpt_manifest,
        cpt_adapter,
        provider,
    ):
        source = inspect.getsource(module)
        for name in protected:
            assert name not in source, f"{module.__name__} references {name}"


# --- 50 + end-to-end: deterministic build, real-shape outputs, CLI --------------------------------


def test_50_identical_input_gives_identical_output(tmp_path: Path, fake_http: list[str]):
    manifest = _manifest(tmp_path)
    first = evidence_build.run_cpt_evidence_build(manifest)
    second = evidence_build.run_cpt_evidence_build(manifest)  # cache hit, same bytes
    assert second.acquisitions[0]["already_cached"] is True
    pd.testing.assert_frame_equal(first.measurements, second.measurements)
    pd.testing.assert_frame_equal(first.inventory, second.inventory)
    assert first.qa == second.qa
    assert first.cpt_readiness.to_dict() == second.cpt_readiness.to_dict()
    assert first.liquefaction_readiness == second.liquefaction_readiness
    assert first.measurements.to_csv(index=False) == second.measurements.to_csv(index=False)


def test_end_to_end_build_outputs_and_semantics(tmp_path: Path, fake_http: list[str]):
    manifest = _manifest(tmp_path)
    result = evidence_build.run_cpt_evidence_build(manifest)
    outputs = result.outputs
    for name in (
        "acquisition_manifest",
        "source_file_inventory",
        "cpt_source_evidence",
        "cpt_measurements",
        "cpt_tests",
        "cpt_locations",
        "cpt_metadata",
        "cpt_readiness",
        "liquefaction_readiness",
    ):
        assert name in outputs and outputs[name].exists(), name

    measurements = pd.read_parquet(outputs["cpt_measurements"])
    assert len(measurements) == len(SYN_ROWS_A) + len(SYN_ROWS_B)
    assert sorted(measurements[contract.TEST_ID].unique()) == ["CPT-SYN1", "CPT-SYN2"]
    assert measurements[contract.QT_MPA].isna().all()
    np.testing.assert_allclose(
        measurements.loc[measurements[contract.TEST_ID] == "CPT-SYN1", contract.FS_KPA],
        [r[3] * 1000.0 for r in SYN_ROWS_A],
    )
    np.testing.assert_allclose(
        measurements.loc[measurements[contract.TEST_ID] == "CPT-SYN1", contract.U2_KPA],
        [r[4] * 1000.0 for r in SYN_ROWS_A],
    )
    assert measurements[contract.DEPTH_REFERENCE_FIELD].unique().tolist() == [
        contract.DEPTH_BELOW_SEABED
    ]
    assert "raw__Incl" in measurements.columns and "raw__Time" in measurements.columns
    assert result.qa["negative_depth_row_count"] == 2  # preserved, never removed

    metadata = json.loads(outputs["cpt_metadata"].read_text(encoding="utf-8"))
    assert metadata["source_declared_test_count"] == 101
    assert metadata["source_declared_location_count"] == 86
    assert metadata["machine_readable_cpt_profile_available"] is True
    assert metadata["documentary_cpt_evidence_available"] is True
    assert metadata["CRR_computed"] is False and metadata["CSR_computed"] is False
    assert metadata["depth_reference"] == contract.DEPTH_BELOW_SEABED
    assert metadata["source_units"]["Depth"] == "m."
    assert metadata["normalized_units"][contract.FS_KPA] == "kPa"
    assert metadata["coordinate_reference"]["resolved_crs"] == "EPSG:32631"
    assert metadata["roles"]["measurements"] == contract.MEASURED_CPT_CPTU_PROFILE
    labelled = json.dumps(metadata["roles"]) + json.dumps(list(metadata))
    assert not any(label in labelled for label in contract.PROHIBITED_PRODUCT_LABELS)
    assert "prohibited_product_labels" not in metadata
    assert any(u.startswith("raw__Time") for u in metadata["unresolved"])  # Time unit not stated

    readiness = json.loads(outputs["cpt_readiness"].read_text(encoding="utf-8"))
    assert readiness["cpt_evidence_readiness"]["status"] == contract.READY_WITH_LIMITATIONS
    assert readiness["DIGITAL_NUMERIC_CPT_PROFILE_READY"] is True
    axes = {a["axis"]: a["status"] for a in readiness["cpt_evidence_readiness"]["axes"]}
    assert axes[contract.DIGITAL_PROFILE] == contract.READY
    assert axes[contract.SPATIAL_REFERENCE] == contract.READY
    assert axes[contract.IDENTITY] == contract.READY_WITH_LIMITATIONS  # 2 parsed vs 101 declared

    liq = json.loads(outputs["liquefaction_readiness"].read_text(encoding="utf-8"))
    assert liq["status"] == contract.NOT_EVALUABLE
    assert liq["pl854"]["CPT_GEOTECHNICAL_PROFILE"] == contract.NOT_AVAILABLE
    assert liq["pl854"]["EARTHQUAKE_LIQUEFACTION_TRIGGERING"] == contract.NOT_EVALUABLE
    assert liq["pl854"]["WAVE_INDUCED_LIQUEFACTION"] == contract.NOT_EVALUABLE

    inventory = pd.read_parquet(outputs["source_file_inventory"])
    reports = inventory[inventory["package"] == "reports"]
    assert reports["is_documentary"].sum() == 3
    assert set(reports.loc[reports["is_documentary"], "candidate_role"]) == {
        contract.ROLE_FACTUAL_REPORT
    }
    assert (inventory["candidate_role"] == contract.ROLE_MACHINE_READABLE_CPT_PROFILE).sum() == 2

    import geopandas as gpd

    gdf = gpd.read_file(outputs["cpt_locations"], layer="cpt_locations")
    assert len(gdf) == 2 and gdf.crs is not None and gdf.crs.to_epsg() == 32631
    assert gdf.geometry.x.tolist() == [1000.5, 1000.5]


def test_documentary_only_package_creates_readiness_but_no_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fake_get(url: str, timeout: float = 0.0) -> _FakeResponse:
        return _FakeResponse(_reports_package_zip(), url)

    monkeypatch.setattr(provider.requests, "get", fake_get)
    manifest = _manifest(tmp_path, reports=False)
    result = evidence_build.run_cpt_evidence_build(manifest)
    assert result.measurements is None
    assert "cpt_measurements" not in result.outputs
    assert "cpt_locations" not in result.outputs
    assert result.outputs["cpt_readiness"].exists()
    assert result.outputs["liquefaction_readiness"].exists()
    readiness = json.loads(result.outputs["cpt_readiness"].read_text(encoding="utf-8"))
    assert readiness["DOCUMENTARY_CPT_EVIDENCE_AVAILABLE"] is True
    assert readiness["DIGITAL_NUMERIC_CPT_PROFILE_READY"] is False
    assert readiness["cpt_evidence_readiness"]["status"] == contract.NOT_READY
    evidence = json.loads(result.outputs["cpt_source_evidence"].read_text(encoding="utf-8"))
    assert evidence["ocr_or_chart_digitization_performed"] is False
    assert evidence["machine_readable_files"] == []


def test_no_declared_crs_writes_no_locations_layer(tmp_path: Path, fake_http: list[str]):
    result = evidence_build.run_cpt_evidence_build(_manifest(tmp_path, crs=None))
    assert "cpt_locations" not in result.outputs
    assert "cpt_tests" in result.outputs  # raw coordinates still preserved in the tests table
    assert result.facts.coordinates_available is True and result.facts.crs_resolved is False
    assert result.cpt_readiness.axis(contract.SPATIAL_REFERENCE).status == (
        contract.READY_WITH_LIMITATIONS
    )


def test_conflicting_declared_crs_blocks_and_writes_no_layer(tmp_path: Path, fake_http: list[str]):
    result = evidence_build.run_cpt_evidence_build(_manifest(tmp_path, crs="EPSG:32632"))
    assert "cpt_locations" not in result.outputs
    assert result.cpt_readiness.cpt_profile_status == contract.NOT_READY
    assert result.cpt_readiness.axis(contract.SPATIAL_REFERENCE).status == contract.NOT_READY


def test_strict_geo_reader_rejects_non_conforming_structure(tmp_path: Path):
    good = _write_geo_csv(tmp_path / "CPT-OK.csv", "CPT-OK", SYN_ROWS_A)
    parsed = provider.parse_geo_cpt_csv(good)
    assert parsed.columns == provider.GEO_COLUMNS
    assert len(parsed.observations) == len(SYN_ROWS_A)
    bad_units = _geo_csv_text("CPT-BAD", SYN_ROWS_A).replace("#;m.;MPa;MPa;MPa", "#;ft;MPa;MPa;MPa")
    (tmp_path / "CPT-BAD.csv").write_bytes(bad_units.encode("latin-1"))
    with pytest.raises(provider.GeoCptFormatError, match="unit row"):
        provider.parse_geo_cpt_csv(tmp_path / "CPT-BAD.csv")
    bad_row = _geo_csv_text("CPT-BAD2", SYN_ROWS_A) + "1;2;3;\r\n"
    (tmp_path / "CPT-BAD2.csv").write_bytes(bad_row.encode("latin-1"))
    with pytest.raises(provider.GeoCptFormatError, match="fields"):
        provider.parse_geo_cpt_csv(tmp_path / "CPT-BAD2.csv")
    # The probe never accepts a PDF even when named .csv, and never accepts a plain numeric CSV
    # lacking the GEO header block (its semantics/units would be unknown).
    (tmp_path / "plain.csv").write_text("depth,qc\n0,1\n", encoding="utf-8")
    assert provider.probe_geo_cpt_csv(tmp_path / "plain.csv", b"depth,qc\n0,1\n") is False


def test_readiness_transitions_and_vocabulary():
    empty = cpt_readiness.assess_cpt_readiness(cpt_readiness.CptEvidenceFacts())
    assert empty.cpt_profile_status == contract.NOT_READY
    assert empty.axis(contract.SOURCE_PACKAGE).status == contract.NOT_AVAILABLE
    assert empty.axis(contract.DIGITAL_PROFILE).status == contract.NOT_AVAILABLE
    ready = cpt_readiness.assess_cpt_readiness(
        cpt_readiness.CptEvidenceFacts(
            source_package_resolved=True,
            source_checksum_recorded=True,
            machine_readable_profile_available=True,
            canonical_profile_created=True,
            row_count=3,
            test_count=1,
            depth_reference=contract.DEPTH_BELOW_SEABED,
            depth_bsf_available=True,
            channels_present=tuple(contract.CANONICAL_MEASUREMENT_FIELDS),
            coordinates_available=True,
            crs_resolved=True,
        )
    )
    assert ready.cpt_profile_status == contract.READY
    for axis in ready.axes:
        assert axis.status in contract.READINESS_STATUSES
    payload = json.dumps(ready.to_dict())
    assert not re.search(r'"score"|"percent"|"confidence"', payload)
    assert all(s in contract.READINESS_STATUSES for s in (a.status for a in empty.axes))
    assert tuple(a.axis for a in ready.axes) == contract.CPT_PROFILE_AXES


def test_manifest_schema_is_strict(tmp_path: Path):
    path = tmp_path / "m.yaml"
    path.write_text(
        "evidence_id: x\nsource:\n  provider: sheringham_shoal_2008_cptu\npaths:\n"
        "  raw_dir: a\n  interim_dir: b\n  processed_dir: c\nunexpected: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        cpt_manifest.load_cpt_evidence_manifest(path)
    ok = Path("configs/geotechnical/sheringham_shoal_2008_cptu.yaml")
    manifest = cpt_manifest.load_cpt_evidence_manifest(ok)
    assert manifest.source.provider == provider.PROVIDER_ID
    assert manifest.declared.coordinate_crs == "EPSG:32631"
    assert manifest.source.acquire_reports_package is True
    with pytest.raises(ValueError, match="unknown CPT evidence provider"):
        evidence_build.run_cpt_evidence_build(
            cpt_manifest.CptEvidenceManifest.model_validate(
                {
                    "evidence_id": "x",
                    "source": {"provider": "nope"},
                    "paths": {"raw_dir": "a", "interim_dir": "b", "processed_dir": "c"},
                }
            )
        )


def test_cli_build_cpt_evidence_poc(
    tmp_path: Path, fake_http: list[str], capsys: pytest.CaptureFixture[str]
):
    manifest_path = tmp_path / "cpt.yaml"
    manifest_path.write_text(
        f"""evidence_id: syn_cli
source:
  provider: {provider.PROVIDER_ID}
  acquire_reports_package: true
paths:
  raw_dir: {(tmp_path / "raw").as_posix()}
  interim_dir: {(tmp_path / "interim").as_posix()}
  processed_dir: {(tmp_path / "processed").as_posix()}
declared:
  coordinate_crs: EPSG:32631
""",
        encoding="utf-8",
    )
    exit_code = cli.main(["build-cpt-evidence-poc", str(manifest_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "DOES CPT FILE EXISTENCE ALONE IMPLY LIQUEFACTION READINESS? NO" in out
    assert "DOES THE PACKAGE CONTAIN MACHINE-READABLE NUMERIC CPT/CPTU PROFILES? YES" in out
    assert "CAN QC SILENTLY BECOME QT? NO" in out
    assert "IS EARTHQUAKE CSR COMPUTED IN MAR-032? NO" in out
    assert "ARE EARTHQUAKE- AND WAVE-INDUCED LIQUEFACTION KEPT SEPARATE? YES" in out
    assert (tmp_path / "processed" / "cpt_measurements.parquet").exists()
    # Second run is fully offline (cache) and idempotent.
    assert cli.main(["build-cpt-evidence-poc", str(manifest_path)]) == 0
    assert len(fake_http) == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("evidence_id: x\n", encoding="utf-8")
    assert cli.main(["build-cpt-evidence-poc", str(bad)]) == 1
    assert "rejected" in capsys.readouterr().out

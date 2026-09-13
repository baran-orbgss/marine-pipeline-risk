"""Tests for api.staging: path containment for uploads, CSV coordinate columns are never
guessed, and promoting a session only ever writes a manifest pointing at the existing bytes --
it never copies, moves, or rewrites the uploaded file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from api import staging
from api.errors import UnsafePathRequestError
from api.models import AssetDeclaration
from rasterio.transform import from_origin

from marine_engine.project.manifest import load_project_manifest


def _raster_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    data = np.linspace(0.0, 50.0, 100, dtype="float32").reshape(10, 10)
    transform = from_origin(500000, 5900000, 10, 10)
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return buf.getvalue()


def test_save_upload_inspects_and_reports_spatial_metadata(api_sandbox: Path) -> None:
    session = staging.create_session()

    staged = staging.save_upload(session.session_id, "bathy.tif", _raster_bytes())

    assert staged.inspection.kind == "raster"
    assert staged.inspection.observed_crs is not None
    absolute = api_sandbox / staged.relative_path
    assert absolute.is_file()


def test_save_upload_rejects_unsupported_extension(api_sandbox: Path) -> None:
    session = staging.create_session()

    with pytest.raises(UnsafePathRequestError):
        staging.save_upload(session.session_id, "malware.exe", b"not spatial data")


def test_save_upload_never_escapes_the_session_directory_for_a_traversal_filename(
    api_sandbox: Path,
) -> None:
    session = staging.create_session()

    # `safe_filename` strips this down to a bare "passwd.tif" rather than raising -- the property
    # that actually matters is containment: the saved file must land inside the session's own
    # staging directory, never anywhere its traversal-shaped name might otherwise imply.
    staged = staging.save_upload(session.session_id, "../../etc/passwd.tif", _raster_bytes())

    absolute = (api_sandbox / staged.relative_path).resolve()
    session_dir = (api_sandbox / "data" / "staging" / session.session_id).resolve()
    assert absolute.is_relative_to(session_dir)


def test_csv_upload_requires_explicit_coordinate_declaration(api_sandbox: Path) -> None:
    session = staging.create_session()
    staged = staging.save_upload(
        session.session_id, "samples.csv", b"easting_m,northing_m,grain_mm\n1,2,0.3\n"
    )

    assert staged.inspection.coordinate_columns_declared is False

    updated = staging.declare_coordinate_columns(
        session.session_id,
        staged.file_id,
        x_column="easting_m",
        y_column="northing_m",
        crs="EPSG:32631",
    )

    assert updated.inspection.coordinate_columns_declared is True
    assert updated.inspection.feature_count == 1


def test_promote_writes_manifest_pointing_at_existing_bytes_without_copying(
    api_sandbox: Path,
) -> None:
    session = staging.create_session()
    original_bytes = _raster_bytes()
    staged = staging.save_upload(session.session_id, "bathy.tif", original_bytes)
    original_absolute = api_sandbox / staged.relative_path
    original_mtime = original_absolute.stat().st_mtime_ns

    manifest_path = staging.promote_session(
        session.session_id,
        project_id="My New Project",
        display_name="My New Project",
        working_crs="EPSG:32631",
        asset_declarations={
            staged.file_id: AssetDeclaration(category="BATHYMETRY_RASTER", evidence_role="DERIVED")
        },
    )

    # source bytes untouched -- promotion only ever writes the new manifest file.
    assert original_absolute.read_bytes() == original_bytes
    assert original_absolute.stat().st_mtime_ns == original_mtime

    manifest, manifest_dir = load_project_manifest(manifest_path)
    assert manifest.project.id == "my new project"
    assert len(manifest.assets) == 1
    from marine_engine.project.manifest import resolve_asset_path

    assert resolve_asset_path(manifest.assets[0], manifest_dir) == original_absolute.resolve()


def test_promote_defaults_unclassified_assets_to_other_derived(api_sandbox: Path) -> None:
    session = staging.create_session()
    staging.save_upload(session.session_id, "bathy.tif", _raster_bytes())

    manifest_path = staging.promote_session(
        session.session_id,
        project_id="proj",
        display_name="Proj",
        working_crs="EPSG:32631",
        asset_declarations={},
    )

    manifest, _ = load_project_manifest(manifest_path)
    assert manifest.assets[0].category == "OTHER"
    assert manifest.assets[0].evidence_role == "DERIVED"

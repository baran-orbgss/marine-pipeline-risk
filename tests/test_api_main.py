"""Endpoint-level wiring tests: status codes and response shapes for api.main, using the
FastAPI TestClient against an isolated sandbox (never the real repository data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import yaml
from api.main import app
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_project_returns_404(api_sandbox: Path) -> None:
    response = client.get("/api/projects/does-not-exist")
    assert response.status_code == 404


def test_unknown_layer_returns_404(api_sandbox: Path) -> None:
    (api_sandbox / "configs" / "pl854.yaml").write_text(
        yaml.safe_dump(
            {"study": {"id": "pl854", "name": "PL854"}, "crs": {"horizontal": "EPSG:32631"}}
        ),
        encoding="utf-8",
    )
    (api_sandbox / "data" / "processed" / "pl854").mkdir(parents=True)

    response = client.get("/api/projects/pl854/layers/pl854:asset:does-not-exist")
    assert response.status_code == 404


def test_project_list_and_layer_round_trip(api_sandbox: Path) -> None:
    raster_path = api_sandbox / "data" / "processed" / "pl854" / "bathymetry" / "baseline.tif"
    raster_path.parent.mkdir(parents=True)
    data = np.linspace(0.0, 40.0, 16, dtype="float32").reshape(4, 4)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500000, 5900000, 25, 25),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    (api_sandbox / "configs" / "pl854.yaml").write_text(
        yaml.safe_dump(
            {"study": {"id": "pl854", "name": "PL854"}, "crs": {"horizontal": "EPSG:32631"}}
        ),
        encoding="utf-8",
    )
    (api_sandbox / "configs" / "project_manifests" / "pl854.yaml").write_text(
        yaml.safe_dump(
            {
                "project": {"id": "pl854", "name": "PL854", "working_crs": "EPSG:32631"},
                "assets": [
                    {
                        "asset_id": "bathy",
                        "category": "BATHYMETRY_RASTER",
                        "evidence_role": "DERIVED",
                        "path": "../../data/processed/pl854/bathymetry/baseline.tif",
                        "provenance": {"source_name": "test"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    projects_response = client.get("/api/projects")
    assert projects_response.status_code == 200
    assert projects_response.json()["projects"][0]["project_id"] == "pl854"

    detail_response = client.get("/api/projects/pl854")
    assert detail_response.status_code == 200
    assert detail_response.json()["layer_count"] == 1

    layers_response = client.get("/api/projects/pl854/layers")
    layer = layers_response.json()["layers"][0]

    # Use the server-provided URL rather than rebuilding one from the raw `layer_id`: the URL's
    # `{layer_id}` path segment is an opaque encoded token, not the raw id itself (see
    # api.layers._url_segment/decode_layer_id).
    tilejson_response = client.get(layer["tilejson_url"])
    assert tilejson_response.status_code == 200
    assert tilejson_response.json()["tilejson"] == "2.2.0"

    capabilities_response = client.get("/api/projects/pl854/capabilities")
    assert capabilities_response.status_code == 200
    assert len(capabilities_response.json()) > 0


def test_tile_and_features_routes_work_for_a_layer_id_with_nested_path_segments(
    api_sandbox: Path,
) -> None:
    # A regression guard: an "output" layer's id embeds its relative output path verbatim (e.g.
    # "sheringham_shoal_2020:output:terrain/aspect.tif"), which contains '/' and ':'. Percent-
    # encoding that into the URL is not enough on its own -- uvicorn percent-decodes
    # `scope["path"]` (a %2F back into a literal '/') before Starlette matches routes, so the
    # request falls between routes entirely (no `{layer_id}` pattern matches an embedded '/') and,
    # in this repo's static-file-fallback setup, was observed reaching the filesystem and raising
    # an OSError on Windows rather than hitting the intended handler at all. The fix
    # (`api.layers._url_segment`) encodes layer_id as an opaque base64url token containing no '/'
    # at any decoding layer, which also means this test can now exercise the real HTTP round trip
    # end-to-end through TestClient (a raw '/' is what TestClient's own URL handling didn't
    # reliably preserve; an opaque token never has one to begin with).
    output_root = api_sandbox / "data" / "processed" / "sheringham_shoal_2020"
    raster_path = output_root / "terrain" / "aspect.tif"
    raster_path.parent.mkdir(parents=True)
    data = np.linspace(0.0, 359.0, 16, dtype="float32").reshape(4, 4)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500000, 5900000, 25, 25),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    (api_sandbox / "configs" / "project_manifests" / "sheringham_shoal_2020.yaml").write_text(
        yaml.safe_dump(
            {
                "project": {
                    "id": "sheringham_shoal_2020",
                    "name": "Sheringham",
                    "working_crs": "EPSG:32631",
                },
                "assets": [],
            }
        ),
        encoding="utf-8",
    )

    layers_response = client.get("/api/projects/sheringham_shoal_2020/layers")
    layer = layers_response.json()["layers"][0]
    assert "/" in layer["layer_id"]  # the raw id genuinely embeds a nested output path

    tilejson_response = client.get(layer["tilejson_url"])
    assert tilejson_response.status_code == 200
    assert tilejson_response.json()["tilejson"] == "2.2.0"

    tile_url = layer["tile_url_template"].format(z=0, x=0, y=0)
    tile_response = client.get(tile_url)
    assert tile_response.status_code == 200
    assert tile_response.headers["content-type"] == "image/png"


def test_unknown_layer_token_returns_404_not_500(api_sandbox: Path) -> None:
    (api_sandbox / "configs" / "project_manifests" / "pl854.yaml").write_text(
        yaml.safe_dump(
            {
                "project": {"id": "pl854", "name": "PL854", "working_crs": "EPSG:32631"},
                "assets": [],
            }
        ),
        encoding="utf-8",
    )
    (api_sandbox / "data" / "processed" / "pl854").mkdir(parents=True)

    response = client.get("/api/projects/pl854/layers/not-a-valid-base64url-token!!!")
    assert response.status_code == 404


def test_run_capability_rejects_non_allowlisted_capability(api_sandbox: Path) -> None:
    (api_sandbox / "configs" / "pl854.yaml").write_text(
        yaml.safe_dump(
            {"study": {"id": "pl854", "name": "PL854"}, "crs": {"horizontal": "EPSG:32631"}}
        ),
        encoding="utf-8",
    )
    (api_sandbox / "data" / "processed" / "pl854").mkdir(parents=True)

    response = client.post(
        "/api/projects/pl854/capabilities/not-a-real-one/run", json={"allow_network": False}
    )
    assert response.status_code == 403


def test_staging_session_upload_and_promote_round_trip(api_sandbox: Path) -> None:
    session_response = client.post("/api/staging/sessions")
    session_id = session_response.json()["session_id"]

    raster_path = api_sandbox / "scratch_upload.tif"
    data = np.zeros((4, 4), dtype="float32")
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500000, 5900000, 1, 1),
    ) as dst:
        dst.write(data, 1)

    with raster_path.open("rb") as fh:
        upload_response = client.post(
            f"/api/staging/sessions/{session_id}/files",
            files={"file": ("upload.tif", fh, "image/tiff")},
        )
    assert upload_response.status_code == 200
    assert upload_response.json()["inspection"]["kind"] == "raster"

    promote_response = client.post(
        f"/api/staging/sessions/{session_id}/promote",
        json={
            "project_id": "temp-project",
            "display_name": "Temp Project",
            "working_crs": "EPSG:32631",
        },
    )
    assert promote_response.status_code == 200

    catalog_response = client.get("/api/projects")
    project_ids = {p["project_id"] for p in catalog_response.json()["projects"]}
    assert "temp-project" in project_ids

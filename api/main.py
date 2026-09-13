"""FastAPI app: routers, CORS, exception handlers, static-serve of the built frontend.

This module only wires together the other `api/*` modules and translates their results into HTTP
responses -- it contains no scientific logic, no science-formula duplication, and no unbounded
command execution (see `api/allowlist.py` / `api/jobs.py` for the only commands this API can run).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from api import allowlist, jobs, layers, projects, settings, staging, tiles
from api import display_specs as ds
from api.discovery import read_vector_any
from api.errors import ApiError, NotFoundError, UnsafePathRequestError, install_exception_handlers
from api.identify import cpt_profile as build_cpt_profile
from api.identify import features_geojson
from api.models import (
    AnalysisCapabilityDescriptor,
    CoordinateColumnsRequest,
    JobRecord,
    JobSubmitRequest,
    LayerCatalog,
    LayerDescriptor,
    ProjectCatalog,
    ProjectSummary,
    PromoteRequest,
    StagedFile,
    StagingSessionRecord,
)
from api.paths import UnsafePathError, safe_join

settings.ensure_runtime_dirs()

app = FastAPI(title="Marine GIS Workspace API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
install_exception_handlers(app)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/projects", response_model=ProjectCatalog)
def list_projects() -> ProjectCatalog:
    return projects.discover_projects()


@app.get("/api/projects/{project_id}", response_model=ProjectSummary)
def get_project(project_id: str) -> ProjectSummary:
    summary = projects.get_project_summary(project_id)
    if summary is None:
        raise NotFoundError(f"unknown project {project_id!r}")
    catalog = layers.build_layer_catalog(project_id)
    return summary.model_copy(
        update={
            "layer_count": len(catalog.layers),
            "extent_wgs84": layers.catalog_extent_wgs84(catalog),
        }
    )


@app.post("/api/projects/{project_id}/refresh", response_model=ProjectSummary)
def refresh_project(project_id: str) -> ProjectSummary:
    return get_project(project_id)


@app.get("/api/projects/{project_id}/layers", response_model=LayerCatalog)
def list_layers(project_id: str) -> LayerCatalog:
    return layers.build_layer_catalog(project_id)


def _require_layer(project_id: str, layer_id_token: str) -> LayerDescriptor:
    # `layer_id_token` is always the base64url token `layers._url_segment` produced for a
    # LayerDescriptor's own URL fields (see api/layers.py for why a raw layer_id -- which may
    # contain '/' -- can never be a plain `{layer_id}` path parameter). A malformed token (e.g. a
    # hand-typed URL) is a 404, not an unhandled 500.
    try:
        layer_id = layers.decode_layer_id(layer_id_token)
    except ValueError as exc:
        raise NotFoundError(f"unknown layer {layer_id_token!r}") from exc
    layer = layers.get_layer(project_id, layer_id)
    if layer is None:
        raise NotFoundError(f"unknown layer {layer_id!r}")
    return layer


@app.get("/api/projects/{project_id}/layers/{layer_id}", response_model=LayerDescriptor)
def get_layer(project_id: str, layer_id: str) -> LayerDescriptor:
    return _require_layer(project_id, layer_id)


@app.get("/api/projects/{project_id}/layers/{layer_id}/tiles/{z}/{x}/{y}.png")
def get_tile(project_id: str, layer_id: str, z: int, x: int, y: int) -> Response:
    layer = _require_layer(project_id, layer_id)
    if layer.layer_type != "raster":
        raise ApiError("layer is not a raster layer")
    absolute_path = settings.REPO_ROOT / layer.relative_path
    content = tiles.render_tile(layer, absolute_path, z, x, y)
    return Response(content=content, media_type="image/png")


@app.get("/api/projects/{project_id}/layers/{layer_id}/tilejson.json")
def get_tilejson(project_id: str, layer_id: str) -> dict:
    layer = _require_layer(project_id, layer_id)
    if layer.layer_type != "raster" or not layer.tile_url_template:
        raise ApiError("layer is not a raster layer")
    absolute_path = settings.REPO_ROOT / layer.relative_path
    return tiles.tilejson_document(layer, absolute_path, layer.tile_url_template)


@app.get("/api/projects/{project_id}/layers/{layer_id}/features")
def get_features(project_id: str, layer_id: str) -> dict:
    layer = _require_layer(project_id, layer_id)
    if layer.layer_type != "vector":
        raise ApiError("layer is not a vector layer")
    absolute_path = settings.REPO_ROOT / layer.relative_path
    return features_geojson(layer, absolute_path)


@app.get("/api/projects/{project_id}/figures", response_model=list[str])
def list_figures(project_id: str) -> list[str]:
    return layers.list_figure_paths(project_id)


@app.get("/api/projects/{project_id}/figures/{figure_path:path}")
def get_figure(project_id: str, figure_path: str) -> FileResponse:
    summary = projects.get_project_summary(project_id)
    if summary is None or summary.output_root is None:
        raise NotFoundError(f"unknown project {project_id!r}")
    output_root = settings.REPO_ROOT / summary.output_root
    try:
        absolute = safe_join(output_root, *figure_path.split("/"))
    except UnsafePathError as exc:
        raise UnsafePathRequestError(str(exc)) from exc
    if absolute.suffix.lower() != ".png" or not absolute.is_file():
        raise NotFoundError(f"unknown figure {figure_path!r}")
    return FileResponse(absolute, media_type="image/png")


def _cpt_measurements_path(layer: LayerDescriptor) -> Path | None:
    layer_path = settings.REPO_ROOT / layer.relative_path
    candidate = layer_path.parent / "cpt_measurements.parquet"
    return candidate if candidate.is_file() else None


@app.get("/api/projects/{project_id}/layers/{layer_id}/cpt-tests/{test_id}/profile")
def get_cpt_profile(project_id: str, layer_id: str, test_id: str) -> dict:
    layer = _require_layer(project_id, layer_id)
    measurements_path = _cpt_measurements_path(layer)
    if measurements_path is None:
        raise NotFoundError("no CPT measurements found alongside this layer")
    profile = build_cpt_profile(measurements_path, test_id)
    if profile is None:
        raise NotFoundError(f"unknown CPT test_id {test_id!r}")
    return profile.model_dump()


@app.get(
    "/api/projects/{project_id}/capabilities",
    response_model=list[AnalysisCapabilityDescriptor],
)
def get_capabilities(project_id: str) -> list[AnalysisCapabilityDescriptor]:
    return allowlist.describe_capabilities(project_id)


@app.post(
    "/api/projects/{project_id}/capabilities/{capability_key}/run",
    response_model=JobRecord,
    status_code=202,
)
def run_capability(project_id: str, capability_key: str, request: JobSubmitRequest) -> JobRecord:
    extra_args = (
        ["--scenario-manifest", request.scenario_manifest_path]
        if request.scenario_manifest_path
        else []
    )
    return jobs.submit_job(
        project_id, capability_key, allow_network=request.allow_network, extra_args=extra_args
    )


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    record = jobs.get_job(job_id)
    if record is None:
        raise NotFoundError(f"unknown job {job_id!r}")
    return record


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str) -> Response:
    if jobs.get_job(job_id) is None:
        raise NotFoundError(f"unknown job {job_id!r}")
    return Response(content=jobs.get_job_log(job_id), media_type="text/plain")


@app.get("/api/jobs", response_model=list[JobRecord])
def list_jobs(project_id: str | None = None) -> list[JobRecord]:
    return jobs.list_jobs(project_id)


@app.post("/api/staging/sessions", response_model=StagingSessionRecord)
def create_staging_session() -> StagingSessionRecord:
    return staging.create_session()


@app.post("/api/staging/sessions/{session_id}/files", response_model=StagedFile)
async def upload_staging_file(session_id: str, file: UploadFile) -> StagedFile:
    content = await file.read()
    return staging.save_upload(session_id, file.filename or "upload", content)


@app.get("/api/staging/sessions/{session_id}/files", response_model=list[StagedFile])
def list_staging_files(session_id: str) -> list[StagedFile]:
    return staging.list_staged_files(session_id)


@app.post(
    "/api/staging/sessions/{session_id}/files/{file_id}/coordinate-columns",
    response_model=StagedFile,
)
def declare_coordinate_columns(
    session_id: str, file_id: str, request: CoordinateColumnsRequest
) -> StagedFile:
    return staging.declare_coordinate_columns(
        session_id, file_id, x_column=request.x_column, y_column=request.y_column, crs=request.crs
    )


def _require_staged_file(session_id: str, file_id: str) -> StagedFile:
    staged = staging.get_staged_file(session_id, file_id)
    if staged is None:
        raise NotFoundError(f"unknown staged file {file_id!r}")
    return staged


@app.get("/api/staging/sessions/{session_id}/files/{file_id}/tiles/{z}/{x}/{y}.png")
def get_staging_tile(session_id: str, file_id: str, z: int, x: int, y: int) -> Response:
    staged = _require_staged_file(session_id, file_id)
    absolute_path = settings.REPO_ROOT / staged.relative_path
    domain = (0.0, 1.0)
    try:
        import rasterio

        with rasterio.open(absolute_path) as src:
            data = src.read(1, masked=True)
            if data.count() > 0:
                domain = (float(data.min()), float(data.max()))
    except Exception:
        pass
    preview_layer = LayerDescriptor(
        layer_id=f"staging:{session_id}:{file_id}",
        project_id=f"staging:{session_id}",
        group="DATA",
        layer_type="raster",
        support_type="AREA_SURFACE",
        role_established=False,
        relative_path=staged.relative_path,
        display=ds.staging_preview_display_spec(domain),
    )
    content = tiles.render_tile(preview_layer, absolute_path, z, x, y)
    return Response(content=content, media_type="image/png")


@app.get("/api/staging/sessions/{session_id}/files/{file_id}/features")
def get_staging_features(session_id: str, file_id: str) -> dict:
    staged = _require_staged_file(session_id, file_id)
    absolute_path = settings.REPO_ROOT / staged.relative_path
    gdf = read_vector_any(absolute_path)
    if gdf.crs is not None:
        gdf = gdf.to_crs(4326)
    return gdf.__geo_interface__


@app.post("/api/staging/sessions/{session_id}/promote")
def promote_session(session_id: str, request: PromoteRequest) -> dict:
    manifest_path = staging.promote_session(
        session_id,
        project_id=request.project_id,
        display_name=request.display_name,
        working_crs=request.working_crs,
        asset_declarations=request.asset_declarations,
    )
    return {"manifest_path": manifest_path.relative_to(settings.REPO_ROOT).as_posix()}


# Serve the built frontend in "production" mode when it exists. Never required for API-only dev
# use (the Vite dev server proxies /api to this app instead); mounted last so it never shadows
# an /api/* route above.
if settings.WEB_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(settings.WEB_DIST_DIR), html=True), name="web")

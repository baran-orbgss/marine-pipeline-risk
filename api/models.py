"""Pydantic request/response schemas for the UI-003 API.

Presentation/transport contracts only. No scientific computation happens here: every status or
value is either echoed verbatim from the engine (readiness, evidence roles) or derived from raw
spatial metadata (CRS, bounds, geometry, pixel size) -- never fabricated, never reinterpreted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SupportType = Literal[
    "AREA_SURFACE",
    "AREA_VECTOR",
    "CORRIDOR",
    "LINEAR_ANALYSIS",
    "LINEAR_ASSET",
    "POINT_EVIDENCE",
    "SOURCE_FOOTPRINT",
    "SUPPORT_NODE",
]

LayerGroup = Literal["DATA", "DERIVED", "ANALYSIS", "EVIDENCE", "ASSETS"]

AvailabilityState = Literal["AVAILABLE", "MISSING_INPUTS", "NOT_APPLICABLE"]

JobState = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]

# The ticket's own vocabulary is 4 states; "Failed" is an addition -- a launcher with no failure
# path isn't usable, and it never leaks stdout/stderr (that stays in the /log endpoint).
SimpleJobStatus = Literal[
    "Preparing data", "Running analysis", "Building map layer", "Ready", "Failed"
]

PaletteKind = Literal["continuous", "diverging", "categorical", "single_color"]


class BoundingBox(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float
    crs: str = "EPSG:4326"


class DeclaredSourceRef(BaseModel):
    kind: Literal["study_config", "project_manifest", "cpt_evidence_manifest", "ad_hoc_manifest"]
    path: str
    declared_id: str


class ProjectSummary(BaseModel):
    project_id: str
    display_name: str
    description: str | None = None
    declared_sources: list[DeclaredSourceRef] = Field(default_factory=list)
    working_crs: str | None = None
    output_root: str | None = None
    has_registered_project: bool = False
    layer_count: int | None = None
    extent_wgs84: BoundingBox | None = None
    is_ad_hoc: bool = False


class UnmatchedOutputDir(BaseModel):
    directory: str
    reason: str


class ProjectCatalog(BaseModel):
    projects: list[ProjectSummary]
    unmatched_output_dirs: list[UnmatchedOutputDir] = Field(default_factory=list)


class PaletteStop(BaseModel):
    value: float
    color: str
    label: str | None = None


class PaletteSpec(BaseModel):
    kind: PaletteKind
    colormap_name: str | None = None
    domain: tuple[float, float] | None = None
    midpoint: float | None = None
    stops: list[PaletteStop] = Field(default_factory=list)
    color: str | None = None  # single_color only


class LegendSpec(BaseModel):
    title: str
    unit: str | None = None
    kind: PaletteKind
    stops: list[PaletteStop] = Field(default_factory=list)
    note: str | None = None


class TooltipField(BaseModel):
    key: str
    label: str
    unit: str | None = None


class DisplaySpec(BaseModel):
    display_name: str
    palette: PaletteSpec
    legend: LegendSpec
    units: str | None = None
    opacity: float = 1.0
    z_index: int = 0
    tooltip_fields: list[TooltipField] = Field(default_factory=list)
    default_visible: bool = True
    scientific_limitations: list[str] = Field(default_factory=list)


class LayerDescriptor(BaseModel):
    layer_id: str
    project_id: str
    group: LayerGroup
    capability_key: str | None = None
    layer_type: Literal["raster", "vector"]
    support_type: SupportType
    semantic_role: str | None = None
    role_established: bool = True
    relative_path: str
    gpkg_layer: str | None = None
    crs_observed: str | None = None
    bounds_native: BoundingBox | None = None
    bounds_wgs84: BoundingBox | None = None
    display: DisplaySpec
    tile_url_template: str | None = None
    tilejson_url: str | None = None
    vector_url: str | None = None
    cpt_profile_url_template: str | None = None
    feature_count: int | None = None
    band_count: int | None = None
    readiness_context: dict | None = None


class LayerCatalog(BaseModel):
    project_id: str
    layers: list[LayerDescriptor]


class SpatialFieldInfo(BaseModel):
    name: str
    dtype: str


class SpatialMetadataInspection(BaseModel):
    kind: Literal["raster", "vector", "tabular_unresolved"]
    observed_crs: str | None = None
    bounds_native: BoundingBox | None = None
    bounds_wgs84: BoundingBox | None = None
    pixel_size_x: float | None = None
    pixel_size_y: float | None = None
    width: int | None = None
    height: int | None = None
    band_count: int | None = None
    nodata: float | None = None
    geometry_type: str | None = None
    feature_count: int | None = None
    fields: list[SpatialFieldInfo] = Field(default_factory=list)
    coordinate_columns_declared: bool = False
    warnings: list[str] = Field(default_factory=list)


class RequiredInputCheck(BaseModel):
    description: str
    satisfied: bool
    detail: str | None = None


class AnalysisCapabilityDescriptor(BaseModel):
    capability_key: str
    title: str
    cli_command: str
    availability: AvailabilityState
    required_inputs: list[RequiredInputCheck] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    support_type: SupportType
    requires_network: bool = False
    disabled: bool = False
    disabled_reason: str | None = None


class JobSubmitRequest(BaseModel):
    allow_network: bool = False
    scenario_manifest_path: str | None = None


class JobRecord(BaseModel):
    job_id: str
    project_id: str
    capability_key: str
    cli_command: str
    argv: list[str]
    status: JobState
    simple_status: SimpleJobStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    produced_layers: list[str] = Field(default_factory=list)
    error_summary: str | None = None


class CptChannel(BaseModel):
    unit: str
    available: bool
    values: list[float | None] | None = None


class CptProfile(BaseModel):
    test_id: str
    depth_bsf_m: list[float]
    channels: dict[str, CptChannel]


class StagingSessionRecord(BaseModel):
    session_id: str
    created_at: datetime


class StagedFile(BaseModel):
    file_id: str
    filename: str
    relative_path: str
    inspection: SpatialMetadataInspection


class CoordinateColumnsRequest(BaseModel):
    x_column: str
    y_column: str
    crs: str


class AssetDeclaration(BaseModel):
    """Explicit, user-supplied classification for one staged file. Never inferred -- when the
    caller omits an entry, promotion falls back to the honest "not yet classified" pairing
    (category `OTHER` / evidence role `DERIVED`), never a guessed category."""

    category: str | None = None
    evidence_role: str | None = None
    source_name: str | None = None


class PromoteRequest(BaseModel):
    project_id: str
    display_name: str
    working_crs: str
    asset_declarations: dict[str, AssetDeclaration] = Field(default_factory=dict)

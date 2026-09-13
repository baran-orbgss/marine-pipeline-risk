// Mirrors api/models.py exactly (field names, JSON shape). Hand-written rather than generated --
// deliberate for this first pass (see README). Never add a field here that duplicates a
// scientific computation; this is a transport contract only.

export type SupportType =
  | 'AREA_SURFACE'
  | 'AREA_VECTOR'
  | 'CORRIDOR'
  | 'LINEAR_ANALYSIS'
  | 'LINEAR_ASSET'
  | 'POINT_EVIDENCE'
  | 'SOURCE_FOOTPRINT'
  | 'SUPPORT_NODE'

export type LayerGroup = 'DATA' | 'DERIVED' | 'ANALYSIS' | 'EVIDENCE' | 'ASSETS'

export type AvailabilityState = 'AVAILABLE' | 'MISSING_INPUTS' | 'NOT_APPLICABLE'

export type JobState = 'QUEUED' | 'RUNNING' | 'SUCCEEDED' | 'FAILED'

export type SimpleJobStatus =
  | 'Preparing data'
  | 'Running analysis'
  | 'Building map layer'
  | 'Ready'
  | 'Failed'

export type PaletteKind = 'continuous' | 'diverging' | 'categorical' | 'single_color'

export interface BoundingBox {
  minx: number
  miny: number
  maxx: number
  maxy: number
  crs: string
}

export interface DeclaredSourceRef {
  kind: 'study_config' | 'project_manifest' | 'cpt_evidence_manifest' | 'ad_hoc_manifest'
  path: string
  declared_id: string
}

export interface ProjectSummary {
  project_id: string
  display_name: string
  description: string | null
  declared_sources: DeclaredSourceRef[]
  working_crs: string | null
  output_root: string | null
  has_registered_project: boolean
  layer_count: number | null
  extent_wgs84: BoundingBox | null
  is_ad_hoc: boolean
}

export interface UnmatchedOutputDir {
  directory: string
  reason: string
}

export interface ProjectCatalog {
  projects: ProjectSummary[]
  unmatched_output_dirs: UnmatchedOutputDir[]
}

export interface PaletteStop {
  value: number
  color: string
  label: string | null
}

export interface PaletteSpec {
  kind: PaletteKind
  colormap_name: string | null
  domain: [number, number] | null
  midpoint: number | null
  stops: PaletteStop[]
  color: string | null
}

export interface LegendSpec {
  title: string
  unit: string | null
  kind: PaletteKind
  stops: PaletteStop[]
  note: string | null
}

export interface TooltipField {
  key: string
  label: string
  unit: string | null
}

export interface DisplaySpec {
  display_name: string
  palette: PaletteSpec
  legend: LegendSpec
  units: string | null
  opacity: number
  z_index: number
  tooltip_fields: TooltipField[]
  default_visible: boolean
  scientific_limitations: string[]
}

export interface LayerDescriptor {
  layer_id: string
  project_id: string
  group: LayerGroup
  capability_key: string | null
  layer_type: 'raster' | 'vector'
  support_type: SupportType
  semantic_role: string | null
  role_established: boolean
  relative_path: string
  gpkg_layer: string | null
  crs_observed: string | null
  bounds_native: BoundingBox | null
  bounds_wgs84: BoundingBox | null
  display: DisplaySpec
  tile_url_template: string | null
  tilejson_url: string | null
  vector_url: string | null
  cpt_profile_url_template: string | null
  feature_count: number | null
  band_count: number | null
  readiness_context: Record<string, unknown> | null
}

export interface LayerCatalog {
  project_id: string
  layers: LayerDescriptor[]
}

export interface SpatialFieldInfo {
  name: string
  dtype: string
}

export interface SpatialMetadataInspection {
  kind: 'raster' | 'vector' | 'tabular_unresolved'
  observed_crs: string | null
  bounds_native: BoundingBox | null
  bounds_wgs84: BoundingBox | null
  pixel_size_x: number | null
  pixel_size_y: number | null
  width: number | null
  height: number | null
  band_count: number | null
  nodata: number | null
  geometry_type: string | null
  feature_count: number | null
  fields: SpatialFieldInfo[]
  coordinate_columns_declared: boolean
  warnings: string[]
}

export interface RequiredInputCheck {
  description: string
  satisfied: boolean
  detail: string | null
}

export interface AnalysisCapabilityDescriptor {
  capability_key: string
  title: string
  cli_command: string
  availability: AvailabilityState
  required_inputs: RequiredInputCheck[]
  reasons: string[]
  support_type: SupportType
  requires_network: boolean
  disabled: boolean
  disabled_reason: string | null
}

export interface JobRecord {
  job_id: string
  project_id: string
  capability_key: string
  cli_command: string
  argv: string[]
  status: JobState
  simple_status: SimpleJobStatus
  started_at: string | null
  finished_at: string | null
  exit_code: number | null
  produced_layers: string[]
  error_summary: string | null
}

export interface CptChannel {
  unit: string
  available: boolean
  values: (number | null)[] | null
}

export interface CptProfile {
  test_id: string
  depth_bsf_m: number[]
  channels: Record<'qc' | 'fs' | 'u2' | 'qt', CptChannel>
}

export interface StagingSessionRecord {
  session_id: string
  created_at: string
}

export interface StagedFile {
  file_id: string
  filename: string
  relative_path: string
  inspection: SpatialMetadataInspection
}

export interface AssetDeclaration {
  category: string | null
  evidence_role: string | null
  source_name: string | null
}

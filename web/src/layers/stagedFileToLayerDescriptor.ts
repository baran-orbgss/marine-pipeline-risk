import { stagingFeaturesUrl, stagingTileUrlTemplate } from '../api/staging'
import type { DisplaySpec, LayerDescriptor, StagedFile, SupportType } from '../types/api'

const UNCLASSIFIED_DISPLAY: DisplaySpec = {
  display_name: 'Unclassified',
  palette: { kind: 'single_color', color: '#999999', colormap_name: null, domain: null, midpoint: null, stops: [] },
  legend: {
    title: 'Unclassified',
    unit: null,
    kind: 'single_color',
    stops: [],
    note: 'Semantic role not established',
  },
  units: null,
  opacity: 0.7,
  z_index: 5,
  tooltip_fields: [],
  default_visible: true,
  scientific_limitations: [
    'Spatial structure detected automatically; no source/config establishes what this data represents.',
  ],
}

function guessSupportType(file: StagedFile): SupportType {
  if (file.inspection.kind === 'raster') return 'AREA_SURFACE'
  const geometryType = (file.inspection.geometry_type ?? '').toLowerCase()
  if (geometryType.includes('point')) return 'POINT_EVIDENCE'
  return 'SOURCE_FOOTPRINT'
}

/**
 * Staged (not-yet-promoted) files are not `LayerDescriptor`s from the backend -- this adapter
 * builds an equivalent, always `role_established: false`, so the same map/legend/layer-tree
 * pipeline used for real project layers also renders a dropped-files preview with zero manual
 * CRS/AOI entry (ticket S9-10/50), without a second rendering code path.
 */
export function stagedFileToLayerDescriptor(sessionId: string, file: StagedFile): LayerDescriptor {
  const isRaster = file.inspection.kind === 'raster'
  const layerId = file.file_id
  return {
    layer_id: layerId,
    project_id: `staging:${sessionId}`,
    group: 'DATA',
    capability_key: null,
    layer_type: isRaster ? 'raster' : 'vector',
    gpkg_layer: null,
    support_type: guessSupportType(file),
    semantic_role: null,
    role_established: false,
    relative_path: file.relative_path,
    crs_observed: file.inspection.observed_crs,
    bounds_native: file.inspection.bounds_native,
    bounds_wgs84: file.inspection.bounds_wgs84,
    display: { ...UNCLASSIFIED_DISPLAY, display_name: file.filename },
    tile_url_template: isRaster ? stagingTileUrlTemplate(sessionId, file.file_id) : null,
    tilejson_url: null,
    vector_url: isRaster ? null : stagingFeaturesUrl(sessionId, file.file_id),
    cpt_profile_url_template: null,
    feature_count: file.inspection.feature_count,
    band_count: file.inspection.band_count,
    readiness_context: null,
  }
}

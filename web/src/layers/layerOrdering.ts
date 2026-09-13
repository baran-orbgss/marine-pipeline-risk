import type { LayerDescriptor, SupportType } from '../types/api'

/**
 * Bottom-to-top band per support type -- this single ordering is what makes "pipeline stays
 * visible above bathymetry" and "pipeline stays visible above a route-only analysis" both fall
 * out of one function, for any project, with no per-project or per-capability special case.
 */
const SUPPORT_TYPE_BAND: Record<SupportType, number> = {
  AREA_SURFACE: 0,
  AREA_VECTOR: 1,
  CORRIDOR: 1,
  SOURCE_FOOTPRINT: 1,
  LINEAR_ANALYSIS: 2,
  SUPPORT_NODE: 2,
  POINT_EVIDENCE: 3,
  LINEAR_ASSET: 4,
}

/** Deterministic bottom-to-top layer id order: band first (support type), then the layer's own
 * `z_index` as a tiebreaker, then layer id for full stability across repeated calls. */
export function computeLayerOrder(layers: LayerDescriptor[]): string[] {
  return [...layers]
    .sort((a, b) => {
      const bandDiff = SUPPORT_TYPE_BAND[a.support_type] - SUPPORT_TYPE_BAND[b.support_type]
      if (bandDiff !== 0) return bandDiff
      const zDiff = a.display.z_index - b.display.z_index
      if (zDiff !== 0) return zDiff
      return a.layer_id.localeCompare(b.layer_id)
    })
    .map((layer) => layer.layer_id)
}

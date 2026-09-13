import type { SupportType } from '../types/api'

/**
 * The backend only ever sends the `support_type` enum -- never free-text prose -- so this
 * vocabulary stays closed, and the "Spatial support: …" wording can never drift or be spoofed by
 * a backend string. This is presentation-only; it never changes what is scientifically true about
 * a layer, only how that truth is captioned.
 */
export const SUPPORT_TYPE_META: Record<SupportType, { glyph: string; label: string; caption: string }> = {
  AREA_SURFACE: { glyph: '▦', label: 'Area', caption: 'Spatial support: area surface' },
  AREA_VECTOR: { glyph: '▦', label: 'Area', caption: 'Spatial support: area' },
  CORRIDOR: { glyph: '▧', label: 'Corridor', caption: 'Spatial support: corridor' },
  LINEAR_ANALYSIS: { glyph: '━', label: 'Route', caption: 'Spatial support: route analysis only' },
  LINEAR_ASSET: { glyph: '━', label: 'Route', caption: 'Spatial support: linear asset' },
  POINT_EVIDENCE: { glyph: '●', label: 'Points', caption: 'Spatial support: point evidence' },
  SOURCE_FOOTPRINT: { glyph: '▧', label: 'Footprint', caption: 'Spatial support: source footprint' },
  SUPPORT_NODE: { glyph: '●', label: 'Points', caption: 'Spatial support: reference points' },
}

import type { LayerSpecification, RasterSourceSpecification, GeoJSONSourceSpecification } from 'maplibre-gl'
import type { LayerDescriptor } from '../types/api'
import { paletteToColorExpression } from './paletteRegistry'

export interface MapLibreRasterBundle {
  kind: 'raster'
  sourceId: string
  source: RasterSourceSpecification
  layers: LayerSpecification[]
}

export interface MapLibreVectorBundle {
  kind: 'vector'
  sourceId: string
  source: GeoJSONSourceSpecification
  layers: LayerSpecification[]
}

export type MapLibreLayerBundle = MapLibreRasterBundle | MapLibreVectorBundle

function sourceIdFor(layerId: string): string {
  return `source:${layerId}`
}

/**
 * One style recipe per support type, applied identically for every project -- this (plus
 * `computeLayerOrder`) is the entire mechanism behind "route-only vs area-wide" and "asset stays
 * above area rasters," with no per-project or per-capability special case anywhere in this file.
 */
export function displaySpecToMapLayer(
  layer: LayerDescriptor,
  featureUrl: string | null,
): MapLibreLayerBundle | null {
  const sourceId = sourceIdFor(layer.layer_id)

  if (layer.layer_type === 'raster') {
    if (!layer.tile_url_template) return null
    return {
      kind: 'raster',
      sourceId,
      source: {
        // A relative URL is used as-is: MapLibre substitutes the literal {z}/{x}/{y} placeholders
        // before any fetch happens, and the browser resolves the relative path against the page
        // origin at request time -- routing it through `new URL()` here first would risk
        // percent-encoding the still-literal braces before that substitution ever runs.
        type: 'raster',
        tiles: [layer.tile_url_template],
        tileSize: 256,
      },
      layers: [
        {
          id: `layer:${layer.layer_id}`,
          type: 'raster',
          source: sourceId,
          paint: { 'raster-opacity': layer.display.opacity },
        },
      ],
    }
  }

  if (!featureUrl) return null
  const source: GeoJSONSourceSpecification = {
    type: 'geojson',
    data: featureUrl,
    promoteId: layer.support_type === 'POINT_EVIDENCE' ? 'test_id' : undefined,
  }

  const baseId = `layer:${layer.layer_id}`
  const color = paletteToColorExpression(layer.display.palette, 'value')
  // MapLibre's own paint-expression types are far stricter (a large literal union of tuple
  // shapes) than a dynamically-built expression can satisfy structurally; the actual runtime
  // shape (a plain JSON array) is exactly what MapLibre expects, so this is a deliberate, narrow
  // cast at the boundary rather than fighting each individual paint property's exact type.
  let layers: unknown[]

  switch (layer.support_type) {
    case 'LINEAR_ANALYSIS':
      // A casing/halo underneath for legibility, then the real data-coloured line on top. The
      // geometry is the same real LineString throughout -- only paint width changes, never a
      // buffered polygon -- so the enhanced visibility never enlarges the scientific support.
      layers = [
        {
          id: `${baseId}:casing`,
          type: 'line',
          source: sourceId,
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: { 'line-color': '#ffffff', 'line-width': 7, 'line-opacity': 0.6 },
        },
        {
          id: baseId,
          type: 'line',
          source: sourceId,
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: { 'line-color': color, 'line-width': 4, 'line-opacity': layer.display.opacity },
        },
      ]
      break
    case 'LINEAR_ASSET':
      layers = [
        {
          id: `${baseId}:halo`,
          type: 'line',
          source: sourceId,
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: { 'line-color': '#ffffff', 'line-width': 5, 'line-opacity': 0.8 },
        },
        {
          id: baseId,
          type: 'line',
          source: sourceId,
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: { 'line-color': color, 'line-width': 2.5 },
        },
      ]
      break
    case 'CORRIDOR':
      layers = [
        {
          id: `${baseId}:fill`,
          type: 'fill',
          source: sourceId,
          paint: { 'fill-color': color, 'fill-opacity': 0.08 },
        },
        {
          id: baseId,
          type: 'line',
          source: sourceId,
          paint: { 'line-color': color, 'line-width': 1.5, 'line-dasharray': [2, 2] },
        },
      ]
      break
    case 'AREA_VECTOR':
    case 'SOURCE_FOOTPRINT':
      layers = [
        {
          id: `${baseId}:fill`,
          type: 'fill',
          source: sourceId,
          paint: { 'fill-color': color, 'fill-opacity': 0.2 },
        },
        {
          id: baseId,
          type: 'line',
          source: sourceId,
          paint: { 'line-color': color, 'line-width': 1 },
        },
      ]
      break
    case 'POINT_EVIDENCE':
    case 'SUPPORT_NODE':
      layers = [
        {
          id: baseId,
          type: 'circle',
          source: sourceId,
          paint: {
            'circle-radius': [
              'case',
              ['boolean', ['feature-state', 'selected'], false],
              7,
              ['boolean', ['feature-state', 'hover'], false],
              5.5,
              4,
            ],
            'circle-color': color,
            'circle-stroke-color': '#ffffff',
            'circle-stroke-width': 1.5,
          },
        },
      ]
      break
    default:
      layers = [
        {
          id: baseId,
          type: 'line',
          source: sourceId,
          paint: { 'line-color': color, 'line-width': 2 },
        },
      ]
  }

  return { kind: 'vector', sourceId, source, layers: layers as LayerSpecification[] }
}

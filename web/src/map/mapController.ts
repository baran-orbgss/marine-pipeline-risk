import type { Map as MapLibreMap, MapGeoJSONFeature } from 'maplibre-gl'
import type { MapLibreLayerBundle } from '../layers/displaySpecToMapLayer'
import type { BoundingBox } from '../types/api'

/**
 * The single adapter boundary between application logic and the real `maplibre-gl` `Map`. Every
 * caller (layer sync, CPT selection, toolbar "fit bounds") goes through this interface, never the
 * raw `Map` directly -- which is what lets that logic be unit-tested with a hand-rolled fake, no
 * WebGL/jsdom limitation involved.
 */
export interface MapController {
  addOrUpdateLayer: (bundle: MapLibreLayerBundle, beforeId?: string) => void
  removeLayer: (layerId: string) => void
  setLayerVisibility: (layerId: string, visible: boolean, isMultiPart?: boolean) => void
  applyLayerOrder: (orderedLayerIds: string[]) => void
  flyToBounds: (bounds: BoundingBox) => void
  queryFeaturesAt: (point: [number, number], mapLayerIds: string[]) => MapGeoJSONFeature[]
  setFeatureState: (sourceId: string, featureId: string | number, state: Record<string, unknown>) => void
  removeFeatureState: (sourceId: string, featureId: string | number) => void
  getMap: () => MapLibreMap
}

const styleLayerIds = (map: MapLibreMap, layerId: string): string[] => {
  const style = map.getStyle()
  if (!style?.layers) return []
  return style.layers
    .map((l: { id: string }) => l.id)
    .filter((id: string) => id === layerId || id.startsWith(`${layerId}:`))
}

export function createMapController(map: MapLibreMap): MapController {
  return {
    addOrUpdateLayer(bundle, beforeId) {
      if (!map.getSource(bundle.sourceId)) {
        map.addSource(bundle.sourceId, bundle.source)
      }
      for (const layerSpec of bundle.layers) {
        if (!map.getLayer(layerSpec.id)) {
          map.addLayer(layerSpec, beforeId)
        }
      }
    },

    removeLayer(layerId) {
      for (const id of styleLayerIds(map, `layer:${layerId}`)) {
        if (map.getLayer(id)) map.removeLayer(id)
      }
      const sourceId = `source:${layerId}`
      if (map.getSource(sourceId)) map.removeSource(sourceId)
    },

    setLayerVisibility(layerId, visible) {
      for (const id of styleLayerIds(map, `layer:${layerId}`)) {
        if (map.getLayer(id)) {
          map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none')
        }
      }
    },

    applyLayerOrder(orderedLayerIds) {
      // moveLayer(id) with no second arg moves a layer to the very top; calling it in the
      // desired bottom-to-top order leaves the style in exactly that final order.
      for (const layerId of orderedLayerIds) {
        for (const id of styleLayerIds(map, `layer:${layerId}`)) {
          if (map.getLayer(id)) map.moveLayer(id)
        }
      }
    },

    flyToBounds(bounds) {
      // duration: 0 (instant) rather than an animated transition -- an animated `fitBounds`
      // interpolates across requestAnimationFrame callbacks, which can silently never progress
      // when the tab isn't actively driving a render loop (e.g. a backgrounded or automated
      // tab), leaving the camera stuck at its pre-transition position. An instant jump has no
      // such dependency and is the more robust choice for "fit to this project's data" moves,
      // which happen on project load/switch rather than as a user-facing animation moment.
      map.fitBounds(
        [
          [bounds.minx, bounds.miny],
          [bounds.maxx, bounds.maxy],
        ],
        { padding: 48, duration: 0, maxZoom: 17 },
      )
    },

    queryFeaturesAt(point, mapLayerIds) {
      const existing = mapLayerIds.filter((id) => map.getLayer(id))
      if (existing.length === 0) return []
      // MapLibre's line/circle hit-testing has no built-in click tolerance -- a single-pixel
      // query routinely misses a thin route line. A small bounding box around the click point is
      // the standard fix (a "fat finger" tolerance), applied once here so every caller benefits.
      const tolerance = 4
      const [x, y] = point
      const bbox: [[number, number], [number, number]] = [
        [x - tolerance, y - tolerance],
        [x + tolerance, y + tolerance],
      ]
      return map.queryRenderedFeatures(bbox, { layers: existing })
    },

    setFeatureState(sourceId, featureId, state) {
      map.setFeatureState({ source: sourceId, id: featureId }, state)
    },

    removeFeatureState(sourceId, featureId) {
      map.removeFeatureState({ source: sourceId, id: featureId })
    },

    getMap() {
      return map
    },
  }
}

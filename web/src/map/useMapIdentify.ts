import { useEffect } from 'react'
import type { MapMouseEvent } from 'maplibre-gl'
import { useIdentifyStore } from '../layers/identifyStore'
import type { LayerDescriptor } from '../types/api'
import type { MapController } from './mapController'

/**
 * Generic click-identify across every currently visible vector layer: whichever feature is
 * topmost at the click point populates the identify store, scoped to its own `layer_id`. The
 * Inspect panel (`FeatureInfoPanel`) renders it only when that id matches the active layer, and
 * only through that layer's own `tooltip_fields` allowlist -- never a raw property dump.
 */
export function useMapIdentify(controller: MapController | null, visibleVectorLayers: LayerDescriptor[]) {
  const setIdentified = useIdentifyStore((s) => s.setIdentified)
  const layerIdsKey = visibleVectorLayers.map((l) => l.layer_id).join(',')

  useEffect(() => {
    if (!controller) return
    // A layer_id may itself contain colons (e.g. "pl854:output:sediment/..."), so matching a
    // rendered MapLibre sub-layer id (e.g. "layer:pl854:output:sediment/...:casing") back to its
    // owning layer_id must be a prefix check against the known base ids, never a fixed split.
    const baseIds = visibleVectorLayers.map((l) => ({ base: `layer:${l.layer_id}`, layerId: l.layer_id }))
    const mapLayerIds = baseIds.flatMap(({ base }) => [base, `${base}:casing`, `${base}:halo`, `${base}:fill`])
    if (mapLayerIds.length === 0) return

    const findOwningLayerId = (mapLayerId: string): string | null => {
      const match = baseIds.find(
        ({ base }) => mapLayerId === base || mapLayerId.startsWith(`${base}:`),
      )
      return match ? match.layerId : null
    }

    const map = controller.getMap()
    const handleClick = (event: MapMouseEvent) => {
      const features = controller.queryFeaturesAt([event.point.x, event.point.y], mapLayerIds)
      if (features.length === 0) {
        setIdentified(null, null)
        return
      }
      const feature = features[0]
      const layerId = findOwningLayerId(String(feature.layer?.id ?? ''))
      setIdentified(layerId, (feature.properties as Record<string, unknown>) ?? null)
    }

    map.on('click', handleClick)
    return () => {
      map.off('click', handleClick)
    }
  }, [controller, layerIdsKey, setIdentified])
}

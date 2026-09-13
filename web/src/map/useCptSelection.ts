import { useEffect, useRef } from 'react'
import type { MapMouseEvent } from 'maplibre-gl'
import { useCptSelectionStore } from '../layers/cptSelectionStore'
import type { LayerDescriptor } from '../types/api'
import type { MapController } from './mapController'

/**
 * Click a CPT (or other POINT_EVIDENCE) feature -> highlight it via `feature-state.selected` and
 * record its id in the shared selection store; click empty space -> clear it. Only ever two
 * `setFeatureState` calls per selection change (clear the previous id, set the new one) -- no
 * re-styling of the whole layer.
 */
export function useCptSelection(controller: MapController | null, pointLayers: LayerDescriptor[]) {
  const select = useCptSelectionStore((s) => s.select)
  const previousRef = useRef<{ sourceId: string; featureId: string | number } | null>(null)
  const pointLayerIdsKey = pointLayers.map((l) => l.layer_id).join(',')

  useEffect(() => {
    if (!controller) return
    const mapLayerIds = pointLayerIdsKey ? pointLayerIdsKey.split(',').map((id) => `layer:${id}`) : []
    if (mapLayerIds.length === 0) return

    const map = controller.getMap()
    const handleClick = (event: MapMouseEvent) => {
      const features = controller.queryFeaturesAt([event.point.x, event.point.y], mapLayerIds)
      if (features.length === 0) {
        if (previousRef.current) {
          controller.removeFeatureState(previousRef.current.sourceId, previousRef.current.featureId)
          previousRef.current = null
        }
        select(null)
        return
      }
      const feature = features[0]
      const sourceId = String(feature.source)
      const featureId = feature.id
      if (featureId === undefined) return

      if (previousRef.current) {
        controller.removeFeatureState(previousRef.current.sourceId, previousRef.current.featureId)
      }
      controller.setFeatureState(sourceId, featureId, { selected: true })
      previousRef.current = { sourceId, featureId }
      const testId = (feature.properties as Record<string, unknown> | null)?.test_id
      select(typeof testId === 'string' ? testId : null)
    }

    map.on('click', handleClick)
    return () => {
      map.off('click', handleClick)
    }
  }, [controller, pointLayerIdsKey, select])
}

import { useEffect, useRef } from 'react'
import type { MapController } from '../map/mapController'
import type { LayerDescriptor } from '../types/api'
import { displaySpecToMapLayer } from './displaySpecToMapLayer'
import { computeLayerOrder } from './layerOrdering'

/**
 * Reconciles (layers, visibility) into the minimal set of `MapController` calls: add a source+
 * layer for anything newly visible, remove anything no longer present, update visibility for
 * everything else, then reapply the deterministic bottom-to-top order. This is the entire
 * "result appears as a map layer automatically" mechanism on the frontend side -- it runs
 * whenever the layer catalog query is invalidated, with no per-capability special case.
 */
export function useSyncLayersToMap(
  controller: MapController | null,
  layers: LayerDescriptor[],
  visibleLayerIds: Record<string, boolean>,
  scopeId: string | null,
) {
  const addedLayerIdsRef = useRef<Set<string>>(new Set())

  useEffect(() => {
    if (!controller || !scopeId) return
    const currentIds = new Set(layers.map((l) => l.layer_id))

    for (const staleId of addedLayerIdsRef.current) {
      if (!currentIds.has(staleId)) {
        controller.removeLayer(staleId)
        addedLayerIdsRef.current.delete(staleId)
      }
    }

    for (const layer of layers) {
      const shouldBeVisible = Boolean(visibleLayerIds[layer.layer_id])
      const alreadyAdded = addedLayerIdsRef.current.has(layer.layer_id)

      if (shouldBeVisible && !alreadyAdded) {
        // Each LayerDescriptor already carries its own correct feature/tile URL (a real project's
        // `/api/projects/.../features` or a staging preview's `/api/staging/.../features`) --
        // using it directly, rather than recomputing one from `scopeId`, is what lets the exact
        // same sync pipeline render both without a staging-specific special case.
        const bundle = displaySpecToMapLayer(layer, layer.vector_url)
        if (bundle) {
          controller.addOrUpdateLayer(bundle)
          addedLayerIdsRef.current.add(layer.layer_id)
        }
      }
      if (alreadyAdded) {
        controller.setLayerVisibility(layer.layer_id, shouldBeVisible)
      }
    }

    controller.applyLayerOrder(computeLayerOrder(layers))
  }, [controller, layers, visibleLayerIds, scopeId])

  useEffect(() => {
    const added = addedLayerIdsRef.current
    return () => {
      if (!controller) return
      for (const layerId of added) {
        controller.removeLayer(layerId)
      }
      added.clear()
    }
    // Only run this cleanup when the project changes (or unmount) -- not on every layer/
    // visibility update, which would otherwise tear down and re-add layers on every toggle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeId])
}

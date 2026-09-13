import { beforeEach, describe, expect, it } from 'vitest'
import { bathymetryLayer, pipelineLayer } from '../test/fixtures/layers.fixture'
import { useLayerTreeStore } from './layerTreeStore'

describe('layerTreeStore', () => {
  beforeEach(() => {
    useLayerTreeStore.getState().reset()
  })

  it('toggling visibility never touches the active layer', () => {
    const store = useLayerTreeStore.getState()
    store.setActiveLayer(bathymetryLayer.layer_id)
    store.toggleVisibility(pipelineLayer.layer_id)

    expect(useLayerTreeStore.getState().activeLayerId).toBe(bathymetryLayer.layer_id)
    expect(useLayerTreeStore.getState().visibleLayerIds[pipelineLayer.layer_id]).toBe(true)
  })

  it('setting the active layer never touches visibility', () => {
    const store = useLayerTreeStore.getState()
    store.toggleVisibility(bathymetryLayer.layer_id)
    store.setActiveLayer(pipelineLayer.layer_id)

    expect(useLayerTreeStore.getState().visibleLayerIds[bathymetryLayer.layer_id]).toBe(true)
    expect(useLayerTreeStore.getState().activeLayerId).toBe(pipelineLayer.layer_id)
  })

  it('seedDefaults fills only missing keys, never overwriting an existing user toggle', () => {
    const store = useLayerTreeStore.getState()
    store.toggleVisibility(bathymetryLayer.layer_id) // user explicitly turns it ON once
    store.toggleVisibility(bathymetryLayer.layer_id) // ...then OFF -- an explicit choice

    store.seedDefaults([bathymetryLayer, pipelineLayer])

    // bathymetry's explicit "off" must survive the reseed even though default_visible is true.
    expect(useLayerTreeStore.getState().visibleLayerIds[bathymetryLayer.layer_id]).toBe(false)
    // pipeline had no prior entry, so its own default_visible fills in.
    expect(useLayerTreeStore.getState().visibleLayerIds[pipelineLayer.layer_id]).toBe(
      pipelineLayer.display.default_visible,
    )
  })

  it('pruneMissing clears activeLayerId when that layer no longer exists', () => {
    const store = useLayerTreeStore.getState()
    store.setActiveLayer(bathymetryLayer.layer_id)
    store.toggleVisibility(bathymetryLayer.layer_id)

    store.pruneMissing([pipelineLayer.layer_id])

    const next = useLayerTreeStore.getState()
    expect(next.activeLayerId).toBeNull()
    expect(next.visibleLayerIds[bathymetryLayer.layer_id]).toBeUndefined()
  })

  it('pruneMissing keeps a still-present active layer', () => {
    const store = useLayerTreeStore.getState()
    store.setActiveLayer(bathymetryLayer.layer_id)

    store.pruneMissing([bathymetryLayer.layer_id, pipelineLayer.layer_id])

    expect(useLayerTreeStore.getState().activeLayerId).toBe(bathymetryLayer.layer_id)
  })
})

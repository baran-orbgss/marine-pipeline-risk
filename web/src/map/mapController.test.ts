import { describe, expect, it } from 'vitest'
import type { MapLibreLayerBundle } from '../layers/displaySpecToMapLayer'
import { createMapController } from './mapController'

function fakeMap() {
  const sources = new Set<string>()
  const layers: { id: string }[] = []
  const calls: Record<string, unknown[][]> = {}
  const record = (name: string, args: unknown[]) => {
    calls[name] = calls[name] ?? []
    calls[name].push(args)
  }

  return {
    calls,
    map: {
      getSource: (id: string) => (sources.has(id) ? {} : undefined),
      addSource: (id: string, spec: unknown) => {
        sources.add(id)
        record('addSource', [id, spec])
      },
      getLayer: (id: string) => layers.find((l) => l.id === id),
      addLayer: (spec: { id: string }) => {
        layers.push(spec)
        record('addLayer', [spec])
      },
      removeLayer: (id: string) => {
        const index = layers.findIndex((l) => l.id === id)
        if (index >= 0) layers.splice(index, 1)
        record('removeLayer', [id])
      },
      removeSource: (id: string) => {
        sources.delete(id)
        record('removeSource', [id])
      },
      setLayoutProperty: (...args: unknown[]) => record('setLayoutProperty', args),
      moveLayer: (...args: unknown[]) => record('moveLayer', args),
      fitBounds: (...args: unknown[]) => record('fitBounds', args),
      queryRenderedFeatures: () => [],
      setFeatureState: (...args: unknown[]) => record('setFeatureState', args),
      removeFeatureState: (...args: unknown[]) => record('removeFeatureState', args),
      getStyle: () => ({ layers: layers.map((l) => ({ id: l.id })) }),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any,
  }
}

const rasterBundle: MapLibreLayerBundle = {
  kind: 'raster',
  sourceId: 'source:test',
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  source: { type: 'raster', tiles: ['x'], tileSize: 256 } as any,
  layers: [
    { id: 'layer:test', type: 'raster', source: 'source:test', paint: {} },
    { id: 'layer:test:halo', type: 'raster', source: 'source:test', paint: {} },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ] as any,
}

describe('mapController', () => {
  it('addOrUpdateLayer adds the source once and every sub-layer', () => {
    const { map, calls } = fakeMap()
    const controller = createMapController(map)
    controller.addOrUpdateLayer(rasterBundle)
    controller.addOrUpdateLayer(rasterBundle) // idempotent second call

    expect(calls.addSource?.length).toBe(1)
    expect(calls.addLayer?.length).toBe(2)
  })

  it('setLayerVisibility calls setLayoutProperty for every sub-layer of that layer id', () => {
    const { map, calls } = fakeMap()
    const controller = createMapController(map)
    controller.addOrUpdateLayer(rasterBundle)
    controller.setLayerVisibility('test', false)

    expect(calls.setLayoutProperty?.length).toBe(2)
    for (const call of calls.setLayoutProperty ?? []) {
      expect(call[1]).toBe('visibility')
      expect(call[2]).toBe('none')
    }
  })

  it('removeLayer removes every sub-layer and the source', () => {
    const { map, calls } = fakeMap()
    const controller = createMapController(map)
    controller.addOrUpdateLayer(rasterBundle)
    controller.removeLayer('test')

    expect(calls.removeLayer?.length).toBe(2)
    expect(calls.removeSource?.length).toBe(1)
  })

  it('applyLayerOrder issues a moveLayer call per matching sub-layer in the given order', () => {
    const { map, calls } = fakeMap()
    const controller = createMapController(map)
    controller.addOrUpdateLayer(rasterBundle)
    controller.applyLayerOrder(['test'])

    expect(calls.moveLayer?.length).toBe(2)
  })

  it('flyToBounds calls fitBounds with the bounding box corners', () => {
    const { map, calls } = fakeMap()
    const controller = createMapController(map)
    controller.flyToBounds({ minx: 1, miny: 2, maxx: 3, maxy: 4, crs: 'EPSG:4326' })

    expect(calls.fitBounds?.[0][0]).toEqual([
      [1, 2],
      [3, 4],
    ])
    // Regression guard: an animated fitBounds (duration > 0) interpolates across
    // requestAnimationFrame callbacks, which silently never progress in some tab states
    // (backgrounded/automated), leaving the camera stuck -- verified directly against a live
    // map instance. duration must stay 0 (instant).
    expect((calls.fitBounds?.[0][1] as { duration?: number })?.duration).toBe(0)
  })

  it('setFeatureState / removeFeatureState pass source and id through untouched', () => {
    const { map, calls } = fakeMap()
    const controller = createMapController(map)
    controller.setFeatureState('source:test', 'CPT-1', { selected: true })
    controller.removeFeatureState('source:test', 'CPT-1')

    expect(calls.setFeatureState?.[0][0]).toEqual({ source: 'source:test', id: 'CPT-1' })
    expect(calls.removeFeatureState?.[0][0]).toEqual({ source: 'source:test', id: 'CPT-1' })
  })
})

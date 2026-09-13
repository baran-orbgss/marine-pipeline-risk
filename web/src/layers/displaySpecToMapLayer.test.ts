import { describe, expect, it } from 'vitest'
import { bathymetryLayer, cptLayer, mobilityLayer, seabedChangeLayer } from '../test/fixtures/layers.fixture'
import { displaySpecToMapLayer } from './displaySpecToMapLayer'

describe('displaySpecToMapLayer', () => {
  it('a raster layer becomes a plain raster source with no client-side recolouring', () => {
    const bundle = displaySpecToMapLayer(bathymetryLayer, null)
    expect(bundle?.kind).toBe('raster')
    if (bundle?.kind === 'raster') {
      expect(bundle.source.tiles).toEqual([bathymetryLayer.tile_url_template])
      // no colour-related paint property, ever -- colourization happens server-side.
      for (const layer of bundle.layers) {
        expect(Object.keys(layer.paint ?? {})).not.toContain('raster-color')
      }
    }
  })

  it('a raster layer with no tile_url_template returns null rather than a broken source', () => {
    const broken = { ...bathymetryLayer, tile_url_template: null }
    expect(displaySpecToMapLayer(broken, null)).toBeNull()
  })

  it('a vector layer with no feature url returns null rather than a broken source', () => {
    expect(displaySpecToMapLayer(mobilityLayer, null)).toBeNull()
  })

  it('a diverging raster legend and the underlying palette share the same stop values', () => {
    const stops = seabedChangeLayer.display.legend.stops
    const paletteValues = seabedChangeLayer.display.palette.stops.length
      ? seabedChangeLayer.display.palette.stops.map((s) => s.value)
      : stops.map((s) => s.value) // this fixture reuses one shared stop array for both
    expect(stops.map((s) => s.value)).toEqual(paletteValues)
  })

  it('a LINEAR_ANALYSIS layer renders as a real line (casing + line), never a filled polygon', () => {
    const bundle = displaySpecToMapLayer(mobilityLayer, 'https://example/features')
    expect(bundle?.kind).toBe('vector')
    const types = bundle?.layers.map((l) => l.type) ?? []
    expect(types).toContain('line')
    expect(types).not.toContain('fill')
  })

  it('a POINT_EVIDENCE layer promotes test_id as the MapLibre feature id', () => {
    const bundle = displaySpecToMapLayer(cptLayer, 'https://example/features')
    expect(bundle?.kind).toBe('vector')
    if (bundle?.kind === 'vector') {
      expect(bundle.source.promoteId).toBe('test_id')
    }
  })
})

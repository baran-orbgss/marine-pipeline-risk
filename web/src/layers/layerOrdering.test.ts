import { describe, expect, it } from 'vitest'
import { bathymetryLayer, cptLayer, mobilityLayer, pipelineLayer, seabedChangeLayer } from '../test/fixtures/layers.fixture'
import { computeLayerOrder } from './layerOrdering'

describe('computeLayerOrder', () => {
  it('always places raster area layers below the pipeline asset', () => {
    const order = computeLayerOrder([pipelineLayer, bathymetryLayer])
    expect(order.indexOf(bathymetryLayer.layer_id)).toBeLessThan(order.indexOf(pipelineLayer.layer_id))
  })

  it('always places a route-only analysis layer below the pipeline asset', () => {
    const order = computeLayerOrder([pipelineLayer, mobilityLayer])
    expect(order.indexOf(mobilityLayer.layer_id)).toBeLessThan(order.indexOf(pipelineLayer.layer_id))
  })

  it('places point evidence above area/route layers but below the asset', () => {
    const order = computeLayerOrder([pipelineLayer, mobilityLayer, cptLayer, bathymetryLayer])
    const indices = {
      bathymetry: order.indexOf(bathymetryLayer.layer_id),
      mobility: order.indexOf(mobilityLayer.layer_id),
      cpt: order.indexOf(cptLayer.layer_id),
      pipeline: order.indexOf(pipelineLayer.layer_id),
    }
    expect(indices.bathymetry).toBeLessThan(indices.mobility)
    expect(indices.mobility).toBeLessThan(indices.cpt)
    expect(indices.cpt).toBeLessThan(indices.pipeline)
  })

  it('is stable and deterministic regardless of input order', () => {
    const forward = computeLayerOrder([bathymetryLayer, pipelineLayer, mobilityLayer, cptLayer, seabedChangeLayer])
    const shuffled = computeLayerOrder([cptLayer, seabedChangeLayer, pipelineLayer, bathymetryLayer, mobilityLayer])
    expect(shuffled).toEqual(forward)
  })
})

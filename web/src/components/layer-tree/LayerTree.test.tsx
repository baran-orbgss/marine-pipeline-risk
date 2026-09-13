import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { useLayerTreeStore } from '../../layers/layerTreeStore'
import { bathymetryLayer, cptLayer, mobilityLayer, pipelineLayer } from '../../test/fixtures/layers.fixture'
import { LayerTree } from './LayerTree'

describe('LayerTree', () => {
  beforeEach(() => {
    useLayerTreeStore.getState().reset()
  })

  it('renders groups in the fixed DATA/DERIVED/ANALYSIS/EVIDENCE/ASSETS order regardless of input order', () => {
    render(<LayerTree layers={[pipelineLayer, cptLayer, mobilityLayer, bathymetryLayer]} />)
    const headers = screen.getAllByRole('button', { name: /Data|Derived|Analysis|Evidence|Assets/ })
    const labels = headers.map((h) => h.textContent?.replace(/^[▾▸]/, '').replace(/\d+$/, '').trim())
    expect(labels).toEqual(['Data', 'Analysis', 'Evidence', 'Assets'])
  })

  it('clicking a visibility checkbox toggles only that layer, never the active layer', () => {
    render(<LayerTree layers={[bathymetryLayer, pipelineLayer]} />)
    useLayerTreeStore.getState().setActiveLayer(pipelineLayer.layer_id)

    const checkbox = screen.getByLabelText(`Toggle visibility of ${bathymetryLayer.display.display_name}`)
    fireEvent.click(checkbox)

    expect(useLayerTreeStore.getState().visibleLayerIds[bathymetryLayer.layer_id]).toBe(true)
    expect(useLayerTreeStore.getState().activeLayerId).toBe(pipelineLayer.layer_id)
  })

  it('clicking a layer name sets exactly that layer active, never touching visibility', () => {
    render(<LayerTree layers={[bathymetryLayer, pipelineLayer]} />)

    fireEvent.click(screen.getByText(pipelineLayer.display.display_name))

    expect(useLayerTreeStore.getState().activeLayerId).toBe(pipelineLayer.layer_id)
    expect(useLayerTreeStore.getState().visibleLayerIds[pipelineLayer.layer_id]).toBeUndefined()
  })

  it('shows an empty state when there are no layers', () => {
    render(<LayerTree layers={[]} />)
    expect(screen.getByText('No layers yet')).toBeInTheDocument()
  })
})

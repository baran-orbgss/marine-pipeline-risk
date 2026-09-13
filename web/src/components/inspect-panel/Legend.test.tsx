import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { DisplaySpec } from '../../types/api'
import { Legend } from './Legend'

function baseDisplay(overrides: Partial<DisplaySpec> = {}): DisplaySpec {
  return {
    display_name: 'Layer',
    palette: { kind: 'single_color', color: '#333333', colormap_name: null, domain: null, midpoint: null, stops: [] },
    legend: { title: 'Layer', unit: null, kind: 'single_color', stops: [], note: null },
    units: null,
    opacity: 1,
    z_index: 0,
    tooltip_fields: [],
    default_visible: true,
    scientific_limitations: [],
    ...overrides,
  }
}

describe('Legend', () => {
  it('renders a note even when there are no stops (e.g. hillshade, unclassified)', () => {
    const display = baseDisplay({
      legend: { title: 'Display hillshade', unit: null, kind: 'single_color', stops: [], note: 'Display hillshade — visualization only' },
    })
    render(<Legend display={display} />)
    expect(screen.getByText(/visualization only/i)).toBeInTheDocument()
  })

  it('renders nothing when there are neither stops nor a note', () => {
    const { container } = render(<Legend display={baseDisplay()} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders a labeled swatch for a single-color spec with one stop', () => {
    const display = baseDisplay({
      legend: {
        title: 'Assets',
        unit: null,
        kind: 'single_color',
        stops: [{ value: 0, color: '#111111', label: 'Pipeline / cable route' }],
        note: null,
      },
    })
    render(<Legend display={display} />)
    expect(screen.getByText('Pipeline / cable route')).toBeInTheDocument()
  })
})

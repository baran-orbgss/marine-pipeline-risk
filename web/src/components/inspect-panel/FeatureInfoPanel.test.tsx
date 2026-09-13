import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { FeatureInfoPanel } from './FeatureInfoPanel'

describe('FeatureInfoPanel', () => {
  it('renders only allowlisted fields, never a raw property dump', () => {
    render(
      <FeatureInfoPanel
        tooltipFields={[
          { key: 'chainage_m', label: 'Chainage', unit: 'm' },
          { key: 'screening_state', label: 'Screening state', unit: null },
        ]}
        properties={{
          chainage_m: 1250,
          screening_state: 'NOT_AVAILABLE',
          internal_debug_column: 'should never render',
          another_secret_field: 42,
        }}
      />,
    )

    expect(screen.getByText('Chainage')).toBeInTheDocument()
    expect(screen.getByText('Screening state')).toBeInTheDocument()
    expect(screen.queryByText(/internal_debug_column/)).not.toBeInTheDocument()
    expect(screen.queryByText(/should never render/)).not.toBeInTheDocument()
    expect(screen.queryByText(/another_secret_field/)).not.toBeInTheDocument()
    expect(screen.queryByText('42')).not.toBeInTheDocument()
  })

  it('renders a placeholder rather than crashing when a field is missing from properties', () => {
    render(
      <FeatureInfoPanel
        tooltipFields={[{ key: 'missing_key', label: 'Missing', unit: null }]}
        properties={{ other: 1 }}
      />,
    )
    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it('renders nothing when there are no properties at all', () => {
    const { container } = render(
      <FeatureInfoPanel tooltipFields={[{ key: 'x', label: 'X', unit: null }]} properties={null} />,
    )
    expect(container).toBeEmptyDOMElement()
  })
})

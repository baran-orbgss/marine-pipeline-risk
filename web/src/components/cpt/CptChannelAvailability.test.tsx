import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CptChannelAvailability } from './CptChannelAvailability'

describe('CptChannelAvailability', () => {
  it('shows an explicit "Unavailable for this source" notice for qt, never a silent gap', () => {
    render(
      <CptChannelAvailability label="qt" channel={{ unit: 'MPa', available: false, values: null }} depths={[]} />,
    )
    expect(screen.getByText('Unavailable for this source')).toBeInTheDocument()
  })

  it('renders a chart for an available channel', () => {
    render(
      <CptChannelAvailability
        label="qc"
        channel={{ unit: 'MPa', available: true, values: [1, 2, 3] }}
        depths={[0, 1, 2]}
      />,
    )
    expect(screen.queryByText('Unavailable for this source')).not.toBeInTheDocument()
    expect(document.querySelector('svg')).not.toBeNull()
  })

  it('never renders CRR, CSR, factor-of-safety, or soil-classification text', () => {
    const { container } = render(
      <CptChannelAvailability label="qt" channel={{ unit: 'MPa', available: false, values: null }} depths={[]} />,
    )
    expect(container.textContent ?? '').not.toMatch(/CRR|CSR|factor of safety|liquefaction|soil classification/i)
  })
})

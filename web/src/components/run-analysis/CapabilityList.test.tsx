import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { AnalysisCapabilityDescriptor } from '../../types/api'
import { CapabilityList } from './CapabilityList'

function capability(
  overrides: Partial<AnalysisCapabilityDescriptor>,
): AnalysisCapabilityDescriptor {
  return {
    capability_key: 'terrain',
    title: 'Terrain',
    cli_command: 'build-highres-terrain-poc',
    availability: 'AVAILABLE',
    required_inputs: [],
    reasons: [],
    support_type: 'AREA_SURFACE',
    requires_network: false,
    disabled: false,
    disabled_reason: null,
    ...overrides,
  }
}

describe('CapabilityList', () => {
  it('enables Run only when availability is AVAILABLE', () => {
    render(
      <CapabilityList
        capabilities={[capability({ availability: 'AVAILABLE' })]}
        onRun={() => {}}
        runningKey={null}
      />,
    )
    expect(screen.getByRole('button', { name: 'Run' })).toBeEnabled()
  })

  it('disables Run for MISSING_INPUTS and shows the backend reason verbatim', () => {
    render(
      <CapabilityList
        capabilities={[
          capability({ availability: 'MISSING_INPUTS', reasons: ['needs a registered bathymetry raster asset'] }),
        ]}
        onRun={() => {}}
        runningKey={null}
      />,
    )
    expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled()
    expect(screen.getByText('needs a registered bathymetry raster asset')).toBeInTheDocument()
  })

  it('disables Run for NOT_APPLICABLE even if a naive client heuristic would disagree', () => {
    // A naive client-side check might see `required_inputs` all satisfied and conclude
    // "available" -- this fixture deliberately sets that trap. The component must still obey the
    // backend's own `availability` value, not recompute one from `required_inputs` itself.
    render(
      <CapabilityList
        capabilities={[
          capability({
            availability: 'NOT_APPLICABLE',
            required_inputs: [{ description: 'looks satisfied', satisfied: true, detail: null }],
            reasons: ['not generalized in this release'],
          }),
        ]}
        onRun={() => {}}
        runningKey={null}
      />,
    )
    expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled()
  })

  it('calls onRun with the capability key when clicked', () => {
    const onRun = vi.fn()
    render(<CapabilityList capabilities={[capability({})]} onRun={onRun} runningKey={null} />)
    fireEvent.click(screen.getByRole('button', { name: 'Run' }))
    expect(onRun).toHaveBeenCalledWith('terrain')
  })

  it('shows "Running…" and disables the button for the currently running capability', () => {
    render(<CapabilityList capabilities={[capability({})]} onRun={() => {}} runningKey="terrain" />)
    expect(screen.getByRole('button', { name: 'Running…' })).toBeDisabled()
  })
})

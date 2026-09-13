import { describe, expect, it } from 'vitest'
import { seabedChangeLayer } from '../test/fixtures/layers.fixture'
import { legendFromDisplaySpec } from './legendFromDisplaySpec'

describe('legendFromDisplaySpec', () => {
  it('renders the exact diverging seabed-change caption', () => {
    const legend = legendFromDisplaySpec(seabedChangeLayer.display)
    expect(legend.note).toBe('negative ← 0 → positive')
    expect(legend.gradient).toBe(true)
  })

  it('never returns a stop without a usable label or value', () => {
    const legend = legendFromDisplaySpec(seabedChangeLayer.display)
    for (const stop of legend.stops) {
      expect(stop.label !== null || stop.value !== null).toBe(true)
    }
  })

  it('sorts stops ascending regardless of input order', () => {
    const legend = legendFromDisplaySpec(seabedChangeLayer.display)
    const values = legend.stops.map((s) => s.value)
    expect(values).toEqual([...values].sort((a, b) => a - b))
  })
})

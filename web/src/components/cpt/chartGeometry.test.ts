import { describe, expect, it } from 'vitest'
import { computeChartGeometry, type ChartDimensions } from './chartGeometry'

const DIMS: ChartDimensions = { width: 100, height: 200, marginTop: 10, marginBottom: 10, marginLeft: 10, marginRight: 10 }

describe('computeChartGeometry', () => {
  it('maps greater depth to a greater pixel-y (inverted depth axis)', () => {
    const depths = [0, 5, 10]
    const values = [1, 2, 3]
    const geometry = computeChartGeometry(depths, values, DIMS)
    // extract the path's y coordinates in order (a plain "M x,y L x,y L x,y" line)
    const points = geometry.path.match(/-?\d+(\.\d+)?/g)?.map(Number) ?? []
    const ys = points.filter((_, i) => i % 2 === 1)
    expect(ys[0]).toBeLessThan(ys[1])
    expect(ys[1]).toBeLessThan(ys[2])
  })

  it('skips null values without breaking the remaining path', () => {
    const geometry = computeChartGeometry([0, 5, 10], [1, null, 3], DIMS)
    expect(geometry.path).not.toBe('')
  })

  it('returns an empty path for no data rather than throwing', () => {
    const geometry = computeChartGeometry([], [], DIMS)
    expect(geometry.path).toBe('')
  })
})

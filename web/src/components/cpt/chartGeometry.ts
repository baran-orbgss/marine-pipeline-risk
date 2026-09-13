import { scaleLinear } from 'd3-scale'
import { line as d3line } from 'd3-shape'

export interface ChartDimensions {
  width: number
  height: number
  marginTop: number
  marginBottom: number
  marginLeft: number
  marginRight: number
}

export interface ChartGeometry {
  path: string
  xTicks: { value: number; x: number }[]
  yTicks: { value: number; y: number }[]
  xScaleRange: [number, number]
  yScaleRange: [number, number]
}

/**
 * Pure geometry computation, no DOM/SVG rendering involved -- depth (larger = further down)
 * always maps to a larger pixel-y, matching the existing engine convention of an inverted depth
 * axis. Testable with plain data assertions, no rendering required.
 */
export function computeChartGeometry(
  depths: number[],
  values: (number | null)[],
  dims: ChartDimensions,
): ChartGeometry {
  const innerWidth = dims.width - dims.marginLeft - dims.marginRight
  const innerHeight = dims.height - dims.marginTop - dims.marginBottom

  const finiteValues = values.filter((v): v is number => v !== null && Number.isFinite(v))
  const valueExtent: [number, number] =
    finiteValues.length > 0 ? [Math.min(...finiteValues, 0), Math.max(...finiteValues)] : [0, 1]
  const depthExtent: [number, number] =
    depths.length > 0 ? [Math.min(...depths), Math.max(...depths)] : [0, 1]

  const xScale = scaleLinear()
    .domain(valueExtent)
    .range([dims.marginLeft, dims.marginLeft + innerWidth])
  const yScale = scaleLinear()
    .domain(depthExtent) // depth increases downward -> larger depth = larger pixel-y
    .range([dims.marginTop, dims.marginTop + innerHeight])

  const points: [number, number][] = []
  for (let i = 0; i < depths.length; i++) {
    const value = values[i]
    if (value === null || !Number.isFinite(value)) continue
    points.push([xScale(value), yScale(depths[i])])
  }

  const lineGenerator = d3line<[number, number]>(
    (d) => d[0],
    (d) => d[1],
  )
  const path = lineGenerator(points) ?? ''

  const xTicks = xScale.ticks(4).map((value) => ({ value, x: xScale(value) }))
  const yTicks = yScale.ticks(4).map((value) => ({ value, y: yScale(value) }))

  return {
    path,
    xTicks,
    yTicks,
    xScaleRange: [dims.marginLeft, dims.marginLeft + innerWidth],
    yScaleRange: [dims.marginTop, dims.marginTop + innerHeight],
  }
}

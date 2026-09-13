import type { PaletteSpec } from '../types/api'

export type MapLibreExpression = (string | number | (string | number)[])[]

/**
 * A vector layer's paint colour comes from the exact same `stops` array the legend renders --
 * this is the mechanism that keeps a legend from ever drifting out of sync with what's drawn (a
 * raster's colour, by contrast, is applied server-side; see displaySpecToMapLayer.ts).
 */
export function paletteToColorExpression(
  palette: PaletteSpec,
  propertyName: string,
): string | MapLibreExpression {
  if (palette.kind === 'single_color') {
    return palette.color ?? '#3388ff'
  }

  const stops = [...palette.stops].sort((a, b) => a.value - b.value)
  if (stops.length === 0) return '#999999'
  // A single-stop categorical spec (e.g. bedforms) has nothing to key off of a property for --
  // it is, in practice, one flat colour.
  if (stops.length === 1) return stops[0].color

  if (palette.kind === 'categorical') {
    const expression: MapLibreExpression = ['match', ['get', propertyName]]
    for (const stop of stops) {
      expression.push(stop.value, stop.color)
    }
    expression.push('#999999')
    return expression
  }

  // continuous or diverging: an interpolate expression over the property's numeric value.
  const expression: MapLibreExpression = ['interpolate', ['linear'], ['get', propertyName]]
  for (const stop of stops) {
    expression.push(stop.value, stop.color)
  }
  return expression
}

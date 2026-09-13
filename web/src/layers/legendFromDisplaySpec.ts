import type { DisplaySpec, PaletteStop } from '../types/api'

export interface LegendRenderModel {
  title: string
  unit: string | null
  gradient: boolean
  stops: PaletteStop[]
  note: string | null
}

/** The single place that turns a layer's `DisplaySpec.legend` into what `Legend.tsx` renders --
 * kept as a named function (not inline JSX) so it stays independently testable and reusable. */
export function legendFromDisplaySpec(display: DisplaySpec): LegendRenderModel {
  const { legend } = display
  return {
    title: legend.title,
    unit: legend.unit,
    gradient: legend.kind === 'continuous' || legend.kind === 'diverging',
    stops: [...legend.stops].sort((a, b) => a.value - b.value),
    note: legend.note,
  }
}

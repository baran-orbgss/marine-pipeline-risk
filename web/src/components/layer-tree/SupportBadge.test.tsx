import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SUPPORT_TYPE_META } from '../../layers/supportType'
import type { SupportType } from '../../types/api'
import { SupportBadge } from './SupportBadge'

const ALL_SUPPORT_TYPES: SupportType[] = [
  'AREA_SURFACE',
  'AREA_VECTOR',
  'CORRIDOR',
  'LINEAR_ANALYSIS',
  'LINEAR_ASSET',
  'POINT_EVIDENCE',
  'SOURCE_FOOTPRINT',
  'SUPPORT_NODE',
]

describe('SupportBadge', () => {
  it.each(ALL_SUPPORT_TYPES)('renders a defined label for %s', (supportType) => {
    render(<SupportBadge supportType={supportType} />)
    expect(screen.getByText(SUPPORT_TYPE_META[supportType].label)).toBeInTheDocument()
  })

  it('has an entry for every SupportType value with no gaps', () => {
    expect(Object.keys(SUPPORT_TYPE_META).sort()).toEqual([...ALL_SUPPORT_TYPES].sort())
  })
})

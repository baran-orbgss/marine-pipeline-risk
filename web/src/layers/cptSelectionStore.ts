import { create } from 'zustand'

interface CptSelectionState {
  selectedTestId: string | null
  select: (testId: string | null) => void
}

export const useCptSelectionStore = create<CptSelectionState>((set) => ({
  selectedTestId: null,
  select: (testId) => set({ selectedTestId: testId }),
}))

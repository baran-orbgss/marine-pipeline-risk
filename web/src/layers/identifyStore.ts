import { create } from 'zustand'

interface IdentifyState {
  layerId: string | null
  properties: Record<string, unknown> | null
  setIdentified: (layerId: string | null, properties: Record<string, unknown> | null) => void
}

/** The last feature clicked on the map, scoped to whichever layer it belongs to -- the Inspect
 * panel only shows this when it matches the currently active layer. */
export const useIdentifyStore = create<IdentifyState>((set) => ({
  layerId: null,
  properties: null,
  setIdentified: (layerId, properties) => set({ layerId, properties }),
}))

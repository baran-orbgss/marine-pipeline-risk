import { create } from 'zustand'
import type { LayerDescriptor, LayerGroup } from '../types/api'

const GROUPS: LayerGroup[] = ['DATA', 'DERIVED', 'ANALYSIS', 'EVIDENCE', 'ASSETS']

interface LayerTreeState {
  visibleLayerIds: Record<string, boolean>
  activeLayerId: string | null
  collapsedGroups: Record<string, boolean>
  /** Visibility and active-layer selection are independent axes: toggling one must never touch
   * the other -- several layers may be visible, only one is ever "active" for inspection. */
  toggleVisibility: (layerId: string) => void
  setActiveLayer: (layerId: string) => void
  toggleGroupCollapsed: (group: LayerGroup) => void
  /** Fills in ONLY missing keys from each layer's own `default_visible` -- a re-fetch (e.g. after
   * a job completes) must never clobber a visibility toggle the user already made. */
  seedDefaults: (layers: LayerDescriptor[]) => void
  /** Drops visibility/active-layer state for layers that no longer exist (e.g. a project switch),
   * so a stale id never lingers as "active" for a layer that isn't there anymore. */
  pruneMissing: (availableLayerIds: string[]) => void
  reset: () => void
}

export const useLayerTreeStore = create<LayerTreeState>((set, get) => ({
  visibleLayerIds: {},
  activeLayerId: null,
  collapsedGroups: {},

  toggleVisibility: (layerId) =>
    set((state) => ({
      visibleLayerIds: { ...state.visibleLayerIds, [layerId]: !state.visibleLayerIds[layerId] },
    })),

  setActiveLayer: (layerId) => set({ activeLayerId: layerId }),

  toggleGroupCollapsed: (group) =>
    set((state) => ({
      collapsedGroups: { ...state.collapsedGroups, [group]: !state.collapsedGroups[group] },
    })),

  seedDefaults: (layers) => {
    const current = get().visibleLayerIds
    const next = { ...current }
    let changed = false
    for (const layer of layers) {
      if (!(layer.layer_id in next)) {
        next[layer.layer_id] = layer.display.default_visible
        changed = true
      }
    }
    if (changed) set({ visibleLayerIds: next })
  },

  pruneMissing: (availableLayerIds) => {
    const available = new Set(availableLayerIds)
    set((state) => {
      const visibleLayerIds = Object.fromEntries(
        Object.entries(state.visibleLayerIds).filter(([id]) => available.has(id)),
      )
      const activeLayerId =
        state.activeLayerId !== null && available.has(state.activeLayerId)
          ? state.activeLayerId
          : null
      return { visibleLayerIds, activeLayerId }
    })
  },

  reset: () => set({ visibleLayerIds: {}, activeLayerId: null, collapsedGroups: {} }),
}))

export function groupOrder(): LayerGroup[] {
  return GROUPS
}

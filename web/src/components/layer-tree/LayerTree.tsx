import { groupOrder, useLayerTreeStore } from '../../layers/layerTreeStore'
import type { LayerDescriptor } from '../../types/api'
import { EmptyState } from '../common/EmptyState'
import { LayerGroup } from './LayerGroup'
import styles from './LayerTree.module.css'

export function LayerTree({ layers }: { layers: LayerDescriptor[] }) {
  const visibleLayerIds = useLayerTreeStore((s) => s.visibleLayerIds)
  const activeLayerId = useLayerTreeStore((s) => s.activeLayerId)
  const collapsedGroups = useLayerTreeStore((s) => s.collapsedGroups)
  const toggleVisibility = useLayerTreeStore((s) => s.toggleVisibility)
  const setActiveLayer = useLayerTreeStore((s) => s.setActiveLayer)
  const toggleGroupCollapsed = useLayerTreeStore((s) => s.toggleGroupCollapsed)

  if (layers.length === 0) {
    return <EmptyState title="No layers yet" hint="Add data or run an analysis to see layers here." />
  }

  return (
    <div className={styles.tree}>
      {groupOrder().map((group) => (
        <LayerGroup
          key={group}
          group={group}
          layers={layers.filter((l) => l.group === group)}
          collapsed={Boolean(collapsedGroups[group])}
          onToggleCollapsed={() => toggleGroupCollapsed(group)}
          visibleLayerIds={visibleLayerIds}
          activeLayerId={activeLayerId}
          onToggleVisibility={toggleVisibility}
          onSelectLayer={setActiveLayer}
        />
      ))}
    </div>
  )
}

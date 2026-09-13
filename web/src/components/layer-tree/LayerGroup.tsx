import type { LayerDescriptor, LayerGroup as LayerGroupType } from '../../types/api'
import { LayerRow } from './LayerRow'
import styles from './LayerGroup.module.css'

const GROUP_LABELS: Record<LayerGroupType, string> = {
  DATA: 'Data',
  DERIVED: 'Derived',
  ANALYSIS: 'Analysis',
  EVIDENCE: 'Evidence',
  ASSETS: 'Assets',
}

export function LayerGroup({
  group,
  layers,
  collapsed,
  onToggleCollapsed,
  visibleLayerIds,
  activeLayerId,
  onToggleVisibility,
  onSelectLayer,
}: {
  group: LayerGroupType
  layers: LayerDescriptor[]
  collapsed: boolean
  onToggleCollapsed: () => void
  visibleLayerIds: Record<string, boolean>
  activeLayerId: string | null
  onToggleVisibility: (layerId: string) => void
  onSelectLayer: (layerId: string) => void
}) {
  if (layers.length === 0) return null
  const sorted = [...layers].sort((a, b) => a.display.z_index - b.display.z_index)

  return (
    <div className={styles.group}>
      <button type="button" className={styles.header} onClick={onToggleCollapsed}>
        <span className={styles.chevron}>{collapsed ? '▸' : '▾'}</span>
        {GROUP_LABELS[group]}
        <span className={styles.count}>{layers.length}</span>
      </button>
      {!collapsed && (
        <div>
          {sorted.map((layer) => (
            <LayerRow
              key={layer.layer_id}
              layer={layer}
              visible={Boolean(visibleLayerIds[layer.layer_id])}
              active={activeLayerId === layer.layer_id}
              onToggleVisibility={() => onToggleVisibility(layer.layer_id)}
              onSelect={() => onSelectLayer(layer.layer_id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

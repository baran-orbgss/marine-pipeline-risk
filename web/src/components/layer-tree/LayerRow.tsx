import type { LayerDescriptor } from '../../types/api'
import { SupportBadge } from './SupportBadge'
import styles from './LayerRow.module.css'

export function LayerRow({
  layer,
  visible,
  active,
  onToggleVisibility,
  onSelect,
}: {
  layer: LayerDescriptor
  visible: boolean
  active: boolean
  onToggleVisibility: () => void
  onSelect: () => void
}) {
  return (
    <div className={`${styles.row} ${active ? styles.active : ''}`}>
      <input
        type="checkbox"
        checked={visible}
        onChange={onToggleVisibility}
        aria-label={`Toggle visibility of ${layer.display.display_name}`}
        className={styles.checkbox}
      />
      <button type="button" className={styles.name} onClick={onSelect}>
        {layer.display.display_name}
        {!layer.role_established && <span className={styles.unclassified}>Unclassified</span>}
      </button>
      <SupportBadge supportType={layer.support_type} />
    </div>
  )
}

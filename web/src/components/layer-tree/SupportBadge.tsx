import { SUPPORT_TYPE_META } from '../../layers/supportType'
import type { SupportType } from '../../types/api'
import styles from './SupportBadge.module.css'

export function SupportBadge({ supportType }: { supportType: SupportType }) {
  const meta = SUPPORT_TYPE_META[supportType]
  return (
    <span className={styles.badge} title={meta.caption}>
      <span aria-hidden="true">{meta.glyph}</span>
      {meta.label}
    </span>
  )
}

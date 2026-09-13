import type { ReactNode } from 'react'
import styles from './Panel.module.css'

export function Panel({
  title,
  children,
  actions,
}: {
  title?: string
  children: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className={styles.panel}>
      {title && (
        <div className={styles.header}>
          <h3 className={styles.title}>{title}</h3>
          {actions}
        </div>
      )}
      <div className={styles.body}>{children}</div>
    </div>
  )
}

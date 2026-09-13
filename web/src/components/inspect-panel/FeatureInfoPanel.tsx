import type { TooltipField } from '../../types/api'
import styles from './FeatureInfoPanel.module.css'

/**
 * Iterates `tooltipFields` ONE AT A TIME and looks up each key from the feature's properties --
 * there is no `Object.entries(properties)`-style loop anywhere near this component. That is the
 * structural reason an extra backend/source property can never leak into the identify panel,
 * regardless of what the underlying GeoDataFrame happens to carry.
 */
export function FeatureInfoPanel({
  tooltipFields,
  properties,
}: {
  tooltipFields: TooltipField[]
  properties: Record<string, unknown> | null
}) {
  if (!properties) return null
  return (
    <dl className={styles.list}>
      {tooltipFields.map((field) => (
        <div key={field.key} className={styles.row}>
          <dt>{field.label}</dt>
          <dd>
            {formatValue(properties[field.key])}
            {field.unit && properties[field.key] !== undefined && ` ${field.unit}`}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function formatValue(value: unknown): string {
  if (value === undefined || value === null) return '—'
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '—'
  return String(value)
}

import type { StagedFile } from '../../types/api'
import styles from './DataReadinessPanel.module.css'

function checkRow(ok: boolean, label: string) {
  return (
    <li key={label} className={ok ? styles.ok : styles.missing}>
      <span aria-hidden="true">{ok ? '✓' : '—'}</span> {label}
    </li>
  )
}

export function DataReadinessPanel({ file }: { file: StagedFile }) {
  const { inspection } = file
  return (
    <div className={styles.panel}>
      <div className={styles.name}>{file.filename}</div>
      <ul className={styles.checks}>
        {checkRow(Boolean(inspection.observed_crs), 'CRS')}
        {checkRow(Boolean(inspection.bounds_native), 'Bounds')}
        {inspection.kind === 'raster' &&
          checkRow(Boolean(inspection.pixel_size_x), 'Pixel size')}
        {inspection.kind === 'vector' &&
          checkRow((inspection.feature_count ?? 0) > 0, 'Features')}
        {inspection.kind === 'tabular_unresolved' &&
          checkRow(inspection.coordinate_columns_declared, 'Coordinate columns declared')}
      </ul>
      {inspection.warnings.length > 0 && (
        <ul className={styles.warnings}>
          {inspection.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

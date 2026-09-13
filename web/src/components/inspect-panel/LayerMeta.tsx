import { SUPPORT_TYPE_META } from '../../layers/supportType'
import type { LayerDescriptor } from '../../types/api'
import styles from './LayerMeta.module.css'

export function LayerMeta({ layer }: { layer: LayerDescriptor }) {
  const support = SUPPORT_TYPE_META[layer.support_type]
  return (
    <div className={styles.meta}>
      <h2 className={styles.name}>{layer.display.display_name}</h2>
      <p className={styles.support}>{support.caption}</p>
      {!layer.role_established && (
        <p className={styles.unclassified}>
          Semantic role not established for this layer — classify it explicitly before relying on
          it.
        </p>
      )}
      <dl className={styles.kv}>
        {layer.crs_observed && (
          <>
            <dt>CRS</dt>
            <dd>{layer.crs_observed}</dd>
          </>
        )}
        {layer.feature_count !== null && (
          <>
            <dt>Features</dt>
            <dd>{layer.feature_count}</dd>
          </>
        )}
        {layer.band_count !== null && (
          <>
            <dt>Bands</dt>
            <dd>{layer.band_count}</dd>
          </>
        )}
        <dt>Source</dt>
        <dd className={styles.path}>{layer.relative_path}</dd>
      </dl>
    </div>
  )
}

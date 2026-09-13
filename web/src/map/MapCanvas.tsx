import { useMap } from './useMap'
import styles from './MapCanvas.module.css'

export function MapCanvas() {
  const { containerRef, basemapAvailable } = useMap()
  return (
    <div className={styles.wrapper}>
      <div ref={containerRef} className={styles.canvas} />
      {!basemapAvailable && (
        <div className={styles.basemapNotice}>
          Basemap unavailable — local project layers remain available.
        </div>
      )}
    </div>
  )
}

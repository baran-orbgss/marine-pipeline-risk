import { legendFromDisplaySpec } from '../../layers/legendFromDisplaySpec'
import type { DisplaySpec } from '../../types/api'
import styles from './Legend.module.css'

export function Legend({ display }: { display: DisplaySpec }) {
  const legend = legendFromDisplaySpec(display)
  // A note-only spec (e.g. hillshade's "visualization only", an unclassified raster's "semantic
  // role not established") still needs to render -- only skip entirely when there is truly
  // nothing to show.
  if (legend.stops.length === 0 && !legend.note) return null

  return (
    <div className={styles.legend}>
      <div className={styles.title}>
        {legend.title}
        {legend.unit && <span className={styles.unit}> ({legend.unit})</span>}
      </div>
      {legend.gradient ? (
        <div className={styles.gradientWrapper}>
          <div
            className={styles.gradientBar}
            style={{
              background: `linear-gradient(to right, ${legend.stops.map((s) => s.color).join(', ')})`,
            }}
          />
          <div className={styles.gradientLabels}>
            <span>{legend.stops[0].value}</span>
            <span>{legend.stops[legend.stops.length - 1].value}</span>
          </div>
        </div>
      ) : (
        <ul className={styles.swatchList}>
          {legend.stops.map((stop, index) => (
            <li key={`${stop.value}-${index}`} className={styles.swatchRow}>
              <span className={styles.swatch} style={{ background: stop.color }} />
              <span>{stop.label ?? stop.value}</span>
            </li>
          ))}
        </ul>
      )}
      {legend.note && <p className={styles.note}>{legend.note}</p>}
    </div>
  )
}

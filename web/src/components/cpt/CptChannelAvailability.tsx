import type { CptChannel } from '../../types/api'
import { computeChartGeometry, type ChartDimensions } from './chartGeometry'
import styles from './CptChannelAvailability.module.css'

const DIMS: ChartDimensions = { width: 120, height: 220, marginTop: 8, marginBottom: 20, marginLeft: 4, marginRight: 4 }

export function CptChannelAvailability({
  label,
  channel,
  depths,
}: {
  label: string
  channel: CptChannel
  depths: number[]
}) {
  if (!channel.available || channel.values === null) {
    return (
      <div className={styles.panel}>
        <div className={styles.label}>{label}</div>
        <div className={styles.unavailable}>Unavailable for this source</div>
      </div>
    )
  }

  const geometry = computeChartGeometry(depths, channel.values, DIMS)

  return (
    <div className={styles.panel}>
      <div className={styles.label}>
        {label} <span className={styles.unit}>({channel.unit})</span>
      </div>
      <svg viewBox={`0 0 ${DIMS.width} ${DIMS.height}`} className={styles.chart}>
        {geometry.yTicks.map((tick) => (
          <line
            key={tick.value}
            x1={geometry.xScaleRange[0]}
            x2={geometry.xScaleRange[1]}
            y1={tick.y}
            y2={tick.y}
            className={styles.gridline}
          />
        ))}
        <path d={geometry.path} className={styles.line} />
      </svg>
    </div>
  )
}

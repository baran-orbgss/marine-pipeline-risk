import type { AnalysisCapabilityDescriptor } from '../../types/api'
import { Badge } from '../common/Badge'
import styles from './CapabilityList.module.css'

const TONE: Record<AnalysisCapabilityDescriptor['availability'], 'success' | 'warning' | 'neutral'> = {
  AVAILABLE: 'success',
  MISSING_INPUTS: 'warning',
  NOT_APPLICABLE: 'neutral',
}

/**
 * Renders strictly from the backend-reported `availability` value -- the Run button's disabled
 * state is a direct function of `availability === 'AVAILABLE'`, nothing recomputed client-side.
 */
export function CapabilityList({
  capabilities,
  onRun,
  runningKey,
}: {
  capabilities: AnalysisCapabilityDescriptor[]
  onRun: (capabilityKey: string) => void
  runningKey: string | null
}) {
  return (
    <ul className={styles.list}>
      {capabilities.map((capability) => (
        <li key={capability.capability_key} className={styles.row}>
          <div className={styles.info}>
            <div className={styles.titleRow}>
              <span className={styles.title}>{capability.title}</span>
              <Badge tone={TONE[capability.availability]}>{capability.availability}</Badge>
            </div>
            {capability.reasons.map((reason) => (
              <p key={reason} className={styles.reason}>
                {reason}
              </p>
            ))}
          </div>
          <button
            type="button"
            className={styles.runButton}
            disabled={capability.availability !== 'AVAILABLE' || runningKey === capability.capability_key}
            onClick={() => onRun(capability.capability_key)}
          >
            {runningKey === capability.capability_key ? 'Running…' : 'Run'}
          </button>
        </li>
      ))}
    </ul>
  )
}

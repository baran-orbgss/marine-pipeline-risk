import { useCptProfileQuery } from '../../api/queries'
import { useCptSelectionStore } from '../../layers/cptSelectionStore'
import type { LayerDescriptor } from '../../types/api'
import { EmptyState } from '../common/EmptyState'
import { CptChannelAvailability } from './CptChannelAvailability'
import styles from './CptProfileChart.module.css'

const CHANNEL_LABELS: Record<'qc' | 'fs' | 'u2' | 'qt', string> = {
  qc: 'qc',
  fs: 'fs',
  u2: 'u2',
  qt: 'qt',
}

export function CptProfileChart({ layer }: { layer: LayerDescriptor }) {
  const selectedTestId = useCptSelectionStore((s) => s.selectedTestId)
  const { data: profile, isLoading } = useCptProfileQuery(
    layer.cpt_profile_url_template,
    selectedTestId,
  )

  if (!selectedTestId) {
    return <EmptyState title="No CPT selected" hint="Click a CPT point on the map to see its profile." />
  }
  if (isLoading || !profile) {
    return <EmptyState title="Loading profile…" />
  }

  return (
    <div className={styles.wrapper}>
      <h3 className={styles.title}>{profile.test_id}</h3>
      <div className={styles.channels}>
        {(Object.keys(CHANNEL_LABELS) as (keyof typeof CHANNEL_LABELS)[]).map((key) => (
          <CptChannelAvailability
            key={key}
            label={CHANNEL_LABELS[key]}
            channel={profile.channels[key]}
            depths={profile.depth_bsf_m}
          />
        ))}
      </div>
      <p className={styles.axisNote}>Depth below seabed (m), top → bottom</p>
    </div>
  )
}

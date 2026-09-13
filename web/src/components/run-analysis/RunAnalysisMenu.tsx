import { useState } from 'react'
import { useCapabilitiesQuery, useRunCapabilityMutation } from '../../api/queries'
import { EmptyState } from '../common/EmptyState'
import { CapabilityList } from './CapabilityList'
import { RunAnalysisJobPanel } from './RunAnalysisJobPanel'
import styles from './RunAnalysisMenu.module.css'

export function RunAnalysisMenu({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const { data: capabilities } = useCapabilitiesQuery(projectId)
  const runMutation = useRunCapabilityMutation(projectId)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)

  const handleRun = (capabilityKey: string) => {
    runMutation.mutate(
      { capabilityKey },
      { onSuccess: (job) => setActiveJobId(job.job_id) },
    )
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.menu} onClick={(event) => event.stopPropagation()}>
        <div className={styles.header}>
          <span>Run Analysis</span>
          <button type="button" className={styles.close} onClick={onClose}>
            ✕
          </button>
        </div>
        {activeJobId && <RunAnalysisJobPanel jobId={activeJobId} projectId={projectId} />}
        {capabilities && capabilities.length > 0 ? (
          <CapabilityList
            capabilities={capabilities}
            onRun={handleRun}
            runningKey={runMutation.isPending ? (runMutation.variables?.capabilityKey ?? null) : null}
          />
        ) : (
          <EmptyState title="No capabilities available" />
        )}
      </div>
    </div>
  )
}

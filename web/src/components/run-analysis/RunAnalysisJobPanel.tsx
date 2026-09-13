import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useJobStatusQuery } from '../../api/queries'
import type { SimpleJobStatus } from '../../types/api'
import styles from './RunAnalysisJobPanel.module.css'

const STEPS: SimpleJobStatus[] = ['Preparing data', 'Running analysis', 'Building map layer', 'Ready']

/**
 * Renders only the four ticket-defined states plus a failure banner -- this component never
 * reads or displays `stdout`/`stderr` even if such a field is present on the job payload; raw
 * logs stay Engineering-View-only (see `/api/jobs/{id}/log`).
 */
export function RunAnalysisJobPanel({ jobId, projectId }: { jobId: string; projectId: string }) {
  const { data: job } = useJobStatusQuery(jobId)
  const queryClient = useQueryClient()

  useEffect(() => {
    if (job?.status === 'SUCCEEDED') {
      queryClient.invalidateQueries({ queryKey: ['layers', projectId] })
      queryClient.invalidateQueries({ queryKey: ['project', projectId] })
    }
  }, [job?.status, projectId, queryClient])

  if (!job) return null

  if (job.status === 'FAILED') {
    return (
      <div className={styles.failureBanner}>
        Analysis failed — see Engineering view for details.
      </div>
    )
  }

  const currentIndex = STEPS.indexOf(job.simple_status as (typeof STEPS)[number])

  return (
    <div className={styles.stepper}>
      {STEPS.map((step, index) => (
        <div
          key={step}
          className={`${styles.step} ${index <= currentIndex ? styles.stepDone : ''} ${
            index === currentIndex ? styles.stepCurrent : ''
          }`}
        >
          {step}
        </div>
      ))}
    </div>
  )
}

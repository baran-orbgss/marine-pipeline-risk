import type { JobRecord } from '../types/api'
import { apiGet, apiPost } from './client'

export interface RunCapabilityRequest {
  allow_network?: boolean
  scenario_manifest_path?: string | null
}

export function runCapability(
  projectId: string,
  capabilityKey: string,
  request: RunCapabilityRequest = {},
): Promise<JobRecord> {
  return apiPost<JobRecord>(
    `/projects/${encodeURIComponent(projectId)}/capabilities/${encodeURIComponent(capabilityKey)}/run`,
    request,
  )
}

export function fetchJob(jobId: string): Promise<JobRecord> {
  return apiGet<JobRecord>(`/jobs/${encodeURIComponent(jobId)}`)
}

export const TERMINAL_JOB_STATES = new Set(['SUCCEEDED', 'FAILED'])

export function isTerminalJob(job: JobRecord | undefined): boolean {
  return job !== undefined && TERMINAL_JOB_STATES.has(job.status)
}

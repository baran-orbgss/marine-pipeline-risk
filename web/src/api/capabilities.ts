import type { AnalysisCapabilityDescriptor } from '../types/api'
import { apiGet } from './client'

export function fetchCapabilities(projectId: string): Promise<AnalysisCapabilityDescriptor[]> {
  return apiGet<AnalysisCapabilityDescriptor[]>(
    `/projects/${encodeURIComponent(projectId)}/capabilities`,
  )
}

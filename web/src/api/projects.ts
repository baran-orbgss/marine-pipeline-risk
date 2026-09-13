import type { ProjectCatalog, ProjectSummary } from '../types/api'
import { apiGet, apiPost } from './client'

export function fetchProjectCatalog(): Promise<ProjectCatalog> {
  return apiGet<ProjectCatalog>('/projects')
}

export function fetchProject(projectId: string): Promise<ProjectSummary> {
  return apiGet<ProjectSummary>(`/projects/${encodeURIComponent(projectId)}`)
}

export function refreshProject(projectId: string): Promise<ProjectSummary> {
  return apiPost<ProjectSummary>(`/projects/${encodeURIComponent(projectId)}/refresh`)
}

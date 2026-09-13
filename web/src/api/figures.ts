import { apiGet, rawUrl } from './client'

export function fetchFigurePaths(projectId: string): Promise<string[]> {
  return apiGet<string[]>(`/projects/${encodeURIComponent(projectId)}/figures`)
}

export function figureUrl(projectId: string, figurePath: string): string {
  return rawUrl(`/projects/${encodeURIComponent(projectId)}/figures/${figurePath}`)
}

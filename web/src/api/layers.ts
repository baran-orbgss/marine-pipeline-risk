import type { LayerCatalog } from '../types/api'
import { apiGet } from './client'

export function fetchLayerCatalog(projectId: string): Promise<LayerCatalog> {
  return apiGet<LayerCatalog>(`/projects/${encodeURIComponent(projectId)}/layers`)
}

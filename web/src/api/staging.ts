import type { AssetDeclaration, StagedFile, StagingSessionRecord } from '../types/api'
import { apiGet, apiPost, rawUrl } from './client'

export function createStagingSession(): Promise<StagingSessionRecord> {
  return apiPost<StagingSessionRecord>('/staging/sessions')
}

export async function uploadStagingFile(sessionId: string, file: File): Promise<StagedFile> {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch(rawUrl(`/staging/sessions/${encodeURIComponent(sessionId)}/files`), {
    method: 'POST',
    body: form,
  })
  if (!response.ok) throw new Error(`upload failed for ${file.name}`)
  return (await response.json()) as StagedFile
}

export function listStagingFiles(sessionId: string): Promise<StagedFile[]> {
  return apiGet<StagedFile[]>(`/staging/sessions/${encodeURIComponent(sessionId)}/files`)
}

export function declareCoordinateColumns(
  sessionId: string,
  fileId: string,
  request: { x_column: string; y_column: string; crs: string },
): Promise<StagedFile> {
  return apiPost<StagedFile>(
    `/staging/sessions/${encodeURIComponent(sessionId)}/files/${encodeURIComponent(fileId)}/coordinate-columns`,
    request,
  )
}

export interface PromoteRequest {
  project_id: string
  display_name: string
  working_crs: string
  asset_declarations?: Record<string, AssetDeclaration>
}

export function promoteSession(
  sessionId: string,
  request: PromoteRequest,
): Promise<{ manifest_path: string }> {
  return apiPost(`/staging/sessions/${encodeURIComponent(sessionId)}/promote`, request)
}

export function stagingFeaturesUrl(sessionId: string, fileId: string): string {
  return rawUrl(
    `/staging/sessions/${encodeURIComponent(sessionId)}/files/${encodeURIComponent(fileId)}/features`,
  )
}

export function stagingTileUrlTemplate(sessionId: string, fileId: string): string {
  return rawUrl(
    `/staging/sessions/${encodeURIComponent(sessionId)}/files/${encodeURIComponent(fileId)}/tiles/{z}/{x}/{y}.png`,
  )
}

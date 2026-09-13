const PROJECT_PARAM = 'project'

export function readProjectIdFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get(PROJECT_PARAM)
}

export function writeProjectIdToUrl(projectId: string | null): void {
  const params = new URLSearchParams(window.location.search)
  if (projectId) {
    params.set(PROJECT_PARAM, projectId)
  } else {
    params.delete(PROJECT_PARAM)
  }
  const query = params.toString()
  const next = `${window.location.pathname}${query ? `?${query}` : ''}`
  window.history.replaceState(null, '', next)
}

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchCapabilities } from './capabilities'
import { fetchCptProfile } from './cpt'
import { fetchJob, isTerminalJob, runCapability, type RunCapabilityRequest } from './jobs'
import { fetchLayerCatalog } from './layers'
import { fetchProject, fetchProjectCatalog } from './projects'

export function useProjectCatalogQuery() {
  return useQuery({ queryKey: ['projects'], queryFn: fetchProjectCatalog })
}

export function useProjectQuery(projectId: string | null) {
  return useQuery({
    queryKey: ['project', projectId],
    queryFn: () => fetchProject(projectId as string),
    enabled: projectId !== null,
  })
}

export function useLayerCatalogQuery(projectId: string | null) {
  return useQuery({
    queryKey: ['layers', projectId],
    queryFn: () => fetchLayerCatalog(projectId as string),
    enabled: projectId !== null,
  })
}

export function useCapabilitiesQuery(projectId: string | null) {
  return useQuery({
    queryKey: ['capabilities', projectId],
    queryFn: () => fetchCapabilities(projectId as string),
    enabled: projectId !== null,
  })
}

export function useCptProfileQuery(profileUrlTemplate: string | null, testId: string | null) {
  return useQuery({
    queryKey: ['cpt-profile', profileUrlTemplate, testId],
    queryFn: () => fetchCptProfile(profileUrlTemplate as string, testId as string),
    enabled: profileUrlTemplate !== null && testId !== null,
  })
}

export function useJobStatusQuery(jobId: string | null) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => fetchJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => (isTerminalJob(query.state.data) ? false : 1500),
  })
}

export function useRunCapabilityMutation(projectId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      capabilityKey,
      request,
    }: {
      capabilityKey: string
      request?: RunCapabilityRequest
    }) => runCapability(projectId, capabilityKey, request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['capabilities', projectId] })
    },
  })
}

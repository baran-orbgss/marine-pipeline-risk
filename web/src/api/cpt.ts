import type { CptProfile } from '../types/api'
import { ApiRequestError } from './client'

/**
 * `profileUrlTemplate` is `LayerDescriptor.cpt_profile_url_template`, a server-built URL with a
 * literal `{test_id}` placeholder (mirrors `tile_url_template`'s `{z}/{x}/{y}`). Its `layer_id`
 * segment is already an opaque, server-encoded token -- rebuilding this URL client-side from a raw
 * `layer_id` would reintroduce the routing bug that opaque encoding exists to prevent (see
 * api/layers.py's `_url_segment`), since a raw id can contain '/'.
 */
export async function fetchCptProfile(profileUrlTemplate: string, testId: string): Promise<CptProfile> {
  const url = profileUrlTemplate.replace('{test_id}', encodeURIComponent(testId))
  const response = await fetch(url)
  if (!response.ok) {
    throw new ApiRequestError(response.status, response.statusText)
  }
  return (await response.json()) as CptProfile
}

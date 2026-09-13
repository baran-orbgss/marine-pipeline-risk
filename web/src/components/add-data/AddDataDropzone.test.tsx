import { fireEvent, render, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { StagedFile, StagingSessionRecord } from '../../types/api'
import { AddDataDropzone } from './AddDataDropzone'

const { createStagingSession, uploadStagingFile } = vi.hoisted(() => ({
  createStagingSession: vi.fn(),
  uploadStagingFile: vi.fn(),
}))

vi.mock('../../api/staging', () => ({
  createStagingSession,
  uploadStagingFile,
  promoteSession: vi.fn(),
}))

const SESSION: StagingSessionRecord = { session_id: 'session-1', created_at: '2026-01-01T00:00:00Z' }

const STAGED: StagedFile = {
  file_id: 'file-1',
  filename: 'dropped.geojson',
  relative_path: 'data/staging/session-1/dropped.geojson',
  inspection: {
    kind: 'vector',
    observed_crs: 'EPSG:4326',
    bounds_native: null,
    bounds_wgs84: null,
    pixel_size_x: null,
    pixel_size_y: null,
    width: null,
    height: null,
    band_count: null,
    nodata: null,
    geometry_type: 'Point',
    feature_count: 2,
    fields: [],
    coordinate_columns_declared: false,
    warnings: [],
  },
}

function dropFile(dropzone: Element) {
  const file = new File(['{"type":"FeatureCollection","features":[]}'], 'dropped.geojson', {
    type: 'application/geo+json',
  })
  fireEvent.drop(dropzone, { dataTransfer: { files: [file] } })
}

describe('AddDataDropzone', () => {
  it('notifies the parent exactly once per upload, without an update-depth loop', async () => {
    // Regression guard: notifying the parent from inside the AddDataDropzone render (or from an
    // effect that lists the parent's inline callback as a dependency) either violates React's
    // render-purity rules or re-fires on every resulting parent re-render -- an infinite loop
    // (observed directly as a live "Maximum update depth exceeded" crash while fixing this). The
    // parent must hear about a real upload exactly once.
    createStagingSession.mockResolvedValue(SESSION)
    uploadStagingFile.mockResolvedValue(STAGED)
    const onSessionReady = vi.fn()

    const { container } = render(<AddDataDropzone onSessionReady={onSessionReady} onClose={() => {}} />)
    const dropzone = container.querySelector('[class*="dropzone"]')
    if (!dropzone) throw new Error('dropzone not found')

    dropFile(dropzone)

    await waitFor(() => expect(onSessionReady).toHaveBeenCalled())
    expect(onSessionReady).toHaveBeenCalledTimes(1)
    expect(onSessionReady).toHaveBeenCalledWith('session-1', [STAGED])
  })

  it('reuses the same staging session across a second drop instead of creating a new one', async () => {
    createStagingSession.mockResolvedValue(SESSION)
    uploadStagingFile.mockResolvedValue(STAGED)
    const onSessionReady = vi.fn()

    const { container } = render(<AddDataDropzone onSessionReady={onSessionReady} onClose={() => {}} />)
    const dropzone = container.querySelector('[class*="dropzone"]')
    if (!dropzone) throw new Error('dropzone not found')

    dropFile(dropzone)
    await waitFor(() => expect(onSessionReady).toHaveBeenCalledTimes(1))

    dropFile(dropzone)
    await waitFor(() => expect(onSessionReady).toHaveBeenCalledTimes(2))

    expect(createStagingSession).toHaveBeenCalledTimes(1)
    expect(onSessionReady).toHaveBeenLastCalledWith('session-1', [STAGED, STAGED])
  })
})

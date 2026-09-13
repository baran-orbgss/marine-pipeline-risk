import { useEffect, useMemo, useState } from 'react'
import { CptProfileChart } from '../components/cpt/CptProfileChart'
import { LayerTree } from '../components/layer-tree/LayerTree'
import { InspectPanel } from '../components/inspect-panel/InspectPanel'
import { useCptSelectionStore } from '../layers/cptSelectionStore'
import { useLayerTreeStore } from '../layers/layerTreeStore'
import { stagedFileToLayerDescriptor } from '../layers/stagedFileToLayerDescriptor'
import { MapCanvas } from '../map/MapCanvas'
import { useCptSelection } from '../map/useCptSelection'
import { useMapIdentify } from '../map/useMapIdentify'
import { useMap } from '../map/useMap'
import { useLayerCatalogQuery, useProjectQuery } from '../api/queries'
import type { LayerDescriptor, StagedFile } from '../types/api'
import { useSyncLayersToMap } from '../layers/useSyncLayersToMap'
import { readProjectIdFromUrl, writeProjectIdToUrl } from './urlState'
import { Toolbar } from './Toolbar'
import styles from './AppShell.module.css'

export function AppShell() {
  const [activeProjectId, setActiveProjectId] = useState<string | null>(readProjectIdFromUrl)
  const [stagingSession, setStagingSession] = useState<{ sessionId: string; files: StagedFile[] } | null>(
    null,
  )
  const { controller } = useMap()

  const { data: project } = useProjectQuery(activeProjectId)
  const { data: catalog } = useLayerCatalogQuery(activeProjectId)

  const stagingLayers = useMemo(
    () =>
      stagingSession
        ? stagingSession.files.map((file) => stagedFileToLayerDescriptor(stagingSession.sessionId, file))
        : [],
    [stagingSession],
  )

  const layers: LayerDescriptor[] = useMemo(
    () => [...(catalog?.layers ?? []), ...stagingLayers],
    [catalog, stagingLayers],
  )
  const scopeId = stagingSession ? `staging:${stagingSession.sessionId}` : activeProjectId

  const visibleLayerIds = useLayerTreeStore((s) => s.visibleLayerIds)
  const activeLayerId = useLayerTreeStore((s) => s.activeLayerId)
  const seedDefaults = useLayerTreeStore((s) => s.seedDefaults)
  const pruneMissing = useLayerTreeStore((s) => s.pruneMissing)
  const cptSelectedTestId = useCptSelectionStore((s) => s.selectedTestId)

  useEffect(() => {
    seedDefaults(layers)
    pruneMissing(layers.map((l) => l.layer_id))
  }, [layers, seedDefaults, pruneMissing])

  useSyncLayersToMap(controller, layers, visibleLayerIds, scopeId)

  const pointLayers = useMemo(() => layers.filter((l) => l.support_type === 'POINT_EVIDENCE'), [layers])
  const vectorLayers = useMemo(() => layers.filter((l) => l.layer_type === 'vector'), [layers])
  useCptSelection(controller, pointLayers)
  useMapIdentify(controller, vectorLayers)

  useEffect(() => {
    const extent = project?.extent_wgs84
    if (controller && extent) {
      controller.flyToBounds(extent)
    }
  }, [controller, project?.extent_wgs84])

  useEffect(() => {
    if (!stagingSession) return
    if (stagingLayers.length === 0) return
    const withBounds = stagingLayers.find((l) => l.bounds_wgs84)
    if (controller && withBounds?.bounds_wgs84) {
      controller.flyToBounds(withBounds.bounds_wgs84)
    }
  }, [controller, stagingSession, stagingLayers])

  const handleSelectProject = (projectId: string) => {
    setStagingSession(null)
    setActiveProjectId(projectId)
    writeProjectIdToUrl(projectId)
  }

  const activeLayer = layers.find((l) => l.layer_id === activeLayerId) ?? null
  const cptLayerForProfile = pointLayers.find((l) => l.layer_id === activeLayerId) ?? pointLayers[0]

  return (
    <div className={styles.shell}>
      <Toolbar
        activeProjectId={activeProjectId}
        onSelectProject={handleSelectProject}
        onStagingSessionReady={(sessionId, files) => setStagingSession({ sessionId, files })}
      />
      <div className={styles.body}>
        <aside className={styles.left}>
          <LayerTree layers={layers} />
        </aside>
        <main className={styles.center}>
          <MapCanvas />
        </main>
        <aside className={styles.right}>
          <InspectPanel projectId={scopeId ?? ''} activeLayer={activeLayer} />
        </aside>
      </div>
      {cptSelectedTestId && cptLayerForProfile && activeProjectId && (
        <div className={styles.bottom}>
          <CptProfileChart layer={cptLayerForProfile} />
        </div>
      )}
    </div>
  )
}

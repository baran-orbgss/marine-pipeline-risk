import { useState } from 'react'
import { AddDataDropzone } from '../components/add-data/AddDataDropzone'
import { RunAnalysisMenu } from '../components/run-analysis/RunAnalysisMenu'
import type { StagedFile } from '../types/api'
import { ProjectPicker } from './ProjectPicker'
import styles from './Toolbar.module.css'

const ENGINEERING_WORKBENCH_URL = 'http://localhost:8501'

export function Toolbar({
  activeProjectId,
  onSelectProject,
  onStagingSessionReady,
}: {
  activeProjectId: string | null
  onSelectProject: (projectId: string) => void
  onStagingSessionReady: (sessionId: string, files: StagedFile[]) => void
}) {
  const [showAddData, setShowAddData] = useState(false)
  const [showRunAnalysis, setShowRunAnalysis] = useState(false)

  return (
    <div className={styles.toolbar}>
      <span className={styles.brand}>MARINE GIS</span>
      <ProjectPicker activeProjectId={activeProjectId} onSelect={onSelectProject} />
      <button type="button" className={styles.button} onClick={() => setShowAddData(true)}>
        + Add Data
      </button>
      <button type="button" className={styles.button} disabled title="Coming soon">
        Add Asset
      </button>
      <button type="button" className={styles.button} disabled title="Optional advanced operation">
        Analysis Area
      </button>
      <button
        type="button"
        className={styles.button}
        disabled={!activeProjectId}
        onClick={() => setShowRunAnalysis(true)}
      >
        Run Analysis ▾
      </button>
      <button type="button" className={styles.button} disabled title="Coming soon">
        Export
      </button>
      <button type="button" className={styles.button} disabled title="Coming soon">
        Report
      </button>
      <a
        className={styles.engineeringLink}
        href={ENGINEERING_WORKBENCH_URL}
        target="_blank"
        rel="noreferrer"
      >
        Engineering
      </a>

      {showAddData && (
        <AddDataDropzone
          onSessionReady={onStagingSessionReady}
          onClose={() => setShowAddData(false)}
        />
      )}
      {showRunAnalysis && activeProjectId && (
        <RunAnalysisMenu projectId={activeProjectId} onClose={() => setShowRunAnalysis(false)} />
      )}
    </div>
  )
}

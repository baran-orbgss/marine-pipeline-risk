import { useEffect, useRef, useState } from 'react'
import { createStagingSession, promoteSession, uploadStagingFile } from '../../api/staging'
import type { StagedFile } from '../../types/api'
import { DataReadinessPanel } from './DataReadinessPanel'
import styles from './AddDataDropzone.module.css'

export function AddDataDropzone({
  onSessionReady,
  onClose,
}: {
  onSessionReady: (sessionId: string, files: StagedFile[]) => void
  onClose: () => void
}) {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [files, setFiles] = useState<StagedFile[]>([])
  const [projectName, setProjectName] = useState('Untitled Project')
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFiles = async (fileList: FileList) => {
    let currentSessionId = sessionId
    if (!currentSessionId) {
      const session = await createStagingSession()
      currentSessionId = session.session_id
      setSessionId(currentSessionId)
    }
    const uploaded: StagedFile[] = []
    for (const file of Array.from(fileList)) {
      const staged = await uploadStagingFile(currentSessionId, file)
      uploaded.push(staged)
    }
    setFiles((prev) => [...prev, ...uploaded])
  }

  // Notifying the parent belongs in an effect, not inside the `setFiles` updater above -- calling
  // it there updates AppShell's state while AddDataDropzone is still rendering (React warns
  // "Cannot update a component while rendering a different component"), which is exactly the kind
  // of render-purity violation that breaks under Strict Mode / concurrent rendering.
  useEffect(() => {
    if (sessionId && files.length > 0) onSessionReady(sessionId, files)
    // `onSessionReady` is a fresh inline callback on every AppShell render (it closes over
    // `setStagingSession`) -- listing it here would re-fire this effect after every resulting
    // parent re-render, which calls it again, which re-renders the parent again, forever. Only
    // `sessionId`/`files` changing should trigger a notification; the latest `onSessionReady` is
    // still what runs, since the effect closure is rebuilt fresh each render regardless.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, files])

  const handleSave = async () => {
    if (!sessionId || files.length === 0) return
    const workingCrs = files.find((f) => f.inspection.observed_crs)?.inspection.observed_crs
    await promoteSession(sessionId, {
      project_id: projectName,
      display_name: projectName,
      working_crs: workingCrs ?? 'EPSG:4326',
    })
    onClose()
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <span>Add Data</span>
          <button type="button" className={styles.close} onClick={onClose}>
            ✕
          </button>
        </div>
        <div
          className={`${styles.dropzone} ${isDragging ? styles.dragging : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setIsDragging(false)
            if (e.dataTransfer.files.length > 0) void handleFiles(e.dataTransfer.files)
          }}
          onClick={() => inputRef.current?.click()}
        >
          <p>Drop a GeoTIFF, GeoPackage, or GeoJSON file here, or click to browse.</p>
          <input
            ref={inputRef}
            type="file"
            accept=".tif,.tiff,.gpkg,.geojson,.json,.csv"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) void handleFiles(e.target.files)
            }}
          />
        </div>
        {files.length > 0 && (
          <div className={styles.results}>
            {files.map((file) => (
              <DataReadinessPanel key={file.file_id} file={file} />
            ))}
            <div className={styles.saveRow}>
              <input
                className={styles.nameInput}
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                placeholder="Project name"
              />
              <button type="button" className={styles.saveButton} onClick={() => void handleSave()}>
                Save Project
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

import { useProjectCatalogQuery } from '../api/queries'
import styles from './ProjectPicker.module.css'

export function ProjectPicker({
  activeProjectId,
  onSelect,
}: {
  activeProjectId: string | null
  onSelect: (projectId: string) => void
}) {
  const { data: catalog, isLoading } = useProjectCatalogQuery()

  return (
    <select
      className={styles.select}
      value={activeProjectId ?? ''}
      onChange={(event) => onSelect(event.target.value)}
      disabled={isLoading}
    >
      <option value="" disabled>
        {isLoading ? 'Loading projects…' : 'Select a project'}
      </option>
      {catalog?.projects.map((project) => (
        <option key={project.project_id} value={project.project_id}>
          {project.display_name}
        </option>
      ))}
    </select>
  )
}

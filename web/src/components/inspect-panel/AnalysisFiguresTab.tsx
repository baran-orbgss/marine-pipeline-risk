import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { fetchFigurePaths, figureUrl } from '../../api/figures'
import { EmptyState } from '../common/EmptyState'
import styles from './AnalysisFiguresTab.module.css'

export function AnalysisFiguresTab({ projectId }: { projectId: string }) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const { data: figures } = useQuery({
    queryKey: ['figures', projectId],
    queryFn: () => fetchFigurePaths(projectId),
  })

  if (!figures || figures.length === 0) {
    return <EmptyState title="No figures" hint="Existing scientific PNG figures appear here." />
  }

  return (
    <div className={styles.wrapper}>
      <p className={styles.hint}>Reference material from prior engine runs — not the primary map.</p>
      <div className={styles.grid}>
        {figures.map((path) => (
          <button key={path} className={styles.thumb} onClick={() => setExpanded(path)} type="button">
            <img src={figureUrl(projectId, path)} alt={path} loading="lazy" />
            <span className={styles.caption}>{path.split('/').pop()}</span>
          </button>
        ))}
      </div>
      {expanded && (
        <div className={styles.lightbox} onClick={() => setExpanded(null)}>
          <img src={figureUrl(projectId, expanded)} alt={expanded} />
        </div>
      )}
    </div>
  )
}

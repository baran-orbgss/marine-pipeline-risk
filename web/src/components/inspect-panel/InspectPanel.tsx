import { useState } from 'react'
import { useIdentifyStore } from '../../layers/identifyStore'
import type { LayerDescriptor } from '../../types/api'
import { EmptyState } from '../common/EmptyState'
import { AnalysisFiguresTab } from './AnalysisFiguresTab'
import { FeatureInfoPanel } from './FeatureInfoPanel'
import { Legend } from './Legend'
import { LayerMeta } from './LayerMeta'
import { LimitationsList } from './LimitationsList'
import styles from './InspectPanel.module.css'

export function InspectPanel({
  projectId,
  activeLayer,
}: {
  projectId: string
  activeLayer: LayerDescriptor | null
}) {
  const [tab, setTab] = useState<'inspect' | 'figures'>('inspect')
  const identified = useIdentifyStore()

  return (
    <div className={styles.panel}>
      <div className={styles.tabs}>
        <button
          type="button"
          className={tab === 'inspect' ? styles.tabActive : styles.tab}
          onClick={() => setTab('inspect')}
        >
          Inspect
        </button>
        <button
          type="button"
          className={tab === 'figures' ? styles.tabActive : styles.tab}
          onClick={() => setTab('figures')}
        >
          Analysis Figures
        </button>
      </div>
      <div className={styles.content}>
        {tab === 'figures' ? (
          <AnalysisFiguresTab projectId={projectId} />
        ) : activeLayer ? (
          <>
            <LayerMeta layer={activeLayer} />
            <Legend display={activeLayer.display} />
            {identified.layerId === activeLayer.layer_id && (
              <FeatureInfoPanel
                tooltipFields={activeLayer.display.tooltip_fields}
                properties={identified.properties}
              />
            )}
            <LimitationsList limitations={activeLayer.display.scientific_limitations} />
          </>
        ) : (
          <EmptyState title="No layer selected" hint="Click a layer name in the layer tree." />
        )}
      </div>
    </div>
  )
}

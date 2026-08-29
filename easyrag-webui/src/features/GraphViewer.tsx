import { CSSProperties, useCallback, useEffect, useState } from 'react'
import { SigmaContainer } from '@react-sigma/core'
import '@react-sigma/core/lib/style.css'
import { createNodeBorderProgram } from '@sigma/node-border'
import { createEdgeCurveProgram } from '@sigma/edge-curve'
import { NodeCircleProgram, EdgeLineProgram, EdgeArrowProgram } from 'sigma/rendering'
import type { Settings } from 'sigma/settings'

import GraphControl from '@/components/graph/GraphControl'
import GraphLabels from '@/components/graph/GraphLabels'
import GraphSearch from '@/components/graph/GraphSearch'
import PropertiesView from '@/components/graph/PropertiesView'
import ZoomControl from '@/components/graph/ZoomControl'
import FullScreenControl from '@/components/graph/FullScreenControl'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import { graphApi } from '@/api/easyrag'
import useEasyRagGraph, { NodeType, EdgeType } from '@/hooks/useEasyRagGraph'
import useIsDarkMode from '@/hooks/useIsDarkMode'

const sigmaSettings: Partial<Settings<NodeType, EdgeType>> = {
  renderLabels: true,
  renderEdgeLabels: false,
  labelRenderedSizeThreshold: 1,
  defaultNodeType: 'bordered',
  defaultEdgeType: 'curvedNoArrow',
  nodeProgramClasses: {
    circle: NodeCircleProgram,
    bordered: createNodeBorderProgram({
      borders: [
        {
          color: { attribute: 'borderColor', defaultValue: '#FFFFFF' },
          size: { attribute: 'borderSize', defaultValue: 0.12, mode: 'relative' }
        },
        {
          color: { attribute: 'color', defaultValue: '#5D6D7E' },
          size: { fill: true }
        }
      ]
    })
  },
  edgeProgramClasses: {
    line: EdgeLineProgram,
    arrow: EdgeArrowProgram,
    rect: EdgeLineProgram,
    curvedNoArrow: createEdgeCurveProgram()
  }
}

const GraphViewer = () => {
  // Drives graph data fetching, node expand/prune and retries
  useEasyRagGraph()

  const isFetching = useGraphStore.use.isFetching()
  const graphIsEmpty = useGraphStore.use.graphIsEmpty()
  const graphNodeCount = useGraphStore.use.graphNodeCount()
  const graphEdgeCount = useGraphStore.use.graphEdgeCount()
  const showNodeSearchBar = useSettingsStore.use.showNodeSearchBar()
  const showPropertyPanel = useSettingsStore.use.showPropertyPanel()
  const workspace = useSettingsStore.use.workspace()
  const isDarkTheme = useIsDarkMode()
  const [graphStats, setGraphStats] = useState<{ node_count: number; edge_count: number } | null>(null)

  // Refresh stats on mount and whenever workspace changes.
  useEffect(() => {
    let cancelled = false
    graphApi
      .getStats()
      .then((s) => {
        if (!cancelled) setGraphStats(s)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [workspace])

  const handleRefresh = useCallback(() => {
    graphApi.getStats().then(setGraphStats).catch(() => {})
    const state = useGraphStore.getState()
    state.setGraphDataFetchAttempted(false)
    state.incrementGraphDataVersion()
  }, [])

  const containerStyle = {
    '--sigma-background-color': isDarkTheme ? 'hsl(240 10% 3.9%)' : 'hsl(0 0% 100%)'
  } as CSSProperties

  return (
    <div className="relative h-full w-full overflow-hidden flex flex-col">
      {/* Graph info bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b bg-background/95">
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span>Nodes: {graphNodeCount}</span>
          <span>Edges: {graphEdgeCount}</span>
          {graphStats && (
            <span className="text-xs">
              (Total: {graphStats.node_count} nodes, {graphStats.edge_count} edges)
            </span>
          )}
        </div>
        <button
          onClick={handleRefresh}
          className="text-xs px-2 py-1 rounded bg-primary/10 hover:bg-primary/20 transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Graph visualization area */}
      <div className="flex-1 relative" style={containerStyle}>
        <SigmaContainer settings={sigmaSettings}>
          <GraphControl />

          {showNodeSearchBar && (
            <div className="absolute top-2 left-2 z-10 w-56 rounded-lg border bg-background/90 p-2 shadow-lg backdrop-blur">
              <GraphSearch />
            </div>
          )}

          <div className="absolute top-2 right-2 z-10 w-64 rounded-lg border bg-background/90 shadow-lg backdrop-blur">
            <GraphLabels />
          </div>

          <div className="absolute bottom-3 left-3 z-10 flex flex-col gap-2">
            <ZoomControl />
            <FullScreenControl />
          </div>

          {showPropertyPanel && (
            <div className="absolute bottom-3 right-3 z-10">
              <PropertiesView />
            </div>
          )}
        </SigmaContainer>

        {isFetching && (
          <div className="bg-background/80 absolute inset-0 z-20 flex items-center justify-center">
            <div className="text-center">
              <div className="border-primary mx-auto mb-2 h-8 w-8 animate-spin rounded-full border-4 border-t-transparent"></div>
              <p>Loading Graph Data...</p>
            </div>
          </div>
        )}

        {!isFetching && graphIsEmpty && (
          <div className="absolute inset-0 z-10 flex items-center justify-center">
            <div className="text-center text-muted-foreground">
              <p className="text-lg mb-2">No graph data</p>
              <p className="text-sm">Insert documents to build the knowledge graph</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default GraphViewer

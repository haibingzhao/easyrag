import { useRegisterEvents, useSetSettings, useSigma } from '@react-sigma/core'
import { AbstractGraph } from 'graphology-types'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import FA2LayoutSupervisor from 'graphology-layout-forceatlas2/worker'
import { useEffect, useRef } from 'react'

import { EdgeType, NodeType } from '@/hooks/useEasyRagGraph'
import useIsDarkMode from '@/hooks/useIsDarkMode'
import * as Constants from '@/lib/constants'

import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'

const isButtonPressed = (ev: MouseEvent | TouchEvent) => {
  if (ev.type.startsWith('mouse')) {
    if ((ev as MouseEvent).buttons !== 0) {
      return true
    }
  }
  return false
}

const GraphControl = ({ disableHoverEffect }: { disableHoverEffect?: boolean }) => {
  const sigma = useSigma<NodeType, EdgeType>()
  const registerEvents = useRegisterEvents<NodeType, EdgeType>()
  const setSettings = useSetSettings<NodeType, EdgeType>()

  const isDarkTheme = useIsDarkMode()
  const hideUnselectedEdges = useSettingsStore.use.enableHideUnselectedEdges()
  const enableEdgeEvents = useSettingsStore.use.enableEdgeEvents()
  const renderEdgeLabels = useSettingsStore.use.showEdgeLabel()
  const renderLabels = useSettingsStore.use.showNodeLabel()
  const minEdgeSize = useSettingsStore.use.minEdgeSize()
  const maxEdgeSize = useSettingsStore.use.maxEdgeSize()
  const selectedNode = useGraphStore.use.selectedNode()
  const focusedNode = useGraphStore.use.focusedNode()
  const selectedEdge = useGraphStore.use.selectedEdge()
  const focusedEdge = useGraphStore.use.focusedEdge()
  const sigmaGraph = useGraphStore.use.sigmaGraph()
  const graphEdgeCount = useGraphStore.use.graphEdgeCount()

  const effectiveEdgeEvents = enableEdgeEvents && graphEdgeCount <= Constants.EDGE_PERF_LIMIT
  const laidOutGraphRef = useRef<unknown>(null)
  const edgeTypeRef = useRef<{ sigma: unknown; curved: boolean } | null>(null)

  // Bind graph to sigma and run initial layout
  useEffect(() => {
    if (!(sigmaGraph && sigma)) return

    try {
      if (typeof sigma.setGraph === 'function') {
        sigma.setGraph(sigmaGraph as unknown as AbstractGraph<NodeType, EdgeType>)
      }
    } catch (error) {
      console.error('Error setting graph on sigma instance:', error)
    }

    if (sigmaGraph.order === 0) return
    if (laidOutGraphRef.current === sigmaGraph) return

    let layout: { start: () => void; stop: () => void; kill: () => void } | null = null
    try {
      layout = new FA2LayoutSupervisor(sigmaGraph as never, {
        settings: forceAtlas2.inferSettings(sigmaGraph.order)
      })
      layout.start()
      useGraphStore.getState().setActiveLayoutSupervisor(layout)
    } catch (error) {
      console.error('Error starting FA2 worker layout:', error)
      return
    }

    const budgetMs = Constants.workerBudgetMs(sigmaGraph.order)
    const timer = window.setTimeout(() => {
      try {
        layout?.stop()
        laidOutGraphRef.current = sigmaGraph
        useGraphStore.getState().releaseLayoutSupervisor(layout)
        sigma.setCustomBBox(null)
        sigma.refresh()
      } catch (error) {
        console.error('Error stopping FA2 worker layout:', error)
      }
    }, budgetMs)

    return () => {
      window.clearTimeout(timer)
      useGraphStore.getState().releaseLayoutSupervisor(layout)
    }
  }, [sigma, sigmaGraph])

  // Set sigma instance in store
  useEffect(() => {
    if (sigma) {
      const currentInstance = useGraphStore.getState().sigmaInstance
      if (currentInstance !== sigma) {
        useGraphStore.getState().setSigmaInstance(sigma)
      }
    }
  }, [sigma])

  // Adapt edge geometry to graph size
  useEffect(() => {
    if (!sigma) return
    const curved = graphEdgeCount > 0 && graphEdgeCount <= Constants.EDGE_PERF_LIMIT
    const prev = edgeTypeRef.current
    if (prev && prev.sigma === sigma && prev.curved === curved) return
    edgeTypeRef.current = { sigma, curved }
    setSettings({ defaultEdgeType: curved ? 'curvedNoArrow' : 'rect' })
    try {
      sigma.refresh()
    } catch {
      /* sigma instance already killed */
    }
  }, [sigma, graphEdgeCount, setSettings])

  // Drop residual edge focus/selection when edge events gated off
  useEffect(() => {
    if (effectiveEdgeEvents) return
    const { selectedEdge, focusedEdge, setSelectedEdge, setFocusedEdge } = useGraphStore.getState()
    if (selectedEdge !== null) setSelectedEdge(null)
    if (focusedEdge !== null) setFocusedEdge(null)
  }, [effectiveEdgeEvents])

  // Register events
  useEffect(() => {
    const { setFocusedNode, setSelectedNode, setFocusedEdge, setSelectedEdge, clearSelection } =
      useGraphStore.getState()

    type NodeEvent = { node: string; event: { original: MouseEvent | TouchEvent } }
    type EdgeEvent = { edge: string; event: { original: MouseEvent | TouchEvent } }

    const events: Record<string, any> = {
      enterNode: (event: NodeEvent) => {
        if (!isButtonPressed(event.event.original)) {
          const graph = sigma.getGraph()
          if (graph.hasNode(event.node)) {
            setFocusedNode(event.node)
          }
        }
      },
      leaveNode: (event: NodeEvent) => {
        if (!isButtonPressed(event.event.original)) {
          setFocusedNode(null)
        }
      },
      clickNode: (event: NodeEvent) => {
        const graph = sigma.getGraph()
        if (graph.hasNode(event.node)) {
          setSelectedNode(event.node)
          setSelectedEdge(null)
        }
      },
      clickStage: () => clearSelection()
    }

    if (effectiveEdgeEvents) {
      events.clickEdge = (event: EdgeEvent) => {
        setSelectedEdge(event.edge)
        setSelectedNode(null)
      }
      events.enterEdge = (event: EdgeEvent) => {
        if (!isButtonPressed(event.event.original)) {
          setFocusedEdge(event.edge)
        }
      }
      events.leaveEdge = (event: EdgeEvent) => {
        if (!isButtonPressed(event.event.original)) {
          setFocusedEdge(null)
        }
      }
    }

    registerEvents(events)
  }, [registerEvents, effectiveEdgeEvents, sigma])

  // Recalculate edge sizes when settings change
  useEffect(() => {
    if (sigma && sigmaGraph) {
      const graph = sigma.getGraph()
      let minWeight = Number.MAX_SAFE_INTEGER
      let maxWeight = 0
      graph.forEachEdge((edge) => {
        const weight = graph.getEdgeAttribute(edge, 'originalWeight') || 1
        if (typeof weight === 'number') {
          minWeight = Math.min(minWeight, weight)
          maxWeight = Math.max(maxWeight, weight)
        }
      })

      const weightRange = maxWeight - minWeight
      const sizeScale = maxEdgeSize - minEdgeSize
      graph.updateEachEdgeAttributes(
        (_edge, attr) => {
          if (weightRange > 0) {
            const weight = typeof attr.originalWeight === 'number' ? attr.originalWeight : 1
            attr.size = minEdgeSize + sizeScale * Math.pow((weight - minWeight) / weightRange, 0.5)
          } else {
            attr.size = minEdgeSize
          }
          return attr
        },
        { attributes: ['size'] }
      )
      sigma.refresh()
    }
  }, [sigma, sigmaGraph, minEdgeSize, maxEdgeSize])

  // Focus/selection reducers
  useEffect(() => {
    const labelColor = isDarkTheme ? Constants.labelColorDarkTheme : undefined
    const edgeColor = isDarkTheme ? Constants.edgeColorDarkTheme : undefined

    const graph = sigma.getGraph()
    const graphOrder = graph ? graph.order : 0
    const effectiveRenderLabels = renderLabels && graphOrder <= Constants.LABEL_RENDER_LIMIT

    const _focusedNode = focusedNode || selectedNode
    const _focusedEdge = effectiveEdgeEvents ? focusedEdge || selectedEdge : null

    if (disableHoverEffect || (!_focusedNode && !_focusedEdge)) {
      setSettings({
        enableEdgeEvents: effectiveEdgeEvents,
        renderEdgeLabels,
        renderLabels: effectiveRenderLabels,
        nodeReducer: null,
        edgeReducer: null
      })
      return
    }

    const neighborSet = new Set<string>()
    let focusedNodeValid = false
    if (_focusedNode && graph.hasNode(_focusedNode)) {
      focusedNodeValid = true
      graph.forEachNeighbor(_focusedNode, (neighbor) => neighborSet.add(neighbor))
    }
    let focusedEdgeSource = ''
    let focusedEdgeTarget = ''
    let focusedEdgeValid = false
    if (!focusedNodeValid && _focusedEdge && graph.hasEdge(_focusedEdge)) {
      focusedEdgeValid = true
      focusedEdgeSource = graph.source(_focusedEdge)
      focusedEdgeTarget = graph.target(_focusedEdge)
    }

    const edgeHighlightColor = isDarkTheme
      ? Constants.edgeColorHighlightedDarkTheme
      : Constants.edgeColorHighlightedLightTheme

    setSettings({
      enableEdgeEvents: effectiveEdgeEvents,
      renderEdgeLabels,
      renderLabels: effectiveRenderLabels,
      nodeReducer: (node, data) => {
        const newData: NodeType & { labelColor?: string; borderColor?: string } = {
          ...data,
          highlighted: false,
          labelColor
        }
        if (focusedNodeValid) {
          if (node === _focusedNode || neighborSet.has(node)) {
            newData.highlighted = true
            if (node === selectedNode) {
              newData.borderColor = Constants.nodeBorderColorSelected
            }
            if (isDarkTheme) {
              newData.labelColor = Constants.LabelColorHighlightedDarkTheme
            }
          } else {
            newData.color = Constants.nodeColorDisabled
          }
        } else if (focusedEdgeValid) {
          if (node === focusedEdgeSource || node === focusedEdgeTarget) {
            newData.highlighted = true
            newData.size = 3
            if (isDarkTheme) {
              newData.labelColor = Constants.LabelColorHighlightedDarkTheme
            }
          } else {
            newData.color = Constants.nodeColorDisabled
          }
        }
        return newData
      },
      edgeReducer: (edge, data) => {
        const newData = { ...data, hidden: false, labelColor, color: edgeColor }
        if (focusedNodeValid) {
          const touchesFocused =
            graph.source(edge) === _focusedNode || graph.target(edge) === _focusedNode
          if (hideUnselectedEdges) {
            if (!touchesFocused) newData.hidden = true
          } else if (touchesFocused) {
            newData.color = edgeHighlightColor
          }
        } else if (focusedEdgeValid) {
          if (edge === selectedEdge) {
            newData.color = Constants.edgeColorSelected
          } else if (edge === _focusedEdge) {
            newData.color = edgeHighlightColor
          } else if (hideUnselectedEdges) {
            newData.hidden = true
          }
        }
        return newData
      }
    })
  }, [
    selectedNode, focusedNode, selectedEdge, focusedEdge,
    setSettings, sigma, sigmaGraph, disableHoverEffect, isDarkTheme,
    hideUnselectedEdges, effectiveEdgeEvents, renderEdgeLabels, renderLabels
  ])

  return null
}

export default GraphControl

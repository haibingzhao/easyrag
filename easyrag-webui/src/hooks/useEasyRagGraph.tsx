import Graph, { UndirectedGraph } from 'graphology'
import { useCallback, useEffect, useRef, useState } from 'react'
import { errorMessage } from '@/lib/utils'
import * as Constants from '@/lib/constants'
import { useGraphStore, RawGraph, RawNodeType, RawEdgeType } from '@/stores/graph'
import { toast } from 'sonner'
import { queryGraphs } from '@/api/easyrag'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'

import { resolveNodeColor, DEFAULT_NODE_COLOR } from '@/utils/graphColor'
import { hasEdgeSafe, addUndirectedEdgeSafe } from '@/utils/graphEdgeSafety'

const NODE_BORDER_COLOR = '#FFFFFF'
const MAX_FETCH_RETRIES = 3
const RETRY_BASE_DELAY_MS = 1000

class GraphBuildError extends Error {
  originalCause: unknown
  constructor(cause: unknown) {
    super('Graph build/apply failed')
    this.name = 'GraphBuildError'
    this.originalCause = cause
  }
}

const hashNodeIdToPosition = (id: string): { x: number; y: number } => {
  let h1 = 0xdeadbeef ^ id.length
  let h2 = 0x41c6ce57 ^ id.length
  for (let i = 0; i < id.length; i++) {
    const ch = id.charCodeAt(i)
    h1 = Math.imul(h1 ^ ch, 2654435761)
    h2 = Math.imul(h2 ^ ch, 1597334677)
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909)
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909)
  return { x: (h1 >>> 0) / 4294967296, y: (h2 >>> 0) / 4294967296 }
}

const yieldToBrowser = (): Promise<void> => {
  const scheduler = (globalThis as { scheduler?: { yield?: () => Promise<void> } }).scheduler
  if (scheduler?.yield) return scheduler.yield()
  return new Promise<void>((resolve) => {
    const { port1, port2 } = new MessageChannel()
    port1.onmessage = () => {
      port1.close()
      resolve()
    }
    port2.postMessage(null)
  })
}

const FRAME_BUDGET_MS = 12
const CHECK_EVERY = 256

const createTypeColorResolver = () => {
  let typeColorMap = useGraphStore.getState().typeColorMap
  const cache = new Map<string, string>()
  let mapUpdated = false

  return {
    colorFor(entityType: string | undefined): string {
      const key = entityType ?? ''
      let color = cache.get(key)
      if (color === undefined) {
        const resolved = resolveNodeColor(entityType, typeColorMap)
        if (resolved.updated) {
          typeColorMap = resolved.map
          mapUpdated = true
        }
        color = resolved.color || DEFAULT_NODE_COLOR
        cache.set(key, color)
      }
      return color
    },
    commit() {
      if (mapUpdated) {
        useGraphStore.setState({ typeColorMap })
        mapUpdated = false
      }
    }
  }
}

const parseEdgeWeight = (properties: Record<string, unknown> | undefined): number => {
  const w = Number(properties?.weight)
  return Number.isFinite(w) ? w : 1
}

const safeNodeLabel = (labels: unknown, fallbackId: string): string =>
  Array.isArray(labels) ? labels.join(', ') : fallbackId

export type NodeType = {
  x: number
  y: number
  label: string
  size: number
  color: string
  highlighted?: boolean
}
export type EdgeType = {
  label: string
  originalWeight?: number
  size?: number
  color?: string
  hidden?: boolean
}

const fetchGraph = async (label: string, maxDepth: number, maxNodes: number) => {
  let rawData: any

  useGraphStore.getState().setLabelsFetchAttempted(true)
  const queryLabel = label || '*'

  try {
    console.log(`Fetching graph label: ${queryLabel}, depth: ${maxDepth}, nodes: ${maxNodes}`)
    rawData = await queryGraphs(queryLabel, maxDepth, maxNodes)
  } catch (e) {
    useBackendState.getState().setErrorMessage(errorMessage(e), 'Query Graphs Error!')
    throw e
  }

  let rawGraph = null

  if (rawData) {
    const nodeIdMap: Record<string, number> = Object.create(null)
    const edgeIdMap: Record<string, number> = Object.create(null)

    for (let i = 0; i < rawData.nodes.length; i++) {
      const node = rawData.nodes[i]
      nodeIdMap[node.id] = i
      node.degree = 0
      node.size = 10
    }

    for (let i = 0; i < rawData.edges.length; i++) {
      const edge = rawData.edges[i]
      // Ensure every edge has a stable id: prefer the backend-supplied one,
      // fall back to a synthetic src↔tgt key to keep `RawEdgeType.id` filled
      // and the `edgeIdMap` index populated regardless of payload shape.
      if (!edge.id) edge.id = `${edge.source}::${edge.target}`
      edgeIdMap[edge.id] = i
      const source = nodeIdMap[edge.source]
      const target = nodeIdMap[edge.target]
      if (source !== undefined && target !== undefined) {
        const sourceNode = rawData.nodes[source]
        const targetNode = rawData.nodes[target]
        if (sourceNode && targetNode) {
          sourceNode.degree += 1
          targetNode.degree += 1
        }
      }
    }

    let minDegree = Number.MAX_SAFE_INTEGER
    let maxDegree = 0
    for (const node of rawData.nodes) {
      minDegree = Math.min(minDegree, node.degree)
      maxDegree = Math.max(maxDegree, node.degree)
    }
    const range = maxDegree - minDegree
    if (range > 0) {
      const scale = Constants.maxNodeSize - Constants.minNodeSize
      for (const node of rawData.nodes) {
        node.size = Math.round(
          Constants.minNodeSize + scale * Math.pow((node.degree - minDegree) / range, 0.5)
        )
      }
    }

    rawGraph = new RawGraph()
    rawGraph.nodes = rawData.nodes
    rawGraph.edges = rawData.edges
    rawGraph.nodeIdMap = nodeIdMap
    rawGraph.edgeIdMap = edgeIdMap
  }

  return { rawGraph, is_truncated: rawData?.is_truncated }
}

const createSigmaGraph = async (rawGraph: RawGraph | null): Promise<UndirectedGraph | null> => {
  if (!rawGraph || !rawGraph.nodes.length) return null

  const graph = new UndirectedGraph()
  const typeColors = createTypeColorResolver()
  let sliceStart = performance.now()

  const nodes = rawGraph.nodes
  for (let i = 0; i < nodes.length; i++) {
    if ((i & (CHECK_EVERY - 1)) === 0 && performance.now() - sliceStart > FRAME_BUDGET_MS) {
      await yieldToBrowser()
      sliceStart = performance.now()
    }

    const rawNode = nodes[i]
    if (graph.hasNode(rawNode.id)) continue

    const { x, y } = hashNodeIdToPosition(rawNode.id)
    rawNode.color = typeColors.colorFor(rawNode.properties?.entity_type as string | undefined)

    graph.addNode(rawNode.id, {
      label: safeNodeLabel(rawNode.labels, rawNode.id),
      color: rawNode.color,
      x,
      y,
      size: rawNode.size,
      borderColor: NODE_BORDER_COLOR
    })
  }

  typeColors.commit()

  rawGraph.edgeDynamicIdMap = Object.create(null) as Record<string, number>
  let skippedEdges = 0

  const edges = rawGraph.edges
  for (let i = 0; i < edges.length; i++) {
    if ((i & (CHECK_EVERY - 1)) === 0 && performance.now() - sliceStart > FRAME_BUDGET_MS) {
      await yieldToBrowser()
      sliceStart = performance.now()
    }

    const rawEdge = edges[i]
    if (
      !graph.hasNode(rawEdge.source) ||
      !graph.hasNode(rawEdge.target) ||
      hasEdgeSafe(graph, rawEdge.source, rawEdge.target)
    ) {
      continue
    }

    const attributes = {
      label: (rawEdge.properties?.keywords as string | undefined) || undefined,
      originalWeight: parseEdgeWeight(rawEdge.properties)
    }
    const dynamicId = addUndirectedEdgeSafe(graph, rawEdge.source, rawEdge.target, attributes)
    if (dynamicId === null) {
      skippedEdges++
      continue
    }
    rawEdge.dynamicId = dynamicId
    rawGraph.edgeDynamicIdMap[rawEdge.dynamicId] = i
  }

  if (skippedEdges > 0) {
    console.warn(`[useEasyRagGraph] ${skippedEdges} edges could not be added to the graph`)
  }

  return graph
}

const useEasyRagGraph = () => {
  const workspace = useSettingsStore.use.workspace()
  const queryLabel = useSettingsStore.use.queryLabel()
  const rawGraph = useGraphStore.use.rawGraph()
  const sigmaGraph = useGraphStore.use.sigmaGraph()
  const maxQueryDepth = useSettingsStore.use.graphQueryMaxDepth()
  const maxNodes = useSettingsStore.use.graphMaxNodes()
  const isFetching = useGraphStore.use.isFetching()
  const nodeToExpand = useGraphStore.use.nodeToExpand()
  const nodeToPrune = useGraphStore.use.nodeToPrune()
  const graphDataVersion = useGraphStore.use.graphDataVersion()

  const dataLoadedRef = useRef(false)
  const initialLoadRef = useRef(false)
  const emptyDataHandledRef = useRef(false)

  const getNode = useCallback(
    (nodeId: string) => {
      return rawGraph?.getNode(nodeId) || null
    },
    [rawGraph]
  )

  const getEdge = useCallback(
    (edgeId: string, dynamicId: boolean = true) => {
      return rawGraph?.getEdge(edgeId, dynamicId) || null
    },
    [rawGraph]
  )

  const fetchInProgressRef = useRef(false)
  const lastFetchSignatureRef = useRef<string | null>(null)
  const retryStateRef = useRef<{ signature: string; attempts: number }>({
    signature: '',
    attempts: 0
  })
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [retryNonce, setRetryNonce] = useState(0)

  // Reset graph when query label is cleared OR workspace changes.
  // `useEffect` scope: when switching tenants (e.g. default → easyai) the
  // old "graph is empty" placeholder + stale fetch signature must be wiped
  // so the label/knowledge endpoints are re-issued against the new scope.
  const lastWorkspaceRef = useRef<string>(workspace)
  useEffect(() => {
    const wsChanged = lastWorkspaceRef.current !== workspace
    if (wsChanged) {
      lastWorkspaceRef.current = workspace
      const state = useGraphStore.getState()
      state.reset()
      state.setGraphDataFetchAttempted(false)
      state.setLabelsFetchAttempted(false)
      dataLoadedRef.current = false
      initialLoadRef.current = false
      emptyDataHandledRef.current = false
      lastFetchSignatureRef.current = null
      retryStateRef.current = { signature: '', attempts: 0 }
      if (retryTimerRef.current !== null) {
        clearTimeout(retryTimerRef.current)
        retryTimerRef.current = null
      }
      return
    }
    if (!queryLabel && (rawGraph !== null || sigmaGraph !== null)) {
      const state = useGraphStore.getState()
      state.reset()
      state.setGraphDataFetchAttempted(false)
      state.setLabelsFetchAttempted(false)
      dataLoadedRef.current = false
      initialLoadRef.current = false
    }
  }, [workspace, queryLabel, rawGraph, sigmaGraph])

  // Graph data fetching logic
  useEffect(() => {
    if (fetchInProgressRef.current) return
    if (!queryLabel && emptyDataHandledRef.current) return

    if (!isFetching && !useGraphStore.getState().graphDataFetchAttempted) {
      const fetchSignature = `${workspace}|${queryLabel}|${maxQueryDepth}|${maxNodes}|v${graphDataVersion}`
      if (lastFetchSignatureRef.current === fetchSignature) {
        useGraphStore.getState().setGraphDataFetchAttempted(true)
        return
      }

      if (retryTimerRef.current !== null) {
        clearTimeout(retryTimerRef.current)
        retryTimerRef.current = null
      }

      fetchInProgressRef.current = true
      useGraphStore.getState().setGraphDataFetchAttempted(true)

      const state = useGraphStore.getState()
      state.setIsFetching(true)
      state.clearSelection()
      if (state.sigmaGraph) {
        state.sigmaGraph.forEachNode((node, attributes) => {
          if (attributes.highlighted) {
            state.sigmaGraph?.setNodeAttribute(node, 'highlighted', false)
          }
        })
      }

      const currentQueryLabel = queryLabel
      const currentMaxQueryDepth = maxQueryDepth
      const currentMaxNodes = maxNodes

      let dataPromise: Promise<{
        rawGraph: RawGraph | null
        is_truncated: boolean | undefined
      } | null>

      if (currentQueryLabel) {
        dataPromise = fetchGraph(currentQueryLabel, currentMaxQueryDepth, currentMaxNodes)
      } else {
        dataPromise = Promise.resolve({ rawGraph: null, is_truncated: false })
      }

      dataPromise
        .then(async (result) => {
          const state = useGraphStore.getState()
          const data = result?.rawGraph

          if (result?.is_truncated) {
            toast.info('Graph data is truncated to Max Nodes')
          }

          state.reset()

          if (!data || !data.nodes || data.nodes.length === 0) {
            const emptyGraph = new UndirectedGraph()
            emptyGraph.addNode('empty-graph-node', {
              label: 'Graph Is Empty',
              color: '#5D6D7E',
              x: 0.5,
              y: 0.5,
              size: 15,
              borderColor: Constants.nodeBorderColor,
              borderSize: 0.2
            })

            state.setSigmaGraph(emptyGraph)
            state.setRawGraph(null)
            state.setGraphCounts(0, 0)
            state.setGraphIsEmpty(true)

            const errMsg = useBackendState.getState().message
            const isAuthError = errMsg && errMsg.includes('Authentication required')

            if (!isAuthError && currentQueryLabel) {
              useSettingsStore.getState().setQueryLabel('')
            }
            if (!isAuthError) {
              state.setLastSuccessfulQueryLabel('')
            }
          } else {
            let newSigmaGraph: UndirectedGraph | null
            try {
              newSigmaGraph = await createSigmaGraph(data)
            } catch (buildError) {
              throw new GraphBuildError(buildError)
            }

            try {
              state.setSigmaGraph(newSigmaGraph)
              state.setRawGraph(data)
              state.setGraphIsEmpty(false)
            } catch (subscriberError) {
              throw new GraphBuildError(subscriberError)
            }

            state.setLastSuccessfulQueryLabel(currentQueryLabel)
            state.setMoveToSelectedNode(true)
          }

          lastFetchSignatureRef.current = fetchSignature
          retryStateRef.current = { signature: '', attempts: 0 }
          dataLoadedRef.current = true
          initialLoadRef.current = true
          fetchInProgressRef.current = false
          state.setIsFetching(false)

          if ((!data || !data.nodes || data.nodes.length === 0) && !currentQueryLabel) {
            emptyDataHandledRef.current = true
          }
        })
        .catch((error) => {
          console.error('[useEasyRagGraph] graph load failed:', error)

          const state = useGraphStore.getState()
          state.setIsFetching(false)
          dataLoadedRef.current = false
          fetchInProgressRef.current = false
          state.setLastSuccessfulQueryLabel('')

          if (error instanceof GraphBuildError) {
            lastFetchSignatureRef.current = fetchSignature
            retryStateRef.current = { signature: '', attempts: 0 }
            toast.error('Failed to render graph data. Use refresh to retry.')
            return
          }

          const retry = retryStateRef.current
          if (retry.signature !== fetchSignature) {
            retry.signature = fetchSignature
            retry.attempts = 0
          }

          if (retry.attempts < MAX_FETCH_RETRIES) {
            retry.attempts += 1
            const delay = RETRY_BASE_DELAY_MS * 2 ** (retry.attempts - 1)
            if (retryTimerRef.current !== null) {
              clearTimeout(retryTimerRef.current)
            }
            retryTimerRef.current = setTimeout(() => {
              retryTimerRef.current = null
              useGraphStore.getState().setGraphDataFetchAttempted(false)
              setRetryNonce((n) => n + 1)
            }, delay)
          } else {
            lastFetchSignatureRef.current = fetchSignature
            toast.error('Failed to load graph data. Use refresh to try again.')
          }
        })
    }
  }, [workspace, queryLabel, maxQueryDepth, maxNodes, isFetching, graphDataVersion, retryNonce])

  // Clean up retry timer on unmount
  useEffect(() => {
    return () => {
      if (retryTimerRef.current !== null) {
        clearTimeout(retryTimerRef.current)
        retryTimerRef.current = null
      }
    }
  }, [])

  // Handle node expansion
  useEffect(() => {
    const nodeId = useGraphStore.getState().nodeToExpand
    if (!nodeId) return

    const handleNodeExpand = async () => {
      const { sigmaGraph, rawGraph } = useGraphStore.getState()
      if (!sigmaGraph || !rawGraph) return

      try {
        const nodeToExpand = rawGraph.getNode(nodeId)
        if (!nodeToExpand) return

        const label = nodeToExpand.labels[0]
        if (!label) return

        const extendedGraph = await queryGraphs(label, 2, 1000)
        if (!extendedGraph || !extendedGraph.nodes || !extendedGraph.edges) return

        const typeColors = createTypeColorResolver()
        const processedNodes: RawNodeType[] = []
        for (const node of extendedGraph.nodes) {
          const { x, y } = hashNodeIdToPosition(node.id)
          const color = typeColors.colorFor(node.properties?.entity_type as string | undefined)
          processedNodes.push({
            id: node.id,
            labels: node.labels,
            properties: node.properties,
            size: 10,
            x,
            y,
            color,
            degree: 0
          })
        }
        typeColors.commit()

        const processedEdges: RawEdgeType[] = []
        for (const edge of extendedGraph.edges) {
          processedEdges.push({
            id: edge.id,
            source: edge.source,
            target: edge.target,
            type: edge.type,
            properties: edge.properties,
            dynamicId: ''
          })
        }

        const processedNodeById = new Map(processedNodes.map((n) => [n.id, n]))
        const processedEdgeById = new Map(processedEdges.map((e) => [e.id, e]))
        const neighborsOfExpandedNode = new Set<string>()
        for (const edge of processedEdges) {
          if (edge.source === nodeId) neighborsOfExpandedNode.add(edge.target)
          else if (edge.target === nodeId) neighborsOfExpandedNode.add(edge.source)
        }

        const nodePositions: Record<string, { x: number; y: number }> = Object.create(null)
        sigmaGraph.forEachNode((node) => {
          nodePositions[node] = {
            x: sigmaGraph.getNodeAttribute(node, 'x'),
            y: sigmaGraph.getNodeAttribute(node, 'y')
          }
        })

        const existingNodeIds = new Set(sigmaGraph.nodes())
        const nodesToAdd = new Set<string>()
        const edgesToAdd = new Set<string>()

        let maxDegree = 0
        let minWeight = Number.MAX_SAFE_INTEGER
        let maxWeight = 0

        sigmaGraph.forEachNode((node) => {
          maxDegree = Math.max(maxDegree, sigmaGraph.degree(node))
        })
        sigmaGraph.forEachEdge((edge) => {
          const weight = sigmaGraph.getEdgeAttribute(edge, 'originalWeight') || 1
          minWeight = Math.min(minWeight, weight)
          maxWeight = Math.max(maxWeight, weight)
        })

        for (const node of processedNodes) {
          if (existingNodeIds.has(node.id)) continue
          if (neighborsOfExpandedNode.has(node.id)) {
            nodesToAdd.add(node.id)
          }
        }

        const nodeDegrees = new Map<string, number>()
        for (const edge of processedEdges) {
          const sourceExists = existingNodeIds.has(edge.source) || nodesToAdd.has(edge.source)
          const targetExists = existingNodeIds.has(edge.target) || nodesToAdd.has(edge.target)
          if (sourceExists && targetExists) {
            edgesToAdd.add(edge.id)
            if (nodesToAdd.has(edge.source)) {
              nodeDegrees.set(edge.source, (nodeDegrees.get(edge.source) || 0) + 1)
            }
            if (nodesToAdd.has(edge.target)) {
              nodeDegrees.set(edge.target, (nodeDegrees.get(edge.target) || 0) + 1)
            }
          }
        }

        if (nodesToAdd.size === 0) {
          toast.info('No new connected nodes found')
          return
        }

        for (const [, degree] of nodeDegrees.entries()) {
          maxDegree = Math.max(maxDegree, degree)
        }

        const minDegree = 1
        const range = maxDegree - minDegree || 1
        const scale = Constants.maxNodeSize - Constants.minNodeSize

        const cameraRatio = useGraphStore.getState().sigmaInstance?.getCamera().ratio || 1
        const spreadFactor =
          Math.max(
            Math.sqrt(nodeToExpand.size) * 4,
            Math.sqrt(nodesToAdd.size) * 3
          ) / cameraRatio
        const randomAngle = Math.random() * 2 * Math.PI

        const nodesToAddList = Array.from(nodesToAdd)
        for (let idx = 0; idx < nodesToAddList.length; idx++) {
          const nid = nodesToAddList[idx]
          const newNode = processedNodeById.get(nid)!
          const nodeDegree = nodeDegrees.get(nid) || 0
          const limitedDegree = Math.min(nodeDegree, maxDegree + 1)
          const nodeSize = Math.round(
            Constants.minNodeSize + scale * Math.pow((limitedDegree - minDegree) / range, 0.5)
          )

          const angle = 2 * Math.PI * (idx / nodesToAddList.length)
          const x =
            nodePositions[nid]?.x ||
            nodePositions[nodeToExpand.id].x + Math.cos(randomAngle + angle) * spreadFactor
          const y =
            nodePositions[nid]?.y ||
            nodePositions[nodeToExpand.id].y + Math.sin(randomAngle + angle) * spreadFactor

          sigmaGraph.addNode(nid, {
            label: safeNodeLabel(newNode.labels, nid),
            color: newNode.color,
            x,
            y,
            size: nodeSize,
            borderColor: NODE_BORDER_COLOR
          })

          if (!rawGraph.getNode(nid)) {
            newNode.size = nodeSize
            newNode.x = x
            newNode.y = y
            newNode.degree = nodeDegree
            rawGraph.nodes.push(newNode)
            rawGraph.nodeIdMap[nid] = rawGraph.nodes.length - 1
          }
        }

        for (const edgeId of edgesToAdd) {
          const newEdge = processedEdgeById.get(edgeId)!
          if (hasEdgeSafe(sigmaGraph, newEdge.source, newEdge.target)) continue

          const weight = parseEdgeWeight(newEdge.properties)
          minWeight = Math.min(minWeight, weight)
          maxWeight = Math.max(maxWeight, weight)

          const dynamicId = addUndirectedEdgeSafe(sigmaGraph, newEdge.source, newEdge.target, {
            label: newEdge.properties?.keywords || undefined,
            size: weight,
            originalWeight: weight
          })
          if (dynamicId === null) continue
          newEdge.dynamicId = dynamicId

          if (!rawGraph.getEdge(newEdge.id, false)) {
            rawGraph.edges.push(newEdge)
            rawGraph.edgeIdMap[newEdge.id] = rawGraph.edges.length - 1
            rawGraph.edgeDynamicIdMap[newEdge.dynamicId] = rawGraph.edges.length - 1
          }
        }

        rawGraph.buildDynamicMap()
        useGraphStore.getState().resetSearchEngine()
        useGraphStore.getState().setGraphCounts(sigmaGraph.order, sigmaGraph.size)

        // Update edge sizes
        const minEdgeSize = useSettingsStore.getState().minEdgeSize
        const maxEdgeSize = useSettingsStore.getState().maxEdgeSize
        const weightRange = maxWeight - minWeight || 1
        const sizeScale = maxEdgeSize - minEdgeSize
        sigmaGraph.forEachEdge((edge) => {
          const weight = sigmaGraph.getEdgeAttribute(edge, 'originalWeight') || 1
          const scaledSize =
            minEdgeSize + sizeScale * Math.pow((weight - minWeight) / weightRange, 0.5)
          sigmaGraph.setEdgeAttribute(edge, 'size', scaledSize)
        })
      } catch (error) {
        console.error('Error expanding node:', error)
      }
    }

    handleNodeExpand()
    window.setTimeout(() => {
      useGraphStore.getState().triggerNodeExpand(null)
    }, 0)
  }, [nodeToExpand, sigmaGraph, rawGraph])

  // Handle node pruning
  useEffect(() => {
    const nodeId = useGraphStore.getState().nodeToPrune
    if (!nodeId) return

    const handleNodePrune = () => {
      const state = useGraphStore.getState()
      const { sigmaGraph, rawGraph } = state
      if (!sigmaGraph || !rawGraph) return

      try {
        if (!sigmaGraph.hasNode(nodeId)) return

        const nodesToDelete = new Set<string>([nodeId])
        sigmaGraph.forEachNeighbor(nodeId, (neighbor) => {
          if (sigmaGraph.degree(neighbor) === 1) {
            nodesToDelete.add(neighbor)
          }
        })

        if (nodesToDelete.size === sigmaGraph.nodes().length) {
          toast.error('Cannot delete all nodes')
          return
        }

        state.clearSelection()

        for (const nodeToDelete of nodesToDelete) {
          sigmaGraph.dropNode(nodeToDelete)
        }

        rawGraph.nodes = rawGraph.nodes.filter((node) => !nodesToDelete.has(node.id))
        rawGraph.edges = rawGraph.edges.filter(
          (edge) => !nodesToDelete.has(edge.source) && !nodesToDelete.has(edge.target)
        )

        rawGraph.nodeIdMap = Object.create(null) as Record<string, number>
        for (let i = 0; i < rawGraph.nodes.length; i++) {
          rawGraph.nodeIdMap[rawGraph.nodes[i].id] = i
        }
        rawGraph.edgeIdMap = Object.create(null) as Record<string, number>
        for (let i = 0; i < rawGraph.edges.length; i++) {
          rawGraph.edgeIdMap[rawGraph.edges[i].id] = i
        }

        rawGraph.buildDynamicMap()
        useGraphStore.getState().resetSearchEngine()
        useGraphStore.getState().setGraphCounts(sigmaGraph.order, sigmaGraph.size)

        if (nodesToDelete.size > 1) {
          toast.info(`${nodesToDelete.size} nodes removed`)
        }
      } catch (error) {
        console.error('Error pruning node:', error)
      }
    }

    handleNodePrune()
    window.setTimeout(() => {
      useGraphStore.getState().triggerNodePrune(null)
    }, 0)
  }, [nodeToPrune, sigmaGraph, rawGraph])

  const easyRagGraph = useCallback(() => {
    if (sigmaGraph) {
      return sigmaGraph as Graph<NodeType, EdgeType>
    }
    const graph = new UndirectedGraph()
    useGraphStore.getState().setSigmaGraph(graph)
    return graph as Graph<NodeType, EdgeType>
  }, [sigmaGraph])

  return { easyRagGraph, getNode, getEdge }
}

export default useEasyRagGraph

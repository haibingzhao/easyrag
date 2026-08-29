import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { DirectedGraph } from 'graphology'

export type RawNodeType = {
  id: string
  labels: string[]
  properties: Record<string, any>
  size: number
  x: number
  y: number
  color: string
  degree: number
}

export type RawEdgeType = {
  id: string
  source: string
  target: string
  type?: string
  properties: Record<string, any>
  dynamicId: string
}

export class RawGraph {
  nodes: RawNodeType[] = []
  edges: RawEdgeType[] = []
  nodeIdMap: Record<string, number> = {}
  edgeIdMap: Record<string, number> = {}
  edgeDynamicIdMap: Record<string, number> = {}

  getNode = (nodeId: string) => {
    const nodeIndex = this.nodeIdMap[nodeId]
    if (nodeIndex !== undefined) {
      return this.nodes[nodeIndex]
    }
    return undefined
  }

  getEdge = (edgeId: string, dynamicId: boolean = true) => {
    const edgeIndex = dynamicId ? this.edgeDynamicIdMap[edgeId] : this.edgeIdMap[edgeId]
    if (edgeIndex !== undefined) {
      return this.edges[edgeIndex]
    }
    return undefined
  }

  buildDynamicMap = () => {
    this.edgeDynamicIdMap = Object.create(null) as Record<string, number>
    for (let i = 0; i < this.edges.length; i++) {
      const edge = this.edges[i]
      this.edgeDynamicIdMap[edge.dynamicId] = i
    }
  }
}

interface GraphState {
  selectedNode: string | null
  focusedNode: string | null
  selectedEdge: string | null
  focusedEdge: string | null

  rawGraph: RawGraph | null
  sigmaGraph: DirectedGraph | null
  sigmaInstance: any | null

  graphNodeCount: number
  graphEdgeCount: number

  moveToSelectedNode: boolean
  isFetching: boolean
  graphIsEmpty: boolean
  lastSuccessfulQueryLabel: string

  typeColorMap: Map<string, string>

  graphDataFetchAttempted: boolean
  labelsFetchAttempted: boolean

  isLayoutComputing: boolean

  setSigmaInstance: (instance: any) => void
  setSelectedNode: (nodeId: string | null, moveToSelectedNode?: boolean) => void
  setFocusedNode: (nodeId: string | null) => void
  setSelectedEdge: (edgeId: string | null) => void
  setFocusedEdge: (edgeId: string | null) => void
  clearSelection: () => void
  reset: () => void

  setMoveToSelectedNode: (moveToSelectedNode: boolean) => void
  setGraphIsEmpty: (isEmpty: boolean) => void
  setLastSuccessfulQueryLabel: (label: string) => void

  setRawGraph: (rawGraph: RawGraph | null) => void
  setSigmaGraph: (sigmaGraph: DirectedGraph | null) => void
  setGraphCounts: (nodeCount: number, edgeCount: number) => void
  setIsFetching: (isFetching: boolean) => void
  setIsLayoutComputing: (running: boolean) => void

  setTypeColorMap: (typeColorMap: Map<string, string>) => void

  setGraphDataFetchAttempted: (attempted: boolean) => void
  setLabelsFetchAttempted: (attempted: boolean) => void

  triggerNodeExpand: (nodeId: string | null) => void
  triggerNodePrune: (nodeId: string | null) => void

  nodeToExpand: string | null
  nodeToPrune: string | null

  graphDataVersion: number
  incrementGraphDataVersion: () => void

  activeLayoutSupervisor: any | null
  setActiveLayoutSupervisor: (supervisor: any | null) => void
  releaseLayoutSupervisor: (supervisor: any | null) => void
  resetSearchEngine: () => void
}

const useGraphStoreBase = create<GraphState>()((set, get) => ({
  selectedNode: null,
  focusedNode: null,
  selectedEdge: null,
  focusedEdge: null,

  moveToSelectedNode: false,
  isFetching: false,
  graphIsEmpty: false,
  lastSuccessfulQueryLabel: '',

  graphDataFetchAttempted: false,
  labelsFetchAttempted: false,

  rawGraph: null,
  sigmaGraph: null,
  sigmaInstance: null,

  graphNodeCount: 0,
  graphEdgeCount: 0,

  typeColorMap: new Map<string, string>(),

  isLayoutComputing: false,

  nodeToExpand: null,
  nodeToPrune: null,

  graphDataVersion: 0,

  activeLayoutSupervisor: null,

  setGraphIsEmpty: (isEmpty: boolean) => set({ graphIsEmpty: isEmpty }),
  setLastSuccessfulQueryLabel: (label: string) => set({ lastSuccessfulQueryLabel: label }),
  setIsFetching: (isFetching: boolean) => set({ isFetching }),
  setIsLayoutComputing: (running: boolean) => set({ isLayoutComputing: running }),

  setSelectedNode: (nodeId: string | null, moveToSelectedNode?: boolean) =>
    set({ selectedNode: nodeId, moveToSelectedNode }),
  setFocusedNode: (nodeId: string | null) => set({ focusedNode: nodeId }),
  setSelectedEdge: (edgeId: string | null) => set({ selectedEdge: edgeId }),
  setFocusedEdge: (edgeId: string | null) => set({ focusedEdge: edgeId }),
  clearSelection: () =>
    set({
      selectedNode: null,
      focusedNode: null,
      selectedEdge: null,
      focusedEdge: null
    }),
  reset: () => {
    set({
      selectedNode: null,
      focusedNode: null,
      selectedEdge: null,
      focusedEdge: null,
      rawGraph: null,
      sigmaGraph: null,
      moveToSelectedNode: false,
      graphIsEmpty: false,
      graphNodeCount: 0,
      graphEdgeCount: 0
    })
  },

  setRawGraph: (rawGraph: RawGraph | null) => set({ rawGraph }),

  setSigmaGraph: (sigmaGraph: DirectedGraph | null) => {
    set({
      sigmaGraph,
      graphNodeCount: sigmaGraph ? sigmaGraph.order : 0,
      graphEdgeCount: sigmaGraph ? sigmaGraph.size : 0
    })
  },

  setGraphCounts: (nodeCount: number, edgeCount: number) =>
    set({ graphNodeCount: nodeCount, graphEdgeCount: edgeCount }),

  setMoveToSelectedNode: (moveToSelectedNode?: boolean) => set({ moveToSelectedNode }),
  setSigmaInstance: (instance: any) => set({ sigmaInstance: instance }),
  setTypeColorMap: (typeColorMap: Map<string, string>) => set({ typeColorMap }),

  setGraphDataFetchAttempted: (attempted: boolean) => set({ graphDataFetchAttempted: attempted }),
  setLabelsFetchAttempted: (attempted: boolean) => set({ labelsFetchAttempted: attempted }),

  triggerNodeExpand: (nodeId: string | null) => set({ nodeToExpand: nodeId }),
  triggerNodePrune: (nodeId: string | null) => set({ nodeToPrune: nodeId }),

  incrementGraphDataVersion: () => set((state) => ({ graphDataVersion: state.graphDataVersion + 1 })),

  setActiveLayoutSupervisor: (supervisor: any | null) => set({ activeLayoutSupervisor: supervisor }),
  releaseLayoutSupervisor: (supervisor: any | null) => {
    const current = get().activeLayoutSupervisor
    if (current === supervisor || supervisor === null) {
      try { current?.kill?.() } catch { /* ignore */ }
      set({ activeLayoutSupervisor: null })
    }
  },
  resetSearchEngine: () => { /* no-op: search is local in GraphSearch component */ }
}))

const useGraphStore = createSelectors(useGraphStoreBase)

export { useGraphStore }

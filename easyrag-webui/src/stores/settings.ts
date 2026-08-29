import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'
import { defaultQueryLabel } from '@/lib/constants'
import { useGraphStore } from '@/stores/graph'

type Theme = 'dark' | 'light' | 'system'
type Tab = 'documents' | 'knowledge-graph' | 'retrieval' | 'api'

export interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface QuerySettings {
  mode: string
  top_k: number
  chunk_top_k: number
  max_entity_tokens: number
  max_relation_tokens: number
  max_total_tokens: number
  only_need_context: boolean
  only_need_prompt: boolean
  stream: boolean
  history_turns: number
  user_prompt: string
  enable_rerank: boolean
}

interface SettingsState {
  workspace: string
  setWorkspace: (workspace: string) => void

  recentWorkspaces: string[]
  addRecentWorkspace: (workspace: string) => void

  showFileName: boolean
  setShowFileName: (show: boolean) => void

  documentsPageSize: number
  setDocumentsPageSize: (size: number) => void

  userPromptHistory: string[]
  addUserPromptToHistory: (prompt: string) => void

  showPropertyPanel: boolean
  showNodeSearchBar: boolean
  showLegend: boolean
  setShowLegend: (show: boolean) => void

  showNodeLabel: boolean
  enableNodeDrag: boolean

  showEdgeLabel: boolean
  enableHideUnselectedEdges: boolean
  enableEdgeEvents: boolean

  minEdgeSize: number
  setMinEdgeSize: (size: number) => void

  maxEdgeSize: number
  setMaxEdgeSize: (size: number) => void

  graphQueryMaxDepth: number
  setGraphQueryMaxDepth: (depth: number) => void

  graphMaxNodes: number
  setGraphMaxNodes: (nodes: number, triggerRefresh?: boolean) => void

  backendMaxGraphNodes: number | null
  setBackendMaxGraphNodes: (maxNodes: number | null) => void

  queryLabel: string
  setQueryLabel: (queryLabel: string) => void

  retrievalHistory: Message[]
  setRetrievalHistory: (history: Message[]) => void

  querySettings: QuerySettings
  updateQuerySettings: (settings: Partial<QuerySettings>) => void

  apiKey: string | null
  setApiKey: (key: string | null) => void

  theme: Theme
  setTheme: (theme: Theme) => void

  enableHealthCheck: boolean
  setEnableHealthCheck: (enable: boolean) => void

  currentTab: Tab
  setCurrentTab: (tab: Tab) => void
}

const useSettingsStoreBase = create<SettingsState>()(
  persist(
    (set, get) => ({
      theme: 'system',
      workspace: 'default',
      recentWorkspaces: ['default'],
      showPropertyPanel: true,
      showNodeSearchBar: true,
      showLegend: false,

      showNodeLabel: true,
      enableNodeDrag: true,

      showEdgeLabel: false,
      enableHideUnselectedEdges: true,
      enableEdgeEvents: false,

      minEdgeSize: 1,
      maxEdgeSize: 1,

      graphQueryMaxDepth: 3,
      graphMaxNodes: 1000,
      backendMaxGraphNodes: null,

      queryLabel: defaultQueryLabel,

      enableHealthCheck: true,

      apiKey: null,

      currentTab: 'documents',
      showFileName: false,
      documentsPageSize: 10,

      retrievalHistory: [],
      userPromptHistory: [],

      querySettings: {
        mode: 'mix',
        top_k: 40,
        chunk_top_k: 20,
        max_entity_tokens: 6000,
        max_relation_tokens: 8000,
        max_total_tokens: 30000,
        only_need_context: false,
        only_need_prompt: false,
        stream: true,
        history_turns: 0,
        user_prompt: '',
        enable_rerank: true
      },

      setWorkspace: (workspace: string) => {
        if (get().workspace === workspace) return
        // Labels are workspace-scoped, so restore the default query label
        // (a stale '' persisted from an empty workspace would otherwise
        // block the graph from auto-loading in the new workspace).
        set({ workspace, queryLabel: defaultQueryLabel })
        const graphState = useGraphStore.getState()
        graphState.reset()
        graphState.setGraphDataFetchAttempted(false)
        graphState.setLabelsFetchAttempted(false)
      },

      addRecentWorkspace: (workspace: string) => {
        if (!workspace.trim()) return
        const current = get().recentWorkspaces
        const filtered = current.filter((w) => w !== workspace)
        const updated = [workspace, ...filtered].slice(0, 10)
        set({ recentWorkspaces: updated })
      },

      setTheme: (theme: Theme) => set({ theme }),
      setQueryLabel: (queryLabel: string) => set({ queryLabel }),
      setGraphQueryMaxDepth: (depth: number) => set({ graphQueryMaxDepth: depth }),

      setGraphMaxNodes: (nodes: number, triggerRefresh: boolean = false) => {
        const state = useSettingsStore.getState()
        if (state.graphMaxNodes === nodes) return

        if (triggerRefresh) {
          const currentLabel = state.queryLabel
          set({ graphMaxNodes: nodes, queryLabel: '' })
          setTimeout(() => {
            set({ queryLabel: currentLabel })
          }, 300)
        } else {
          set({ graphMaxNodes: nodes })
        }
      },

      setBackendMaxGraphNodes: (maxNodes: number | null) => set({ backendMaxGraphNodes: maxNodes }),
      setMinEdgeSize: (size: number) => set({ minEdgeSize: size }),
      setMaxEdgeSize: (size: number) => set({ maxEdgeSize: size }),
      setEnableHealthCheck: (enable: boolean) => set({ enableHealthCheck: enable }),
      setApiKey: (apiKey: string | null) => set({ apiKey }),
      setCurrentTab: (tab: Tab) => set({ currentTab: tab }),
      setRetrievalHistory: (history: Message[]) => set({ retrievalHistory: history }),

      updateQuerySettings: (settings: Partial<QuerySettings>) => {
        const filteredSettings = { ...settings }
        delete filteredSettings.history_turns
        set((state) => ({
          querySettings: { ...state.querySettings, ...filteredSettings, history_turns: 0 }
        }))
      },

      setShowFileName: (show: boolean) => set({ showFileName: show }),
      setShowLegend: (show: boolean) => set({ showLegend: show }),
      setDocumentsPageSize: (size: number) => set({ documentsPageSize: size }),

      addUserPromptToHistory: (prompt: string) => {
        if (!prompt.trim()) return
        set((state) => {
          const newHistory = [...state.userPromptHistory]
          const existingIndex = newHistory.indexOf(prompt)
          if (existingIndex !== -1) {
            newHistory.splice(existingIndex, 1)
          }
          newHistory.unshift(prompt)
          if (newHistory.length > 12) {
            newHistory.splice(12)
          }
          return { userPromptHistory: newHistory }
        })
      }
    }),
    {
      name: 'easyrag-settings',
      storage: createJSONStorage(() => localStorage),
      version: 1
    }
  )
)

const useSettingsStore = createSelectors(useSettingsStoreBase)

export { useSettingsStore, type Theme }

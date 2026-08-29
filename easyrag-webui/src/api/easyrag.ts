import axios from 'axios'

const apiClient = axios.create({
  baseURL: '/api',
  timeout: 300000,
  headers: { 'Content-Type': 'application/json' },
})

// Read current workspace from persisted settings store (avoids circular import)
function getCurrentWorkspace(): string {
  try {
    const raw = localStorage.getItem('easyrag-settings')
    if (raw) {
      const parsed = JSON.parse(raw)
      return parsed?.state?.workspace || 'default'
    }
  } catch {}
  return 'default'
}

// Request interceptor: auth token + workspace routing
apiClient.interceptors.request.use((config) => {
  // Auth token
  const token = localStorage.getItem('easyrag_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }

  // Workspace routing: attach to all API requests
  const workspace = getCurrentWorkspace()
  if (workspace && workspace !== 'default') {
    const method = config.method?.toLowerCase()
    if (method === 'get' || method === 'delete' || method === 'head') {
      // Query parameter for read-only methods
      config.params = { ...config.params, workspace }
    } else {
      // Body field for POST/PUT/PATCH with JSON data
      if (config.data && typeof config.data === 'object' && !(config.data instanceof FormData)) {
        config.data = { ...config.data, workspace }
      } else {
        // Fallback to query param for non-JSON bodies (e.g. multipart)
        config.params = { ...config.params, workspace }
      }
    }
  }

  return config
})

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('easyrag_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export interface QueryRequest {
  query: string
  mode?: string
  top_k?: number
  response_type?: string
  language?: string
  user_prompt?: string
  stream?: boolean
}

export interface QueryResponse {
  response: string
}

export interface DocumentInfo {
  id: string
  filePath: string
  status: string
  contentSummary: string
  contentLength: number
  createdAt: string
  updatedAt: string
  chunksCount?: number
  bizId?: string
}

export interface PaginatedResponse {
  documents: DocumentInfo[]
  total: number
  page: number
  pageSize: number
}

export interface GraphNode {
  id: string
  labels: string[]
  properties: Record<string, string>
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  type?: string
  properties: Record<string, string>
}

export interface KnowledgeGraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
  is_truncated: boolean
}

// Query API
export const queryApi = {
  async query(request: QueryRequest): Promise<QueryResponse> {
    const { data } = await apiClient.post('/query', request)
    return data
  },

  queryStream(_request: QueryRequest): EventSource | null {
    // Use fetch for streaming
    return null // Implemented via fetch in component
  },
}

// Document API
export const documentApi = {
  async insertText(text: string, filePath?: string) {
    const { data } = await apiClient.post('/documents/text', { text, file_path: filePath })
    return data
  },

  async uploadFile(file: File) {
    const formData = new FormData()
    formData.append('file', file)
    const { data } = await apiClient.post('/documents/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  },

  async processDocuments() {
    const { data } = await apiClient.post('/documents/process')
    return data
  },

  async listDocuments(page = 1, pageSize = 50, status?: string): Promise<PaginatedResponse> {
    const params: Record<string, any> = { page, pageSize }
    if (status) params.status = status
    const { data } = await apiClient.get('/documents/list', { params })
    return data
  },

  async deleteDocument(docId: string) {
    const { data } = await apiClient.delete(`/documents/${docId}`)
    return data
  },

  async getPipelineStatus() {
    const { data } = await apiClient.get('/documents/pipeline_status')
    return data
  },

  async resetProcessing() {
    const { data } = await apiClient.post('/documents/reset_processing')
    return data
  },

  async reprocessFailed() {
    const { data } = await apiClient.post('/documents/reprocess_failed')
    return data
  },

  async reprocessDocument(docId: string) {
    const { data } = await apiClient.post(`/documents/${docId}/reprocess`)
    return data
  },
}

// Graph API
export const graphApi = {
  async getLabels(limit = 300): Promise<{ labels: string[]; total: number }> {
    const { data } = await apiClient.get('/graph/labels', { params: { limit } })
    return data
  },

  async searchLabels(query: string, limit = 50): Promise<{ labels: string[]; total: number }> {
    const { data } = await apiClient.get('/graph/search', { params: { query, limit } })
    return data
  },

  async getKnowledgeGraph(nodeLabel = '*', maxDepth = 3, maxNodes = 1000): Promise<KnowledgeGraphResponse> {
    const { data } = await apiClient.get('/graph/knowledge', {
      params: { nodeLabel, max_depth: maxDepth, max_nodes: maxNodes },
    })
    return data
  },

  async getNode(nodeId: string) {
    const { data } = await apiClient.get(`/graph/node/${encodeURIComponent(nodeId)}`)
    return data
  },

  async getStats(): Promise<{ node_count: number; edge_count: number }> {
    const { data } = await apiClient.get('/graph/stats')
    return data
  },
}

// Convenience aliases for components
export const uploadDocument = documentApi.uploadFile
export const getAllGraphLabels = async (): Promise<string[]> => {
  const result = await graphApi.getLabels()
  return result.labels
}
export const queryGraphs = graphApi.getKnowledgeGraph

// Auth API
export interface AuthStatus {
  auth_required: boolean
  guest_token: string | null
  version?: string
  webui_title?: string | null
}

export const authApi = {
  async login(username: string, password: string): Promise<{ token: string }> {
    const { data } = await apiClient.post('/auth/login', { username, password })
    return data
  },

  async authStatus(): Promise<AuthStatus> {
    const { data } = await apiClient.get('/auth-status')
    return data
  },
}

// Legacy alias
export const api = {
  login: authApi.login,
}

export default apiClient

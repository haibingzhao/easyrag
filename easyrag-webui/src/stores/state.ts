import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { healthCheckInterval } from '@/lib/constants'
import apiClient from '@/api/easyrag'

interface BackendState {
  health: boolean
  message: string | null
  messageTitle: string | null
  lastCheckTime: number
  pipelineBusy: boolean
  pipelineActive: boolean
  healthCheckIntervalId: ReturnType<typeof setInterval> | null
  healthCheckFunction: (() => void) | null
  healthCheckIntervalValue: number

  check: () => Promise<boolean>
  clear: () => void
  setErrorMessage: (message: string, messageTitle: string) => void
  setPipelineBusy: (busy: boolean) => void
  setHealthCheckFunction: (fn: () => void) => void
  resetHealthCheckTimer: () => void
  resetHealthCheckTimerDelayed: (delayMs: number) => void
  clearHealthCheckTimer: () => void
}

interface AuthState {
  isAuthenticated: boolean
  isGuestMode: boolean
  coreVersion: string | null
  apiVersion: string | null
  username: string | null
  webuiTitle: string | null
  webuiDescription: string | null

  login: (token: string, isGuest?: boolean, coreVersion?: string | null, apiVersion?: string | null, webuiTitle?: string | null, webuiDescription?: string | null) => void
  logout: () => void
  setVersion: (coreVersion: string | null, apiVersion: string | null) => void
}

const useBackendStateStoreBase = create<BackendState>()((set, get) => ({
  health: true,
  message: null,
  messageTitle: null,
  lastCheckTime: Date.now(),
  pipelineBusy: false,
  pipelineActive: false,
  healthCheckIntervalId: null,
  healthCheckFunction: null,
  healthCheckIntervalValue: healthCheckInterval * 1000,

  check: async () => {
    try {
      const { data } = await apiClient.get('/health')
      if (data.status === 'healthy') {
        set({
          health: true,
          message: null,
          messageTitle: null,
          lastCheckTime: Date.now(),
          pipelineBusy: data.pipeline_busy ?? false,
          pipelineActive: data.pipeline_active ?? data.pipeline_busy ?? false
        })
        return true
      }
      set({
        health: false,
        message: data.message || 'Backend unhealthy',
        messageTitle: 'Backend Health Check Error!',
        lastCheckTime: Date.now()
      })
      return false
    } catch (error: any) {
      set({
        health: false,
        message: error.message || 'Cannot connect to backend',
        messageTitle: 'Connection Error!',
        lastCheckTime: Date.now()
      })
      return false
    }
  },

  clear: () => {
    set({ health: true, message: null, messageTitle: null })
  },

  setErrorMessage: (message: string, messageTitle: string) => {
    set({ health: false, message, messageTitle })
  },

  setPipelineBusy: (busy: boolean) => {
    set({ pipelineBusy: busy })
  },

  setHealthCheckFunction: (fn: () => void) => {
    set({ healthCheckFunction: fn })
  },

  resetHealthCheckTimer: () => {
    const { healthCheckIntervalId, healthCheckFunction, healthCheckIntervalValue } = get()
    if (healthCheckIntervalId) {
      clearInterval(healthCheckIntervalId)
    }
    if (healthCheckFunction) {
      healthCheckFunction()
      const newIntervalId = setInterval(healthCheckFunction, healthCheckIntervalValue)
      set({ healthCheckIntervalId: newIntervalId })
    }
  },

  resetHealthCheckTimerDelayed: (delayMs: number) => {
    setTimeout(() => {
      get().resetHealthCheckTimer()
    }, delayMs)
  },

  clearHealthCheckTimer: () => {
    const { healthCheckIntervalId } = get()
    if (healthCheckIntervalId) {
      clearInterval(healthCheckIntervalId)
      set({ healthCheckIntervalId: null })
    }
  }
}))

const useBackendState = createSelectors(useBackendStateStoreBase)

export { useBackendState }

const parseTokenPayload = (token: string): { sub?: string; role?: string; exp?: number } => {
  try {
    const parts = token.split('.')
    if (parts.length !== 3) return {}
    const payload = JSON.parse(atob(parts[1]))
    return payload
  } catch {
    return {}
  }
}

const getUsernameFromToken = (token: string): string | null => {
  const payload = parseTokenPayload(token)
  return payload.sub || null
}

const initAuthState = () => {
  const token = localStorage.getItem('easyrag_token')
  const coreVersion = localStorage.getItem('easyrag_core_version')
  const apiVersion = localStorage.getItem('easyrag_api_version')
  const username = token ? getUsernameFromToken(token) : null

  if (!token) {
    return {
      isAuthenticated: false,
      isGuestMode: false,
      coreVersion,
      apiVersion,
      username: null,
      webuiTitle: null as string | null,
      webuiDescription: null as string | null
    }
  }

  return {
    isAuthenticated: true,
    isGuestMode: parseTokenPayload(token).role === 'guest',
    coreVersion,
    apiVersion,
    username,
    webuiTitle: null as string | null,
    webuiDescription: null as string | null
  }
}

export const useAuthStore = create<AuthState>((set) => {
  const initialState = initAuthState()

  return {
    ...initialState,

    login: (token, isGuest = false, coreVersion = null, apiVersion = null, webuiTitle = null, webuiDescription = null) => {
      localStorage.setItem('easyrag_token', token)
      if (coreVersion) localStorage.setItem('easyrag_core_version', coreVersion)
      if (apiVersion) localStorage.setItem('easyrag_api_version', apiVersion)

      set({
        isAuthenticated: true,
        isGuestMode: isGuest,
        username: getUsernameFromToken(token),
        coreVersion,
        apiVersion,
        webuiTitle,
        webuiDescription
      })
    },

    logout: () => {
      localStorage.removeItem('easyrag_token')
      set({
        isAuthenticated: false,
        isGuestMode: false,
        username: null
      })
    },

    setVersion: (coreVersion, apiVersion) => {
      if (coreVersion) localStorage.setItem('easyrag_core_version', coreVersion)
      if (apiVersion) localStorage.setItem('easyrag_api_version', apiVersion)
      set({ coreVersion, apiVersion })
    }
  }
})

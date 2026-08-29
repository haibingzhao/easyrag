import { useState, useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import ThemeProvider from '@/components/ThemeProvider'
import { SiteInfo } from '@/lib/constants'
import { useBackendState, useAuthStore } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { authApi } from '@/api/easyrag'
import SiteHeader from '@/features/SiteHeader'
import GraphViewer from '@/features/GraphViewer'
import DocumentManager from '@/features/DocumentManager'
import RetrievalView from '@/features/RetrievalView'
import { Tabs, TabsContent } from '@/components/ui/Tabs'
import { ZapIcon } from 'lucide-react'

function App() {
  const enableHealthCheck = useSettingsStore.use.enableHealthCheck()
  const currentTab = useSettingsStore.use.currentTab()
  const [initializing, setInitializing] = useState(true)
  const healthCheckInitializedRef = useRef(false)
  const isMountedRef = useRef(true)
  const navigate = useNavigate()

  useEffect(() => {
    isMountedRef.current = true
    const handleBeforeUnload = () => {
      isMountedRef.current = false
    }
    window.addEventListener('beforeunload', handleBeforeUnload)
    return () => {
      isMountedRef.current = false
      window.removeEventListener('beforeunload', handleBeforeUnload)
    }
  }, [])

  // Health check
  useEffect(() => {
    const performHealthCheck = async () => {
      try {
        if (isMountedRef.current) {
          await useBackendState.getState().check()
        }
      } catch (error) {
        console.error('Health check error:', error)
      }
    }

    useBackendState.getState().setHealthCheckFunction(performHealthCheck)

    if (!enableHealthCheck) {
      useBackendState.getState().clearHealthCheckTimer()
      return
    }

    if (!healthCheckInitializedRef.current) {
      healthCheckInitializedRef.current = true
    }

    useBackendState.getState().resetHealthCheckTimer()

    return () => {
      useBackendState.getState().clearHealthCheckTimer()
    }
  }, [enableHealthCheck])

  // Initialization
  useEffect(() => {
    const init = async () => {
      try {
        setInitializing(true)
        // Auth bootstrap: consult /api/auth-status before anything else.
        // - auth required + no token -> redirect to /login
        // - auth disabled + guest_token -> auto guest login (keeps the
        //   previous login-free behavior)
        try {
          const status = await authApi.authStatus()
          const authStore = useAuthStore.getState()
          if (status.auth_required) {
            if (!authStore.isAuthenticated) {
              navigate('/login', { replace: true })
              return
            }
          } else if (status.guest_token && !authStore.isAuthenticated) {
            authStore.login(
              status.guest_token,
              true,
              status.version ?? null,
              null,
              status.webui_title ?? null
            )
          }
        } catch (error) {
          console.error('Auth status check error:', error)
        }
        // Quick health check on startup
        await useBackendState.getState().check()
      } catch (error) {
        console.error('Initialization error:', error)
      } finally {
        if (isMountedRef.current) {
          setInitializing(false)
        }
      }
    }
    init()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleTabChange = useCallback(
    (tab: string) => useSettingsStore.getState().setCurrentTab(tab as any),
    []
  )

  return (
    <ThemeProvider>
      {initializing ? (
        <div className="flex h-screen w-screen flex-col">
          <header className="border-border/40 bg-background/95 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
            <div className="min-w-[200px] w-auto flex items-center">
              <a href="/" className="flex items-center gap-2">
                <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
                <span className="font-bold md:inline-block">{SiteInfo.name}</span>
              </a>
            </div>
            <div className="flex h-10 flex-1 items-center justify-center" />
            <nav className="w-[200px] flex items-center justify-end" />
          </header>
          <div className="flex flex-1 items-center justify-center">
            <div className="text-center">
              <div className="mb-2 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto"></div>
              <p>Initializing...</p>
            </div>
          </div>
        </div>
      ) : (
        <main className="flex h-screen w-screen overflow-hidden">
          <Tabs
            defaultValue={currentTab}
            className="!m-0 flex grow flex-col !p-0 overflow-hidden"
            onValueChange={handleTabChange}
          >
            <SiteHeader />
            <div className="relative grow">
              <TabsContent value="documents" className="absolute top-0 right-0 bottom-0 left-0 overflow-auto">
                <DocumentManager />
              </TabsContent>
              <TabsContent value="knowledge-graph" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                <GraphViewer />
              </TabsContent>
              <TabsContent value="retrieval" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                <RetrievalView />
              </TabsContent>
            </div>
          </Tabs>
        </main>
      )}
    </ThemeProvider>
  )
}

export default App

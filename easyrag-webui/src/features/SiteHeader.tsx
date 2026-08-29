import Button from '@/components/ui/Button'
import { SiteInfo } from '@/lib/constants'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore } from '@/stores/state'
import { cn } from '@/lib/utils'
import { ZapIcon, LogOutIcon } from 'lucide-react'
import WorkspaceSelector from '@/components/WorkspaceSelector'

interface NavigationTabProps {
  value: string
  currentTab: string
  children: React.ReactNode
}

function NavigationTab({ value, currentTab, children }: NavigationTabProps) {
  return (
    <TabsTrigger
      value={value}
      className={cn(
        'cursor-pointer px-2 py-1 transition-all',
        currentTab === value ? '!bg-emerald-400 !text-zinc-50' : 'hover:bg-background/60'
      )}
    >
      {children}
    </TabsTrigger>
  )
}

function TabsNavigation() {
  const currentTab = useSettingsStore.use.currentTab()

  return (
    <div className="flex h-8 self-center">
      <TabsList className="h-full gap-2">
        <NavigationTab value="documents" currentTab={currentTab}>
          Documents
        </NavigationTab>
        <NavigationTab value="knowledge-graph" currentTab={currentTab}>
          Knowledge Graph
        </NavigationTab>
        <NavigationTab value="retrieval" currentTab={currentTab}>
          Retrieval
        </NavigationTab>
      </TabsList>
    </div>
  )
}

export default function SiteHeader() {
  const { isGuestMode, coreVersion, apiVersion, username } = useAuthStore()

  const versionDisplay = coreVersion && apiVersion ? `${coreVersion}/${apiVersion}` : null

  const handleLogout = () => {
    useAuthStore.getState().logout()
    window.location.href = '/login'
  }

  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <div className="min-w-[200px] w-auto flex items-center">
        <a href="/" className="flex items-center gap-2">
          <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
          <span className="font-bold md:inline-block">{SiteInfo.name}</span>
        </a>
      </div>

      <div className="flex h-10 flex-1 items-center justify-center">
        <TabsNavigation />
        {isGuestMode && (
          <div className="ml-2 self-center px-2 py-1 text-xs bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 rounded-md">
            Guest Mode
          </div>
        )}
      </div>

      <nav className="w-[200px] flex items-center justify-end">
        <div className="flex items-center gap-2">
          <WorkspaceSelector />
          {versionDisplay && (
            <span className="text-xs text-gray-500 dark:text-gray-400 mr-1 cursor-default">
              v{versionDisplay}
            </span>
          )}
          {!isGuestMode && (
            <Button
              variant="ghost"
              size="icon"
              tooltip={`Logout (${username})`}
              onClick={handleLogout}
            >
              <LogOutIcon className="size-4" aria-hidden="true" />
            </Button>
          )}
        </div>
      </nav>
    </header>
  )
}

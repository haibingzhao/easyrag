import { useState, useRef, useEffect, useCallback } from 'react'
import { useSettingsStore } from '@/stores/settings'
import { cn } from '@/lib/utils'
import { FolderIcon, CheckIcon, PlusIcon, XIcon } from 'lucide-react'
import { Input } from '@/components/ui/Input'

const WORKSPACE_PATTERN = /^[a-zA-Z0-9_-][a-zA-Z0-9_.-]{0,63}$/

export default function WorkspaceSelector() {
  const workspace = useSettingsStore.use.workspace()
  const recentWorkspaces = useSettingsStore.use.recentWorkspaces()
  const setWorkspace = useSettingsStore.use.setWorkspace()
  const addRecentWorkspace = useSettingsStore.use.addRecentWorkspace()

  const [open, setOpen] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState('')
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [open])

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setInputValue('')
      setError('')
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  const selectWorkspace = useCallback(
    (ws: string) => {
      const trimmed = ws.trim()
      if (!trimmed) return
      if (!WORKSPACE_PATTERN.test(trimmed)) {
        setError('Invalid: alphanumeric, _ - . only, 1-64 chars')
        return
      }
      setWorkspace(trimmed)
      addRecentWorkspace(trimmed)
      setOpen(false)
      setInputValue('')
      setError('')
    },
    [setWorkspace, addRecentWorkspace]
  )

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      selectWorkspace(inputValue)
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  const filteredRecents = recentWorkspaces.filter(
    (w) => w !== workspace && w.toLowerCase().includes(inputValue.toLowerCase())
  )

  return (
    <div className="relative" ref={containerRef}>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          'flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs transition-colors',
          'hover:bg-accent hover:text-accent-foreground',
          workspace !== 'default'
            ? 'border-emerald-400/50 text-emerald-600 dark:text-emerald-400'
            : 'border-border text-muted-foreground'
        )}
        title={`Workspace: ${workspace}`}
      >
        <FolderIcon className="size-3" />
        <span className="max-w-[100px] truncate">{workspace}</span>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 w-64 rounded-md border bg-popover p-2 shadow-md">
          <div className="mb-2">
            <div className="text-xs font-medium text-muted-foreground mb-1">Switch Workspace</div>
            <div className="relative">
              <Input
                ref={inputRef}
                value={inputValue}
                onChange={(e) => {
                  setInputValue(e.target.value)
                  setError('')
                }}
                onKeyDown={handleKeyDown}
                placeholder="Type workspace name..."
                className="h-7 text-xs pr-7"
              />
              {inputValue && (
                <button
                  onClick={() => {
                    setInputValue('')
                    setError('')
                    inputRef.current?.focus()
                  }}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  <XIcon className="size-3" />
                </button>
              )}
            </div>
            {error && <p className="text-xs text-red-500 mt-1">{error}</p>}
          </div>

          {/* Current workspace indicator */}
          <div className="mb-1 flex items-center gap-1.5 rounded-sm px-2 py-1.5 bg-accent text-xs">
            <CheckIcon className="size-3 text-emerald-500" />
            <span className="truncate font-medium">{workspace}</span>
            <span className="ml-auto text-muted-foreground text-[10px]">current</span>
          </div>

          {/* Recent workspaces */}
          {filteredRecents.length > 0 && (
            <div className="mb-1">
              <div className="text-[10px] text-muted-foreground px-2 py-0.5">Recent</div>
              {filteredRecents.map((ws) => (
                <button
                  key={ws}
                  onClick={() => selectWorkspace(ws)}
                  className="flex w-full items-center gap-1.5 rounded-sm px-2 py-1.5 text-xs hover:bg-accent transition-colors text-left"
                >
                  <FolderIcon className="size-3 text-muted-foreground" />
                  <span className="truncate">{ws}</span>
                </button>
              ))}
            </div>
          )}

          {/* Create new workspace */}
          {inputValue.trim() &&
            !recentWorkspaces.includes(inputValue.trim()) &&
            WORKSPACE_PATTERN.test(inputValue.trim()) && (
              <div className="border-t pt-1 mt-1">
                <button
                  onClick={() => selectWorkspace(inputValue)}
                  className="flex w-full items-center gap-1.5 rounded-sm px-2 py-1.5 text-xs hover:bg-accent transition-colors text-left text-emerald-600 dark:text-emerald-400"
                >
                  <PlusIcon className="size-3" />
                  <span>Create & switch to "{inputValue.trim()}"</span>
                </button>
              </div>
            )}
        </div>
      )}
    </div>
  )
}

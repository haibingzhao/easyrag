import { useState, useRef, useCallback, useEffect } from 'react'
import Textarea from '@/components/ui/Textarea'
import Button from '@/components/ui/Button'
import ChatMessageBubble from '@/components/retrieval/ChatMessage'
import { queryApi, authHeaders, handleUnauthorized, withWorkspace, type QueryRequest } from '@/api/easyrag'
import { useSettingsStore } from '@/stores/settings'
import { errorMessage } from '@/lib/utils'
import { SendIcon, EraserIcon, SquareIcon } from 'lucide-react'

interface ViewMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  displayContent?: string
  thinkingContent?: string
  thinkingTime?: number | null
  isThinking?: boolean
  isError?: boolean
}

const generateId = () => `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`

const THINK_OPEN_RE = /<(think|thinking)>/i

/** Split raw streamed text into reasoning-model thinking and body content. */
function splitThinking(content: string): { thinking: string; body: string; thinkingActive: boolean } {
  const openMatch = content.match(THINK_OPEN_RE)
  if (!openMatch || openMatch.index === undefined) {
    return { thinking: '', body: content, thinkingActive: false }
  }
  const afterOpen = openMatch.index + openMatch[0].length
  const closeMatch = content.slice(afterOpen).match(new RegExp(`</${openMatch[1]}>`, 'i'))
  if (closeMatch && closeMatch.index !== undefined) {
    return {
      thinking: content.slice(afterOpen, afterOpen + closeMatch.index),
      body: content.slice(afterOpen + closeMatch.index + closeMatch[0].length),
      thinkingActive: false
    }
  }
  return { thinking: content.slice(afterOpen), body: '', thinkingActive: true }
}

export default function RetrievalView() {
  const [messages, setMessages] = useState<ViewMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const thinkingStartRef = useRef<number | null>(null)
  const thinkingTimeRef = useRef<number | null>(null)

  const querySettings = useSettingsStore.use.querySettings()

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const handleQuery = async () => {
    const query = inputValue.trim()
    if (!query || isLoading) return

    const userMessage: ViewMessage = { id: generateId(), role: 'user', content: query }
    setMessages((prev) => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    const assistantId = generateId()
    setMessages((prev) => [...prev, { id: assistantId, role: 'assistant', content: '' }])

    const abortController = new AbortController()
    abortControllerRef.current = abortController

    thinkingStartRef.current = null
    thinkingTimeRef.current = null

    const elapsedSeconds = () =>
      thinkingStartRef.current === null
        ? null
        : Math.round((Date.now() - thinkingStartRef.current) / 100) / 10

    // Parse raw text into thinking + body and update the assistant message.
    const applyContent = (raw: string) => {
      const { thinking, body, thinkingActive } = splitThinking(raw)
      if (thinking && thinkingStartRef.current === null) {
        thinkingStartRef.current = Date.now()
      }
      if (!thinkingActive && thinkingStartRef.current !== null && thinkingTimeRef.current === null) {
        thinkingTimeRef.current = elapsedSeconds()
      }
      const hasThinking = thinkingStartRef.current !== null
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content: raw,
                displayContent: body,
                thinkingContent: hasThinking ? thinking : undefined,
                isThinking: hasThinking ? thinkingActive : undefined,
                thinkingTime: thinkingTimeRef.current
              }
            : msg
        )
      )
    }

    try {
      const request: QueryRequest = {
        query,
        mode: querySettings.mode,
        top_k: querySettings.top_k,
        stream: querySettings.stream
      }

      if (querySettings.stream) {
        // Streaming response: axios cannot hand back a ReadableStream in the
        // browser, so this stays on native fetch — but through the shared
        // helpers, so auth/workspace injection and 401 recovery match every
        // apiClient call.
        const streamBody = withWorkspace<Record<string, unknown>>({ ...request })
        const response = await fetch('/api/query/stream', {
          method: 'POST',
          headers: authHeaders(),
          body: JSON.stringify(streamBody),
          signal: abortController.signal
        })

        if (response.status === 401) {
          // BC-14 (A-12): an expired session now clears the token and returns
          // to the login page instead of surfacing a bare "HTTP 401" bubble.
          handleUnauthorized()
          return
        }

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const reader = response.body?.getReader()
        const decoder = new TextDecoder()

        if (reader) {
          let raw = ''
          let sseBuffer = ''
          const handleSseLine = (line: string) => {
            if (!line.startsWith('data: ')) return
            const data = line.slice(6)
            if (data === '[DONE]') return
            let text = data
            try {
              const parsed = JSON.parse(data)
              text = parsed.response || parsed.content || data
            } catch {
              // Plain text SSE data
            }
            raw += text
          }

          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            // Buffer across reads so SSE lines split between chunks stay intact
            sseBuffer += decoder.decode(value, { stream: true })
            const lines = sseBuffer.split('\n')
            sseBuffer = lines.pop() ?? ''
            for (const line of lines) handleSseLine(line)
            applyContent(raw)
          }
          if (sseBuffer) handleSseLine(sseBuffer)
          applyContent(raw)
        }
      } else {
        // Non-streaming response (reasoning models may still embed thinking tags)
        const result = await queryApi.query(request)
        applyContent(result.response || '')
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content: msg.content || '(Cancelled)',
                  isThinking: false,
                  thinkingTime: thinkingTimeRef.current ?? elapsedSeconds()
                }
              : msg
          )
        )
      } else {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content: `Error: ${errorMessage(err)}`,
                  displayContent: undefined,
                  isThinking: false,
                  isError: true
                }
              : msg
          )
        )
      }
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  const handleStop = () => {
    abortControllerRef.current?.abort()
  }

  const handleClear = () => {
    setMessages([])
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleQuery()
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Settings bar */}
      <div className="flex items-center gap-4 border-b px-4 py-2">
        <div className="flex items-center gap-2">
          <label className="text-sm text-muted-foreground">Mode:</label>
          <select
            className="rounded-md border bg-background px-2 py-1 text-sm"
            value={querySettings.mode}
            onChange={(e) => useSettingsStore.getState().updateQuerySettings({ mode: e.target.value })}
          >
            <option value="local">Local</option>
            <option value="global">Global</option>
            <option value="hybrid">Hybrid</option>
            <option value="naive">Naive</option>
            <option value="mix">Mix</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-muted-foreground">Top K:</label>
          <input
            type="number"
            className="w-16 rounded-md border bg-background px-2 py-1 text-sm"
            value={querySettings.top_k}
            min={1}
            max={100}
            onChange={(e) => useSettingsStore.getState().updateQuerySettings({ top_k: parseInt(e.target.value) || 40 })}
          />
        </div>
        <div className="flex-1" />
        <Button variant="ghost" size="sm" onClick={handleClear} disabled={messages.length === 0}>
          <EraserIcon className="size-4 mr-1" />
          Clear
        </Button>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <div className="text-center">
              <p className="text-lg mb-1">Ask a question</p>
              <p className="text-sm">Query your knowledge base using RAG</p>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <ChatMessageBubble message={msg} />
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t p-4">
        <div className="flex items-end gap-2">
          <Textarea
            className="min-h-[44px] max-h-[200px] flex-1 resize-none"
            placeholder="Ask a question about your documents..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
          />
          {isLoading ? (
            <Button variant="destructive" size="icon" className="size-11" onClick={handleStop}>
              <SquareIcon className="size-4" />
            </Button>
          ) : (
            <Button size="icon" className="size-11" onClick={handleQuery} disabled={!inputValue.trim()}>
              <SendIcon className="size-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

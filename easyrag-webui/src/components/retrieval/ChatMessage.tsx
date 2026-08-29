import { ReactNode, useMemo, useState, useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeRaw from 'rehype-raw'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism'
import useTheme from '@/hooks/useTheme'
import { LoaderIcon, ClockIcon, ZapIcon, ChevronRightIcon } from 'lucide-react'

export interface Message {
  role: 'user' | 'assistant'
  content: string
  thinkingContent?: string
  displayContent?: string
  thinkingTime?: number | null
  responseTime?: number
  firstTokenTime?: number
}

export type MessageWithError = Message & {
  id: string
  isError?: boolean
  isThinking?: boolean
  isAborted?: boolean
  mermaidRendered?: boolean
  latexRendered?: boolean
}

export const ChatMessage = ({
  message,
  isTabActive = true,
  activeProgress = null,
  isQuerying = false
}: {
  message: MessageWithError
  isTabActive?: boolean
  activeProgress?: string | null
  isQuerying?: boolean
}) => {
  const { theme } = useTheme()
  const { thinkingContent, displayContent, thinkingTime, responseTime, firstTokenTime, isThinking } = message

  const hasThinking = !!thinkingContent && thinkingContent.trim() !== ''

  // Expanded while thinking is streaming; auto-collapse once it completes.
  const [thinkingExpanded, setThinkingExpanded] = useState(!!isThinking)
  const wasThinkingRef = useRef(!!isThinking)
  useEffect(() => {
    if (isThinking) {
      setThinkingExpanded(true)
    } else if (wasThinkingRef.current) {
      setThinkingExpanded(false)
    }
    wasThinkingRef.current = !!isThinking
  }, [isThinking])

  const finalDisplayContent = message.role === 'user'
    ? message.content
    : (displayContent !== undefined ? displayContent : (message.content || ''))

  const hasContent = !!finalDisplayContent && finalDisplayContent.trim() !== ''

  const timingRow =
    message.role === 'assistant' && typeof responseTime === 'number' ? (
      <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1 tabular-nums">
          <ClockIcon className={cn('size-3', isQuerying && 'animate-spin')} />
          {responseTime.toFixed(1)}s
        </span>
        {typeof firstTokenTime === 'number' && (
          <span className="flex items-center gap-1 tabular-nums">
            <ZapIcon className="size-3" />
            {firstTokenTime.toFixed(1)}s
          </span>
        )}
      </div>
    ) : null

  const markdownComponents = useMemo(() => ({
    code: (props: any) => {
      const { inline, className, children, ...restProps } = props
      const match = /language-(\w+)/.exec(className || '')
      const language = match ? match[1] : undefined
      const contentStr = String(children || '').replace(/\n$/, '')

      if (!inline && language) {
        return (
          <SyntaxHighlighter
            style={theme === 'dark' ? oneDark : oneLight}
            PreTag="div"
            language={language}
            {...restProps}
          >
            {contentStr}
          </SyntaxHighlighter>
        )
      }
      return (
        <code className="mx-1 rounded-sm bg-muted px-1 py-0.5 font-mono text-sm" {...restProps}>
          {children}
        </code>
      )
    },
    p: ({ children }: { children?: ReactNode }) => <div className="my-2">{children}</div>,
    h1: ({ children }: { children?: ReactNode }) => <h1 className="text-xl font-bold mt-4 mb-2">{children}</h1>,
    h2: ({ children }: { children?: ReactNode }) => <h2 className="text-lg font-bold mt-4 mb-2">{children}</h2>,
    h3: ({ children }: { children?: ReactNode }) => <h3 className="text-base font-bold mt-3 mb-2">{children}</h3>,
    ul: ({ children }: { children?: ReactNode }) => <ul className="list-disc pl-5 my-2">{children}</ul>,
    ol: ({ children }: { children?: ReactNode }) => <ol className="list-decimal pl-5 my-2">{children}</ol>,
    li: ({ children }: { children?: ReactNode }) => <li className="my-1">{children}</li>
  }), [theme])

  return (
    <div
      className={`${
        message.role === 'user'
          ? 'max-w-[80%] bg-primary text-primary-foreground'
          : message.isError
            ? 'w-[95%] bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400'
            : 'w-[95%] bg-muted'
      } rounded-lg px-4 py-2`}
    >
      {!hasContent && timingRow}

      {/* Thinking block: streamed live (expanded), collapsed after completion */}
      {message.role === 'assistant' && (isThinking || hasThinking || typeof thinkingTime === 'number') && (
        <div className={cn('mb-2', !isTabActive && 'opacity-50')}>
          <button
            type="button"
            className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400 text-sm"
            onClick={() => setThinkingExpanded((v) => !v)}
            disabled={isThinking}
          >
            {isThinking ? (
              <>
                <LoaderIcon className="size-3.5 animate-spin" />
                <span>Thinking...</span>
              </>
            ) : (
              <>
                <ChevronRightIcon
                  className={cn('size-3.5 transition-transform', thinkingExpanded && 'rotate-90')}
                />
                <span>{typeof thinkingTime === 'number' ? `Thought for ${thinkingTime}s` : 'Thought'}</span>
              </>
            )}
          </button>
          {thinkingExpanded && hasThinking && (
            <div className="mt-2 pl-4 border-l-2 border-primary/20 text-sm text-muted-foreground break-words">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeRaw]}
                skipHtml={false}
                components={markdownComponents}
              >
                {thinkingContent!}
              </ReactMarkdown>
            </div>
          )}
        </div>
      )}

      {/* Main content */}
      {finalDisplayContent && (
        <div className={cn(
          'prose dark:prose-invert max-w-none text-sm break-words',
          message.role === 'user' ? 'text-primary-foreground' : 'text-foreground'
        )}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeRaw]}
            skipHtml={false}
            components={markdownComponents}
          >
            {finalDisplayContent}
          </ReactMarkdown>
        </div>
      )}

      {hasContent && timingRow}

      {message.isAborted && (
        <div className="mt-1 text-xs italic text-muted-foreground">
          Response was terminated by user
        </div>
      )}

      {isTabActive && !message.isAborted && !hasContent && !isThinking && !timingRow && (
        <LoaderIcon className="animate-spin" />
      )}
    </div>
  )
}

export default ChatMessage

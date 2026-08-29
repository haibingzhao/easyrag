import { ReactNode, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeRaw from 'rehype-raw'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism'
import useTheme from '@/hooks/useTheme'
import { cn } from '@/lib/utils'

// Shared react-markdown element mapping (theme-aware code highlighting +
// typography). Project Tailwind has no typography plugin, so styles come
// from these overrides. Single source used by chat messages and document
// preview.
function useMarkdownComponents() {
  const { theme } = useTheme()
  return useMemo(
    () => ({
      code: (props: any) => {
        const { className, children, ...restProps } = props
        const match = /language-(\w+)/.exec(className || '')
        const contentStr = String(children || '').replace(/\n$/, '')

        // react-markdown v10 dropped the `inline` prop; treat fenced blocks
        // (with a language or newlines) as code blocks.
        if (match || contentStr.includes('\n')) {
          return (
            <SyntaxHighlighter
              style={theme === 'dark' ? oneDark : oneLight}
              PreTag="div"
              language={match ? match[1] : 'text'}
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
      h4: ({ children }: { children?: ReactNode }) => <h4 className="text-sm font-bold mt-3 mb-1">{children}</h4>,
      ul: ({ children }: { children?: ReactNode }) => <ul className="list-disc pl-5 my-2">{children}</ul>,
      ol: ({ children }: { children?: ReactNode }) => <ol className="list-decimal pl-5 my-2">{children}</ol>,
      li: ({ children }: { children?: ReactNode }) => <li className="my-1">{children}</li>,
      blockquote: ({ children }: { children?: ReactNode }) => (
        <blockquote className="my-2 border-l-2 border-border pl-3 text-muted-foreground">{children}</blockquote>
      ),
      a: ({ children, ...restProps }: any) => (
        <a className="text-primary underline underline-offset-2" target="_blank" rel="noreferrer" {...restProps}>
          {children}
        </a>
      ),
      img: (restProps: any) => <img className="my-2 max-w-full rounded-md" alt="" {...restProps} />,
      hr: () => <hr className="my-4 border-border" />,
      table: ({ children }: { children?: ReactNode }) => (
        <div className="my-2 overflow-x-auto">
          <table className="w-full border-collapse text-sm">{children}</table>
        </div>
      ),
      th: ({ children }: { children?: ReactNode }) => (
        <th className="border border-border bg-muted/50 px-2 py-1 text-left font-medium">{children}</th>
      ),
      td: ({ children }: { children?: ReactNode }) => (
        <td className="border border-border px-2 py-1 align-top">{children}</td>
      ),
    }),
    [theme]
  )
}

const Markdown = ({
  content,
  className,
}: {
  content: string
  className?: string
}) => {
  const components = useMarkdownComponents()
  return (
    <div className={cn('max-w-none break-words text-sm', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw]}
        skipHtml={false}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

export default Markdown

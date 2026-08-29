import { useEffect, useMemo, useState } from 'react'
import { FileTextIcon, LoaderIcon } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/Dialog'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/Tabs'
import Markdown from '@/components/markdown/Markdown'
import { documentApi, type DocumentFull, type DocumentInfo } from '@/api/easyrag'
import { errorMessage } from '@/lib/utils'

const isMarkdownFile = (filePath: string) => /\.(md|markdown)$/i.test(filePath)

// Read-only preview of a stored document: markdown files render through the
// shared Markdown component (with a raw-source tab), everything else shows
// the extracted plain text. Content comes from GET /api/documents/{docId}.
const DocumentContentViewer = ({
  doc,
  open,
  onOpenChange,
}: {
  doc: DocumentInfo | null
  open: boolean
  onOpenChange: (open: boolean) => void
}) => {
  const [full, setFull] = useState<DocumentFull | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<'preview' | 'source'>('preview')

  const isMarkdown = isMarkdownFile(doc?.filePath ?? '')

  useEffect(() => {
    if (!open || !doc) return
    let cancelled = false
    setIsLoading(true)
    setError(null)
    setFull(null)
    setView('preview')
    documentApi
      .getDocument(doc.id, doc.bizId)
      .then((data) => {
        if (!cancelled) setFull(data)
      })
      .catch((err) => {
        if (!cancelled) setError(errorMessage(err))
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, doc])

  const content = typeof full?.content === 'string' ? full.content : ''

  const fileName = useMemo(() => {
    const path = doc?.filePath ?? ''
    return path.split('/').pop() || path || 'Untitled'
  }, [doc?.filePath])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[85vh] max-w-4xl flex-col gap-0 overflow-hidden p-0">
        <DialogHeader className="space-y-1 border-b px-6 py-4">
          <div className="flex items-center gap-2">
            <FileTextIcon className="size-4 shrink-0 text-muted-foreground" />
            <DialogTitle className="truncate text-base" title={doc?.filePath}>
              {fileName}
            </DialogTitle>
            {doc?.status && (
              <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                {full?.status ?? doc.status}
              </span>
            )}
          </div>
          <DialogDescription className="flex flex-wrap items-center gap-x-3 text-xs">
            <span className="truncate" title={doc?.filePath}>
              {doc?.filePath}
            </span>
            {full?.chunks_count != null && <span>{full.chunks_count} chunks</span>}
            {doc?.createdAt && <span>Created {new Date(doc.createdAt).toLocaleString()}</span>}
          </DialogDescription>
        </DialogHeader>

        {isMarkdown && !isLoading && !error && content && (
          <div className="px-6 pt-3">
            <Tabs value={view} onValueChange={(v) => setView(v as 'preview' | 'source')}>
              <TabsList>
                <TabsTrigger value="preview">Preview</TabsTrigger>
                <TabsTrigger value="source">Source</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-auto px-6 py-4">
          {isLoading ? (
            <div className="flex h-40 items-center justify-center gap-2 text-sm text-muted-foreground">
              <LoaderIcon className="size-4 animate-spin" />
              Loading document...
            </div>
          ) : error ? (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
              {error}
            </div>
          ) : !content ? (
            <div className="py-8 text-center text-sm text-muted-foreground">
              No content stored for this document.
            </div>
          ) : isMarkdown && view === 'preview' ? (
            <Markdown content={content} />
          ) : (
            <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">
              {content}
            </pre>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

export default DocumentContentViewer

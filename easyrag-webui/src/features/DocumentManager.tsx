import { useState, useEffect, useCallback } from 'react'
import Button from '@/components/ui/Button'
import { documentApi, type DocumentInfo, type PaginatedResponse } from '@/api/easyrag'
import { errorMessage } from '@/lib/utils'
import { UploadIcon, TrashIcon, RefreshCwIcon, FileTextIcon, RotateCcwIcon, RotateCwIcon } from 'lucide-react'

export default function DocumentManager() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [isLoading, setIsLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isResetting, setIsResetting] = useState(false)
  const [isRetrying, setIsRetrying] = useState(false)
  const [retryingDocId, setRetryingDocId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pipelineStatus, setPipelineStatus] = useState<any>(null)

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response: PaginatedResponse = await documentApi.listDocuments(page, pageSize)
      setDocuments(response.documents || [])
      setTotal(response.total || 0)
    } catch (err: any) {
      setError(errorMessage(err))
    } finally {
      setIsLoading(false)
    }
  }, [page, pageSize])

  const fetchPipelineStatus = useCallback(async () => {
    try {
      const status = await documentApi.getPipelineStatus()
      setPipelineStatus(status)
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    fetchDocuments()
    fetchPipelineStatus()
  }, [fetchDocuments, fetchPipelineStatus])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setIsUploading(true)
    setError(null)
    try {
      for (const file of Array.from(files)) {
        await documentApi.uploadFile(file)
      }
      await fetchDocuments()
    } catch (err: any) {
      setError(errorMessage(err))
    } finally {
      setIsUploading(false)
      e.target.value = ''
    }
  }

  const handleProcess = async () => {
    setIsLoading(true)
    setError(null)
    try {
      await documentApi.processDocuments()
      await fetchDocuments()
      await fetchPipelineStatus()
    } catch (err: any) {
      setError(errorMessage(err))
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = async () => {
    if (!confirm('Reset all stuck "processing" documents back to pending?')) return
    setIsResetting(true)
    setError(null)
    try {
      await documentApi.resetProcessing()
      await fetchDocuments()
      await fetchPipelineStatus()
    } catch (err: any) {
      setError(errorMessage(err))
    } finally {
      setIsResetting(false)
    }
  }

  // Failed count from status counts (covers pages that show no failed rows);
  // falls back to the current page when counts are unavailable.
  const failedTotal =
    pipelineStatus?.counts?.failed ??
    documents.filter((d) => d.status === 'failed').length

  const handleRetryFailed = async () => {
    if (!confirm(`Reprocess ${failedTotal} failed document(s)? They will be reset to pending and processed again.`)) return
    setIsRetrying(true)
    setError(null)
    try {
      await documentApi.reprocessFailed()
      await fetchDocuments()
      await fetchPipelineStatus()
    } catch (err: any) {
      setError(errorMessage(err))
    } finally {
      setIsRetrying(false)
    }
  }

  const handleDelete = async (docId: string) => {
    if (!confirm('Are you sure you want to delete this document?')) return
    setError(null)
    try {
      await documentApi.deleteDocument(docId)
      await fetchDocuments()
    } catch (err: any) {
      setError(errorMessage(err))
    }
  }

  const handleRetry = async (docId: string) => {
    setRetryingDocId(docId)
    setError(null)
    try {
      await documentApi.reprocessDocument(docId)
      await fetchDocuments()
      await fetchPipelineStatus()
    } catch (err: any) {
      setError(errorMessage(err))
    } finally {
      setRetryingDocId(null)
    }
  }

  const totalPages = Math.ceil(total / pageSize)

  const statusColor = (status: string) => {
    switch (status) {
      case 'processed': return 'text-green-600 bg-green-50 dark:bg-green-900/20'
      case 'processing': return 'text-blue-600 bg-blue-50 dark:bg-blue-900/20'
      case 'pending': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20'
      case 'failed': return 'text-red-600 bg-red-50 dark:bg-red-900/20'
      default: return 'text-gray-600 bg-gray-50 dark:bg-gray-900/20'
    }
  }

  return (
    <div className="flex h-full flex-col p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Document Management</h2>
        <div className="flex items-center gap-2">
          <label className="cursor-pointer">
            <input
              type="file"
              multiple
              className="hidden"
              onChange={handleUpload}
              disabled={isUploading}
            />
            <span className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 cursor-pointer">
              <UploadIcon className="size-4" />
              {isUploading ? 'Uploading...' : 'Upload'}
            </span>
          </label>
          <Button variant="outline" size="sm" onClick={handleProcess} disabled={isLoading}>
            <RefreshCwIcon className="size-4" />
            Process
          </Button>
          {failedTotal > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleRetryFailed}
              disabled={isRetrying || pipelineStatus?.busy}
              title={
                pipelineStatus?.busy
                  ? 'Pipeline is busy; retry once it finishes'
                  : `Reprocess ${failedTotal} failed document(s)`
              }
            >
              <RotateCwIcon className="size-4" />
              {isRetrying ? 'Retrying...' : `Retry Failed (${failedTotal})`}
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={handleReset} disabled={isResetting}>
            <RotateCcwIcon className="size-4" />
            Reset Stuck
          </Button>
        </div>
      </div>

      {/* Pipeline status */}
      {pipelineStatus && (
        <div className="mb-4 rounded-md border p-3 text-sm">
          <span className="font-medium">Pipeline: </span>
          <span className={pipelineStatus.busy ? 'text-blue-600' : 'text-green-600'}>
            {pipelineStatus.busy ? 'Processing...' : 'Idle'}
          </span>
          {pipelineStatus.pending_count > 0 && (
            <span className="ml-2 text-muted-foreground">
              ({pipelineStatus.pending_count} pending)
            </span>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-4 rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Document list */}
      <div className="flex-1 overflow-auto rounded-md border">
        <table className="w-full text-sm">
          <thead className="border-b bg-muted/50">
            <tr>
              <th className="px-4 py-2 text-left font-medium">File</th>
              <th className="px-4 py-2 text-left font-medium">BizId</th>
              <th className="px-4 py-2 text-left font-medium">Status</th>
              <th className="px-4 py-2 text-left font-medium">Size</th>
              <th className="px-4 py-2 text-left font-medium">Created</th>
              <th className="px-4 py-2 text-right font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">
                  Loading...
                </td>
              </tr>
            ) : documents.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">
                  <FileTextIcon className="mx-auto mb-2 size-8 opacity-50" />
                  No documents yet. Upload files to get started.
                </td>
              </tr>
            ) : (
              documents.map((doc) => (
                <tr key={doc.id} className="border-b last:border-0 hover:bg-muted/30">
                  <td className="px-4 py-2 max-w-[300px] truncate" title={doc.filePath}>
                    {doc.filePath}
                  </td>
                  <td className="px-4 py-2 text-muted-foreground">
                    {doc.bizId || '-'}
                  </td>
                  <td className="px-4 py-2">
                    <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${statusColor(doc.status)}`}>
                      {doc.status}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-muted-foreground">
                    {doc.contentLength ? `${(doc.contentLength / 1024).toFixed(1)} KB` : '-'}
                  </td>
                  <td className="px-4 py-2 text-muted-foreground">
                    {doc.createdAt ? new Date(doc.createdAt).toLocaleDateString() : '-'}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7"
                      title="Reprocess this document (re-run extraction)"
                      disabled={retryingDocId === doc.id || pipelineStatus?.busy || doc.status === 'processing'}
                      onClick={() => handleRetry(doc.id)}
                    >
                      <RotateCwIcon
                        className={`size-3.5 ${retryingDocId === doc.id ? 'animate-spin' : ''}`}
                      />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7 text-destructive hover:text-destructive"
                      onClick={() => handleDelete(doc.id)}
                    >
                      <TrashIcon className="size-3.5" />
                    </Button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4">
          <span className="text-sm text-muted-foreground">
            {total} document{total !== 1 ? 's' : ''} total
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
            >
              Previous
            </Button>
            <span className="text-sm">
              Page {page} of {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage(page + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

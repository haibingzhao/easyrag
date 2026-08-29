import { useCallback } from 'react'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { Button } from '@/components/ui/Button'
import { XIcon, ExpandIcon, TrashIcon } from 'lucide-react'

const PropertiesView = () => {
  const selectedNode = useGraphStore.use.selectedNode()
  const selectedEdge = useGraphStore.use.selectedEdge()
  const rawGraph = useGraphStore.use.rawGraph()
  const clearSelection = useGraphStore.use.clearSelection()
  const triggerNodeExpand = useGraphStore.use.triggerNodeExpand()
  const triggerNodePrune = useGraphStore.use.triggerNodePrune()

  const getNode = useCallback(
    (nodeId: string) => rawGraph?.getNode(nodeId) || null,
    [rawGraph]
  )

  const getEdge = useCallback(
    (edgeId: string) => rawGraph?.getEdge(edgeId, true) || null,
    [rawGraph]
  )

  if (!selectedNode && !selectedEdge) return null

  const node = selectedNode ? getNode(selectedNode) : null
  const edge = selectedEdge ? getEdge(selectedEdge) : null

  return (
    <div className="bg-background border rounded-lg shadow-lg w-80 max-h-[60vh] flex flex-col">
      <div className="flex items-center justify-between p-3 border-b">
        <h3 className="font-semibold text-sm truncate">
          {node ? 'Node Properties' : 'Edge Properties'}
        </h3>
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={clearSelection}>
          <XIcon className="h-4 w-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1 p-3">
        {node && (
          <div className="space-y-3">
            <div>
              <span className="text-xs font-medium text-muted-foreground">ID</span>
              <p className="text-sm break-all">{node.id}</p>
            </div>
            <div>
              <span className="text-xs font-medium text-muted-foreground">Labels</span>
              <p className="text-sm">{node.labels?.join(', ') || 'N/A'}</p>
            </div>
            {node.properties && (
              <div>
                <span className="text-xs font-medium text-muted-foreground">Properties</span>
                <div className="mt-1 space-y-1">
                  {Object.entries(node.properties).map(([key, value]) => (
                    <div key={key} className="text-xs">
                      <span className="font-medium">{key}: </span>
                      <span className="text-muted-foreground break-all">
                        {typeof value === 'string' ? value : JSON.stringify(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div>
              <span className="text-xs font-medium text-muted-foreground">Degree</span>
              <p className="text-sm">{node.degree ?? 0}</p>
            </div>
            <div className="flex gap-2 pt-2">
              <Button
                variant="outline"
                size="sm"
                className="flex-1"
                onClick={() => triggerNodeExpand(node.id)}
              >
                <ExpandIcon className="h-3 w-3 mr-1" />
                Expand
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="flex-1 text-destructive"
                onClick={() => triggerNodePrune(node.id)}
              >
                <TrashIcon className="h-3 w-3 mr-1" />
                Prune
              </Button>
            </div>
          </div>
        )}

        {edge && (
          <div className="space-y-3">
            <div>
              <span className="text-xs font-medium text-muted-foreground">ID</span>
              <p className="text-sm break-all">{edge.id}</p>
            </div>
            <div>
              <span className="text-xs font-medium text-muted-foreground">Source</span>
              <p className="text-sm break-all">{edge.source}</p>
            </div>
            <div>
              <span className="text-xs font-medium text-muted-foreground">Target</span>
              <p className="text-sm break-all">{edge.target}</p>
            </div>
            {edge.properties && (
              <div>
                <span className="text-xs font-medium text-muted-foreground">Properties</span>
                <div className="mt-1 space-y-1">
                  {Object.entries(edge.properties).map(([key, value]) => (
                    <div key={key} className="text-xs">
                      <span className="font-medium">{key}: </span>
                      <span className="text-muted-foreground break-all">
                        {typeof value === 'string' ? value : JSON.stringify(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </ScrollArea>
    </div>
  )
}

export default PropertiesView

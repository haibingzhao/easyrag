import { useState, useCallback, useRef } from 'react'
import { useGraphStore } from '@/stores/graph'
import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { SearchIcon, XIcon } from 'lucide-react'

const GraphSearch = () => {
  const [searchTerm, setSearchTerm] = useState('')
  const [results, setResults] = useState<string[]>([])
  const [isOpen, setIsOpen] = useState(false)
  const sigmaGraph = useGraphStore.use.sigmaGraph()
  const setSelectedNode = useGraphStore.use.setSelectedNode()
  const setMoveToSelectedNode = useGraphStore.use.setMoveToSelectedNode()
  const inputRef = useRef<HTMLInputElement>(null)

  const handleSearch = useCallback(
    (term: string) => {
      setSearchTerm(term)
      if (!sigmaGraph || term.length < 1) {
        setResults([])
        setIsOpen(false)
        return
      }

      const matches: string[] = []
      const lowerTerm = term.toLowerCase()
      sigmaGraph.forEachNode((node, attributes) => {
        if (matches.length >= 20) return
        const label = (attributes.label || node).toLowerCase()
        if (label.includes(lowerTerm)) {
          matches.push(node)
        }
      })

      setResults(matches)
      setIsOpen(matches.length > 0)
    },
    [sigmaGraph]
  )

  const handleSelect = useCallback(
    (nodeId: string) => {
      setSelectedNode(nodeId)
      setMoveToSelectedNode(true)
      setIsOpen(false)
      setSearchTerm('')
      setResults([])
    },
    [setSelectedNode, setMoveToSelectedNode]
  )

  const handleClear = useCallback(() => {
    setSearchTerm('')
    setResults([])
    setIsOpen(false)
  }, [])

  return (
    <div className="relative">
      <div className="flex items-center gap-1">
        <div className="relative flex-1">
          <SearchIcon className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            ref={inputRef}
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
            className="h-7 pl-7 text-xs"
          />
          {searchTerm && (
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-0 top-0 h-7 w-7"
              onClick={handleClear}
            >
              <XIcon className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>

      {isOpen && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-popover border rounded-md shadow-md z-50 max-h-48 overflow-y-auto">
          {results.map((nodeId) => (
            <button
              key={nodeId}
              className="w-full text-left px-3 py-1.5 text-xs hover:bg-accent truncate"
              onClick={() => handleSelect(nodeId)}
            >
              {sigmaGraph?.getNodeAttribute(nodeId, 'label') || nodeId}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default GraphSearch

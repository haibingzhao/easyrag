import { useEffect, useState } from 'react'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import { getAllGraphLabels } from '@/api/easyrag'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { TagIcon } from 'lucide-react'

const GraphLabels = () => {
  const [labels, setLabels] = useState<string[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const queryLabel = useSettingsStore.use.queryLabel()
  const setQueryLabel = useSettingsStore.use.setQueryLabel()
  const labelsFetchAttempted = useGraphStore.use.labelsFetchAttempted()

  useEffect(() => {
    if (!labelsFetchAttempted) return

    const fetchLabels = async () => {
      try {
        const result = await getAllGraphLabels()
        if (result && Array.isArray(result)) {
          setLabels(result)
        }
      } catch (error) {
        console.error('Failed to fetch graph labels:', error)
      }
    }

    fetchLabels()
  }, [labelsFetchAttempted])

  const filteredLabels = labels.filter((label) =>
    label.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="flex flex-col gap-2 p-2">
      <div className="flex items-center gap-2">
        <TagIcon className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">Labels</span>
      </div>
      <Input
        placeholder="Search labels..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="h-7 text-xs"
      />
      <div className="flex flex-wrap gap-1 max-h-32 overflow-y-auto">
        {filteredLabels.map((label) => (
          <Button
            key={label}
            variant={queryLabel === label ? 'default' : 'outline'}
            size="sm"
            className="h-6 text-xs px-2"
            onClick={() => setQueryLabel(label === queryLabel ? '' : label)}
          >
            {label}
          </Button>
        ))}
      </div>
    </div>
  )
}

export default GraphLabels

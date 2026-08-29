import { useSettingsStore } from '@/stores/settings'
import { Textarea } from '@/components/ui/Textarea'
import { Label } from '@/components/ui/Label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/Select'

export interface QuerySettingsProps {
  onSettingsChange?: (settings: Record<string, any>) => void
}

const QuerySettings = ({ onSettingsChange: _onSettingsChange }: QuerySettingsProps) => {
  const querySettings = useSettingsStore.use.querySettings()
  const updateQuerySettings = useSettingsStore.use.updateQuerySettings()

  const modes = [
    { value: 'local', label: 'Local' },
    { value: 'global', label: 'Global' },
    { value: 'hybrid', label: 'Hybrid' },
    { value: 'naive', label: 'Naive' },
    { value: 'mix', label: 'Mix' },
    { value: 'structure', label: 'Structure' },
    { value: 'bypass', label: 'Bypass' }
  ]

  return (
    <div className="space-y-4 p-4">
      <div className="space-y-2">
        <Label>Query Mode</Label>
        <Select
          value={querySettings.mode}
          onValueChange={(v) => updateQuerySettings({ mode: v })}
        >
          <SelectTrigger className="h-8 text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {modes.map((m) => (
              <SelectItem key={m.value} value={m.value}>
                {m.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Top K</Label>
          <input
            type="number"
            value={querySettings.top_k}
            onChange={(e) => updateQuerySettings({ top_k: parseInt(e.target.value) || 10 })}
            className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            min={1}
            max={100}
          />
        </div>
        <div className="space-y-2">
          <Label>Chunk Top K</Label>
          <input
            type="number"
            value={querySettings.chunk_top_k}
            onChange={(e) => updateQuerySettings({ chunk_top_k: parseInt(e.target.value) || 5 })}
            className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            min={1}
            max={50}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Max Entity Tokens</Label>
          <input
            type="number"
            value={querySettings.max_entity_tokens}
            onChange={(e) => updateQuerySettings({ max_entity_tokens: parseInt(e.target.value) || 4000 })}
            className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            min={100}
            max={32000}
          />
        </div>
        <div className="space-y-2">
          <Label>Max Relation Tokens</Label>
          <input
            type="number"
            value={querySettings.max_relation_tokens}
            onChange={(e) => updateQuerySettings({ max_relation_tokens: parseInt(e.target.value) || 4000 })}
            className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            min={100}
            max={32000}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Max Total Tokens</Label>
          <input
            type="number"
            value={querySettings.max_total_tokens}
            onChange={(e) => updateQuerySettings({ max_total_tokens: parseInt(e.target.value) || 4000 })}
            className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            min={100}
            max={64000}
          />
        </div>
        <div className="space-y-2">
          <Label>History Turns</Label>
          <input
            type="number"
            value={querySettings.history_turns}
            onChange={(e) => updateQuerySettings({ history_turns: parseInt(e.target.value) || 0 })}
            className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            min={0}
            max={20}
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label>User Prompt (Optional)</Label>
        <Textarea
          value={querySettings.user_prompt}
          onChange={(e) => updateQuerySettings({ user_prompt: e.target.value })}
          placeholder="Additional instructions for the response..."
          className="min-h-[60px] text-sm"
        />
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="enableRerank"
          checked={querySettings.enable_rerank}
          onChange={(e) => updateQuerySettings({ enable_rerank: e.target.checked })}
          className="rounded border-gray-300"
        />
        <Label htmlFor="enableRerank" className="cursor-pointer">Enable Rerank</Label>
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="streamMode"
          checked={querySettings.stream}
          onChange={(e) => updateQuerySettings({ stream: e.target.checked })}
          className="rounded border-gray-300"
        />
        <Label htmlFor="streamMode" className="cursor-pointer">Stream Response</Label>
      </div>
    </div>
  )
}

export default QuerySettings

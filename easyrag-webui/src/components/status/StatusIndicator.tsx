import { useBackendState } from '@/stores/state'
import { cn } from '@/lib/utils'

const StatusIndicator = () => {
  const isHealthy = useBackendState.use.health()
  const message = useBackendState.use.message()

  return (
    <div className="flex items-center gap-2">
      <div
        className={cn(
          'h-2 w-2 rounded-full',
          isHealthy ? 'bg-green-500' : 'bg-red-500'
        )}
        title={message || (isHealthy ? 'Connected' : 'Disconnected')}
      />
      <span className="text-xs text-muted-foreground">
        {isHealthy ? 'Online' : 'Offline'}
      </span>
    </div>
  )
}

export default StatusIndicator

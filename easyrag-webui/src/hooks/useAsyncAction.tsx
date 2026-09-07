import { useCallback, useEffect, useRef, useState } from 'react'
import { errorMessage } from '@/lib/utils'

interface AsyncActionOptions {
  /** Invoked right before the action starts (e.g. to clear a shared error). */
  onStart?: () => void
  /** Invoked with the normalized message when the action throws. */
  onError?: (message: string) => void
}

/**
 * Wraps the repeated "set pending -> clear error -> try/catch/finally"
 * boilerplate around a one-shot async UI action.
 *
 * Each call site gets its own `isPending` flag while sharing error reporting
 * through `onError`, so several buttons can track their busy state
 * independently but surface failures on one banner.
 *
 * The options are mirrored into a ref (kept current in an effect, never
 * mutated during render), so `run` stays referentially stable and callers can
 * pass inline callbacks without it landing in a dependency array.
 */
export function useAsyncAction(options: AsyncActionOptions = {}) {
  const [isPending, setIsPending] = useState(false)
  const optionsRef = useRef(options)

  useEffect(() => {
    optionsRef.current = options
  })

  const run = useCallback(async (action: () => Promise<void>) => {
    optionsRef.current.onStart?.()
    setIsPending(true)
    try {
      await action()
    } catch (err) {
      optionsRef.current.onError?.(errorMessage(err))
    } finally {
      setIsPending(false)
    }
  }, [])

  return { run, isPending }
}

export default useAsyncAction

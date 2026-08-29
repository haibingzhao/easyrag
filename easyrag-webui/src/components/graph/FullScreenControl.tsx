import { useState, useCallback } from 'react'
import { useSigma } from '@react-sigma/core'
import { Button } from '@/components/ui/Button'
import { MaximizeIcon, MinimizeIcon } from 'lucide-react'

const FullScreenControl = () => {
  const [isFullScreen, setIsFullScreen] = useState(false)
  const sigma = useSigma()

  const toggleFullScreen = useCallback(() => {
    const container = sigma.getContainer()
    if (!document.fullscreenElement) {
      container.requestFullscreen().then(() => setIsFullScreen(true))
    } else {
      document.exitFullscreen().then(() => setIsFullScreen(false))
    }
  }, [sigma])

  return (
    <Button variant="outline" size="icon" className="h-8 w-8" onClick={toggleFullScreen}>
      {isFullScreen ? <MinimizeIcon className="h-4 w-4" /> : <MaximizeIcon className="h-4 w-4" />}
    </Button>
  )
}

export default FullScreenControl

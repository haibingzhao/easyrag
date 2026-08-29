import { useSigma } from '@react-sigma/core'
import { Button } from '@/components/ui/Button'
import { ZoomInIcon, ZoomOutIcon, MaximizeIcon } from 'lucide-react'

const ZoomControl = () => {
  const sigma = useSigma()

  const zoomIn = () => {
    const camera = sigma.getCamera()
    camera.animatedZoom({ duration: 300 })
  }

  const zoomOut = () => {
    const camera = sigma.getCamera()
    camera.animatedUnzoom({ duration: 300 })
  }

  const resetZoom = () => {
    const camera = sigma.getCamera()
    camera.animatedReset({ duration: 300 })
  }

  return (
    <div className="flex flex-col gap-1">
      <Button variant="outline" size="icon" className="h-8 w-8" onClick={zoomIn}>
        <ZoomInIcon className="h-4 w-4" />
      </Button>
      <Button variant="outline" size="icon" className="h-8 w-8" onClick={zoomOut}>
        <ZoomOutIcon className="h-4 w-4" />
      </Button>
      <Button variant="outline" size="icon" className="h-8 w-8" onClick={resetZoom}>
        <MaximizeIcon className="h-4 w-4" />
      </Button>
    </div>
  )
}

export default ZoomControl

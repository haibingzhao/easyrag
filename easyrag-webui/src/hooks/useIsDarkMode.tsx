import { useEffect, useState } from 'react'

const useIsDarkMode = (): boolean => {
  const [isDark, setIsDark] = useState(() => {
    return document.documentElement.classList.contains('dark')
  })

  useEffect(() => {
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
          setIsDark(document.documentElement.classList.contains('dark'))
        }
      }
    })

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    })

    // Set initial value
    setIsDark(document.documentElement.classList.contains('dark'))

    return () => observer.disconnect()
  }, [])

  return isDark
}

export default useIsDarkMode

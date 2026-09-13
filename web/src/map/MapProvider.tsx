import { Map as MapLibreMap, NavigationControl, ScaleControl } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { createContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { BLANK_STYLE, CARTO_LIGHT_STYLE } from './basemap'
import { createMapController, type MapController } from './mapController'

interface MapContextValue {
  controller: MapController | null
  containerRef: React.RefObject<HTMLDivElement | null>
  basemapAvailable: boolean
}

export const MapContext = createContext<MapContextValue>({
  controller: null,
  containerRef: { current: null },
  basemapAvailable: true,
})

export function MapProvider({ children }: { children: ReactNode }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [controller, setController] = useState<MapController | null>(null)
  const [basemapAvailable, setBasemapAvailable] = useState(true)

  useEffect(() => {
    if (!containerRef.current) return

    // No "created once ever" ref guard: React 18/19 StrictMode runs this effect's mount ->
    // cleanup -> mount again in development specifically to catch effects that aren't safely
    // re-runnable. A guard that only allows creation once would skip the second mount after the
    // first cleanup already called `map.remove()`, leaving no live map at all -- so this must
    // create and tear down a real map on every run, which is what makes it StrictMode-safe.
    const map = new MapLibreMap({
      container: containerRef.current,
      style: CARTO_LIGHT_STYLE,
      center: [0, 55],
      zoom: 4,
      attributionControl: { compact: true },
    })
    map.addControl(new NavigationControl({ showCompass: false }), 'top-right')
    map.addControl(new ScaleControl({ unit: 'metric' }), 'bottom-left')

    map.on('error', (event) => {
      // A tile/style load failure (e.g. no network reaching the basemap host) falls back to a
      // blank ground -- project layers must remain fully usable regardless.
      const error = (event as unknown as { error?: { message?: string } }).error
      if (error && /carto|tile/i.test(String(error.message ?? ''))) {
        setBasemapAvailable(false)
        map.setStyle(BLANK_STYLE)
      }
    })

    map.on('load', () => {
      setController(createMapController(map))
    })

    const resizeObserver = new ResizeObserver(() => map.resize())
    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      map.remove()
    }
  }, [])

  return (
    <MapContext.Provider value={{ controller, containerRef, basemapAvailable }}>
      {children}
    </MapContext.Provider>
  )
}

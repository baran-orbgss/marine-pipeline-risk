import { AppShell } from './app/AppShell'
import { MapProvider } from './map/MapProvider'

export function App() {
  return (
    <MapProvider>
      <AppShell />
    </MapProvider>
  )
}

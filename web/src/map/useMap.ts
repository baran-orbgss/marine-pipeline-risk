import { useContext } from 'react'
import { MapContext } from './MapProvider'

export function useMap() {
  return useContext(MapContext)
}

import type { StyleSpecification } from 'maplibre-gl'

// OpenStreetMap's standard tile server -- a no-token, no-API-key raster basemap (the CARTO
// Positron CDN this originally targeted now gates its subdomain endpoint behind an API key).
// Chosen per the product requirement that the workspace must remain usable if external tiles are
// unavailable: this style has no vector glyphs/sprites to fail to load, just raster tiles, so a
// network failure degrades to "blank ground" rather than a broken/half-rendered style.
export const CARTO_LIGHT_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    'osm-light': {
      type: 'raster',
      tiles: [
        'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png',
        'https://b.tile.openstreetmap.org/{z}/{x}/{y}.png',
        'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png',
      ],
      tileSize: 256,
      attribution: '&copy; OpenStreetMap contributors',
    },
  },
  layers: [
    { id: 'background', type: 'background', paint: { 'background-color': '#eef2f5' } },
    { id: 'osm-light', type: 'raster', source: 'osm-light' },
  ],
}

// Used when the basemap tiles genuinely fail to load (offline environment, blocked host): a
// plain background colour, so project layers remain fully usable with no broken tile requests.
export const BLANK_STYLE: StyleSpecification = {
  version: 8,
  sources: {},
  layers: [{ id: 'background', type: 'background', paint: { 'background-color': '#dfe6ea' } }],
}

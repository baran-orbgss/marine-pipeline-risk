/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // maplibre-gl loads its own worker bundle via a dynamically constructed URL; Vite's dependency
  // pre-bundling (esbuild) doesn't preserve that worker file at the path maplibre-gl expects,
  // which 404s and silently stalls all tile loading. Excluding it from pre-bundling lets the
  // package's own worker-loading logic resolve against its real, unbundled file layout instead.
  optimizeDeps: {
    exclude: ['maplibre-gl'],
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setupTests.ts'],
    globals: true,
  },
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    /** Listen on LAN so http://localhost:3000 works from the host browser (not only inside sandboxes). */
    host: true,
    strictPort: true,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ['plotly.js', 'react-plotly.js'],
          three: ['three', '@react-three/fiber', '@react-three/drei'],
          framer: ['framer-motion'],
        }
      }
    }
  }
})

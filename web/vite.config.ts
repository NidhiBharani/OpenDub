import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 127.0.0.1, not localhost: on Node 18+ localhost resolves to IPv6 ::1
      // first, but the API binds IPv4 only — proxying to localhost 500s.
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})

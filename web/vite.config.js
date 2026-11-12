import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  base: './', // relative URLs so the build works under any GitHub Pages path
  optimizeDeps: { exclude: ['@mujoco/mujoco', 'onnxruntime-web'] }, // keep loaders next to their .wasm
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    rollupOptions: {
      input: { main: resolve(import.meta.dirname, 'index.html'), openarm: resolve(import.meta.dirname, 'openarm.html') },
    },
  },
  server: { port: 5173, strictPort: false },
});

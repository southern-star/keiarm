// Vite also emits a hashed copy of onnxruntime-web's .wasm into dist/assets/. The app loads the
// runtime from dist/ort/ (see policy_worker.js), so the copy is dead weight (29 MB): remove it.
import { readdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const assets = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist', 'assets');
for (const f of readdirSync(assets)) {
  if (/^ort-wasm-.*\.wasm$/.test(f)) {
    rmSync(join(assets, f));
    console.log(`removed unused dist/assets/${f}`);
  }
}

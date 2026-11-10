// Copy onnxruntime-web's runtime files (.wasm + loader .mjs) into public/ort/ so the app
// serves them itself instead of fetching them from a CDN. Runs before dev and build.
import { cpSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(web, 'node_modules', 'onnxruntime-web', 'dist');
const dst = join(web, 'public', 'ort');
rmSync(dst, { recursive: true, force: true });
mkdirSync(dst, { recursive: true });
// The 'onnxruntime-web/webgpu' entry requests only the asyncify build (it serves both the
// WebGPU and the WASM execution providers).
const wanted = /^ort-wasm-simd-threaded\.asyncify\.(mjs|wasm)$/;
const files = readdirSync(src).filter((f) => wanted.test(f));
for (const f of files) cpSync(join(src, f), join(dst, f));
console.log(`copied ${files.length} onnxruntime-web files to public/ort/`);

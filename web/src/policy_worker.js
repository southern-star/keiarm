// Runs the KeiGrasp ONNX model off the main thread so rendering never stalls.
// WebGPU when available, otherwise WASM (single-threaded unless the page is cross-origin isolated).
import * as ort from 'onnxruntime-web/webgpu';

let session = null;
let shape = null; // { cameras, height, width, chunk, dim }

async function init({ modelUrl, ortPath, prefer, cameras, height, width, chunk, dim = 6 }) {
  ort.env.wasm.wasmPaths = ortPath;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  shape = { cameras, height, width, chunk, dim };
  const order = prefer === 'wasm' || !('gpu' in navigator) ? ['wasm'] : ['webgpu', 'wasm'];
  const errors = [];
  for (const backend of order) {
    try {
      session = await ort.InferenceSession.create(modelUrl, { executionProviders: [backend] });
      // Warm-up run: the first call compiles shaders / allocates and is much slower.
      const t0 = performance.now();
      await run(new Array(cameras).fill(new Uint8ClampedArray(height * width * 4)), new Float32Array(dim));
      return { backend, threads: ort.env.wasm.numThreads, warmupMs: performance.now() - t0 };
    } catch (err) {
      errors.push(`${backend}: ${err.message ?? err}`);
      session = null;
    }
  }
  throw new Error(`could not create an inference session (${errors.join('; ')})`);
}

// images: one RGBA byte array per camera, row 0 = top. Returns Float32Array(chunk * dim) [rad].
async function run(images, joints) {
  const { cameras, height, width, dim } = shape;
  const plane = height * width;
  const input = new Float32Array(cameras * 3 * plane);
  for (let c = 0; c < cameras; c++) {
    const rgba = images[c];
    const base = c * 3 * plane;
    for (let i = 0; i < plane; i++) {
      input[base + i] = rgba[4 * i] / 255;
      input[base + plane + i] = rgba[4 * i + 1] / 255;
      input[base + 2 * plane + i] = rgba[4 * i + 2] / 255;
    }
  }
  const out = await session.run({
    images: new ort.Tensor('float32', input, [1, cameras, 3, height, width]),
    joints: new ort.Tensor('float32', Float32Array.from(joints), [1, dim]),
  });
  return out.actions.data;
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      self.postMessage({ id: data.id, ok: true, result: await init(data) });
    } else if (data.type === 'infer') {
      const t0 = performance.now();
      const actions = await run(data.images, data.joints);
      self.postMessage({ id: data.id, ok: true, result: { actions, ms: performance.now() - t0 } });
    }
  } catch (err) {
    self.postMessage({ id: data.id, ok: false, error: String(err.message ?? err) });
  }
};

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import { Sim } from './sim.js';
import { SceneView } from './render.js';
import { ChunkRunner, PolicyClient } from './policy.js';

const $ = (id) => document.getElementById(id);
const base = import.meta.env.BASE_URL;
const params = new URLSearchParams(location.search);

function showError(err) {
  console.error(err);
  $('status').innerHTML = `<span class="err">エラー: ${err.message}</span>`;
}

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas: $('main'), antialias: true, preserveDrawingBuffer: true });
} catch (err) {
  showError(new Error('WebGL を初期化できません。WebGL2 が有効なブラウザで開いてください。'));
  throw err;
}
renderer.outputColorSpace = THREE.LinearSRGBColorSpace; // write shader output as it is
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const viewCamera = new THREE.PerspectiveCamera(40, 1, 0.01, 20);
viewCamera.up.set(0, 0, 1);
viewCamera.position.set(0.75, -0.55, 0.5);
const controls = new OrbitControls(viewCamera, renderer.domElement);
controls.target.set(0.15, 0, 0.06);
controls.update();

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  viewCamera.aspect = window.innerWidth / window.innerHeight;
  viewCamera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

const meshCache = new Map();
async function fetchMesh(name) {
  if (!meshCache.has(name)) {
    const res = await fetch(`${base}assets/so101/${name}`);
    if (!res.ok) throw new Error(`mesh ${name}: HTTP ${res.status}`);
    meshCache.set(name, new Uint8Array(await res.arrayBuffer()));
  }
  return meshCache.get(name);
}

const app = {
  mujoco: null, sim: null, view: null, bundle: null, episode: null,
  mode: 'model', // 'model' | 'expert'
  playing: false, cursor: 0,
  policy: null, runner: null, policyError: null,
};
const camCtx = [$('cam0').getContext('2d'), $('cam1').getContext('2d')];

async function loadScene(seed) {
  const res = await fetch(`${base}scenes/${seed}.json`);
  if (!res.ok) throw new Error(`scene ${seed}: HTTP ${res.status}`);
  const bundle = await res.json();
  const meshes = new Map(await Promise.all(bundle.mesh_files.map(async (f) => [f, await fetchMesh(f)])));
  app.playing = false;
  // Tear down the previous scene first: its view reads from WASM objects that are about to
  // be freed (they are not garbage collected, so they must be deleted explicitly).
  app.view?.dispose();
  app.view = null;
  app.sim?.dispose();
  app.sim = null;
  app.sim = new Sim(app.mujoco, bundle, meshes);
  app.bundle = bundle;
  $('episode').replaceChildren(...bundle.episodes.map((e, i) => new Option(String(e.episode_seed), String(i))));
  selectEpisode(0);
  app.view = new SceneView(app.sim);
}

function selectEpisode(index) {
  app.episode = app.bundle.episodes[index];
  $('episode').value = String(index);
  app.sim.reset(app.episode);
  app.view?.sync();
  app.cursor = 0;
  app.playing = false;
  app.runner?.reset();
}

function setMode(mode) {
  app.mode = mode;
  $('mode-model').setAttribute('aria-pressed', String(mode === 'model'));
  $('mode-expert').setAttribute('aria-pressed', String(mode === 'expert'));
  if (app.sim) selectEpisode(Number($('episode').value));
}

// What the policy sees: both camera images (RGBA bytes) and the six joint angles.
function observe() {
  app.view.sync();
  return {
    images: app.view.cameras.map(({ name }) => app.view.renderCamera(renderer, name)),
    joints: app.sim.joints(),
  };
}

function drawPolicyCameras() {
  const { view } = app;
  view.cameras.forEach(({ name }, i) => {
    const pixels = view.renderCamera(renderer, name);
    camCtx[i].putImageData(new ImageData(pixels, view.width, view.height), 0, 0);
  });
}

function updateStatus() {
  const { sim, runner } = app;
  const verdict = sim.succeeded ? '<span class="ok">皿の上に置けています</span>' : '未達';
  const lines = [`ステップ ${sim.steps}（${(sim.steps / sim.c.control_hz).toFixed(1)} 秒） 判定: ${verdict}`];
  if (app.mode === 'model') {
    if (app.policyError) lines.push(`<span class="err">モデルを読み込めません: ${app.policyError}</span>`);
    else if (!app.policy?.info) lines.push('モデルを読み込み中…');
    else {
      const ms = runner.lastMs ? `${runner.lastMs.toFixed(0)} ms` : '—';
      lines.push(`推論: ${app.policy.info.backend.toUpperCase()}、直近 ${ms}、${runner.inferences} 回`);
    }
  } else {
    lines.push(`エキスパートの記録を再生（${app.cursor} / ${app.episode.expert.actions.length}）`);
  }
  $('status').innerHTML = lines.join('\n');
}

// One 30 Hz control tick in real time. Returns true if the simulation advanced.
function tick() {
  if (!app.playing) return false;
  if (app.mode === 'expert') {
    const actions = app.episode.expert.actions;
    if (app.cursor >= actions.length) {
      app.playing = false;
      return false;
    }
    app.sim.step(actions[app.cursor++]);
    return true;
  }
  if (!app.policy?.info) return false;
  const action = app.runner.actRealtime(app.sim.steps, observe);
  if (!action) return false; // waiting for the first plan: hold the simulation
  app.sim.step(action);
  return true;
}

let last = performance.now();
let accumulator = 0;
function frame(now) {
  requestAnimationFrame(frame);
  if (!app.sim || !app.view) return;
  const period = 1000 / app.sim.c.control_hz;
  accumulator = Math.min(accumulator + (now - last), 5 * period);
  last = now;
  while (accumulator >= period) {
    accumulator -= period;
    if (drag.active) drag.apply();
    tick();
  }
  app.view.sync();
  controls.update();
  drawPolicyCameras();
  renderer.setRenderTarget(null);
  renderer.render(app.view.scene, viewCamera);
  updateStatus();
}

// --- Dragging the cube ---------------------------------------------------------------------
// Pointer on the cube: move it across the table (kinematically, at rest) until released.
const raycaster = new THREE.Raycaster();
const drag = {
  active: false,
  target: null, // [x, y] on the table
  apply() {
    if (this.target) app.sim.placeCube(this.target[0], this.target[1]);
  },
};

function pointerRay(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, viewCamera);
  return raycaster.ray;
}

function tablePoint(event) {
  const { sim } = app;
  const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -sim.scene.cube_size / 2);
  const hit = pointerRay(event).intersectPlane(plane, new THREE.Vector3());
  if (!hit) return null;
  const [cx, cy] = sim.c.table_center;
  const [hx, hy] = sim.c.table_half_size;
  const margin = sim.scene.cube_size;
  return [Math.min(cx + hx - margin, Math.max(cx - hx + margin, hit.x)),
    Math.min(cy + hy - margin, Math.max(cy - hy + margin, hit.y))];
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  if (!app.view || event.button !== 0) return;
  const cube = app.view.geoms.find((g) => g.id === app.sim.cubeGeom)?.mesh;
  pointerRay(event);
  if (!cube || raycaster.intersectObject(cube).length === 0) return;
  drag.active = true;
  drag.target = tablePoint(event);
  controls.enabled = false;
  renderer.domElement.setPointerCapture(event.pointerId);
}, { capture: true });
renderer.domElement.addEventListener('pointermove', (event) => {
  if (drag.active) drag.target = tablePoint(event) ?? drag.target;
});
const endDrag = () => {
  if (!drag.active) return;
  drag.apply();
  drag.active = false;
  drag.target = null;
  controls.enabled = true;
};
renderer.domElement.addEventListener('pointerup', endDrag);
renderer.domElement.addEventListener('pointercancel', endDrag);

// --- Hooks for headless tools (tools/browser_check.py, tools/browser_eval.py) -------------

// Replay an episode's recorded expert actions as fast as possible (no rendering).
function replayNow(index) {
  selectEpisode(index);
  for (const action of app.episode.expert.actions) app.sim.step(action);
  app.view.sync();
  const s = app.sim.state();
  return { success: app.sim.succeeded, steps: app.sim.steps, cubePos: s.cubePos,
    pythonSuccess: app.episode.expert.success, pythonCubePos: app.episode.expert.final_cube_pos };
}

// Render both policy cameras for an explicit simulator state (render comparison tool).
function renderState(qpos, mocapPos) {
  app.playing = false;
  app.sim.setState(qpos, mocapPos);
  app.view.sync();
  const out = {};
  for (const { name } of app.view.cameras) out[name] = Array.from(app.view.renderCamera(renderer, name));
  return out;
}

// Closed-loop model episode in simulation time (the simulation waits for each inference, as in
// the Python harness). `disturb`: apply the episode's planned cube teleport.
async function runModelEpisode(index, { maxSteps, disturb = false } = {}) {
  await policyReady;
  selectEpisode(index);
  const { sim, runner, episode } = app;
  const limit = maxSteps ?? sim.c.max_episode_steps;
  const teleport = disturb ? episode.teleport : null;
  while (sim.steps < limit && !sim.succeeded) {
    if (teleport && sim.steps === teleport.step) sim.placeCube(teleport.x, teleport.y, teleport.yaw);
    sim.step(await runner.actBlocking(sim.steps, observe));
  }
  app.view.sync();
  return { success: sim.succeeded, steps: sim.steps, inferences: runner.inferences,
    meanInferMs: runner.inferences ? runner.totalMs / runner.inferences : 0 };
}

let policyReady;
async function initPolicy() {
  const meta = await (await fetch(`${base}models/keigrasp.json`)).json();
  app.policy = new PolicyClient();
  app.runner = new ChunkRunner(app.policy, { chunk: meta.chunk, execute: app.sim.c.policy_execute_steps });
  return app.policy.init({
    modelUrl: new URL(`${base}models/keigrasp.onnx`, location.href).href,
    ortPath: new URL(`${base}ort/`, location.href).href,
    prefer: params.get('backend') ?? 'auto',
    cameras: meta.cameras.length, height: meta.image_height, width: meta.image_width, chunk: meta.chunk,
  });
}

async function init() {
  app.mujoco = await loadMujoco();
  const index = await (await fetch(`${base}scenes/index.json`)).json();
  $('scene').replaceChildren(...index.scenes.map((s) => new Option(String(s), String(s))));
  const first = params.get('scene') ?? String(index.scenes[0]);
  $('scene').value = first;
  await loadScene(Number(first));
  if (params.has('episode')) selectEpisode(Number(params.get('episode')));
  if (params.get('mode') === 'expert') setMode('expert');

  $('scene').addEventListener('change', (e) => loadScene(Number(e.target.value)).catch(showError));
  $('episode').addEventListener('change', (e) => selectEpisode(Number(e.target.value)));
  $('reset').addEventListener('click', () => selectEpisode(Number($('episode').value)));
  $('mode-model').addEventListener('click', () => setMode('model'));
  $('mode-expert').addEventListener('click', () => setMode('expert'));
  $('play').addEventListener('click', () => {
    if (app.mode === 'expert' && app.cursor >= app.episode.expert.actions.length) {
      selectEpisode(Number($('episode').value));
    }
    app.playing = true;
  });
  requestAnimationFrame(frame);

  // The model (37 MB) loads in the background; the scene is usable meanwhile.
  policyReady = initPolicy();
  policyReady.catch((err) => {
    app.policyError = err.message;
    console.error(err);
  });
  return index;
}

window.keiarm = { app, THREE, camera: viewCamera, loadScene, selectEpisode, setMode, replayNow, renderState, runModelEpisode,
  policyReady: () => policyReady, ready: init() };
window.keiarm.ready.catch(showError);

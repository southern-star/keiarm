// OpenArm page (milestone 2, Phase 4): replay the expert's recorded episodes in MuJoCo WASM,
// draw the free view and the three policy cameras. Hooks for tools/browser_check_openarm.py
// are on window.keiarm.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import { OpenArmSim } from './sim.js';
import { SceneView } from '../render.js';

const $ = (id) => document.getElementById(id);
const base = import.meta.env.BASE_URL;
const params = new URLSearchParams(location.search);
const DISTURB_JA = { move: '棒を移動', knock: '持ち上げた棒を落とす', drop: '受け渡し後に棒を落とす', plate: '皿を移動' };

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
viewCamera.position.set(1.15, -0.55, 0.75);
const controls = new OrbitControls(viewCamera, renderer.domElement);
controls.target.set(0.25, 0, 0.08);
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
    const res = await fetch(`${base}assets/openarm/${name}`);
    if (!res.ok) throw new Error(`mesh ${name}: HTTP ${res.status}`);
    meshCache.set(name, new Uint8Array(await res.arrayBuffer()));
  }
  return meshCache.get(name);
}

const app = { mujoco: null, sim: null, view: null, bundle: null, episode: null, playing: false, cursor: 0 };
const camCtx = [0, 1, 2].map((i) => $(`cam${i}`).getContext('2d'));

async function loadScene(seed) {
  const res = await fetch(`${base}openarm_scenes/${seed}.json`);
  if (!res.ok) throw new Error(`scene ${seed}: HTTP ${res.status}`);
  const bundle = await res.json();
  const meshes = new Map(await Promise.all(bundle.mesh_files.map(async (f) => [f, await fetchMesh(f)])));
  app.playing = false;
  // Free the previous scene first (WASM objects are not garbage collected).
  app.view?.dispose();
  app.view = null;
  app.sim?.dispose();
  app.sim = null;
  app.sim = new OpenArmSim(app.mujoco, bundle, meshes);
  app.bundle = bundle;
  $('episode').replaceChildren(...bundle.episodes.map((e, i) => new Option(
    e.disturbance ? `${e.episode_seed}（外乱: ${DISTURB_JA[e.disturbance.kind]}）` : String(e.episode_seed), String(i))));
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
}

function drawPolicyCameras() {
  const { view } = app;
  view.cameras.forEach(({ name }, i) => {
    const pixels = view.renderCamera(renderer, name);
    camCtx[i].putImageData(new ImageData(pixels, view.width, view.height), 0, 0);
  });
}

function updateStatus() {
  const { sim, episode } = app;
  const verdict = sim.succeeded ? '<span class="ok">皿の上に置けています</span>' : '未達';
  const dist = episode.disturbance;
  $('status').innerHTML = [
    `ステップ ${sim.steps}（${(sim.steps / sim.c.control_hz).toFixed(1)} 秒） 判定: ${verdict}`,
    `エキスパートの記録を再生（${app.cursor} / ${episode.expert.actions.length}）`
      + (dist ? `、外乱「${DISTURB_JA[dist.kind]}」は ${dist.step} ステップ目` : ''),
  ].join('\n');
}

function tick() {
  if (!app.playing) return;
  const { actions } = app.episode.expert;
  if (app.cursor >= actions.length) {
    app.playing = false;
    return;
  }
  const dist = app.episode.disturbance;
  if (dist && dist.step === app.cursor) app.sim.applyDisturbance(dist);
  app.sim.step(actions[app.cursor++]);
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
    tick();
  }
  app.view.sync();
  controls.update();
  drawPolicyCameras();
  renderer.setRenderTarget(null);
  renderer.render(app.view.scene, viewCamera);
  updateStatus();
}

// --- Hooks for headless tools (tools/browser_check_openarm.py) -----------------------------

// Replay an episode's recorded actions (and its disturbance) as fast as possible.
function replayNow(index) {
  selectEpisode(index);
  const t0 = performance.now();
  const s = app.sim.replay(app.episode);
  const ms = performance.now() - t0;
  app.view.sync();
  return { success: app.sim.succeeded, successStep: app.sim.successStep, steps: app.sim.steps, ms,
    barPos: s.barPos, pythonSuccess: app.episode.expert.success, pythonSteps: app.episode.expert.steps,
    pythonBarPos: app.episode.expert.final_bar_pos };
}

// Render the three policy cameras for an explicit simulator state.
function renderState(qpos, mocapPos) {
  app.playing = false;
  app.sim.setState(qpos, mocapPos);
  app.view.sync();
  const out = {};
  for (const { name } of app.view.cameras) out[name] = Array.from(app.view.renderCamera(renderer, name));
  return out;
}

// Time per control step of what closed-loop inference will need: physics + the three policy
// cameras read back (+ the free view). Runs `n` steps of the episode's expert actions.
function benchStep(index, n = 120, { mainView = true } = {}) {
  selectEpisode(index);
  const { actions } = app.episode.expert;
  let physics = 0;
  let cams = 0;
  let view = 0;
  for (let t = 0; t < Math.min(n, actions.length); t++) {
    const t0 = performance.now();
    app.sim.step(actions[t]);
    const t1 = performance.now();
    app.view.sync();
    for (const { name } of app.view.cameras) app.view.renderCamera(renderer, name);
    const t2 = performance.now();
    if (mainView) {
      renderer.setRenderTarget(null);
      renderer.render(app.view.scene, viewCamera);
    }
    view += performance.now() - t2;
    physics += t1 - t0;
    cams += t2 - t1;
  }
  const k = Math.min(n, actions.length);
  return { steps: k, physicsMs: physics / k, camerasMs: cams / k, mainViewMs: view / k };
}

async function init() {
  app.mujoco = await loadMujoco();
  const index = await (await fetch(`${base}openarm_scenes/index.json`)).json();
  $('scene').replaceChildren(...index.scenes.map((s) => new Option(String(s), String(s))));
  const first = params.get('scene') ?? String(index.scenes[0]);
  $('scene').value = first;
  await loadScene(Number(first));
  if (params.has('episode')) selectEpisode(Number(params.get('episode')));

  $('scene').addEventListener('change', (e) => loadScene(Number(e.target.value)).catch(showError));
  $('episode').addEventListener('change', (e) => selectEpisode(Number(e.target.value)));
  $('reset').addEventListener('click', () => selectEpisode(Number($('episode').value)));
  $('play').addEventListener('click', () => {
    if (app.cursor >= app.episode.expert.actions.length) selectEpisode(Number($('episode').value));
    app.playing = true;
  });
  requestAnimationFrame(frame);
  return index;
}

window.keiarm = { app, THREE, camera: viewCamera, loadScene, selectEpisode, replayNow, renderState, benchStep,
  ready: init() };
window.keiarm.ready.catch(showError);

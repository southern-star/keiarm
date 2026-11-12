// OpenArm page (milestone 2): the learned policy (or the expert's recording) hands the bar over
// in MuJoCo WASM; the free view, the three policy cameras, and dragging the bar or the plate.
// Hooks for tools/browser_check_openarm.py and tools/browser_eval_openarm.py are on window.keiarm.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import { OpenArmSim } from './sim.js';
import { SceneView } from '../render.js';
import { ChunkRunner, PolicyClient } from '../policy.js';

const $ = (id) => document.getElementById(id);
const base = import.meta.env.BASE_URL;
const params = new URLSearchParams(location.search);
const DISTURB_JA = { move: '棒を移動', knock: '持ち上げた棒を落とす', drop: '受け渡し後に棒を落とす', plate: '皿を移動' };
const STAGE_JA = ['開始', '右手が接近', '右手で把持', '持ち上げ', '両手で保持', '左手だけで保持', '皿の上', '完了'];

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

const app = {
  mujoco: null, sim: null, view: null, bundle: null, episode: null,
  mode: 'model', // 'model' | 'expert'
  playing: false, cursor: 0,
  policy: null, runner: null, policyError: null,
};
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
    e.disturbance ? `${e.episode_seed}（記録に外乱: ${DISTURB_JA[e.disturbance.kind]}）` : String(e.episode_seed), String(i))));
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

// What the policy sees: the three camera images (RGBA bytes) and the 16 joint angles.
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
  const { sim, runner, episode } = app;
  const verdict = sim.succeeded ? '<span class="ok">皿の上に置けています</span>' : STAGE_JA[sim.current];
  const lines = [`ステップ ${sim.steps}（${(sim.steps / sim.c.control_hz).toFixed(1)} 秒） 段階: ${verdict}`];
  if (app.mode === 'model') {
    if (app.policyError) lines.push(`<span class="err">モデルを読み込めません: ${app.policyError}</span>`);
    else if (!app.policy?.info) lines.push('モデルを読み込み中…');
    else {
      const ms = runner.lastMs ? `${runner.lastMs.toFixed(0)} ms` : '—';
      lines.push(`推論: ${app.policy.info.backend.toUpperCase()}、直近 ${ms}、${runner.inferences} 回`);
    }
  } else {
    const dist = episode.disturbance;
    lines.push(`エキスパートの記録を再生（${app.cursor} / ${episode.expert.actions.length}）`
      + (dist ? `、外乱「${DISTURB_JA[dist.kind]}」は ${dist.step} ステップ目` : ''));
  }
  $('status').innerHTML = lines.join('\n');
}

// One 30 Hz control tick in real time.
function tick() {
  if (!app.playing) return;
  if (app.mode === 'expert') {
    const { actions } = app.episode.expert;
    if (app.cursor >= actions.length) {
      app.playing = false;
      return;
    }
    const dist = app.episode.disturbance;
    if (dist && dist.step === app.cursor) app.sim.applyDisturbance(dist);
    app.sim.step(actions[app.cursor++]);
    return;
  }
  if (!app.policy?.info) return;
  const action = app.runner.actRealtime(app.sim.steps, observe);
  if (!action) return; // waiting for the first plan: hold the simulation
  app.sim.step(action);
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

// --- Dragging the bar or the plate -----------------------------------------------------------
// Pointer on the bar (or the plate): move it across the table, at rest, until released. The bar
// keeps its yaw. A disturbance the policy has to recover from, like the evaluation suite's.
const raycaster = new THREE.Raycaster();
const drag = {
  active: false,
  what: null, // 'bar' | 'plate'
  target: null, // [x, y] on the table
  yaw: 0,
  apply() {
    if (!this.target) return;
    if (this.what === 'bar') app.sim.teleportBar(this.target, this.yaw);
    else app.sim.movePlate(this.target);
  },
};

function pointerRay(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, viewCamera);
  return raycaster.ray;
}

function tablePoint(event, height, margin) {
  const hit = pointerRay(event).intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -height), new THREE.Vector3());
  if (!hit) return null;
  const [cx, cy] = app.sim.c.table_center;
  const [hx, hy] = app.sim.c.table_half_size;
  // Keep clear of the pedestal (x < 0.10) and of the table edges.
  return [Math.min(cx + hx - margin, Math.max(0.10 + margin, hit.x)), Math.min(cy + hy - margin, Math.max(cy - hy + margin, hit.y))];
}

function geomMesh(name) {
  const { mujoco, model } = app.sim;
  const id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, name);
  return app.view.geoms.find((g) => g.id === id)?.mesh;
}

function dragHeight() {
  const { scene } = app.sim;
  return drag.what === 'bar' ? scene.bar_size[2] / 2 : 0.002;
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  if (!app.view || event.button !== 0) return;
  pointerRay(event);
  const bar = geomMesh('bar');
  const plate = geomMesh('plate');
  if (bar && raycaster.intersectObject(bar).length) drag.what = 'bar';
  else if (plate && raycaster.intersectObject(plate).length) drag.what = 'plate';
  else return;
  const s = app.sim.state();
  // Bar yaw from its long axis (column 1 of the rotation): yaw 0 = along +y.
  drag.yaw = Math.atan2(-s.barMat[1], s.barMat[4]);
  drag.active = true;
  const margin = drag.what === 'bar' ? app.sim.scene.bar_size[1] / 2 : app.sim.scene.plate_radius;
  drag.margin = margin;
  drag.target = tablePoint(event, dragHeight(), margin);
  controls.enabled = false;
  renderer.domElement.setPointerCapture(event.pointerId);
}, { capture: true });
renderer.domElement.addEventListener('pointermove', (event) => {
  if (drag.active) drag.target = tablePoint(event, dragHeight(), drag.margin) ?? drag.target;
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

// --- Hooks for headless tools --------------------------------------------------------------

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

// Time per control step: physics + the three policy cameras read back (+ the free view).
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

// Closed-loop model episode in simulation time (the simulation waits for each inference, as in
// the Python harness). `disturb`: one of the suite's kinds, armed from the episode's plans and
// fired by the same stage rule as keiarm/openarm/eval.py.
async function runModelEpisode(index, { maxSteps, disturb = 'none' } = {}) {
  await policyReady;
  selectEpisode(index);
  const { sim, runner, episode } = app;
  sim.armDisturbance(disturb === 'none' ? null : episode.planned_disturbances[disturb]);
  const limit = maxSteps ?? sim.c.max_episode_steps;
  while (sim.steps < limit && !sim.succeeded) {
    sim.disturbIfDue();
    sim.step(await runner.actBlocking(sim.steps, observe));
  }
  app.view.sync();
  return { success: sim.succeeded, steps: sim.steps, stage: sim.c.stages[sim.stage],
    disturbStep: sim.disturbance?.step ?? null, inferences: runner.inferences,
    meanInferMs: runner.inferences ? runner.totalMs / runner.inferences : 0 };
}

let policyReady;
async function initPolicy() {
  const meta = await (await fetch(`${base}models/openarm.json`)).json();
  app.policy = new PolicyClient();
  app.runner = new ChunkRunner(app.policy, { chunk: meta.chunk, execute: app.sim.c.policy_execute_steps,
    dim: meta.action_dim });
  return app.policy.init({
    modelUrl: new URL(`${base}models/openarm.onnx`, location.href).href,
    ortPath: new URL(`${base}ort/`, location.href).href,
    prefer: params.get('backend') ?? 'auto',
    cameras: meta.cameras.length, height: meta.image_height, width: meta.image_width, chunk: meta.chunk,
    dim: meta.action_dim,
  });
}

async function init() {
  app.mujoco = await loadMujoco();
  const index = await (await fetch(`${base}openarm_scenes/index.json`)).json();
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

  // The model (47 MB) loads in the background; the scene is usable meanwhile.
  policyReady = initPolicy();
  policyReady.catch((err) => {
    app.policyError = err.message;
    console.error(err);
  });
  return index;
}

window.keiarm = { app, THREE, camera: viewCamera, loadScene, selectEpisode, setMode, replayNow, renderState, benchStep,
  runModelEpisode, policyReady: () => policyReady, ready: init() };
window.keiarm.ready.catch(showError);

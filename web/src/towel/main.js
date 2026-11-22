// Towel page (milestone 3): the half fold of a towel by both arms.
// - live: MuJoCo WASM runs the physics with a coarse towel (N = N_web). The learned policy
//   (onnxruntime-web, Phase 7) folds it, or the expert's recorded actions are replayed (Phase 4).
//   In the policy's run the towel can be dragged: it lies flat again where it is dropped (inside
//   the spawn region) and the policy folds it again.
// - record: playback of episodes recorded in Python with the training mesh (N = 16), no physics
// Hooks for tools/browser_check_towel.py are on window.keiarm.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import { TowelSim } from './sim.js';
import { TowelView } from './view.js';
import { ChunkRunner, PolicyClient } from '../policy.js';

const $ = (id) => document.getElementById(id);
const base = import.meta.env.BASE_URL;
const params = new URLSearchParams(location.search);
const DISTURB_JA = { shift: '布を移動', slip: '片手が滑る', drop: '両手が滑る', nudge: '角をはじく' };
const STAGE_JA = ['開始', '接近', '片手で把持', '両手で把持', '持ち上げ', '運搬', '置いた', '離した', '完了'];

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
viewCamera.position.set(0.78, -0.32, 0.5);
const controls = new OrbitControls(viewCamera, renderer.domElement);
controls.target.set(0.2, 0, 0.05);
controls.update();

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  viewCamera.aspect = w / h;
  // Keep the towel clear of the panel on the left: move the picture's centre to the right.
  const shift = w > 900 ? Math.min(270, 0.25 * w) : 0;
  if (shift) viewCamera.setViewOffset(w, h, -shift, 0, w, h);
  else viewCamera.clearViewOffset();
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

async function fetchJson(path) {
  const res = await fetch(`${base}${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

function decode(b64, Type) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

// Physics runs in a worker by default (physics_worker.js); ?physics=main runs it on this thread.
const useWorker = params.get('physics') !== 'main';
let worker = null;
let workerLoaded = null; // resolves when the worker has the current scene
let framesIn = 0; // worker frames received since the last drawn frame

function startWorker() {
  worker = new Worker(new URL('./physics_worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'loaded') workerLoaded.resolve();
    else if (m.type === 'error') showError(new Error(`physics worker: ${m.message}`));
    else if (m.runId !== undefined && m.runId !== app.runId) return; // a stopped run's late message
    else if (m.type === 'frame' && app.sim && !app.record) {
      const d = app.sim.data;
      d.geom_xpos.set(m.geom_xpos);
      d.geom_xmat.set(m.geom_xmat);
      d.cam_xpos.set(m.cam_xpos);
      d.cam_xmat.set(m.cam_xmat);
      d.flexvert_xpos.set(m.flexvert_xpos, 3 * app.sim.vadr);
      app.remote = m.status;
      app.cursor = m.cursor;
      framesIn += 1;
      if (app.liveMode === 'model' && app.playing) maybeInfer();
    } else if (m.type === 'done') {
      app.remote = m.status;
      app.playing = false;
      app.workerDone?.(m);
    }
  };
}

const app = {
  mujoco: null, sim: null, view: null, bundle: null, episode: null,
  mode: 'live', // 'live' | 'record'
  liveMode: 'model', // in live mode: 'model' (the learned policy) | 'expert' (its recorded actions)
  playing: false, cursor: 0,
  runId: 0, workerRun: false, // the worker's current run (frames of older runs are ignored)
  policy: null, runner: null, policyError: null, meta: null,
  record: null, // { data, qpos, verts, ... } while in record mode
  sceneIndex: null, recordIndex: null,
};
const camCtx = [0, 1, 2].map((i) => $(`cam${i}`).getContext('2d'));

function disposeScene() {
  app.playing = false;
  app.view?.dispose();
  app.view = null;
  app.sim?.dispose();
  app.sim = null;
}

async function makeSim(bundle, { physics = false } = {}) {
  const meshes = new Map(await Promise.all(bundle.mesh_files.map(async (f) => [f, await fetchMesh(f)])));
  disposeScene();
  if (physics && worker) {
    worker.postMessage({ type: 'stop' });
    let resolve;
    workerLoaded = new Promise((r) => { resolve = r; });
    workerLoaded.resolve = resolve;
    worker.postMessage({ type: 'load', bundle, meshes: [...meshes] });
  }
  app.sim = new TowelSim(app.mujoco, bundle, meshes);
  app.view = new TowelView(app.sim);
}

// --- live (physics in the browser) ---------------------------------------------------------

async function loadScene(name) {
  app.loading = true; // play() waits for the new scene (a run started meanwhile would be dropped)
  try {
    await loadSceneNow(name);
  } finally {
    app.loading = false;
  }
}

async function loadSceneNow(name) {
  const bundle = await fetchJson(`towel_scenes/${name}.json`);
  app.record = null;
  await makeSim(bundle, { physics: true });
  if (workerLoaded) await workerLoaded;
  app.bundle = bundle;
  $('episode').replaceChildren(...bundle.episodes.map((e, i) => new Option(
    e.disturbance ? `${e.episode_seed}（外乱: ${DISTURB_JA[e.disturbance.kind]}）` : String(e.episode_seed), String(i))));
  selectEpisode(0);
}

function selectEpisode(index) {
  app.episode = app.bundle.episodes[index];
  $('episode').value = String(index);
  worker?.postMessage({ type: 'stop' });
  app.runId += 1;
  app.workerRun = false;
  remote.reset();
  app.runner?.reset();
  drag.centre = null;
  app.sim.reset(app.episode);
  app.view.sync();
  app.cursor = 0;
  app.playing = false;
  app.remote = null;
  camsStale = true;
}

function play() {
  if (app.loading) return;
  app.playing = true;
  if (app.record) return;
  if (app.liveMode === 'model') {
    remote.reset();
    app.runner?.reset();
    if (worker) {
      app.workerRun = true;
      worker.postMessage({ type: 'playModel', episode: app.episode, chunk: app.meta.chunk, runId: app.runId });
    }
  } else if (worker) {
    app.workerRun = true;
    worker.postMessage({ type: 'play', episode: app.episode, runId: app.runId });
  }
}

// --- the learned policy ------------------------------------------------------------------------

// What the policy sees: the three camera images (RGBA bytes) and the 16 joint angles. With the
// physics in the worker, the poses and joints are those of the newest frame it posted.
function observe() {
  app.view.sync();
  return {
    images: app.view.cameras.map(({ name }) => app.view.renderCamera(renderer, name)),
    joints: app.workerRun ? app.remote.joints : app.sim.joints(),
  };
}

// The worker's run: request a plan when the current one has been followed for
// policy_execute_steps (ChunkRunner.actRealtime, with the plan held by the worker).
const remote = {
  reset() {
    this.planStep = null;
    this.inFlight = false;
    this.generation = (this.generation ?? 0) + 1;
    this.inferences = 0;
    this.lastMs = 0;
    this.totalMs = 0;
  },
};
remote.reset();

function maybeInfer() {
  if (!app.workerRun || !app.policy?.info || remote.inFlight || !app.remote) return;
  const step = app.remote.steps;
  if (remote.planStep !== null && step - remote.planStep < app.sim.c.policy_execute_steps) return;
  remote.inFlight = true;
  const { generation } = remote;
  const obs = observe();
  app.policy.infer(obs.images, obs.joints).then(({ actions, ms }) => {
    if (generation !== remote.generation) return;
    remote.planStep = step;
    remote.lastMs = ms;
    remote.inferences += 1;
    remote.totalMs += ms;
    worker.postMessage({ type: 'plan', planStep: step, actions }, [actions.buffer]);
  }).catch(showError).finally(() => {
    if (generation === remote.generation) remote.inFlight = false;
  });
}

let policyReady;
async function initPolicy() {
  app.meta = await fetchJson('models/towel.json');
  app.policy = new PolicyClient();
  app.runner = new ChunkRunner(app.policy, { chunk: app.meta.chunk, execute: 10, dim: app.meta.action_dim });
  return app.policy.init({
    modelUrl: new URL(`${base}models/towel.onnx`, location.href).href,
    ortPath: new URL(`${base}ort/`, location.href).href,
    prefer: params.get('backend') ?? 'auto',
    cameras: app.meta.cameras.length, height: app.meta.image_height, width: app.meta.image_width,
    chunk: app.meta.chunk, dim: app.meta.action_dim,
  });
}

// --- dragging the towel (the policy's run) -----------------------------------------------------
// Pointer on the towel: the towel lies flat under the pointer (keeping its offset from the grab
// point) until released, clamped to the spawn region, at the episode's yaw. Both hands let go.

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const tablePlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const drag = { active: false, offset: null, target: null, centre: null };

function pointerRay(event) {
  const r = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - r.left) / r.width) * 2 - 1, -((event.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(pointer, viewCamera);
}

function clampCentre([x, y]) {
  const c = app.sim.c;
  const [cx, cy] = c.towel_center;
  const [jx, jy] = c.towel_center_jitter;
  return [Math.min(cx + jx, Math.max(cx - jx, x)), Math.min(cy + jy, Math.max(cy - jy, y))];
}

function dragTarget(event) {
  pointerRay(event);
  const p = raycaster.ray.intersectPlane(tablePlane, new THREE.Vector3());
  if (!p) return null;
  const xy = clampCentre([p.x - drag.offset[0], p.y - drag.offset[1]]);
  return { xy, yaw: app.episode.towel_yaw };
}

function sendDrag(target) {
  drag.target = target;
  if (target) drag.centre = target.xy;
  if (worker) worker.postMessage({ type: 'drag', target });
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  if (app.record || app.liveMode !== 'model' || !app.playing || !app.view) return;
  pointerRay(event);
  // The vertices move every step; three.js caches the bounding sphere it tests rays against
  // (computed once, possibly before the first pose arrived), so refresh it here.
  app.view.towelGeom.computeBoundingSphere();
  const hit = raycaster.intersectObjects(app.view.towelMeshes)[0];
  if (!hit) return;
  const centre = drag.centre ?? app.episode.towel_xy;
  drag.offset = [hit.point.x - centre[0], hit.point.y - centre[1]];
  drag.active = true;
  controls.enabled = false;
  renderer.domElement.setPointerCapture(event.pointerId);
  sendDrag(dragTarget(event));
});
renderer.domElement.addEventListener('pointermove', (event) => {
  if (drag.active) sendDrag(dragTarget(event) ?? drag.target);
});
const endDrag = () => {
  if (!drag.active) return;
  drag.active = false;
  controls.enabled = true;
  sendDrag(null);
};
renderer.domElement.addEventListener('pointerup', endDrag);
renderer.domElement.addEventListener('pointercancel', endDrag);

// --- record playback (N = 16, no physics) ----------------------------------------------------

async function loadRecord(name) {
  const rec = await fetchJson(`towel_records/${name}.json`);
  await makeSim(rec); // the record carries the N = 16 model; it is only posed, never stepped
  const { model: m } = app.sim;
  const T = app.mujoco.mjtObj;
  app.record = {
    ...rec,
    qadr: rec.robot_joints.map((j) => m.jnt_qposadr[app.mujoco.mj_name2id(m, T.mjOBJ_JOINT.value, j)]),
    qpos: decode(rec.qpos, Float32Array),
    verts: decode(rec.verts, Int16Array),
    speed: decode(rec.speed, Uint16Array),
    touch: decode(rec.touch, Uint8Array),
    stage: decode(rec.stage, Uint8Array),
  };
  app.bundle = null;
  app.cursor = 0;
  app.playing = false;
  showFrame(0);
}

function frameVerts(k) {
  const r = app.record;
  const n = 3 * app.sim.nvert;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = r.verts[k * n + i] * r.vert_scale;
  return out;
}

function showFrame(k) {
  const r = app.record;
  const { mujoco, model: m, data: d } = app.sim;
  const nj = r.qadr.length;
  r.qadr.forEach((a, j) => { d.qpos[a] = r.qpos[k * nj + j]; });
  mujoco.mj_kinematics(m, d);
  mujoco.mj_camlight(m, d);
  app.view.sync();
  app.view.setTowel(frameVerts(k));
  camsStale = true;
}

// The success check of keiarm/towel/task.py on a recorded frame (held counts, finger touch and
// the largest vertex speed are recorded with it).
function recordCheck(k) {
  const r = app.record;
  const v = frameVerts(k);
  const s = {
    verts: v, maxSpeed: r.speed[k] / 1000,
    held: { left: Array(r.held[k][0]).fill(0), right: Array(r.held[k][1]).fill(0) },
    contacts: { touch: { left: Boolean(r.touch[k] & 1), right: Boolean(r.touch[k] & 2) }, arms: false },
  };
  return app.sim.checkFold(s);
}

// Success as the Python Monitor counts it: the check held for success_hold_steps frames.
function recordSuccess() {
  const r = app.record;
  let run = 0;
  for (let k = 1; k < r.frames; k++) { // frame 0 is the initial state, frame k follows control step k
    run = recordCheck(k).success ? run + 1 : 0;
    if (run >= app.sim.c.success_hold_steps) return { success: true, step: k };
  }
  return { success: false, step: null };
}

// --- UI ----------------------------------------------------------------------------------------

function setMode(mode) {
  app.mode = mode;
  $('mode-live').setAttribute('aria-pressed', String(mode === 'live'));
  $('mode-record').setAttribute('aria-pressed', String(mode === 'record'));
  $('scene-label').hidden = mode !== 'live';
  $('episode-label').hidden = mode !== 'live';
  $('record-label').hidden = mode !== 'record';
  $('live-label').hidden = mode !== 'live';
  if (mode === 'live') setHint();
  else $('hint').textContent = '学習と同じ布（16 × 16 頂点）で Python が記録したエピソードを再生します（ブラウザでは物理を回しません）。';
  const load = mode === 'live' ? loadScene($('scene').value) : loadRecord($('record').value);
  return load.catch(showError);
}

function drawPolicyCameras() {
  const { view } = app;
  view.cameras.forEach(({ name }, i) => {
    const pixels = view.renderCamera(renderer, name);
    camCtx[i].putImageData(new ImageData(pixels, view.width, view.height), 0, 0);
  });
}

function updateStatus() {
  const { sim } = app;
  if (app.record) {
    const r = app.record;
    const k = Math.min(app.cursor, r.frames - 1);
    const stage = STAGE_JA[r.stage[k]];
    const done = r.stage[k] === STAGE_JA.length - 1 ? '<span class="ok">たためました</span>' : stage;
    $('status').innerHTML = `記録 ${r.label}\nステップ ${k + 1} / ${r.frames}（${((k + 1) / sim.c.control_hz).toFixed(1)} 秒） 段階: ${done}`;
    return;
  }
  const st = app.remote ?? { steps: sim.steps, current: sim.current, succeeded: sim.succeeded };
  const verdict = st.succeeded ? '<span class="ok">たためました</span>' : STAGE_JA[st.current];
  const rt = stats.realtime ? `、実時間の ${stats.realtime.toFixed(2)} 倍` : '';
  const head = `ステップ ${st.steps}（${(st.steps / sim.c.control_hz).toFixed(1)} 秒） 段階: ${verdict}\n`;
  if (app.liveMode === 'model') {
    let line;
    if (app.policyError) line = `<span class="err">モデルを読み込めません: ${app.policyError}</span>`;
    else if (!app.policy?.info) line = 'モデルを読み込み中…';
    else {
      const r = worker ? remote : app.runner;
      line = `推論: ${app.policy.info.backend.toUpperCase()}、直近 ${r.lastMs ? `${r.lastMs.toFixed(0)} ms` : '—'}、${r.inferences} 回`;
    }
    $('status').innerHTML = head + line + rt;
    return;
  }
  const dist = app.episode.disturbance;
  $('status').innerHTML = head
    + `エキスパートの記録を再生（${app.cursor} / ${app.episode.expert.actions.length}）`
    + (dist ? `、外乱「${DISTURB_JA[dist.kind]}」は ${dist.step} ステップ目` : '') + rt;
}

// One 30 Hz control tick.
function tick() {
  if (!app.playing) return;
  if (app.record) {
    if (app.cursor >= app.record.frames - 1) { app.playing = false; return; }
    app.cursor += 1;
    showFrame(app.cursor);
    return;
  }
  if (worker) return; // the worker steps the physics
  if (app.liveMode === 'model') {
    if (!app.policy?.info) return;
    if (drag.target) app.sim.dragTowel(drag.target.xy, drag.target.yaw);
    const action = app.runner.actRealtime(app.sim.steps, observe);
    if (action) app.sim.step(action); // until the first plan arrives the simulation waits
    return;
  }
  const { actions } = app.episode.expert;
  if (app.cursor >= actions.length) {
    app.playing = false;
    stats.ticks = 0;
    stats.wallMs = 0;
    return;
  }
  const dist = app.episode.disturbance;
  if (dist && dist.step === app.cursor) app.sim.applyDisturbance(dist);
  app.sim.step(actions[app.cursor++]);
}

// Real time: at most MAX_TICKS control steps per frame; if the physics is slower than real
// time the playback slows down (it never skips physics).
const MAX_TICKS = 3;
const stats = { realtime: null, ticks: 0, wallMs: 0 };
let last = performance.now();
let accumulator = 0;
let camsStale = true; // set when the scene changes outside a tick
function frame(now) {
  requestAnimationFrame(frame);
  if (!app.sim || !app.view) return;
  const period = 1000 / app.sim.c.control_hz;
  const dt = now - last;
  last = now;
  accumulator = Math.min(accumulator + dt, MAX_TICKS * period);
  let ticks = 0;
  const t0 = performance.now();
  while (accumulator >= period && ticks < MAX_TICKS) {
    accumulator -= period;
    tick();
    ticks += 1;
  }
  if (worker && !app.record) {
    ticks = framesIn ? 1 : 0; // poses arrived from the worker
    stats.ticks += framesIn ? app.cursor - (stats.lastCursor ?? 0) : 0;
    if (framesIn) stats.lastCursor = app.cursor;
    framesIn = 0;
  }
  if (app.playing && !app.record) {
    if (!worker) stats.ticks += ticks;
    stats.wallMs += dt;
    if (stats.wallMs > 1000) {
      stats.realtime = (stats.ticks * period) / stats.wallMs;
      stats.ticks = 0;
      stats.wallMs = 0;
    }
  }
  if (!app.record) app.view.sync();
  controls.update();
  // The policy cameras change only with a control step; reading them back stalls the GPU, so
  // they are redrawn only on frames that advanced the simulation (or showed a new frame).
  if (ticks > 0 || camsStale) {
    drawPolicyCameras();
    camsStale = false;
  }
  renderer.setRenderTarget(null);
  renderer.render(app.view.scene, viewCamera);
  updateStatus();
  stats.frameMs = performance.now() - t0;
}

// --- Hooks for headless tools --------------------------------------------------------------

function replayNow(index) {
  selectEpisode(index);
  const t0 = performance.now();
  const s = app.sim.replay(app.episode);
  const ms = performance.now() - t0;
  app.view.sync();
  let vertErr = 0;
  app.episode.expert.final_verts.forEach((p, k) => {
    vertErr = Math.max(vertErr, Math.hypot(p[0] - s.verts[3 * k], p[1] - s.verts[3 * k + 1], p[2] - s.verts[3 * k + 2]));
  });
  return { success: app.sim.succeeded || app.sim.successStep !== null, successStep: app.sim.successStep,
    steps: app.sim.steps, ms, grabs: app.sim.grabs, vertErr, pythonSuccess: app.episode.expert.success,
    pythonSteps: app.episode.expert.steps, pythonGrabs: app.episode.expert.grabs };
}

// Render the three policy cameras for an explicit simulator state (qpos, incl. the towel's
// vertex joints).
function renderState(qpos) {
  app.playing = false;
  app.sim.data.qpos.set(qpos);
  app.mujoco.mj_forward(app.sim.model, app.sim.data);
  app.view.sync();
  const out = {};
  for (const { name } of app.view.cameras) out[name] = Array.from(app.view.renderCamera(renderer, name));
  return out;
}

// Time per control step: physics + the three policy cameras read back (+ the free view).
function benchStep(index, n = 150, { mainView = true } = {}) {
  selectEpisode(index);
  const { actions } = app.episode.expert;
  let physics = 0;
  let cams = 0;
  let view = 0;
  const k = Math.min(n, actions.length);
  for (let t = 0; t < k; t++) {
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
  return { steps: k, physicsMs: physics / k, camerasMs: cams / k, mainViewMs: view / k };
}

// Play an episode in real time as a viewer would (requestAnimationFrame, the three policy
// cameras drawn every frame) for `seconds`; returns frames per second and the simulated time.
function realtimeRun(index, seconds = 8) {
  app.liveMode = 'expert';
  selectEpisode(index);
  stats.lastCursor = 0;
  play();
  return new Promise((resolve) => {
    const start = performance.now();
    let frames = 0;
    const count = (now) => {
      frames += 1;
      if (now - start < seconds * 1000 && app.playing) requestAnimationFrame(count);
      else {
        const wall = (now - start) / 1000;
        const steps = worker ? (app.remote?.steps ?? 0) : app.sim.steps;
        resolve({ fps: frames / wall, wall, simulated: steps / app.sim.c.control_hz,
          realtime: steps / app.sim.c.control_hz / wall, steps, worker: Boolean(worker),
          meanStepMs: worker ? app.remote?.meanStepMs : null });
      }
    };
    requestAnimationFrame(count);
  });
}

// Closed-loop model episode in simulation time on this thread (the simulation waits for each
// inference, as in the Python harness). `disturb`: one of the suite's kinds, armed from the
// episode's plans and fired by the same stage rule as keiarm/towel/eval.py. Also times physics,
// the policy cameras and inference per control step (the cost if one thread did all three).
async function runModelEpisode(index, { maxSteps, disturb = 'none' } = {}) {
  await policyReady;
  selectEpisode(index);
  const { sim, runner, episode } = app;
  sim.armDisturbance(disturb === 'none' ? null : episode.planned_disturbances[disturb]);
  const limit = maxSteps ?? sim.c.max_episode_steps;
  let physicsMs = 0;
  let camerasMs = 0;
  const timedObserve = () => {
    const t0 = performance.now();
    const o = observe();
    camerasMs += performance.now() - t0;
    return o;
  };
  while (sim.steps < limit && !sim.succeeded) {
    sim.disturbIfDue();
    const action = await runner.actBlocking(sim.steps, timedObserve);
    const t0 = performance.now();
    sim.step(action);
    physicsMs += performance.now() - t0;
  }
  app.view.sync();
  return { success: sim.succeeded, steps: sim.steps, stage: sim.c.stages[sim.stage],
    disturbStep: sim.disturbance?.step ?? null, inferences: runner.inferences,
    meanInferMs: runner.inferences ? runner.totalMs / runner.inferences : 0,
    physicsMsPerStep: physicsMs / sim.steps, camerasMsPerStep: camerasMs / sim.steps,
    inferMsPerStep: runner.totalMs / sim.steps };
}

// The policy's run in real time as a viewer sees it (physics in the worker, inference in its
// worker, drawing here) for `seconds`: simulated seconds per wall-clock second.
function realtimeModelRun(index, seconds = 10) {
  return policyReady.then(() => {
    app.liveMode = 'model';
    selectEpisode(index);
    play();
    return new Promise((resolve) => {
      let start = null;
      let steps0 = 0;
      let frames = 0;
      const count = (now) => {
        const steps = worker ? (app.remote?.steps ?? 0) : app.sim.steps;
        if (start === null && steps > 0) { start = now; steps0 = steps; } // from the first plan on
        if (start !== null) frames += 1;
        if (start === null || now - start < seconds * 1000) requestAnimationFrame(count);
        else {
          const wall = (now - start) / 1000;
          const simulated = (steps - steps0) / app.sim.c.control_hz;
          resolve({ fps: frames / wall, wall, simulated, realtime: simulated / wall, steps,
            succeeded: worker ? app.remote?.succeeded : app.sim.succeeded, worker: Boolean(worker),
            meanStepMs: worker ? app.remote?.meanStepMs : null, inferences: worker ? remote.inferences : app.runner.inferences,
            meanInferMs: worker ? remote.totalMs / Math.max(1, remote.inferences)
              : app.runner.totalMs / Math.max(1, app.runner.inferences) });
        }
      };
      requestAnimationFrame(count);
    });
  });
}

function setLiveMode(mode) {
  app.liveMode = mode;
  $('live-model').setAttribute('aria-pressed', String(mode === 'model'));
  $('live-expert').setAttribute('aria-pressed', String(mode === 'expert'));
  if (app.bundle) selectEpisode(Number($('episode').value));
  setHint();
}

function setHint() {
  const n = app.sceneIndex?.default_n ?? '?';
  if (app.mode !== 'live') return;
  $('hint').textContent = `ブラウザ版は粗い布（${n} × ${n} 頂点）で物理を回します。`
    + (app.liveMode === 'model'
      ? '学習した方策がカメラ 3 台の画像からたたみます。動作中に布をドラッグすると、広げ直してまたたたみます。'
      : 'Python で記録したエキスパートの行動列を再生します。')
    + '視点は背景をドラッグ。';
}

function recordFrames(name) {
  return loadRecord(name).then(() => {
    const r = app.record;
    const res = recordSuccess();
    return { ...res, python: r.python, frames: r.frames };
  });
}

async function init() {
  app.mujoco = await loadMujoco();
  if (useWorker) startWorker();
  app.sceneIndex = await fetchJson('towel_scenes/index.json');
  const n = params.get('n') ?? String(app.sceneIndex.default_n ?? app.sceneIndex.meshes[0]);
  const names = app.sceneIndex.scenes.map((s) => `${s}_n${n}`);
  $('scene').replaceChildren(...names.map((s) => new Option(s.replace(`_n${n}`, ''), s)));
  $('scene').value = params.has('scene') ? `${params.get('scene')}_n${n}` : names[0];
  try {
    app.recordIndex = await fetchJson('towel_records/index.json');
    $('record').replaceChildren(...app.recordIndex.records.map((r) => new Option(r.label, r.name)));
  } catch (err) {
    console.warn('no towel records', err);
    app.recordIndex = { records: [] };
  }
  $('mode-record').disabled = !app.recordIndex.records.length;
  if (params.get('live') === 'expert') app.liveMode = 'expert';
  await setMode(params.get('mode') === 'record' && app.recordIndex.records.length ? 'record' : 'live');
  if (params.has('episode') && app.bundle) selectEpisode(Number(params.get('episode')));
  for (const k of ['model', 'expert']) $(`live-${k}`).setAttribute('aria-pressed', String(app.liveMode === k));

  $('scene').addEventListener('change', () => loadScene($('scene').value).catch(showError));
  $('record').addEventListener('change', () => loadRecord($('record').value).catch(showError));
  $('episode').addEventListener('change', (e) => selectEpisode(Number(e.target.value)));
  $('reset').addEventListener('click', () => (app.record ? (app.cursor = 0, showFrame(0), app.playing = false)
    : selectEpisode(Number($('episode').value))));
  $('mode-live').addEventListener('click', () => setMode('live'));
  $('live-model').addEventListener('click', () => setLiveMode('model'));
  $('live-expert').addEventListener('click', () => setLiveMode('expert'));
  $('mode-record').addEventListener('click', () => setMode('record'));
  $('play').addEventListener('click', () => {
    if (app.record && app.cursor >= app.record.frames - 1) { app.cursor = 0; showFrame(0); }
    if (!app.record && (app.liveMode === 'model' || app.cursor >= app.episode.expert.actions.length || app.remote)) {
      selectEpisode(Number($('episode').value));
    }
    stats.lastCursor = 0;
    play();
  });
  requestAnimationFrame(frame);

  // The model (47 MB) loads in the background; the expert's replay is usable meanwhile.
  policyReady = initPolicy();
  policyReady.catch((err) => {
    app.policyError = err.message;
    console.error(err);
  });
  return app.sceneIndex;
}

window.keiarm = { app, THREE, camera: viewCamera, loadScene, loadRecord, selectEpisode, setMode, setLiveMode, replayNow,
  renderState, benchStep, realtimeRun, recordFrames, runModelEpisode, realtimeModelRun, drag,
  policyReady: () => policyReady, ready: init() };
window.keiarm.ready.catch(showError);

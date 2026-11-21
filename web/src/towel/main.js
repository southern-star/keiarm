// Towel page (milestone 3): the half fold of a towel by both arms.
// - live: MuJoCo WASM runs the physics with a coarse towel (N = N_web) and replays the expert's
//   recorded actions (Phase 4; the learned policy comes in Phase 7)
// - record: playback of episodes recorded in Python with the training mesh (N = 16), no physics
// Hooks for tools/browser_check_towel.py are on window.keiarm.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import { TowelSim } from './sim.js';
import { TowelView } from './view.js';

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
  playing: false, cursor: 0,
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
  app.sim.reset(app.episode);
  app.view.sync();
  app.cursor = 0;
  app.playing = false;
  app.remote = null;
  camsStale = true;
}

function play() {
  app.playing = true;
  if (worker && !app.record) worker.postMessage({ type: 'play', episode: app.episode });
}

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
  $('hint').textContent = mode === 'live'
    ? `ブラウザ版は粗い布（${app.sceneIndex?.default_n ?? '?'} × ${app.sceneIndex?.default_n ?? '?'} 頂点）で物理を回し、`
      + 'Python で記録したエキスパートの行動列を再生します。視点は背景をドラッグ。'
    : '学習と同じ布（16 × 16 頂点）で Python が記録したエピソードを再生します（ブラウザでは物理を回しません）。';
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
  const dist = app.episode.disturbance;
  const rt = stats.realtime ? `、物理は実時間の ${stats.realtime.toFixed(2)} 倍` : '';
  $('status').innerHTML = `ステップ ${st.steps}（${(st.steps / sim.c.control_hz).toFixed(1)} 秒） 段階: ${verdict}\n`
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
  await setMode(params.get('mode') === 'record' && app.recordIndex.records.length ? 'record' : 'live');
  if (params.has('episode') && app.bundle) selectEpisode(Number(params.get('episode')));

  $('scene').addEventListener('change', () => loadScene($('scene').value).catch(showError));
  $('record').addEventListener('change', () => loadRecord($('record').value).catch(showError));
  $('episode').addEventListener('change', (e) => selectEpisode(Number(e.target.value)));
  $('reset').addEventListener('click', () => (app.record ? (app.cursor = 0, showFrame(0), app.playing = false)
    : selectEpisode(Number($('episode').value))));
  $('mode-live').addEventListener('click', () => setMode('live'));
  $('mode-record').addEventListener('click', () => setMode('record'));
  $('play').addEventListener('click', () => {
    if (app.record && app.cursor >= app.record.frames - 1) { app.cursor = 0; showFrame(0); }
    if (!app.record && (app.cursor >= app.episode.expert.actions.length || app.remote)) {
      selectEpisode(Number($('episode').value));
    }
    stats.lastCursor = 0;
    play();
  });
  requestAnimationFrame(frame);
  return app.sceneIndex;
}

window.keiarm = { app, THREE, camera: viewCamera, loadScene, loadRecord, selectEpisode, setMode, replayNow,
  renderState, benchStep, realtimeRun, recordFrames, ready: init() };
window.keiarm.ready.catch(showError);

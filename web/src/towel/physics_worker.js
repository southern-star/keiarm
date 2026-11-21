// Physics in a worker (milestone 3, Phase 4): the towel scene's MuJoCo WASM runs here at real
// time, so that drawing (main thread) and physics (this thread) overlap. At N = 10 a control
// step takes ~25 ms and drawing the free view and the three policy cameras ~3 ms; on one thread
// the page reached only 0.75-0.8 x real time at 25 fps.
// After every control step the poses the view needs are posted back (geom and camera poses,
// towel vertices) with the task status. Messages:
//   in:  {type: 'load', bundle, meshes}  {type: 'play', episode}  {type: 'stop'}
//   out: {type: 'loaded'}  {type: 'frame', ...poses, status}  {type: 'done', status}  {type: 'error', message}
import loadMujoco from '@mujoco/mujoco';
import { TowelSim } from './sim.js';

let mujoco = null;
let sim = null;
let run = null; // { episode, cursor, t0, stopped }

function status() {
  const ms = run?.stepMs ?? [];
  return { steps: sim.steps, current: sim.current, stage: sim.stage, succeeded: sim.succeeded,
    successStep: sim.successStep, meanStepMs: ms.length ? ms.reduce((a, b) => a + b, 0) / ms.length : null };
}

function post() {
  const d = sim.data;
  const v0 = 3 * sim.vadr;
  const frame = {
    type: 'frame',
    geom_xpos: d.geom_xpos.slice(), geom_xmat: d.geom_xmat.slice(),
    cam_xpos: d.cam_xpos.slice(), cam_xmat: d.cam_xmat.slice(),
    flexvert_xpos: d.flexvert_xpos.slice(v0, v0 + 3 * sim.nvert),
    cursor: run.cursor, status: status(),
  };
  self.postMessage(frame, [frame.geom_xpos.buffer, frame.geom_xmat.buffer, frame.cam_xpos.buffer,
    frame.cam_xmat.buffer, frame.flexvert_xpos.buffer]);
}

// Advance to wall-clock time in 30 Hz control steps (at most a few per call, then yield so that
// 'stop' messages get through). Behind real time, it runs flat out and never skips physics.
function loop() {
  if (!run || run.stopped) return;
  const { actions } = run.episode.expert;
  const period = 1000 / sim.c.control_hz;
  const due = Math.floor((performance.now() - run.t0) / period);
  let n = 0;
  while (run.cursor < due && run.cursor < actions.length && n < 3) {
    const dist = run.episode.disturbance;
    if (dist && dist.step === run.cursor) sim.applyDisturbance(dist);
    const s0 = performance.now();
    sim.step(actions[run.cursor++]);
    run.stepMs.push(performance.now() - s0);
    n += 1;
  }
  if (n) post();
  if (run.cursor >= actions.length) {
    self.postMessage({ type: 'done', status: status(), wallMs: performance.now() - run.t0 });
    run = null;
    return;
  }
  const wait = run.t0 + (run.cursor + 1) * period - performance.now();
  // Behind real time: continue at once through a message (setTimeout(0) is clamped to 4 ms
  // when nested, ~10% of a step); ahead: sleep until the next step is due.
  if (wait <= 0) yieldNow();
  else setTimeout(loop, wait);
}

const channel = new MessageChannel();
channel.port1.onmessage = () => loop();
function yieldNow() {
  channel.port2.postMessage(null);
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') {
      mujoco ??= await loadMujoco();
      sim?.dispose();
      sim = new TowelSim(mujoco, msg.bundle, new Map(msg.meshes));
      self.postMessage({ type: 'loaded' });
    } else if (msg.type === 'play') {
      sim.reset(msg.episode);
      run = { episode: msg.episode, cursor: 0, t0: performance.now(), stopped: false, stepMs: [] };
      loop();
    } else if (msg.type === 'stop') {
      if (run) run.stopped = true;
      run = null;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
};

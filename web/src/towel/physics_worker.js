// Physics in a worker (milestone 3, Phase 4): the towel scene's MuJoCo WASM runs here at real
// time, so that drawing (main thread) and physics (this thread) overlap. At N = 10 a control
// step takes ~25 ms and drawing the free view and the three policy cameras ~3 ms; on one thread
// the page reached only 0.75-0.8 x real time at 25 fps.
// After every control step the poses the view needs are posted back (geom and camera poses,
// towel vertices) with the task status and the 16 joint angles. Two kinds of run:
// - 'play': the expert's recorded actions (Phase 4)
// - 'playModel': the learned policy (Phase 7). The page renders the policy cameras from the
//   posted poses, runs inference and posts the action chunk back ('plan', indexed by the step
//   its observation was taken at, as keigrasp/policy.py and web/src/policy.js ChunkRunner).
//   The physics follows the newest plan and waits only for the first one; it never waits for
//   inference after that, so a late plan skips the steps already taken.
// Messages:
//   in:  {type: 'load', bundle, meshes}  {type: 'play', episode, runId}
//        {type: 'playModel', episode, chunk, runId}  {type: 'plan', planStep, actions}
//        {type: 'drag', target: {xy, yaw} | null}  {type: 'stop'}
//   out: {type: 'loaded'}  {type: 'frame', ...poses, status, runId}  {type: 'done', status, runId}  {type: 'error', message}
// The page ignores frames of a run it has stopped (they can still be in flight).
import loadMujoco from '@mujoco/mujoco';
import { TowelSim } from './sim.js';

let mujoco = null;
let sim = null;
let run = null; // { episode, cursor, t0, stopped, stepMs, model?, plan?, planStep?, chunk?, drag? }
const STEP_MS_KEEP = 300; // recent physics step times kept for the mean

function status() {
  const ms = run?.stepMs ?? [];
  return { steps: sim.steps, current: sim.current, stage: sim.stage, succeeded: sim.succeeded,
    successStep: sim.successStep, meanStepMs: ms.length ? ms.reduce((a, b) => a + b, 0) / ms.length : null,
    joints: sim.joints() };
}

function timedStep(action) {
  const s0 = performance.now();
  sim.step(action);
  run.stepMs.push(performance.now() - s0);
  if (run.stepMs.length > STEP_MS_KEEP) run.stepMs.shift();
}

function post() {
  const d = sim.data;
  const v0 = 3 * sim.vadr;
  const frame = {
    type: 'frame',
    geom_xpos: d.geom_xpos.slice(), geom_xmat: d.geom_xmat.slice(),
    cam_xpos: d.cam_xpos.slice(), cam_xmat: d.cam_xmat.slice(),
    flexvert_xpos: d.flexvert_xpos.slice(v0, v0 + 3 * sim.nvert),
    cursor: run.cursor, status: status(), runId: run.runId,
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
    timedStep(actions[run.cursor++]);
    n += 1;
  }
  if (n) post();
  if (run.cursor >= actions.length) {
    self.postMessage({ type: 'done', status: status(), wallMs: performance.now() - run.t0, runId: run.runId });
    run = null;
    return;
  }
  const wait = run.t0 + (run.cursor + 1) * period - performance.now();
  // Behind real time: continue at once through a message (setTimeout(0) is clamped to 4 ms
  // when nested, ~10% of a step); ahead: sleep until the next step is due.
  if (wait <= 0) yieldNow();
  else setTimeout(loop, wait);
}

// The learned policy's run: real time from the first plan on; it runs until stopped (the towel
// can be dragged flat again after a fold, and the policy folds it again).
function loopModel() {
  if (!run || run.stopped || !run.plan) return;
  const period = 1000 / sim.c.control_hz;
  const due = Math.floor((performance.now() - run.t0) / period);
  let n = 0;
  while (sim.steps < due && n < 3) {
    if (run.drag) sim.dragTowel(run.drag.xy, run.drag.yaw);
    sim.disturbIfDue();
    const k = Math.min(sim.steps - run.planStep, run.chunk - 1);
    timedStep(run.plan.subarray(16 * k, 16 * (k + 1)));
    n += 1;
  }
  run.cursor = sim.steps;
  if (n) post();
  const wait = run.t0 + (sim.steps + 1) * period - performance.now();
  if (wait <= 0) yieldNow();
  else setTimeout(loopModel, wait);
}

const channel = new MessageChannel();
channel.port1.onmessage = () => (run?.model ? loopModel() : loop());
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
      run = { episode: msg.episode, cursor: 0, t0: performance.now(), stopped: false, stepMs: [], runId: msg.runId };
      loop();
    } else if (msg.type === 'playModel') {
      sim.reset(msg.episode);
      run = { episode: msg.episode, cursor: 0, t0: null, stopped: false, stepMs: [], model: true,
        plan: null, planStep: 0, chunk: msg.chunk, drag: null, runId: msg.runId };
      post(); // the initial observation
    } else if (msg.type === 'plan') {
      if (!run?.model || run.stopped) return;
      const first = !run.plan;
      run.plan = msg.actions;
      run.planStep = msg.planStep;
      if (first) {
        run.t0 = performance.now() - sim.steps * (1000 / sim.c.control_hz);
        loopModel();
      }
    } else if (msg.type === 'drag') {
      if (run?.model) run.drag = msg.target;
    } else if (msg.type === 'stop') {
      if (run) run.stopped = true;
      run = null;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
};

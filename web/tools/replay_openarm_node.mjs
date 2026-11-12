// Replay the OpenArm expert's recorded actions (and disturbances) in MuJoCo WASM under Node.
// Usage: node tools/replay_openarm_node.mjs [scene seeds...]   (run from web/)
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco from '@mujoco/mujoco';
import { OpenArmSim } from '../src/openarm/sim.js';

const pub = join(dirname(fileURLToPath(import.meta.url)), '..', 'public');
const indexPath = join(pub, 'openarm_scenes', 'index.json');
if (!existsSync(indexPath)) {
  console.error('web/public/openarm_scenes is missing: run `uv run python scripts/export_web_openarm.py`');
  process.exit(1);
}
const index = JSON.parse(readFileSync(indexPath));
const seeds = process.argv.slice(2).map(Number);
const mujoco = await loadMujoco();
console.log(`MuJoCo WASM ${mujoco.mj_versionString()} (OpenArm)`);

const meshCache = new Map();
const results = [];
for (const seed of seeds.length ? seeds : index.scenes) {
  const bundle = JSON.parse(readFileSync(join(pub, 'openarm_scenes', `${seed}.json`)));
  for (const f of bundle.mesh_files) {
    if (!meshCache.has(f)) meshCache.set(f, readFileSync(join(pub, 'assets', 'openarm', f)));
  }
  const meshes = new Map(bundle.mesh_files.map((f) => [f, meshCache.get(f)]));
  const t0 = performance.now();
  const sim = new OpenArmSim(mujoco, bundle, meshes);
  const loadMs = performance.now() - t0;
  for (const ep of bundle.episodes) {
    const t1 = performance.now();
    const s = sim.replay(ep);
    const ms = performance.now() - t1;
    // Stage bookkeeping vs. the Python Monitor (first step of each stage).
    const stages = bundle.constants.stages;
    // WASM and Python are not bit-identical (contacts can flicker a step apart), so allow a
    // few steps; a wrong rule would be off by much more or miss a stage entirely.
    const stageDiff = Math.max(0, ...Object.entries(ep.expert.stage_steps)
      .map(([name, step]) => Math.abs((sim.stageStep[stages.indexOf(name)] ?? Infinity) - step)));
    const first = { wasm: sim.succeeded, step: sim.successStep, steps: sim.steps };
    // Disturbance timing: arm the plan (step unknown) and let the stage trigger fire it; it must
    // fire at the step Python recorded.
    let fired = null;
    if (ep.disturbance) {
      sim.reset(ep);
      sim.armDisturbance(ep.planned_disturbances[ep.disturbance.kind]);
      for (const action of ep.expert.actions) {
        sim.disturbIfDue();
        sim.step(action);
      }
      fired = sim.disturbance.step;
    }
    const barErr = Math.hypot(...s.barPos.map((v, i) => v - ep.expert.final_bar_pos[i]));
    const jointErr = Math.max(...s.joints.map((v, i) => Math.abs(v - ep.expert.final_joints[i])));
    results.push({ scene: seed, episode: ep.episode_seed, disturb: ep.disturbance?.kind ?? 'none',
      python: ep.expert.success, wasm: first.wasm, python_step: ep.expert.steps, wasm_step: first.step,
      stage_step_diff: stageDiff, disturb_step_ok: ep.disturbance ? fired === ep.disturbance.step : '',
      bar_err_mm: +(barErr * 1000).toFixed(3), joint_err_rad: +jointErr.toExponential(2),
      realtime: +((first.steps / bundle.constants.control_hz) / (ms / 1000)).toFixed(1) });
  }
  results.at(-1).load_ms = Math.round(loadMs);
  sim.dispose();
}
console.table(results);
const ok = results.filter((r) => r.wasm && r.stage_step_diff <= 3 && r.disturb_step_ok !== false).length;
console.log(JSON.stringify({ episodes: results.length, wasm_success: ok,
  python_success: results.filter((r) => r.python).length,
  max_bar_err_mm: Math.max(...results.map((r) => r.bar_err_mm)),
  max_stage_step_diff: Math.max(...results.map((r) => r.stage_step_diff)),
  max_success_step_diff: Math.max(...results.map((r) => Math.abs((r.wasm_step ?? Infinity) - r.python_step))) }));
process.exit(ok === results.length ? 0 : 1);

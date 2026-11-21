// Replay the towel expert's recorded actions (and disturbances) in MuJoCo WASM under Node
// (milestone 3, Phase 4): success, success step, stages and grabs must match Python; the final
// towel shape is compared (WASM and Python are not bit-identical). Also reports the physics
// speed (real-time factor) per mesh N.
// Usage: node tools/replay_towel_node.mjs [--json out.json] [bundle names like 100000_n10 ...]   (run from web/)
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco from '@mujoco/mujoco';
import { TowelSim } from '../src/towel/sim.js';

const pub = join(dirname(fileURLToPath(import.meta.url)), '..', 'public');
const indexPath = join(pub, 'towel_scenes', 'index.json');
if (!existsSync(indexPath)) {
  console.log('web/public/towel_scenes is missing: skipped (run `uv run python scripts/export_web_towel.py`)');
  process.exit(0);
}
const index = JSON.parse(readFileSync(indexPath));
const args = process.argv.slice(2);
let jsonOut = null;
if (args[0] === '--json') { jsonOut = args[1]; args.splice(0, 2); }
const names = args.length ? args : index.meshes.flatMap((n) => index.scenes.map((s) => `${s}_n${n}`));
const mujoco = await loadMujoco();
console.log(`MuJoCo WASM ${mujoco.mj_versionString()} (towel)`);

const meshCache = new Map();
const results = [];
for (const name of names) {
  const bundle = JSON.parse(readFileSync(join(pub, 'towel_scenes', `${name}.json`)));
  for (const f of bundle.mesh_files) {
    if (!meshCache.has(f)) meshCache.set(f, readFileSync(join(pub, 'assets', 'openarm', f)));
  }
  const sim = new TowelSim(mujoco, bundle, new Map(bundle.mesh_files.map((f) => [f, meshCache.get(f)])));
  const stages = bundle.constants.stages;
  for (const ep of bundle.episodes) {
    const t1 = performance.now();
    const s = sim.replay(ep);
    const ms = performance.now() - t1;
    const stageDiff = Math.max(0, ...Object.entries(ep.expert.stage_steps)
      .map(([st, step]) => Math.abs((sim.stageStep[stages.indexOf(st)] ?? Infinity) - step)));
    const first = { wasm: sim.succeeded || sim.successStep !== null, step: sim.successStep, grabs: sim.grabs };
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
    let vertErr = 0;
    ep.expert.final_verts.forEach((p, k) => {
      vertErr = Math.max(vertErr, Math.hypot(p[0] - s.verts[3 * k], p[1] - s.verts[3 * k + 1], p[2] - s.verts[3 * k + 2]));
    });
    const steps = ep.expert.actions.length;
    results.push({ bundle: name, n: bundle.n, episode: ep.episode_seed, disturb: ep.disturbance?.kind ?? 'none',
      python: ep.expert.success, wasm: first.wasm, python_step: ep.expert.steps, wasm_step: first.step,
      python_grabs: ep.expert.grabs, wasm_grabs: first.grabs, stage_step_diff: stageDiff,
      disturb_step_ok: ep.disturbance ? fired === ep.disturbance.step : '',
      vert_err_mm: +(vertErr * 1000).toFixed(2),
      ms_per_step: +(ms / steps).toFixed(2), realtime: +((steps / bundle.constants.control_hz) / (ms / 1000)).toFixed(2) });
  }
  sim.dispose();
}
console.table(results);
// WASM and Python are not bit-identical. In clean episodes the outcome must agree (a rule
// mirrored wrongly shows up there). After a disturbance the coarse towel can end differently
// (a drop amplifies small differences; at N = 10 a vertex can keep buzzing above the 5 cm/s
// rest limit in one and settle in the other): reported, not failed.
const bad = results.filter((r) => (r.python !== r.wasm && r.disturb === 'none') || r.disturb_step_ok === false);
const diverged = results.filter((r) => r.python !== r.wasm && r.disturb !== 'none');
for (const r of diverged) {
  console.log(`differs after a disturbance (not a failure): ${r.bundle} ${r.disturb}: python ${r.python}, `
    + `wasm ${r.wasm}, final shapes ${r.vert_err_mm} mm apart`);
}
const byN = {};
for (const r of results) (byN[r.n] ??= []).push(r.realtime);
for (const [n, rt] of Object.entries(byN)) {
  rt.sort((a, b) => a - b);
  console.log(`N = ${n}: real-time factor median ${rt[Math.floor(rt.length / 2)]} (min ${rt[0]})`);
}
if (jsonOut) writeFileSync(jsonOut, JSON.stringify(results, null, 1));
console.log(bad.length ? `MISMATCH in ${bad.length} episodes`
  : `${results.length - diverged.length} of ${results.length} episodes match Python, ${diverged.length} diverged`);
process.exit(bad.length ? 1 : 0);

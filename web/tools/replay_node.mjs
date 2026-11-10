// Replay the expert's recorded actions in MuJoCo WASM under Node and report the outcome.
// Usage: node tools/replay_node.mjs [scene seeds...]   (run from web/)
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco from '@mujoco/mujoco';
import { Sim } from '../src/sim.js';

const pub = join(dirname(fileURLToPath(import.meta.url)), '..', 'public');
const index = JSON.parse(readFileSync(join(pub, 'scenes', 'index.json')));
const seeds = process.argv.slice(2).map(Number);
const mujoco = await loadMujoco();
console.log(`MuJoCo WASM ${mujoco.mj_versionString()}`);

const meshCache = new Map();
const results = [];
for (const seed of seeds.length ? seeds : index.scenes) {
  const bundle = JSON.parse(readFileSync(join(pub, 'scenes', `${seed}.json`)));
  for (const f of bundle.mesh_files) {
    if (!meshCache.has(f)) meshCache.set(f, readFileSync(join(pub, 'assets', 'so101', f)));
  }
  const meshes = new Map(bundle.mesh_files.map((f) => [f, meshCache.get(f)]));
  const sim = new Sim(mujoco, bundle, meshes);
  for (const ep of bundle.episodes) {
    sim.reset(ep);
    for (const action of ep.expert.actions) sim.step(action);
    const s = sim.state();
    const cubeErr = Math.hypot(...s.cubePos.map((v, i) => v - ep.expert.final_cube_pos[i]));
    const jointErr = Math.max(...s.joints.map((v, i) => Math.abs(v - ep.expert.final_joints[i])));
    results.push({ scene: seed, episode: ep.episode_seed, python: ep.expert.success, wasm: sim.succeeded,
      steps: sim.steps, cube_err_mm: +(cubeErr * 1000).toFixed(3), joint_err_rad: +jointErr.toExponential(2) });
  }
  sim.dispose();
}
console.table(results);
const ok = results.filter((r) => r.wasm).length;
console.log(JSON.stringify({ episodes: results.length, wasm_success: ok,
  python_success: results.filter((r) => r.python).length,
  max_cube_err_mm: Math.max(...results.map((r) => r.cube_err_mm)) }));
process.exit(ok === results.length ? 0 : 1);

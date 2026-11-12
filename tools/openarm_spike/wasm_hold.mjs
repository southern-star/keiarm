// Load OpenArm v2 move_puck scene in @mujoco/mujoco (WASM) under Node, hold home, time it.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
const mj = (await import(process.argv[2])).default;
const root = process.argv[3];
const mujoco = await mj();
const vfs = new mujoco.MjVFS();
let bytes = 0, files = 0;
const walk = (dir) => { for (const f of readdirSync(dir)) { const p = join(dir, f);
  if (statSync(p).isDirectory()) walk(p); else if (/\.(xml|stl|obj)$/.test(f) && !p.includes('/cell/') ) { const b = readFileSync(p); vfs.addBuffer(relative(root, p), new Uint8Array(b)); bytes += b.length; files++; } } };
walk(root);
let t0 = performance.now();
const m = mujoco.MjModel.from_xml_path('pedestal/move_puck_scene.xml', vfs);
const tLoad = performance.now() - t0;
console.log(`loaded ${files} files ${(bytes/1e6).toFixed(1)} MB, compile ${tLoad.toFixed(0)} ms, nq=${m.nq} nu=${m.nu} ngeom=${m.ngeom} nmesh=${m.nmesh} timestep=${m.opt.timestep}`);
for (const ts of [0.001, 1/240]) {
  m.opt.timestep = ts;
  // gravity compensation on arm bodies (body_gravcomp) and route it through actuators (jnt_actgravcomp)
  const gc = m.body_gravcomp; for (let b = 1; b < m.nbody; b++) { const n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY.value, b); if (n && n.startsWith('openarm_')) gc[b] = 1; }
  const ag = m.jnt_actgravcomp; for (let j = 0; j < m.njnt; j++) { const n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT.value, j); if (n && n.startsWith('openarm_')) ag[j] = 1; }
  m.ngravcomp = Array.from(m.body_gravcomp).filter(x => x > 0).length; if (ts === 0.001) console.log('gravcomp bodies', m.ngravcomp);
  const d = new mujoco.MjData(m);
  mujoco.mj_resetDataKeyframe(m, d, 0);
  const n = Math.round(2 / ts);
  t0 = performance.now();
  for (let i = 0; i < n; i++) mujoco.mj_step(m, d);
  const wall = (performance.now() - t0) / 1000;
  const q = d.qpos; let maxErr = 0; const ctrl = d.ctrl;
  for (let a = 0; a < m.nu; a++) { const j = m.actuator_trnid[2*a]; maxErr = Math.max(maxErr, Math.abs(q[m.jnt_qposadr[j]] - ctrl[a])); }
  console.log(`ts=${ts.toFixed(5)} 2s sim in ${wall.toFixed(2)} s -> RTF ${(2/wall).toFixed(1)}x, max|q-ctrl| ${maxErr.toFixed(4)} rad, puck z ${q[20].toFixed(4)}, finite ${Array.from(q).every(Number.isFinite)}`);
  d.delete();
}
m.delete(); vfs.delete();

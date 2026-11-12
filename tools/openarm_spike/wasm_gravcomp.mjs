import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
const mujoco = await (await import(process.argv[2])).default();
const root = process.argv[3];
const vfs = new mujoco.MjVFS();
const walk = (dir) => { for (const f of readdirSync(dir)) { const p = join(dir, f);
  if (statSync(p).isDirectory()) walk(p); else if (/\.(xml|stl|obj)$/.test(f) && !p.includes('/cell/')) vfs.addBuffer(relative(root, p), new Uint8Array(readFileSync(p))); } };
walk(root);
const m = mujoco.MjModel.from_xml_path('pedestal/move_puck_scene.xml', vfs);
const a = m.body_gravcomp; a[5] = 1; console.log('typed?', a.constructor.name, 'readback', m.body_gravcomp[5]);
// spec route
const spec = mujoco.parseXMLString(readFileSync(join(root, 'pedestal/move_puck_scene.xml'), 'utf8'), vfs);
console.log('spec methods', Object.getOwnPropertyNames(Object.getPrototypeOf(spec)).filter(n => /bod|joint|first|next|find|option/i.test(n)).join(' '));
for (const mode of ['none', 'body', 'body+act']) {
  const mm = mujoco.MjModel.from_xml_path('pedestal/move_puck_scene.xml', vfs);
  mm.opt.timestep = 1/240;
  if (mode !== 'none') {
    for (let b = 1; b < mm.nbody; b++) { const n = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY.value, b); if (n && n.startsWith('openarm_')) mm.body_gravcomp[b] = 1; }
    mm.ngravcomp = 20;
  }
  if (mode === 'body+act') for (let j = 0; j < mm.njnt; j++) { const n = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT.value, j); if (n && n.startsWith('openarm_')) mm.jnt_actgravcomp[j] = 1; }
  const d = new mujoco.MjData(mm); mujoco.mj_resetDataKeyframe(mm, d, 0);
  for (let i = 0; i < 480; i++) mujoco.mj_step(mm, d);
  let e = 0; for (let k = 0; k < mm.nu; k++) { const j = mm.actuator_trnid[2*k]; e = Math.max(e, Math.abs(d.qpos[mm.jnt_qposadr[j]] - d.ctrl[k])); }
  const g = Array.from(d.qfrc_gravcomp).reduce((s, x) => s + Math.abs(x), 0);
  console.log(mode, 'actgravcomp type', mm.jnt_actgravcomp.constructor.name, 'sum|qfrc_gravcomp|', g.toFixed(2), 'max err', e.toFixed(4), 'ctrl0..3', Array.from(d.ctrl).slice(0,4).map(x=>x.toFixed(2)).join(','));
  d.delete(); mm.delete();
}
{
  const mm = mujoco.MjModel.from_xml_path('pedestal/move_puck_scene.xml', vfs);
  mm.ngravcomp = 20; console.log('ngravcomp readback', mm.ngravcomp); mm.delete();
  const xml = readFileSync(join(root, 'openarm_bimanual.xml'), 'utf8').replaceAll('<body name="openarm_', '<body gravcomp="1" name="openarm_');
  vfs.deleteFile?.('openarm_bimanual.xml');
  vfs.addBuffer('openarm_bimanual.xml', new TextEncoder().encode(xml));
  const m2 = mujoco.MjModel.from_xml_path('pedestal/move_puck_scene.xml', vfs);
  m2.opt.timestep = 1/240;
  const d = new mujoco.MjData(m2); mujoco.mj_resetDataKeyframe(m2, d, 0);
  const t0 = performance.now();
  for (let i = 0; i < 480; i++) mujoco.mj_step(m2, d);
  let e = 0; for (let k = 0; k < m2.nu; k++) { const j = m2.actuator_trnid[2*k]; e = Math.max(e, Math.abs(d.qpos[m2.jnt_qposadr[j]] - d.ctrl[k])); }
  console.log('xml gravcomp: ngravcomp', m2.ngravcomp, 'max err', e.toFixed(4), 'RTF', (2 / ((performance.now()-t0)/1000)).toFixed(1));
}

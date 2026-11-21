// E5（Node）: 書き出したシーンを固定版の @mujoco/mujoco（web/node_modules、単一スレッド版）で
// たたみ、Python の結果と並べる。
// 実行（リポジトリの最上位から）: node spikes/cloth/web/node_check.mjs [シーン名...]
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const { default: loadMujoco } = await import(join(here, '../../../web/node_modules/@mujoco/mujoco/mujoco.js'));
const { ClothFold } = await import(join(here, 'fold.mjs'));

const mujoco = await loadMujoco();
console.log(`MuJoCo WASM ${mujoco.mj_versionString()}`);
const names = process.argv.slice(2).length ? process.argv.slice(2) : JSON.parse(readFileSync(join(here, 'scenes/index.json')));
const rows = [];
for (const name of names) {
  const bundle = JSON.parse(readFileSync(join(here, 'scenes', `${name}.json`)));
  const sim = new ClothFold(mujoco, bundle);
  while (sim.step()) { /* run */ }
  const js = sim.summary();
  sim.delete();
  const py = bundle.python;
  const row = { name, js, py: Object.fromEntries(['status', 'half_corner_mm', 'half_area_ratio', 'half2s_corner_mm', 'half2s_area_ratio',
    'quarter_corner_mm', 'quarter_area_ratio', 'quarter2s_corner_mm', 'quarter2s_area_ratio', 'rtf_fold', 'rtf_rest'].map((k) => [k, py[k]])) };
  rows.push(row);
  console.log(name, JSON.stringify(js), '| python', JSON.stringify(row.py));
}
// 既存の結果に名前で上書きして足す
const outPath = join(here, '../out/e5_node.json');
const prev = existsSync(outPath) ? JSON.parse(readFileSync(outPath)) : [];
const merged = [...prev.filter((r) => !rows.some((q) => q.name === r.name)), ...rows].sort((a, b) => a.name.localeCompare(b.name));
writeFileSync(outPath, JSON.stringify(merged, null, 1));

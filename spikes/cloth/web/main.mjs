// E5（ブラウザ）: 布単体のシーンを MuJoCo WASM で動かし、flexvert_xpos から three.js の
// BufferGeometry を毎フレーム更新して描く（表は格子、裏は単色の両面描画、法線は毎フレーム再計算）。
// ?scene=A_n16_s0&mode=realtime|fast
//   realtime: 実時間に合わせて物理を進める（1 フレームの上限 MAX_STEPS）。fps と遅れを測る
//   fast:     1 フレームに FAST_STEPS ずつ進める（物理の ms/step を測る）
// 終わると window.__result に結果を置く。
import * as THREE from 'three';
import loadMujoco from '@mujoco/mujoco';
import { ClothFold } from './fold.mjs';

const qs = new URLSearchParams(location.search);
const sceneName = qs.get('scene') ?? 'A_n16_s0';
const mode = qs.get('mode') ?? 'realtime';
const MAX_STEPS = 60;
const FAST_STEPS = 24;
const hud = document.getElementById('hud');

const mujoco = await loadMujoco();
const bundle = await (await fetch(`./scenes/${sceneName}.json`)).json();
const sim = new ClothFold(mujoco, bundle);
const m = sim.model;

// --- three.js ---
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(1);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202428);
const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.01, 10);
camera.up.set(0, 0, 1);
camera.position.set(0.05, -0.45, 0.35);
camera.lookAt(0, 0, 0.02);
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(0.3, -0.4, 1.0);
scene.add(sun);
const table = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.8), new THREE.MeshLambertMaterial({ color: 0x8c8c99 }));
scene.add(table);

// 布: flex の三角形と texcoord から BufferGeometry
const f = sim.flex;
const nv = sim.nvert;
const ne = m.flex_elemnum[f];
const eadr = m.flex_elemdataadr[f];
const index = new Uint32Array(3 * ne);
for (let k = 0; k < 3 * ne; k++) index[k] = m.flex_elem[eadr + k];
const pos = new Float32Array(3 * nv);
const uv = new Float32Array(2 * nv);
const tadr = m.flex_texcoordadr[f];
for (let k = 0; k < nv; k++) {
  uv[2 * k] = tadr >= 0 ? m.flex_texcoord[2 * (tadr + k)] : 0;
  uv[2 * k + 1] = tadr >= 0 ? m.flex_texcoord[2 * (tadr + k) + 1] : 0;
}
const geom = new THREE.BufferGeometry();
geom.setIndex(new THREE.BufferAttribute(index, 1));
const posAttr = new THREE.BufferAttribute(pos, 3);
posAttr.setUsage(THREE.DynamicDrawUsage);
geom.setAttribute('position', posAttr);
geom.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
const cv = document.createElement('canvas');
cv.width = cv.height = 256;
const g2 = cv.getContext('2d');
for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
  g2.fillStyle = (i + j) % 2 ? '#f2d94c' : '#e64d33';
  g2.fillRect(32 * i, 32 * j, 32, 32);
}
const tex = new THREE.CanvasTexture(cv);
tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
tex.repeat.set(4, 4);
const front = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ map: tex, side: THREE.FrontSide }));
const back = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ color: 0x3359d9, side: THREE.BackSide }));
scene.add(front, back);
const grips = sim.grab.map(() => {
  const s = new THREE.Mesh(new THREE.SphereGeometry(0.006), new THREE.MeshLambertMaterial({ color: 0x1a99e6 }));
  scene.add(s);
  return s;
});

function sync() {
  const x = sim.data.flexvert_xpos;
  const a = 3 * sim.vadr;
  for (let k = 0; k < 3 * nv; k++) pos[k] = x[a + k];
  posAttr.needsUpdate = true;
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
  sim.grab.forEach((g, q) => grips[q].position.set(...[0, 1, 2].map((c) => sim.data.mocap_pos[3 * g.mocap + c])));
}

// --- ループ ---
const frameMs = [];
const renderMs = [];
let t0 = null;
let last = null;
let maxLag = 0;
function frame(now) {
  if (t0 === null) { t0 = now; last = now; }
  frameMs.push(now - last);
  last = now;
  if (mode === 'fast') {
    for (let k = 0; k < FAST_STEPS && sim.step(); k++);
  } else {
    const wall = (now - t0) / 1000;
    let k = 0;
    while (sim.simTime < wall && k < MAX_STEPS && sim.step()) k++;
    maxLag = Math.max(maxLag, wall - sim.simTime);
  }
  const r0 = performance.now();
  sync();
  renderer.render(scene, camera);
  renderMs.push(performance.now() - r0);
  hud.textContent = `${sceneName} ${mode}  sim ${sim.simTime.toFixed(2)} s  ${sim.phase?.name ?? ''}`;
  if (sim.done) {
    const fm = frameMs.slice(1);
    const fps = 1000 / (fm.reduce((a, b) => a + b, 0) / Math.max(fm.length, 1));
    const wallTotal = (now - t0) / 1000;
    window.__result = { scene: sceneName, mode, ...sim.summary(), fps: +fps.toFixed(1), wall_s: +wallTotal.toFixed(2),
      sim_s: +sim.simTime.toFixed(2), max_lag_s: +maxLag.toFixed(3),
      render_ms_mean: +(renderMs.reduce((a, b) => a + b, 0) / renderMs.length).toFixed(2),
      gl: renderer.getContext().getParameter(renderer.getContext().VERSION) };
    hud.textContent += '\ndone';
    return;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

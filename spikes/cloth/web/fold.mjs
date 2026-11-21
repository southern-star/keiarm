// E5: spikes/cloth/e2_fold.py の trial() を写したもの（DOM なし。Node とブラウザで共用）。
// 布単体のシーンで、mocap 2 個と connect 拘束（スロットの相手を実行時に差し替える）で
// 二つ折り → 四つ折りにする。step() を呼ぶたびに 1 物理ステップ進む。

const NEQDATA = 12; // mjNEQDATA（3.17.0）
const N_SLOTS = 4;

function id(mujoco, model, type, name) {
  const i = mujoco.mj_name2id(model, type.value, name);
  if (i < 0) throw new Error(`not found: ${name}`);
  return i;
}

function arc(p0, p1, s, lift) {
  const u = 0.5 - 0.5 * Math.cos(Math.PI * s);
  const p = p0.map((v, k) => v + u * (p1[k] - v));
  p[2] += lift * Math.sin(Math.PI * u);
  return p;
}

export function convexArea(xy) {
  const pts = xy.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) { while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), p) <= 0) lower.pop(); lower.push(p); }
  for (const p of pts.reverse()) { while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), p) <= 0) upper.pop(); upper.push(p); }
  const h = lower.slice(0, -1).concat(upper.slice(0, -1));
  let a = 0;
  for (let i = 0; i < h.length; i++) { const p = h[i], q = h[(i + 1) % h.length]; a += p[0] * q[1] - q[0] * p[1]; }
  return Math.abs(a) / 2;
}

export class ClothFold {
  constructor(mujoco, bundle) {
    this.mujoco = mujoco;
    this.p = bundle.params;
    this.model = mujoco.MjModel.from_xml_string(bundle.xml);
    this.model.opt.timestep = this.p.timestep; // XML の丸めを戻す
    this.data = new mujoco.MjData(this.model);
    const m = this.model, T = mujoco.mjtObj;
    mujoco.mj_forward(m, this.data);
    this.flex = id(mujoco, m, T.mjOBJ_FLEX, 'cloth');
    this.vadr = m.flex_vertadr[this.flex];
    this.nvert = m.flex_vertnum[this.flex];
    this.vertBody = Array.from({ length: this.nvert }, (_, k) => id(mujoco, m, T.mjOBJ_BODY, `cloth_${k}`));
    this.grab = ['grip_a', 'grip_b'].map((g) => ({
      mocap: m.body_mocapid[id(mujoco, m, T.mjOBJ_BODY, g)],
      slots: Array.from({ length: N_SLOTS }, (_, k) => id(mujoco, m, T.mjOBJ_EQUALITY, `${g}_slot${k}`)),
    }));
    this.result = { status: 'ok' };
    this.stepMs = [];
    this.simTime = 0;
    this.plan = this._plan();
    this.phase = null;
    this.done = false;
  }

  vid(i, j) { return i * this.p.n + j; }

  verts() {
    const x = this.data.flexvert_xpos;
    const out = new Array(this.nvert);
    for (let k = 0; k < this.nvert; k++) { const a = 3 * (this.vadr + k); out[k] = [x[a], x[a + 1], x[a + 2]]; }
    return out;
  }

  grabVerts(g, verts) {
    const m = this.model, d = this.data;
    const mp = [0, 1, 2].map((k) => d.mocap_pos[3 * g.mocap + k]);
    const v = this.verts();
    verts.forEach((vk, s) => {
      const slot = g.slots[s];
      m.eq_obj2id[slot] = this.vertBody[vk];
      for (let k = 0; k < 3; k++) {
        m.eq_data[slot * NEQDATA + k] = v[vk][k] - mp[k]; // mocap の向きは単位（回さない）
        m.eq_data[slot * NEQDATA + 3 + k] = 0;
      }
      d.eq_active[slot] = 1;
    });
  }

  release(g) { for (const s of g.slots) this.data.eq_active[s] = 0; }

  cornerPatch(i, j, k) {
    const di = i === 0 ? 1 : -1, dj = j === 0 ? 1 : -1;
    const cand = [[i, j], [i + di, j], [i, j + dj], [i + di, j + dj]];
    return cand.filter(([a, b]) => Math.hypot(a - i, b - j) * this.p.spacing <= 0.02).slice(0, k).map(([a, b]) => this.vid(a, b));
  }

  // 計画: [{name, steps, onStart(), onStep(k, steps)}]
  _plan() {
    const n = this.p.n, h = n / 2, dt = this.p.timestep, top = 2 * this.p.radius;
    const S = (t) => Math.round(t / dt);
    const mean = (vs) => [0, 1, 2].map((k) => vs.reduce((a, v) => a + v[k], 0) / vs.length);
    const plan = [];
    const fold = (tag, grabsFn, targetsFn, t, liftRatio) => {
      let starts;
      plan.push({
        name: `${tag}_fold`, steps: S(t),
        onStart: () => {
          const v = this.verts();
          const grabs = grabsFn(), targets = targetsFn(v);
          starts = grabs.map(([g, vs], q) => {
            const p0 = mean(vs.map((k) => v[k]));
            for (let k = 0; k < 3; k++) this.data.mocap_pos[3 * g.mocap + k] = p0[k];
            this.grabVerts(g, vs);
            return [g, p0, targets[q]];
          });
        },
        onStep: (k, steps) => {
          for (const [g, p0, p1] of starts) {
            const lift = liftRatio * Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 2;
            const p = arc(p0, p1, k / steps, lift);
            for (let c = 0; c < 3; c++) this.data.mocap_pos[3 * g.mocap + c] = p[c];
          }
        },
      });
      plan.push({ name: `${tag}_hold`, steps: S(0.3) });
      plan.push({ name: `${tag}_release`, steps: 0, onStart: () => this.grab.forEach((g) => this.release(g)) });
    };
    const [A, B] = this.grab;
    plan.push({ name: 'settle', steps: S(0.5), onEnd: () => { this.area0 = convexArea(this.verts().map((v) => v.slice(0, 2))); } });
    fold('half', () => [[A, this.cornerPatch(n - 1, 0, 3)], [B, this.cornerPatch(n - 1, n - 1, 3)]],
      (v) => [this.vid(0, 0), this.vid(0, n - 1)].map((k) => [v[k][0], v[k][1], v[k][2] + top + 0.004]),
      this.p.t_fold[0], this.p.lift_ratio[0]);
    const halfCorners = [[this.vid(n - 1, 0), this.vid(0, 0)], [this.vid(n - 1, n - 1), this.vid(0, n - 1)]];
    plan.push({ name: 'rest', steps: S(0.2), onEnd: () => this.measure('half', halfCorners) });
    plan.push({ name: 'rest', steps: S(2.0), onEnd: () => this.measure('half2s', halfCorners) });
    fold('quarter',
      () => [[A, [this.vid(0, n - 1), this.vid(n - 1, n - 1), this.vid(0, n - 2), this.vid(n - 1, n - 2)]],
        [B, [this.vid(h - 1, n - 1), this.vid(h, n - 1), this.vid(h - 1, n - 2), this.vid(h, n - 2)]]],
      (v) => [mean([v[this.vid(0, 0)], v[this.vid(n - 1, 0)]]), mean([v[this.vid(h - 1, 0)], v[this.vid(h, 0)]])]
        .map((p) => [p[0], p[1], p[2] + 2 * top + 0.004]),
      this.p.t_fold[1], this.p.lift_ratio[1]);
    const qCorners = [[this.vid(0, n - 1), this.vid(0, 0)], [this.vid(h - 1, n - 1), this.vid(h - 1, 0)]];
    plan.push({ name: 'rest', steps: S(0.2), onEnd: () => this.measure('quarter', qCorners) });
    plan.push({ name: 'rest', steps: S(2.0), onEnd: () => this.measure('quarter2s', qCorners) });
    return plan;
  }

  measure(tag, cornerPairs) {
    const v = this.verts();
    const dist = (a, b) => Math.hypot(v[a][0] - v[b][0], v[a][1] - v[b][1], v[a][2] - v[b][2]);
    this.result[`${tag}_corner_mm`] = +(1000 * Math.max(...cornerPairs.map(([a, b]) => dist(a, b)))).toFixed(1);
    this.result[`${tag}_area_ratio`] = +(convexArea(v.map((p) => p.slice(0, 2))) / this.area0).toFixed(4);
    this.result[`${tag}_zmax_mm`] = +(1000 * Math.max(...v.map((p) => p[2]))).toFixed(2);
  }

  // 1 物理ステップ進める。終わったら false
  step() {
    if (this.done) return false;
    if (!this.phase) {
      const ph = this.plan.shift();
      if (!ph) { this.done = true; return false; }
      this.phase = { ...ph, k: 0 };
      ph.onStart?.();
      if (ph.steps === 0) { ph.onEnd?.(); this.phase = null; return this.step(); }
    }
    const ph = this.phase;
    ph.k += 1;
    ph.onStep?.(ph.k, ph.steps);
    const t0 = performance.now();
    this.mujoco.mj_step(this.model, this.data);
    this.stepMs.push(performance.now() - t0);
    this.simTime += this.p.timestep;
    const x = this.data.flexvert_xpos;
    for (let k = 3 * this.vadr; k < 3 * (this.vadr + this.nvert); k++) {
      if (!Number.isFinite(x[k]) || Math.abs(x[k]) > 1) { this.result.status = 'broken'; this.done = true; return false; }
    }
    if (ph.k >= ph.steps) { ph.onEnd?.(); this.phase = null; }
    return true;
  }

  summary() {
    const s = this.stepMs.slice().sort((a, b) => a - b);
    const mean = s.reduce((a, b) => a + b, 0) / Math.max(s.length, 1);
    return { ...this.result, steps: s.length, step_ms_mean: +mean.toFixed(3), step_ms_p95: +(s[Math.floor(0.95 * s.length)] ?? 0).toFixed(3),
      rtf_physics: +(this.p.timestep / (mean / 1000)).toFixed(3) };
  }

  delete() { this.data.delete(); this.model.delete(); }
}

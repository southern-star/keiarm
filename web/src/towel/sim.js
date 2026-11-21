// MuJoCo WASM wrapper for one towel scene (milestone 3): load, reset, step at 30 Hz with the
// grasp rule, read the task state, check the fold, apply disturbances. Mirrors
// keiarm/towel/scene.py (reset_data, TowelIndex.place), keiarm/towel/grasp.py (GraspRule),
// keiarm/towel/task.py (read_state, check_fold, stage_of), keiarm/towel/eval.py (Monitor's
// stage bookkeeping, Disturbance.apply) and keiarm/towel/env.py (place_towel, slip, kick).
// Constants come from the bundle (keiarm/towel/export.py). No DOM or three.js here: the same
// code runs in the browser and under Node (tools/replay_towel_node.mjs).

const WORK_DIR = '/towel';
const SIDES = ['left', 'right'];
const MJ_NEQDATA = 12; // mujoco.mjNEQDATA in 3.17 (checked against the model below)

function nameToId(mujoco, model, objType, name) {
  const id = mujoco.mj_name2id(model, objType.value, name);
  if (id < 0) throw new Error(`MuJoCo object not found: ${name}`);
  return id;
}

// R (row-major 3x3 at offset o of a flat array) times v
function rotate(R, o, v) {
  return [0, 1, 2].map((r) => R[o + 3 * r] * v[0] + R[o + 3 * r + 1] * v[1] + R[o + 3 * r + 2] * v[2]);
}

function convexArea(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  let a = 0;
  for (let i = 0; i < hull.length; i++) {
    const [x1, y1] = hull[i];
    const [x0, y0] = hull[(i + hull.length - 1) % hull.length];
    a += x1 * y0 - y1 * x0;
  }
  return Math.abs(a) / 2;
}

export class TowelSim {
  // bundle: parsed web/public/towel_scenes/<seed>_n<N>.json
  // meshes: Map<relative mesh path, Uint8Array> for every entry of bundle.mesh_files
  constructor(mujoco, bundle, meshes) {
    this.mujoco = mujoco;
    this.bundle = bundle;
    // milestone 2's appearance fields (render.js reads them at the top level) and the towel's
    this.scene = { ...bundle.scene.base, ...bundle.scene };
    this.c = bundle.constants;
    const c = this.c;

    const dir = `${WORK_DIR}/${this.scene.scene_seed}_n${bundle.n}`;
    const FS = mujoco.FS;
    for (const [name, bytes] of meshes) {
      const path = `${dir}/${bundle.mesh_dir}/${name}`;
      FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
      FS.writeFile(path, bytes);
    }
    FS.writeFile(`${dir}/scene.xml`, bundle.xml);
    this.model = mujoco.MjModel.from_xml_path(`${dir}/scene.xml`);
    this.model.opt.timestep = c.timestep; // the MJCF text carries a rounded value
    this.data = new mujoco.MjData(this.model);

    const m = this.model;
    const T = mujoco.mjtObj;
    this.act = c.actuators.map((n) => nameToId(mujoco, m, T.mjOBJ_ACTUATOR, n));
    this.qadr = this.act.map((a) => m.jnt_qposadr[m.actuator_trnid[2 * a]]);
    this.resetQadr = c.reset_joints.map((names) => names.map((n) => m.jnt_qposadr[nameToId(mujoco, m, T.mjOBJ_JOINT, n)]));
    this.ctrlRange = this.act.map((a) => [m.actuator_ctrlrange[2 * a], m.actuator_ctrlrange[2 * a + 1]]);
    if (m.eq_data.length !== MJ_NEQDATA * m.neq) throw new Error('unexpected mjNEQDATA');

    // towel (keiarm/towel/scene.py TowelIndex)
    this.flex = nameToId(mujoco, m, T.mjOBJ_FLEX, c.towel);
    this.vadr = m.flex_vertadr[this.flex];
    this.nvert = m.flex_vertnum[this.flex];
    this.vertBody = new Int32Array(this.nvert);
    this.vertQadr = new Int32Array(3 * this.nvert);
    this.vertDadr = new Int32Array(3 * this.nvert);
    this.rest = new Float64Array(3 * this.nvert);
    for (let k = 0; k < this.nvert; k++) {
      const b = nameToId(mujoco, m, T.mjOBJ_BODY, `${c.towel}_${k}`);
      this.vertBody[k] = b;
      for (let a = 0; a < 3; a++) {
        const j = m.body_jntadr[b] + a;
        this.vertQadr[3 * k + a] = m.jnt_qposadr[j];
        this.vertDadr[3 * k + a] = m.jnt_dofadr[j];
        this.rest[3 * k + a] = m.body_pos[3 * b + a];
      }
    }

    // grasp rule (keiarm/towel/grasp.py)
    this.slots = Object.fromEntries(SIDES.map((s) => [s,
      [...Array(c.grab_slots).keys()].map((k) => nameToId(mujoco, m, T.mjOBJ_EQUALITY, `${s}_grab${k}`))]));
    this.hand = Object.fromEntries(SIDES.map((s) => [s, nameToId(mujoco, m, T.mjOBJ_BODY, c.hand_bodies[s])]));
    this.site = Object.fromEntries(SIDES.map((s) => [s, nameToId(mujoco, m, T.mjOBJ_SITE, c.ee_sites[s])]));
    this.gripAct = Object.fromEntries(SIDES.map((s) => [s, nameToId(mujoco, m, T.mjOBJ_ACTUATOR, c.grip_actuators[s])]));
    this.fingerQadr = Object.fromEntries(SIDES.map((s) => [s,
      m.jnt_qposadr[nameToId(mujoco, m, T.mjOBJ_JOINT, c.finger_joint[s])]]));

    // finger and arm geoms (keiarm/towel/task.py TaskIndex)
    this.fingerGeom = new Map(); // geom -> side
    this.armGeom = new Map(); // colliding geom -> side
    for (let g = 0; g < m.ngeom; g++) {
      const name = mujoco.mj_id2name(m, T.mjOBJ_BODY.value, m.geom_bodyid[g]) ?? '';
      for (const s of SIDES) {
        if (name.startsWith(c.finger_body_prefix[s]) && name.endsWith(c.finger_body_suffix)) this.fingerGeom.set(g, s);
        if (name.startsWith(c.arm_body_prefix[s]) && m.geom_contype[g]) this.armGeom.set(g, s);
      }
    }
    this.towelSize = this.scene.towel_size;
    this.grasp = null;
  }

  // --- episode -----------------------------------------------------------------------------

  reset(episode) {
    const { mujoco, model: m, data: d } = this;
    mujoco.mj_resetData(m, d);
    episode.start_ctrl.forEach((q, i) => {
      for (const a of this.resetQadr[i]) d.qpos[a] = q;
      d.ctrl[this.act[i]] = q;
    });
    this.placeVerts(episode.towel_xy, episode.towel_yaw);
    this.grasp = Object.fromEntries(SIDES.map((s) => [s, { latched: false, verts: [], anchors: [] }]));
    this.applyGrasp();
    mujoco.mj_forward(m, d);
    this.episode = episode;
    this.steps = 0;
    this.successRun = 0;
    this.successStep = null; // control step at which success was first confirmed
    this.stage = 0; // furthest stage (index into constants.stages; the last only once success held)
    this.current = 0; // instantaneous stage, for display
    this.stageStep = {}; // stage index -> first control step (1-based count) it was reached
    this.grabs = 0;
    this.disturbance = null;
    this.lastState = this.state();
  }

  // keiarm/towel/scene.py TowelIndex.place: towel flat at (xy, yaw), at rest
  placeVerts([x, y], yaw) {
    const d = this.data;
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    for (let k = 0; k < this.nvert; k++) {
      const lx = this.rest[3 * k];
      const ly = this.rest[3 * k + 1];
      d.qpos[this.vertQadr[3 * k]] = c * lx - s * ly + x - lx;
      d.qpos[this.vertQadr[3 * k + 1]] = s * lx + c * ly + y - ly;
      d.qpos[this.vertQadr[3 * k + 2]] = 0;
      for (let a = 0; a < 3; a++) d.qvel[this.vertDadr[3 * k + a]] = 0;
    }
  }

  // --- grasp rule ----------------------------------------------------------------------------

  fingertip(side) {
    const d = this.data;
    const i = this.site[side];
    const r = rotate(d.site_xmat, 9 * i, this.c.fingertip_offset);
    return [0, 1, 2].map((a) => d.site_xpos[3 * i + a] + r[a]);
  }

  pinch(side) {
    const d = this.data;
    const i = this.site[side];
    const r = rotate(d.site_xmat, 9 * i, this.c.pinch_offset);
    return [0, 1, 2].map((a) => d.site_xpos[3 * i + a] + r[a]);
  }

  vert(k) {
    const x = this.data.flexvert_xpos;
    const o = 3 * (this.vadr + k);
    return [x[o], x[o + 1], x[o + 2]];
  }

  verts() {
    const x = this.data.flexvert_xpos;
    return x.slice(3 * this.vadr, 3 * (this.vadr + this.nvert));
  }

  updateGrasp() {
    const events = [];
    const d = this.data;
    for (const s of SIDES) {
      const h = this.grasp[s];
      const cmd = d.ctrl[this.gripAct[s]] / this.c.gripper_open[s];
      if (h.latched) {
        if (cmd > this.c.open_cmd) {
          this.grasp[s] = { latched: false, verts: [], anchors: [] };
          events.push([s, 'release']);
        }
      } else if (cmd < this.c.close_cmd && d.qpos[this.fingerQadr[s]] / this.c.gripper_open[s] < this.c.close_opening) {
        this.grab(s);
        events.push([s, 'grab']);
      }
    }
    if (events.length) this.applyGrasp();
    return events;
  }

  grab(side) {
    const d = this.data;
    const tip = this.fingertip(side);
    const dist = [];
    for (let k = 0; k < this.nvert; k++) {
      const v = this.vert(k);
      dist.push([Math.hypot(v[0] - tip[0], v[1] - tip[1], v[2] - tip[2]), k]);
    }
    dist.sort((a, b) => a[0] - b[0] || a[1] - b[1]); // stable as numpy's argsort(kind="stable")
    const order = dist.slice(0, this.c.grab_slots).filter(([r]) => r < this.c.grab_radius).map(([, k]) => k);
    const b = this.hand[side];
    const pos = [0, 1, 2].map((a) => d.xpos[3 * b + a]);
    const R = d.xmat;
    const anchors = order.map((k) => {
      const v = this.vert(k).map((x, a) => x - pos[a]);
      // R^T v
      return [0, 1, 2].map((col) => R[9 * b + col] * v[0] + R[9 * b + 3 + col] * v[1] + R[9 * b + 6 + col] * v[2]);
    });
    this.grasp[side] = { latched: true, verts: order, anchors };
    const other = this.grasp[side === 'left' ? 'right' : 'left'];
    const keep = other.verts.map((v, i) => [v, other.anchors[i]]).filter(([v]) => !order.includes(v));
    other.verts = keep.map(([v]) => v);
    other.anchors = keep.map(([, a]) => a);
  }

  applyGrasp() {
    const { model: m, data: d } = this;
    for (const s of SIDES) {
      const h = this.grasp[s];
      this.slots[s].forEach((slot, k) => {
        if (k < h.verts.length) {
          m.eq_obj2id[slot] = this.vertBody[h.verts[k]];
          for (let a = 0; a < 3; a++) {
            m.eq_data[MJ_NEQDATA * slot + a] = h.anchors[k][a];
            m.eq_data[MJ_NEQDATA * slot + 3 + a] = 0;
          }
          d.eq_active[slot] = 1;
        } else {
          d.eq_active[slot] = 0;
        }
      });
    }
  }

  // --- disturbances (keiarm/towel/eval.py Disturbance.apply) --------------------------------

  applyDisturbance(dist) {
    const { mujoco, model: m, data: d } = this;
    if (dist.kind === 'shift') {
      this.placeVerts(dist.xy, dist.yaw);
    } else if (dist.kind === 'slip' || dist.kind === 'drop') {
      for (const s of dist.sides) this.grasp[s] = { latched: true, verts: [], anchors: [] };
      this.applyGrasp();
    } else if (dist.kind === 'nudge') {
      dist.kick_verts.forEach((k, i) => {
        for (let a = 0; a < 3; a++) d.qvel[this.vertDadr[3 * k + a]] += dist.kick_vel[i][a];
      });
    } else {
      throw new Error(`unknown disturbance ${dist.kind}`);
    }
    mujoco.mj_forward(m, d);
    this.lastState = this.state();
    this.successRun = 0;
  }

  // Arm one planned disturbance (step unset): it fires `delay` steps after the episode first
  // reaches `after_stage`, as in the Python harness.
  armDisturbance(plan) {
    this.disturbance = plan ? { ...plan, step: null } : null;
  }

  // Call before choosing the action for the next step. Returns true when it fired.
  disturbIfDue() {
    const dist = this.disturbance;
    if (!dist || dist.step !== null) return false;
    const first = dist.after_stage === 0 ? 0 : this.stageStep[dist.after_stage];
    if (first === undefined || this.steps < first + dist.delay) return false;
    dist.step = this.steps;
    this.applyDisturbance(dist);
    return true;
  }

  // --- step ------------------------------------------------------------------------------------

  // One control period: hold `action` (16 position targets, rad) for `substeps` physics steps,
  // then the grasp rule. Returns the grasp events.
  step(action) {
    const { mujoco, model: m, data: d } = this;
    for (let i = 0; i < this.act.length; i++) {
      const [lo, hi] = this.ctrlRange[i];
      d.ctrl[this.act[i]] = Math.min(hi, Math.max(lo, action[i]));
    }
    for (let k = 0; k < this.c.substeps; k++) mujoco.mj_step(m, d);
    this.steps += 1;
    const events = this.updateGrasp();
    this.grabs += events.filter(([, e]) => e === 'grab').length;
    // Mirrors the Monitor in keiarm/towel/eval.py: 'success' counts only once it has held.
    const s = this.state();
    this.lastState = s;
    const ok = this.checkFold(s).success;
    this.successRun = ok ? this.successRun + 1 : 0;
    const last = this.c.stages.length - 1;
    const now = this.succeeded ? last : Math.min(this.stageOf(s), last - 1);
    for (let k = 1; k <= now; k++) this.stageStep[k] ??= this.steps;
    this.stage = Math.max(this.stage, now);
    this.current = now;
    if (this.successStep === null && this.succeeded) this.successStep = this.steps;
    return events;
  }

  get succeeded() {
    return this.successRun >= this.c.success_hold_steps;
  }

  joints() {
    return this.qadr.map((a) => this.data.qpos[a]);
  }

  // --- task state (keiarm/towel/task.py) -----------------------------------------------------

  // The parts of read_state the checks need. Contacts (finger touch, arms touch) are read
  // lazily (contacts()), only when the rest of the success check passes.
  state() {
    const d = this.data;
    const v = this.verts();
    let maxSpeed = 0;
    for (let k = 0; k < this.nvert; k++) {
      const o = this.vertDadr[3 * k];
      maxSpeed = Math.max(maxSpeed, Math.hypot(d.qvel[o], d.qvel[o + 1], d.qvel[o + 2]));
    }
    const held = Object.fromEntries(SIDES.map((s) => [s, this.grasp[s].verts.slice()]));
    const fingertip = Object.fromEntries(SIDES.map((s) => [s, this.fingertip(s)]));
    return { verts: v, maxSpeed, held, fingertip };
  }

  // Finger-towel and arm-arm contacts.
  contacts() {
    const d = this.data;
    const touch = { left: false, right: false };
    let arms = false;
    const contacts = d.contact;
    for (let i = 0; i < d.ncon; i++) {
      const con = contacts.get(i);
      if (con.dist <= 0) {
        const g1 = con.geom[0];
        const g2 = con.geom[1];
        const f = con.flex;
        if (f[0] === this.flex || f[1] === this.flex) {
          const g = g1 >= 0 ? g1 : g2;
          const side = this.fingerGeom.get(g);
          if (side) touch[side] = true;
        } else {
          const s1 = this.armGeom.get(g1);
          const s2 = this.armGeom.get(g2);
          if (s1 && s2 && s1 !== s2) arms = true;
        }
      }
      con.delete();
    }
    contacts.delete();
    return { touch, arms };
  }

  cornerGap(s, side) {
    const v = s.verts;
    const f = 3 * this.c.far[side];
    const n = 3 * this.c.near[side];
    return [Math.hypot(v[f] - v[n], v[f + 1] - v[n + 1]), Math.abs(v[f + 2] - v[n + 2])];
  }

  // Mirrors check_fold / FoldCheck.success.
  checkFold(s) {
    const c = this.c;
    const v = s.verts;
    const gaps = Object.fromEntries(SIDES.map((side) => [side, this.cornerGap(s, side)]));
    const pts = [];
    for (let k = 0; k < this.nvert; k++) pts.push([v[3 * k], v[3 * k + 1]]);
    const area = convexArea(pts) / (this.towelSize[0] * this.towelSize[1]);
    const [lo, hi] = c.layers;
    let bad = 0;
    for (let i = 0; i < lo.length; i++) bad += v[3 * hi[i] + 2] < v[3 * lo[i] + 2];
    const violation = bad / lo.length;
    const placed = SIDES.every((side) => gaps[side][0] < c.corner_h && gaps[side][1] < c.corner_v);
    const held = SIDES.some((side) => s.held[side].length);
    let success = placed && area >= c.area_range[0] && area <= c.area_range[1] && violation <= c.max_violation
      && !held && s.maxSpeed <= c.max_speed;
    if (success) {
      s.contacts ??= this.contacts();
      success = !s.contacts.touch.left && !s.contacts.touch.right;
    }
    return { gaps, area, violation, placed, success };
  }

  // Mirrors stage_of (instantaneous stage, index into constants.stages).
  stageOf(s) {
    const c = this.c;
    const chk = this.checkFold(s);
    if (chk.success) return 8;
    const anyHeld = SIDES.some((side) => s.held[side].length);
    if (chk.placed && !anyHeld) return 7;
    if (chk.placed) return 6;
    const holds = Object.fromEntries(SIDES.map((side) => [side, s.held[side].length > 0]));
    if (holds.left && holds.right) {
      if (SIDES.every((side) => chk.gaps[side][0] < c.stage_carry)) return 5;
      if (SIDES.every((side) => s.verts[3 * c.far[side] + 2] > c.stage_lift)) return 4;
      return 3;
    }
    if (holds.left || holds.right) return 2;
    for (const side of SIDES) {
      const tip = s.fingertip[side];
      const f = 3 * c.far[side];
      if (Math.hypot(tip[0] - s.verts[f], tip[1] - s.verts[f + 1], tip[2] - s.verts[f + 2]) < c.stage_reach) return 1;
    }
    return 0;
  }

  // Replay an episode's recorded actions (and its disturbance at the recorded step).
  replay(episode) {
    this.reset(episode);
    const dist = episode.disturbance;
    episode.expert.actions.forEach((action, t) => {
      if (dist && dist.step === t) this.applyDisturbance(dist);
      this.step(action);
    });
    return this.state();
  }

  // WASM objects are not garbage collected: free them explicitly.
  dispose() {
    this.data.delete();
    this.model.delete();
  }
}

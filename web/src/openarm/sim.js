// MuJoCo WASM wrapper for one OpenArm scene (milestone 2): load, reset, step at 30 Hz, read
// the task state, apply disturbances. Mirrors keiarm/openarm/scene.py (reset_data),
// keiarm/openarm/env.py (teleport_bar, move_plate) and keiarm/openarm/task.py (is_success).
// No DOM or three.js here: the same code runs in the browser and under Node (tests).
// Exposes the fields render.js reads (mujoco, model, data, scene, c), so the SO-101 view is reused.

const WORK_DIR = '/openarm';

function nameToId(mujoco, model, objType, name) {
  const id = mujoco.mj_name2id(model, objType.value, name);
  if (id < 0) throw new Error(`MuJoCo object not found: ${name}`);
  return id;
}

function quatToMat(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

export class OpenArmSim {
  // bundle: parsed web/public/openarm_scenes/<seed>.json
  // meshes: Map<relative mesh path, Uint8Array> for every entry of bundle.mesh_files
  constructor(mujoco, bundle, meshes) {
    this.mujoco = mujoco;
    this.bundle = bundle;
    this.scene = bundle.scene;
    this.c = bundle.constants;
    this.steps = 0;
    this.successRun = 0;
    this.successStep = null;

    const dir = `${WORK_DIR}/${this.scene.scene_seed}`;
    const FS = mujoco.FS;
    for (const [name, bytes] of meshes) {
      const path = `${dir}/${bundle.mesh_dir}/${name}`;
      FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
      FS.writeFile(path, bytes);
    }
    FS.writeFile(`${dir}/scene.xml`, bundle.xml);
    this.model = mujoco.MjModel.from_xml_path(`${dir}/scene.xml`);
    // The MJCF text carries a rounded timestep; restore the exact value Python uses.
    this.model.opt.timestep = this.c.timestep;
    this.data = new mujoco.MjData(this.model);

    const m = this.model;
    const T = mujoco.mjtObj;
    this.act = this.c.actuators.map((n) => nameToId(mujoco, m, T.mjOBJ_ACTUATOR, n));
    // Joint read back per actuator: the one it drives (finger joint 1 for the grippers).
    this.qadr = this.act.map((a) => m.jnt_qposadr[m.actuator_trnid[2 * a]]);
    this.resetQadr = this.c.reset_joints.map((names) =>
      names.map((n) => m.jnt_qposadr[nameToId(mujoco, m, T.mjOBJ_JOINT, n)]));
    const barJoint = nameToId(mujoco, m, T.mjOBJ_JOINT, 'bar');
    this.barQadr = m.jnt_qposadr[barJoint];
    this.barDadr = m.jnt_dofadr[barJoint];
    this.barGeom = nameToId(mujoco, m, T.mjOBJ_GEOM, 'bar');
    this.plateMocap = m.body_mocapid[nameToId(mujoco, m, T.mjOBJ_BODY, 'plate')];
    // Finger geoms per side: geom id -> side.
    this.fingerSide = new Map();
    for (const [side, names] of Object.entries(this.c.finger_bodies)) {
      const bodies = new Set(names.map((n) => nameToId(mujoco, m, T.mjOBJ_BODY, n)));
      for (let g = 0; g < m.ngeom; g++) if (bodies.has(m.geom_bodyid[g])) this.fingerSide.set(g, side);
    }
    this.ctrlRange = this.act.map((a) => [m.actuator_ctrlrange[2 * a], m.actuator_ctrlrange[2 * a + 1]]);
  }

  reset(episode) {
    const { mujoco, model: m, data: d } = this;
    mujoco.mj_resetData(m, d);
    episode.start_ctrl.forEach((q, i) => {
      for (const a of this.resetQadr[i]) d.qpos[a] = q;
      d.ctrl[this.act[i]] = q;
    });
    this.setBar(episode.bar_xy, episode.bar_yaw);
    this.setPlate(episode.plate_xy);
    mujoco.mj_forward(m, d);
    this.episode = episode;
    this.steps = 0;
    this.successRun = 0;
    this.successStep = null; // control step at which success was first confirmed
  }

  setBar([x, y], yaw) {
    const d = this.data;
    const a = this.barQadr;
    d.qpos[a] = x;
    d.qpos[a + 1] = y;
    d.qpos[a + 2] = this.scene.bar_size[2] / 2 + 1e-3;
    d.qpos[a + 3] = Math.cos(yaw / 2);
    d.qpos[a + 4] = 0;
    d.qpos[a + 5] = 0;
    d.qpos[a + 6] = Math.sin(yaw / 2);
    for (let k = 0; k < 6; k++) d.qvel[this.barDadr + k] = 0;
  }

  setPlate([x, y]) {
    const p = 3 * this.plateMocap;
    this.data.mocap_pos[p] = x;
    this.data.mocap_pos[p + 1] = y;
    this.data.mocap_pos[p + 2] = this.c.plate_height / 2;
  }

  // Disturbances (keiarm/openarm/eval.py Disturbance.apply): bar flat on the table at rest,
  // or the plate moved. Also used for dragging the bar by hand.
  teleportBar(xy, yaw) {
    this.setBar(xy, yaw);
    this.mujoco.mj_forward(this.model, this.data);
  }

  movePlate(xy) {
    this.setPlate(xy);
    this.mujoco.mj_forward(this.model, this.data);
  }

  applyDisturbance(dist) {
    if (dist.kind === 'plate') this.movePlate([dist.x, dist.y]);
    else this.teleportBar([dist.x, dist.y], dist.yaw);
  }

  // One control period: hold `action` (16 position targets, rad) for `substeps` physics steps.
  step(action) {
    const { mujoco, model: m, data: d } = this;
    for (let i = 0; i < this.act.length; i++) {
      const [lo, hi] = this.ctrlRange[i];
      d.ctrl[this.act[i]] = Math.min(hi, Math.max(lo, action[i]));
    }
    for (let k = 0; k < this.c.substeps; k++) mujoco.mj_step(m, d);
    this.steps += 1;
    this.successRun = this.isSuccessNow() ? this.successRun + 1 : 0;
    if (this.successStep === null && this.succeeded) this.successStep = this.steps;
  }

  get succeeded() {
    return this.successRun >= this.c.success_hold_steps;
  }

  joints() {
    return this.qadr.map((a) => this.data.qpos[a]);
  }

  // Privileged task state (the parts of keiarm/openarm/task.py read_state the checks need).
  state() {
    const d = this.data;
    const a = this.barQadr;
    const v = this.barDadr;
    const p = 3 * this.plateMocap;
    const touch = { left: false, right: false };
    const contacts = d.contact;
    for (let i = 0; i < d.ncon; i++) {
      const c = contacts.get(i);
      if (c.dist <= 0 && (c.geom1 === this.barGeom || c.geom2 === this.barGeom)) {
        const side = this.fingerSide.get(c.geom1 === this.barGeom ? c.geom2 : c.geom1);
        if (side) touch[side] = true;
      }
      c.delete();
    }
    contacts.delete();
    const barQuat = [d.qpos[a + 3], d.qpos[a + 4], d.qpos[a + 5], d.qpos[a + 6]];
    return {
      joints: this.joints(),
      barPos: [d.qpos[a], d.qpos[a + 1], d.qpos[a + 2]],
      barMat: quatToMat(barQuat),
      barLinvel: [d.qvel[v], d.qvel[v + 1], d.qvel[v + 2]],
      platePos: [d.mocap_pos[p], d.mocap_pos[p + 1], d.mocap_pos[p + 2]],
      touch,
    };
  }

  // Mirrors keiarm/openarm/task.py is_success (the instantaneous condition).
  isSuccessNow() {
    const s = this.state();
    const dx = s.barPos[0] - s.platePos[0];
    const dy = s.barPos[1] - s.platePos[1];
    const plateTop = s.platePos[2] + this.c.plate_height / 2;
    const tilt = Math.asin(Math.min(1, Math.abs(s.barMat[7]))); // row z, column 1 (long axis)
    return Math.hypot(dx, dy) < this.scene.plate_radius
      && tilt < this.c.rest_tilt
      && Math.abs(s.barPos[2] - (plateTop + this.scene.bar_size[2] / 2)) < this.c.rest_z_tol
      && Math.hypot(...s.barLinvel) < this.c.rest_speed
      && !s.touch.left && !s.touch.right;
  }

  // Overwrite the full state (render comparison tool), then recompute poses.
  setState(qpos, mocapPos) {
    this.data.qpos.set(qpos);
    this.data.mocap_pos.set(mocapPos);
    this.mujoco.mj_forward(this.model, this.data);
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

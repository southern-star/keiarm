// MuJoCo WASM wrapper for one KeiArm scene: load, reset, step at 30 Hz, read the task state.
// No DOM or three.js here, so the same code runs in the browser and under Node (tests).

const WORK_DIR = '/keiarm';

function nameToId(mujoco, model, objType, name) {
  const id = mujoco.mj_name2id(model, objType.value, name);
  if (id < 0) throw new Error(`MuJoCo object not found: ${name}`);
  return id;
}

export class Sim {
  // bundle: parsed scene JSON from web/public/scenes/<seed>.json
  // meshes: Map<file name, Uint8Array> for every entry of bundle.mesh_files
  constructor(mujoco, bundle, meshes) {
    this.mujoco = mujoco;
    this.bundle = bundle;
    this.scene = bundle.scene;
    this.c = bundle.constants;
    this.steps = 0;
    this.successRun = 0;

    const dir = `${WORK_DIR}/${this.scene.scene_seed}`;
    const FS = mujoco.FS;
    FS.mkdirTree(`${dir}/${bundle.mesh_dir}`);
    for (const [name, bytes] of meshes) FS.writeFile(`${dir}/${bundle.mesh_dir}/${name}`, bytes);
    FS.writeFile(`${dir}/scene.xml`, bundle.xml);
    this.model = mujoco.MjModel.from_xml_path(`${dir}/scene.xml`);
    // The MJCF text carries a rounded timestep; restore the exact value Python uses.
    this.model.opt.timestep = this.c.timestep;
    this.data = new mujoco.MjData(this.model);

    const m = this.model;
    const T = mujoco.mjtObj;
    this.qadr = this.c.joint_names.map((n) => m.jnt_qposadr[nameToId(mujoco, m, T.mjOBJ_JOINT, n)]);
    this.act = this.c.joint_names.map((n) => nameToId(mujoco, m, T.mjOBJ_ACTUATOR, n));
    const cubeJoint = nameToId(mujoco, m, T.mjOBJ_JOINT, 'cube');
    this.cubeQadr = m.jnt_qposadr[cubeJoint];
    this.cubeDadr = m.jnt_dofadr[cubeJoint];
    this.cubeGeom = nameToId(mujoco, m, T.mjOBJ_GEOM, 'cube');
    this.plateMocap = m.body_mocapid[nameToId(mujoco, m, T.mjOBJ_BODY, 'plate')];
    this.eeSite = nameToId(mujoco, m, T.mjOBJ_SITE, this.c.ee_site);
    const bodies = (names) => new Set(names.map((n) => nameToId(mujoco, m, T.mjOBJ_BODY, n)));
    const fixed = bodies(this.c.fixed_jaw_bodies);
    const moving = bodies(this.c.moving_jaw_bodies);
    this.fixedGeoms = new Set();
    this.movingGeoms = new Set();
    for (let g = 0; g < m.ngeom; g++) {
      if (fixed.has(m.geom_bodyid[g])) this.fixedGeoms.add(g);
      if (moving.has(m.geom_bodyid[g])) this.movingGeoms.add(g);
    }
    this.ctrlRange = this.act.map((a) => [m.actuator_ctrlrange[2 * a], m.actuator_ctrlrange[2 * a + 1]]);
  }

  reset(episode) {
    const { mujoco, model: m, data: d } = this;
    mujoco.mj_resetData(m, d);
    episode.arm_qpos.forEach((q, i) => {
      d.qpos[this.qadr[i]] = q;
      d.ctrl[this.act[i]] = q;
    });
    const a = this.cubeQadr;
    d.qpos[a] = episode.cube_xy[0];
    d.qpos[a + 1] = episode.cube_xy[1];
    d.qpos[a + 2] = this.scene.cube_size / 2 + 1e-3;
    d.qpos[a + 3] = Math.cos(episode.cube_yaw / 2);
    d.qpos[a + 4] = 0;
    d.qpos[a + 5] = 0;
    d.qpos[a + 6] = Math.sin(episode.cube_yaw / 2);
    const p = 3 * this.plateMocap;
    d.mocap_pos[p] = episode.plate_xy[0];
    d.mocap_pos[p + 1] = episode.plate_xy[1];
    d.mocap_pos[p + 2] = this.c.plate_height / 2;
    mujoco.mj_forward(m, d);
    this.episode = episode;
    this.steps = 0;
    this.successRun = 0;
  }

  // One control period: hold `action` (6 joint targets, rad) for `substeps` physics steps.
  step(action) {
    const { mujoco, model: m, data: d } = this;
    for (let i = 0; i < 6; i++) {
      const [lo, hi] = this.ctrlRange[i];
      d.ctrl[this.act[i]] = Math.min(hi, Math.max(lo, action[i]));
    }
    for (let k = 0; k < this.c.substeps; k++) mujoco.mj_step(m, d);
    this.steps += 1;
    this.successRun = this.isSuccessNow() ? this.successRun + 1 : 0;
  }

  get succeeded() {
    return this.successRun >= this.c.success_hold_steps;
  }

  joints() {
    return this.qadr.map((a) => this.data.qpos[a]);
  }

  // Privileged task state (mirrors keiarm/task.py read_state).
  state() {
    const d = this.data;
    const a = this.cubeQadr;
    const v = this.cubeDadr;
    const p = 3 * this.plateMocap;
    let touchFixed = false;
    let touchMoving = false;
    const contacts = d.contact;
    for (let i = 0; i < d.ncon; i++) {
      const c = contacts.get(i);
      if (c.dist <= 0 && (c.geom1 === this.cubeGeom || c.geom2 === this.cubeGeom)) {
        const other = c.geom1 === this.cubeGeom ? c.geom2 : c.geom1;
        touchFixed ||= this.fixedGeoms.has(other);
        touchMoving ||= this.movingGeoms.has(other);
      }
      c.delete();
    }
    contacts.delete();
    return {
      joints: this.joints(),
      cubePos: [d.qpos[a], d.qpos[a + 1], d.qpos[a + 2]],
      cubeQuat: [d.qpos[a + 3], d.qpos[a + 4], d.qpos[a + 5], d.qpos[a + 6]],
      cubeLinvel: [d.qvel[v], d.qvel[v + 1], d.qvel[v + 2]],
      platePos: [d.mocap_pos[p], d.mocap_pos[p + 1], d.mocap_pos[p + 2]],
      eePos: [d.site_xpos[3 * this.eeSite], d.site_xpos[3 * this.eeSite + 1], d.site_xpos[3 * this.eeSite + 2]],
      touchFixed,
      touchMoving,
    };
  }

  // Mirrors keiarm/task.py is_success (the instantaneous condition).
  isSuccessNow() {
    const s = this.state();
    const dx = s.cubePos[0] - s.platePos[0];
    const dy = s.cubePos[1] - s.platePos[1];
    const restZ = s.platePos[2] + this.c.plate_height / 2 + this.scene.cube_size / 2;
    const speed = Math.hypot(...s.cubeLinvel);
    return Math.hypot(dx, dy) < this.scene.plate_radius
      && Math.abs(s.cubePos[2] - restZ) < this.c.rest_z_tol
      && speed < this.c.rest_speed && !s.touchFixed && !s.touchMoving;
  }

  // Put the cube flat on the table at (x, y), at rest (drag interaction and scripted disturbances).
  placeCube(x, y, yaw = 0) {
    const d = this.data;
    const a = this.cubeQadr;
    d.qpos[a] = x;
    d.qpos[a + 1] = y;
    d.qpos[a + 2] = this.scene.cube_size / 2 + 1e-3;
    d.qpos[a + 3] = Math.cos(yaw / 2);
    d.qpos[a + 4] = 0;
    d.qpos[a + 5] = 0;
    d.qpos[a + 6] = Math.sin(yaw / 2);
    for (let k = 0; k < 6; k++) d.qvel[this.cubeDadr + k] = 0;
    this.mujoco.mj_forward(this.model, d);
  }

  // Overwrite the full state (used by the render comparison tool), then recompute poses.
  setState(qpos, mocapPos) {
    this.data.qpos.set(qpos);
    this.data.mocap_pos.set(mocapPos);
    this.mujoco.mj_forward(this.model, this.data);
  }

  // WASM objects are not garbage collected: free them explicitly.
  dispose() {
    this.data.delete();
    this.model.delete();
  }
}

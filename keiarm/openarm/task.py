"""Task "hand the bar over and put it on the plate": state readout, success, stages.

Privileged simulator state only. The policy never sees it.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from keiarm.openarm.robot import ACTUATOR_NAMES, PINCH_OFFSET, SIDES, ee_site, finger_joints
from keiarm.openarm.scene import PLATE_HEIGHT, SceneParams

STAGES = ("none", "reach", "grasp_right", "lift", "handover", "left_carry", "over_plate", "success")
REST_Z_TOL = 0.004  # bar centre height vs. lying flat on the plate [m]
REST_TILT = np.radians(5.0)
REST_SPEED = 0.01  # [m/s]


class TaskIndex:
    """Model addresses needed to read the task state."""

    def __init__(self, model: mujoco.MjModel):
        self.act = np.array([model.actuator(n).id for n in ACTUATOR_NAMES])
        self.trn_qadr = np.array([model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in self.act])
        self.bar_qadr = model.joint("bar").qposadr[0]
        self.bar_dadr = model.joint("bar").dofadr[0]
        self.bar_geom = model.geom("bar").id
        self.plate_mocap = model.body("plate").mocapid[0]
        self.site = {s: model.site(ee_site(s)).id for s in SIDES}
        self.finger_qadr = {s: model.joint(finger_joints(s)[0]).qposadr[0] for s in SIDES}
        self.finger_dadr = {s: model.joint(finger_joints(s)[0]).dofadr[0] for s in SIDES}
        self.finger_geoms = {}
        for s in SIDES:
            for part in ("inner", "outer"):
                body = model.body(f"openarm_{s}_ee_{part}_finger").id
                self.finger_geoms[(s, part)] = {g for g in range(model.ngeom) if model.geom_bodyid[g] == body}
        self.arm_geoms = {s: {g for g in range(model.ngeom)
                              if model.body(model.geom_bodyid[g]).name.startswith(f"openarm_{s}_")
                              and model.geom_contype[g]} for s in SIDES}


@dataclass
class TaskState:
    ctrl_joints: np.ndarray  # (16,) joint angles in ACTUATOR_NAMES order (finger joint 1 for grippers)
    bar_pos: np.ndarray
    bar_mat: np.ndarray  # (3, 3); column 1 = long axis
    bar_linvel: np.ndarray
    plate_pos: np.ndarray  # centre of the plate disc
    pinch_pos: dict[str, np.ndarray]
    pinch_mat: dict[str, np.ndarray]  # (3, 3); column 1 = jaw closing direction, column 2 = up
    finger_q: dict[str, float]
    finger_qvel: dict[str, float]
    touch: dict[tuple[str, str], bool]  # (side, "inner"/"outer") -> touches the bar
    arms_touch: bool  # a left-arm geom touches a right-arm geom

    def held(self, side: str) -> bool:
        return self.touch[(side, "inner")] and self.touch[(side, "outer")]

    def touches(self, side: str) -> bool:
        return self.touch[(side, "inner")] or self.touch[(side, "outer")]

    @property
    def plate_top(self) -> float:
        return float(self.plate_pos[2] + PLATE_HEIGHT / 2)

    @property
    def bar_axis(self) -> np.ndarray:
        """Horizontal unit long axis, oriented towards +y (the left hand's end)."""
        a = self.bar_mat[:2, 1].copy()
        n = np.linalg.norm(a)
        a = a / n if n > 1e-6 else np.array([0.0, 1.0])
        return a if a[1] >= 0 else -a

    @property
    def bar_tilt(self) -> float:
        """Angle between the bar's long axis and the table plane."""
        return float(np.arcsin(min(1.0, abs(self.bar_mat[2, 1]))))


def read_state(model: mujoco.MjModel, ix: TaskIndex, data: mujoco.MjData) -> TaskState:
    """Read the task state from `data` without modifying it (derived quantities are those
    of the last physics substep)."""
    a = ix.bar_qadr
    bar_mat = np.empty(9)
    mujoco.mju_quat2Mat(bar_mat, data.qpos[a + 3 : a + 7])
    touch = dict.fromkeys(ix.finger_geoms, False)
    arms_touch = False
    for c in data.contact[: data.ncon]:
        if c.dist > 0:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        other = g2 if g1 == ix.bar_geom else g1 if g2 == ix.bar_geom else -1
        if other >= 0:
            for key, geoms in ix.finger_geoms.items():
                touch[key] |= other in geoms
        if (g1 in ix.arm_geoms["left"] and g2 in ix.arm_geoms["right"]) or (
                g2 in ix.arm_geoms["left"] and g1 in ix.arm_geoms["right"]):
            arms_touch = True
    offset = np.array(PINCH_OFFSET)
    pinch_pos, pinch_mat = {}, {}
    for s in SIDES:
        rot = data.site_xmat[ix.site[s]].reshape(3, 3).copy()
        pinch_mat[s] = rot
        pinch_pos[s] = data.site_xpos[ix.site[s]] + rot @ offset
    return TaskState(
        ctrl_joints=data.qpos[ix.trn_qadr].copy(),
        bar_pos=data.qpos[a : a + 3].copy(), bar_mat=bar_mat.reshape(3, 3),
        bar_linvel=data.qvel[ix.bar_dadr : ix.bar_dadr + 3].copy(),
        plate_pos=data.mocap_pos[ix.plate_mocap].copy(),
        pinch_pos=pinch_pos, pinch_mat=pinch_mat,
        finger_q={s: float(data.qpos[ix.finger_qadr[s]]) for s in SIDES},
        finger_qvel={s: float(data.qvel[ix.finger_dadr[s]]) for s in SIDES},
        touch=touch, arms_touch=arms_touch,
    )


def over_plate(st: TaskState, scene: SceneParams, margin: float = 0.0) -> bool:
    """Bar centre is inside the plate's radius (seen from above)."""
    return bool(np.linalg.norm(st.bar_pos[:2] - st.plate_pos[:2]) < scene.plate_radius - margin)


def rests_on_plate(st: TaskState, scene: SceneParams, z_tol: float = REST_Z_TOL) -> bool:
    return (over_plate(st, scene) and st.bar_tilt < REST_TILT
            and abs(st.bar_pos[2] - (st.plate_top + scene.bar_size[2] / 2)) < z_tol)


def is_success(st: TaskState, scene: SceneParams) -> bool:
    """Bar centre inside the plate radius, lying flat on it, at rest, no finger touching it."""
    return (rests_on_plate(st, scene) and float(np.linalg.norm(st.bar_linvel)) < REST_SPEED
            and not st.touches("left") and not st.touches("right"))


def stage_of(st: TaskState, scene: SceneParams) -> int:
    """Instantaneous progress stage (index into STAGES). Track the max over an episode."""
    if is_success(st, scene):
        return 7
    lifted = st.bar_pos[2] > scene.bar_size[2] / 2 + 0.02
    if st.held("left") and not st.touches("right") and over_plate(st, scene):
        return 6
    if st.held("left") and not st.touches("right") and lifted:
        return 5
    if st.held("left") and st.held("right"):
        return 4
    if st.held("right") and lifted:
        return 3
    if st.held("right"):
        return 2
    if np.linalg.norm(st.pinch_pos["right"] - st.bar_pos) < 0.10:
        return 1
    return 0

"""Task definition for "pick the cube, place it on the plate": state readout, success, stages.

Everything here uses privileged simulator state. The policy never sees it.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from keiarm.robot import EE_SITE, JOINT_NAMES
from keiarm.scene import PLATE_HEIGHT, SceneParams

STAGES = ("none", "reach", "grasp", "lift", "over_plate", "success")
REST_Z_TOL = 0.003  # cube centre height vs. resting on the plate [m]
REST_SPEED = 0.01  # [m/s]


class TaskIndex:
    """Model addresses needed to read the task state."""

    def __init__(self, model: mujoco.MjModel):
        self.qadr = np.array([model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.act = np.array([model.actuator(n).id for n in JOINT_NAMES])
        self.cube_qadr = model.joint("cube").qposadr[0]
        self.cube_dadr = model.joint("cube").dofadr[0]
        self.cube_body = model.body("cube").id
        self.cube_geom = model.geom("cube").id
        self.plate_mocap = model.body("plate").mocapid[0]
        self.ee = model.site(EE_SITE).id
        fixed = {model.body("gripper").id, model.body("camera_mount").id}
        moving = {model.body("moving_jaw_so101_v1").id}
        self.fixed_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in fixed}
        self.moving_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in moving}


@dataclass
class TaskState:
    joints: np.ndarray  # (6,) [rad], JOINT_NAMES order
    cube_pos: np.ndarray
    cube_mat: np.ndarray  # (3, 3)
    cube_linvel: np.ndarray
    plate_pos: np.ndarray  # centre of the plate disc
    ee_pos: np.ndarray
    ee_mat: np.ndarray  # (3, 3); columns: approach, hinge, jaw opening
    touch_fixed: bool  # cube touches the fixed jaw
    touch_moving: bool  # cube touches the moving jaw

    @property
    def held(self) -> bool:
        return self.touch_fixed and self.touch_moving

    @property
    def plate_top(self) -> float:
        return float(self.plate_pos[2] + PLATE_HEIGHT / 2)


def read_state(model: mujoco.MjModel, ix: TaskIndex, data: mujoco.MjData) -> TaskState:
    """Read the task state from `data` without modifying it.

    Derived quantities (site pose, contacts) are those of the last physics substep.
    """
    a = ix.cube_qadr
    cube_mat = np.empty(9)
    mujoco.mju_quat2Mat(cube_mat, data.qpos[a + 3 : a + 7])
    touch_fixed = touch_moving = False
    for c in data.contact:
        if c.dist > 0:
            continue
        other = c.geom2 if c.geom1 == ix.cube_geom else c.geom1 if c.geom2 == ix.cube_geom else -1
        touch_fixed |= other in ix.fixed_geoms
        touch_moving |= other in ix.moving_geoms
    return TaskState(
        joints=data.qpos[ix.qadr].copy(),
        cube_pos=data.qpos[a : a + 3].copy(), cube_mat=cube_mat.reshape(3, 3),
        cube_linvel=data.qvel[ix.cube_dadr : ix.cube_dadr + 3].copy(),
        plate_pos=data.mocap_pos[ix.plate_mocap].copy(),
        ee_pos=data.site_xpos[ix.ee].copy(), ee_mat=data.site_xmat[ix.ee].reshape(3, 3).copy(),
        touch_fixed=touch_fixed, touch_moving=touch_moving,
    )


def over_plate(st: TaskState, scene: SceneParams, margin: float = 0.0) -> bool:
    """Cube centre is inside the plate's radius (seen from above)."""
    return bool(np.linalg.norm(st.cube_pos[:2] - st.plate_pos[:2]) < scene.plate_radius - margin)


def rests_on_plate(st: TaskState, scene: SceneParams, z_tol: float = REST_Z_TOL) -> bool:
    return over_plate(st, scene) and abs(st.cube_pos[2] - (st.plate_top + scene.cube_size / 2)) < z_tol


def is_success(st: TaskState, scene: SceneParams) -> bool:
    """Design doc 10.1: cube inside the plate radius, resting on it, gripper not touching it."""
    return (
        rests_on_plate(st, scene)
        and float(np.linalg.norm(st.cube_linvel)) < REST_SPEED
        and not st.touch_fixed
        and not st.touch_moving
    )


def stage_of(st: TaskState, scene: SceneParams) -> int:
    """Instantaneous progress stage (index into STAGES). Track the max over an episode."""
    if is_success(st, scene):
        return 5
    lifted = st.cube_pos[2] > scene.cube_size / 2 + 0.015
    if st.held and over_plate(st, scene):
        return 4
    if st.held and lifted:
        return 3
    if st.held:
        return 2
    if np.linalg.norm(st.ee_pos - st.cube_pos) < 0.04:
        return 1
    return 0

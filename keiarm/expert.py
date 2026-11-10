"""Privileged scripted expert for pick-and-place.

The expert is a pure function of the simulator state: `act(data)` looks at where the cube,
plate and gripper are right now, decides which phase that corresponds to, and returns joint
position targets (the policy's action space). It keeps no memory between calls, so it can
label any state a learner wanders into (DAgger) and recovers from slips and drops by
construction.

Targets are "carrots": at most a few millimetres from the current pose towards the goal,
and exactly the goal once it is within one step. Goals are absolute or relative to the
cube/plate, never "stay where you are" in the vertical direction (a position-controlled arm
commanded to its own sagging pose slowly sinks).
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from keiarm.ik import ArmIK
from keiarm.scene import PAN_AXIS_XY, SceneParams
from keiarm.task import TaskIndex, TaskState, over_plate, read_state, rests_on_plate

PHASES = ("APPROACH", "DESCEND", "CLOSE", "LIFT", "MOVE", "LOWER", "OPEN", "RETREAT")

Q_OPEN = 0.9  # gripper joint target when open [rad] (jaw gap about 85 mm)
Q_CLOSED = -0.17
FIXED_JAW_Z = -0.020  # fixed fingertip in the gripper site frame, along the opening axis [m]
JAW_CLEARANCE = 0.010  # gap left between the fixed jaw and the cube face before closing
GRASP_HEIGHT = 0.010  # site height above the cube's bottom face when grasping
HOVER_CLEARANCE = 0.035  # site height above the cube's top face when hovering
CARRY_LIFT = 0.025  # how far above its resting-on-plate height the cube is carried

STEP_FAST = 0.008  # carrot length per control step [m] (0.24 m/s at 30 Hz)
STEP_SLOW = 0.005


@dataclass
class ExpertInfo:
    phase: str
    target_pos: np.ndarray


def _carrot(pos: np.ndarray, goal: np.ndarray, step: float) -> np.ndarray:
    d = goal - pos
    n = float(np.linalg.norm(d))
    return goal.copy() if n <= step else pos + d * (step / n)


class Expert:
    def __init__(self, model: mujoco.MjModel, scene: SceneParams):
        self.model = model
        self.scene = scene
        self.ix = TaskIndex(model)
        self.ik = ArmIK(model)

    # --- geometry helpers --------------------------------------------------------------

    def _grasp(self, st: TaskState) -> tuple[np.ndarray, np.ndarray]:
        """Site position and jaw opening direction for a top-down grasp of the cube.

        The jaws open along the horizontal cube face normal closest to the radial direction
        (robot -> cube), so the fixed jaw ends up on the robot's side of the cube.
        """
        radial = st.cube_pos[:2] - np.array(PAN_AXIS_XY)
        radial /= np.linalg.norm(radial)
        best, best_dot = radial, -np.inf
        for axis in st.cube_mat.T:
            if abs(axis[2]) > 0.5:
                continue  # this cube axis points up/down
            h = axis[:2] / np.linalg.norm(axis[:2])
            for cand in (h, -h):
                if cand @ radial > best_dot:
                    best, best_dot = cand, float(cand @ radial)
        s = self.scene.cube_size
        z_off = FIXED_JAW_Z + s / 2 + JAW_CLEARANCE  # cube centre along the opening axis, site frame
        pos = np.array([*(st.cube_pos[:2] - z_off * best), st.cube_pos[2] - s / 2 + GRASP_HEIGHT])
        return pos, best

    # --- phase logic -------------------------------------------------------------------

    def plan(self, st: TaskState) -> tuple[str, np.ndarray, np.ndarray | None, float]:
        """Return (phase, site goal, jaw direction or None, gripper target) for a state."""
        scene, s = self.scene, self.scene.cube_size
        ee, grip = st.ee_pos, st.joints[5]
        rest_z = st.plate_top + s / 2  # cube centre height when resting on the plate
        hover_z = s + HOVER_CLEARANCE

        # 1. Cube already sits on the plate: let go, then back off upwards.
        if rests_on_plate(st, scene, z_tol=0.006) and over_plate(st, scene, margin=0.005):
            near = np.linalg.norm(ee[:2] - st.cube_pos[:2]) < 0.04 and ee[2] < st.plate_top + s + 0.01
            if near and (st.held or grip < Q_OPEN - 0.2):
                goal = np.array([ee[0], ee[1], st.plate_top + GRASP_HEIGHT])
                return "OPEN", _carrot(ee, goal, STEP_SLOW), None, Q_OPEN
            goal = np.array([ee[0], ee[1], st.plate_top + hover_z])
            return "RETREAT", _carrot(ee, goal, STEP_FAST), None, Q_OPEN

        # 2. Cube is in the gripper: lift, carry over the plate, lower. Servo on the cube itself.
        if st.held:
            carry_z = rest_z + CARRY_LIFT
            to_plate = st.plate_pos[:2] - st.cube_pos[:2]
            low = st.cube_pos[2] < carry_z - 0.008
            if np.linalg.norm(to_plate) < (0.020 if low else 0.012):
                phase, step = "LOWER", STEP_SLOW
                cube_goal = np.array([*st.plate_pos[:2], rest_z + 0.001])
            elif low:
                phase, step = "LIFT", STEP_FAST
                cube_goal = np.array([*st.cube_pos[:2], carry_z])
            else:
                phase, step = "MOVE", STEP_FAST
                cube_goal = np.array([*st.plate_pos[:2], carry_z])
            return phase, _carrot(ee, ee + (cube_goal - st.cube_pos), step), None, Q_CLOSED

        # 3. Cube is somewhere else and free: go and grasp it.
        grasp, jaw_dir = self._grasp(st)
        jaw_now = st.ee_mat[:2, 2]
        yaw_err = abs(np.arctan2(jaw_now[0] * jaw_dir[1] - jaw_now[1] * jaw_dir[0], jaw_now @ jaw_dir))
        height = ee[2] - grasp[2]
        # Funnel: tolerate more sideways error the higher the gripper still is.
        aligned = (np.linalg.norm(ee[:2] - grasp[:2]) < 0.007 + 0.15 * max(height, 0.0)
                   and yaw_err < np.radians(10.0))
        if aligned and height < 0.008:
            return "CLOSE", grasp, jaw_dir, Q_CLOSED
        if aligned and grip > Q_OPEN - 0.15:
            return "DESCEND", _carrot(ee, grasp, STEP_SLOW), jaw_dir, Q_OPEN
        hover = np.array([grasp[0], grasp[1], st.cube_pos[2] - s / 2 + hover_z])
        if ee[2] < st.cube_pos[2] + s / 2 + 0.012:
            hover[:2] = ee[:2]  # fingertips below the cube's top: rise in place before travelling
        return "APPROACH", _carrot(ee, hover, STEP_FAST), jaw_dir, Q_OPEN

    def act(self, data: mujoco.MjData) -> tuple[np.ndarray, ExpertInfo]:
        """Joint position targets (6,) for the current state. Does not modify `data`."""
        st = read_state(self.model, self.ix, data)
        phase, target, jaw_dir, grip = self.plan(st)
        q = self.ik.solve_topdown(target, data.qpos, jaw_dir).q
        action = q[self.ix.qadr]
        action[5] = grip
        return action, ExpertInfo(phase, target)

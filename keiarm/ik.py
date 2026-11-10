"""Arm inverse kinematics for the SO-101 (mink, differential IK iterated to convergence).

The gripper site frame (`gripperframe`) has +x pointing out through the fingertips (the
approach direction), z along the jaw opening direction (fixed jaw at -z, moving jaw
towards +z) and y along the jaw hinge. A top-down grasp therefore means site +x = world -z.

With 5 arm joints the arm can reach a position (3) with a chosen approach direction lying
in the arm's vertical plane (1) and a chosen roll about it (1); nothing else is free.
"""

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from keiarm.robot import ARM_JOINT_NAMES, EE_SITE

DOWN = np.array([0.0, 0.0, -1.0])
_DT = 1.0  # one iteration = one damped Gauss-Newton step


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray  # full qpos (gripper joint untouched)
    pos_err: float  # [m]
    tilt: float  # angle between site +x and the requested approach direction [rad]
    iters: int


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to a.
        p = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0.0, 1.0, 0.0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s**2)


def _quat(mat: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, mat.reshape(9))
    return q


class ArmIK:
    """Reusable IK solver bound to one model (keeps its own scratch MjData)."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.config = mink.Configuration(model)
        self.limits = [mink.ConfigurationLimit(model)]
        self.site = model.site(EE_SITE).id
        self.grip_dof = model.joint("gripper").dofadr[0]
        self.roll_dof = model.joint("wrist_roll").dofadr[0]
        self.roll_adr = model.joint("wrist_roll").qposadr[0]
        self.roll_range = model.jnt_range[model.joint("wrist_roll").id].copy()

    def solve(
        self,
        target_pos: np.ndarray,
        q_init: np.ndarray,
        approach: np.ndarray | None = DOWN,
        orientation_cost: float = 1.0,
        freeze_roll: bool = False,
        max_iters: int = 100,
        pos_tol: float = 5e-4,
        tilt_tol: float = np.radians(0.5),
    ) -> IKResult:
        """Move the gripper site to `target_pos`, optionally aligning site +x with `approach`.

        Roll about the approach axis is left free: each iteration targets the orientation
        closest to the current one that satisfies the approach constraint.
        """
        config = self.config
        config.update(np.asarray(q_init, dtype=float))
        use_ori = approach is not None
        task = mink.FrameTask(
            EE_SITE, "site", position_cost=1.0,
            orientation_cost=orientation_cost if use_ori else 0.0, lm_damping=1e-4,
        )
        sid = self.site
        target_pos = np.asarray(target_pos, dtype=float)

        def errors() -> tuple[float, float, np.ndarray]:
            rot = config.data.site_xmat[sid].reshape(3, 3)
            pos_err = float(np.linalg.norm(config.data.site_xpos[sid] - target_pos))
            tilt = float(np.arccos(np.clip(rot[:, 0] @ approach, -1, 1))) if use_ori else 0.0
            return pos_err, tilt, rot

        it = 0
        for it in range(max_iters):
            pos_err, tilt, rot = errors()
            if pos_err < pos_tol and tilt < tilt_tol:
                break
            target_rot = _rotation_between(rot[:, 0], approach) @ rot if use_ori else rot
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(_quat(target_rot)), target_pos))
            vel = mink.solve_ik(config, [task], _DT, "daqp", limits=self.limits)
            vel[self.grip_dof] = 0.0
            vel[6:] = 0.0  # anything beyond the robot (e.g. the cube's free joint)
            if freeze_roll:
                vel[self.roll_dof] = 0.0
            if np.linalg.norm(vel) < 1e-5:  # stalled (limit or unreachable)
                break
            # Cap the step so far-away targets do not fling the arm through its limits.
            vel *= min(1.0, 0.5 / max(np.abs(vel).max(), 1e-12))
            config.integrate_inplace(vel, _DT)
        pos_err, tilt, _ = errors()
        return IKResult(config.q.copy(), pos_err, tilt, it)

    def jaw_yaw(self, q: np.ndarray) -> float:
        """World yaw of the jaw opening direction (site +z projected on the table) at q."""
        self.config.update(q)
        z = self.config.data.site_xmat[self.site].reshape(3, 3)[:, 2]
        return float(np.arctan2(z[1], z[0]))

    def solve_topdown(
        self,
        target_pos: np.ndarray,
        q_init: np.ndarray,
        jaw_dir: np.ndarray | None = None,
        max_roll_step: float = 0.25,
        max_iters: int = 30,
    ) -> IKResult:
        """Top-down pose at `target_pos`. If `jaw_dir` (horizontal unit vector) is given, turn the
        wrist roll towards making the jaws open along it, by at most `max_roll_step` rad."""
        r = self.solve(target_pos, q_init, freeze_roll=True, max_iters=max_iters)
        if jaw_dir is None:
            return r
        q = r.q.copy()
        yaw = self.jaw_yaw(q)
        probe = q.copy()
        probe[self.roll_adr] += 0.02
        sign = np.sign(wrap_angle(self.jaw_yaw(probe) - yaw)) or 1.0
        err = wrap_angle(np.arctan2(jaw_dir[1], jaw_dir[0]) - yaw)
        q[self.roll_adr] = np.clip(q[self.roll_adr] + sign * np.clip(err, -max_roll_step, max_roll_step),
                                   *self.roll_range)
        return self.solve(target_pos, q, freeze_roll=True, max_iters=max_iters)


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def solve_arm_ik(model: mujoco.MjModel, target_pos: np.ndarray, q_init: np.ndarray, **kwargs) -> IKResult:
    """One-off convenience wrapper around ArmIK.solve."""
    return ArmIK(model).solve(target_pos, q_init, **kwargs)


def initial_guesses(model: mujoco.MjModel, target_pos: np.ndarray) -> list[np.ndarray]:
    """A few elbow-up starting poses with the shoulder already panned towards the target."""
    pan_anchor = model.body("shoulder").pos[:2]
    # shoulder_pan is mounted upside down: positive joint angle turns towards -y.
    pan = -np.arctan2(target_pos[1] - pan_anchor[1], target_pos[0] - pan_anchor[0])
    lo, hi = model.jnt_range[model.joint("shoulder_pan").id]
    pan = float(np.clip(pan, lo, hi))
    guesses = []
    for lift, elbow, flex in [(-0.3, 0.6, 1.2), (0.6, -0.2, 1.2), (-1.0, 1.3, 1.0)]:
        q = np.zeros(model.nq)
        for name, v in zip(ARM_JOINT_NAMES, (pan, lift, elbow, flex, 0.0)):
            q[model.joint(name).qposadr[0]] = v
        guesses.append(q)
    return guesses

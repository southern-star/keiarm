"""Damped least-squares IK for one OpenArm arm, on the pinch point between the pads.

Solves on a scratch MjData (kinematics only) so the simulation state is never touched.
"""

import mujoco
import numpy as np

from keiarm.openarm.robot import PINCH_OFFSET, arm_joints, ee_site


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def topdown_rotation(yaw: float) -> np.ndarray:
    """ee_site orientation with the fingers pointing straight down (site +z = world +z;
    the fingertips lie along the site's -z) and the jaws closing along rot_z(yaw) @ y."""
    return rot_z(yaw)


class ArmIK:
    def __init__(self, model: mujoco.MjModel, side: str, damping: float = 1e-3):
        self.model = model
        self.side = side
        self.scratch = mujoco.MjData(model)
        ids = [model.joint(n).id for n in arm_joints(side)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in ids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in ids])
        self.lo, self.hi = model.jnt_range[ids, 0], model.jnt_range[ids, 1]
        self.site = model.site(ee_site(side)).id
        self.offset = np.array(PINCH_OFFSET)
        self.damping = damping

    def pinch_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(position, rotation) of the pinch point, from data's current kinematics."""
        rot = data.site_xmat[self.site].reshape(3, 3)
        return data.site_xpos[self.site] + rot @ self.offset, rot

    def solve(self, qpos: np.ndarray, q_arm: np.ndarray, target_pos, target_rot,
              iters: int = 100, step: float = 0.5, tol_pos: float = 2e-4, tol_rot: float = 2e-3,
              q_rest=None, rest_gain: float = 0.1):
        """Return (q_arm, position error [m], rotation error [rad]) starting from q_arm.

        qpos supplies the rest of the configuration (other arm, objects). With q_rest, each
        iteration also moves the arm towards q_rest inside the Jacobian's null space (the
        7th degree of freedom, the elbow's swivel), which keeps it away from joint limits
        where a warm-started solve otherwise stalls.
        """
        m, d = self.model, self.scratch
        d.qpos[:] = qpos
        d.qpos[self.qadr] = q_arm
        target_pos = np.asarray(target_pos, float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos, rot = self.pinch_pose(d)
            e_pos = target_pos - pos
            e_rot = 0.5 * sum(np.cross(rot[:, i], target_rot[:, i]) for i in range(3))
            if np.linalg.norm(e_pos) < tol_pos and np.linalg.norm(e_rot) < tol_rot:
                break
            mujoco.mj_jac(m, d, jp, jr, pos, m.site_bodyid[self.site])
            jac = np.vstack([jp[:, self.vadr], jr[:, self.vadr]])
            err = np.concatenate([e_pos, e_rot])
            pinv = jac.T @ np.linalg.inv(jac @ jac.T + self.damping * np.eye(6))
            dq = pinv @ err
            if q_rest is not None:
                dq += (np.eye(len(self.qadr)) - pinv @ jac) @ (rest_gain * (q_rest - d.qpos[self.qadr]))
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + step * dq, self.lo, self.hi)
        else:
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos, rot = self.pinch_pose(d)
            e_pos = target_pos - pos
            e_rot = 0.5 * sum(np.cross(rot[:, i], target_rot[:, i]) for i in range(3))
        return d.qpos[self.qadr].copy(), float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))

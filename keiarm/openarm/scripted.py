"""Open-loop scripted grasp used to check the physics (Phase 0). Not the expert.

One arm moves its pinch point above the object, descends, closes, and lifts. Joint
targets come from ArmIK on a carrot that advances at most CARROT [m] per control step.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from keiarm import config
from keiarm.openarm.ik import ArmIK, topdown_rotation
from keiarm.openarm.robot import ARM_DOF, GRIPPER_CLOSED, GRIPPER_OPEN

CARROT = 0.008


@dataclass
class GraspResult:
    lift: float  # cube height gain at the end [m]
    arm_table_contact_steps: int  # control steps ending with an arm geom touching the table
    max_tracking_error: float  # max |pinch - carrot| over the motion [m]
    log: list[str] = field(default_factory=list)


def _arm_touches(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> bool:
    for c in data.contact[: data.ncon]:
        if geom_id in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == geom_id else c.geom1
            if model.body(model.geom_bodyid[other]).name.startswith("openarm_"):
                return True
    return False


# Lowest pinch height: the fingertips reach 2 cm below the pinch point.
MIN_PINCH_Z = 0.024
YAW_STEP = np.radians(3.0)  # max change of the commanded jaw yaw per control step


def scripted_grasp(model: mujoco.MjModel, data: mujoco.MjData, side: str, grasp_xy, jaw_yaw: float,
                   grasp_z: float, body: str = "bar", lift_height: float = 0.15) -> GraspResult:
    """Grasp `body` from above at grasp_xy (pinch height grasp_z) with the jaws closing along
    rot_z(jaw_yaw) @ y, then lift it by lift_height."""
    ik = ArmIK(model, side)
    act = np.array([model.actuator(f"{side}_joint{i}_ctrl").id for i in range(1, ARM_DOF + 1)])
    grip = model.actuator(f"{side}_finger1_ctrl").id
    obj = model.body(body).id
    table = model.geom("table").id

    mujoco.mj_forward(model, data)
    carrot, rot = ik.pinch_pose(data)
    carrot = carrot.copy()
    yaw = float(np.arctan2(-rot[0, 1], rot[1, 1]))  # jaw direction = rot[:, 1] = (-sin, cos)
    # the jaws are symmetric: pick the equivalent target yaw nearest to the current one
    jaw_yaw = jaw_yaw + np.pi * np.round((yaw - jaw_yaw) / np.pi)
    q_cmd = data.qpos[ik.qadr].copy()
    z0 = float(data.xpos[obj][2])
    res = GraspResult(lift=0.0, arm_table_contact_steps=0, max_tracking_error=0.0)

    at = np.array([*grasp_xy, max(grasp_z, MIN_PINCH_Z)])
    plan = [
        ("above", at + [0, 0, 0.10], GRIPPER_OPEN[side], 75),
        ("down", at, GRIPPER_OPEN[side], 45),
        ("close", at, GRIPPER_CLOSED[side], 25),
        ("lift", at + [0, 0, lift_height], GRIPPER_CLOSED[side], 60),
        ("hold", at + [0, 0, lift_height], GRIPPER_CLOSED[side], 30),
    ]
    for name, target, gripper, steps in plan:
        for _ in range(steps):
            delta = target - carrot
            dist = np.linalg.norm(delta)
            carrot = carrot + (delta if dist < CARROT else delta / dist * CARROT)
            yaw += float(np.clip(jaw_yaw - yaw, -YAW_STEP, YAW_STEP))
            q_cmd, _, _ = ik.solve(data.qpos, q_cmd, carrot, topdown_rotation(yaw), iters=20)
            data.ctrl[act] = q_cmd
            data.ctrl[grip] = gripper
            mujoco.mj_step(model, data, nstep=config.SUBSTEPS)
            res.arm_table_contact_steps += _arm_touches(model, data, table)
            res.max_tracking_error = max(res.max_tracking_error,
                                         float(np.linalg.norm(ik.pinch_pose(data)[0] - carrot)))
        res.log.append(f"{name}: pinch error {np.linalg.norm(ik.pinch_pose(data)[0] - target) * 1000:.1f} mm, "
                       f"{body} z {data.xpos[obj][2]:.3f}")
    res.lift = float(data.xpos[obj][2] - z0)
    return res

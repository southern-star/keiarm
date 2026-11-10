"""Phase 0 check: can mink solve position-only IK for the SO-101 gripper site?

Kinematics only (no physics). Usage: uv run python tools/check_ik.py
"""

import mink
import mujoco
import numpy as np

from keiarm.robot import ARM_JOINT_NAMES, EE_SITE, so101_spec

DT = 1 / 30
MAX_ITERS = 300
TOL = 1e-3  # 1 mm

# World-frame targets for the gripper site (robot base at origin, +x forward, +z up).
TARGETS = [
    (0.25, 0.00, 0.10),
    (0.25, 0.10, 0.05),
    (0.25, -0.10, 0.05),
    (0.15, 0.15, 0.05),
    (0.30, 0.00, 0.03),
    (0.35, 0.10, 0.03),
    (0.20, 0.00, 0.25),
    (0.45, 0.00, 0.05),  # near / beyond full reach
    (0.60, 0.00, 0.05),  # unreachable on purpose
]


def solve(model: mujoco.MjModel, target: np.ndarray) -> tuple[float, int, np.ndarray]:
    config = mink.Configuration(model)
    q0 = np.zeros(model.nq)
    q0[model.joint("shoulder_lift").qposadr[0]] = -0.5
    q0[model.joint("elbow_flex").qposadr[0]] = 0.8
    q0[model.joint("wrist_flex").qposadr[0]] = 0.8
    config.update(q0)
    ee = mink.FrameTask(EE_SITE, "site", position_cost=1.0, orientation_cost=0.0, lm_damping=1e-3)
    ee.set_target(mink.SE3.from_translation(target))
    posture = mink.PostureTask(model, cost=1e-3)
    posture.set_target(q0)
    limits = [mink.ConfigurationLimit(model)]
    sid = model.site(EE_SITE).id
    err = np.inf
    for i in range(MAX_ITERS):
        err = float(np.linalg.norm(config.data.site_xpos[sid] - target))
        if err < TOL:
            return err, i, config.q.copy()
        vel = mink.solve_ik(config, [ee, posture], DT, "daqp", limits=limits)
        vel[model.joint("gripper").dofadr[0]] = 0.0  # jaw is not part of the arm IK
        config.integrate_inplace(vel, DT)
    return err, MAX_ITERS, config.q.copy()


def main() -> None:
    model = so101_spec().compile()
    ok = 0
    for t in TARGETS:
        err, iters, q = solve(model, np.array(t))
        in_limits = bool(np.all(q >= model.jnt_range[:, 0] - 1e-6) and np.all(q <= model.jnt_range[:, 1] + 1e-6))
        status = "OK  " if err < TOL else "FAIL"
        ok += err < TOL
        arm = [round(float(q[model.joint(n).qposadr[0]]), 2) for n in ARM_JOINT_NAMES]
        print(f"{status} target={t} err={err * 1000:7.2f} mm iters={iters:3d} in_limits={in_limits} q={arm}")
    print(f"{ok}/{len(TARGETS)} solved to < {TOL * 1000:.0f} mm")


if __name__ == "__main__":
    main()

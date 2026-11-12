"""Scripted top-down grasp+lift of the puck with the right arm; compare timesteps and gravcomp."""

import time

import mujoco
import numpy as np

SCENE = "runs/spike/openarm_mujoco/v2/pedestal/move_puck_scene.xml"
PINCH = np.array([0.0, 0.0, -0.150])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build(ts, gc):
    spec = mujoco.MjSpec.from_file(SCENE)
    spec.option.timestep = ts
    if gc:
        for b in spec.bodies:
            if b.name.startswith("openarm_"):
                b.gravcomp = 1.0
        for j in spec.joints:
            if j.name.startswith("openarm_"):
                j.actgravcomp = True
    return spec.compile()


class IK:
    def __init__(s, m, side="right"):
        s.m = m
        s.d = mujoco.MjData(m)
        arm = [m.joint(f"openarm_{side}_joint{i}").id for i in range(1, 8)]
        s.qa = np.array([m.jnt_qposadr[j] for j in arm])
        s.va = np.array([m.jnt_dofadr[j] for j in arm])
        s.lo, s.hi = m.jnt_range[arm, 0], m.jnt_range[arm, 1]
        s.sid = m.site(f"{side}_ee_control_point").id

    def pinch(s, d):
        R = d.site_xmat[s.sid].reshape(3, 3)
        return d.site_xpos[s.sid] + R @ PINCH, R

    def solve(s, qpos, q0, target, Rt, it=100):
        m, d = s.m, s.d
        d.qpos[:] = qpos
        d.qpos[s.qa] = q0
        for _ in range(it):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p, R = s.pinch(d)
            ep = target - p
            er = 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            jp = np.zeros((3, m.nv))
            jr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, p, m.site_bodyid[s.sid])
            J = np.vstack([jp[:, s.va], jr[:, s.va]])
            e = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            d.qpos[s.qa] = np.clip(d.qpos[s.qa] + 0.5 * dq, s.lo, s.hi)
        return d.qpos[s.qa].copy(), np.linalg.norm(ep), np.linalg.norm(er)


def run(ts, gc):
    m = build(ts, gc)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    ik = IK(m)
    act = [m.actuator(f"right_joint{i}_ctrl").id for i in range(1, 8)]
    grip = m.actuator("right_finger1_ctrl").id
    puck = m.body("puck").id
    table = m.geom("table_top").id
    p0 = d.xpos[puck].copy()
    Rt = rz(np.pi)
    # pre-solve the "above" pose from home (global seed), then go there in joint space
    q_above, e1, r1 = ik.solve(d.qpos, d.qpos[ik.qa], p0 + [0, 0, 0.10], Rt, it=600)
    nsub = round((1 / 30) / ts)
    steps = 0
    arm_table_hits = 0
    t0 = time.perf_counter()
    q_cmd = d.qpos[ik.qa].copy()

    def step(qc, g):
        nonlocal steps, arm_table_hits
        d.ctrl[act] = qc
        d.ctrl[grip] = g
        for _ in range(nsub):
            mujoco.mj_step(m, d)
            steps += 1
        for c in d.contact[: d.ncon]:
            if table in (c.geom1, c.geom2):
                o = c.geom2 if c.geom1 == table else c.geom1
                # counts control steps that end with any arm geom touching the table
                if mujoco.mj_id2name(m, 1, m.geom_bodyid[o]).startswith("openarm"):
                    arm_table_hits += 1

    q_home = q_cmd.copy()
    for k in range(1, 46):
        step(q_home + (q_above - q_home) * k / 45, -0.785)
    q_cmd = q_above
    goal = ik.pinch(d)[0].copy()
    log = [f"presolve err {e1 * 1000:.1f}mm/{r1:.3f}"]
    for name, tgt, g, n in [
        ("down", p0 + [0, 0, 0.005], -0.785, 45),
        ("close", p0 + [0, 0, 0.005], 0.4, 25),
        ("lift", p0 + [0, 0, 0.15], 0.4, 60),
        ("hold", p0 + [0, 0, 0.15], 0.4, 60),
    ]:
        for k in range(n):
            delta = tgt - goal
            dist = np.linalg.norm(delta)
            goal = goal + (delta if dist < 0.008 else delta / dist * 0.008)
            q_cmd, _, _ = ik.solve(d.qpos, q_cmd, goal, Rt, it=20)
            step(q_cmd, g)
        log.append(
            f"{name}: pinch_err={np.linalg.norm(ik.pinch(d)[0] - tgt) * 1000:.1f}mm puck_z={d.xpos[puck][2]:.3f}"
        )
    wall = time.perf_counter() - t0
    print(
        f"ts={ts:.5f} gravcomp={gc} lifted={(d.xpos[puck][2] - p0[2]) * 1000:.0f}mm arm_table_contact_steps={arm_table_hits} RTF(phys+IK)={steps * ts / wall:.1f}x nan={not np.isfinite(d.qpos).all()}"
    )
    print("   " + " | ".join(log))


if __name__ == "__main__":
    for gc in (True, False):
        for ts in (0.001, 1 / 480, 1 / 240):
            run(ts, gc)

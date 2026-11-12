import sys

import imageio
import mujoco
import numpy as np
from grasp import IK, build, rz

m = build(1 / 240, True)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
ik = IK(m)
p0 = d.body("puck").xpos.copy()
rows = []
r = mujoco.Renderer(m, 240, 320)
for h in (0.15, 0.10, 0.04):
    q, _, _ = ik.solve(d.qpos, d.qpos[ik.qa], p0 + [0, 0, h], rz(np.pi), it=600)
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    d2.qpos[ik.qa] = q
    for n in ("openarm_right_finger_joint1", "openarm_right_finger_joint2"):
        d2.qpos[m.joint(n).qposadr[0]] = -0.785
    mujoco.mj_forward(m, d2)
    r.update_scene(d2, "camera_wrist_right")
    a = r.render()
    r.update_scene(d2, "tablecam")
    b = r.render()
    rows.append(np.concatenate([a, b], 1))
imageio.imwrite(sys.argv[1], np.concatenate(rows, 0))
r.close()

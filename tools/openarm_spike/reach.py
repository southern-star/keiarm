import sys

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grasp import IK, build, rz

m = build(1 / 240, True)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
xs = np.arange(0.075, 0.651, 0.025)
ys = np.arange(-0.50, 0.501, 0.025)
Z = 0.42
res = {}
for side in ("right", "left"):
    ik = IK(m, side)
    q_home = d.qpos[ik.qa].copy()
    ok = np.zeros((len(ys), len(xs)), bool)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            for yaw in np.deg2rad([180, 90, 0, -90, 135, -135, 45, -45]):
                q, ep, er = ik.solve(d.qpos, q_home, np.array([x, y, Z]), rz(yaw), it=300)
                if ep < 3e-3 and er < 0.03:
                    ok[i, j] = True
                    break
    res[side] = ok
    print(side, "reachable cells", ok.sum(), "of", ok.size)
both = res["right"] & res["left"]
print(
    "both arms",
    both.sum(),
    "cells; y range both:",
    ys[both.any(1)].min() if both.any() else None,
    ys[both.any(1)].max() if both.any() else None,
    "x range both:",
    xs[both.any(0)].min() if both.any() else None,
    xs[both.any(0)].max() if both.any() else None,
)
img = res["right"].astype(int) + 2 * res["left"].astype(int)
fig, ax = plt.subplots(figsize=(6, 6))
cmap = matplotlib.colors.ListedColormap(["#eeeeee", "#e07b39", "#3a7bd5", "#6a3d9a"])
ax.imshow(
    img,
    origin="lower",
    extent=[xs[0] - 0.0125, xs[-1] + 0.0125, ys[0] - 0.0125, ys[-1] + 0.0125],
    cmap=cmap,
    vmin=0,
    vmax=3,
)
ax.add_patch(plt.Rectangle((0.06, -0.55), 0.82, 1.10, fill=False, ls="--", color="k"))
ax.plot([0, 0], [-0.152, 0.152], "ko")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(
    f"OpenArm v2 top-down grasp reach at z={Z} (table+2cm)\norange=right only, blue=left only, purple=both"
)
fig.savefig(sys.argv[1], dpi=110, bbox_inches="tight")

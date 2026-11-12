import sys
import time

import mujoco
import numpy as np

spec = mujoco.MjSpec.from_file(sys.argv[1])
spec.option.timestep = 1 / 240
for b in spec.bodies:
    if b.name.startswith("openarm_"):
        b.gravcomp = 1.0
m = spec.compile()
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
cams = ["frontcam", "camera_wrist_left", "camera_wrist_right"]
r = mujoco.Renderer(m, 120, 160)
opt = mujoco.MjvOption()  # default groups 0-2 visible
for c in cams:
    r.update_scene(d, c)
    r.render()
N = 300
t0 = time.perf_counter()
for i in range(N):
    for _ in range(8):
        mujoco.mj_step(m, d)
    for c in cams:
        r.update_scene(d, c)
        img = r.render()
wall = time.perf_counter() - t0
print(
    f"steps+3cam render: {N / wall:.1f} control-steps/s  RTF={N / 30 / wall:.2f}x  ngeom={m.ngeom}  mesh faces={m.nmeshface}"
)
t0 = time.perf_counter()
for i in range(N):
    for _ in range(8):
        mujoco.mj_step(m, d)
print(f"physics only: RTF={N / 30 / (time.perf_counter() - t0):.1f}x")
import imageio

imgs = []
for c in cams + ["tablecam"]:
    r.update_scene(d, c)
    imgs.append(r.render())
r2 = mujoco.Renderer(m, 480, 640)
big = []
for c in ["frontcam", "tablecam", "camera_wrist_right"]:
    r2.update_scene(d, c)
    big.append(r2.render())
imageio.imwrite(sys.argv[2], np.concatenate([np.concatenate(imgs, 1)], 0))
imageio.imwrite(sys.argv[3], np.concatenate(big, 1))
r.close()
r2.close()

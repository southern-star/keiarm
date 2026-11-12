import sys

import imageio
import mujoco
import numpy as np

rows = []
for root in sys.argv[2:]:
    m = mujoco.MjModel.from_xml_path(f"{root}/pedestal/move_puck_scene.xml")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 240, 320)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.25, 0, 0.5]
    cam.distance = 0.9
    cam.azimuth = 200
    cam.elevation = -25
    r.update_scene(d, cam)
    a = r.render()
    cam.lookat[:] = [0.28, -0.15, 0.45]
    cam.distance = 0.3
    cam.azimuth = 150
    cam.elevation = -20
    r.update_scene(d, cam)
    b = r.render()
    r.update_scene(d, "camera_wrist_right")
    c = r.render()
    rows.append(np.concatenate([a, b, c], 1))
    r.close()
    print(root, "faces", m.nmeshface)
imageio.imwrite(sys.argv[1], np.concatenate(rows, 0))

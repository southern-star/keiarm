"""E0: 環境確認。

- 固定版 MuJoCo（Python）で flex 関連の機能がそろっているか
- 本家 model/flex/poncho.xml と同じ flexcomp 設定（3.17.0 タグの値）を、マネキンの代わりに
  球にかぶせて動かし、動画を保存する（poncho.xml 本体は mannequin.xml の include と頂点列を含む
  ので取り込まず、設定値だけ写した）

実行: MUJOCO_GL=egl uv run python spikes/cloth/e0_env.py
"""

import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# poncho.xml（3.17.0）の値: integrator="discrete" solver="CG" tolerance="1e-6"、
# radius 0.01、mass 1、<edge equality="vert" damping="0.1"/>、
# <elasticity young="3e5" poisson="0" thickness="8e-3" elastic2d="bend" damping="0.02"/>、
# <contact solref="0.003"/>。timestep は既定（0.002）。
PONCHO_LIKE = """<mujoco model="poncho_like">
  <option integrator="discrete" solver="CG" tolerance="1e-6"/>
  <visual><global offwidth="480" offheight="480"/><quality shadowsize="0"/></visual>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom type="plane" size="3 3 .1" rgba=".3 .35 .4 1"/>
    <geom type="sphere" size="0.4" pos="0 0 1" rgba=".7 .7 .7 1"/>
    <camera name="cam" pos="0 -3.2 2.0" xyaxes="1 0 0 0 0.45 1" fovy="45"/>
    <flexcomp name="f1" type="grid" count="24 24 1" spacing=".08 .08 .08" rgba=".8 .2 .2 1" radius="0.01"
              dim="2" pos="0 0 2" mass="1">
      <edge equality="vert" damping="0.1"/>
      <elasticity young="3e5" poisson="0" thickness="8e-3" elastic2d="bend" damping="0.02"/>
      <contact solref="0.003"/>
    </flexcomp>
  </worldbody>
</mujoco>"""


def main():
    info = {"mujoco_version": mujoco.__version__, "python": sys.version.split()[0]}
    info["mjtEq"] = [x for x in dir(mujoco.mjtEq) if x.startswith("mjEQ")]
    info["mjtFlexSelf"] = [x for x in dir(mujoco.mjtFlexSelf) if x.startswith("mjFLEX")]
    info["has_flexvert_xpos"] = hasattr(mujoco.MjData(mujoco.MjModel.from_xml_string("<mujoco/>")), "flexvert_xpos")

    model = mujoco.MjModel.from_xml_string(PONCHO_LIKE)
    data = mujoco.MjData(model)
    info["poncho_like"] = {"nbody": model.nbody, "nflexvert": int(model.nflexvert), "nv": model.nv, "neq": model.neq,
                           "eq_types": sorted({int(t) for t in model.eq_type}),
                           "timestep": model.opt.timestep}
    r = mujoco.Renderer(model, 480, 480)
    frames = []
    t_wall = 0.0
    steps = int(3.0 / model.opt.timestep)
    every = round(1 / 30 / model.opt.timestep)
    for k in range(steps):
        t = time.perf_counter()
        mujoco.mj_step(model, data)
        t_wall += time.perf_counter() - t
        if k % every == 0:
            r.update_scene(data, "cam")
            frames.append(r.render())
    r.close()
    verts = data.flexvert_xpos
    info["poncho_like"].update(finite=bool(np.isfinite(verts).all()), rtf=round(steps * model.opt.timestep / t_wall, 3),
                               final_zmin=float(verts[:, 2].min()), final_zmax=float(verts[:, 2].max()))
    imageio.mimwrite(OUT / "e0_poncho_like.mp4", frames, fps=30, macro_block_size=1)
    (OUT / "e0_env.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

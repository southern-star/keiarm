"""時間の内訳（mjData.timer）を見る診断。

実行: uv run python spikes/cloth/diag_profile.py [N] [solver] [selfcollide] [equality] [radius]
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cloth import ClothCfg, StepTimer, cloth_verts, edge_strain, table_scene_xml


def profile(cfg: ClothCfg, sim_t=1.0, drop_z=0.01, extra=""):
    m = mujoco.MjModel.from_xml_string(table_scene_xml(cfg, cloth_pos=(0, 0, drop_z), extra_world=extra))
    d = mujoco.MjData(m)
    t = StepTimer()
    for _ in range(int(sim_t / cfg.timestep)):
        t.step(m, d)
    names = [x for x in dir(mujoco.mjtTimer) if x.startswith("mjTIMER_")]
    tot = d.timer[mujoco.mjtTimer.mjTIMER_STEP].duration
    rows = []
    for n in names:
        tm = d.timer[getattr(mujoco.mjtTimer, n)]
        if tm.duration > 0.02 * tot:
            rows.append(f"{n[8:]}={tm.duration / max(tm.number, 1) * 1e3:.2f}ms")
    v = cloth_verts(m, d)
    print(f"{cfg.n=} {cfg.solver=} {cfg.selfcollide=} {cfg.equality=} r={cfg.radius} rtf={t.rtf:.3f} "
          f"ncon={d.ncon} nefc={d.nefc} solver_niter={d.solver_niter[0]} strain={edge_strain(m, d):.3f} "
          f"zmin={v[:, 2].min() * 1000:.1f}mm | " + " ".join(rows))


if __name__ == "__main__":
    a = sys.argv[1:]
    cfg = ClothCfg(n=int(a[0]) if a else 16, solver=a[1] if len(a) > 1 else "Newton",
                   selfcollide=a[2] if len(a) > 2 else "auto", equality=a[3] if len(a) > 3 else "vert",
                   radius=float(a[4]) if len(a) > 4 else 0.002)
    profile(cfg)
    np.set_printoptions(precision=3)

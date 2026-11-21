"""たたんだ状態（層が重なった状態）での時間の内訳と、速くする設定の候補を比べる。

二つ折りまで E2 と同じ手順で進めた状態を保存し、その状態から 1 秒ずつ設定を変えて回す
（物理パラメータは状態を保ったまま model.opt を書き換える）。

実行: uv run python spikes/cloth/diag_fold_speed.py [N]
"""

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cloth import ClothCfg, Grabber, cloth_verts, table_scene_xml, vid
from e2_fold import arc, corner_patch

OUT = Path(__file__).parent / "out"


def folded_state(cfg: ClothCfg):
    model = mujoco.MjModel.from_xml_string(table_scene_xml(cfg, cloth_pos=(0, 0, cfg.radius + 0.001)))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, int(0.5 / cfg.timestep))
    n = cfg.n
    v = cloth_verts(model, data)
    gs = [Grabber(model, "grip_a"), Grabber(model, "grip_b")]
    grabs = [corner_patch(cfg, n - 1, 0, 3), corner_patch(cfg, n - 1, n - 1, 3)]
    tg = [v[vid(cfg, 0, 0)] + [0, 0, 0.008], v[vid(cfg, 0, n - 1)] + [0, 0, 0.008]]
    p0 = []
    for g, vs in zip(gs, grabs):
        data.mocap_pos[g.mocap_id] = v[vs].mean(0)
        g.grab(data, vs)
        p0.append(v[vs].mean(0))
    steps = int(2.0 / cfg.timestep)
    mid_state = None
    for k in range(1, steps + 1):
        for g, a, b in zip(gs, p0, tg):
            data.mocap_pos[g.mocap_id] = arc(a, b, k / steps, 0.6 * np.linalg.norm((b - a)[:2]) / 2)
        mujoco.mj_step(model, data)
        if k == steps // 2:
            mid_state = snapshot(model, data)
    for g in gs:
        g.release(data)
    mujoco.mj_step(model, data, int(0.5 / cfg.timestep))
    return model, data, mid_state, snapshot(model, data)


def snapshot(model, data):
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    s = np.empty(size)
    mujoco.mj_getState(model, data, s, mujoco.mjtState.mjSTATE_INTEGRATION)
    return s


def bench(model, data, state, secs=1.0) -> dict:
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    for t in data.timer:
        t.duration, t.number = 0.0, 0
    steps = max(1, int(secs / model.opt.timestep))
    t0 = time.perf_counter()
    niter = []
    for _ in range(steps):
        mujoco.mj_step(model, data)
        niter.append(int(data.solver_niter[0]))
    wall = time.perf_counter() - t0
    per = lambda k: data.timer[k].duration / max(data.timer[k].number, 1) * 1e3
    T = mujoco.mjtTimer
    zs = cloth_verts(model, data)[:, 2]
    return {"rtf": round(steps * model.opt.timestep / wall, 3), "ms_step": round(wall / steps * 1e3, 2),
            "ncon": int(data.ncon), "nefc": int(data.nefc), "niter_mean": round(float(np.mean(niter)), 1),
            "collision_ms": round(per(T.mjTIMER_POS_COLLISION), 2), "make_ms": round(per(T.mjTIMER_POS_MAKE), 2),
            "solve_ms": round(per(T.mjTIMER_CONSTRAINT), 2), "zmin_mm": round(float(zs.min()) * 1000, 2),
            "finite": bool(np.isfinite(zs).all())}


S = mujoco.mjtSolver
# (opt の上書き, model 配列の上書き)。PGS は discrete + flex で未対応（mj_checkDiscrete）
VARIANTS = {
    "base(CG, iter50, tol1e-8)": ({}, {}),
    "iterations=20": ({"iterations": 20}, {}),
    "iterations=10": ({"iterations": 10}, {}),
    "tolerance=1e-4": ({"tolerance": 1e-4}, {}),
    "Newton": ({"solver": S.mjSOL_NEWTON}, {}),
    "elliptic": ({"cone": mujoco.mjtCone.mjCONE_ELLIPTIC}, {}),
    "self condim=1": ({}, {"flex_condim": 1}),
    "self condim=1 iter=10": ({"iterations": 10}, {"flex_condim": 1}),
    "passive contact": ({}, {"flex_passive": 1}),
    "dt=1/120": ({"timestep": 1 / 120}, {}),
    "dt=1/120 iter=10": ({"timestep": 1 / 120, "iterations": 10}, {}),
}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    out = {}
    for name, cfg in {"A": ClothCfg(n=n), "B": ClothCfg(n=n, equality="false", elastic2d="both", young=2e5)}.items():
        model, data, mid, rest = folded_state(cfg)
        base = {k: getattr(model.opt, k) for k in ("iterations", "tolerance", "solver", "cone", "timestep")}
        base_arr = {k: getattr(model, k).copy() for k in ("flex_condim", "flex_passive")}
        for vname, (kw, arr) in VARIANTS.items():
            for k, v in base.items():
                setattr(model.opt, k, v)
            for k, v in base_arr.items():
                getattr(model, k)[:] = v
            for k, v in kw.items():
                setattr(model.opt, k, v)
            for k, v in arr.items():
                getattr(model, k)[:] = v
            try:
                row = {"moving": bench(model, data, mid, 0.5), "rest": bench(model, data, rest, 0.5)}
            except Exception as e:  # noqa: BLE001
                row = {"error": str(e).splitlines()[0]}
                print(name, vname, row, flush=True)
                out[f"{name}_n{n} {vname}"] = row
                continue
            out[f"{name}_n{n} {vname}"] = row
            print(f"{name}_n{n} {vname:28s} moving {row['moving']}\n{'':36s} rest   {row['rest']}", flush=True)
    (OUT / f"diag_fold_speed_n{n}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

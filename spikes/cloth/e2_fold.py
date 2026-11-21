"""E2: スクリプトでたたむ（ロボットなし）。

mocap 2 個を布の頂点と connect 拘束でつなぎ（拘束スロットの相手を実行時に差し替える）、
弧の軌道で反対側まで運んで離す。二つ折り → 続けて四つ折り。

各試行は (seed) で決まる: 布の位置（±2 cm）と向き（±10°）、軌道の所要時間と持ち上げ高さ。

測るもの
- 層の順序: 上層の頂点が対応する下層の頂点より下にある割合（tol = 0 と tol = radius）
- 形の保持: 角どうしの距離、上から見た凸包の面積比（理想は 0.5 / 0.25）。離した直後と 2 秒後
- 実時間比（たたんでいる間、静止後）、破綻

実行:
  uv run python spikes/cloth/e2_fold.py --trials 100 --n 16 --workers 10
  MUJOCO_GL=egl uv run python spikes/cloth/e2_fold.py --video --seed 0 --n 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cloth import (
    ClothCfg,
    Grabber,
    StepTimer,
    broken,
    cloth_verts,
    convex_area,
    flex_triangles,
    fold_layers_half,
    fold_layers_quarter,
    layer_violation,
    self_intersections,
    strain_stats,
    table_scene_xml,
    vid,
)

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
ROTATE = False  # --rotate: 掴んだ点を折り目の軸まわりに 180° 回す（グリッパが手首を返す場合）


def arc(p0: np.ndarray, p1: np.ndarray, s: float, lift: float) -> np.ndarray:
    """p0 から p1 へ。水平は滑らかに（cos 補間）、高さは sin で持ち上げる。"""
    u = 0.5 - 0.5 * np.cos(np.pi * s)
    p = p0 + u * (p1 - p0)
    p[2] += lift * np.sin(np.pi * u)
    return p


class Recorder:
    def __init__(self, model, every: int, views=("side", "top"), w=480, h=360):
        self.r = mujoco.Renderer(model, h, w)
        self.every = every
        self.views = views
        self.frames = {v: [] for v in views}
        self.k = 0

    def __call__(self, data):
        if self.k % self.every == 0:
            for v in self.views:
                self.r.update_scene(data, v)
                self.frames[v].append(self.r.render())
        self.k += 1

    def save(self, stem: str):
        import imageio.v2 as imageio

        for v, fr in self.frames.items():
            imageio.mimwrite(OUT / f"{stem}_{v}.mp4", fr, fps=30, macro_block_size=1)
        self.r.close()


def corner_patch(cfg: ClothCfg, i: int, j: int, k: int) -> list[int]:
    """角 (i, j) と、その隣の頂点（グリッパの幅 ≈ 2 cm 以内）から最大 k 個。"""
    di = 1 if i == 0 else -1
    dj = 1 if j == 0 else -1
    cand = [(i, j), (i + di, j), (i, j + dj), (i + di, j + dj)]
    reach = 0.02
    out = [vid(cfg, a, b) for a, b in cand if np.hypot(a - i, b - j) * cfg.spacing() <= reach]
    return out[:k]


def trial(args) -> dict:
    global ROTATE
    cfg_d, seed, video, ROTATE = args
    cfg = ClothCfg(**cfg_d)
    rng = np.random.default_rng(np.random.SeedSequence([0xC107, seed]))
    off = rng.uniform(-0.02, 0.02, 2)
    yaw = rng.uniform(-np.radians(10), np.radians(10))
    t_fold = [rng.uniform(1.5, 3.0), rng.uniform(1.5, 3.0)]
    lift_ratio = [rng.uniform(0.45, 0.8), rng.uniform(0.45, 0.8)]

    xml = table_scene_xml(cfg, cloth_pos=(off[0], off[1], cfg.radius + 0.001), cloth_euler=(0, 0, np.degrees(yaw)))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ga, gb = Grabber(model, "grip_a"), Grabber(model, "grip_b")
    dt = cfg.timestep
    rec = Recorder(model, every=round(1 / 30 / dt)) if video else None
    timers = {"fold": StepTimer(), "rest": StepTimer()}
    res = {"seed": seed, "n": cfg.n, "rotate": ROTATE, "status": "ok", "off": off.round(4).tolist(), "yaw_deg": round(float(np.degrees(yaw)), 2),
           "t_fold": [round(x, 3) for x in t_fold], "lift_ratio": [round(x, 3) for x in lift_ratio]}
    wall0 = time.perf_counter()

    def step(phase: str, n: int = 1) -> bool:
        for _ in range(n):
            timers[phase].step(model, data)
            if rec:
                rec(data)
            b = broken(model, data, cloth_verts(model, data), cfg.radius)
            if b:
                res["status"] = b
                return False
        return True

    def fold(grabs, targets, t, lift_ratio) -> bool:
        """grabs: [(grabber, verts)]。targets: 各 mocap の終点。"""
        v = cloth_verts(model, data)
        starts = []
        for (g, verts), tgt in zip(grabs, targets):
            p0 = v[verts].mean(axis=0)
            data.mocap_pos[g.mocap_id] = p0
            g.grab(data, verts)
            starts.append((p0.copy(), tgt))
        steps = round(t / dt)
        for k in range(1, steps + 1):
            s = k / steps
            for (g, _), (p0, p1) in zip(grabs, starts):
                lift = lift_ratio * np.linalg.norm((p1 - p0)[:2]) / 2
                data.mocap_pos[g.mocap_id] = arc(p0, p1, s, lift)
                if ROTATE:
                    # 掴んだ点ごと、折り目の軸（水平で運ぶ向きに直交）まわりに 0 → 180° 回す
                    d = (p1 - p0)[:2] / np.linalg.norm((p1 - p0)[:2])
                    axis = np.array([-d[1], d[0], 0.0])
                    u = 0.5 - 0.5 * np.cos(np.pi * s)
                    q = np.zeros(4)
                    mujoco.mju_axisAngle2Quat(q, axis, -np.pi * u)
                    data.mocap_quat[g.mocap_id] = q
            if not step("fold"):
                return False
        if not step("fold", int(0.3 / dt)):  # 終点で保持
            return False
        for g, _ in grabs:
            g.release(data)
            data.mocap_quat[g.mocap_id] = [1, 0, 0, 0]
        return True

    def measure(tag: str, layers, area0: float, corner_pairs):
        v = cloth_verts(model, data)
        pairs, core = layers(cfg), layers(cfg, skip_crease=True)
        res[f"{tag}_viol0"] = round(layer_violation(v, pairs, 0.0), 4)
        res[f"{tag}_viol_r"] = round(layer_violation(v, pairs, cfg.radius), 4)
        bad = []
        for k, (lo, hi) in enumerate(pairs):
            for a, b in zip(lo[v[hi, 2] < v[lo, 2]], hi[v[hi, 2] < v[lo, 2]]):
                bad.append([k, int(a) // cfg.n, int(a) % cfg.n, round(float(v[b, 2] - v[a, 2]) * 1000, 1)])
        res[f"{tag}_bad"] = bad
        res[f"{tag}_xings"] = self_intersections(v, tris)  # 層のすり抜け（中心面の交差）  # [層の組, 下層の i, j, z_上 - z_下 (mm)]
        res[f"{tag}_core_viol0"] = round(layer_violation(v, core, 0.0), 4)  # 折り目の行・列を除く
        res[f"{tag}_core_viol_r"] = round(layer_violation(v, core, cfg.radius), 4)
        res[f"{tag}_area_ratio"] = round(convex_area(v) / area0, 4)
        res[f"{tag}_corner_mm"] = round(max(float(np.linalg.norm(v[a] - v[b])) for a, b in corner_pairs) * 1000, 1)
        res[f"{tag}_zmax_mm"] = round(float(v[:, 2].max()) * 1000, 2)
        res[f"{tag}_strain_p95"] = round(strain_stats(model, data)[1], 4)

    n, h, top = cfg.n, cfg.n // 2, cfg.radius * 2
    # 0. 静止
    if not step("rest", int(0.5 / dt)):
        return finish(res, timers, rec, wall0, seed)
    area0 = convex_area(cloth_verts(model, data))
    tris = flex_triangles(model)

    # 1. 二つ折り: x の遠い側の角 (n-1, 0), (n-1, n-1) を近い側の角 (0, 0), (0, n-1) の上へ
    v = cloth_verts(model, data)
    grabs = [(ga, corner_patch(cfg, n - 1, 0, 3)), (gb, corner_patch(cfg, n - 1, n - 1, 3))]
    targets = [v[vid(cfg, 0, 0)] + [0, 0, top + 0.004], v[vid(cfg, 0, n - 1)] + [0, 0, top + 0.004]]
    if not fold(grabs, targets, t_fold[0], lift_ratio[0]):
        return finish(res, timers, rec, wall0, seed)
    half_corners = [(vid(cfg, n - 1, 0), vid(cfg, 0, 0)), (vid(cfg, n - 1, n - 1), vid(cfg, 0, n - 1))]
    if not step("rest", int(0.2 / dt)):
        return finish(res, timers, rec, wall0, seed)
    measure("half", fold_layers_half, area0, half_corners)
    if not step("rest", int(2.0 / dt)):
        return finish(res, timers, rec, wall0, seed)
    measure("half2s", fold_layers_half, area0, half_corners)

    # 2. 四つ折り: +y 側の重なった角を -y 側へ。A は x の近い辺の角（2 層）、B は折り目の角（2 層）
    v = cloth_verts(model, data)
    a_verts = [vid(cfg, 0, n - 1), vid(cfg, n - 1, n - 1), vid(cfg, 0, n - 2), vid(cfg, n - 1, n - 2)]
    b_verts = [vid(cfg, h - 1, n - 1), vid(cfg, h, n - 1), vid(cfg, h - 1, n - 2), vid(cfg, h, n - 2)]
    grabs = [(ga, a_verts), (gb, b_verts)]
    targets = [v[[vid(cfg, 0, 0), vid(cfg, n - 1, 0)]].mean(0) + [0, 0, 2 * top + 0.004],
               v[[vid(cfg, h - 1, 0), vid(cfg, h, 0)]].mean(0) + [0, 0, 2 * top + 0.004]]
    if not fold(grabs, targets, t_fold[1], lift_ratio[1]):
        return finish(res, timers, rec, wall0, seed)
    q_corners = [(vid(cfg, 0, n - 1), vid(cfg, 0, 0)), (vid(cfg, h - 1, n - 1), vid(cfg, h - 1, 0))]
    if not step("rest", int(0.2 / dt)):
        return finish(res, timers, rec, wall0, seed)
    measure("quarter", fold_layers_quarter, area0, q_corners)
    if not step("rest", int(2.0 / dt)):
        return finish(res, timers, rec, wall0, seed)
    measure("quarter2s", fold_layers_quarter, area0, q_corners)
    return finish(res, timers, rec, wall0, seed)


def finish(res, timers, rec, wall0, seed):
    res["rtf_fold"] = round(timers["fold"].rtf, 3)
    res["rtf_rest"] = round(timers["rest"].rtf, 3)
    res["sim_s"] = round(timers["fold"].sim + timers["rest"].sim, 2)
    res["wall_s"] = round(time.perf_counter() - wall0, 2)
    res["ncon_max"] = int(max(timers["fold"].ncon + timers["rest"].ncon or [0]))
    if rec:
        rec.save(f"e2_fold_n{res['n']}_seed{seed}{'_rot' if res['rotate'] else ''}")
    return res


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    s = {"trials": len(rows), "broken": len(rows) - len(ok),
         "broken_kinds": sorted({r["status"] for r in rows if r["status"] != "ok"})}
    for k in ("half_viol0", "half_viol_r", "quarter_viol0", "quarter_viol_r", "half_core_viol0", "quarter_core_viol0",
              "quarter_core_viol_r", "quarter2s_viol0", "quarter2s_core_viol0", "half_xings", "half2s_xings",
              "quarter_xings", "quarter2s_xings", "half_area_ratio", "half2s_area_ratio",
              "quarter_area_ratio", "quarter2s_area_ratio", "half_corner_mm", "half2s_corner_mm", "quarter_corner_mm",
              "quarter2s_corner_mm", "half_strain_p95", "quarter_strain_p95", "rtf_fold", "rtf_rest", "wall_s"):
        x = np.array([r[k] for r in ok if k in r])
        if len(x):
            s[k] = {"mean": round(float(x.mean()), 4), "p95": round(float(np.percentile(x, 95)), 4),
                    "max": round(float(x.max()), 4), "min": round(float(x.min()), 4)}
    for tag in ("half", "quarter"):
        x = np.array([r[f"{tag}_viol0"] for r in ok if f"{tag}_viol0" in r])
        s[f"{tag}_trials_viol_lt1pct"] = int((x < 0.01).sum())
        x = np.array([r[f"{tag}_core_viol0"] for r in ok if f"{tag}_core_viol0" in r])
        s[f"{tag}_trials_core_viol_lt1pct"] = int((x < 0.01).sum())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--rotate", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="ClothCfg の上書き 例: radius=0.003 solver=Newton")
    args = ap.parse_args()
    global ROTATE
    ROTATE = args.rotate
    kw = {"n": args.n}
    for item in args.set:
        k, v = item.split("=")
        f = ClothCfg.__dataclass_fields__[k]
        kw[k] = f.type == "str" and v or (int(v) if f.type == "int" else float(v) if f.type == "float" else v)
    cfg = ClothCfg(**kw)
    if args.video:
        r = trial((cfg.to_dict(), args.seed, True, ROTATE))
        print(json.dumps(r, indent=1))
        return
    jobs = [(cfg.to_dict(), s, False, ROTATE) for s in range(args.seed, args.seed + args.trials)]
    with Pool(args.workers) as pool:
        rows = []
        for r in pool.imap_unordered(trial, jobs):
            rows.append(r)
            print(r["seed"], r["status"], r.get("half_viol0"), r.get("quarter_viol0"), r.get("rtf_fold"), flush=True)
    rows.sort(key=lambda r: r["seed"])
    summ = summarize(rows)
    name = f"e2_n{cfg.n}{args.tag}"
    (OUT / f"{name}.json").write_text(json.dumps({"cfg": cfg.to_dict(), "summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

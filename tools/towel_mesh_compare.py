"""Coarse towel mesh vs the training mesh (milestone 3, Phase 4; docs/milestone3.md, checks V2, V4 a).

V2: the N = 16 expert's actions replayed open loop on a coarse mesh (same scene and episode):
    how far the final shape moves (corner gaps, area ratio) and whether it still succeeds.
V4 a: the policy cameras' images of the initial state (flat towel) at N = 16 and the coarse N.

Usage: uv run python tools/towel_mesh_compare.py [--n 10] [--scenes 100] [--workers 10]
Writes docs/towel/mesh_compare_n<N>.json and prints a summary.
"""

import argparse
import json
import os
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from keiarm import config
from keiarm.paths import REPO_ROOT
from keiarm.towel.env import CAMERAS, TowelEnv
from keiarm.towel.rollout import run_expert
from keiarm.towel.task import SUCCESS_HOLD_STEPS, TaskIndex, check_fold, read_state


def _final(env: TowelEnv) -> dict:
    ix = TaskIndex(env.model)
    chk = check_fold(read_state(env.model, ix, env.data), ix, env.scene)
    return {"gap_mm": {s: [round(1000 * v, 2) for v in g] for s, g in chk.corner_gap.items()},
            "area": round(chk.area_ratio, 4), "violation": round(chk.violation, 4), "success_now": chk.success}


def one(job) -> dict:
    seed, n = job
    fine = TowelEnv(seed, render=True, n=16)
    coarse = TowelEnv(seed, render=True, n=n)
    try:
        res = run_expert(fine, 0)
        out = {"scene": seed, "fine_success": res.success, "fine_steps": res.steps, "fine": _final(fine)}
        # V2: open-loop replay on the coarse mesh (the same actions, then hold the last one)
        coarse.reset(0)
        ix = TaskIndex(coarse.model)
        run, ok, ok_step = 0, False, None
        for t, a in enumerate(res.actions):
            coarse.step(a)
            run = run + 1 if check_fold(read_state(coarse.model, ix, coarse.data), ix, coarse.scene).success else 0
            if run >= SUCCESS_HOLD_STEPS and not ok:
                ok, ok_step = True, t + 1
        out.update({"coarse_replay_success": ok, "coarse_replay_step": ok_step, "coarse": _final(coarse)})
        # V4 a: initial images
        fine.reset(0)
        coarse.reset(0)
        a, b = fine.observe().images, coarse.observe().images
        out["image_mean_abs"] = {c: float(np.abs(a[c].astype(np.int16) - b[c]).mean()) for c in CAMERAS}
        return out
    finally:
        fine.close()
        coarse.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--scenes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    seeds = list(config.EVAL_SCENE_SEEDS)[: args.scenes]
    with Pool(args.workers) as pool:
        rows = pool.map(one, [(s, args.n) for s in seeds], chunksize=1)
    d_gap = [abs(r["coarse"]["gap_mm"][s][0] - r["fine"]["gap_mm"][s][0]) for r in rows for s in ("left", "right")]
    d_area = [abs(r["coarse"]["area"] - r["fine"]["area"]) for r in rows]
    summary = {
        "n": args.n, "scenes": len(rows),
        "fine_success": sum(r["fine_success"] for r in rows),
        "coarse_replay_success": sum(r["coarse_replay_success"] for r in rows),
        "corner_gap_diff_mm": {"median": float(np.median(d_gap)), "p90": float(np.percentile(d_gap, 90)),
                               "max": float(max(d_gap))},
        "area_ratio_diff": {"median": float(np.median(d_area)), "max": float(max(d_area))},
        "initial_image_mean_abs": {c: float(np.mean([r["image_mean_abs"][c] for r in rows])) for c in CAMERAS},
    }
    print(json.dumps(summary, indent=1))
    path = REPO_ROOT / "docs" / "towel" / f"mesh_compare_n{args.n}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

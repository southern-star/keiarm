"""Scripted towel half fold (milestone 3, Phase 0): N episodes in parallel, or one with a video.

Usage:
  uv run python scripts/run_scripted_towel.py --episodes 20 --workers 10
  MUJOCO_GL=egl uv run python scripts/run_scripted_towel.py --scene 100000 --video runs/towel_scripted.mp4

Episode k uses scene_seed 100000 + k, episode_seed 0. Prints per-episode results and a
summary, including episodes per hour at this level of parallelism.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from keiarm.towel.env import TowelEnv
from keiarm.towel.scripted import scripted_fold


def run(scene_seed: int) -> dict:
    env = TowelEnv(scene_seed, render=False)
    env.reset(0)
    r = scripted_fold(env)
    env.close()
    sim = r.steps / 30
    return {"scene": scene_seed, "status": r.status, "completed": r.completed, "grabbed": r.grabbed,
            "corner_gap_mm": r.corner_gap_mm, "violation": r.violation, "crossings": r.crossings,
            "area_ratio": r.area_ratio, "max_anchor_err_mm": round(r.max_anchor_err_mm, 2),
            "max_track_mm": round(r.max_track_mm, 1), "sim_s": round(sim, 2), "wall_s": round(r.wall_total, 2),
            "rtf_physics": round(sim / r.wall_physics, 3), "rtf_total": round(sim / r.wall_total, 3)}


def video(scene_seed: int, path: str) -> None:
    import imageio.v2 as imageio

    env = TowelEnv(scene_seed, render=True, width=320, height=240)
    env.reset(0)
    frames = []

    def grab(e):
        frames.append(np.concatenate([e.render(c) for c in ("cam_front", "camera_wrist_left", "camera_wrist_right")], 1))

    r = scripted_fold(env, on_step=grab)
    env.close()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimwrite(path, frames, fps=30, macro_block_size=1)
    print(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--scene", type=int, default=100000)
    ap.add_argument("--video")
    ap.add_argument("--out", help="write the per-episode results as JSON")
    args = ap.parse_args()
    if args.video:
        video(args.scene, args.video)
        return
    seeds = [args.scene + k for k in range(args.episodes)]
    t = time.perf_counter()
    with Pool(args.workers) as pool:
        rows = pool.map(run, seeds)
    wall = time.perf_counter() - t
    for r in rows:
        print(json.dumps(r))
    done = sum(r["completed"] for r in rows)
    summary = {
        "completed": f"{done}/{len(rows)}",
        "status": sorted({r["status"] for r in rows}),
        "workers": args.workers,
        "wall_s": round(wall, 1),
        "episodes_per_hour": round(len(rows) / wall * 3600),
        "max_crossings": max((r["crossings"] or 0) for r in rows),
        "violation_max": max((r["violation"] or 0) for r in rows),
        "area_ratio_range": [min(r["area_ratio"] or 1 for r in rows), max(r["area_ratio"] or 0 for r in rows)],
        "corner_gap_h_mm_max": max(max(g[0] for g in r["corner_gap_mm"].values()) for r in rows if r["corner_gap_mm"]),
        "max_anchor_err_mm": max(r["max_anchor_err_mm"] for r in rows),
        "rtf_physics_mean": round(float(np.mean([r["rtf_physics"] for r in rows])), 3),
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "episodes": rows}, f, indent=1)


if __name__ == "__main__":
    main()

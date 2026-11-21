"""Evaluate the towel expert (milestone 3, Phase 2) on the eval scenes.

Usage: uv run python scripts/eval_expert_towel.py [--n 100] [--mesh 16] [--workers 10] [--dart 0.0065]
         [--out runs/towel/expert_eval.json]

Prints the success rate, time to success, arm-arm contact, the final fold measurements and,
for failures, the furthest stage and the last phases. Speed: episodes per hour at this
level of parallelism, and physics + expert time per control step.
"""

import argparse
import json
import time
from collections import Counter
from itertools import groupby
from multiprocessing import Pool

import numpy as np

from keiarm import config
from keiarm.towel.env import TowelEnv
from keiarm.towel.rollout import run_expert
from keiarm.towel.task import STAGES


def _one(job):
    scene_seed, episode_seed, mesh, dart = job
    env = TowelEnv(scene_seed, render=False, n=mesh)
    t = time.perf_counter()
    res = run_expert(env, episode_seed, dart_sigma=dart)
    wall = time.perf_counter() - t
    env.close()
    f = res.final
    return {
        "scene": scene_seed, "episode": episode_seed, "success": res.success, "steps": res.steps,
        "stage": STAGES[res.stage], "arms_touch_steps": res.arms_touch_steps,
        "corner_gap_mm": {s: [round(1000 * x, 1) for x in g] for s, g in f.corner_gap.items()},
        "area_ratio": round(f.area_ratio, 4), "violation": round(f.violation, 4),
        "max_speed": round(f.max_speed, 4), "free": f.free,
        "switches": sum(1 for _ in groupby(res.phases)),
        "last_phases": [k for k, _ in groupby(res.phases)][-6:],
        "wall_s": round(wall, 2), "ms_per_step": round(1000 * wall / res.steps, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--n", type=int, default=100, help="number of scenes")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per scene")
    ap.add_argument("--mesh", type=int, default=16, help="towel vertices per side")
    ap.add_argument("--dart", type=float, default=0.0, help="DART noise std [rad]")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", help="write the per-episode results as JSON")
    args = ap.parse_args()

    seeds = list(config.EVAL_SCENE_SEEDS if args.split == "eval" else config.TRAIN_SCENE_SEEDS)[: args.n]
    jobs = [(s, e, args.mesh, args.dart) for s in seeds for e in range(args.episodes)]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    wall = time.perf_counter() - t0
    ok = [r for r in rows if r["success"]]
    fail = [r for r in rows if not r["success"]]
    summary = {
        "split": args.split, "mesh": args.mesh, "dart": args.dart, "episodes": len(rows),
        "success": len(ok), "success_rate": round(len(ok) / len(rows), 3),
        "workers": args.workers, "wall_s": round(wall, 1), "episodes_per_hour": round(len(rows) / wall * 3600),
        "ms_per_step_mean": round(float(np.mean([r["ms_per_step"] for r in rows])), 1),
        "arms_touch_episodes": sum(r["arms_touch_steps"] > 0 for r in rows),
    }
    if ok:
        steps = np.array([r["steps"] for r in ok])
        summary["steps_to_success"] = {"median": int(np.median(steps)), "p90": int(np.percentile(steps, 90)),
                                       "max": int(steps.max())}
        summary["success_corner_gap_mm_max"] = max(max(g[0] for g in r["corner_gap_mm"].values()) for r in ok)
        summary["success_violation_max"] = max(r["violation"] for r in ok)
        summary["success_area_ratio"] = [min(r["area_ratio"] for r in ok), max(r["area_ratio"] for r in ok)]
    summary["switches_median"] = int(np.median([r["switches"] for r in rows]))
    print(json.dumps(summary, indent=1))
    if fail:
        print("failures by furthest stage:", dict(Counter(r["stage"] for r in fail)))
        for r in fail:
            print(f"  scene {r['scene']} ep {r['episode']}: stage {r['stage']}, gaps {r['corner_gap_mm']}, "
                  f"area {r['area_ratio']}, viol {r['violation']}, speed {r['max_speed']}, free {r['free']}, "
                  f"last {' '.join(r['last_phases'])}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "episodes": rows}, f, indent=1)


if __name__ == "__main__":
    main()

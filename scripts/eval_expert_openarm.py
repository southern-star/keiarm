"""Evaluate the OpenArm expert (milestone 2) on the eval scenes.

Usage: uv run python scripts/eval_expert_openarm.py [--split eval|train] [--n 100] [--dart 0.01]
"""

import argparse
import time
from collections import Counter
from itertools import groupby
from multiprocessing import Pool

import numpy as np

from keiarm import config
from keiarm.openarm.env import OpenArmEnv
from keiarm.openarm.rollout import run_expert
from keiarm.openarm.task import STAGES


def _one(job):
    scene_seed, episode_seed, dart = job
    env = OpenArmEnv(scene_seed, render=False)
    res = run_expert(env, episode_seed, dart_sigma=dart)
    env.close()
    switches = sum(1 for _ in groupby(res.phases))
    return res, switches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--n", type=int, default=100, help="number of scenes")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per scene")
    ap.add_argument("--dart", type=float, default=0.0, help="DART noise std [rad]")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    seeds = list(config.EVAL_SCENE_SEEDS if args.split == "eval" else config.TRAIN_SCENE_SEEDS)[: args.n]
    jobs = [(s, e, args.dart) for s in seeds for e in range(args.episodes)]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        out = pool.map(_one, jobs)
    results = [r for r, _ in out]
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    print(f"split={args.split} scenes={len(seeds)} episodes={len(jobs)} dart={args.dart} "
          f"({time.perf_counter() - t0:.0f}s)")
    print(f"success: {len(ok)}/{len(jobs)} = {100 * len(ok) / len(jobs):.1f}%")
    if ok:
        steps = np.array([r.steps for r in ok])
        print(f"steps to success: median {np.median(steps):.0f} ({np.median(steps) / config.CONTROL_HZ:.1f}s), "
              f"p90 {np.percentile(steps, 90):.0f}, max {steps.max()} ({steps.max() / config.CONTROL_HZ:.1f}s)")
    touching = [r for r in results if r.arms_touch_steps]
    print(f"episodes with arm-arm contact: {len(touching)} (steps: {sum(r.arms_touch_steps for r in touching)})"
          + (f" scenes {[r.scene_seed for r in touching][:10]}" if touching else ""))
    print(f"phase switches per episode: median {np.median([sw for _, sw in out]):.0f}, max {max(sw for _, sw in out)}")
    if fail:
        print("failures by furthest stage:", dict(Counter(STAGES[r.stage] for r in fail)))
        for r in fail:
            last = [k for k, _ in groupby(r.phases)][-6:]
            print(f"  scene {r.scene_seed} ep {r.episode_seed}: stage {STAGES[r.stage]}, last phases {' '.join(last)}")


if __name__ == "__main__":
    main()

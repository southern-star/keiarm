"""Evaluate the expert on a range of scenes (default: the 100-scene eval suite).

Usage: uv run python scripts/eval_expert.py [--dart 0.03] [--split eval|train] [--n 100]
Failures are saved (seeds + executed actions) to runs/expert_failures/ for replay.
"""

import argparse
import os
import time
from collections import Counter
from itertools import groupby
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.paths import REPO_ROOT
from keiarm.rollout import run_expert
from keiarm.task import STAGES


def _one(job):
    scene_seed, episode_seed, dart, teleport = job
    env = KeiArmEnv(scene_seed, render=False)
    res = run_expert(env, episode_seed, dart_sigma=dart, teleport_at=teleport)
    switches = sum(1 for _ in groupby(res.phases))
    return res, switches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--n", type=int, default=100, help="number of scenes")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per scene")
    ap.add_argument("--dart", type=float, default=0.0, help="DART noise std [rad]")
    ap.add_argument("--teleport-at", type=int, help="move the cube to a random spot at this step")
    args = ap.parse_args()

    seeds = list(config.EVAL_SCENE_SEEDS if args.split == "eval" else config.TRAIN_SCENE_SEEDS)[: args.n]
    jobs = [(s, e, args.dart, args.teleport_at) for s in seeds for e in range(args.episodes)]
    t0 = time.perf_counter()
    with Pool() as pool:
        out = pool.map(_one, jobs)
    results = [r for r, _ in out]
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    steps = np.array([r.steps for r in ok])
    print(f"split={args.split} scenes={len(seeds)} episodes={len(jobs)} dart={args.dart} teleport_at={args.teleport_at} "
          f"({time.perf_counter() - t0:.0f}s)")
    print(f"success: {len(ok)}/{len(jobs)} = {100 * len(ok) / len(jobs):.1f}%")
    if len(ok):
        print(f"steps to success: median {np.median(steps):.0f}, p90 {np.percentile(steps, 90):.0f}, "
              f"max {steps.max()} ({steps.max() / config.CONTROL_HZ:.1f}s)")
    print(f"phase switches per episode: median {np.median([sw for _, sw in out]):.0f}, "
          f"max {max(sw for _, sw in out)}")
    stages = Counter(STAGES[r.stage] for r in fail)
    if fail:
        print("failures by furthest stage:", dict(stages))
        d = REPO_ROOT / "runs" / "expert_failures"
        d.mkdir(parents=True, exist_ok=True)
        for r in fail:
            np.savez_compressed(d / f"s{r.scene_seed}_e{r.episode_seed}_dart{args.dart}.npz",
                                scene_seed=r.scene_seed, episode_seed=r.episode_seed, actions=r.actions)
        print("failed (scene, episode):", [(r.scene_seed, r.episode_seed) for r in fail][:20])


if __name__ == "__main__":
    main()

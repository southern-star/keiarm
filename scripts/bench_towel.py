"""Towel speed benchmark (milestone 3, Phase 0). Single process: stop other heavy jobs first.

Usage: uv run python scripts/bench_towel.py [--scene 100000]

Runs the scripted half fold once (scene_seed, episode 0) and records its actions, then replays
them twice: physics only, and physics + the policy's three cameras every control step. The
first run itself gives physics + IK. For the rigid milestone 2 scene see scripts/bench_openarm.py.
Episodes per hour in parallel: scripts/run_scripted_towel.py.
"""

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from keiarm import config
from keiarm.towel.env import CAMERAS, TowelEnv
from keiarm.towel.scripted import scripted_fold


def record(scene_seed: int) -> tuple[list[np.ndarray], float, float, int]:
    env = TowelEnv(scene_seed, render=False)
    env.reset(0)
    actions = []
    step = env.step

    def rec(a):
        actions.append(np.array(a, copy=True))
        return step(a)

    env.step = rec
    t = time.perf_counter()
    r = scripted_fold(env)
    wall = time.perf_counter() - t
    env.close()
    assert r.completed, r
    return actions, wall, r.wall_physics, r.steps


def replay(scene_seed: int, actions, render: bool) -> float:
    env = TowelEnv(scene_seed, render=render)
    env.reset(0)
    t = time.perf_counter()
    for a in actions:
        env.step(a)
        if render:
            env.observe()
    wall = time.perf_counter() - t
    env.close()
    return wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=100000)
    args = ap.parse_args()
    actions, wall_ik, _, steps = record(args.scene)
    sim = steps / config.CONTROL_HZ
    wall_phys = replay(args.scene, actions, render=False)
    wall_render = replay(args.scene, actions, render=True)
    print(f"Scripted half fold, scene {args.scene}: {steps} control steps = {sim:.1f} s simulated")
    print("| 測定 | 実時間比 | 1 制御ステップ [ms] |")
    print("|---|---|---|")
    for name, wall in (("物理のみ（行動列の再生）", wall_phys),
                       (f"物理 + 描画（{len(CAMERAS)} 台、160×120）", wall_render),
                       ("物理 + IK（決め打ちの軌道）", wall_ik)):
        print(f"| {name} | {sim / wall:.2f}× | {wall / steps * 1000:.1f} |")


if __name__ == "__main__":
    main()

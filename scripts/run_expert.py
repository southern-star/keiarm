"""Run the expert on one episode, print the phase timeline, optionally write a video.

Usage: uv run python scripts/run_expert.py --scene 100000 --episode 0 --video runs/expert.mp4
"""

import argparse
import os
from itertools import groupby
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.rollout import run_expert
from keiarm.task import STAGES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=100000)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--dart", type=float, default=0.0)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--trace", action="store_true", help="print the state every 10 steps")
    args = ap.parse_args()

    env = KeiArmEnv(args.scene, render=args.video is not None,
                    width=config.IMAGE_WIDTH * args.scale, height=config.IMAGE_HEIGHT * args.scale)
    frames = []

    def on_step(env, info):
        if args.video:
            frames.append(np.concatenate([env.render(c) for c in config.CAMERAS], axis=1))
        if args.trace and env.steps % 10 == 0:
            st = env.privileged()
            print(f"{env.steps:4d} {info.phase:8s} ee={np.round(st.ee_pos, 3)} cube={np.round(st.cube_pos, 3)} "
                  f"grip={st.joints[5]:+.2f} touch={int(st.touch_fixed)}{int(st.touch_moving)}")

    res = run_expert(env, args.episode, dart_sigma=args.dart, on_step=on_step)
    timeline = " ".join(f"{k}x{len(list(g))}" for k, g in groupby(res.phases))
    print(f"scene={args.scene} episode={args.episode} success={res.success} steps={res.steps} "
          f"stage={STAGES[res.stage]}\n{timeline}")
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=config.CONTROL_HZ, macro_block_size=1)
        print(f"wrote {args.video}")


if __name__ == "__main__":
    main()

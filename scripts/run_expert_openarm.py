"""Run the OpenArm expert on one episode, print the phase timeline, optionally write a video.

Usage: uv run python scripts/run_expert_openarm.py --scene 100000 --episode 0 --video runs/openarm_expert.mp4
The video shows the three policy cameras (front, left wrist, right wrist) and an overview.
"""

import argparse
import os
from itertools import groupby
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

from keiarm import config
from keiarm.openarm.env import CAMERAS, OpenArmEnv
from keiarm.openarm.rollout import run_expert
from keiarm.openarm.task import STAGES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=100000)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--dart", type=float, default=0.0)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    w, h = config.IMAGE_WIDTH * args.scale, config.IMAGE_HEIGHT * args.scale
    env = OpenArmEnv(args.scene, render=args.video is not None, width=w, height=h)
    overview = mujoco.MjvCamera()
    overview.lookat[:] = [0.25, 0.0, 0.08]
    overview.distance, overview.azimuth, overview.elevation = 1.1, 200.0, -35.0
    frames = []

    def on_step(env, info):
        if args.video:
            tiles = [env.render(c) for c in CAMERAS]
            env._renderer.update_scene(env.data, overview)
            tiles.append(env._renderer.render())
            frames.append(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0))

    try:
        res = run_expert(env, args.episode, dart_sigma=args.dart, on_step=on_step)
    finally:
        env.close()
    timeline = " ".join(f"{k}x{len(list(g))}" for k, g in groupby(res.phases))
    print(f"scene={args.scene} episode={args.episode} success={res.success} steps={res.steps} "
          f"stage={STAGES[res.stage]} arm_contact_steps={res.arms_touch_steps}\n{timeline}")
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=config.CONTROL_HZ, macro_block_size=1)
        print(f"wrote {args.video}")


if __name__ == "__main__":
    main()

"""Render a contact sheet of generated scenes (both policy cameras) and optionally a video.

Usage: uv run python scripts/show_scenes.py --seeds 0 1 2 3 100000 100001 --out runs/scenes.png
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.paths import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 100000, 100001])
    ap.add_argument("--episode-seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=2, help="render at this multiple of the policy resolution")
    ap.add_argument("--settle-steps", type=int, default=15)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "scenes.png")
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        env = KeiArmEnv(seed, width=config.IMAGE_WIDTH * args.scale, height=config.IMAGE_HEIGHT * args.scale)
        env.reset(args.episode_seed)
        hold = env.joints()
        for _ in range(args.settle_steps):
            env.step(hold)
        obs = env.observe()
        rows.append(np.concatenate([obs.images[c] for c in config.CAMERAS], axis=1))
        env.close()
    ncol = 2 if len(rows) > 3 else 1
    while len(rows) % ncol:
        rows.append(np.zeros_like(rows[0]))
    sheet = np.concatenate([np.concatenate(rows[i::ncol], axis=0) for i in range(ncol)], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, sheet)
    print(f"wrote {args.out} {sheet.shape}")


if __name__ == "__main__":
    main()

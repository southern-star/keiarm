"""Grid of OpenArm scenes: one row per scene, columns = the policy cameras + an overview.

Usage: uv run python scripts/show_scenes_openarm.py [--seeds 0 1 2 ...] [--out docs/openarm/scenes_phase1.png]
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import mujoco
import numpy as np

from keiarm.openarm.robot import wrist_camera
from keiarm.openarm.scene import build_model, reset_data, sample_episode, sample_scene

CAMERAS = ("cam_front", wrist_camera("left"), wrist_camera("right"))


def render_scene(seed: int, episode: int, width: int, height: int) -> np.ndarray:
    scene = sample_scene(seed)
    model = build_model(scene)
    data = mujoco.MjData(model)
    reset_data(model, data, scene, sample_episode(scene, episode))
    renderer = mujoco.Renderer(model, height, width)
    try:
        tiles = []
        for cam in CAMERAS:
            renderer.update_scene(data, cam)
            tiles.append(renderer.render())
        free = mujoco.MjvCamera()
        free.lookat[:] = [0.25, 0.0, 0.05]
        free.distance, free.azimuth, free.elevation = 1.3, 200.0, -45.0
        renderer.update_scene(data, free)
        tiles.append(renderer.render())
    finally:
        renderer.close()
    return np.concatenate(tiles, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--size", type=int, nargs=2, default=(240, 180), metavar=("W", "H"))
    ap.add_argument("--out", default="docs/openarm/scenes_phase1.png")
    args = ap.parse_args()
    rows = [render_scene(s, args.episode, *args.size) for s in args.seeds]
    iio.imwrite(args.out, np.concatenate(rows, axis=0))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

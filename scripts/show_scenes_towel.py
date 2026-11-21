"""Grid of towel scenes (milestone 3): one row per scene.

Columns: the policy's three cameras at the start (cam_front, left wrist, right wrist), an
overview, cam_front after the scripted half fold (the folded half shows the plain back), and
the towel's segmentation mask in that image. Prints each scene's towel parameters and the
towel's share of the cam_front image, flat and folded.

Usage: uv run python scripts/show_scenes_towel.py [--seeds 0 1 2 ...] [--out docs/towel/scenes_phase1.png]
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from multiprocessing import Pool

import imageio.v3 as iio
import mujoco
import numpy as np

from keiarm.towel.env import CAMERAS, TowelEnv
from keiarm.towel.scripted import scripted_fold


def overview() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.22, 0.0, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 1.1, 200.0, -45.0
    return cam


def render_scene(args) -> tuple[np.ndarray, dict]:
    seed, episode, width, height = args
    env = TowelEnv(seed, render=True, width=width, height=height)
    try:
        env.reset(episode)
        tiles = [env.render(cam) for cam in CAMERAS] + [env.render(overview())]
        flat = float(env.towel_mask("cam_front").mean())
        res = scripted_fold(env)
        tiles.append(env.render("cam_front"))
        mask = env.towel_mask("cam_front")
        tiles.append(np.repeat(mask[..., None], 3, axis=2).astype(np.uint8) * 255)
        sc = env.scene
        info = {"seed": seed, "depth_x_width": sc.towel_size, "mass_g": round(sc.towel_mass * 1000, 1),
                "friction": sc.towel_friction, "young": sc.towel_young,
                "towel_share_flat": round(flat, 3), "towel_share_folded": round(float(mask.mean()), 3),
                "fold_completed": res.completed}
    finally:
        env.close()
    return np.concatenate(tiles, axis=1), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--size", type=int, nargs=2, default=(240, 180), metavar=("W", "H"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="docs/towel/scenes_phase1.png")
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        out = pool.map(render_scene, [(s, args.episode, *args.size) for s in args.seeds])
    for _, info in out:
        print(info)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    iio.imwrite(args.out, np.concatenate([row for row, _ in out], axis=0))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

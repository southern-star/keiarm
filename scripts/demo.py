"""Phase 0 smoke test: load the SO-101, sweep every joint, write a two-camera video.

Usage: MUJOCO_GL=egl uv run python scripts/demo.py [--out runs/phase0_demo.mp4]
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

from keiarm.paths import REPO_ROOT
from keiarm.robot import JOINT_NAMES, WRIST_CAMERA, so101_spec

FPS = 30
W, H = 320, 240


def build_model() -> mujoco.MjModel:
    spec = so101_spec()
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                   rgba=[0.75, 0.75, 0.72, 1])
    world.add_light(pos=[0.2, 0, 1.5], dir=[0, 0, -1])
    # Fixed camera looking at the workspace in front of the arm; refined in Phase 1.
    world.add_camera(name="cam_front", pos=[0.75, -0.45, 0.45], mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
                     targetbody="upper_arm", fovy=45)
    return spec.compile()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "phase0_demo.mp4")
    ap.add_argument("--seconds-per-joint", type=float, default=2.0)
    args = ap.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    assert [model.actuator(i).name for i in range(model.nu)] == list(JOINT_NAMES)
    substeps = round(1 / FPS / model.opt.timestep)  # 0.005 s -> 7 (not exact; see docs/so101_model.md)
    renderer = mujoco.Renderer(model, height=H, width=W)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lo, hi = model.actuator_ctrlrange.T
    n = int(args.seconds_per_joint * FPS)
    frames = 0
    t0 = time.perf_counter()
    with imageio.get_writer(args.out, fps=FPS, macro_block_size=1) as writer:
        for j in range(model.nu):
            for k in range(n):
                # One sine period scaled to 60% of the joint's range, starting and ending at 0.
                s = np.sin(2 * np.pi * k / n)
                data.ctrl[:] = 0
                data.ctrl[j] = 0.6 * (hi[j] if s > 0 else -lo[j]) * s
                mujoco.mj_step(model, data, nstep=substeps)
                tiles = []
                for cam in ("cam_front", WRIST_CAMERA):
                    renderer.update_scene(data, camera=cam)
                    tiles.append(renderer.render())
                writer.append_data(np.concatenate(tiles, axis=1))
                frames += 1
    dt = time.perf_counter() - t0
    print(f"wrote {args.out} ({frames} frames, {frames / FPS:.1f}s sim) in {dt:.1f}s "
          f"-> {frames / FPS / dt:.1f}x realtime incl. 2 cameras + encoding")
    print(f"GL backend: {os.environ['MUJOCO_GL']}  final qpos: {np.round(data.qpos, 3)}")


if __name__ == "__main__":
    main()

"""OpenArm speed benchmark (milestone 2, Phase 0).

Usage: uv run python scripts/bench_openarm.py [--steps 600]
Single process. Actions are a slow random walk around the start pose (both arms move).
Rendering: cam_front + both wrist cameras at the policy resolution, every control step.
"""

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from keiarm import config
from keiarm.openarm.robot import wrist_camera
from keiarm.openarm.scene import build_model, reset_data, sample_episode, sample_scene

CAMERAS = ("cam_front", wrist_camera("left"), wrist_camera("right"))


SCENE = sample_scene(0)


def _run(model, steps: int, renderer: mujoco.Renderer | None) -> float:
    data = mujoco.MjData(model)
    reset_data(model, data, SCENE, sample_episode(SCENE, 0))
    rng = np.random.default_rng(0)
    target = data.ctrl.copy()
    lo, hi = model.actuator_ctrlrange.T
    t0 = time.perf_counter()
    for _ in range(steps):
        target = np.clip(target + rng.normal(0, 0.02, model.nu), lo, hi)
        data.ctrl[:] = target
        mujoco.mj_step(model, data, nstep=config.SUBSTEPS)
        if renderer is not None:
            for cam in CAMERAS:
                renderer.update_scene(data, cam)
                renderer.render()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    t0 = time.perf_counter()
    model = build_model(SCENE)
    build = time.perf_counter() - t0
    sim_seconds = args.steps / config.CONTROL_HZ
    phys = _run(model, args.steps, None)
    renderer = mujoco.Renderer(model, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    try:
        rend = _run(model, args.steps, renderer)
    finally:
        renderer.close()
    print(f"MuJoCo {mujoco.__version__}, {args.steps} control steps, nmeshface={model.nmeshface}")
    print("| 測定 | 実時間比 | 制御ステップ/秒 |")
    print("|---|---|---|")
    print(f"| 物理のみ | {sim_seconds / phys:.1f}× | {args.steps / phys:.0f} |")
    print(f"| 物理 + 描画（{len(CAMERAS)} 台、{config.IMAGE_WIDTH}×{config.IMAGE_HEIGHT}） | "
          f"{sim_seconds / rend:.1f}× | {args.steps / rend:.0f} |")
    print(f"| シーン構築（MjSpec → MjModel） | {build:.2f} 秒 | |")


if __name__ == "__main__":
    main()

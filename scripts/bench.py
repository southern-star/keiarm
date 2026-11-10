"""Speed benchmark: prints the markdown table used in the README.

Usage: uv run python scripts/bench.py [--steps 600] [--scenes 8]
Single process. Actions are a slow random walk around the start pose (arm keeps moving).
"""

import argparse
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.scene import build_model, sample_scene


def _cpu() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def _gpu() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown"


def _run(env: KeiArmEnv, steps: int, render: bool) -> float:
    env.reset(0)
    rng = np.random.default_rng(0)
    target = env.joints()
    lo, hi = env.ctrl_range.T
    t0 = time.perf_counter()
    for _ in range(steps):
        target = np.clip(target + rng.normal(0, 0.03, 6), lo, hi)
        env.step(target)
        if render:
            env.observe()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--scenes", type=int, default=8)
    args = ap.parse_args()

    sim_seconds = args.steps / config.CONTROL_HZ
    phys, rend, gen, gen_r = [], [], [], []
    for seed in range(args.scenes):
        t0 = time.perf_counter()
        build_model(sample_scene(seed))
        gen.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        env = KeiArmEnv(seed)
        gen_r.append(time.perf_counter() - t0)
        phys.append(_run(env, args.steps, render=False))
        rend.append(_run(env, args.steps, render=True))
        env.close()

    def rt(ts):  # median realtime factor
        return sim_seconds / float(np.median(ts))

    w, h = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
    print(f"MuJoCo {mujoco.__version__}, {_cpu()}, {_gpu()} (EGL), single process, "
          f"{args.scenes} scenes x {args.steps} control steps, medians\n")
    print("| 測定 | 実時間比 | 制御ステップ/秒 |")
    print("|---|---|---|")
    print(f"| 物理のみ（{config.CONTROL_HZ} Hz 制御、{config.SUBSTEPS} サブステップ） | "
          f"{rt(phys):.0f}x | {rt(phys) * config.CONTROL_HZ:.0f} |")
    print(f"| 物理 + カメラ 2 台（{w}x{h}） | {rt(rend):.0f}x | {rt(rend) * config.CONTROL_HZ:.0f} |")
    print("\n| 測定 | 時間 |\n|---|---|")
    print(f"| シーン生成（サンプリング + モデルのコンパイル） | {np.median(gen) * 1000:.0f} ms |")
    print(f"| 同上 + レンダラ生成 | {np.median(gen_r) * 1000:.0f} ms |")


if __name__ == "__main__":
    main()

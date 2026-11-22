"""Browser speed of the towel page per mesh N (milestone 3, Phase 7), in headless Chrome.

For each N and scene, three numbers, as a viewer gets them:
1. physics only: the expert's recorded actions in the physics worker, nothing else drawn but the
   view (window.keiarm.realtimeRun) -> simulated seconds per wall-clock second
2. physics + the three policy cameras, one thread: time per control step (benchStep)
3. the learned policy in real time: physics in its worker, inference (onnxruntime-web) in its
   worker, cameras and view here (realtimeModelRun), timed from the first plan on
Also the blocking closed loop's per-step cost (physics + cameras + inference / 10, one thread).

Usage: npm --prefix web run build && uv run python tools/browser_speed_towel.py --gpu [--meshes 10 11] [--scenes 100000 100001]
(needs web/public/towel_scenes exported for those meshes and web/public/models/towel.onnx)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from browser_check import DIST, serve

from keiarm.paths import REPO_ROOT

GPU_ARGS = ["--use-gl=angle", "--use-angle=gl-egl", "--enable-gpu"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", type=int, nargs="+", default=[10, 11])
    ap.add_argument("--scenes", type=int, nargs="+", default=[100000, 100001, 100002])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--gpu", action="store_true", help="WebGL on the machine's GPU (default: SwiftShader)")
    args = ap.parse_args()
    server, url = serve(DIST)
    rows, errors = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True, args=[
            "--ignore-gpu-blocklist", *(GPU_ARGS if args.gpu else ["--enable-unsafe-swiftshader"])])
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        page.set_default_timeout(600_000)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        for n in args.meshes:
            for seed in args.scenes:
                page.goto(f"{url}towel.html?scene={seed}&n={n}")
                page.evaluate("() => window.keiarm.ready")
                info = page.evaluate("() => window.keiarm.policyReady()")
                phys = page.evaluate(f"() => window.keiarm.realtimeRun(0, {args.seconds})")
                bench = page.evaluate("() => window.keiarm.benchStep(0, 150, {mainView: false})")
                model = page.evaluate(f"() => window.keiarm.realtimeModelRun(0, {args.seconds})")
                block = page.evaluate("() => window.keiarm.runModelEpisode(0)")
                row = {"n": n, "scene": seed, "backend": info["backend"],
                       "physics_only_realtime": phys["realtime"], "physics_worker_ms": phys["meanStepMs"],
                       "physics_ms": bench["physicsMs"], "cameras_ms": bench["camerasMs"],
                       "physics_cameras_realtime": 1000 / 30 / (bench["physicsMs"] + bench["camerasMs"]),
                       "model_realtime": model["realtime"], "model_fps": model["fps"],
                       "model_physics_worker_ms": model["meanStepMs"], "infer_ms": model["meanInferMs"],
                       "model_succeeded": model["succeeded"],
                       "blocking_ms_per_step": block["physicsMsPerStep"] + block["camerasMsPerStep"] + block["inferMsPerStep"],
                       "blocking_success": block["success"]}
                rows.append(row)
                print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        browser.close()
    server.shutdown()
    summary = {}
    for n in args.meshes:
        r = [x for x in rows if x["n"] == n]
        summary[n] = {k: float(np.median([x[k] for x in r])) for k in (
            "physics_only_realtime", "physics_worker_ms", "physics_ms", "cameras_ms", "physics_cameras_realtime",
            "model_realtime", "model_fps", "model_physics_worker_ms", "infer_ms", "blocking_ms_per_step")}
        s = summary[n]
        print(f"N = {n} (median of {len(r)} scenes): physics only {s['physics_only_realtime']:.2f} x real time "
              f"({s['physics_worker_ms']:.1f} ms/step in the worker); physics + 3 cameras on one thread "
              f"{s['physics_ms']:.1f} + {s['cameras_ms']:.1f} ms/step = {s['physics_cameras_realtime']:.2f} x; "
              f"with the policy {s['model_realtime']:.2f} x real time at {s['model_fps']:.0f} fps "
              f"(physics {s['model_physics_worker_ms']:.1f} ms/step, inference {s['infer_ms']:.0f} ms); "
              f"blocking loop {s['blocking_ms_per_step']:.1f} ms/step")
    out = REPO_ROOT / "docs" / "towel" / f"browser_speed_model{'_gpu' if args.gpu else ''}.json"
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(f"wrote {out}")
    if errors:
        print("console errors:", *sorted(set(errors))[:8], sep="\n  ")


if __name__ == "__main__":
    main()

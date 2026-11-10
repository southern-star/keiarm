"""Headless-browser checks for the web demo (design doc 11.3 and Phase 4).

1. Replay: every exported expert episode is replayed in MuJoCo WASM inside the page and must
   end in success.
2. Rendering: the same simulator states are drawn by mujoco.Renderer (Python) and by the
   page's three.js policy cameras; images are compared pixel by pixel.

Usage:
  (cd web && npm run build)
  uv run python tools/browser_check.py            # writes docs/render_compare.{png,json}
Uses the system Google Chrome through Playwright (no browser download).
"""

import argparse
import functools
import http.server
import json
import os
import threading
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from playwright.sync_api import sync_playwright

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.expert import Expert
from keiarm.paths import REPO_ROOT

DIST = REPO_ROOT / "web" / "dist"
CAPTURE_STEPS = (0, 20, 40, 60)  # control steps into the expert episode at which to compare


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(_Quiet, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def python_states(scene_seed: int, episode_seed: int):
    """Yield (step, qpos, mocap_pos, {camera: image}) along an expert episode."""
    env = KeiArmEnv(scene_seed)
    env.reset(episode_seed)
    expert = Expert(env.model, env.scene)
    for step in range(max(CAPTURE_STEPS) + 1):
        if step in CAPTURE_STEPS:
            # mj_forward so that rendered poses match qpos exactly (as the browser will do).
            import mujoco
            mujoco.mj_forward(env.model, env.data)
            yield step, env.data.qpos.copy(), env.data.mocap_pos.copy(), env.observe().images
        env.step(expert.act(env.data)[0])
    env.close()


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    worst = diff.max(axis=2)
    return {"mean_abs": float(diff.mean()), "p99": float(np.percentile(worst, 99)),
            "frac_gt8": float((worst > 8).mean()), "frac_gt32": float((worst > 32).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+", help="default: all exported scenes")
    ap.add_argument("--sheet-scenes", type=int, default=4, help="scenes shown in the comparison image")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    if not (DIST / "index.html").exists():
        raise SystemExit("web/dist is missing: run `npm run build` in web/ first")

    index = json.loads((DIST / "scenes" / "index.json").read_text())
    seeds = args.scenes or index["scenes"]
    server, url = serve(DIST)
    replay, render, sheet_rows, console = [], [], [], []
    w, h = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=not args.headed,
                                     args=["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
        for i, seed in enumerate(seeds):
            page.goto(f"{url}?scene={seed}")
            page.evaluate("() => window.keiarm.ready")
            if i == 0:
                gl = page.evaluate("""() => { const g = document.createElement('canvas').getContext('webgl2');
                    const e = g.getExtension('WEBGL_debug_renderer_info');
                    return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; }""")
                version = page.evaluate("() => window.keiarm.app.mujoco.mj_versionString()")
                print(f"browser MuJoCo {version}, WebGL renderer: {gl}")
            episodes = page.evaluate("() => window.keiarm.app.bundle.episodes.map(e => e.episode_seed)")
            for k, ep in enumerate(episodes):
                r = page.evaluate("(k) => window.keiarm.replayNow(k)", k)
                err = float(np.linalg.norm(np.subtract(r["cubePos"], r["pythonCubePos"])))
                replay.append({"scene": seed, "episode": ep, "browser_success": r["success"],
                               "python_success": r["pythonSuccess"], "cube_err_mm": round(err * 1000, 3)})
            for step, qpos, mocap, images in python_states(seed, episodes[0]):
                got = page.evaluate("([q, m]) => window.keiarm.renderState(q, m)",
                                    [qpos.tolist(), mocap.ravel().tolist()])
                for cam in config.CAMERAS:
                    web = np.array(got[cam], dtype=np.uint8).reshape(h, w, 4)[..., :3]
                    render.append({"scene": seed, "step": step, "camera": cam, **compare(images[cam], web)})
                    if i < args.sheet_scenes and step in (0, 40):
                        diff = np.clip(np.abs(images[cam].astype(np.int16) - web) * 4, 0, 255).astype(np.uint8)
                        sheet_rows.append(np.concatenate([images[cam], web, diff], axis=1))
        browser.close()
    server.shutdown()

    ok = sum(r["browser_success"] for r in replay)
    print(f"\nreplay in browser: {ok}/{len(replay)} episodes succeed "
          f"(python: {sum(r['python_success'] for r in replay)}/{len(replay)}), "
          f"max final cube position difference {max(r['cube_err_mm'] for r in replay):.2f} mm")
    print("\nrender comparison (0-255 scale), mujoco.Renderer vs three.js:")
    summary = {}
    for cam in config.CAMERAS:
        rows = [r for r in render if r["camera"] == cam]
        summary[cam] = {k: float(np.mean([r[k] for r in rows])) for k in ("mean_abs", "p99", "frac_gt8", "frac_gt32")}
        summary[cam]["worst_mean_abs"] = float(max(r["mean_abs"] for r in rows))
        s = summary[cam]
        print(f"  {cam:10s} images={len(rows)} mean|diff|={s['mean_abs']:.2f} (worst image {s['worst_mean_abs']:.2f}) "
              f"pixels off by >8: {100 * s['frac_gt8']:.1f}%  >32: {100 * s['frac_gt32']:.1f}%")
    if console:
        print("\nbrowser console:", *sorted(set(console))[:10], sep="\n  ")

    docs = REPO_ROOT / "docs"
    if sheet_rows:
        half = (len(sheet_rows) + 1) // 2
        cols = [np.concatenate(sheet_rows[:half], axis=0), np.concatenate(sheet_rows[half:], axis=0)]
        height = max(c.shape[0] for c in cols)
        cols = [np.pad(c, ((0, height - c.shape[0]), (0, 0), (0, 0))) for c in cols]
        sheet = np.concatenate(cols, axis=1)
        imageio.imwrite(docs / "render_compare.png", np.kron(sheet, np.ones((2, 2, 1), dtype=np.uint8)))
    (docs / "render_compare.json").write_text(json.dumps(
        {"replay": replay, "render_summary": summary, "render": render}, indent=1))
    print(f"\nwrote {docs / 'render_compare.png'} (columns: MuJoCo | three.js | 4x difference) and render_compare.json")
    raise SystemExit(0 if ok == len(replay) else 1)


if __name__ == "__main__":
    main()

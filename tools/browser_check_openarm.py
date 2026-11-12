"""Headless-browser checks for the OpenArm page (milestone 2, Phase 4).

1. Replay: every exported expert episode (clean and disturbed) is replayed in MuJoCo WASM
   inside the page and must succeed, at the same step as in Python.
2. Rendering: the same simulator states are drawn by mujoco.Renderer (Python, original
   meshes) and by the page's three.js policy cameras (decimated meshes). To tell the two
   causes apart, Python also renders the states with the decimated meshes.
3. Size: bytes the page downloads (raw and gzip), and time per control step for physics +
   three policy cameras + the free view.

Usage:
  uv run python scripts/export_web_openarm.py && npm --prefix web run build
  uv run python tools/browser_check_openarm.py      # writes docs/openarm/render_compare.{png,json}
"""

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from browser_check import DIST, compare, serve
from playwright.sync_api import sync_playwright

from keiarm import config
from keiarm.openarm.env import CAMERAS, OpenArmEnv
from keiarm.openarm.expert import Expert
from keiarm.paths import REPO_ROOT

CAPTURE_STEPS = (0, 50, 100, 150, 200)  # control steps into the expert episode


def python_states(scene_seed: int, episode_seed: int):
    """Yield (step, qpos, mocap_pos, {camera: image}) along a clean expert episode."""
    env = OpenArmEnv(scene_seed)
    env.reset(episode_seed)
    expert = Expert(env.model, env.scene)
    for step in range(max(CAPTURE_STEPS) + 1):
        if step in CAPTURE_STEPS:
            mujoco.mj_forward(env.model, env.data)  # poses exactly from qpos, as the browser does
            yield step, env.data.qpos.copy(), env.data.mocap_pos.copy(), env.observe().images
        env.step(expert.act(env.data)[0])
    env.close()


class DecimatedRenderer:
    """mujoco.Renderer on the exported model (decimated meshes), configured like OpenArmEnv."""

    def __init__(self, bundle: dict, mesh_root: Path):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / bundle["mesh_dir"]).symlink_to(mesh_root.resolve(), target_is_directory=True)
        (d / "scene.xml").write_text(bundle["xml"])
        self.model = mujoco.MjModel.from_xml_path(str(d / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.r = mujoco.Renderer(self.model, height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

    def render(self, qpos, mocap) -> dict:
        self.data.qpos[:] = qpos
        self.data.mocap_pos[:] = mocap
        mujoco.mj_forward(self.model, self.data)
        out = {}
        for cam in CAMERAS:
            self.r.update_scene(self.data, camera=cam)
            out[cam] = self.r.render()
        return out

    def close(self):
        self.r.close()
        self.tmp.cleanup()


def page_bytes(index: dict, scene_seed: int) -> dict:
    """What opening the page for one scene downloads (from dist/), raw and gzip-compressed."""
    def size(paths):
        raw = sum(p.stat().st_size for p in paths)
        gz = sum(len(gzip.compress(p.read_bytes(), 6)) for p in paths)
        return {"raw": raw, "gzip": gz}
    assets = DIST / "assets"
    meshes = [p for p in (assets / "openarm").rglob("*") if p.is_file() and p.name != "LICENSE"]
    code = [DIST / "openarm.html", *assets.glob("openarm-*.js"), *assets.glob("__vite-browser-external-*.js"),
            *assets.glob("mujoco-*.wasm")]
    scene = [DIST / "openarm_scenes" / f"{scene_seed}.json", DIST / "openarm_scenes" / "index.json"]
    out = {"meshes": size(meshes), "scene_json": size(scene), "code_and_wasm": size(code)}
    out["total"] = {k: sum(v[k] for v in out.values()) for k in ("raw", "gzip")}
    out["visual_faces"] = index["visual_faces_after"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+", help="default: all exported scenes")
    ap.add_argument("--sheet-scenes", type=int, default=3, help="scenes shown in the comparison image")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--gpu", action="store_true",
                    help="use the machine's GPU (ANGLE on EGL) instead of SwiftShader; the render comparison "
                         "and docs/openarm/ outputs are for SwiftShader, so --gpu only prints")
    args = ap.parse_args()
    if not (DIST / "openarm.html").exists() or not (DIST / "openarm_scenes" / "index.json").exists():
        raise SystemExit("run `uv run python scripts/export_web_openarm.py && npm --prefix web run build` first")

    index = json.loads((DIST / "openarm_scenes" / "index.json").read_text())
    seeds = args.scenes or index["scenes"]
    server, url = serve(DIST)
    replay, render, sheet_rows, console, bench = [], [], [], [], []
    w, h = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
    with sync_playwright() as pw:
        gl_args = (["--use-gl=angle", "--use-angle=gl-egl", "--enable-gpu"] if args.gpu
                   else ["--enable-unsafe-swiftshader"])
        browser = pw.chromium.launch(channel="chrome", headless=not args.headed,
                                     args=["--ignore-gpu-blocklist", *gl_args])
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
        for i, seed in enumerate(seeds):
            page.goto(f"{url}openarm.html?scene={seed}")
            page.evaluate("() => window.keiarm.ready")
            if i == 0:
                gl = page.evaluate("""() => { const g = document.createElement('canvas').getContext('webgl2');
                    const e = g.getExtension('WEBGL_debug_renderer_info');
                    return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; }""")
                version = page.evaluate("() => window.keiarm.app.mujoco.mj_versionString()")
                print(f"browser MuJoCo {version}, WebGL renderer: {gl}")
            bundle = page.evaluate("() => window.keiarm.app.bundle")
            for k, ep in enumerate(bundle["episodes"]):
                r = page.evaluate("(k) => window.keiarm.replayNow(k)", k)
                err = float(np.linalg.norm(np.subtract(r["barPos"], r["pythonBarPos"])))
                kind = ep["disturbance"]["kind"] if ep["disturbance"] else "none"
                replay.append({"scene": seed, "episode": ep["episode_seed"], "disturb": kind,
                               "browser_success": r["success"], "python_success": r["pythonSuccess"],
                               "browser_step": r["successStep"], "python_step": r["pythonSteps"],
                               "bar_err_mm": round(err * 1000, 3),
                               "realtime": round(r["steps"] / config.CONTROL_HZ / (r["ms"] / 1000), 1)})
            bench.append(page.evaluate("() => window.keiarm.benchStep(0, 120)"))
            dec = DecimatedRenderer(bundle, DIST / "assets" / "openarm")
            ep0 = bundle["episodes"][0]["episode_seed"]
            for step, qpos, mocap, images in python_states(seed, ep0):
                got = page.evaluate("([q, m]) => window.keiarm.renderState(q, m)",
                                    [qpos.tolist(), mocap.ravel().tolist()])
                dimgs = dec.render(qpos, mocap)
                for cam in CAMERAS:
                    web = np.array(got[cam], dtype=np.uint8).reshape(h, w, 4)[..., :3]
                    render.append({"scene": seed, "step": step, "camera": cam,
                                   "python_vs_web": compare(images[cam], web),
                                   "python_vs_python_decimated": compare(images[cam], dimgs[cam]),
                                   "python_decimated_vs_web": compare(dimgs[cam], web)})
                if i < args.sheet_scenes and step in (0, 100, 200):
                    row = []
                    for cam in CAMERAS:
                        web = np.array(got[cam], dtype=np.uint8).reshape(h, w, 4)[..., :3]
                        diff = np.clip(np.abs(images[cam].astype(np.int16) - web) * 4, 0, 255).astype(np.uint8)
                        row.append(np.concatenate([images[cam], web, diff], axis=0))
                    sheet_rows.append(np.concatenate(row, axis=1))
            dec.close()
        browser.close()
    server.shutdown()

    ok = sum(r["browser_success"] for r in replay)
    same_step = sum(r["browser_step"] == r["python_step"] for r in replay)
    print(f"\nreplay in browser: {ok}/{len(replay)} episodes succeed "
          f"(python: {sum(r['python_success'] for r in replay)}/{len(replay)}); success confirmed at the same "
          f"step in {same_step}, max step difference "
          f"{max(abs((r['browser_step'] or 10**6) - r['python_step']) for r in replay)}; "
          f"max final bar position difference {max(r['bar_err_mm'] for r in replay):.2f} mm; "
          f"physics {np.median([r['realtime'] for r in replay]):.0f}x real time")
    print("\nrender comparison (0-255 scale):")
    summary = {}
    for pair in ("python_vs_web", "python_vs_python_decimated", "python_decimated_vs_web"):
        summary[pair] = {}
        for cam in CAMERAS:
            rows = [r[pair] for r in render if r["camera"] == cam]
            s = {k: float(np.mean([r[k] for r in rows])) for k in ("mean_abs", "p99", "frac_gt8", "frac_gt32")}
            s["worst_mean_abs"] = float(max(r["mean_abs"] for r in rows))
            s["images"] = len(rows)
            summary[pair][cam] = s
            print(f"  {pair:27s} {cam:19s} n={len(rows)} mean|diff|={s['mean_abs']:.2f} "
                  f"(worst {s['worst_mean_abs']:.2f}) >8: {100 * s['frac_gt8']:.1f}%  >32: {100 * s['frac_gt32']:.1f}%")
    timing = {k: float(np.median([b[k] for b in bench])) for k in ("physicsMs", "camerasMs", "mainViewMs")}
    print(f"\nper control step (median over scenes, this browser): physics {timing['physicsMs']:.1f} ms, "
          f"3 policy cameras {timing['camerasMs']:.1f} ms, free view {timing['mainViewMs']:.1f} ms "
          f"(budget at 30 Hz: {1000 / config.CONTROL_HZ:.1f} ms)")
    size = page_bytes(index, seeds[0])
    print("download for one scene: " + ", ".join(
        f"{k} {v['raw'] / 1e6:.2f} MB (gzip {v['gzip'] / 1e6:.2f})" for k, v in size.items() if isinstance(v, dict)))
    if console:
        print("\nbrowser console:", *sorted(set(console))[:10], sep="\n  ")

    if args.gpu:
        raise SystemExit(0 if ok == len(replay) else 1)
    docs = REPO_ROOT / "docs" / "openarm"
    docs.mkdir(parents=True, exist_ok=True)
    if sheet_rows:
        sheet = np.concatenate(sheet_rows, axis=0)
        imageio.imwrite(docs / "render_compare.png", np.kron(sheet, np.ones((2, 2, 1), dtype=np.uint8)))
    (docs / "render_compare.json").write_text(json.dumps(
        {"webgl": gl, "replay": replay, "render_summary": summary, "timing_ms": timing, "bench": bench,
         "download": size, "render": render}, indent=1))
    print(f"\nwrote {docs / 'render_compare.png'} (per camera: MuJoCo / three.js / 4x difference) and render_compare.json")
    raise SystemExit(0 if ok == len(replay) else 1)


if __name__ == "__main__":
    main()

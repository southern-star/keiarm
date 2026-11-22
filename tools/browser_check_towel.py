"""Headless-browser checks for the towel page (milestone 3, Phase 4).

1. Replay: every exported expert episode (clean and disturbed, coarse mesh N_web) is replayed
   in MuJoCo WASM inside the page: success, the step it is confirmed and the grabs must match
   Python.
2. Rendering (check V4 b): the same simulator states are drawn by mujoco.Renderer (Python, with
   the back faces of keiarm/towel/render.py) and by the page's three.js policy cameras. Python
   also renders them with the decimated meshes the browser uses, to tell the causes apart.
3. Speed: time per control step for physics + three policy cameras + the free view, and a
   real-time run (requestAnimationFrame, as a viewer sees it): frames per second and simulated
   seconds per wall-clock second.
4. Records (N = 16, no physics): the page's success check on the recorded frames must agree
   with the Python evaluation.
5. Size: bytes the page downloads (raw and gzip).

Usage:
  uv run python scripts/export_web_towel.py --meshes 11 && uv run python scripts/export_towel_records.py
  npm --prefix web run build && uv run python tools/browser_check_towel.py [--gpu]
  (writes docs/towel/render_compare.{png,json} unless --gpu)
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
from keiarm.paths import REPO_ROOT
from keiarm.towel.env import CAMERAS, TowelEnv
from keiarm.towel.expert import Expert
from keiarm.towel.render import BacksideDrawer
from keiarm.towel.scene import TowelIndex, sample_scene

CAPTURE_STEPS = (0, 50, 100, 150, 200)  # control steps into the expert episode


def python_states(scene_seed: int, episode_seed: int, n: int):
    """Yield (step, qpos, {camera: image}, {camera: towel mask}) along a clean expert episode."""
    env = TowelEnv(scene_seed, n=n)
    env.reset(episode_seed)
    expert = Expert(env.model, env.scene)
    try:
        for step in range(max(CAPTURE_STEPS) + 1):
            if step in CAPTURE_STEPS:
                mujoco.mj_forward(env.model, env.data)
                yield (step, env.data.qpos.copy(), env.observe().images,
                       {cam: env.towel_mask(cam) for cam in CAMERAS})
            env.step(expert.act(env.data)[0])
    finally:
        env.close()


class DecimatedRenderer:
    """mujoco.Renderer on the exported model (decimated meshes), configured like TowelEnv."""

    def __init__(self, bundle: dict, mesh_root: Path):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / bundle["mesh_dir"]).symlink_to(mesh_root.resolve(), target_is_directory=True)
        (d / "scene.xml").write_text(bundle["xml"])
        self.model = mujoco.MjModel.from_xml_path(str(d / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.back = BacksideDrawer(self.model, TowelIndex(self.model), sample_scene(bundle["scene"]["scene_seed"]).towel_back_rgb)
        self.r = mujoco.Renderer(self.model, height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

    def render(self, qpos) -> dict:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        out = {}
        for cam in CAMERAS:
            self.r.update_scene(self.data, camera=cam)
            self.back.add(self.r.scene, self.data)
            out[cam] = self.r.render()
        return out

    def close(self):
        self.r.close()
        self.tmp.cleanup()


def page_bytes(name: str, record: str | None) -> dict:
    """What opening the page for one scene (or one record) downloads, raw and gzip."""
    def size(paths):
        paths = [p for p in paths if p.exists()]
        raw = sum(p.stat().st_size for p in paths)
        gz = sum(len(gzip.compress(p.read_bytes(), 6)) for p in paths)
        return {"raw": raw, "gzip": gz}
    assets = DIST / "assets"
    meshes = [p for p in (assets / "openarm").rglob("*") if p.is_file() and p.name != "LICENSE"]
    code = [DIST / "towel.html", *assets.glob("towel-*.js"), *assets.glob("__vite-browser-external-*.js"),
            *assets.glob("mujoco-*.wasm")]
    out = {"meshes": size(meshes), "code_and_wasm": size(code),
           "scene_json": size([DIST / "towel_scenes" / f"{name}.json", DIST / "towel_scenes" / "index.json"])}
    if record:
        out["record_json"] = size([DIST / "towel_records" / f"{record}.json", DIST / "towel_records" / "index.json"])
        out["all_records"] = size(list((DIST / "towel_records").glob("*.json")))
    out["total_live"] = {k: out["meshes"][k] + out["code_and_wasm"][k] + out["scene_json"][k] for k in ("raw", "gzip")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="towel mesh N (default: the index's default)")
    ap.add_argument("--scenes", type=int, nargs="+", help="default: all exported scenes")
    ap.add_argument("--sheet-scenes", type=int, default=3)
    ap.add_argument("--realtime-seconds", type=float, default=8.0)
    ap.add_argument("--physics", choices=("worker", "main"), default="worker",
                    help="where the page runs the physics in the real-time run (default: a Web Worker)")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--gpu", action="store_true",
                    help="use the machine's GPU (ANGLE on EGL) instead of SwiftShader; only prints")
    args = ap.parse_args()
    if not (DIST / "towel.html").exists() or not (DIST / "towel_scenes" / "index.json").exists():
        raise SystemExit("run `uv run python scripts/export_web_towel.py && npm --prefix web run build` first")
    index = json.loads((DIST / "towel_scenes" / "index.json").read_text())
    n = args.n or index["default_n"]
    seeds = args.scenes or index["scenes"]
    rec_index_path = DIST / "towel_records" / "index.json"
    records = json.loads(rec_index_path.read_text())["records"] if rec_index_path.exists() else []
    server, url = serve(DIST)
    replay, render, sheet_rows, console, bench, realtime, rec_rows = [], [], [], [], [], [], []
    w, h = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
    with sync_playwright() as pw:
        gl_args = (["--use-gl=angle", "--use-angle=gl-egl", "--enable-gpu"] if args.gpu
                   else ["--enable-unsafe-swiftshader"])
        browser = pw.chromium.launch(channel="chrome", headless=not args.headed,
                                     args=["--ignore-gpu-blocklist", *gl_args])
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
        gl = "unknown"
        for i, seed in enumerate(seeds):
            page.goto(f"{url}towel.html?scene={seed}&n={n}&physics={args.physics}")
            page.evaluate("() => window.keiarm.ready")
            if i == 0:
                gl = page.evaluate("""() => { const g = document.createElement('canvas').getContext('webgl2');
                    const e = g.getExtension('WEBGL_debug_renderer_info');
                    return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; }""")
                print(f"browser MuJoCo {page.evaluate('() => window.keiarm.app.mujoco.mj_versionString()')}, "
                      f"WebGL renderer: {gl}, towel N = {n}")
            bundle = page.evaluate("() => window.keiarm.app.bundle")
            for k, ep in enumerate(bundle["episodes"]):
                r = page.evaluate("(k) => window.keiarm.replayNow(k)", k)
                kind = ep["disturbance"]["kind"] if ep["disturbance"] else "none"
                replay.append({"scene": seed, "episode": ep["episode_seed"], "disturb": kind,
                               "browser_success": r["success"], "python_success": r["pythonSuccess"],
                               "browser_step": r["successStep"], "python_step": r["pythonSteps"],
                               "browser_grabs": r["grabs"], "python_grabs": r["pythonGrabs"],
                               "vert_err_mm": round(r["vertErr"] * 1000, 2),
                               "realtime": round(r["steps"] / config.CONTROL_HZ / (r["ms"] / 1000), 2)})
            bench.append(page.evaluate("() => window.keiarm.benchStep(0, 150)"))
            if i < 2:
                realtime.append({"scene": seed, **page.evaluate(
                    f"() => window.keiarm.realtimeRun(0, {args.realtime_seconds})")})
            dec = DecimatedRenderer(bundle, DIST / "assets" / "openarm")
            for step, qpos, images, masks in python_states(seed, bundle["episodes"][0]["episode_seed"], n):
                got = page.evaluate("(q) => window.keiarm.renderState(q)", qpos.tolist())
                dimgs = dec.render(qpos)
                for cam in CAMERAS:
                    web = np.array(got[cam], dtype=np.uint8).reshape(h, w, 4)[..., :3]
                    m = masks[cam]
                    towel = (float(np.abs(images[cam][m].astype(np.int16) - web[m]).mean()) if m.any() else None)
                    render.append({"scene": seed, "step": step, "camera": cam, "towel_pixels": int(m.sum()),
                                   "towel_mean_abs": towel,
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
        for rec in records:
            page.goto(f"{url}towel.html?mode=record")
            page.evaluate("() => window.keiarm.ready")
            r = page.evaluate("(name) => window.keiarm.recordFrames(name)", rec["name"])
            rec_rows.append({"record": rec["name"], "browser_success": r["success"], "browser_step": r["step"],
                             "python_success": r["python"]["success"], "python_step": r["python"]["steps"],
                             "frames": r["frames"], "bytes": rec["bytes"]})
        browser.close()
    server.shutdown()

    ok = sum(r["browser_success"] == r["python_success"] for r in replay)
    same = sum(r["browser_step"] == r["python_step"] for r in replay if r["python_success"])
    print(f"\nreplay in browser (N = {n}): success agrees with Python in {ok}/{len(replay)} episodes "
          f"(python successes {sum(r['python_success'] for r in replay)}); confirmed at the same step in {same}, "
          f"grabs agree in {sum(r['browser_grabs'] == r['python_grabs'] for r in replay)}; "
          f"final towel max vertex difference median {np.median([r['vert_err_mm'] for r in replay]):.2f} mm "
          f"(max {max(r['vert_err_mm'] for r in replay):.2f})")
    for r in replay:
        if r["browser_success"] != r["python_success"] or r["browser_step"] != r["python_step"]:
            print("  differs:", r)
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
    print("  towel pixels only (python vs web, mask from Python segmentation; edges of the towel included):")
    for cam in CAMERAS:
        vals = [r["towel_mean_abs"] for r in render if r["camera"] == cam and r["towel_mean_abs"] is not None
                and r["towel_pixels"] >= 50]
        if vals:
            summary.setdefault("towel_only", {})[cam] = {"mean_abs": float(np.mean(vals)), "worst": float(max(vals)),
                                                          "images": len(vals)}
            print(f"    {cam:19s} n={len(vals)} mean|diff|={np.mean(vals):.2f} (worst {max(vals):.2f})")
    timing = {k: float(np.median([b[k] for b in bench])) for k in ("physicsMs", "camerasMs", "mainViewMs")}
    print(f"\nper control step (median over scenes, this browser): physics {timing['physicsMs']:.1f} ms, "
          f"3 policy cameras {timing['camerasMs']:.1f} ms, free view {timing['mainViewMs']:.1f} ms "
          f"(budget at 30 Hz: {1000 / config.CONTROL_HZ:.1f} ms)")
    for r in realtime:
        print(f"real-time run ({args.physics}), scene {r['scene']}: {r['fps']:.1f} fps, simulated {r['simulated']:.2f} s in "
              f"{r['wall']:.2f} s ({r['realtime']:.2f} x real time)"
              + (f", physics {r['meanStepMs']:.1f} ms per step in the worker" if r.get("meanStepMs") else ""))
    if rec_rows:
        agree = sum(r["browser_success"] == r["python_success"]
                    and (not r["python_success"] or r["browser_step"] == r["python_step"]) for r in rec_rows)
        print(f"\nrecords (N = 16, no physics): the page's success check agrees with Python in {agree}/{len(rec_rows)}")
        for r in rec_rows:
            print("  ", r)
    size = page_bytes(f"{seeds[0]}_n{n}", records[0]["name"] if records else None)
    print("download: " + ", ".join(f"{k} {v['raw'] / 1e6:.2f} MB (gzip {v['gzip'] / 1e6:.2f})" for k, v in size.items()))
    if console:
        print("\nbrowser console:", *sorted(set(console))[:10], sep="\n  ")
    # As tools/replay_towel_node.mjs: clean episodes must agree; after a disturbance the coarse
    # towel can end differently in WASM and Python (reported above, not failed).
    rule_bugs = [r for r in replay if r["browser_success"] != r["python_success"] and r["disturb"] == "none"]
    passed = not rule_bugs and all(r["browser_success"] == r["python_success"] for r in rec_rows)
    if args.gpu:
        out = {"webgl": gl, "n": n, "replay": replay, "timing_ms": timing, "bench": bench, "realtime": realtime}
        path = REPO_ROOT / "docs" / "towel" / f"browser_speed_gpu_n{n}.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {path}")
        raise SystemExit(0 if passed else 1)
    docs = REPO_ROOT / "docs" / "towel"
    docs.mkdir(parents=True, exist_ok=True)
    if sheet_rows:
        sheet = np.concatenate(sheet_rows, axis=0)
        imageio.imwrite(docs / f"render_compare_n{n}.png", np.kron(sheet, np.ones((2, 2, 1), dtype=np.uint8)))
    (docs / f"render_compare_n{n}.json").write_text(json.dumps(
        {"webgl": gl, "n": n, "replay": replay, "render_summary": summary, "timing_ms": timing, "bench": bench,
         "realtime": realtime, "records": rec_rows, "download": size, "render": render}, indent=1))
    print(f"\nwrote {docs / f'render_compare_n{n}.png'} (per camera: MuJoCo / three.js / 4x difference) and .json")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

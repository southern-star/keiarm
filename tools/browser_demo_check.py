"""Exercise the demo page the way a visitor would, in a headless browser, in real time.

For each scene: press 開始 in model mode, wait for the cube to be placed, then drag the cube
off the plate with real mouse events and wait for the policy to pick it up and place it again.
Unlike tools/browser_eval.py nothing waits for inference here: the simulation runs at 30 Hz
and plans arrive late by however long inference takes on this machine.

Usage: (cd web && npm run build) && uv run python tools/browser_demo_check.py [--scenes 100000 ...]
       uv run python tools/browser_demo_check.py --url https://southern-star.github.io/keiarm/
"""

import argparse
import sys
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from browser_check import DIST, serve

from keiarm.paths import REPO_ROOT

# Screen position of a world point, and the cube/plate state, evaluated in the page.
JS_STATE = """() => {
  const { sim, view } = window.keiarm.app;
  const s = sim.state();
  return { cube: s.cubePos, plate: s.platePos, steps: sim.steps, ok: sim.succeeded,
           ms: window.keiarm.app.runner.lastMs, n: window.keiarm.app.runner.inferences };
}"""
JS_PROJECT = """([x, y, z]) => {
  const cam = window.keiarm.camera;
  const v = new window.keiarm.THREE.Vector3(x, y, z).project(cam);
  const r = document.getElementById('main').getBoundingClientRect();
  return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height];
}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+", help="default: every scene in the build")
    ap.add_argument("--timeout", type=float, default=25.0, help="seconds allowed for each placement")
    ap.add_argument("--shot", type=Path, default=REPO_ROOT / "docs" / "web_demo.png")
    ap.add_argument("--url", help="check a deployed site instead of the local build, e.g. the GitHub Pages URL")
    args = ap.parse_args()

    server, url = (None, args.url.rstrip("/") + "/") if args.url else serve(DIST)
    rows, errors = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True,
                                     args=["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1100, "height": 700})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.set_default_timeout(180_000)  # a deployed site downloads ~60 MB before it is ready
        page.goto(url)
        page.evaluate("() => window.keiarm.ready")
        info = page.evaluate("() => window.keiarm.policyReady()")
        seeds = args.scenes or page.evaluate("() => [...document.getElementById('scene').options].map(o => Number(o.value))")
        print(f"inference: {info['backend']}, {info['threads']} thread(s)")
        for i, seed in enumerate(seeds):
            page.select_option("#scene", str(seed))
            page.wait_for_function(f"() => window.keiarm.app.view && window.keiarm.app.sim.scene.scene_seed === {seed}")
            page.click("#play")
            row = {"scene": seed, "first": False, "after_drag": False}
            try:
                page.wait_for_function("() => window.keiarm.app.sim.succeeded", timeout=args.timeout * 1000)
                row["first"] = True
                st = page.evaluate(JS_STATE)
                row["first_steps"] = st["steps"]
                # Drag the cube from the plate to a point 9 cm away, towards the middle of the workspace.
                cx, cy, cz = st["cube"]
                tx, ty = (0.20, -0.02) if abs(cy - (-0.02)) > 0.09 or abs(cx - 0.20) > 0.09 else (0.16, 0.10)
                x0, y0 = page.evaluate(JS_PROJECT, [cx, cy, cz])
                x1, y1 = page.evaluate(JS_PROJECT, [tx, ty, cz])
                page.mouse.move(x0, y0)
                page.mouse.down()
                page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, steps=5)
                page.mouse.move(x1, y1, steps=5)
                page.wait_for_timeout(150)
                page.mouse.up()
                moved = page.evaluate(JS_STATE)
                row["dragged_cm"] = round(100 * ((moved["cube"][0] - cx) ** 2 + (moved["cube"][1] - cy) ** 2) ** 0.5, 1)
                page.wait_for_function("() => !window.keiarm.app.sim.succeeded", timeout=5000)
                if i == 0:
                    page.wait_for_timeout(1200)
                    page.screenshot(path=str(args.shot))
                page.wait_for_function("() => window.keiarm.app.sim.succeeded", timeout=args.timeout * 1000)
                end = page.evaluate(JS_STATE)
                row |= {"after_drag": True, "total_steps": end["steps"], "infer_ms": round(end["ms"])}
            except PlaywrightTimeout:
                pass
            rows.append(row)
            print(row, flush=True)
        browser.close()
    if server is not None:
        server.shutdown()
    n = len(rows)
    print(f"\nreal-time demo: first placement {sum(r['first'] for r in rows)}/{n}, "
          f"re-placement after a mouse drag {sum(r['after_drag'] for r in rows)}/{n}")
    if errors:
        print("console errors:", *sorted(set(errors))[:8], sep="\n  ")
    raise SystemExit(0 if all(r["after_drag"] for r in rows) else 1)


if __name__ == "__main__":
    main()

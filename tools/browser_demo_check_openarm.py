"""Exercise the OpenArm page the way a visitor would, in a headless browser, in real time.

For each scene: press 開始 in model mode and wait for the bar to be put on the plate; then drag
the bar off the plate back to the right-hand side of the table with real mouse events, and wait
for the policy to hand it over and place it again. Nothing waits for inference here: the
simulation runs at 30 Hz and plans arrive late by however long inference takes.

Usage: npm --prefix web run build && uv run python tools/browser_demo_check_openarm.py [--scenes 100000 ...]
       uv run python tools/browser_demo_check_openarm.py --url https://southern-star.github.io/keiarm/
"""

import argparse
import sys
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from browser_check import DIST, serve
from browser_demo_check import JS_PROJECT

from keiarm.paths import REPO_ROOT

JS_STATE = """() => {
  const { sim, runner } = window.keiarm.app;
  const s = sim.state();
  return { bar: s.barPos, plate: s.platePos, steps: sim.steps, ok: sim.succeeded, ms: runner.lastMs, n: runner.inferences };
}"""
DRAG_TO = (0.21, -0.23)  # bar centre after the drag: inside the right hand's spawn region


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+", help="default: every scene in the build")
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds allowed for each placement")
    ap.add_argument("--shot", type=Path, default=REPO_ROOT / "docs" / "openarm" / "web_demo.png")
    ap.add_argument("--url", help="check a deployed site instead of the local build")
    ap.add_argument("--gpu", action="store_true", help="WebGL on the machine's GPU (default: SwiftShader)")
    args = ap.parse_args()

    server, url = (None, args.url.rstrip("/") + "/") if args.url else serve(DIST)
    gl = ["--use-gl=angle", "--use-angle=gl-egl", "--enable-gpu"] if args.gpu else ["--enable-unsafe-swiftshader"]
    rows, errors = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True, args=["--ignore-gpu-blocklist", *gl])
        page = browser.new_page(viewport={"width": 1100, "height": 700})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.set_default_timeout(180_000)
        page.goto(f"{url}openarm.html")
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
                bx, by, bz = st["bar"]
                x0, y0 = page.evaluate(JS_PROJECT, [bx, by, bz])
                x1, y1 = page.evaluate(JS_PROJECT, [DRAG_TO[0], DRAG_TO[1], bz])
                page.mouse.move(x0, y0)
                page.mouse.down()
                page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, steps=5)
                page.mouse.move(x1, y1, steps=5)
                page.wait_for_timeout(150)
                page.mouse.up()
                moved = page.evaluate(JS_STATE)
                row["dragged_cm"] = round(100 * ((moved["bar"][0] - bx) ** 2 + (moved["bar"][1] - by) ** 2) ** 0.5, 1)
                page.wait_for_function("() => !window.keiarm.app.sim.succeeded", timeout=5000)
                if i == 0:
                    page.wait_for_timeout(4000)  # mid hand-over
                    args.shot.parent.mkdir(parents=True, exist_ok=True)
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
          f"re-placement after dragging the bar {sum(r['after_drag'] for r in rows)}/{n}")
    if errors:
        print("console errors:", *sorted(set(errors))[:8], sep="\n  ")
    raise SystemExit(0 if all(r["after_drag"] for r in rows) else 1)


if __name__ == "__main__":
    main()

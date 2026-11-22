"""Exercise the towel page the way a visitor would, in a headless browser, in real time (milestone 3, Phase 7).

For each scene: press 開始 with the learned policy and wait for the fold; then drag the folded
towel with real mouse events to the opposite corner of the spawn region (it lies flat again there,
both hands let go), and wait for the policy to fold it again. Nothing waits for inference: the
physics runs in its worker at (up to) real time and plans arrive late by however long inference
takes.

Usage: npm --prefix web run build && uv run python tools/browser_demo_check_towel.py --gpu [--scenes 100000 ...]
       uv run python tools/browser_demo_check_towel.py --gpu --url https://southern-star.github.io/keiarm/
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
  const { app } = window.keiarm;
  const x = app.sim.data.flexvert_xpos, a = 3 * app.sim.vadr, n = app.sim.nvert;
  const c = [0, 0, 0];
  for (let k = 0; k < n; k++) for (let i = 0; i < 3; i++) c[i] += x[a + 3 * k + i] / n;
  const r = app.remote ?? {};
  return { centre: c, steps: r.steps ?? 0, ok: Boolean(r.succeeded), xy: app.episode.towel_xy,
           region: [app.sim.c.towel_center, app.sim.c.towel_center_jitter] };
}"""
JS_SUCCEEDED = "() => Boolean(window.keiarm.app.remote?.succeeded)"
JS_NOT_SUCCEEDED = "() => !window.keiarm.app.remote?.succeeded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+", help="default: every scene in the build")
    ap.add_argument("--n", type=int, help="towel mesh N (default: the build's default)")
    ap.add_argument("--timeout", type=float, default=60.0, help="wall-clock seconds allowed for each fold")
    ap.add_argument("--shot", type=Path, default=REPO_ROOT / "docs" / "towel" / "web_demo.png")
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
        page.goto(f"{url}towel.html" + (f"?n={args.n}" if args.n else ""))
        page.evaluate("() => window.keiarm.ready")
        info = page.evaluate("() => window.keiarm.policyReady()")
        seeds = args.scenes or page.evaluate(
            "() => [...document.getElementById('scene').options].map(o => Number(o.textContent))")
        n = page.evaluate("() => window.keiarm.app.bundle.n")
        print(f"inference: {info['backend']}, {info['threads']} thread(s); towel N = {n}")
        for i, seed in enumerate(seeds):
            if page.evaluate("() => document.getElementById('scene').value") != f"{seed}_n{n}":
                page.select_option("#scene", f"{seed}_n{n}")
            page.wait_for_function(f"() => !window.keiarm.app.loading && window.keiarm.app.bundle?.scene.scene_seed === {seed}")
            page.click("#play")
            row = {"scene": seed, "first": False, "after_drag": False}
            try:
                page.wait_for_function(JS_SUCCEEDED, timeout=args.timeout * 1000)
                row["first"] = True
                st = page.evaluate(JS_STATE)
                row["first_steps"] = st["steps"]
                (cx, cy), (jx, jy) = st["region"]
                x0, y0 = st["xy"]
                goal = (cx - jx if x0 > cx else cx + jx, cy - jy if y0 > cy else cy + jy)  # opposite corner
                px, py, pz = st["centre"]
                sx0, sy0 = page.evaluate(JS_PROJECT, [px, py, pz])
                sx1, sy1 = page.evaluate(JS_PROJECT, [px + goal[0] - x0, py + goal[1] - y0, pz])
                page.mouse.move(sx0, sy0)
                page.mouse.down()
                page.mouse.move((sx0 + sx1) / 2, (sy0 + sy1) / 2, steps=5)
                page.mouse.move(sx1, sy1, steps=5)
                page.wait_for_timeout(300)
                page.mouse.up()
                page.wait_for_function(JS_NOT_SUCCEEDED, timeout=5000)
                moved = page.evaluate("() => window.keiarm.drag.centre")
                row["dragged_to"] = moved
                if i == 0:
                    page.wait_for_timeout(5000)  # mid fold
                    args.shot.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(args.shot))
                page.wait_for_function(JS_SUCCEEDED, timeout=args.timeout * 1000)
                end = page.evaluate(JS_STATE)
                row |= {"after_drag": True, "total_steps": end["steps"]}
            except PlaywrightTimeout:
                pass
            row["status"] = page.evaluate("() => document.getElementById('status').innerText").replace("\n", " / ")
            rows.append(row)
            print(row, flush=True)
        browser.close()
    if server is not None:
        server.shutdown()
    k = len(rows)
    print(f"\nreal-time demo: first fold {sum(r['first'] for r in rows)}/{k}, "
          f"fold again after dragging the towel {sum(r['after_drag'] for r in rows)}/{k}")
    if errors:
        print("console errors:", *sorted(set(errors))[:8], sep="\n  ")
    raise SystemExit(0 if all(r["after_drag"] for r in rows) else 1)


if __name__ == "__main__":
    main()

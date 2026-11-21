"""E5（ブラウザ）: ヘッドレス Chrome で spikes/cloth/web/index.html を開き、布をたたませて測る。

リポジトリの最上位を http で配信する（import map が /web/node_modules の固定版を指す）。
実行: uv run python spikes/cloth/web/browser_check.py [--gpu] [--scenes A_n16_s0 ...] [--modes fast realtime]
出力: spikes/cloth/out/e5_browser.json と e5_browser_<scene>.png
"""

import argparse
import functools
import http.server
import json
import threading
from pathlib import Path
from typing import ClassVar

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "spikes" / "cloth" / "out"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map: ClassVar[dict] = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm"}

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="実 GPU（ANGLE/EGL）。既定は SwiftShader")
    ap.add_argument("--scenes", nargs="*", default=["A_n10_s0", "A_n16_s0", "A_n24_s0", "B_n16_s0"])
    ap.add_argument("--modes", nargs="*", default=["fast", "realtime"])
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    gl = ["--use-gl=angle", "--use-angle=gl-egl", "--enable-gpu"] if args.gpu else []
    rows = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True, args=["--ignore-gpu-blocklist", *gl])
        for scene in args.scenes:
            for mode in args.modes:
                page = browser.new_page(viewport={"width": 960, "height": 640})
                logs = []
                page.on("console", lambda msg, logs=logs: logs.append(msg.text))
                page.on("pageerror", lambda e, logs=logs: logs.append(f"pageerror: {e}"))
                page.goto(f"http://127.0.0.1:{args.port}/spikes/cloth/web/index.html?scene={scene}&mode={mode}")
                try:
                    page.wait_for_function("window.__result !== undefined", timeout=600_000)
                    r = page.evaluate("window.__result")
                except Exception as e:  # noqa: BLE001
                    r = {"scene": scene, "mode": mode, "status": f"timeout/error: {e}", "logs": logs[-5:]}
                if mode == "realtime":
                    page.screenshot(path=str(OUT / f"e5_browser_{scene}{'_gpu' if args.gpu else ''}.png"))
                r["gpu_flag"] = args.gpu
                rows.append(r)
                print(json.dumps(r), flush=True)
                page.close()
        browser.close()
    srv.shutdown()
    name = "e5_browser_gpu.json" if args.gpu else "e5_browser.json"
    (OUT / name).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

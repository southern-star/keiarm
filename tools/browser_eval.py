"""Closed-loop evaluation of the exported model inside a headless browser.

Physics (MuJoCo WASM), rendering (three.js) and inference (onnxruntime-web) all run in the
page, exactly as in the public demo, except that the simulation waits for each inference so
the result does not depend on the machine's speed. Results are compared, episode by episode,
with the Python harness run of the same model (paired sign test).

Usage:
  uv run python scripts/export_onnx.py --run r1 && (cd web && npm run build)
  uv run python tools/browser_eval.py --python-run keigrasp-r1_eval3
  uv run python tools/browser_eval.py --disturb --python-run keigrasp-r1_eval_disturb
"""

import argparse
import functools
import http.server
import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from keiarm import config
from keiarm.eval import sign_test
from keiarm.export import export_web
from keiarm.paths import REPO_ROOT

DIST = REPO_ROOT / "web" / "dist"


class _Overlay(http.server.SimpleHTTPRequestHandler):
    """Serve files from the first directory that has them (eval scenes on top of the built site)."""

    roots: tuple[Path, ...] = ()

    def translate_path(self, path):
        rel = Path(super().translate_path(path)).relative_to(Path(self.directory).resolve())
        for root in self.roots:
            if (root / rel).is_file():
                return str(root / rel)
        return str(self.roots[-1] / rel)

    def log_message(self, *args):
        pass


def serve(roots: tuple[Path, ...]):
    handler = functools.partial(type("H", (_Overlay,), {"roots": roots}), directory=str(roots[-1]))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of eval scenes")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--disturb", action="store_true", help="apply each episode's planned cube teleport")
    ap.add_argument("--backend", default="auto", choices=["auto", "wasm"])
    ap.add_argument("--python-run", help="runs/eval/<name> of the same model in the Python harness, to compare")
    ap.add_argument("--name", help="summary name (default: derived)")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    if not (DIST / "models" / "keigrasp.onnx").exists():
        raise SystemExit("web/dist has no model: run scripts/export_onnx.py and `npm run build` in web/")

    meta = json.loads((DIST / "models" / "keigrasp.json").read_text())
    seeds = list(config.EVAL_SCENE_SEEDS)[: args.n]
    name = args.name or f"keigrasp-{meta['run']}_browser{'_disturb' if args.disturb else ''}"
    results, console = [], []
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        export_web(seeds, list(range(args.episodes)), out=Path(tmp))
        server, url = serve((Path(tmp), DIST))
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome", headless=not args.headed,
                                         args=["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"])
            page = browser.new_page(viewport={"width": 900, "height": 600})
            page.set_default_timeout(600_000)
            page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
            page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
            page.goto(f"{url}?scene={seeds[0]}&backend={args.backend}")
            page.evaluate("() => window.keiarm.ready")
            info = page.evaluate("() => window.keiarm.policyReady()")
            gl = page.evaluate("""() => { const g = document.createElement('canvas').getContext('webgl2');
                const e = g.getExtension('WEBGL_debug_renderer_info');
                return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; }""")
            print(f"inference backend: {info['backend']} ({info['threads']} thread), warm-up {info['warmupMs']:.0f} ms; "
                  f"WebGL: {gl}", flush=True)
            for i, seed in enumerate(seeds):
                page.evaluate("(s) => window.keiarm.loadScene(s)", seed)
                for k in range(args.episodes):
                    r = page.evaluate("([k, d]) => window.keiarm.runModelEpisode(k, {disturb: d})", [k, args.disturb])
                    results.append({"scene_seed": seed, "episode_seed": k, **r})
                if (i + 1) % 10 == 0:
                    ok = sum(r["success"] for r in results)
                    print(f"  {i + 1}/{len(seeds)} scenes: {ok}/{len(results)} ({time.perf_counter() - t0:.0f}s)", flush=True)
            browser.close()
        server.shutdown()

    n = len(results)
    ok = [r for r in results if r["success"]]
    steps = [r["steps"] for r in ok]
    infer = float(np.mean([r["meanInferMs"] for r in results]))
    summary = {
        "name": name, "model": meta["run"], "episodes": n, "success": len(ok), "success_rate": len(ok) / n,
        "disturb": args.disturb, "backend": info["backend"], "webgl": gl, "mean_infer_ms": infer,
        "steps_to_success_median": float(np.median(steps)) if steps else None,
        "failed": [(r["scene_seed"], r["episode_seed"]) for r in results if not r["success"]],
    }
    lines = [
        f"# ブラウザ内評価: {name}", "",
        f"- モデル: `{meta['run']}`（ONNX、{meta['bytes'] / 1e6:.1f} MB）/ 評価 {len(seeds)} シーン × {args.episodes} エピソード",
        f"- 外乱: {'各エピソードの計画どおりに立方体を 1 回移動' if args.disturb else 'なし'}",
        f"- 物理 MuJoCo WASM、描画 three.js（{gl}）、推論 onnxruntime-web（{info['backend']}、平均 {infer:.0f} ms）",
        "- シミュレーションは推論を待つ（同期実行）。所要 " + f"{time.perf_counter() - t0:.0f} 秒", "",
        "| 指標 | 値 |", "|---|---|",
        f"| 成功率 | **{len(ok)}/{n}（{100 * len(ok) / n:.1f}%）** |",
    ]
    if steps:
        lines.append(f"| 所要ステップ数（成功時、中央値） | {np.median(steps):.0f} |")
    if args.python_run:
        path = REPO_ROOT / "runs" / "eval" / args.python_run / "episodes.json"
        py = {(r["scene_seed"], r["episode_seed"]): r["success"] for r in json.loads(path.read_text())}
        web = {(r["scene_seed"], r["episode_seed"]): r["success"] for r in results}
        t = sign_test(py, web)
        summary["vs_python"] = {"run": args.python_run, **t}
        lines += ["", f"## Python 描画（`{args.python_run}`）との対比較", "",
                  f"共通 {t['paired_episodes']} エピソード: Python {t['both'] + t['a_only']} 成功、ブラウザ {t['both'] + t['b_only']} 成功。",
                  f"両方成功 {t['both']}、両方失敗 {t['neither']}、Python のみ {t['a_only']}、ブラウザのみ {t['b_only']}。",
                  f"対応のある符号検定（両側、正確）: p = {t['p_value']:.3g}"]
    if summary["failed"]:
        lines += ["", "失敗した (scene_seed, episode_seed): " + ", ".join(f"({s}, {e})" for s, e in summary["failed"][:40])]
    if console:
        lines += ["", "ブラウザのコンソールエラー:", *[f"- {c}" for c in sorted(set(console))[:8]]]
    out = REPO_ROOT / "runs" / "eval_summaries"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

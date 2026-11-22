"""Closed-loop evaluation of the exported towel model inside headless browsers (milestone 3, check V5).

Physics (MuJoCo WASM, the coarse towel N_web), rendering (three.js, decimated meshes) and
inference (onnxruntime-web) all run in the page, as in the demo, except that the simulation waits
for each inference (window.keiarm.runModelEpisode), so the result does not depend on the
machine's speed. Disturbances are armed from the same plans as the Python suite and fired by the
same stage rule. Results are compared episode by episode with the Python harness run of the same
model on the same mesh (paired sign test). The scenes are split over --pages browsers.

Usage:
  uv run python scripts/export_onnx.py --robot openarm --run towel_r3 --out web/public/models/towel.onnx \\
      --check-data towel_bc1 && npm --prefix web run build
  uv run python tools/browser_eval_towel.py --gpu --python-run towel_keigrasp-towel_r3_eval_n11 [--disturb suite]
"""

import argparse
import json
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from browser_eval import DIST, serve

from keiarm import config
from keiarm.eval import sign_test
from keiarm.paths import REPO_ROOT
from keiarm.towel.eval import DISTURBANCES
from keiarm.towel.export import export_web

GPU_ARGS = ["--use-gl=angle", "--use-angle=gl-egl", "--enable-gpu"]
SWIFTSHADER_ARGS = ["--enable-unsafe-swiftshader"]


def _run_pages(job) -> dict:
    """One browser on its own server: evaluate `seeds` for every kind."""
    root, seeds, kinds, n, gpu, backend = job
    server, url = serve((Path(root), DIST))
    out = {"results": {k: [] for k in kinds}, "console": [], "info": None, "gl": None}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome", headless=True,
                                         args=["--ignore-gpu-blocklist", *(GPU_ARGS if gpu else SWIFTSHADER_ARGS)])
            page = browser.new_page(viewport={"width": 900, "height": 600})
            page.set_default_timeout(1_800_000)
            page.on("pageerror", lambda e: out["console"].append(f"pageerror: {e}"))
            page.on("console", lambda m: out["console"].append(m.text) if m.type == "error" else None)
            page.goto(f"{url}towel.html?scene={seeds[0]}&n={n}&backend={backend}")
            page.evaluate("() => window.keiarm.ready")
            out["info"] = page.evaluate("() => window.keiarm.policyReady()")
            out["gl"] = page.evaluate("""() => { const g = document.createElement('canvas').getContext('webgl2');
                const e = g.getExtension('WEBGL_debug_renderer_info');
                return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; }""")
            for seed in seeds:
                page.evaluate("(s) => window.keiarm.loadScene(s)", f"{seed}_n{n}")
                eps = page.evaluate("() => window.keiarm.app.bundle.episodes.map(e => [e.episode_seed, !!e.disturbance])")
                for k, (ep, recorded_disturbance) in enumerate(eps):
                    if recorded_disturbance:
                        continue  # the extra replay episode, not an evaluation episode
                    for kind in kinds:
                        r = page.evaluate("([k, d]) => window.keiarm.runModelEpisode(k, {disturb: d})", [k, kind])
                        out["results"][kind].append({"scene_seed": seed, "episode_seed": ep, **r})
            browser.close()
    finally:
        server.shutdown()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of eval scenes")
    ap.add_argument("--mesh", type=int, default=11, help="towel mesh N in the browser")
    ap.add_argument("--disturb", choices=[*DISTURBANCES, "suite"], default="none")
    ap.add_argument("--backend", default="auto", choices=["auto", "wasm"])
    ap.add_argument("--pages", type=int, default=3, help="browsers in parallel (each: physics + inference threads)")
    ap.add_argument("--gpu", action="store_true", help="use the machine's GPU for WebGL (default: SwiftShader)")
    ap.add_argument("--python-run", help="runs/eval/<name> of the same model and mesh in the Python harness")
    args = ap.parse_args()
    if not (DIST / "models" / "towel.onnx").exists():
        raise SystemExit("web/dist has no towel model: run scripts/export_onnx.py (see Usage) and `npm run build`")

    meta = json.loads((DIST / "models" / "towel.json").read_text())
    kinds = DISTURBANCES if args.disturb == "suite" else (args.disturb,)
    seeds = list(config.EVAL_SCENE_SEEDS)[: args.n]
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        with Pool(6) as pool:
            export_web(seeds, [args.mesh], [0], out=Path(tmp), map_fn=pool.map)
        print(f"exported {len(seeds)} scenes (N = {args.mesh}) in {time.perf_counter() - t0:.0f}s", flush=True)
        chunks = [seeds[i:: args.pages] for i in range(args.pages)]
        jobs = [(tmp, c, kinds, args.mesh, args.gpu, args.backend) for c in chunks if c]
        with Pool(len(jobs)) as pool:
            parts = pool.map(_run_pages, jobs)
    results = {k: sorted((r for p in parts for r in p["results"][k]), key=lambda r: r["scene_seed"]) for k in kinds}
    console = [c for p in parts for c in p["console"]]
    info, gl = parts[0]["info"], parts[0]["gl"]

    out = REPO_ROOT / "runs" / "eval_summaries"
    name = f"towel_keigrasp-{meta['run']}_browser_n{args.mesh}" + ("" if args.disturb == "none" else f"_{args.disturb}")
    infer = float(np.mean([r["meanInferMs"] for v in results.values() for r in v]))
    lines = [
        f"# ブラウザ内評価: {name}", "",
        f"- モデル: `{meta['run']}`（ONNX、{meta['bytes'] / 1e6:.1f} MB）/ 評価 {len(seeds)} シーン × 1 エピソード、布 N = {args.mesh}",
        f"- 物理 MuJoCo WASM、描画 three.js（間引いたメッシュ、{gl}）、推論 onnxruntime-web（{info['backend']}、平均 {infer:.0f} ms）",
        f"- シミュレーションは推論を待つ（同期実行）。ブラウザ {len(jobs)} 個で分担、所要 {time.perf_counter() - t0:.0f} 秒", "",
        "| 外乱 | ブラウザ | Python | 両方成功 | Python のみ | ブラウザのみ | 両方失敗 | 符号検定 p | 所要ステップ中央値（ブラウザ） |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    summary = {"name": name, "model": meta["run"], "mesh": args.mesh, "backend": info["backend"], "webgl": gl,
               "mean_infer_ms": infer, "kinds": {}}
    for kind, rows in results.items():
        ok = [r for r in rows if r["success"]]
        med = float(np.median([r["steps"] for r in ok])) if ok else None
        s = {"episodes": len(rows), "success": len(ok), "steps_to_success_median": med,
             "failed": [(r["scene_seed"], r["episode_seed"], r["stage"]) for r in rows if not r["success"]]}
        cells = ["—"] * 6
        if args.python_run:
            run = args.python_run + ("" if kind == "none" else f"_{kind}")
            path = REPO_ROOT / "runs" / "eval" / run / "episodes.json"
            py = {(r["scene_seed"], r["episode_seed"]): r["success"] for r in json.loads(path.read_text())}
            web = {(r["scene_seed"], r["episode_seed"]): r["success"] for r in rows}
            t = sign_test(py, web)
            s["vs_python"] = {"run": run, **t}
            cells = [f"{t['both'] + t['a_only']}/{t['paired_episodes']}", str(t["both"]), str(t["a_only"]),
                     str(t["b_only"]), str(t["neither"]), f"{t['p_value']:.3g}"]
        summary["kinds"][kind] = s
        lines.append(f"| {kind} | **{len(ok)}/{len(rows)}** | " + " | ".join(cells) + f" | {med if med is None else f'{med:.0f}'} |")
    fails = [f"{k}: " + ", ".join(f"({a}, {b}, {c})" for a, b, c in v["failed"][:25])
             for k, v in summary["kinds"].items() if v["failed"]]
    if fails:
        lines += ["", "失敗した (scene_seed, episode_seed, 最遠段階):", *[f"- {f}" for f in fails]]
    if console:
        lines += ["", "ブラウザのコンソールエラー:", *[f"- {c}" for c in sorted(set(console))[:8]]]
    (out / f"{name}.json").write_text(json.dumps({**summary, "episodes": results}, indent=1, ensure_ascii=False))
    (out / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

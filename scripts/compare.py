"""Paired comparison of two evaluation runs on the seeds they share (exact sign test).

Usage: uv run python scripts/compare.py expert_eval expert-noisy-0.02_eval
"""

import argparse
import json

from keiarm.eval import sign_test
from keiarm.paths import REPO_ROOT


def load(name: str) -> dict[tuple[int, int], bool]:
    path = REPO_ROOT / "runs" / "eval" / name / "episodes.json"
    return {(r["scene_seed"], r["episode_seed"]): r["success"] for r in json.loads(path.read_text())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    t = sign_test(a, b)
    n = t["paired_episodes"]
    print(f"A = {args.a}: {t['both'] + t['a_only']}/{n} 成功")
    print(f"B = {args.b}: {t['both'] + t['b_only']}/{n} 成功")
    print(f"両方成功 {t['both']} / 両方失敗 {t['neither']} / A のみ成功 {t['a_only']} / B のみ成功 {t['b_only']}")
    verdict = "差あり" if t["p_value"] < 0.05 else "差があるとは言えない"
    print(f"対応のある符号検定（両側、正確）: p = {t['p_value']:.4g} → {verdict}（有意水準 5%）")


if __name__ == "__main__":
    main()

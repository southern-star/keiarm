"""Evaluate a policy on a suite and write a report, per-episode results and failure logs.

Usage:
  uv run python scripts/eval.py --policy expert
  uv run python scripts/eval.py --policy expert-noisy:0.02 --name noisy02 --videos 5

Outputs:
  runs/eval/<name>/episodes.json        every episode (seeds, outcome, infractions)
  runs/eval_summaries/<name>.{json,md}  the summary (tracked in git)
  runs/failures/<name>/                 per failure: seeds + executed actions (.npz), video (.mp4)
"""

import argparse
import json
import os
import shutil
import time
from dataclasses import astuple
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.eval import TABLE_HIT_FORCE, EpisodeRecord, run_episode, summarize
from keiarm.paths import REPO_ROOT
from keiarm.policies import make_policy
from keiarm.task import STAGES

FAILURE_JA = {
    "approach_failed": "接近失敗", "grasp_failed": "掴み損ね", "dropped_in_transit": "搬送中の落下",
    "transit_timeout": "搬送中の時間切れ", "place_failed": "置き損ね", "out_of_workspace": "作業領域外",
}
STAGE_JA = {"reach": "到達", "grasp": "把持", "lift": "持ち上げ", "over_plate": "皿上", "success": "完了"}


def DISTURB_JA(mode) -> str:
    if mode is None:
        return "なし"
    when = "シードで決まるステップ（10〜59）" if mode == "random" else f"{mode} ステップ目"
    return f"{when}に立方体を机上の別の場所へ移動（1 回）"


def _one(job) -> EpisodeRecord:
    spec, scene_seed, episode_seed, disturb = job
    policy = make_policy(spec)
    env = KeiArmEnv(scene_seed, render=policy.needs_images)
    rec = run_episode(env, policy, episode_seed, disturb=disturb)
    env.close()
    return rec


def write_failure(rec: EpisodeRecord, out: Path, video: bool, scale: int = 3) -> None:
    stem = out / f"s{rec.scene_seed}_e{rec.episode_seed}_{rec.failure}"
    np.savez_compressed(f"{stem}.npz", scene_seed=rec.scene_seed, episode_seed=rec.episode_seed,
                        actions=rec.actions, failure=rec.failure,
                        expert_could_finish=bool(rec.expert_could_finish),
                        teleport=np.array(astuple(rec.teleport) if rec.teleport else [], dtype=float))
    if not video:
        return
    env = KeiArmEnv(rec.scene_seed, width=config.IMAGE_WIDTH * scale, height=config.IMAGE_HEIGHT * scale)
    env.reset(rec.episode_seed)
    with imageio.get_writer(f"{stem}.mp4", fps=config.CONTROL_HZ, macro_block_size=1) as writer:
        for t, action in enumerate(rec.actions):
            if rec.teleport is not None and t == rec.teleport.step:
                rec.teleport.apply(env)
            env.step(action)
            writer.append_data(np.concatenate([env.render(c) for c in config.CAMERAS], axis=1))
    env.close()


def report_md(name: str, spec: str, split: str, summary: dict, records: list[EpisodeRecord], seconds: float) -> str:
    n = summary["episodes"]
    lines = [
        f"# 評価レポート: {name}", "",
        f"- 方策: `{spec}` / スイート: {split}（{n} エピソード）/ 制限 {config.EPISODE_SECONDS} 秒",
        f"- 外乱: {DISTURB_JA(summary.get('disturb'))}",
        f"- MuJoCo {mujoco.__version__}、所要 {seconds:.0f} 秒", "",
        "## 指標", "",
        "| 指標 | 値 |", "|---|---|",
        f"| 成功率 | **{summary['success']}/{n}（{100 * summary['success_rate']:.1f}%）** |",
    ]
    if summary["steps_to_success"]:
        s = summary["steps_to_success"]
        lines.append(f"| 所要ステップ数（成功時） | 中央値 {s['median']:.0f}、90% 点 {s['p90']:.0f}、最大 {s['max']} "
                     f"（{s['max'] / config.CONTROL_HZ:.1f} 秒） |")
    inf = summary["infractions_per_episode"]
    lines += [
        f"| 机への強い衝突（{TABLE_HIT_FORCE:.0f} N 超） | {inf['table_hits']:.2f} 回/エピソード |",
        f"| 立方体の落下 | {inf['drops']:.2f} 回/エピソード |",
        f"| 作業領域外への逸脱 | {inf['workspace_exits']:.2f} 回/エピソード |", "",
        "## 段階進捗（そこまで到達したエピソード数）", "",
        "| " + " | ".join(STAGE_JA[s] for s in STAGES[1:]) + " |", "|" + "---|" * 5,
        "| " + " | ".join(str(summary["stage_reached"][s]) for s in STAGES[1:]) + " |", "",
    ]
    fails = [r for r in records if not r.success]
    if fails:
        lines += ["## 失敗の内訳", "", "| 分類 | 件数 |", "|---|---|"]
        lines += [f"| {FAILURE_JA[c]} | {k} |" for c, k in summary["failures"].items()]
        lines += ["", (f"失敗 {len(fails)} 件のうち、最終状態からエキスパートなら完了できたもの: "
                  f"{summary['timeouts_expert_could_finish']} 件（方策の停止・迷い）、"
                  f"できなかったもの: {summary['timeouts_stuck']} 件（状況が詰んでいる）。"), "",
                  "| scene_seed | episode_seed | 分類 | 最遠段階 | エキスパートなら完了 | 机衝突 | 落下 |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {r.scene_seed} | {r.episode_seed} | {FAILURE_JA[r.failure]} | {STAGE_JA.get(r.stage, '未到達')} | "
                  f"{'はい' if r.expert_could_finish else 'いいえ'} | {r.table_hits} | {r.drops} |" for r in fails[:40]]
        if len(fails) > 40:
            lines.append(f"\n（残り {len(fails) - 40} 件は episodes.json）")
    else:
        lines += ["## 失敗の内訳", "", "失敗なし。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert", help="expert | expert-noisy:<sigma> | hold | grasp-only")
    ap.add_argument("--name", help="run name (default: derived from --policy and --split)")
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--n", type=int, default=100, help="number of scenes")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per scene")
    ap.add_argument("--disturb", action="store_true",
                    help="move the cube to a new spot once per episode, at a seed-determined step in 10-59")
    ap.add_argument("--teleport-at", type=int, help="like --disturb but at this fixed step")
    ap.add_argument("--videos", type=int, default=10, help="max failure videos to render")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    disturb = args.teleport_at if args.teleport_at is not None else ("random" if args.disturb else None)
    suffix = "" if disturb is None else "_disturb" if disturb == "random" else f"_teleport{disturb}"
    name = args.name or f"{args.policy.replace(':', '-')}_{args.split}{suffix}"
    seeds = list(config.EVAL_SCENE_SEEDS if args.split == "eval" else config.TRAIN_SCENE_SEEDS)[: args.n]
    jobs = [(args.policy, s, e, disturb) for s in seeds for e in range(args.episodes)]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        records = pool.map(_one, jobs)
    seconds = time.perf_counter() - t0
    summary = summarize(records)
    summary |= {"disturb": disturb, "name": name, "policy": args.policy, "split": args.split, "mujoco": mujoco.__version__}
    report = report_md(name, args.policy, args.split, summary, records, seconds)

    run_dir = REPO_ROOT / "runs" / "eval" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episodes.json").write_text(json.dumps([r.to_json() for r in records], indent=1))
    sum_dir = REPO_ROOT / "runs" / "eval_summaries"
    sum_dir.mkdir(parents=True, exist_ok=True)
    (sum_dir / f"{name}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (sum_dir / f"{name}.md").write_text(report)

    fail_dir = REPO_ROOT / "runs" / "failures" / name
    shutil.rmtree(fail_dir, ignore_errors=True)
    fails = [r for r in records if not r.success]
    if fails:
        fail_dir.mkdir(parents=True)
        for i, r in enumerate(fails):
            write_failure(r, fail_dir, video=i < args.videos)

    print(report)
    print(f"summary: {sum_dir / (name + '.md')}\nepisodes: {run_dir / 'episodes.json'}")
    if fails:
        print(f"failures: {fail_dir} ({len(fails)} logs, {min(len(fails), args.videos)} videos)")


if __name__ == "__main__":
    main()

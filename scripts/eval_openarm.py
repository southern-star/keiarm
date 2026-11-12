"""Evaluate a policy on the OpenArm hand-over (milestone 2) and write a report.

Usage:
  uv run python scripts/eval_openarm.py --policy expert
  uv run python scripts/eval_openarm.py --policy expert --disturb suite
  uv run python scripts/eval_openarm.py --policy expert-noisy:0.0065 --disturb knock

--disturb: none | move | knock | drop | plate | suite (all five, one run each, plus a
combined report). Every run is named openarm_<policy>_<split>[_<disturbance>] and writes
  runs/eval/<name>/episodes.json        every episode (works with scripts/compare.py)
  runs/eval_summaries/<name>.{json,md}  the summary (tracked in git)
  runs/failures/<name>/                 per failure: seeds, actions, disturbance (.npz), video (.mp4)
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
from keiarm.openarm.env import CAMERAS, EPISODE_SECONDS, OpenArmEnv
from keiarm.openarm.eval import (
    DISTURBANCES,
    TABLE_HIT_FORCE,
    EpisodeRecord,
    replay,
    run_episode,
    summarize,
)
from keiarm.openarm.policies import make_policy
from keiarm.openarm.task import STAGES
from keiarm.paths import REPO_ROOT

STAGE_JA = {"reach": "右手が接近", "grasp_right": "右手で把持", "lift": "持ち上げ", "handover": "両手で保持",
            "left_carry": "左手だけで保持", "over_plate": "皿の上", "success": "完了"}
FAILURE_JA = {"grasp_failed": "右手の掴み損ね", "lift_failed": "持ち上げ失敗", "handover_failed": "受け渡し失敗",
              "release_failed": "右手が離せない", "carry_failed": "左手の搬送失敗", "place_failed": "置き損ね",
              "dropped": "落下", "bar_off_table": "棒が机の外へ"}
DISTURB_JA = {
    "none": "なし",
    "move": "右手が閉じる前（10〜39 ステップ目）に、棒を右側の別の場所へ移す",
    "knock": "右手が持ち上げてから 5〜29 ステップ後に、棒を右側の机の上へ落とす（掴み直し）",
    "drop": "左手だけで持ってから 0〜19 ステップ後に、棒を右側の机の上へ落とす（受け渡しからやり直し）",
    "plate": "左手だけで持ってから 0〜19 ステップ後に、皿を別の場所へ移す",
}


def _one(job) -> EpisodeRecord:
    spec, scene_seed, episode_seed, disturb = job
    policy = make_policy(spec)
    env = OpenArmEnv(scene_seed, render=policy.needs_images)
    rec = run_episode(env, policy, episode_seed, disturb=disturb)
    env.close()
    return rec


def write_failure(rec: EpisodeRecord, out: Path, video: bool, scale: int = 2) -> None:
    stem = out / f"s{rec.scene_seed}_e{rec.episode_seed}_{rec.failure}"
    d = rec.disturbance
    np.savez_compressed(f"{stem}.npz", scene_seed=rec.scene_seed, episode_seed=rec.episode_seed,
                        actions=rec.actions, failure=rec.failure, disturb=rec.disturb,
                        expert_could_finish=bool(rec.expert_could_finish),
                        disturbance=np.array(astuple(d)[1:] if d and d.step is not None else [], dtype=float))
    if not video:
        return
    env = OpenArmEnv(rec.scene_seed, width=config.IMAGE_WIDTH * scale, height=config.IMAGE_HEIGHT * scale)
    with imageio.get_writer(f"{stem}.mp4", fps=config.CONTROL_HZ, macro_block_size=1) as writer:
        replay(env, rec, on_step=lambda e: writer.append_data(np.concatenate([e.render(c) for c in CAMERAS], axis=1)))
    env.close()


def _fmt_step(v) -> str:
    return "—" if v is None else f"{v:.0f}（{v / config.CONTROL_HZ:.1f} 秒）"


def report_md(name: str, spec: str, split: str, disturb: str, summary: dict, records: list[EpisodeRecord],
              seconds: float) -> str:
    n = summary["episodes"]
    inf = summary["infractions_per_episode"]
    lines = [
        f"# 評価レポート: {name}", "",
        f"- 方策: `{spec}` / スイート: {split}（{n} エピソード）/ 制限 {EPISODE_SECONDS} 秒",
        f"- 外乱: {DISTURB_JA[disturb]}"
        + (f"（実際に起きたのは {summary['disturbance_applied']} 本）" if disturb != "none" else ""),
        f"- MuJoCo {mujoco.__version__}、所要 {seconds:.0f} 秒", "",
        "## 指標", "",
        "| 指標 | 値 |", "|---|---|",
        f"| 成功率 | **{summary['success']}/{n}（{100 * summary['success_rate']:.1f}%）** |",
    ]
    if summary["steps_to_success"]:
        s = summary["steps_to_success"]
        lines.append(f"| 所要ステップ数（成功時） | 中央値 {s['median']:.0f}、90% 点 {s['p90']:.0f}、最大 {s['max']} "
                     f"（{s['max'] / config.CONTROL_HZ:.1f} 秒） |")
    lines += [
        f"| 腕どうしの接触があったエピソード | {inf['arms_touch_episodes']:.2f}（計 {summary['arms_touch_steps']} ステップ） |",
        (f"| 机への強い衝突（{TABLE_HIT_FORCE:.0f} N 超） | {inf['table_hits']:.2f} 回/エピソード"
        f"（最大 {summary['max_table_force']:.1f} N） |"),
        f"| 棒の落下 | {inf['drops']:.2f} 回/エピソード |",
        f"| 棒が机の外へ | {inf['bar_off_table']:.2f} 本/エピソード |", "",
        "## 段階ごとの集計", "",
        "| 段階 | 到達したエピソード | 初めて到達したステップ（中央値） |", "|---|---|---|",
    ]
    lines += [f"| {STAGE_JA[s]} | {summary['stage_reached'][s]} | {_fmt_step(summary['stage_first_step_median'][s])} |"
              for s in STAGES[1:]]
    lines.append("")
    fails = [r for r in records if not r.success]
    if fails:
        lines += ["## 失敗の内訳", "", "| 分類 | 件数 |", "|---|---|"]
        lines += [f"| {FAILURE_JA[c]} | {k} |" for c, k in summary["failures"].items()]
        lines += ["", (f"失敗 {len(fails)} 件のうち、最終状態からエキスパートなら完了できたもの: "
                       f"{summary['failed_expert_could_finish']} 件（方策の停止・迷い）、"
                       f"できなかったもの: {summary['failed_stuck']} 件（状況が詰んでいる）。"), "",
                  "| scene_seed | episode_seed | 分類 | 最遠段階 | 外乱のステップ | エキスパートなら完了 | 腕の接触 | 落下 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in fails[:40]:
            d = r.disturbance.step if r.disturbance and r.disturbance.step is not None else "—"
            could = {True: "はい", False: "いいえ", None: "—"}[r.expert_could_finish]
            lines.append(f"| {r.scene_seed} | {r.episode_seed} | {FAILURE_JA[r.failure]} | "
                         f"{STAGE_JA.get(r.stage, '未到達')} | {d} | {could} | {r.arms_touch_steps} | {r.drops} |")
        if len(fails) > 40:
            lines.append(f"\n（残り {len(fails) - 40} 件は episodes.json）")
    else:
        lines += ["## 失敗の内訳", "", "失敗なし。"]
    return "\n".join(lines) + "\n"


def suite_md(name: str, spec: str, split: str, runs: dict[str, dict]) -> str:
    lines = [f"# 外乱スイート: {name}", "", f"- 方策: `{spec}` / スイート: {split}", "",
             "| 外乱 | 成功率 | 外乱が起きた本数 | 腕の接触 | 落下/本 | 机の外/本 | 所要ステップ中央値 | レポート |",
             "|---|---|---|---|---|---|---|---|"]
    for kind, (run_name, s) in runs.items():
        inf = s["infractions_per_episode"]
        med = s["steps_to_success"]["median"] if s["steps_to_success"] else None
        lines.append(f"| {kind} | **{s['success']}/{s['episodes']}** | "
                     f"{s['disturbance_applied'] if kind != 'none' else '—'} | {inf['arms_touch_episodes']:.2f} | "
                     f"{inf['drops']:.2f} | {inf['bar_off_table']:.2f} | {_fmt_step(med)} | [{run_name}]({run_name}.md) |")
    lines += ["", "外乱の内容:", ""] + [f"- **{k}**: {DISTURB_JA[k]}" for k in runs]
    return "\n".join(lines) + "\n"


def run_one(args, disturb: str, pool: Pool) -> tuple[str, dict]:
    suffix = "" if disturb == "none" else f"_{disturb}"
    name = (args.name or f"openarm_{args.policy.replace(':', '-')}_{args.split}") + suffix
    seeds = list(config.EVAL_SCENE_SEEDS if args.split == "eval" else config.TRAIN_SCENE_SEEDS)[: args.n]
    jobs = [(args.policy, s, e, disturb) for s in seeds for e in range(args.episodes)]
    t0 = time.perf_counter()
    records = pool.map(_one, jobs)
    seconds = time.perf_counter() - t0
    summary = summarize(records)
    summary |= {"disturb": disturb, "name": name, "policy": args.policy, "split": args.split,
                "mujoco": mujoco.__version__}
    report = report_md(name, args.policy, args.split, disturb, summary, records, seconds)

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
    print(f"summary: {sum_dir / (name + '.md')}")
    if fails:
        print(f"failures: {fail_dir} ({len(fails)} logs, {min(len(fails), args.videos)} videos)")
    return name, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert", help="expert | expert-noisy:<sigma> | hold | freeze[:<phase>]")
    ap.add_argument("--name", help="run name prefix (default: openarm_<policy>_<split>)")
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--n", type=int, default=100, help="number of scenes")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per scene")
    ap.add_argument("--disturb", choices=[*DISTURBANCES, "suite"], default="none")
    ap.add_argument("--videos", type=int, default=5, help="max failure videos to render per run")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    kinds = DISTURBANCES if args.disturb == "suite" else (args.disturb,)
    with Pool(args.workers) as pool:
        runs = {k: run_one(args, k, pool) for k in kinds}
    if args.disturb == "suite":
        name = (args.name or f"openarm_{args.policy.replace(':', '-')}_{args.split}") + "_suite"
        path = REPO_ROOT / "runs" / "eval_summaries" / f"{name}.md"
        path.write_text(suite_md(name, args.policy, args.split, runs))
        print(path.read_text())
        print(f"suite: {path}")


if __name__ == "__main__":
    main()

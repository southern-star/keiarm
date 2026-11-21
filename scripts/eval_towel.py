"""Evaluate a policy on the towel half fold (milestone 3) and write a report.

Usage:
  uv run python scripts/eval_towel.py --policy expert
  uv run python scripts/eval_towel.py --policy expert --disturb suite [--mesh 10]
  uv run python scripts/eval_towel.py --policy expert-noisy:0.0065 --disturb slip

--disturb: none | shift | slip | drop | nudge | suite (all five, one run each, plus a
combined report). Every run is named towel_<policy>_<split>_n<mesh>[_<disturbance>] and writes
  runs/eval/<name>/episodes.json        every episode (works with scripts/compare.py)
  runs/eval_summaries/<name>.{json,md}  the summary (tracked in git)
  runs/failures/<name>/                 per failure: seeds, actions, disturbance (.npz), video (.mp4)
"""

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

from keiarm import config
from keiarm.paths import REPO_ROOT
from keiarm.towel.env import CAMERAS, EPISODE_SECONDS, TowelEnv
from keiarm.towel.eval import DISTURBANCES, EpisodeRecord, replay, run_episode, summarize
from keiarm.towel.policies import make_policy
from keiarm.towel.scene import TOWEL_N
from keiarm.towel.task import STAGES

STAGE_JA = {"reach": "接近", "grasp_one": "片手で把持", "grasp_both": "両手で把持", "lift": "持ち上げ",
            "carry": "運搬", "placed": "置く", "released": "離す", "success": "完了"}
FAILURE_JA = {"sim_broken": "シミュレーションの破綻", "towel_off_table": "布が机の外へ",
              "layers_crossed": "層のすり抜け", "grasp_failed": "掴み損ね", "lift_failed": "持ち上げ失敗",
              "carry_failed": "運搬失敗", "place_failed": "置き損ね", "release_failed": "離せない",
              "shape_failed": "離した後の形が基準外"}
DISTURB_JA = {
    "none": "なし",
    "shift": "両手が閉じる前（10〜37 ステップ目）に、布を出現範囲の別の位置・向きへ移す（角が 8 mm 以上動く）",
    "slip": "持ち上げ（奥の角が 3 cm）から 0〜19 ステップ後に、片手（乱択）の拘束を外す。指は閉じたまま",
    "drop": "持ち上げから 0〜19 ステップ後に、両手の拘束を外す（布が半端に折れて落ちる）",
    "nudge": "両手が離してから 3〜10 ステップ後に、上の層の片方の角（半径 5 cm）を 0.8〜1.2 m/s で "
             "斜め上（仰角 60°）、折り目の向こうへ（±30°）はじく。角は半分めくれて元の位置から 3〜5 cm に落ちる",
}


def _one(job) -> EpisodeRecord:
    spec, scene_seed, episode_seed, disturb, mesh = job
    policy = make_policy(spec)
    env = TowelEnv(scene_seed, render=policy.needs_images, n=mesh)
    try:
        return run_episode(env, policy, episode_seed, disturb=disturb)
    finally:
        env.close()


def write_failure(rec: EpisodeRecord, out, video: bool, scale: int = 2) -> None:
    stem = out / f"s{rec.scene_seed}_e{rec.episode_seed}_{rec.failure}"
    d = rec.disturbance
    np.savez_compressed(f"{stem}.npz", scene_seed=rec.scene_seed, episode_seed=rec.episode_seed, mesh=rec.mesh,
                        actions=rec.actions, failure=rec.failure, disturb=rec.disturb,
                        expert_could_finish=bool(rec.expert_could_finish),
                        disturbance=json.dumps(asdict(d) if d else None))
    if not video:
        return
    env = TowelEnv(rec.scene_seed, n=rec.mesh, width=config.IMAGE_WIDTH * scale, height=config.IMAGE_HEIGHT * scale)
    try:
        with imageio.get_writer(f"{stem}.mp4", fps=config.CONTROL_HZ, macro_block_size=1) as writer:
            replay(env, rec, on_step=lambda e: writer.append_data(
                np.concatenate([e.render(c) for c in CAMERAS], axis=1)))
    finally:
        env.close()


def _fmt_step(v) -> str:
    return "—" if v is None else f"{v:.0f}（{v / config.CONTROL_HZ:.1f} 秒）"


def report_md(name: str, spec: str, split: str, disturb: str, summary: dict, records: list[EpisodeRecord],
              seconds: float) -> str:
    n = summary["episodes"]
    inf = summary["infractions"]
    lines = [
        f"# 評価レポート: {name}", "",
        (f"- 方策: `{spec}` / スイート: {split}（{n} エピソード）/ 布のメッシュ N = {summary['mesh']} / "
        f"制限 {EPISODE_SECONDS} 秒"),
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
        lines.append(f"| 角の距離（成功時の最大） | {summary['success_corner_gap_mm_max']:.1f} mm |")
    lines += [
        f"| 掴んだ回数（中央値、外乱なしは 2） | {summary['grabs_median']:.0f} |",
    ]
    if disturb != "none":
        lines.append(f"| 外乱の後の角の距離の最大（中央値） | {summary['max_gap_after_mm_median']:.0f} mm |")
    lines += [
        f"| 腕どうしの接触があったエピソード | {inf['arms_touch_episodes']}（計 {inf['arms_touch_steps']} ステップ） |",
        f"| 層のすり抜け（中心面の交差）があったエピソード | {inf['crossing_episodes']}（最大 {inf['max_crossings']} 箇所） |",
        f"| 布が机の外へ | {inf['towel_off_table']} |",
        f"| 掴み損ね（閉じたが何も掴まなかった） | {inf['empty_grasps']} 回 |",
        f"| 早すぎる手放し（角が手前の角の上に無いまま開いた） | {inf['early_releases']} 回 |",
        f"| シミュレーションの破綻 | {inf['sim_broken']} |", "",
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
                  ("| scene_seed | episode_seed | 分類 | 最遠段階（外乱後） | 外乱のステップ | エキスパートなら完了 | "
                  "角の距離 [mm]（右, 左） | 面積比 | 違反 | 交差 |"),
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in fails[:40]:
            d = r.disturbance.step if r.disturbance and r.disturbance.step is not None else "—"
            could = {True: "はい", False: "いいえ", None: "—"}[r.expert_could_finish]
            st = r.stage_after_disturbance or r.stage
            g = r.final_corner_gap_mm
            lines.append(f"| {r.scene_seed} | {r.episode_seed} | {FAILURE_JA[r.failure]} | "
                         f"{STAGE_JA.get(st, '未到達')} | {d} | {could} | {g['right'][0]}, {g['left'][0]} | "
                         f"{r.final_area_ratio:.2f} | {r.final_violation:.2f} | {r.crossing_steps} |")
        if len(fails) > 40:
            lines.append(f"\n（残り {len(fails) - 40} 件は episodes.json）")
    else:
        lines += ["## 失敗の内訳", "", "失敗なし。"]
    return "\n".join(lines) + "\n"


def suite_md(name: str, spec: str, split: str, runs: dict) -> str:
    lines = [f"# 外乱スイート: {name}", "", f"- 方策: `{spec}` / スイート: {split}", "",
             ("| 外乱 | 成功率 | 外乱が起きた本数 | 腕の接触 | すり抜け | 机の外 | 掴み損ね | 掴んだ回数 | "
             "所要ステップ中央値 | レポート |"),
             "|---|---|---|---|---|---|---|---|---|---|"]
    for kind, (run_name, s) in runs.items():
        inf = s["infractions"]
        med = s["steps_to_success"]["median"] if s["steps_to_success"] else None
        lines.append(f"| {kind} | **{s['success']}/{s['episodes']}** | "
                     f"{s['disturbance_applied'] if kind != 'none' else '—'} | {inf['arms_touch_episodes']} | "
                     f"{inf['crossing_episodes']} | {inf['towel_off_table']} | {inf['empty_grasps']} | "
                     f"{s['grabs_median']:.0f} | {_fmt_step(med)} | [{run_name}]({run_name}.md) |")
    lines += ["", "外乱の内容:", ""] + [f"- **{k}**: {DISTURB_JA[k]}" for k in runs]
    return "\n".join(lines) + "\n"


def base_name(args) -> str:
    ep = f"_ep{args.episodes}" if args.episodes > 1 else ""  # keeps 1- and 3-episode runs apart
    return args.name or f"towel_{args.policy.replace(':', '-')}_{args.split}_n{args.mesh}{ep}"


def run_one(args, disturb: str, pool: Pool) -> tuple[str, dict]:
    name = base_name(args) + ("" if disturb == "none" else f"_{disturb}")
    seeds = list(config.EVAL_SCENE_SEEDS if args.split == "eval" else config.TRAIN_SCENE_SEEDS)[: args.n]
    jobs = [(args.policy, s, e, disturb, args.mesh) for s in seeds for e in range(args.episodes)]
    t0 = time.perf_counter()
    records = pool.map(_one, jobs, chunksize=1)
    seconds = time.perf_counter() - t0
    summary = summarize(records)
    summary |= {"disturb": disturb, "name": name, "policy": args.policy, "split": args.split,
                "mujoco": mujoco.__version__, "seconds": round(seconds)}
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
    print(report, flush=True)
    print(f"summary: {sum_dir / (name + '.md')}")
    if fails:
        print(f"failures: {fail_dir} ({len(fails)} logs, {min(len(fails), args.videos)} videos)", flush=True)
    return name, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert", help="expert | expert-noisy:<sigma> | hold | freeze[:<phase>] | keigrasp:<run> | keigrasp-blind:<run>[:<camera>]")
    ap.add_argument("--name", help="run name prefix (default: towel_<policy>_<split>_n<mesh>)")
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--n", type=int, default=100, help="number of scenes")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per scene")
    ap.add_argument("--mesh", type=int, default=TOWEL_N, help="towel vertices per side")
    ap.add_argument("--disturb", choices=[*DISTURBANCES, "suite"], default="none")
    ap.add_argument("--videos", type=int, default=5, help="max failure videos to render per run")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    kinds = DISTURBANCES if args.disturb == "suite" else (args.disturb,)
    with Pool(args.workers) as pool:
        runs = {k: run_one(args, k, pool) for k in kinds}
    if args.disturb == "suite":
        path = REPO_ROOT / "runs" / "eval_summaries" / f"{base_name(args)}_suite.md"
        path.write_text(suite_md(base_name(args) + "_suite", args.policy, args.split, runs))
        print(path.read_text())
        print(f"suite: {path}")


if __name__ == "__main__":
    main()

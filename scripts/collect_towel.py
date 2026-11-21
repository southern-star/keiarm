"""Collect towel training data (milestone 3): expert demonstrations with DART noise (BC),
or DAgger rollouts of a policy labelled by the expert.

Usage:
  uv run python scripts/collect_towel.py --name towel_bc1 --scenes 0 600 --workers 10
  uv run python scripts/collect_towel.py --name towel_dagger1 --scenes 600 900 --disturb mix --policy keigrasp:towel_bc1
Writes runs/data/<name>/*.npz. Scene seeds must be in the training split. Episodes already
on disk are skipped (a stopped collection can be restarted with the same command).
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")

from keiarm.paths import REPO_ROOT
from keiarm.towel.eval import DISTURBANCES
from keiarm.towel.policies import make_policy
from keigrasp.data import require_train_seed, save_episode
from keigrasp.towel import DART_SIGMAS, collect_episode

MIX = DISTURBANCES[1:]  # --disturb mix: shift / slip / drop / nudge in turn over the scenes


def _path(out, seed, ep, source):
    return out / f"s{seed:06d}_e{ep}_{source}.npz"


def _one(job):
    out, scene_seed, episode_seed, sigma, label_every, disturb, policy_spec, mesh = job
    t0 = time.perf_counter()
    policy = make_policy(policy_spec) if policy_spec else None
    ep = collect_episode(scene_seed, episode_seed, sigma if policy is None else 0.0, label_every=label_every,
                         disturb=disturb, policy=policy, n=mesh)
    path = save_episode(out, ep)
    return {"scene": scene_seed, "episode": episode_seed, "sigma": sigma, "disturb": disturb,
            "samples": len(ep["steps"]), "success": bool(ep["success_at"] >= 0), "stage": ep["stage"],
            "bytes": path.stat().st_size, "seconds": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--scenes", type=int, nargs=2, default=[0, 600], metavar=("FIRST", "END"))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--label-every", type=int, default=3)
    ap.add_argument("--disturb", default="none",
                    help="none | shift | slip | drop | nudge | mix (= shift,slip,drop,nudge) | a comma list rotated over scenes")
    ap.add_argument("--policy", help="DAgger: let this policy drive (e.g. keigrasp:towel_bc1); the expert only labels")
    ap.add_argument("--mesh", type=int, help="towel vertices per side (default: the training mesh, N = 16)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    kinds = MIX if args.disturb == "mix" else tuple(args.disturb.split(","))
    if any(k not in DISTURBANCES for k in kinds):
        ap.error(f"--disturb: unknown kind in {args.disturb!r} (choose from {', '.join(DISTURBANCES)})")
    out = REPO_ROOT / "runs" / "data" / args.name
    source = "dagger" if args.policy else "expert"
    jobs = []
    for seed in range(*args.scenes):
        require_train_seed(seed)
        for ep in range(args.episodes):
            if _path(out, seed, ep, source).exists():
                continue
            disturb = kinds[(seed + ep) % len(kinds)]
            jobs.append((out, seed, ep, DART_SIGMAS[(seed + ep) % len(DART_SIGMAS)], args.label_every,
                         disturb, args.policy, args.mesh))
    print(f"{len(jobs)} episodes to collect into {out}", flush=True)
    t0 = time.perf_counter()
    rows = []
    log = out / "collect_log.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(_one, jobs, chunksize=1):
            rows.append(row)
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if len(rows) % 10 == 0 or len(rows) == len(jobs):
                h = (time.perf_counter() - t0) / 3600
                print(f"{len(rows)}/{len(jobs)} done, {len(rows) / h:.0f} episodes/hour", flush=True)
    wall = time.perf_counter() - t0
    samples = sum(r["samples"] for r in rows)
    size = sum(r["bytes"] for r in rows) / 1e6
    who = args.policy or "expert"
    ok = sum(r["success"] for r in rows)
    print(f"{len(rows)} episodes, {samples} labelled samples ({samples / max(1, len(rows)):.0f}/episode), "
          f"{who} success {ok}/{len(rows)}, {size:.0f} MB ({size / max(1, len(rows)):.1f} MB/episode) in {out}, "
          f"{wall:.0f}s ({len(rows) / (wall / 3600):.0f} episodes/hour with {args.workers} workers)")


if __name__ == "__main__":
    main()

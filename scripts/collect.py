"""Collect training data: expert demonstrations with DART noise (BC), or DAgger rollouts.

Usage:
  uv run python scripts/collect.py --name bc1 --scenes 0 1200
  uv run python scripts/collect.py --name bc1d --scenes 1200 2000 --disturb
  uv run python scripts/collect.py --name dagger1 --scenes 2000 2600 --disturb --policy keigrasp:bc1 --workers 6
Writes runs/data/<name>/*.npz. Scene seeds must be in the training split.
"""

import argparse
import os
import time
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")

from keiarm.paths import REPO_ROOT
from keiarm.policies import make_policy
from keigrasp.data import DART_SIGMAS, collect_episode, require_train_seed, save_episode


def _one(job):
    out, scene_seed, episode_seed, sigma, label_every, disturb, policy_spec = job
    policy = make_policy(policy_spec) if policy_spec else None
    ep = collect_episode(scene_seed, episode_seed, sigma if policy is None else 0.0, label_every=label_every,
                         disturb=disturb, policy=policy)
    save_episode(out, ep)
    return len(ep["steps"]), ep["success_at"] >= 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--scenes", type=int, nargs=2, default=[0, 1000], metavar=("FIRST", "END"))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--label-every", type=int, default=3)
    ap.add_argument("--disturb", action="store_true", help="move the cube once mid-episode")
    ap.add_argument("--policy", help="DAgger: let this policy drive (e.g. keigrasp:bc1); the expert only labels")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    out = REPO_ROOT / "runs" / "data" / args.name
    jobs = []
    for seed in range(*args.scenes):
        require_train_seed(seed)
        for ep in range(args.episodes):
            jobs.append((out, seed, ep, DART_SIGMAS[(seed + ep) % len(DART_SIGMAS)], args.label_every,
                         "random" if args.disturb else None, args.policy))
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        results = pool.map(_one, jobs, chunksize=4)
    samples = sum(n for n, _ in results)
    size = sum(f.stat().st_size for f in out.glob("*.npz")) / 1e6
    who = args.policy or "expert"
    print(f"{len(jobs)} episodes, {samples} labelled samples, {who} success {sum(ok for _, ok in results)}/{len(jobs)}, "
          f"{size:.0f} MB in {out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()

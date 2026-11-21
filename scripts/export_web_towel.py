"""Export towel scenes for the browser (milestone 3, Phase 4): web/public/towel_scenes/
(generated, not in git). The robot meshes are milestone 2's (web/public/assets/openarm/);
they are written if missing.

Usage: uv run python scripts/export_web_towel.py [--scenes 100000 ...] [--meshes 8 10 12] [--episodes 0]
Each scene and mesh N gets the clean expert for every --episodes seed, plus one disturbed
episode (shift / slip / drop / nudge in turn), whose disturbance the browser replays.
"""

import argparse
import json
import time
from multiprocessing import Pool

from keiarm import config
from keiarm.towel import export


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+", default=list(config.EVAL_SCENE_SEEDS)[:8])
    ap.add_argument("--meshes", type=int, nargs="+", default=[10])
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        index = export.export_web(args.scenes, args.meshes, args.episodes, map_fn=pool.map)
    print(json.dumps(index, indent=1))
    print(f"{len(args.scenes)} scenes x {len(args.meshes)} meshes in {time.perf_counter() - t0:.0f} s, "
          f"{sum(index['scene_bytes'].values()) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

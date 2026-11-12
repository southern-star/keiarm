"""Export OpenArm scenes for the browser (milestone 2): web/public/openarm_scenes/ and the
(decimated) meshes in web/public/assets/openarm/. Both are generated, not in git.

Usage: uv run python scripts/export_web_openarm.py [--scenes 100000 100001 ...] [--keep 0.25]
Each scene gets the clean expert for every --episodes seed, plus one disturbed episode
(knock / drop / plate / move in turn) so that the browser's disturbance code is replayed too.
"""

import argparse
import json
import time
from multiprocessing import Pool

from keiarm import config
from keiarm.openarm import export


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+", default=list(config.EVAL_SCENE_SEEDS)[:8])
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--keep", type=float, default=export.VISUAL_KEEP, help="fraction of visual mesh faces kept")
    args = ap.parse_args()
    t0 = time.perf_counter()
    with Pool(min(len(args.scenes), 8)) as pool:  # scenes are independent
        index = export.export_web(args.scenes, args.episodes, keep=args.keep, map_fn=pool.map)
    print(json.dumps(index, indent=1))
    mb = index["mesh_bytes"] / 1e6
    print(f"{len(args.scenes)} scenes in {time.perf_counter() - t0:.0f} s; meshes {mb:.2f} MB "
          f"(visual faces {index['visual_faces_before']} -> {index['visual_faces_after']})")


if __name__ == "__main__":
    main()

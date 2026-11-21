"""Record towel episodes with the training mesh (N = 16) for playback in the browser
(milestone 3, Phase 4): web/public/towel_records/ (generated, not in git).

Usage: uv run python scripts/export_towel_records.py
The page (web/towel.html, "記録の再生") poses the robot and the towel from the recorded frames
without running physics. Until a policy exists (Phase 5) the records are the expert's: clean
successes, recoveries from the disturbance suite and a failure (the expert with large DART noise).
"""

import argparse
import json
import time
from multiprocessing import Pool

from keiarm.towel import export

RECORDS = [
    # name, label, scene_seed, episode_seed, disturb, policy
    ("expert_100000", "エキスパート（シーン 100000）", 100000, 0, "none", "expert"),
    ("expert_100003", "エキスパート（シーン 100003）", 100003, 0, "none", "expert"),
    ("expert_100007", "エキスパート（シーン 100007）", 100007, 0, "none", "expert"),
    ("slip_100001", "片手が滑ってから掴み直す（シーン 100001）", 100001, 0, "slip", "expert"),
    ("drop_100002", "両手が滑ってから折り直す（シーン 100002）", 100002, 0, "drop", "expert"),
    ("nudge_100004", "はじかれた角を置き直す（シーン 100004）", 100004, 0, "nudge", "expert"),
    ("noisy_100014", "失敗例: ノイズの大きいエキスパート（σ = 0.0065、シーン 100014）", 100014, 0, "none",
     "expert-noisy:0.0065"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        index = export.export_records(RECORDS, map_fn=pool.map)
    print(json.dumps(index, indent=1, ensure_ascii=False))
    total = sum(r["bytes"] for r in index["records"])
    print(f"{len(index['records'])} records in {time.perf_counter() - t0:.0f} s, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

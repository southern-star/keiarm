"""速度だけを単一プロセスで測り直す（他の重い処理を止めてから実行する）。

- 布単体（E2 の二つ折り → 四つ折り、seed 0）: 候補 A（edge 拘束 + 曲げ）N = 10 / 16 / 24、候補 B（拘束なし + 伸び・曲げの弾性）N = 16
- 双腕 + 布（E3 の二つ折り、seed 0）: N = 10 / 16
- 参考: 既存の OpenArm 剛体タスク（scripts/bench_openarm.py を同じ機械で実行）

実行: uv run python spikes/cloth/bench.py
出力: spikes/cloth/out/bench.json
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import e2_fold
import e3_robot_fold
from cloth import ClothCfg

CLOTH = {
    "A_n10": ClothCfg(n=10),
    "A_n16": ClothCfg(n=16),
    "A_n24": ClothCfg(n=24),
    "B_n16": ClothCfg(n=16, equality="false", elastic2d="both", young=2e5),
    "Bfast_n10": ClothCfg(n=10, equality="false", elastic2d="both", young=2e5, iterations=10),
    "Bfast_c1_n10": ClothCfg(n=10, equality="false", elastic2d="both", young=2e5, iterations=10, condim=1),
}
ROBOT = {
    "A_n10": ClothCfg(n=10, side=0.20),
    "A_n16": ClothCfg(n=16, side=0.20),
    "B_n16": ClothCfg(n=16, side=0.20, equality="false", elastic2d="both", young=2e5),
    "Bfast_n10": ClothCfg(n=10, side=0.20, equality="false", elastic2d="both", young=2e5, iterations=10),
}


def main():
    out = {"cpu": platform.processor() or platform.machine(), "cloth_only": {}, "robot_cloth": {}}
    for name, cfg in CLOTH.items():
        r = e2_fold.trial((cfg.to_dict(), 0, False, False))
        out["cloth_only"][name] = {k: r[k] for k in ("status", "rtf_fold", "rtf_rest", "sim_s", "wall_s", "ncon_max")}
        print("cloth", name, out["cloth_only"][name], flush=True)
    for name, cfg in ROBOT.items():
        r = e3_robot_fold.trial((cfg.to_dict(), 0, False, False))
        out["robot_cloth"][name] = {k: r.get(k) for k in ("status", "rtf_physics", "rtf_with_ik", "sim_s", "viol0", "corner_mm")}
        print("robot", name, out["robot_cloth"][name], flush=True)
    # 決定性: 同じ seed を 2 回（同じプロセス）。時間以外の値が一致するか
    timing = {"rtf_fold", "rtf_rest", "wall_s", "rtf_physics", "rtf_with_ik"}
    det = {}
    for label, fn, cfg in (("cloth_A_n16", e2_fold.trial, CLOTH["A_n16"]), ("cloth_B_n16", e2_fold.trial, CLOTH["B_n16"]),
                           ("robot_A_n16", e3_robot_fold.trial, ROBOT["A_n16"])):
        a, b = (fn((cfg.to_dict(), 7, False, False)) for _ in range(2))
        diff = sorted(k for k in a if k not in timing and a[k] != b.get(k))
        det[label] = {"identical": not diff, "differing_keys": diff}
        print("determinism", label, det[label], flush=True)
    out["determinism"] = det
    rigid = subprocess.run(["uv", "run", "python", "scripts/bench_openarm.py"], cwd=HERE.parents[1], check=False,
                           capture_output=True, text=True)
    out["openarm_rigid_bench_stdout"] = rigid.stdout.strip().splitlines()[-6:]
    print("\n".join(out["openarm_rigid_bench_stdout"]))
    (HERE / "out" / "bench.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""E5 の準備: E2 の布単体シーンと試行のパラメータを JSON に書き出し、同じ試行の Python の結果も添える。

出力: spikes/cloth/web/scenes/<name>.json（xml、パラメータ、Python の結果）と index.json

実行: uv run python spikes/cloth/web/export_scenes.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import e2_fold
from cloth import ClothCfg, table_scene_xml

CONFIGS = {
    "A_n10": ClothCfg(n=10),
    "A_n16": ClothCfg(n=16),
    "A_n24": ClothCfg(n=24),
    "B_n16": ClothCfg(n=16, equality="false", elastic2d="both", young=2e5),
    "B_n10": ClothCfg(n=10, equality="false", elastic2d="both", young=2e5),
}
SEEDS = (0, 1, 2)


def trial_params(seed: int):
    # e2_fold.trial と同じ乱数の引き方
    rng = np.random.default_rng(np.random.SeedSequence([0xC107, seed]))
    off = rng.uniform(-0.02, 0.02, 2)
    yaw = rng.uniform(-np.radians(10), np.radians(10))
    t_fold = [rng.uniform(1.5, 3.0), rng.uniform(1.5, 3.0)]
    lift_ratio = [rng.uniform(0.45, 0.8), rng.uniform(0.45, 0.8)]
    return off, yaw, t_fold, lift_ratio


def main():
    only = set(sys.argv[1:])  # 設定名を指定すればその分だけ書き出す（index.json は全シーン分を作り直す）
    out = HERE / "scenes"
    out.mkdir(exist_ok=True)
    index = []
    for name, cfg in CONFIGS.items():
        for seed in SEEDS:
            key = f"{name}_s{seed}"
            index.append(key)
            if only and name not in only:
                continue
            off, yaw, t_fold, lift = trial_params(seed)
            xml = table_scene_xml(cfg, cloth_pos=(off[0], off[1], cfg.radius + 0.001),
                                  cloth_euler=(0, 0, np.degrees(yaw)))
            py = e2_fold.trial((cfg.to_dict(), seed, False, False))
            py = {k: v for k, v in py.items() if not k.endswith("_bad")}
            (out / f"{key}.json").write_text(json.dumps({
                "name": key, "cfg": cfg.to_dict(), "xml": xml,
                "params": {"n": cfg.n, "spacing": cfg.spacing(), "radius": cfg.radius, "timestep": cfg.timestep,
                           "t_fold": t_fold, "lift_ratio": lift},
                "python": py,
            }))
            print(key, py["status"], py.get("half_corner_mm"), py.get("quarter_corner_mm"), py["rtf_fold"], flush=True)
    (out / "index.json").write_text(json.dumps(index))


if __name__ == "__main__":
    main()

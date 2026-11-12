# ブラウザ内評価: openarm_keigrasp-oa_r1_browser_suite

- モデル: `oa_r1`（ONNX、47.2 MB）/ 評価 100 シーン × 1 エピソード
- 物理 MuJoCo WASM、描画 three.js（間引いたメッシュ、ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 3060/PCIe/SSE2, OpenGL ES 3.2)）、推論 onnxruntime-web（wasm、平均 177 ms）
- シミュレーションは推論を待つ（同期実行）。所要 2805 秒

| 外乱 | ブラウザ | Python | 両方成功 | Python のみ | ブラウザのみ | 両方失敗 | 符号検定 p | 所要ステップ中央値（ブラウザ） |
|---|---|---|---|---|---|---|---|---|
| none | **100/100** | 100/100 | 100 | 0 | 0 | 0 | 1 | 225 |
| move | **100/100** | 100/100 | 100 | 0 | 0 | 0 | 1 | 240 |
| knock | **100/100** | 100/100 | 100 | 0 | 0 | 0 | 1 | 304 |
| drop | **99/100** | 100/100 | 99 | 1 | 0 | 0 | 1 | 392 |
| plate | **100/100** | 100/100 | 100 | 0 | 0 | 0 | 1 | 227 |

失敗した (scene_seed, episode_seed, 最遠段階):
- drop: (100005, 0, left_carry)

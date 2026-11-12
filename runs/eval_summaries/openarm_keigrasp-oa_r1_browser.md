# ブラウザ内評価: openarm_keigrasp-oa_r1_browser

- モデル: `oa_r1`（ONNX、47.2 MB）/ 評価 100 シーン × 3 エピソード
- 物理 MuJoCo WASM、描画 three.js（間引いたメッシュ、ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 3060/PCIe/SSE2, OpenGL ES 3.2)）、推論 onnxruntime-web（wasm、平均 178 ms）
- シミュレーションは推論を待つ（同期実行）。所要 1518 秒

| 外乱 | ブラウザ | Python | 両方成功 | Python のみ | ブラウザのみ | 両方失敗 | 符号検定 p | 所要ステップ中央値（ブラウザ） |
|---|---|---|---|---|---|---|---|---|
| none | **299/300** | 300/300 | 299 | 1 | 0 | 0 | 1 | 225 |

失敗した (scene_seed, episode_seed, 最遠段階):
- none: (100005, 1, lift)

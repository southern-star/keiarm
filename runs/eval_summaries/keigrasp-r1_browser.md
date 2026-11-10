# ブラウザ内評価: keigrasp-r1_browser

- モデル: `r1`（ONNX、36.7 MB）/ 評価 100 シーン × 3 エピソード
- 外乱: なし
- 物理 MuJoCo WASM、描画 three.js（ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)）、推論 onnxruntime-web（wasm、平均 182 ms）
- シミュレーションは推論を待つ（同期実行）。所要 1059 秒

| 指標 | 値 |
|---|---|
| 成功率 | **297/300（99.0%）** |
| 所要ステップ数（成功時、中央値） | 68 |

## Python 描画（`keigrasp-r1_eval3`）との対比較

共通 300 エピソード: Python 298 成功、ブラウザ 297 成功。
両方成功 297、両方失敗 2、Python のみ 1、ブラウザのみ 0。
対応のある符号検定（両側、正確）: p = 1

失敗した (scene_seed, episode_seed): (100016, 2), (100045, 1), (100051, 2)

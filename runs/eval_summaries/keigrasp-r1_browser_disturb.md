# ブラウザ内評価: keigrasp-r1_browser_disturb

- モデル: `r1`（ONNX、36.7 MB）/ 評価 100 シーン × 3 エピソード
- 外乱: 各エピソードの計画どおりに立方体を 1 回移動
- 物理 MuJoCo WASM、描画 three.js（ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)）、推論 onnxruntime-web（wasm、平均 181 ms）
- シミュレーションは推論を待つ（同期実行）。所要 1445 秒

| 指標 | 値 |
|---|---|
| 成功率 | **300/300（100.0%）** |
| 所要ステップ数（成功時、中央値） | 113 |

## Python 描画（`keigrasp-r1_eval_disturb`）との対比較

共通 300 エピソード: Python 299 成功、ブラウザ 300 成功。
両方成功 299、両方失敗 0、Python のみ 0、ブラウザのみ 1。
対応のある符号検定（両側、正確）: p = 1

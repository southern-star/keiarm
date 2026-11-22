# ブラウザ内評価: towel_keigrasp-towel_r3_browser_n11_suite

- モデル: `towel_r3`（ONNX、47.2 MB）/ 評価 100 シーン × 1 エピソード、布 N = 11
- 物理 MuJoCo WASM、描画 three.js（間引いたメッシュ、ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 3060/PCIe/SSE2, OpenGL ES 3.2)）、推論 onnxruntime-web（wasm、平均 215 ms）
- シミュレーションは推論を待つ（同期実行）。ブラウザ 5 個で分担、所要 2145 秒

| 外乱 | ブラウザ | Python | 両方成功 | Python のみ | ブラウザのみ | 両方失敗 | 符号検定 p | 所要ステップ中央値（ブラウザ） |
|---|---|---|---|---|---|---|---|---|
| none | **100/100** | 100/100 | 100 | 0 | 0 | 0 | 1 | 160 |
| shift | **99/100** | 100/100 | 99 | 1 | 0 | 0 | 1 | 164 |
| slip | **85/100** | 92/100 | 80 | 12 | 5 | 3 | 0.143 | 220 |
| drop | **66/100** | 68/100 | 49 | 19 | 17 | 15 | 0.868 | 252 |
| nudge | **95/100** | 97/100 | 94 | 3 | 1 | 2 | 0.625 | 170 |

失敗した (scene_seed, episode_seed, 最遠段階):
- shift: (100071, 0, released)
- slip: (100001, 0, lift), (100013, 0, lift), (100019, 0, lift), (100020, 0, released), (100023, 0, carry), (100035, 0, released), (100050, 0, carry), (100055, 0, carry), (100060, 0, carry), (100061, 0, carry), (100070, 0, lift), (100072, 0, carry), (100078, 0, carry), (100088, 0, lift), (100099, 0, released)
- drop: (100000, 0, released), (100002, 0, carry), (100005, 0, carry), (100007, 0, lift), (100015, 0, released), (100018, 0, lift), (100019, 0, released), (100020, 0, lift), (100023, 0, placed), (100025, 0, lift), (100027, 0, lift), (100028, 0, carry), (100037, 0, released), (100040, 0, carry), (100041, 0, lift), (100042, 0, lift), (100043, 0, lift), (100044, 0, lift), (100046, 0, lift), (100047, 0, placed), (100052, 0, carry), (100054, 0, lift), (100063, 0, carry), (100070, 0, carry), (100072, 0, lift)
- nudge: (100001, 0, released), (100010, 0, released), (100015, 0, released), (100051, 0, released), (100062, 0, released)

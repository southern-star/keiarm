# 外乱スイート: openarm_keigrasp-oa_bc1_eval_suite

- 方策: `keigrasp:oa_bc1` / スイート: eval

| 外乱 | 成功率 | 外乱が起きた本数 | 腕の接触 | 落下/本 | 机の外/本 | 所要ステップ中央値 | レポート |
|---|---|---|---|---|---|---|---|
| none | **100/100** | — | 0.00 | 0.01 | 0.00 | 226（7.5 秒） | [openarm_keigrasp-oa_bc1_eval](openarm_keigrasp-oa_bc1_eval.md) |
| move | **98/100** | 100 | 0.00 | 0.05 | 0.00 | 246（8.2 秒） | [openarm_keigrasp-oa_bc1_eval_move](openarm_keigrasp-oa_bc1_eval_move.md) |
| knock | **83/100** | 100 | 0.00 | 0.20 | 0.00 | 320（10.7 秒） | [openarm_keigrasp-oa_bc1_eval_knock](openarm_keigrasp-oa_bc1_eval_knock.md) |
| drop | **45/100** | 100 | 0.00 | 0.77 | 0.00 | 536（17.9 秒） | [openarm_keigrasp-oa_bc1_eval_drop](openarm_keigrasp-oa_bc1_eval_drop.md) |
| plate | **100/100** | 100 | 0.00 | 0.01 | 0.00 | 228（7.6 秒） | [openarm_keigrasp-oa_bc1_eval_plate](openarm_keigrasp-oa_bc1_eval_plate.md) |

外乱の内容:

- **none**: なし
- **move**: 右手が閉じる前（10〜39 ステップ目）に、棒を右側の別の場所へ移す
- **knock**: 右手が持ち上げてから 5〜29 ステップ後に、棒を右側の机の上へ落とす（掴み直し）
- **drop**: 左手だけで持ってから 0〜19 ステップ後に、棒を右側の机の上へ落とす（受け渡しからやり直し）
- **plate**: 左手だけで持ってから 0〜19 ステップ後に、皿を別の場所へ移す

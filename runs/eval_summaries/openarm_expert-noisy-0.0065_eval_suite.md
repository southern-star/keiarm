# 外乱スイート: openarm_expert-noisy-0.0065_eval_suite

- 方策: `expert-noisy:0.0065` / スイート: eval

| 外乱 | 成功率 | 外乱が起きた本数 | 腕の接触 | 落下/本 | 机の外/本 | 所要ステップ中央値 | レポート |
|---|---|---|---|---|---|---|---|
| none | **90/100** | — | 0.00 | 0.05 | 0.08 | 330（11.0 秒） | [openarm_expert-noisy-0.0065_eval](openarm_expert-noisy-0.0065_eval.md) |
| move | **88/100** | 100 | 0.04 | 0.03 | 0.06 | 350（11.7 秒） | [openarm_expert-noisy-0.0065_eval_move](openarm_expert-noisy-0.0065_eval_move.md) |
| knock | **88/100** | 100 | 0.02 | 0.04 | 0.08 | 418（13.9 秒） | [openarm_expert-noisy-0.0065_eval_knock](openarm_expert-noisy-0.0065_eval_knock.md) |
| drop | **83/100** | 99 | 0.02 | 0.05 | 0.11 | 530（17.7 秒） | [openarm_expert-noisy-0.0065_eval_drop](openarm_expert-noisy-0.0065_eval_drop.md) |
| plate | **87/100** | 99 | 0.00 | 0.01 | 0.07 | 336（11.2 秒） | [openarm_expert-noisy-0.0065_eval_plate](openarm_expert-noisy-0.0065_eval_plate.md) |

外乱の内容:

- **none**: なし
- **move**: 右手が閉じる前（10〜39 ステップ目）に、棒を右側の別の場所へ移す
- **knock**: 右手が持ち上げてから 5〜29 ステップ後に、棒を右側の机の上へ落とす（掴み直し）
- **drop**: 左手だけで持ってから 0〜19 ステップ後に、棒を右側の机の上へ落とす（受け渡しからやり直し）
- **plate**: 左手だけで持ってから 0〜19 ステップ後に、皿を別の場所へ移す

# 外乱スイート: towel_expert_eval_n11_suite

- 方策: `expert` / スイート: eval

| 外乱 | 成功率 | 外乱が起きた本数 | 腕の接触 | すり抜け | 机の外 | 掴み損ね | 掴んだ回数 | 所要ステップ中央値 | レポート |
|---|---|---|---|---|---|---|---|---|---|
| none | **100/100** | — | 0 | 0 | 0 | 0 | 2 | 160（5.3 秒） | [towel_expert_eval_n11](towel_expert_eval_n11.md) |
| shift | **100/100** | 100 | 0 | 0 | 0 | 0 | 2 | 162（5.4 秒） | [towel_expert_eval_n11_shift](towel_expert_eval_n11_shift.md) |
| slip | **96/100** | 100 | 0 | 11 | 0 | 0 | 3 | 188（6.3 秒） | [towel_expert_eval_n11_slip](towel_expert_eval_n11_slip.md) |
| drop | **93/100** | 100 | 0 | 30 | 0 | 0 | 4 | 190（6.3 秒） | [towel_expert_eval_n11_drop](towel_expert_eval_n11_drop.md) |
| nudge | **100/100** | 100 | 0 | 1 | 0 | 0 | 2 | 167（5.6 秒） | [towel_expert_eval_n11_nudge](towel_expert_eval_n11_nudge.md) |

外乱の内容:

- **none**: なし
- **shift**: 両手が閉じる前（10〜37 ステップ目）に、布を出現範囲の別の位置・向きへ移す（角が 8 mm 以上動く）
- **slip**: 持ち上げ（奥の角が 3 cm）から 0〜19 ステップ後に、片手（乱択）の拘束を外す。指は閉じたまま
- **drop**: 持ち上げから 0〜19 ステップ後に、両手の拘束を外す（布が半端に折れて落ちる）
- **nudge**: 両手が離してから 3〜10 ステップ後に、上の層の片方の角（半径 5 cm）を 0.8〜1.2 m/s で 斜め上（仰角 60°）、折り目の向こうへ（±30°）はじく。角は半分めくれて元の位置から 3〜5 cm に落ちる

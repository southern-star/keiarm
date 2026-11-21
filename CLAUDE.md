# KeiArm（仮称）— 恒久的な方針

SO-101 を題材にした軽量クローズドループ・マニピュレーション環境（KeiArm）と、その中で学習する
カメラ入力の小型方策（KeiGrasp）。物理は MuJoCo を借り、それ以外は自作する。
設計の全文は `docs/design.md`、進捗と次の一手は `PROGRESS.md`。**毎セッションの最後に `PROGRESS.md` を更新する。**

## 固定した値（変える時はここと PROGRESS.md を両方直す）

| 項目 | 値 |
|---|---|
| MuJoCo（Python `mujoco` / npm `@mujoco/mujoco`） | **3.17.0**（両方に存在。`tests/test_so101.py` で検査） |
| Python | 3.12（uv 管理）。PyTorch は依存グループ `train`（CI は `--no-group train`） |
| ブラウザ側 | three.js 0.180.0、Vite 7、Node 20。MuJoCo WASM は単一スレッド版 |
| ロボットモデル | Menagerie `robotstudio_so101` @ `98a70c5`、`third_party/` にベンダリング。**編集禁止** |
| 行動 | 関節目標角度 6 個 [rad]、順序 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`（Menagerie・LeRobot と一致） |
| 手先サイト / 手首カメラ | `gripperframe` / `wrist_cam`（Menagerie 既存） |
| 制御周期 | 30 Hz。`timestep = 1/240` × 8 サブステップ（`keiarm/config.py`。Menagerie の 0.005 から変更） |
| 画像 | 160×120、順序 `cam_front, wrist_cam` |
| 出現範囲 | 肩旋回軸から半径 0.13–0.22 m（皿は 0.15–0.22 m）、方位 ±60°（`docs/reachability.png` が根拠） |
| 行動チャンク長 K | 30。実行時は先頭 10 ステップを実行して再推論 |
| 方策 | KeiGrasp 893 万パラメータ。入力 160×120 × 2 台 + 関節角 6。現行の最良は `runs/train/r1/model.pt`（git 管理外。`bc1` に外乱入りデータと DAgger 1 ラウンドを足したもの） |
| DART ノイズ | σ = 0.01 rad、AR(1) 係数 0.9、グリッパには載せない |
| 成功判定 | 設計図 10.1 の条件が 10 ステップ連続（`keiarm/task.py`、`keiarm/rollout.py`） |
| シード分割 | 訓練 scene_seed 0–3999 / 評価 100000–100099 |

### マイルストーン 2: OpenArm v2（両腕、`keiarm/openarm/`、計画は `docs/milestone2.md`）

| 項目 | 値 |
|---|---|
| ロボットモデル | `enactic/openarm_mujoco` @ `1daab06` の `v2/openarm_bimanual.xml`、`third_party/openarm_v2/` にベンダリング。**編集禁止** |
| 物理 | timestep 1/240 × 8（SO-101 と同じ）、implicitfast、elliptic、impratio 10。**アームの全ボディに `gravcomp = 1`（必須。無いと手先が 14 mm 沈む）** |
| 行動 | 16 個 = アクチュエータ順（左 7 関節、左指、右 7 関節、右指）[rad] |
| 座標 | 机の上面 z = 0、アームの取り付け高さ 0.297 m（Enactic の台座と机の高さの差） |
| 開始姿勢 | 指先の中点を (0.25, ±0.20, 0.20) に真下向き。左右は鏡像（`keiarm/openarm/robot.py`） |
| 最初のタスク | 右手側の**棒**（長さ 15–18 cm）を右手が −y 端で取り、中央 (0.27, 0, 0.15) の空中で左手が +y 端を掴んで受け取り、左手側の皿に置く。両手の掴む点は棒の中心から ±6 cm（`BAR_GRIP_OFFSET`。手の間隔 12 cm、9 cm で手首どうしが当たる） |
| 出現範囲 | 右手の掴む点（と鏡像で左手の置く点）を肩付近 (0, ∓0.152) 中心の扇形（半径 0.21–0.27 m、方位 −65°〜−10°）に置く。棒の向きは y 軸から −20°〜+45°（`keiarm/openarm/scene.py`、根拠は `docs/openarm/reach_bar.png`） |
| 把持の向き | 手ごとに「自然な」向きだけを使う: 顎の yaw = `HOME_YAW[side]`（右 +π/2、左 −π/2）+ 棒の角度。180° 反転した把持では棒を y 軸向きに回せない（手首の可動域） |
| DART ノイズ（OpenArm） | σ = 0.0065 rad を基準にする（SO-101 の 0.01 rad と手先の変位が同じになる値。手先の変位/関節角は SO-101 0.35 m/rad、OpenArm 0.54 m/rad）。収集では (0, 0.00325, 0.0065) を回す（0.00975 ではエキスパートが 48%） |
| 方策（OpenArm） | KeiGrasp をカメラ 3 台（`cam_front`、左手首、右手首）・16 次元で 1178 万パラメータ。指の列 (7, 15) は絶対値で予測。現行の最良は `runs/train/oa_r1/model.pt`（git 管理外。`oa_bc1` に外乱入りデータと DAgger 1 ラウンドを足したもの）。**使用済みの訓練シーン: 0–1199** |

### マイルストーン 3: タオルの二つ折り（OpenArm v2、`keiarm/towel/`、計画は `docs/milestone3.md`）

| 項目 | 値 |
|---|---|
| 物理 | timestep 1/240 × 8、**discrete**、**CG**、iterations 50、elliptic、impratio 10（pyramidal では掴んだ布が机を突き抜ける）。アームは M2 と同じ（`openarm_spec()`） |
| 布 | `flexcomp` grid、学習用 **N = 16**（頂点 256）、**ブラウザ用 N = 10**（N = 8・9 はエキスパートが 69%・53%）。radius 2 mm、edge 拘束 + 曲げ、selfcollide sap |
| タオル | 長方形、奥行き（折る向き x）15–16.5 cm × 幅 18.5–20 cm。質量・摩擦・曲げ・色はシードで振る（`keiarm/towel/scene.py`） |
| 出現範囲 | 中心 (0.239 ± 0.008, ±0.010)、向き ±5°、広げて平らに置く（根拠 `tools/towel_reach.py`） |
| タスク | 右手が −y、左手が +y の奥の角（角から 6 mm 内側）を真上から掴み、手前の角の上へ運んで離す |
| 掴みの規則 | 指令が開の 25% 未満かつ指の開度が 35% 未満で、指先の中点から 2 cm 以内の頂点を近い順に最大 4 個 `connect`（`<side>_grab<k>`）で拘束、指令が 50% を超えたら離す（`keiarm/towel/grasp.py`）。**Python と `web/src/towel/sim.js` の両方に書く**。mjNEQDATA は 3.17 で 12 |
| 成功判定 | 奥の角が手前の角から水平 3 cm・垂直 2.5 cm 以内、真上から見た面積 0.40–0.60、層の順序の違反 10% 以下（**折り目をまたぐ 1 組は数えない**）、両手が離して指が触れていない、全頂点 5 cm/s 未満、が 10 ステップ連続（`keiarm/towel/task.py`） |
| エキスパート | `Expert.act(data, commanded=None)`。**DART では前回のノイズなしラベルを `commanded` に渡す**（運搬の起点。`data.ctrl` を起点にするとノイズを積分する）。実行するのはノイズ入りの行動（`keiarm/towel/policies.py` の `ExpertPolicy`） |
| DART ノイズ（タオル） | 収集では σ = (0, 0.00325, 0.0065) を回す（エキスパート 100% / 91%。0.00975 は 10%） |
| ブラウザの速度 | N = 10 の物理は実時間の 0.76〜0.84 倍。0.8 倍速前後で見せる（利用者の決定 2026-11-14）。推論込みの速度は Phase 7 で見直す |

## コマンド

- 環境構築: `./setup.sh`（= `uv sync`）
- テスト: `uv run pytest -q` / lint: `uv run ruff check .`
- 描画を伴う実行は `MUJOCO_GL=egl`（ヘッドレス）
- デモ動画: `uv run python scripts/demo.py` / シーン一覧: `uv run python scripts/show_scenes.py`
- **評価: `uv run python scripts/eval.py --policy expert`**（レポートは `runs/eval_summaries/`、失敗ログは `runs/failures/`）/ 比較: `uv run python scripts/compare.py <A> <B>`
- エキスパート単体: `uv run python scripts/run_expert.py --scene 100000 --video runs/x.mp4` / 外乱つき: `uv run python scripts/eval_expert.py --teleport-at 35`
- ブラウザ: `uv run python scripts/export_web.py` → `npm --prefix web run dev`（開発）/ `npm --prefix web test`（Node で行動列を再生）/ `npm --prefix web run build && uv run python tools/browser_check.py`（ヘッドレス Chrome で再生と描画比較）
- 学習: `uv run python scripts/collect.py --name <データ名> --scenes 0 1200` → `uv run python scripts/train.py --name <run 名> --data <データ名>` → `uv run python scripts/eval.py --policy keigrasp:<run 名> --workers 6 [--disturb] [--episodes 3]`
- DAgger: `uv run python scripts/collect.py --name <名前> --scenes A B --disturb --policy keigrasp:<run 名> --workers 6` → `scripts/train.py --init <run 名> --data <全部>`。**ラウンドごとに新しいシーン範囲を使う**（使用済み: 0–1899、2000–2599）
- ブラウザのモデル: `uv run python scripts/export_onnx.py --run r1` → `npm --prefix web run build` → `uv run python tools/browser_eval.py --python-run keigrasp-r1_eval3`（ブラウザ内の閉ループ評価、約 18 分）/ `uv run python tools/browser_demo_check.py`（実時間 + マウスドラッグ）
- ベンチ: `uv run python scripts/bench.py`（OpenArm は `scripts/bench_openarm.py`。シーン一覧 `scripts/show_scenes_openarm.py`、到達範囲 `tools/openarm_reach.py --object bar`、評価 `scripts/eval_openarm.py --policy expert [--disturb none|move|knock|drop|plate|suite]`（レポートは `runs/eval_summaries/openarm_*`）、エキスパートの簡易評価 `scripts/eval_expert_openarm.py [--dart 0.0065]`、動画 `scripts/run_expert_openarm.py --video runs/x.mp4`、ブラウザ `scripts/export_web_openarm.py` → `npm --prefix web run build` → `tools/browser_check_openarm.py [--gpu]`、ページは `web/openarm.html`、モデル `scripts/export_onnx.py --robot openarm --run oa_r1`、ブラウザ内評価 `tools/browser_eval_openarm.py --gpu --python-run openarm_keigrasp-oa_r1_eval3 [--disturb suite]`、実時間 + ドラッグ `tools/browser_demo_check_openarm.py --gpu [--url ...]`、学習 `scripts/collect_openarm.py --name <名前> --scenes A B [--disturb mix] [--policy keigrasp:<run>]` → `scripts/train.py --robot openarm --name <run> --data <名前>` → `scripts/eval_openarm.py --policy keigrasp:<run> --workers 6`、対照は `--policy keigrasp-blind:<run>[:カメラ]`） / 到達範囲: `uv run python tools/reachability.py`
- タオル（M3）: ベンチ `scripts/bench_towel.py`、シーン一覧 `scripts/show_scenes_towel.py`、到達範囲 `tools/towel_reach.py`、エキスパートの簡易評価 `scripts/eval_expert_towel.py [--mesh 16] [--dart 0.0065]`、評価 `scripts/eval_towel.py --policy expert [--disturb none|shift|slip|drop|nudge|suite] [--mesh 10]`（レポートは `runs/eval_summaries/towel_*`）、粗いメッシュの比較 `tools/towel_mesh_compare.py --n 10`、ブラウザ `scripts/export_web_towel.py` + `scripts/export_towel_records.py` → `npm --prefix web run build` → `tools/browser_check_towel.py [--gpu]`、ページは `web/towel.html`、学習 `scripts/collect_towel.py --name <名前> --scenes A B --label-every 6 --workers 6 [--disturb mix|slip,drop,nudge] [--mesh 10] [--policy keigrasp:<run>]`（`--disturb` のカンマ区切りはシーンごとに順に回す）→ `scripts/train_towel.py --name <run> --data <名前…> [--init <run>]` → `scripts/eval_towel.py --policy keigrasp:<run> --workers 6 --disturb suite [--mesh 10]`、対照は `--policy keigrasp-blind:<run>[:カメラ]`。**使用済みの訓練シーン（M3）: 0–1799**。現行の最良は `runs/train/towel_r2/model.pt`（git 管理外）
- モデル調査: `uv run python tools/inspect_so101.py` / IK 確認: `uv run python tools/check_ik.py`

## 原則

- **重いエンジンを使わない。** Isaac 系は範囲外。物理は MuJoCo のみ、接触と摩擦は簡略化しない
- **シードで全てが決まる。** `(scene_seed, episode_seed)` からシーンとエピソードが一意に決まる。乱数は `numpy.random.Generator` をシードから作り、グローバル乱数は使わない
- **モデルは scene_seed だけで決まり、エピソードは初期状態だけを変える。** 皿は mocap ボディ、立方体は自由関節で、どちらも位置は `MjData` 側にある。シーンは `MjSpec` で組み立てる（`keiarm/scene.py`）
- **真下向きの把持はサイト高さ約 9 cm まで。** それより高い搬送は傾きが必要（`docs/so101_model.md`）
- **決定性は Python 側だけに要求する。** ブラウザ（WASM）と Python のビット一致は要求しない
- **エキスパートは状態からフェーズを判定する。** 経過時間で進めない（DAgger が任意の状態からラベルを求めるため）。`Expert.act(data)` は `MjData` を変更せず、内部状態も持たない。これはテストで保証している
- **エキスパートの目標に「垂直方向は今の位置を保つ」を使わない。** 位置制御のアームは沈み続ける。絶対値か、立方体・皿からの相対で与える
- **方策が見るのはカメラ画像 2 枚と関節角 6 個だけ。** 物体姿勢や接触は特権情報で、エキスパートと評価だけが使う
- **訓練データに評価シードを混ぜない。** テストで保証する
- **方策は `keiarm/policies.py` の `Policy` として実装する**（`reset(env)` / `act(env)`）。学習済み方策は `env.observe()` だけを使い、`needs_images = True` にする。モデル間の比較は `scripts/compare.py`（同一シードの符号検定）で行う
- **フェーズは合格基準を満たしてから次へ。** 特に、エキスパートが基準を満たすまで学習に進まない。ブラウザ描画の検証（Phase 4）は学習より前
- **見た目は単純に。** 影・反射を使わず単色中心（MuJoCo 描画と three.js 描画の差を小さくするため）。両者の画素は `色 × (環境光 + 拡散光 × max(n·l, 0))` で一致させてある（`web/src/render.js`）。**Python 側の見た目（`keiarm/scene.py`）を変えたら `tools/browser_check.py` で差を測り直す**
- **`web/src/sim.js` は `keiarm/scene.py` の `reset_data` と `keiarm/task.py` の成功判定を写したもの。** Python 側を変えたら両方直す。定数は書き出し JSON の `constants` で渡す
- **8 GB を超える画像は `runs/cache/` に展開して memmap で読む**（`keigrasp.data.load_dataset`）。キャッシュは消してもよい（次の学習で作り直す）
- **描画つきの `KeiArmEnv` は必ず `close()` する。** 閉じないまま同一プロセスで大量に作ると、エラーなしで画像が壊れ、方策の成功率が黙って下がる
- **良すぎる結果は疑う。** 方策の評価では、画像を黒塗りした対照と、同じ評価の再実行での一致を確かめてから報告する
- **ブラウザ側の方策実行（`web/src/policy.js`）は `keigrasp/policy.py` を写したもの。** 実行ステップ数（10）は書き出し JSON の `policy_execute_steps`。モデルを変えたら `scripts/export_onnx.py` で書き出し直し、`tools/browser_eval.py` で Python の評価と突き合わせる
- **公開先**: リポジトリ https://github.com/southern-star/keiarm（public）、デモ https://southern-star.github.io/keiarm/。`main` への push でデモが更新されるので、push は利用者の確認を取ってから。公開後の確認は `uv run python tools/browser_demo_check.py --url https://southern-star.github.io/keiarm/`
- **OpenArm の重力補償はコンパイル前（XML / MjSpec）に入れる。** WASM ではコンパイル後に `body_gravcomp` を書き換えても効かない
- **WASM のオブジェクトは GC されない。** `MjModel` / `MjData` / 接触ベクタは使い終わったら `.delete()` する
- 方策は 2,000 万パラメータ以下、行動チャンク回帰（L1）。拡散・VLA・強化学習は使わない
- 「要確認」の項目は実物（XML、README、パッケージ）で確かめ、結果を `PROGRESS.md` か `docs/` に残す。推測で設計を進めない
- 名称（KeiArm / KeiGrasp）は仮称。名前を埋め込む箇所を増やさない

## 構成

```
keiarm/      シミュレーター（シーン生成、環境、エキスパート、評価）
keigrasp/    方策（モデル、学習、DAgger、ONNX 書き出し）
web/         ブラウザデモ（three.js、MuJoCo WASM、onnxruntime-web）。web/public/scenes/ に事前生成シーン（git 管理）、web/public/assets/ はメッシュのコピー（管理外）
keiarm/openarm/  マイルストーン 2（OpenArm v2）。SO-101 側のコードとは独立させ、共通化は必要になってから
keiarm/towel/    マイルストーン 3（タオルの二つ折り）。シーン、掴みの規則、成功判定、エキスパート、評価、ブラウザ書き出し
web/src/towel/   タオルのページ（物理は Web Worker、記録の再生）
third_party/ Menagerie 由来の SO-101、Enactic 由来の OpenArm v2（ライセンス同梱、編集禁止）
scripts/     demo.py ほか、利用者が叩く入口
tools/       調査・検証用（モデル調査、IK 確認、到達範囲、描画一致）
tests/ docs/ runs/（runs は git 管理外。`runs/eval_summaries/` だけ管理）
```

## 範囲外（最初のマイルストーン）

実機接続、LeRobot データセット入出力、言語指示、複数タスク、双腕、遠隔操作によるデータ収集。

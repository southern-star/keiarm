# 布スパイク E0: 調査メモ（MuJoCo 3.17.0）

調べた日: 2026-11-13。一次情報は (1) 固定版の Python パッケージ `mujoco==3.17.0` の列挙型と
フィールド、(2) 3.17.0 版の XML リファレンス（https://mujoco.readthedocs.io/en/3.17.0/XMLreference.html）、
(3) 本家リポジトリ 3.17.0 タグの `model/flex/poncho.xml`、(4) 実際にコンパイルして出たエラー。

## バージョンの判断（設計図 2.2）

**上げなくてよい。** 固定版 3.17.0 は設計図が前提にした 3.5.0 より新しく、必要なものが Python・WASM の両方にある。

| 確認したもの | Python 3.17.0 | npm `@mujoco/mujoco` 3.17.0（`web/node_modules`） |
|---|---|---|
| 等式拘束 `mjEQ_FLEXVERT`（`edge equality="vert"`） | あり（`mjtEq`） | 型定義あり |
| `elastic2d="bend"` の `flexcomp`（プラグイン不要） | 読める | 読める（E5） |
| `data.flexvert_xpos` | あり | あり（`mujoco.d.ts`） |
| `model.eq_obj2id` / `eq_data` / `data.eq_active` の書き換え | 効く（E2、E3） | 効く（E5） |
| 自己衝突の方式 `mjtFlexSelf` | NONE / NARROW / BVH / SAP / AUTO | — |

`poncho.xml` と同じ flexcomp の設定（下記）を、マネキンの代わりに球にかぶせて動かした:
`out/e0_poncho_like.mp4`、`out/e0_env.json`。破綻なし。576 頂点・CG・2 ms で実時間の 0.15 倍。

poncho.xml（3.17.0）の値: `<option integrator="discrete" solver="CG" tolerance="1e-6">`、
`radius="0.01" mass="1"`、`<edge equality="vert" damping="0.1"/>`、
`<elasticity young="3e5" poisson="0" thickness="8e-3" elastic2d="bend" damping="0.02"/>`、`<contact solref="0.003"/>`。
本体は `mannequin.xml` の include と頂点・三角形の列（数百行）を含むので取り込んでいない。

## flexcomp / flex の属性（3.17.0 のリファレンスより、要点）

### flexcomp 本体
- `dof`（既定 `full`）: `full`（頂点ごとに並進 3 自由度）/ `radial` / `trilinear`（全体で 24 自由度）/ `quadratic`（81 自由度）/ `2d`
- `mass`（既定 1）: 自動生成されるボディに点の数で等分
- `rigid`（既定 false）: true なら頂点が全て親ボディに付き、ボディは作られない
- `radius`（flex の既定 0.005）: 衝突と描画の両方に効く。2D では正である必要がある（要素に厚みを持たせる）。層の間隔は 2 × radius

### edge
- `equality`（既定 `false`）: `false` / `true`（辺ごとの長さの拘束）/ `vert`（下記 flexvert、頂点ごとに平均した拘束）/ `strain`（trilinear と quadratic のみ）
- `stiffness`（既定 0）: **1D の flex にだけ効く**。2D・3D では使えない
- `damping`（既定 0）: 全ての辺に効く減衰
- `solref` / `solimp`: 生成される等式拘束に渡される

### elasticity
- `young`（既定 0）、`poisson`（既定 0、[0, 0.5)）、`damping`（既定 0、レイリー減衰の係数で単位は時間）、
  `thickness`（既定 -1、2D の膜の厚み。2 × radius にしてもよい）
- `elastic2d`（既定 `none`）: `none` / `bend` / `stretch` / `both`。2D の弾性力のどれを入れるか

### contact
- `selfcollide`（既定 `auto`）: `none` / `narrow`（全組み合わせ。診断用）/ `bvh` / `sap` / `auto`（**1D・2D は sap、3D は bvh**）
- `activelayers`（既定 1）: **3D の flex 専用**。四面体を外表面からのグラフ距離で層に分け、衝突させる層の数を決める。2D の布には効かない（E1 で 0 にしても接触数は同じ）
- `passive`（既定 false、実験的）: 接触を拘束でなく受動的な垂直力にする。摩擦なし、すり抜けを保証しない。discrete 積分器、CG/Newton、noslip なしが条件
- `condim` / `friction` / `margin` / `gap` / `solref` / `solimp`: geom と同じ。**相手の geom の condim が大きければそちらが使われる**（E1: 布だけ condim=1 にしても机が 3 なら 3）

### 等式拘束 flexvert（3.5.0 で追加）
ひずみテンソルのトレースと行列式を単位行列のものに保つ（Chen, Kry, Vouga 2019）。三角形ごとのひずみを、
頂点に接する三角形で平均する。拘束の数が 2T から 2V に減り、ロッキングを避ける。2D の flex（布）専用。

## 実際に動かして分かった制約（リファレンスに書かれていない、または読み落としやすいもの）

1. **flex の elasticity を使うと `integrator="discrete"` が必須。** implicit / implicitfast ではコンパイル時に
   `flex elasticity is no longer integrated implicitly under integrator='implicit' and 'implicitfast': set integrator='discrete'`。
   OpenArm の既定は implicitfast なので、布と同居させるならロボットも discrete で動かす（E3 の前提確認で、決め打ちの把持は
   implicitfast と同じ結果になった: 持ち上げ 147 mm、追従誤差 9〜12 mm、`out/e3_integrator.json`）
2. **edge の等式拘束（`true` / `vert`）と `young` は、`elastic2d="bend"` のときだけ併用できる。** `stretch` / `both` と
   組み合わせると `flex constraints and elasticity (young) cannot both be present`。`none` で `<elasticity young=…>` を書いても同じエラー
3. **`vertcollide`、`internal` は 3.17.0 のスキーマにない**（`unrecognized attribute`）。新しい版の資料にある属性を使わない
4. **2D の grid で `dof="trilinear"` / `"quadratic"` はコンパイルできない**（`flex grid rotation R0 is not orthonormal`、`cellcount` を変えても同じ）。
   設計図 2.1 の「粗い制御格子 + 複数セル」は 2D の布では試せなかった
5. **`selfcollide="narrow"` は N = 24 以上でスタックが足りなくなる**（`mj_stackAlloc: out of memory`）。診断用の扱い
6. **机を箱にすると、接触が三角形ごとに約 3 点できる**（N = 16 の布を置いただけで 1350 点）。平面なら頂点ごとに 1 点（256 点）で、
   布単体の物理が 2.5 倍速くなった。E3 では布専用の衝突平面を机の上面に置き、布と机の箱は当てない（衝突のビットで分ける）
7. 自動生成されるボディの名前は `<flexcomp の name>_<k>`、頂点番号は grid の順 `k = i * N + j`（i が x、j が y）。
   各ボディに slide 関節が 3 本。`nbody = 1 + N²`
8. 描画: 2D の flex は `material` のテクスチャを貼れる（grid は texcoord を自動で作る）。**表と裏は同じ材質で、裏だけ別の色にする指定はない**
   （回避策は E4）。セグメンテーション描画では布の画素が `mjOBJ_FLEX` になる
9. `MjSpec` から `to_xml()` で書き出した XML は `<compiler angle="radian">` になる。そこへ `flexcomp` を文字列で足すときは
   `euler` をラジアンで書く（度で書いて布が 270° 回った）。メッシュの `meshdir` は相対パスのままなので絶対パスにしてから書き出す
10. 布は `flexcomp` の XML 文字列で差し込んでからコンパイルした（MjSpec には `spec.add_flex` と `body.make_flex` があるが、
    設計図の属性名をそのまま使える XML を優先した。本実装で MjSpec に寄せるなら `make_flex` を調べる）
11. 拘束なし + `elastic2d="both"`（young 2e5）では、掴んだ瞬間に
    `WARNING: Flex stiffness is too ill-conditioned for the effective-metric block preconditioner ...` が出る（破綻はしない）
12. `discrete` 積分器 + flex では PGS ソルバーと noslip が使えない（`mj_checkDiscrete: discrete integrator: PGS and noslip not yet supported with flex`）
13. 層が重なると CG が既定の 50 反復で収束しなくなる。反復を 10 に減らすと速いが、層がすり抜ける（REPORT.md 2.2）

"""E1: 布だけ（ロボットなし）。机の上に布を落とし、一部を箱にかけて静止させる。

条件を 1 つずつ変えて、破綻、実時間比（物理のみ）、接触数・拘束数、静止後の伸び・残留速度・
弾みを測る。結果は out/e1_results.json と out/e1_table.md。

実行: uv run python spikes/cloth/e1_drop.py [--quick]
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cloth import (
    ClothCfg,
    StepTimer,
    broken,
    cloth_verts,
    edge_strain,
    strain_stats,
    table_scene_xml,
    vid,
)

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# 布の -x 側の 1/3 ほどを高さ 4 cm の箱にかける（曲げと、角での接触を見る）
BOX = '<geom name="box" type="box" size="0.105 0.2 0.02" pos="-0.145 0 0.02" rgba="0.4 0.5 0.45 1"/>'
SIM_T = 2.0


def run(cfg: ClothCfg, sim_t: float = SIM_T, extra=BOX, drop_z=0.045) -> dict:
    xml = table_scene_xml(cfg, cloth_pos=(0.0, 0.0, drop_z), extra_world=extra)
    try:
        model = mujoco.MjModel.from_xml_string(xml)
    except Exception as e:  # noqa: BLE001
        return {"status": f"compile_error: {e}"}
    try:
        return _run(cfg, model, sim_t)
    except Exception as e:  # noqa: BLE001  （N=32 + narrow でスタック不足になる）
        return {"status": f"error: {str(e).splitlines()[0][:80]}"}


def _run(cfg, model, sim_t):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    timer = StepTimer()
    steps = round(sim_t / cfg.timestep)
    status = None
    com_z, prev = [], cloth_verts(model, data).copy()
    vmax_end = 0.0
    for k in range(steps):
        timer.step(model, data)
        v = cloth_verts(model, data)
        status = broken(model, data, v, cfg.radius)
        if status:
            break
        com_z.append(float(v[:, 2].mean()))
        if k >= steps - 24:
            vmax_end = max(vmax_end, float(np.linalg.norm(v - prev, axis=1).max() / cfg.timestep))
        prev = v.copy()
    out = {"status": status or "ok", "rtf": round(timer.rtf, 3), "steps": len(timer.ncon),
           "ncon_mean": round(float(np.mean(timer.ncon)), 1), "ncon_max": int(np.max(timer.ncon)),
           "nefc_mean": round(float(np.mean(timer.nefc)), 1), "nv": model.nv, "neq": model.neq}
    if status is None:
        v = cloth_verts(model, data)
        cz = np.array(com_z)
        # 弾み: 重心が最初に最下点に達した後、どれだけ上がり戻ったか
        k0 = int(np.argmin(cz[: int(0.6 / cfg.timestep)]))
        rebound = float(cz[k0:].max() - cz[k0])
        if v[:, 2].min() < -cfg.radius:
            out["status"] = "through_table_end"
        out.update(strain_max=round(edge_strain(model, data), 4), vmax_end=round(vmax_end, 4),
                   rebound_mm=round(rebound * 1000, 2), zmin_mm=round(float(v[:, 2].min()) * 1000, 2),
                   strain_mean=round(strain_stats(model, data)[0], 4), strain_p95=round(strain_stats(model, data)[1], 4),
                   # 箱の縁（x = -0.04）から垂れた部分の中央の列（x = 0）の高さ。曲げ剛性が大きいほど高い
                   mid_z_mm=round(float(v[vid(cfg, cfg.n // 2, 0) : vid(cfg, cfg.n // 2, cfg.n - 1) + 1, 2].mean()) * 1000, 1),
                   far_z_mm=round(float(v[-cfg.n:, 2].mean()) * 1000, 1))
    return out


def variants(quick: bool):
    base = ClothCfg()
    rows = []
    ns = [10, 16] if quick else [10, 16, 24, 32]
    # 1. 解像度 × 伸びの拘束方式
    for n in ns:
        for eq in ("true", "vert"):  # 基準は true
            rows.append((f"N={n} eq={eq}", base.with_(n=n, equality=eq)))
        # 拘束なし: 伸びは elasticity の stretch で持つ（edge 拘束と stretch / both は併用できない）
        rows.append((f"N={n} eq=false both", base.with_(n=n, equality="false", elastic2d="both", young=5e4)))
    # 2. 曲げ
    for n in ([16] if quick else [16, 24]):
        rows.append((f"N={n} elastic2d=none", base.with_(n=n, elastic2d="none")))
        rows.append((f"N={n} eq=false both young=2e5", base.with_(n=n, equality="false", elastic2d="both", young=2e5)))
    # 3. 弾性パラメータ（N=16）
    for y in (1e3, 2e4, 1e5):
        rows.append((f"N=16 young={y:g}", base.with_(young=y)))
    for th in (5e-4, 3e-3):
        rows.append((f"N=16 thickness={th:g}", base.with_(thickness=th)))
    rows.append(("N=16 poisson=0.3", base.with_(poisson=0.3)))
    rows.append(("N=16 damping=0.01", base.with_(damping=0.01)))
    rows.append(("N=16 edge_damping=0.05", base.with_(edge_damping=0.05)))
    # 4. 半径（層の厚み）
    for r in (0.001, 0.004):
        rows.append((f"N=16 radius={r:g}", base.with_(radius=r)))
    # 5. 自己衝突
    for sc in ("none", "narrow", "bvh", "sap"):
        for n in ([16] if quick else [16, 24, 32]):
            rows.append((f"N={n} selfcollide={sc}", base.with_(n=n, selfcollide=sc)))
    # 6. 時間刻み・ソルバー・反復
    for n in ([16] if quick else [16, 24]):
        rows.append((f"N={n} dt=1/480", base.with_(n=n, timestep=1 / 480)))
        rows.append((f"N={n} dt=1/120", base.with_(n=n, timestep=1 / 120)))
        rows.append((f"N={n} Newton", base.with_(n=n, solver="Newton")))
        rows.append((f"N={n} iter=10", base.with_(n=n, iterations=10)))
        rows.append((f"N={n} cone=elliptic", base.with_(n=n, cone="elliptic")))
        rows.append((f"N={n} condim=1", base.with_(n=n, condim=1)))
    return rows


TRILINEAR = """<mujoco>
  <option timestep="{dt}" solver="Newton" integrator="discrete"/>
  <worldbody>
    <geom type="box" size="0.4 0.4 0.02" pos="0 0 -0.02"/>
    <geom type="box" size="0.105 0.2 0.02" pos="-0.145 0 0.02"/>
    <flexcomp name="cloth" type="grid" count="16 16 1" spacing="0.016 0.016 0.016" dim="2" radius="0.002"
              mass="0.02" pos="0 0 0.06" dof="{dof}" cellcount="{cc}">
      <edge equality="false"/>
      <elasticity young="5e3" thickness="1e-3" elastic2d="bend"/>
    </flexcomp>
  </worldbody>
</mujoco>"""


def trilinear_check() -> list[dict]:
    """dof=trilinear / quadratic（+ 複数セル）で、箱の角に沿って折れ曲がるかだけ見る。"""
    res = []
    for dof, cc in (("trilinear", "1 1 1"), ("trilinear", "4 4 1"), ("quadratic", "2 2 1")):
        row = {"dof": dof, "cellcount": cc}
        try:
            m = mujoco.MjModel.from_xml_string(TRILINEAR.format(dt=1 / 240, dof=dof, cc=cc))
        except Exception as e:  # noqa: BLE001
            row["status"] = f"compile_error: {e}"
            res.append(row)
            continue
        d = mujoco.MjData(m)
        t = StepTimer()
        for _ in range(int(SIM_T * 240)):
            t.step(m, d)
        v = d.flexvert_xpos
        row.update(status="ok" if np.isfinite(v).all() else "nan", nv=m.nv, rtf=round(t.rtf, 2),
                   mid_z_mm=round(float(v[8 * 16 : 9 * 16, 2].mean()) * 1000, 1),
                   zmax_mm=round(float(v[:, 2].max()) * 1000, 1), zmin_mm=round(float(v[:, 2].min()) * 1000, 1))
        res.append(row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    results = []
    for name, cfg in variants(args.quick):
        r = run(cfg)
        r["name"] = name
        results.append(r)
        print(name, r, flush=True)
    tri = trilinear_check()
    for r in tri:
        print("trilinear", r, flush=True)
    (OUT / "e1_results.json").write_text(json.dumps({"rows": results, "trilinear": tri}, indent=1, ensure_ascii=False))
    cols = ["name", "status", "rtf", "nv", "neq", "ncon_mean", "nefc_mean", "strain_max", "rebound_mm",
            "strain_mean", "strain_p95", "vmax_end", "zmin_mm", "mid_z_mm", "far_z_mm"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in results:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (OUT / "e1_table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

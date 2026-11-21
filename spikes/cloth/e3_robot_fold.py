"""E3: OpenArm v2（両腕）+ 布。決め打ちの軌道で二つ折りにする（方策ではない）。

- 既存の双腕シーン（keiarm/openarm/scene.py の build_spec）から棒と皿を除き、布を置く。既存コードは変更しない
- 布の大きさ 20 cm: 右手が y = -0.10、左手が y = +0.10 の辺で、x = 0.125〜0.325 を往復する
  （docs/openarm/reach_cube.json の把持高さで両手とも届く範囲）
- 物理: integrator="discrete"（flex の曲げに必須）、ソルバーは CG（E1）
- 衝突のビット: 布と机は「布専用の平面」（机の上面）で当て、机の箱とは当てない（箱だと接触が
  三角形ごとに約 3 点になる）。布はアームの全ジオメトリと当たる
- 掴み（設計図 3.2）: グリッパを閉じた瞬間に、指先の中点から GRAB_RADIUS 以内の頂点（近い順に
  最大 N_SLOTS 個）を手のボディ（ee_base_link）に connect 拘束でつなぐ。拘束スロットは事前に
  非アクティブで定義し、相手ボディ（eq_obj2id）と anchor（eq_data）を実行時に書き換える
- --friction: 拘束を使わず、指の摩擦だけで挟む比較（1 回だけ）

実行:
  MUJOCO_GL=egl uv run python spikes/cloth/e3_robot_fold.py --n 16 --video
  uv run python spikes/cloth/e3_robot_fold.py --n 16 --trials 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cloth import (
    N_SLOTS,
    ClothCfg,
    assets_xml,
    broken,
    cloth_verts,
    convex_area,
    fold_layers_half,
    layer_violation,
    strain_stats,
    vid,
)

from keiarm import config
from keiarm.openarm.ik import ArmIK, topdown_rotation
from keiarm.openarm.robot import (
    ARM_DOF,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_ARM_QPOS,
    HOME_YAW,
    SIDES,
)
from keiarm.openarm.scene import build_spec, sample_episode, sample_scene
from keiarm.paths import OPENARM_XML

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

CLOTH_SIDE = 0.20
CLOTH_CENTER = (0.21, 0.0)
# 顎の向き: 自然な向き（HOME_YAW）のままだと x < 0.2 で IK が解けない。+60°（左は鏡像で -60°）なら
# y = ±0.10 の x = 0.125〜0.30 に真下向きで届く（拘束で掴むので顎の向きは布に関係しない）
YAW_OFFSET = {"right": np.radians(60.0), "left": -np.radians(60.0)}
GRAB_RADIUS = 0.02  # 指先の中点からこの距離以内の頂点を掴む
FINGERTIP_OFFSET = np.array([0.0, 0.0, -0.165])  # ee_site 座標での指先の中点（指先の端は -0.17）
MIN_PINCH_Z = 0.024  # keiarm/openarm/scripted.py と同じ（指先が机の 4 mm 上）
CARROT = 0.006  # 1 制御周期あたりの目標の移動量上限 [m]（0.18 m/s）


def build(cfg: ClothCfg, scene_seed: int, off, yaw_deg: float, friction_only: bool):
    scene = sample_scene(scene_seed)
    spec = build_spec(scene)
    spec.delete(spec.body("bar"))
    spec.delete(spec.body("plate"))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
    spec.option.solver = getattr(mujoco.mjtSolver, f"mjSOL_{cfg.solver.upper()}")
    spec.option.iterations = cfg.iterations
    # 机の箱はアームとだけ当てる（contype 4 / conaffinity 1）。布は contype 2 / conaffinity 3
    table = spec.geom("table")
    table.contype, table.conaffinity = 4, 1
    spec.worldbody.add_geom(name="cloth_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.5, 0.6, 0.01],
                            rgba=[0, 0, 0, 0], contype=2, conaffinity=2)
    spec.worldbody.add_camera(name="side", pos=[0.225, -0.75, 0.28], xyaxes=[1, 0, 0, 0, 0.35, 1], fovy=40)
    spec.worldbody.add_camera(name="top", pos=[0.225, 0.0, 0.75], xyaxes=[0, 1, 0, -1, 0, 0], fovy=40)
    spec.meshdir = str(OPENARM_XML.parent / spec.meshdir)  # XML に書き出すと相対パスが解決できないので絶対パスに
    xml = spec.to_xml()  # <compiler angle="radian"> なので euler はラジアンで書く
    s = CLOTH_SIDE / (cfg.n - 1)
    fric = 'friction="1.0 0.005 0.0001"' if friction_only else f'friction="{cfg.friction} 0.005 0.0001"'
    cloth = f"""
    <flexcomp name="cloth" type="grid" count="{cfg.n} {cfg.n} 1" spacing="{s:.8g} {s:.8g} {s:.8g}" dim="2"
              radius="{cfg.radius:g}" mass="{cfg.mass:g}" material="cloth_mat"
              pos="{CLOTH_CENTER[0] + off[0]:g} {CLOTH_CENTER[1] + off[1]:g} {cfg.radius + 0.001:g}" euler="0 0 {np.radians(yaw_deg):.6g}">
      <edge equality="{cfg.equality}" damping="{cfg.edge_damping}"/>
      <elasticity young="{cfg.young:g}" poisson="{cfg.poisson:g}" thickness="{cfg.thickness:g}" damping="{cfg.damping:g}" elastic2d="{cfg.elastic2d}"/>
      <contact selfcollide="{cfg.selfcollide}" contype="2" conaffinity="3" condim="{cfg.condim}" {fric}
               solref="{cfg.solref[0]} {cfg.solref[1]}"/>
    </flexcomp>"""
    eqs = "\n".join(
        f'<connect name="{side}_slot{k}" body1="openarm_{side}_ee_base_link" body2="cloth_0" anchor="0 0 0" '
        f'active="false" solref="0.01 1"/>' for side in SIDES for k in range(N_SLOTS))
    # MjSpec の XML に布（flexcomp はマクロなので XML で足す）、テクスチャ、拘束スロットを差し込む
    xml = xml.replace("</worldbody>", cloth + "\n</worldbody>", 1)
    xml = xml.replace("<asset>", "<asset>" + assets_xml(cfg), 1)
    xml = xml.replace("</mujoco>", f"<equality>{eqs}</equality>\n</mujoco>", 1) if "<equality>" not in xml else \
        xml.replace("<equality>", "<equality>" + eqs, 1)
    model = mujoco.MjModel.from_xml_string(xml)
    return scene, model


class HandGrabber:
    """手のボディと布の頂点を connect でつなぐ（スロットの相手を実行時に差し替える）。"""

    def __init__(self, model, side: str):
        self.m = model
        self.body = model.body(f"openarm_{side}_ee_base_link").id
        self.slots = [model.equality(f"{side}_slot{k}").id for k in range(N_SLOTS)]
        self.held: list[int] = []

    def grab(self, data, verts: list[int]):
        R = data.xmat[self.body].reshape(3, 3)
        for slot, v in zip(self.slots, verts):
            b = self.m.body(f"cloth_{v}").id
            self.m.eq_obj2id[slot] = b
            self.m.eq_data[slot, 0:3] = R.T @ (data.xpos[b] - data.xpos[self.body])
            self.m.eq_data[slot, 3:6] = 0.0
            data.eq_active[slot] = 1
        self.held = list(verts)

    def release(self, data):
        for slot in self.slots:
            data.eq_active[slot] = 0
        self.held = []

    def anchor_error(self, data) -> float:
        """掴んでいる頂点と anchor の距離の最大値（拘束がちゃんと効いているか）。"""
        if not self.held:
            return 0.0
        R = data.xmat[self.body].reshape(3, 3)
        err = 0.0
        for slot, v in zip(self.slots, self.held):
            a = data.xpos[self.body] + R @ self.m.eq_data[slot, 0:3]
            err = max(err, float(np.linalg.norm(a - data.xpos[self.m.body(f"cloth_{v}").id])))
        return err


def arc(p0, p1, s, lift):
    u = 0.5 - 0.5 * np.cos(np.pi * s)
    p = p0 + u * (p1 - p0)
    p[2] += lift * np.sin(np.pi * u)
    return p


def trial(args) -> dict:
    cfg_d, seed, video, friction_only = args
    cfg = ClothCfg(**cfg_d)
    rng = np.random.default_rng(np.random.SeedSequence([0xC1E3, seed]))
    off = rng.uniform(-0.01, 0.01, 2)
    yaw = rng.uniform(-5.0, 5.0)
    lift_ratio = rng.uniform(0.45, 0.7)
    scene, model = build(cfg, 100000 + seed, off, yaw, friction_only)
    data = mujoco.MjData(model)
    ep = sample_episode(scene, 0)
    # reset_data は棒と皿を前提にしているので、腕の部分だけ写す
    _reset_arms(model, data, ep)
    n, dt = cfg.n, model.opt.timestep
    iks = {s: ArmIK(model, s) for s in SIDES}
    acts = {s: np.array([model.actuator(f"{s}_joint{i}_ctrl").id for i in range(1, ARM_DOF + 1)]) for s in SIDES}
    grip = {s: model.actuator(f"{s}_finger1_ctrl").id for s in SIDES}
    grabbers = {s: HandGrabber(model, s) for s in SIDES}
    q_cmd = {s: data.qpos[iks[s].qadr].copy() for s in SIDES}
    carrot = {s: iks[s].pinch_pose(data)[0].copy() for s in SIDES}
    yaw_cmd = {s: HOME_YAW[s] + YAW_OFFSET[s] for s in SIDES}
    res = {"seed": seed, "n": n, "friction_only": friction_only, "status": "ok", "off": off.round(4).tolist(),
           "yaw_deg": round(yaw, 2), "lift_ratio": round(lift_ratio, 3)}
    t_phys = t_all = sim = 0.0
    arm_cloth_steps = 0
    max_anchor = 0.0
    max_track = 0.0
    frames = {"side": [], "top": [], "cam_front": []} if video else None
    rend = mujoco.Renderer(model, 360, 480) if video else None
    body_is_arm = np.array([model.body(model.geom_bodyid[g]).name.startswith("openarm_") for g in range(model.ngeom)])

    def control_step(targets, gripper) -> bool:
        nonlocal t_phys, t_all, sim, arm_cloth_steps, max_anchor, max_track
        t0 = time.perf_counter()
        for s in SIDES:
            if s in targets:
                d = targets[s] - carrot[s]
                dist = np.linalg.norm(d)
                carrot[s] = carrot[s] + (d if dist < CARROT else d / dist * CARROT)
            q_cmd[s], _, _ = iks[s].solve(data.qpos, q_cmd[s], carrot[s], topdown_rotation(yaw_cmd[s]), iters=20,
                                          q_rest=np.array(HOME_ARM_QPOS[s]))
            data.ctrl[acts[s]] = q_cmd[s]
            data.ctrl[grip[s]] = gripper[s]
        t1 = time.perf_counter()
        mujoco.mj_step(model, data, config.SUBSTEPS)
        t2 = time.perf_counter()
        t_phys += t2 - t1
        sim += config.SUBSTEPS * dt
        # アームのジオメトリと布の接触（flex との接触は geom1 / geom2 の片方が -1）
        touch = any((c.geom[0] >= 0 and body_is_arm[c.geom[0]]) or (c.geom[1] >= 0 and body_is_arm[c.geom[1]])
                    for c in data.contact[: data.ncon] if c.flex[0] >= 0 or c.flex[1] >= 0)
        arm_cloth_steps += int(touch)
        for s in SIDES:
            max_anchor = max(max_anchor, grabbers[s].anchor_error(data))
            max_track = max(max_track, float(np.linalg.norm(iks[s].pinch_pose(data)[0] - carrot[s])))
        if video:
            for v in frames:
                rend.update_scene(data, v)
                frames[v].append(rend.render())
        t_all += time.perf_counter() - t0
        b = broken(model, data, cloth_verts(model, data), cfg.radius, bound=2.0)
        if b:
            res["status"] = b
            return False
        return True

    def go(targets, gripper, steps) -> bool:
        return all(control_step(targets, gripper) for _ in range(steps))

    def fingertip(s):
        rot = data.site_xmat[iks[s].site].reshape(3, 3)
        return data.site_xpos[iks[s].site] + rot @ FINGERTIP_OFFSET

    opened = {s: GRIPPER_OPEN[s] for s in SIDES}
    closed = {s: GRIPPER_CLOSED[s] for s in SIDES}
    try:
        # 0. 布を落ち着かせる（腕は開始姿勢）
        if not go({}, opened, 30):  # 顎の向きも開始姿勢から回す
            return _finish(res, t_phys, t_all, sim, arm_cloth_steps, max_anchor, max_track, frames, rend, seed)
        v = cloth_verts(model, data)
        area0 = convex_area(v)
        far = {"right": vid(cfg, n - 1, 0), "left": vid(cfg, n - 1, n - 1)}
        near = {"right": vid(cfg, 0, 0), "left": vid(cfg, 0, n - 1)}
        # 角の少し内側（指が布の上に乗るように 1 cm 内側）を掴む
        center = v.mean(axis=0)
        grasp_xy = {s: v[far[s], :2] + 0.01 * (center[:2] - v[far[s], :2]) / np.linalg.norm(center[:2] - v[far[s], :2])
                    for s in SIDES}
        above = {s: np.array([*grasp_xy[s], 0.12]) for s in SIDES}
        down = {s: np.array([*grasp_xy[s], MIN_PINCH_Z - (0.006 if friction_only else 0.0)]) for s in SIDES}
        ok = go(above, opened, 60) and go(down, opened, 40)
        if ok:
            ok = go(down, closed, 12)  # 閉じる
        if ok and not friction_only:
            for s in SIDES:
                tip = fingertip(s)
                vv = cloth_verts(model, data)
                d = np.linalg.norm(vv - tip, axis=1)
                cand = [int(i) for i in np.argsort(d)[:N_SLOTS] if d[i] < GRAB_RADIUS]
                res[f"{s}_grabbed"] = len(cand)
                res[f"{s}_tip_to_corner_mm"] = round(float(np.linalg.norm(tip - vv[far[s]])) * 1000, 1)
                res[f"{s}_tip"] = tip.round(3).tolist()
                res[f"{s}_target"] = down[s].round(3).tolist()
                grabbers[s].grab(data, cand)
        if ok:
            ok = go(down, closed, 8)
        # 運ぶ: 弧で近い側の角の上へ
        if ok:
            vv = cloth_verts(model, data)
            p0 = {s: carrot[s].copy() for s in SIDES}
            p1 = {s: np.array([*(vv[near[s], :2] + (p0[s][:2] - vv[far[s], :2])), MIN_PINCH_Z + 0.008]) for s in SIDES}
            steps = 75  # 2.5 秒
            for k in range(1, steps + 1):
                tg = {s: arc(p0[s], p1[s], k / steps, lift_ratio * np.linalg.norm((p1[s] - p0[s])[:2]) / 2)
                      for s in SIDES}
                for s in SIDES:
                    carrot[s] = tg[s].copy()  # 弧は直接たどる（CARROT の制限なし）
                if not control_step(tg, closed):
                    ok = False
                    break
            ok = ok and go(p1, closed, 10)
        if ok:
            res["carried_corner_dist_mm"] = round(float(max(
                np.linalg.norm(cloth_verts(model, data)[far[s]] - cloth_verts(model, data)[near[s]]) for s in SIDES)) * 1000, 1)
            for s in SIDES:
                grabbers[s].release(data)
            up = {s: carrot[s] + [0, 0, 0.10] for s in SIDES}
            ok = go(p1, opened, 10) and go(up, opened, 30) and go({s: carrot[s] for s in SIDES}, opened, 30)
        if ok:
            vv = cloth_verts(model, data)
            res["viol0"] = round(layer_violation(vv, fold_layers_half(cfg), 0.0), 4)
            res["viol_r"] = round(layer_violation(vv, fold_layers_half(cfg), cfg.radius), 4)
            res["area_ratio"] = round(convex_area(vv) / area0, 4)
            res["corner_mm"] = round(max(float(np.linalg.norm(vv[far[s]] - vv[near[s]])) for s in SIDES) * 1000, 1)
            res["strain_p95"] = round(strain_stats(model, data)[1], 4)
    except Exception as e:  # noqa: BLE001
        res["status"] = f"exception: {e}"
    return _finish(res, t_phys, t_all, sim, arm_cloth_steps, max_anchor, max_track, frames, rend, seed)


def _reset_arms(model, data, ep):
    from keiarm.openarm.robot import arm_joints, finger_joints

    mujoco.mj_resetData(model, data)
    for side in SIDES:
        for name, q in zip(arm_joints(side), HOME_ARM_QPOS[side]):
            act = name.removeprefix("openarm_") + "_ctrl"
            data.qpos[model.joint(name).qposadr[0]] = q
            data.ctrl[model.actuator(act).id] = q
        for name in finger_joints(side):
            data.qpos[model.joint(name).qposadr[0]] = GRIPPER_OPEN[side]
        data.ctrl[model.actuator(f"{side}_finger1_ctrl").id] = GRIPPER_OPEN[side]
    mujoco.mj_forward(model, data)


def _finish(res, t_phys, t_all, sim, arm_cloth_steps, max_anchor, max_track, frames, rend, seed):
    res.update(sim_s=round(sim, 2), rtf_physics=round(sim / max(t_phys, 1e-9), 3),
               rtf_with_ik=round(sim / max(t_all, 1e-9), 3) if not frames else None,
               arm_cloth_contact_steps=arm_cloth_steps, max_anchor_err_mm=round(max_anchor * 1000, 2),
               max_track_mm=round(max_track * 1000, 1))
    if frames:
        import imageio.v2 as imageio

        tag = "friction" if res["friction_only"] else "weld"
        for v, fr in frames.items():
            imageio.mimwrite(OUT / f"e3_{tag}_n{res['n']}_seed{seed}_{v}.mp4", fr, fps=30, macro_block_size=1)
        rend.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--trials", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--friction", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[], help="ClothCfg の上書き 例: equality=false iterations=10")
    args = ap.parse_args()
    kw = {"n": args.n, "side": CLOTH_SIDE}
    for item in args.set:
        k, v = item.split("=")
        t = ClothCfg.__dataclass_fields__[k].type
        kw[k] = int(v) if t == "int" else float(v) if t == "float" else v
    cfg = ClothCfg(**kw)
    if not args.trials:
        r = trial((cfg.to_dict(), args.seed, args.video, args.friction))
        print(json.dumps(r, indent=1))
        return
    jobs = [(cfg.to_dict(), s, False, args.friction) for s in range(args.seed, args.seed + args.trials)]
    with Pool(args.workers) as pool:
        rows = sorted(pool.map(trial, jobs), key=lambda r: r["seed"])
    for r in rows:
        print(r)
    ok = [r for r in rows if r["status"] == "ok" and "viol0" in r]
    summ = {"trials": len(rows), "completed": len(ok), "status": sorted({r["status"] for r in rows})}
    for k in ("viol0", "viol_r", "area_ratio", "corner_mm", "rtf_physics", "rtf_with_ik", "arm_cloth_contact_steps",
              "max_anchor_err_mm", "max_track_mm", "right_grabbed", "left_grabbed"):
        x = np.array([r[k] for r in ok if r.get(k) is not None], float)
        if len(x):
            summ[k] = {"mean": round(float(x.mean()), 4), "min": round(float(x.min()), 4), "max": round(float(x.max()), 4)}
    print(json.dumps(summ, indent=1))
    (OUT / f"e3_n{args.n}{args.tag}{'_friction' if args.friction else ''}.json").write_text(
        json.dumps({"cfg": cfg.to_dict(), "summary": summ, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()

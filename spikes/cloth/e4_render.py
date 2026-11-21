"""E4: Python 側の描画（mujoco.Renderer）。

1. 布の模様（テクスチャ）と表裏の色。角を持ち上げて裏を見せる
2. 裏の色を変える回避策: mjvScene に三角形（mjGEOM_TRIANGLE）を足して裏面だけ別の色で描く
3. セグメンテーション描画で布の画素が mjOBJ_FLEX として識別できるか
4. OpenArm + 布で、既存のカメラ構成（cam_front + 手首 2 台、160x120）を毎制御周期描画したときの
   描画の時間（物理の時間は E3 / bench.py）

実行: MUJOCO_GL=egl uv run python spikes/cloth/e4_render.py
"""

import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cloth import ClothCfg, Grabber, cloth_verts, flex_triangles, table_scene_xml, vid

OUT = Path(__file__).parent / "out"


def lifted_corner_scene(cfg: ClothCfg):
    model = mujoco.MjModel.from_xml_string(table_scene_xml(cfg, cloth_pos=(0, 0, cfg.radius + 0.001)))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 60)
    g = Grabber(model, "grip_a")
    v = cloth_verts(model, data)
    corner = [vid(cfg, cfg.n - 1, 0), vid(cfg, cfg.n - 1, 1), vid(cfg, cfg.n - 2, 0)]
    data.mocap_pos[g.mocap_id] = v[corner].mean(0)
    g.grab(data, corner)
    p0 = data.mocap_pos[g.mocap_id].copy()
    for k in range(240):  # 1 秒で角を中央の上 8 cm へ
        u = (k + 1) / 240
        data.mocap_pos[g.mocap_id] = p0 + u * (np.array([0.0, 0.0, 0.08]) - p0 * [1, 1, 0] * 0.6)
        mujoco.mj_step(model, data)
    return model, data


def add_backside(scene: mujoco.MjvScene, verts: np.ndarray, tris: np.ndarray, offset: float, rgba) -> int:
    """布の裏面を三角形で足す（法線の逆向きに offset ずらす）。足した数を返す。"""
    added = 0
    for t in tris:
        if scene.ngeom >= scene.maxgeom:
            break
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n) + 1e-12
        # 裏から見て表になるよう、頂点の順を逆にする（a, c, b）
        e1, e2 = c - a, b - a
        g = scene.geoms[scene.ngeom]
        mat = np.column_stack([e1, e2, -n])  # 三角形 (0,0,0), (1,0,0), (0,1,0) を写す（直交でない行列）
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_TRIANGLE, np.array([1.0, 1.0, 1.0]), a - n * offset,
                            mat.ravel(), np.array(rgba, np.float32))
        g.emission = 0.0
        scene.ngeom += 1
        added += 1
    return added


def main():
    info = {}
    cfg = ClothCfg(n=16)
    model, data = lifted_corner_scene(cfg)
    r = mujoco.Renderer(model, 480, 640)
    # 1. そのまま
    imgs = []
    for cam in ("side", "top"):
        r.update_scene(data, cam)
        imgs.append(r.render())
    # 2. 裏面を青で足す
    tris = flex_triangles(model)
    r.update_scene(data, "side")
    added = add_backside(r.scene, cloth_verts(model, data), tris, cfg.radius * 1.05, (0.2, 0.35, 0.85, 1.0))
    imgs.append(r.render())
    info["backside_triangles_added"] = added
    info["scene_maxgeom"] = r.scene.maxgeom
    # 3. セグメンテーション
    r.enable_segmentation_rendering()
    r.update_scene(data, "side")
    seg = r.render()
    r.disable_segmentation_rendering()
    types = seg[..., 1]
    flex_px = int(np.sum(types == int(mujoco.mjtObj.mjOBJ_FLEX)))
    info["segmentation"] = {"flex_pixels": flex_px, "objtypes_seen": sorted({int(x) for x in np.unique(types)}),
                            "mjOBJ_FLEX": int(mujoco.mjtObj.mjOBJ_FLEX), "mjOBJ_GEOM": int(mujoco.mjtObj.mjOBJ_GEOM)}
    segimg = np.zeros((*types.shape, 3), np.uint8)
    segimg[types == int(mujoco.mjtObj.mjOBJ_FLEX)] = (255, 80, 40)
    segimg[types == int(mujoco.mjtObj.mjOBJ_GEOM)] = (90, 90, 90)
    imgs.append(segimg)
    r.close()
    grid = np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0)
    imageio.imwrite(OUT / "e4_render.png", grid[::2, ::2])

    # 4. OpenArm + 布、3 カメラ 160x120
    from e3_robot_fold import build

    from keiarm.openarm.robot import wrist_camera

    _, m2 = build(ClothCfg(n=16, side=0.20), 100000, (0.0, 0.0), 0.0, False)
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    cams = ["cam_front", wrist_camera("left"), wrist_camera("right")]
    rr = mujoco.Renderer(m2, 120, 160)
    for _ in range(5):  # ウォームアップ
        rr.update_scene(d2, cams[0])
        rr.render()
    t = time.perf_counter()
    reps = 60
    frames = []
    for k in range(reps):
        for c in cams:
            rr.update_scene(d2, c)
            img = rr.render()
            if k == 0:
                frames.append(img)
    ms = (time.perf_counter() - t) / reps * 1000
    rr.close()
    imageio.imwrite(OUT / "e4_openarm_cams.png", np.concatenate(frames, 1))
    info["openarm_3cam_160x120_ms_per_control_step"] = round(ms, 2)
    (OUT / "e4_render.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

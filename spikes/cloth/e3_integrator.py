"""E3 の前提確認: OpenArm を integrator="discrete" で動かしても掴めるか、速さはどうか。

3.17.0 では flex の elasticity（曲げ）を使うと integrator="discrete" が必須になる
（implicitfast だとコンパイルエラー）。OpenArm の既定は implicitfast なので、既存の
決め打ち把持（keiarm/openarm/scripted.py）を両方の積分器で走らせて比べる。既存コードは変更しない。

実行: uv run python spikes/cloth/e3_integrator.py
"""

import json
import time
from pathlib import Path

import mujoco

from keiarm import config
from keiarm.openarm.robot import HOME_YAW
from keiarm.openarm.scene import build_spec, grip_points, reset_data, sample_episode, sample_scene
from keiarm.openarm.scripted import scripted_grasp

OUT = Path(__file__).parent / "out"


def run(integrator: str, scene_seed: int) -> dict:
    scene = sample_scene(scene_seed)
    ep = sample_episode(scene, 0)
    spec = build_spec(scene)
    spec.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{integrator.upper()}")
    model = spec.compile()
    data = mujoco.MjData(model)
    reset_data(model, data, scene, ep)
    grip = grip_points(ep.bar_xy, ep.bar_yaw)["right"]
    t = time.perf_counter()
    res = scripted_grasp(model, data, "right", grip, HOME_YAW["right"] + ep.bar_yaw, scene.bar_size[2] / 2)
    wall = time.perf_counter() - t
    steps = 235 * config.SUBSTEPS
    # 腕を止めた状態での物理だけの実時間比
    t = time.perf_counter()
    mujoco.mj_step(model, data, 2400)
    rtf_hold = 2400 * model.opt.timestep / (time.perf_counter() - t)
    return {"integrator": integrator, "scene": scene_seed, "lift_mm": round(res.lift * 1000, 1),
            "max_track_mm": round(res.max_tracking_error * 1000, 1), "table_contact_steps": res.arm_table_contact_steps,
            "rtf_grasp_incl_ik": round(steps * model.opt.timestep / wall, 2), "rtf_physics_hold": round(rtf_hold, 1)}


def main():
    rows = [run(i, s) for s in (100000, 100001, 100002, 100003) for i in ("implicitfast", "discrete")]
    for r in rows:
        print(r)
    (OUT / "e3_integrator.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

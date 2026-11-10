"""Export scenes for the browser demo.

One JSON bundle per scene: the MJCF (as text), the scene parameters, and per episode the
initial state plus the expert's recorded action sequence and outcome. The browser loads the
same MJCF into MuJoCo WASM; meshes are served once from web/public/assets/so101/.
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.expert import Expert
from keiarm.paths import REPO_ROOT, SO101_DIR
from keiarm.robot import EE_SITE, JOINT_NAMES
from keiarm.rollout import SUCCESS_HOLD_STEPS, planned_teleport, run_expert
from keiarm.scene import PLATE_HEIGHT, TABLE_CENTER, TABLE_HALF_SIZE, build_spec
from keiarm.task import REST_SPEED, REST_Z_TOL

WEB_PUBLIC = REPO_ROOT / "web" / "public"
BUNDLE_VERSION = 1
REPLAY_TAIL_STEPS = 30


def mesh_files(model: mujoco.MjModel) -> list[str]:
    """Mesh file names (relative to the MJCF's meshdir) the model needs."""
    spec_meshes = sorted(p.name for p in (SO101_DIR / "assets").glob("*.stl"))
    used = {model.mesh(i).name for i in range(model.nmesh)}
    return [f for f in spec_meshes if Path(f).stem in used]


def scene_bundle(scene_seed: int, episode_seeds: list[int]) -> dict:
    env = KeiArmEnv(scene_seed, render=False)
    episodes = []
    for ep in episode_seeds:
        env.reset(ep)
        teleport = planned_teleport(env, "random")  # the eval suite's disturbance for this episode
        res = run_expert(env, ep)
        # Keep going a little after success (the expert retreats upwards). An open-loop replay
        # in another MuJoCo build diverges by a step or two, so it needs this margin to see the
        # success condition hold long enough.
        expert = Expert(env.model, env.scene)
        tail = []
        for _ in range(REPLAY_TAIL_STEPS):
            action, info = expert.act(env.data)
            env.step(action)
            tail.append(action)
            res.phases.append(info.phase)
        res.actions = np.concatenate([res.actions, np.array(tail)])
        st = env.privileged()
        episodes.append({
            **env.episode.to_dict(),
            "teleport": asdict(teleport),
            "expert": {
                "success": res.success,
                "steps": res.steps,  # control steps until success was confirmed
                "actions": [[float(v) for v in a] for a in res.actions],
                "phases": res.phases,
                "final_cube_pos": [float(v) for v in st.cube_pos],
                "final_joints": [float(v) for v in st.joints],
            },
        })
    return {
        "version": BUNDLE_VERSION,
        "mujoco": mujoco.__version__,
        "scene": env.scene.to_dict(),
        "xml": build_spec(env.scene).to_xml(),
        "mesh_dir": "assets",
        "mesh_files": mesh_files(env.model),
        "episodes": episodes,
        "constants": {
            "control_hz": config.CONTROL_HZ, "substeps": config.SUBSTEPS, "timestep": config.TIMESTEP,
            "image_width": config.IMAGE_WIDTH, "image_height": config.IMAGE_HEIGHT,
            "cameras": list(config.CAMERAS), "joint_names": list(JOINT_NAMES), "ee_site": EE_SITE,
            "plate_height": PLATE_HEIGHT, "rest_z_tol": REST_Z_TOL, "rest_speed": REST_SPEED,
            "success_hold_steps": SUCCESS_HOLD_STEPS, "max_episode_steps": config.MAX_EPISODE_STEPS,
            "table_center": list(TABLE_CENTER), "table_half_size": list(TABLE_HALF_SIZE),
            "policy_execute_steps": 10,
            "fixed_jaw_bodies": ["gripper", "camera_mount"], "moving_jaw_bodies": ["moving_jaw_so101_v1"],
        },
    }


def export_web(scene_seeds: list[int], episode_seeds: list[int], out: Path = WEB_PUBLIC) -> list[Path]:
    scenes = out / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    assets = out / "assets" / "so101"
    assets.mkdir(parents=True, exist_ok=True)
    written = []
    files: set[str] = set()
    for seed in scene_seeds:
        bundle = scene_bundle(seed, episode_seeds)
        files |= set(bundle["mesh_files"])
        path = scenes / f"{seed}.json"
        path.write_text(json.dumps(bundle, separators=(",", ":")))
        written.append(path)
    for f in sorted(files):
        shutil.copyfile(SO101_DIR / "assets" / f, assets / f)
    shutil.copyfile(SO101_DIR / "LICENSE", assets / "LICENSE")
    index = {"version": BUNDLE_VERSION, "scenes": [int(s) for s in scene_seeds],
             "episodes": [int(e) for e in episode_seeds],
             "mesh_bytes": int(np.sum([(assets / f).stat().st_size for f in files]))}
    (scenes / "index.json").write_text(json.dumps(index))
    return written

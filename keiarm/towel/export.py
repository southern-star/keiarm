"""Export towel scenes for the browser (milestone 3, Phase 4).

The towel counterpart of keiarm/openarm/export.py. One JSON bundle per (scene, mesh N): the
MJCF text (meshes relative to "assets", the same files as milestone 2's
web/public/assets/openarm/), the scene parameters, the constants web/src/towel/sim.js needs
(grasp rule, success check, stages), and per episode the initial state, the expert's
recorded actions, an optional disturbance (applied at its recorded step) and the Python
outcome, which the browser's replay must reproduce (tools/replay_towel_node.mjs,
tools/browser_check_towel.py).

Bundles go to web/public/towel_scenes/ (generated, not in git).
"""

import base64
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from keiarm import config
from keiarm.openarm import export as oa_export
from keiarm.openarm.robot import (
    ACTUATOR_NAMES,
    GRIPPER_OPEN,
    PINCH_OFFSET,
    SIDES,
    arm_joints,
    ee_site,
    finger_joints,
)
from keiarm.openarm.scene import TABLE_CENTER, TABLE_HALF_SIZE
from keiarm.paths import REPO_ROOT
from keiarm.towel import grasp, task
from keiarm.towel.env import CAMERAS, MAX_EPISODE_STEPS, TowelEnv
from keiarm.towel.eval import (
    DISTURBANCES,
    NUDGE_ELEVATION,
    OFF_TABLE_Z,
    Disturbance,
    nudge_region,
    planned_disturbance,
    run_episode,
)
from keiarm.towel.expert import Expert
from keiarm.towel.policies import ExpertPolicy
from keiarm.towel.render import BACK_OFFSET
from keiarm.towel.scene import GRAB_SLOTS, TOWEL, TOWEL_N, build_xml

WEB_PUBLIC = REPO_ROOT / "web" / "public"
SCENES_DIR = "towel_scenes"  # under web/public (generated, not in git)
ASSETS_DIR = oa_export.ASSETS_DIR  # the milestone 2 meshes (same robot)
BUNDLE_VERSION = 1
REPLAY_TAIL_STEPS = 30  # keep the expert running after success (as milestone 2)
EXPORT_DISTURBANCES = ("shift", "slip", "drop", "nudge")


def record_episode(env: TowelEnv, episode_seed: int, disturb: str) -> dict:
    env.reset(episode_seed)
    plans = {k: _plan_dict(env, planned_disturbance(env, k)) for k in DISTURBANCES[1:]}
    rec = run_episode(env, ExpertPolicy(), episode_seed, disturb=disturb, probe=False)
    actions = list(rec.actions)
    expert = Expert(env.model, env.scene)
    for _ in range(REPLAY_TAIL_STEPS):
        action, _ = expert.act(env.data)
        action = np.clip(action, *env.ctrl_range.T)
        env.step(action)
        actions.append(action)
    st = task.read_state(env.model, task.TaskIndex(env.model), env.data)
    return {
        **env.episode.to_dict(),
        "disturbance": _plan_dict(env, rec.disturbance),
        "planned_disturbances": plans,
        "expert": {
            "success": rec.success,
            "steps": rec.steps,  # control steps until success was confirmed
            "stage": rec.stage,
            "stage_steps": rec.stage_steps,
            "grabs": rec.grabs,
            "actions": [[round(float(v), 7) for v in a] for a in actions],
            "final_verts": [[round(float(v), 6) for v in p] for p in st.verts],
            "final_joints": [float(v) for v in env.joints()],
        },
    }


def _plan_dict(env: TowelEnv, d: Disturbance | None) -> dict | None:
    """The disturbance as JSON. A nudge also carries its vertices and their velocities
    (Disturbance.apply), so the browser only adds them."""
    if d is None:
        return None
    out = asdict(d)
    if d.kind == "nudge":
        verts, weight = nudge_region(env, d.sides[0])
        c = np.cos(NUDGE_ELEVATION)
        v = d.speed * np.array([c * np.cos(d.yaw), c * np.sin(d.yaw), np.sin(NUDGE_ELEVATION)])
        out["kick_verts"] = [int(k) for k in verts]
        out["kick_vel"] = [[float(x) for x in w * v] for w in weight]
    return out


def constants(env: TowelEnv) -> dict:
    ix = task.TaskIndex(env.model)
    lo, hi = ix.layers
    return {
        "control_hz": config.CONTROL_HZ, "substeps": config.SUBSTEPS, "timestep": config.TIMESTEP,
        "image_width": config.IMAGE_WIDTH, "image_height": config.IMAGE_HEIGHT, "cameras": list(CAMERAS),
        "actuators": list(ACTUATOR_NAMES),
        "reset_joints": oa_export.reset_joints(),
        "max_episode_steps": MAX_EPISODE_STEPS, "stages": list(task.STAGES),
        "table_center": list(TABLE_CENTER), "table_half_size": list(TABLE_HALF_SIZE), "off_table_z": OFF_TABLE_Z,
        # towel
        "towel": TOWEL, "towel_n": ix.n,
        "far": {s: int(ix.far[s]) for s in SIDES}, "near": {s: int(ix.near[s]) for s in SIDES},
        "layers": [[int(v) for v in lo], [int(v) for v in hi]],
        # grasp rule (keiarm/towel/grasp.py)
        "grab_slots": GRAB_SLOTS, "grab_radius": grasp.GRAB_RADIUS,
        "fingertip_offset": [float(v) for v in grasp.FINGERTIP_OFFSET],
        "close_cmd": grasp.CLOSE_CMD, "close_opening": grasp.CLOSE_OPENING, "open_cmd": grasp.OPEN_CMD,
        "gripper_open": dict(GRIPPER_OPEN), "hand_bodies": {s: f"openarm_{s}_ee_base_link" for s in SIDES},
        "grip_actuators": {s: f"{s}_finger1_ctrl" for s in SIDES},
        "finger_joint": {s: finger_joints(s)[0] for s in SIDES},
        "ee_sites": {s: ee_site(s) for s in SIDES}, "pinch_offset": list(PINCH_OFFSET),
        # success and stages (keiarm/towel/task.py)
        "corner_h": task.CORNER_H, "corner_v": task.CORNER_V, "area_range": list(task.AREA_RANGE),
        "max_violation": task.MAX_VIOLATION, "max_speed": task.MAX_SPEED,
        "success_hold_steps": task.SUCCESS_HOLD_STEPS, "stage_reach": task.STAGE_REACH,
        "stage_lift": task.STAGE_LIFT, "stage_carry": task.STAGE_CARRY,
        # body name prefixes for the finger / arm geoms (task.TaskIndex)
        "finger_body_prefix": {s: f"openarm_{s}_ee_" for s in SIDES}, "finger_body_suffix": "_finger",
        "arm_body_prefix": {s: f"openarm_{s}_" for s in SIDES},
        "policy_execute_steps": 10,
        "back_offset": BACK_OFFSET,  # keiarm/towel/render.py: the back is drawn this far below the mid-surface
    }


def scene_bundle(scene_seed: int, n: int, episode_seeds: list[int], disturbed: list[tuple[int, str]]) -> dict:
    env = TowelEnv(scene_seed, render=False, n=n)
    try:
        episodes = [record_episode(env, ep, "none") for ep in episode_seeds]
        episodes += [record_episode(env, ep, kind) for ep, kind in disturbed]
        return {
            "version": BUNDLE_VERSION,
            "robot": "openarm_v2",
            "task": "towel_half_fold",
            "mujoco": mujoco.__version__,
            "n": n,
            "scene": env.scene.to_dict(),
            "xml": build_xml(env.scene, n, web=True),
            "mesh_dir": "assets",
            "mesh_files": sorted({m.file for m in _web_spec_meshes(env)}),
            "episodes": episodes,
            "constants": constants(env),
        }
    finally:
        env.close()


def _web_spec_meshes(env: TowelEnv):
    from keiarm.openarm.scene import SceneParams

    return oa_export.web_spec(SceneParams(**env.scene.base)).meshes


def _bundle_job(job) -> dict:
    return scene_bundle(*job)


def ensure_meshes(out: Path = WEB_PUBLIC, keep: float = oa_export.VISUAL_KEEP) -> None:
    """The robot meshes are milestone 2's; write them only if they are missing."""
    assets = out / ASSETS_DIR
    if (assets / "LICENSE").exists():
        return
    from keiarm.openarm.scene import sample_scene

    oa_export.export_meshes(oa_export.web_spec(sample_scene(100000)), assets, keep)


def export_web(scene_seeds: list[int], meshes: list[int], episode_seeds: list[int], out: Path = WEB_PUBLIC,
               map_fn=map) -> dict:
    """Write all bundles (one per scene and N) and an index. `map_fn` may be a pool's map."""
    scenes = out / SCENES_DIR
    shutil.rmtree(scenes, ignore_errors=True)
    scenes.mkdir(parents=True)
    ensure_meshes(out)
    jobs = [(seed, n, episode_seeds, [(episode_seeds[0], EXPORT_DISTURBANCES[i % len(EXPORT_DISTURBANCES)])])
            for n in meshes for i, seed in enumerate(scene_seeds)]
    sizes = {}
    for (seed, n, *_), bundle in zip(jobs, map_fn(_bundle_job, jobs)):
        path = scenes / f"{seed}_n{n}.json"
        path.write_text(json.dumps(bundle, separators=(",", ":")))
        sizes[f"{seed}_n{n}"] = path.stat().st_size
    index = {"version": BUNDLE_VERSION, "scenes": [int(s) for s in scene_seeds], "meshes": [int(n) for n in meshes],
             "episodes": [int(e) for e in episode_seeds], "scene_bytes": sizes, "default_n": int(meshes[0])}
    (scenes / "index.json").write_text(json.dumps(index))
    return index


# --- Records (N = 16 playback without physics) -----------------------------------------------

RECORDS_DIR = "towel_records"  # under web/public (generated, not in git)
VERT_SCALE = 1e-4  # [m] per int16 unit (0.1 mm)
RECORD_TAIL_STEPS = 30


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def record_frames(scene_seed: int, episode_seed: int, disturb: str = "none", policy: str = "expert",
                  n: int = TOWEL_N) -> dict:
    """One episode recorded for playback: per frame (the initial state, then after every
    control step) the robot joints, the towel vertices (int16, 0.1 mm), the vertices each hand
    holds (counts), finger contact, the largest vertex speed and the stage. The browser
    recomputes the success check from these (it must agree with `python`)."""
    from keiarm.towel.policies import make_policy

    env = TowelEnv(scene_seed, render=False, n=n)
    try:
        ix = task.TaskIndex(env.model)
        joints = [j for s in SIDES for j in (*arm_joints(s), *finger_joints(s))]
        qadr = [env.model.joint(j).qposadr[0] for j in joints]
        frames = []
        run = [0]

        def grab(env):
            st = task.read_state(env.model, ix, env.data)
            chk = task.check_fold(st, ix, env.scene)
            run[0] = run[0] + 1 if chk.success else 0
            last = len(task.STAGES) - 1
            stage = last if run[0] >= task.SUCCESS_HOLD_STEPS else min(task.stage_of(st, ix, env.scene), last - 1)
            frames.append((env.data.qpos[qadr].copy(), st.verts.copy(), [len(st.held[s]) for s in SIDES],
                           int(st.finger_touch["left"]) | 2 * int(st.finger_touch["right"]),
                           float(st.vert_speed.max()), stage))

        pol = make_policy(policy)
        env.reset(episode_seed)
        grab(env)
        rec = run_episode(env, pol, episode_seed, disturb=disturb, on_step=grab, probe=True)
        for _ in range(RECORD_TAIL_STEPS if rec.success else 0):
            env.step(np.clip(pol.act(env), *env.ctrl_range.T))
            grab(env)
        q = np.array([f[0] for f in frames], np.float32)
        v = np.round(np.array([f[1] for f in frames]) / VERT_SCALE).astype(np.int16)
        bundle = scene_bundle_header(env, n)
        return {
            **bundle,
            "episode": env.episode.to_dict(),
            "robot_joints": joints, "frames": len(frames), "vert_scale": VERT_SCALE,
            "qpos": _b64(q), "verts": _b64(v),
            "held": [f[2] for f in frames],
            "touch": _b64(np.array([f[3] for f in frames], np.uint8)),
            "speed": _b64(np.minimum(np.round(np.array([f[4] for f in frames]) * 1000), 65535).astype(np.uint16)),
            "stage": _b64(np.array([f[5] for f in frames], np.uint8)),
            "python": {"success": rec.success, "steps": rec.steps, "stage": rec.stage, "failure": rec.failure,
                       "disturb": disturb, "disturb_step": None if rec.disturbance is None else rec.disturbance.step,
                       "policy": policy, "expert_could_finish": rec.expert_could_finish},
        }
    finally:
        env.close()


def scene_bundle_header(env: TowelEnv, n: int) -> dict:
    """The scene part of a bundle (model, meshes, constants), without episodes."""
    return {
        "version": BUNDLE_VERSION, "robot": "openarm_v2", "task": "towel_half_fold", "mujoco": mujoco.__version__,
        "n": n, "scene": env.scene.to_dict(), "xml": build_xml(env.scene, n, web=True), "mesh_dir": "assets",
        "mesh_files": sorted({m.file for m in _web_spec_meshes(env)}), "constants": constants(env),
    }


def _record_job(job) -> dict:
    name, label, scene_seed, episode_seed, disturb, policy = job
    return {"name": name, "label": label, **record_frames(scene_seed, episode_seed, disturb, policy)}


def export_records(jobs: list[tuple], out: Path = WEB_PUBLIC, map_fn=map) -> dict:
    """jobs: (name, label, scene_seed, episode_seed, disturb, policy). Writes one JSON per
    record and an index."""
    d = out / RECORDS_DIR
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    ensure_meshes(out)
    index = {"version": BUNDLE_VERSION, "records": []}
    for rec in map_fn(_record_job, jobs):
        path = d / f"{rec['name']}.json"
        path.write_text(json.dumps(rec, separators=(",", ":")))
        index["records"].append({"name": rec["name"], "label": rec["label"], "bytes": path.stat().st_size,
                                 **rec["python"]})
    (d / "index.json").write_text(json.dumps(index))
    return index

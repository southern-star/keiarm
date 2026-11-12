"""Export OpenArm scenes for the browser (milestone 2, Phase 4).

The OpenArm counterpart of keiarm/export.py. One JSON bundle per scene: the MJCF text, the
scene parameters, and per episode the initial state, the expert's recorded actions, an
optional disturbance (applied at its recorded step) and the Python outcome. Meshes are
served once from web/public/assets/openarm/:

- collision meshes (STL) are copied as they are: they decide the physics
- visual meshes (OBJ, 330k faces) are decimated to VISUAL_KEEP of their faces. The arm
  bodies have explicit <inertial>s and the visual geoms do not collide, so this changes
  the picture only, never the physics (tests/test_openarm.py checks bit-identical rollouts)
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from keiarm import config
from keiarm.openarm.env import CAMERAS, MAX_EPISODE_STEPS, OpenArmEnv
from keiarm.openarm.eval import DISTURBANCES, planned_disturbance, run_episode
from keiarm.openarm.expert import Expert
from keiarm.openarm.policies import ExpertPolicy
from keiarm.openarm.robot import (
    ACTUATOR_NAMES,
    PINCH_OFFSET,
    SIDES,
    arm_joints,
    ee_site,
    finger_joints,
)
from keiarm.openarm.rollout import SUCCESS_HOLD_STEPS
from keiarm.openarm.scene import PLATE_HEIGHT, TABLE_CENTER, TABLE_HALF_SIZE, build_spec
from keiarm.openarm.task import REST_SPEED, REST_TILT, REST_Z_TOL, STAGE_LIFT, STAGE_REACH, STAGES
from keiarm.paths import OPENARM_DIR, REPO_ROOT

WEB_PUBLIC = REPO_ROOT / "web" / "public"
SCENES_DIR = "openarm_scenes"  # under web/public (generated, not in git)
ASSETS_DIR = "assets/openarm"  # under web/public (generated, not in git)
BUNDLE_VERSION = 1
REPLAY_TAIL_STEPS = 30  # keep the expert running after success (see keiarm/export.py)
VISUAL_KEEP = 0.25  # fraction of each visual mesh's faces kept (docs/openarm_spike.md)
MIN_FACES = 200  # meshes this small are left alone
# Disturbed episodes in the bundle cycle through these kinds (one per scene).
EXPORT_DISTURBANCES = ("knock", "drop", "plate", "move")

_MESH_ROOT = OPENARM_DIR / "assets"


# --- Meshes --------------------------------------------------------------------------------


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of an OBJ file (positions only; polygons are fanned)."""
    verts, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")


def decimate(verts: np.ndarray, faces: np.ndarray, keep: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate vertices (OBJ/STL exports repeat them per face; without merging the
    simplifier tears holes), then quadric-decimate to `keep` of the faces."""
    import fast_simplification  # dev dependency: only the exporter needs it

    uniq, inverse = np.unique(np.round(verts, 7), axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    if len(faces) <= MIN_FACES or keep >= 1.0:
        return uniq, faces
    v, f = fast_simplification.simplify(uniq.astype(np.float32), faces.astype(np.int32),
                                        target_reduction=1.0 - keep)
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def relative_mesh_file(file: str) -> str:
    p = Path(file)
    return p.relative_to(_MESH_ROOT).as_posix() if p.is_absolute() else p.as_posix()


def web_spec(scene) -> mujoco.MjSpec:
    """The scene's spec with every mesh file relative to one mesh directory ("assets")."""
    spec = build_spec(scene)
    spec.meshdir = "assets"
    for mesh in spec.meshes:
        mesh.file = relative_mesh_file(mesh.file)
    return spec


def export_meshes(spec: mujoco.MjSpec, out: Path, keep: float = VISUAL_KEEP) -> dict:
    """Write every mesh the spec uses under `out` (same relative paths). Returns sizes."""
    stats = {"visual_faces_before": 0, "visual_faces_after": 0, "visual_bytes": 0, "collision_bytes": 0}
    for file in sorted({m.file for m in spec.meshes}):
        src, dst = _MESH_ROOT / file, out / file
        dst.parent.mkdir(parents=True, exist_ok=True)
        if file.startswith("visual/"):
            v, f = read_obj(src)
            stats["visual_faces_before"] += len(f)
            v, f = decimate(v, f, keep)
            stats["visual_faces_after"] += len(f)
            write_obj(dst, v, f)
            stats["visual_bytes"] += dst.stat().st_size
        else:
            shutil.copyfile(src, dst)
            stats["collision_bytes"] += dst.stat().st_size
    shutil.copyfile(OPENARM_DIR / "LICENSE", out / "LICENSE")
    return stats


# --- Episodes ------------------------------------------------------------------------------


def record_episode(env: OpenArmEnv, episode_seed: int, disturb: str) -> dict:
    env.reset(episode_seed)
    # The evaluation suite's disturbances for this episode (not yet fired), for the browser's
    # closed-loop evaluation (tools/browser_eval_openarm.py).
    plans = {k: asdict(planned_disturbance(env, k)) for k in DISTURBANCES[1:]}
    rec = run_episode(env, ExpertPolicy(), episode_seed, disturb=disturb, probe=False)
    actions = list(rec.actions)
    expert = Expert(env.model, env.scene)
    for _ in range(REPLAY_TAIL_STEPS):
        action, _ = expert.act(env.data)
        action = np.clip(action, *env.ctrl_range.T)
        env.step(action)
        actions.append(action)
    st = env.privileged()
    d = rec.disturbance
    return {
        **env.episode.to_dict(),
        "disturbance": None if d is None else asdict(d),
        "planned_disturbances": plans,
        "expert": {
            "success": rec.success,
            "steps": rec.steps,  # control steps until success was confirmed
            "stage": rec.stage,
            "stage_steps": rec.stage_steps,
            "actions": [[float(v) for v in a] for a in actions],
            "final_bar_pos": [float(v) for v in st.bar_pos],
            "final_joints": [float(v) for v in st.ctrl_joints],
        },
    }


def reset_joints() -> list[list[str]]:
    """Per actuator (ACTUATOR_NAMES order), the joints scene.reset_data sets to its start target."""
    out = []
    for side in SIDES:
        out += [[j] for j in arm_joints(side)]
        out.append(list(finger_joints(side)))
    assert len(out) == len(ACTUATOR_NAMES)
    return out


def constants() -> dict:
    return {
        "control_hz": config.CONTROL_HZ, "substeps": config.SUBSTEPS, "timestep": config.TIMESTEP,
        "image_width": config.IMAGE_WIDTH, "image_height": config.IMAGE_HEIGHT, "cameras": list(CAMERAS),
        "actuators": list(ACTUATOR_NAMES),
        # Joints set to each actuator's start target at reset (both finger joints of a gripper).
        "reset_joints": reset_joints(),
        "finger_bodies": {s: [f"openarm_{s}_ee_inner_finger", f"openarm_{s}_ee_outer_finger"] for s in SIDES},
        "ee_sites": {s: ee_site(s) for s in SIDES}, "pinch_offset": list(PINCH_OFFSET),
        "lift_height": STAGE_LIFT, "reach_distance": STAGE_REACH,  # task.stage_of
        "policy_execute_steps": 10,
        "plate_height": PLATE_HEIGHT, "rest_z_tol": REST_Z_TOL, "rest_tilt": float(REST_TILT),
        "rest_speed": REST_SPEED, "success_hold_steps": SUCCESS_HOLD_STEPS,
        "max_episode_steps": MAX_EPISODE_STEPS, "stages": list(STAGES),
        "table_center": list(TABLE_CENTER), "table_half_size": list(TABLE_HALF_SIZE),
    }


def scene_bundle(scene_seed: int, episode_seeds: list[int], disturbed: list[tuple[int, str]]) -> dict:
    env = OpenArmEnv(scene_seed, render=False)
    episodes = [record_episode(env, ep, "none") for ep in episode_seeds]
    episodes += [record_episode(env, ep, kind) for ep, kind in disturbed]
    spec = web_spec(env.scene)
    return {
        "version": BUNDLE_VERSION,
        "robot": "openarm_v2",
        "mujoco": mujoco.__version__,
        "scene": env.scene.to_dict(),
        "xml": spec.to_xml(),
        "mesh_dir": "assets",
        "mesh_files": sorted({m.file for m in spec.meshes}),
        "episodes": episodes,
        "constants": constants(),
    }


def _bundle_job(job) -> dict:
    return scene_bundle(*job)


def export_web(scene_seeds: list[int], episode_seeds: list[int], out: Path = WEB_PUBLIC,
               keep: float = VISUAL_KEEP, map_fn=map) -> dict:
    """Write all bundles, the meshes and an index. `map_fn` may be a process pool's map."""
    scenes = out / SCENES_DIR
    shutil.rmtree(scenes, ignore_errors=True)
    scenes.mkdir(parents=True)
    assets = out / ASSETS_DIR
    shutil.rmtree(assets, ignore_errors=True)
    # The mesh set does not depend on the scene.
    mesh_stats = export_meshes(web_spec(OpenArmEnv(scene_seeds[0], render=False).scene), assets, keep)
    jobs = [(seed, episode_seeds, [(episode_seeds[0], EXPORT_DISTURBANCES[i % len(EXPORT_DISTURBANCES)])])
            for i, seed in enumerate(scene_seeds)]
    sizes = {}
    for seed, bundle in zip(scene_seeds, map_fn(_bundle_job, jobs)):
        path = scenes / f"{seed}.json"
        path.write_text(json.dumps(bundle, separators=(",", ":")))
        sizes[seed] = path.stat().st_size
    index = {"version": BUNDLE_VERSION, "scenes": [int(s) for s in scene_seeds],
             "episodes": [int(e) for e in episode_seeds], "visual_keep": keep, **mesh_stats,
             "mesh_bytes": int(sum(p.stat().st_size for p in assets.rglob("*") if p.is_file())),
             "scene_bytes": sizes}
    (scenes / "index.json").write_text(json.dumps(index))
    return index

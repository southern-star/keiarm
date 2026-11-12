"""OpenArm v2 (milestone 2): model, physics settings, start pose, grasp (Phase 0) and
procedural scenes (Phase 1)."""

import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from keiarm import config
from keiarm.openarm.ik import ArmIK
from keiarm.openarm.robot import (
    ACTION_DIM,
    ACTUATOR_NAMES,
    GRIPPER_OPEN,
    HOME_ARM_QPOS,
    HOME_PINCH,
    SIDES,
    arm_joints,
    openarm_spec,
    wrist_camera,
)
from keiarm.openarm.scene import (
    BAR_GRIP_OFFSET,
    PLATE_HEIGHT,
    bar_corners,
    build_model,
    grip_points,
    in_sector,
    reset_data,
    sample_episode,
    sample_scene,
    set_bar_pose,
)
from keiarm.openarm.scripted import scripted_grasp
from keiarm.paths import REPO_ROOT

N_PLACEMENTS = 300


def nominal_episode(scene, episode_seed: int = 0):
    """The sampled episode with the arms exactly at the start pose and grippers fully open."""
    ctrl = [q for side in SIDES for q in (*HOME_ARM_QPOS[side], GRIPPER_OPEN[side])]
    return replace(sample_episode(scene, episode_seed), start_ctrl=ctrl)


@pytest.fixture(scope="module")
def scene():
    return sample_scene(0)


@pytest.fixture(scope="module")
def model(scene) -> mujoco.MjModel:
    return build_model(scene)


# --- Phase 0: robot and physics ------------------------------------------------------------


def test_vendored_model_actuators():
    m = openarm_spec().compile()
    assert ACTION_DIM == 16
    assert [m.actuator(i).name for i in range(m.nu)] == list(ACTUATOR_NAMES)
    for side in SIDES:
        assert m.camera(wrist_camera(side)).id >= 0
        for name in arm_joints(side):
            assert m.actuator(name.removeprefix("openarm_") + "_ctrl").trnid[0] == m.joint(name).id


def test_physics_overrides(model):
    assert model.opt.timestep == pytest.approx(config.TIMESTEP)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == 10.0
    arm_bodies = [b for b in range(model.nbody) if model.body(b).name.startswith("openarm_")]
    assert len(arm_bodies) == 20
    assert all(model.body_gravcomp[b] == 1.0 for b in arm_bodies)


def test_home_pose_is_mirrored_topdown(scene, model):
    data = mujoco.MjData(model)
    reset_data(model, data, scene, nominal_episode(scene))
    assert data.ncon == 0
    for side in SIDES:
        pos, rot = ArmIK(model, side).pinch_pose(data)
        np.testing.assert_allclose(pos, HOME_PINCH[side], atol=1e-3)
        assert rot[2, 2] > 0.999  # fingers straight down


def test_home_is_held_without_sag(scene, model):
    """Gravity compensation: the low-gain actuators hold the start pose to < 1e-3 rad."""
    data = mujoco.MjData(model)
    reset_data(model, data, scene, nominal_episode(scene))
    mujoco.mj_step(model, data, nstep=2 * config.CONTROL_HZ * config.SUBSTEPS)
    for side in SIDES:
        for name in arm_joints(side):
            act = model.actuator(name.removeprefix("openarm_") + "_ctrl").id
            assert abs(data.qpos[model.joint(name).qposadr[0]] - data.ctrl[act]) < 1e-3, name


def test_stepping_is_deterministic(scene, model):
    finals = []
    for _ in range(2):
        data = mujoco.MjData(model)
        reset_data(model, data, scene, sample_episode(scene, 0))
        rng = np.random.default_rng(0)
        start = data.ctrl.copy()
        for _ in range(60):
            data.ctrl[:] = np.clip(start + rng.normal(0, 0.1, model.nu), *model.actuator_ctrlrange.T)
            mujoco.mj_step(model, data, nstep=config.SUBSTEPS)
        finals.append(data.qpos.copy())
    np.testing.assert_array_equal(finals[0], finals[1])


# --- Phase 1: procedural scenes ----------------------------------------------------------


def test_scenes_and_episodes_are_pure_functions_of_seeds():
    assert sample_scene(7) == sample_scene(7)
    assert sample_episode(sample_scene(7), 3) == sample_episode(sample_scene(7), 3)
    assert sample_scene(7) != sample_scene(8)
    assert sample_episode(sample_scene(7), 3) != sample_episode(sample_scene(7), 4)
    json.dumps(sample_scene(7).to_dict())
    json.dumps(sample_episode(sample_scene(7), 3).to_dict())


def test_model_depends_on_scene_only(scene):
    """Two episodes of one scene share the model; only the initial MjData differs."""
    m, again = build_model(scene), build_model(scene)
    assert np.array_equal(m.geom_size, again.geom_size) and np.array_equal(m.geom_rgba, again.geom_rgba)
    d1, d2 = mujoco.MjData(m), mujoco.MjData(m)
    reset_data(m, d1, scene, sample_episode(scene, 0))
    reset_data(m, d2, scene, sample_episode(scene, 1))
    assert not np.array_equal(d1.qpos, d2.qpos)


def test_scene_seed_changes_appearance_and_geometry():
    a, b = sample_scene(0), sample_scene(1)
    assert a.bar_rgb != b.bar_rgb and a.bar_size != b.bar_size and a.cam_front_pos != b.cam_front_pos
    assert a.wrist_cam_quat["left"] != a.wrist_cam_quat["right"]


def test_placements_respect_the_spawn_rules():
    for i in range(N_PLACEMENTS):
        scene = sample_scene(i % 50)
        ep = sample_episode(scene, i)
        grips = grip_points(ep.bar_xy, ep.bar_yaw)
        assert in_sector(grips["right"], "right")
        assert np.radians(-20.0) - 1e-9 <= ep.bar_yaw <= np.radians(45.0) + 1e-9
        corners = bar_corners(ep.bar_xy, ep.bar_yaw, scene.bar_size)
        assert corners[:, 0].min() >= 0.10 and corners[:, 1].max() < 0.0
        place = np.asarray(ep.plate_xy) + [0.0, BAR_GRIP_OFFSET]
        assert in_sector(place, "left")
        assert ep.plate_xy[1] - scene.plate_radius > corners[:, 1].max()  # plate clear of the bar


def test_spawn_sectors_lie_inside_the_reach_map():
    """Every sampled right grasp point (and mirrored left place point) is in a cell that
    docs/openarm/reach_bar.json marks reachable for grasping and carrying."""
    reach = json.loads((REPO_ROOT / "docs" / "openarm" / "reach_bar.json").read_text())
    xs, ys = np.array(reach["xs"]), np.array(reach["ys"])
    for i in range(N_PLACEMENTS):
        scene = sample_scene(i % 50)
        ep = sample_episode(scene, i)
        points = {"right": grip_points(ep.bar_xy, ep.bar_yaw)["right"],
                  "left": np.asarray(ep.plate_xy) + [0.0, BAR_GRIP_OFFSET]}
        for side, p in points.items():
            ok = np.array(reach["reachable"][side]["grasp"]) & np.array(reach["reachable"][side]["carry"])
            assert ok[np.argmin(abs(ys - p[1])), np.argmin(abs(xs - p[0]))], (i, side, p)


def test_no_contacts_at_reset():
    for seed in range(10):
        scene = sample_scene(seed)
        m = build_model(scene)
        d = mujoco.MjData(m)
        reset_data(m, d, scene, sample_episode(scene, seed))
        assert d.ncon == 0, seed


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_right_hand_grasps_the_spawned_bar(seed):
    scene = sample_scene(seed)
    ep = sample_episode(scene, 0)
    m = build_model(scene)
    d = mujoco.MjData(m)
    reset_data(m, d, scene, ep)
    res = scripted_grasp(m, d, "right", grip_points(ep.bar_xy, ep.bar_yaw)["right"], ep.bar_yaw + np.pi / 2,
                         scene.bar_size[2] / 2)
    assert res.lift > 0.12, res.log
    assert res.arm_table_contact_steps == 0, res.log


def test_left_hand_reaches_the_bar_on_the_plate():
    """The place pose is reachable: a bar lying on the plate (along y) can be grasped by the left hand."""
    scene = sample_scene(3)
    ep = sample_episode(scene, 0)
    m = build_model(scene)
    d = mujoco.MjData(m)
    reset_data(m, d, scene, ep)
    set_bar_pose(m, d, ep.plate_xy, 0.0, PLATE_HEIGHT + scene.bar_size[2] / 2 + 1e-3)
    mujoco.mj_forward(m, d)
    res = scripted_grasp(m, d, "left", grip_points(ep.plate_xy, 0.0)["left"], np.pi / 2,
                         PLATE_HEIGHT + scene.bar_size[2] / 2)
    assert res.lift > 0.12, res.log
    assert res.arm_table_contact_steps == 0, res.log


# --- Phase 2: expert -------------------------------------------------------------------------


def _snapshot(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate([data.qpos, data.qvel, data.ctrl, data.qacc_warmstart, data.mocap_pos.ravel(), [data.time]])


@pytest.mark.parametrize("seed", [0, 100000])
def test_expert_hands_over_and_places_the_bar(seed):
    from keiarm.openarm.env import OpenArmEnv
    from keiarm.openarm.rollout import run_expert
    from keiarm.openarm.task import STAGES

    res = run_expert(OpenArmEnv(seed, render=False), 0)
    assert res.success and res.steps < 330, (res.steps, STAGES[res.stage])
    assert res.arms_touch_steps == 0
    assert res.actions.shape == (res.steps, 16)
    phases = [p.split("/") for p in res.phases]
    assert any(l == "CLOSE" and r in ("CARRY", "HANDOVER") for l, r in phases)  # a real hand-over
    assert any(r == "RELEASE" for _, r in phases)


def test_expert_is_a_pure_function_of_state():
    from keiarm.openarm.env import OpenArmEnv
    from keiarm.openarm.expert import Expert

    env = OpenArmEnv(5, render=False)
    env.reset(0)
    expert, other = Expert(env.model, env.scene), Expert(env.model, env.scene)
    for _ in range(40):
        before = _snapshot(env.data)
        a1, _ = expert.act(env.data)
        np.testing.assert_array_equal(before, _snapshot(env.data))
        a2, _ = other.act(env.data)  # a fresh expert with no history gives the same answer
        np.testing.assert_array_equal(a1, a2)
        env.step(a1)


def test_dart_noise_spares_the_grippers():
    from keiarm.openarm.rollout import GRIPPER_COLUMNS, dart_noise

    noise = dart_noise(np.random.default_rng(0), 50, 0.0065)
    assert np.all(noise[:, GRIPPER_COLUMNS] == 0.0) and noise.std() > 0.003


# --- Phase 3: evaluation harness -----------------------------------------------------------


def test_disturbances_are_pure_functions_of_seeds_and_respect_their_rules():
    from keiarm.openarm.env import OpenArmEnv
    from keiarm.openarm.eval import DISTURBANCES, planned_disturbance

    env = OpenArmEnv(7, render=False)
    for ep in range(20):
        env.reset(ep)
        assert planned_disturbance(env, "none") is None
        for kind in DISTURBANCES[1:]:
            d1, d2 = planned_disturbance(env, kind), planned_disturbance(env, kind)
            assert d1 == d2 and d1.step is None
            if kind == "plate":
                assert np.linalg.norm(np.array([d1.x, d1.y]) - env.episode.plate_xy) >= 0.05
                continue
            grips = grip_points([d1.x, d1.y], d1.yaw)
            assert in_sector(grips["right"], "right"), (kind, ep)
            if kind == "move":  # clear of where the right hand was heading
                old = grip_points(env.episode.bar_xy, env.episode.bar_yaw)["right"]
                assert np.linalg.norm(grips["right"] - old) > 0.04


def test_harness_classifies_baselines_by_stage():
    from keiarm.openarm.env import OpenArmEnv
    from keiarm.openarm.eval import run_episode
    from keiarm.openarm.policies import make_policy

    env = OpenArmEnv(100000, render=False)
    hold = run_episode(env, make_policy("hold"), 0, max_steps=60, probe=False)
    assert not hold.success and hold.failure == "grasp_failed" and hold.stage == "none"
    frozen = run_episode(env, make_policy("freeze"), 0, max_steps=150)
    assert not frozen.success and frozen.stage == "lift" and frozen.failure == "handover_failed"
    assert list(frozen.stage_steps) == ["reach", "grasp_right", "lift"]
    assert frozen.expert_could_finish  # the expert takes over from the frozen state and finishes


@pytest.mark.parametrize("kind", ["knock", "drop", "plate"])
def test_expert_recovers_from_disturbances_and_replay_matches(kind):
    from keiarm.openarm.env import OpenArmEnv
    from keiarm.openarm.eval import replay, run_episode
    from keiarm.openarm.policies import make_policy
    from keiarm.openarm.task import STAGES

    env = OpenArmEnv(100000, render=False)
    rec = run_episode(env, make_policy("expert"), 0, disturb=kind)
    assert rec.disturbance.step is not None
    assert rec.success and rec.arms_touch_steps == 0 and rec.drops == 0, (rec.stage, rec.failure)
    assert rec.disturbance.step > rec.stage_steps[STAGES[rec.disturbance.after_stage]] - 1
    final = env.privileged().bar_pos
    np.testing.assert_array_equal(replay(env, rec).bar_pos, final)


def test_harness_counts_a_real_drop_once():
    from keiarm.openarm.env import OpenArmEnv
    from keiarm.openarm.eval import run_episode
    from keiarm.openarm.policies import ExpertPolicy
    from keiarm.openarm.robot import GRIPPER_OPEN

    class OpensMidCarry(ExpertPolicy):  # lets go of the bar 20 steps into the carry, once
        def reset(self, env):
            super().reset(env)
            self.carry_start = None

        def act(self, env):
            action, info = self.expert.act(env.data)
            if info.phase.endswith("/CARRY") and self.carry_start is None:
                self.carry_start = env.steps
            if self.carry_start is not None and 20 <= env.steps - self.carry_start < 35:
                action[15] = GRIPPER_OPEN["right"]
            return action

    rec = run_episode(OpenArmEnv(100000, render=False), OpensMidCarry(), 0)
    assert rec.drops == 1
    assert rec.success  # the expert picks it up again


# --- Phase 4: browser export ----------------------------------------------------------------


def test_web_export_decimates_visuals_without_changing_physics(tmp_path):
    """The browser model (exported MJCF + decimated visual meshes) steps bit-identically to
    the Python model, and the bundle's reset recipe matches scene.reset_data."""
    from pathlib import Path

    from keiarm.openarm import export
    from keiarm.openarm.env import OpenArmEnv
    from keiarm.openarm.expert import Expert
    from keiarm.openarm.task import TaskIndex

    env = OpenArmEnv(100000, render=False)
    spec = export.web_spec(env.scene)
    assert all(not Path(m.file).is_absolute() for m in spec.meshes)
    stats = export.export_meshes(spec, tmp_path / "assets")
    assert stats["visual_faces_after"] < 0.3 * stats["visual_faces_before"]
    xml = spec.to_xml()
    assert 'gravcomp="1"' in xml  # must be in the XML: WASM ignores body_gravcomp written later
    (tmp_path / "scene.xml").write_text(xml)
    web = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    web.opt.timestep = env.model.opt.timestep
    assert web.nmeshface < env.model.nmeshface

    env.reset(0)
    data = mujoco.MjData(web)
    # The browser's reset (web/src/openarm/sim.js): every joint in reset_joints[i] and
    # actuator i get start_ctrl[i]; bar and plate as in reset_data.
    mujoco.mj_resetData(web, data)
    ix = TaskIndex(web)
    for i, names in enumerate(export.constants()["reset_joints"]):
        for n in names:
            data.qpos[web.joint(n).qposadr[0]] = env.episode.start_ctrl[i]
        data.ctrl[ix.act[i]] = env.episode.start_ctrl[i]
    a = ix.bar_qadr
    data.qpos[a : a + 3] = [*env.episode.bar_xy, env.scene.bar_size[2] / 2 + 1e-3]
    data.qpos[a + 3 : a + 7] = [np.cos(env.episode.bar_yaw / 2), 0, 0, np.sin(env.episode.bar_yaw / 2)]
    data.mocap_pos[ix.plate_mocap] = [*env.episode.plate_xy, export.PLATE_HEIGHT / 2]
    mujoco.mj_forward(web, data)
    np.testing.assert_array_equal(data.qpos, env.data.qpos)

    expert = Expert(env.model, env.scene)
    for _ in range(90):  # through the grasp and lift
        action = np.clip(expert.act(env.data)[0], *env.ctrl_range.T)
        env.step(action)
        data.ctrl[ix.act] = action
        mujoco.mj_step(web, data, nstep=config.SUBSTEPS)
    np.testing.assert_array_equal(data.qpos, env.data.qpos)

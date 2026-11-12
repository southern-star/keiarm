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

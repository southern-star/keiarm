import os

import mujoco
import numpy as np
import pytest

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.scene import (
    AZIMUTH_LIMIT,
    CUBE_RADIUS_RANGE,
    PAN_AXIS_XY,
    PLATE_RADIUS_RANGE,
    build_spec,
    sample_episode,
    sample_scene,
)

os.environ.setdefault("MUJOCO_GL", "egl")
SEEDS = [0, 1, 2, 3999, 100000, 100099]


def test_splits_are_disjoint():
    assert not set(config.TRAIN_SCENE_SEEDS) & set(config.EVAL_SCENE_SEEDS)
    assert config.split_of(0) == "train" and config.split_of(3999) == "train"
    assert config.split_of(100000) == "eval" and config.split_of(100099) == "eval"
    assert config.split_of(4000) is None and config.split_of(100100) is None


def test_control_period_is_exact():
    assert config.TIMESTEP * config.SUBSTEPS == pytest.approx(1 / config.CONTROL_HZ, abs=1e-15)


@pytest.mark.parametrize("seed", SEEDS)
def test_sampling_is_deterministic(seed):
    assert sample_scene(seed) == sample_scene(seed)
    scene = sample_scene(seed)
    assert sample_episode(scene, 7) == sample_episode(scene, 7)
    assert sample_episode(scene, 7) != sample_episode(scene, 8)
    assert build_spec(scene).to_xml() == build_spec(scene).to_xml()


def test_different_seeds_give_different_scenes():
    assert len({str(sample_scene(s).to_dict() | {"scene_seed": 0}) for s in range(50)}) == 50


def test_placement_stays_in_reachable_region():
    for seed in range(30):
        scene = sample_scene(seed)
        for ep in range(10):
            e = sample_episode(scene, ep)
            for xy, (lo, hi) in ((e.cube_xy, CUBE_RADIUS_RANGE), (e.plate_xy, PLATE_RADIUS_RANGE)):
                d = np.subtract(xy, PAN_AXIS_XY)
                assert lo - 1e-6 <= np.hypot(*d) <= hi + 1e-6
                assert abs(np.arctan2(d[1], d[0])) <= AZIMUTH_LIMIT + 1e-6
            gap = np.linalg.norm(np.subtract(e.cube_xy, e.plate_xy)) - scene.plate_radius
            assert gap >= scene.cube_size * np.sqrt(0.5) + 0.03 - 1e-6


def _rollout(seed: int, episode: int, steps: int = 60) -> np.ndarray:
    env = KeiArmEnv(seed, render=False)
    env.reset(episode)
    rng = np.random.default_rng(123)
    home = env.joints()
    for _ in range(steps):
        env.step(home + rng.uniform(-0.3, 0.3, 6))
    return np.concatenate([env.data.qpos, env.data.qvel])


@pytest.mark.parametrize("seed", [0, 100000])
def test_rollout_is_deterministic(seed):
    a, b = _rollout(seed, 3), _rollout(seed, 3)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, _rollout(seed, 4))


@pytest.mark.parametrize("seed", SEEDS)
def test_initial_state_is_clean_and_cube_rests(seed):
    env = KeiArmEnv(seed, render=False)
    for ep in range(3):
        env.reset(ep)
        m, d = env.model, env.data
        robot_bodies = {b for b in range(m.nbody) if m.body_rootid[b] == m.body("base").id}
        for c in d.contact:
            assert m.geom_bodyid[c.geom1] not in robot_bodies and m.geom_bodyid[c.geom2] not in robot_bodies
        start = env.privileged().cube_pos
        hold = env.joints()
        for _ in range(30):
            env.step(hold)
        p = env.privileged()
        assert np.linalg.norm(p.cube_pos[:2] - start[:2]) < 1e-3
        assert p.cube_pos[2] == pytest.approx(env.scene.cube_size / 2, abs=1e-3)
        assert np.linalg.norm(p.cube_linvel) < 1e-3
        np.testing.assert_allclose(env.joints(), hold, atol=0.02)  # arm holds its start pose


def test_images_are_deterministic_and_show_the_scene():
    try:
        env_a, env_b = KeiArmEnv(5), KeiArmEnv(5)
    except Exception as exc:  # noqa: BLE001 - no GL context available (e.g. CI without EGL)
        pytest.skip(f"renderer unavailable: {exc}")
    for env in (env_a, env_b):
        env.reset(0)
    a, b = env_a.observe(), env_b.observe()
    assert list(a.images) == list(config.CAMERAS)
    for cam in config.CAMERAS:
        assert a.images[cam].shape == (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3)
        np.testing.assert_array_equal(a.images[cam], b.images[cam])
        assert a.images[cam].std() > 5  # not a blank frame
    # The cube is visible from the front camera in its own colour.
    env_a._renderer.enable_segmentation_rendering()
    env_a._renderer.update_scene(env_a.data, camera="cam_front")
    seg = env_a._renderer.render()
    assert (seg[..., 0] == env_a.model.geom("cube").id).sum() >= 4
    assert mujoco.mj_id2name(env_a.model, mujoco.mjtObj.mjOBJ_CAMERA, 0) is not None

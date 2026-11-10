import mujoco
import numpy as np
import pytest

from keiarm.env import KeiArmEnv
from keiarm.expert import PHASES, Expert
from keiarm.rollout import CHUNK, expert_chunk, replay, run_expert
from keiarm.task import is_success


def _snapshot(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate([data.qpos, data.qvel, data.ctrl, data.qacc_warmstart, data.mocap_pos.ravel(),
                           [data.time]])


@pytest.mark.parametrize("seed", [0, 1, 100000, 100050])
def test_expert_solves_the_task(seed):
    env = KeiArmEnv(seed, render=False)
    res = run_expert(env, 0)
    assert res.success and res.steps < 300
    assert set(res.phases) <= set(PHASES)
    assert res.actions.shape == res.labels.shape == (res.steps, 6)
    np.testing.assert_array_equal(res.actions, res.labels)  # no DART noise -> executed == label


def test_expert_is_a_pure_function_of_state():
    env = KeiArmEnv(3, render=False)
    env.reset(0)
    expert, other = Expert(env.model, env.scene), Expert(env.model, env.scene)
    for _ in range(25):
        before = _snapshot(env.data)
        a1, _ = expert.act(env.data)
        np.testing.assert_array_equal(before, _snapshot(env.data))
        a2, _ = other.act(env.data)  # a fresh expert with no history gives the same answer
        np.testing.assert_array_equal(a1, a2)
        env.step(a1)


def test_chunk_labels_do_not_disturb_the_episode():
    env = KeiArmEnv(4, render=False)
    env.reset(1)
    expert = Expert(env.model, env.scene)
    for _ in range(10):
        env.step(expert.act(env.data)[0])
    before = _snapshot(env.data)
    chunk = expert_chunk(env, expert)
    np.testing.assert_array_equal(before, _snapshot(env.data))
    assert chunk.shape == (CHUNK, 6)
    # The chunk is exactly what the expert goes on to do when nothing perturbs it.
    for k in range(CHUNK):
        action, _ = expert.act(env.data)
        np.testing.assert_array_equal(action, chunk[k])
        env.step(action)


def test_replay_reproduces_the_episode():
    env = KeiArmEnv(100007, render=False)
    res = run_expert(env, 2, dart_sigma=0.01)
    final = np.concatenate([env.data.qpos, env.data.qvel])
    assert not np.array_equal(res.actions, res.labels)  # DART noise was applied
    np.testing.assert_array_equal(replay(KeiArmEnv(100007, render=False), 2, res.actions), final)


def test_dart_noise_depends_only_on_seeds():
    a = run_expert(KeiArmEnv(8, render=False), 0, dart_sigma=0.01)
    b = run_expert(KeiArmEnv(8, render=False), 0, dart_sigma=0.01)
    np.testing.assert_array_equal(a.actions, b.actions)


@pytest.mark.parametrize("seed,step", [(5, 15), (6, 35), (100020, 50)])
def test_expert_recovers_when_the_cube_is_moved(seed, step):
    env = KeiArmEnv(seed, render=False)
    res = run_expert(env, 0, teleport_at=step)
    assert res.success
    assert is_success(env.privileged(), env.scene)

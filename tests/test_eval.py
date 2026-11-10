import numpy as np
import pytest

from keiarm.env import KeiArmEnv
from keiarm.eval import run_episode, sign_test, summarize
from keiarm.ik import ArmIK
from keiarm.policies import ExpertPolicy, make_policy
from keiarm.rollout import replay


def _run(spec_or_policy, seed=100001, max_steps=200):
    policy = make_policy(spec_or_policy) if isinstance(spec_or_policy, str) else spec_or_policy
    env = KeiArmEnv(seed, render=False)
    return env, run_episode(env, policy, 0, max_steps=max_steps)


def test_expert_episode_is_clean():
    _, rec = _run("expert")
    assert rec.success and rec.stage == "success" and rec.failure is None
    assert (rec.table_hits, rec.drops, rec.workspace_exits) == (0, 0, 0)
    assert rec.expert_could_finish is None


def test_idle_policy_is_an_approach_failure_the_expert_could_fix():
    _, rec = _run("hold")
    assert not rec.success and rec.steps == 200
    assert rec.stage == "none" and rec.failure == "approach_failed"
    assert rec.expert_could_finish is True


def test_freezing_after_the_lift_is_a_transit_timeout():
    _, rec = _run("grasp-only")
    assert rec.stage == "lift" and rec.failure == "transit_timeout" and rec.drops == 0
    assert rec.expert_could_finish is True


class _DropPolicy(ExpertPolicy):
    """Opens the gripper as soon as the cube is being carried, then stays put."""

    def act(self, env):
        action, info = self.expert.act(env.data)
        if info.phase == "MOVE" or getattr(self, "dropped", None) is not None:
            if getattr(self, "dropped", None) is None:
                self.dropped = action.copy()
                self.dropped[5] = 0.9
            return self.dropped
        return action


def test_letting_go_in_the_air_counts_as_a_drop():
    _, rec = _run(_DropPolicy())
    assert rec.drops == 1 and rec.failure == "dropped_in_transit"


class _ThrowAwayPolicy(ExpertPolicy):
    """Moves the cube out of reach (but still on the table) at step 5."""

    def act(self, env):
        if env.steps == 5:
            env.teleport_cube([0.50, 0.30])
        return super().act(env)


def test_unreachable_cube_is_out_of_workspace_and_really_stuck():
    _, rec = _run(_ThrowAwayPolicy())
    assert rec.workspace_exits == 1 and rec.failure == "out_of_workspace"
    assert rec.expert_could_finish is False


class _SlamPolicy:
    """Commands a gripper pose 4 cm below the table surface, fingertips first."""

    needs_images = False

    def reset(self, env):
        q = ArmIK(env.model).solve_topdown(np.array([0.20, 0.0, -0.04]), env.data.qpos).q
        self.target = q[env.ix.qadr]

    def act(self, env):
        return self.target


def test_pushing_into_the_table_is_a_table_hit():
    _, rec = _run(_SlamPolicy(), max_steps=60)
    assert rec.table_hits >= 1 and rec.max_table_force > 20


def test_failure_log_replays_to_the_same_state():
    env, rec = _run("grasp-only")
    final = np.concatenate([env.data.qpos, env.data.qvel])
    np.testing.assert_array_equal(replay(KeiArmEnv(100001, render=False), 0, rec.actions), final)


def test_summary_counts():
    recs = [_run("expert", seed=s)[1] for s in (100001, 100002)] + [_run("hold")[1]]
    s = summarize(recs)
    assert s["episodes"] == 3 and s["success"] == 2
    assert s["stage_reached"] == {"reach": 2, "grasp": 2, "lift": 2, "over_plate": 2, "success": 2}
    assert s["failures"] == {"approach_failed": 1} and s["timeouts_expert_could_finish"] == 1


def test_sign_test():
    a = {(i, 0): True for i in range(10)}
    b = {(i, 0): i >= 5 for i in range(10)}
    t = sign_test(a, b)
    assert (t["a_only"], t["b_only"], t["both"], t["neither"]) == (5, 0, 5, 0)
    assert t["p_value"] == pytest.approx(2 / 32)
    assert sign_test(a, a)["p_value"] == 1.0
    c = {(i, 0): i % 2 == 0 for i in range(10)}
    d = {(i, 0): i % 2 == 1 for i in range(10)}
    assert sign_test(c, d)["p_value"] == 1.0  # 5 vs 5
    assert sign_test(a, {(99, 0): True})["paired_episodes"] == 0


def test_disturbed_episode_is_recorded_and_replays():
    env = KeiArmEnv(100003, render=False)
    rec = run_episode(env, make_policy("expert"), 0, disturb="random")
    final = np.concatenate([env.data.qpos, env.data.qvel])
    assert rec.success and rec.teleport is not None and 10 <= rec.teleport.step < 60
    assert rec.drops == 0  # having the cube taken away is not the policy dropping it
    again = run_episode(KeiArmEnv(100003, render=False), make_policy("expert"), 0, disturb="random")
    assert again.teleport == rec.teleport  # the disturbance depends only on the seeds
    replayed = replay(KeiArmEnv(100003, render=False), 0, rec.actions, teleports=(rec.teleport,))
    np.testing.assert_array_equal(replayed, final)
    assert rec.to_json()["teleport"]["step"] == rec.teleport.step

"""Towel episodes: expert rollouts (with optional DART noise), milestone 3."""

from dataclasses import dataclass, field

import numpy as np

from keiarm.openarm.rollout import dart_noise
from keiarm.towel.env import MAX_EPISODE_STEPS, TowelEnv
from keiarm.towel.expert import Expert
from keiarm.towel.scene import episode_rng
from keiarm.towel.task import SUCCESS_HOLD_STEPS, FoldCheck, check_fold, read_state, stage_of

DART_STREAM = 1


@dataclass
class EpisodeResult:
    scene_seed: int
    episode_seed: int
    n: int  # towel mesh vertices per side
    success: bool
    steps: int
    stage: int  # furthest stage reached (index into task.STAGES)
    arms_touch_steps: int  # control steps ending with the two arms in contact
    final: FoldCheck  # the fold measurements at the last step
    actions: np.ndarray  # (T, 16) executed actions
    labels: np.ndarray  # (T, 16) clean expert actions
    phases: list[str] = field(default_factory=list)


def run_expert(env: TowelEnv, episode_seed: int, dart_sigma: float = 0.0,
               max_steps: int = MAX_EPISODE_STEPS, on_step=None) -> EpisodeResult:
    env.reset(episode_seed)
    expert = Expert(env.model, env.scene)
    ix = expert.ix
    noise = dart_noise(episode_rng(env.scene.scene_seed, episode_seed, DART_STREAM), max_steps, dart_sigma)
    actions, labels, phases = [], [], []
    stage, success, held_for, touch_steps = 0, False, 0, 0
    chk = None
    label = None
    for t in range(max_steps):
        # with noise, the carry continues from the previous clean label (Expert.act)
        label, info = expert.act(env.data, label if dart_sigma > 0 else None)
        action = np.clip(label + noise[t], *env.ctrl_range.T)
        env.step(action)
        actions.append(action)
        labels.append(label)
        phases.append(info.phase)
        st = read_state(env.model, ix, env.data)
        stage = max(stage, stage_of(st, ix, env.scene))
        touch_steps += st.arms_touch
        chk = check_fold(st, ix, env.scene)
        if on_step is not None:
            on_step(env, info)
        held_for = held_for + 1 if chk.success else 0
        if held_for >= SUCCESS_HOLD_STEPS:
            success = True
            break
    return EpisodeResult(env.scene.scene_seed, episode_seed, ix.n, success, len(actions), stage, touch_steps, chk,
                         np.array(actions), np.array(labels), phases)

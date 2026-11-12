"""OpenArm episodes: expert rollouts (with optional DART noise)."""

from dataclasses import dataclass, field

import numpy as np

from keiarm.openarm.env import MAX_EPISODE_STEPS, OpenArmEnv
from keiarm.openarm.expert import Expert
from keiarm.openarm.robot import ACTUATOR_NAMES
from keiarm.openarm.scene import episode_rng
from keiarm.openarm.task import is_success, stage_of

DART_STREAM = 1
SUCCESS_HOLD_STEPS = 10
GRIPPER_COLUMNS = [i for i, n in enumerate(ACTUATOR_NAMES) if "finger" in n]


@dataclass
class EpisodeResult:
    scene_seed: int
    episode_seed: int
    success: bool
    steps: int
    stage: int  # furthest stage reached (index into task.STAGES)
    arms_touch_steps: int  # control steps ending with the two arms in contact
    actions: np.ndarray  # (T, 16) executed actions
    labels: np.ndarray  # (T, 16) clean expert actions
    phases: list[str] = field(default_factory=list)


def dart_noise(rng: np.random.Generator, steps: int, sigma: float, corr: float = 0.9) -> np.ndarray:
    """AR(1) joint-target noise with stationary std `sigma` [rad]; none on the grippers."""
    n = len(ACTUATOR_NAMES)
    out = np.zeros((steps, n))
    if sigma <= 0:
        return out
    x = rng.normal(0, sigma, n)
    for t in range(steps):
        x = corr * x + np.sqrt(1 - corr**2) * rng.normal(0, sigma, n)
        out[t] = x
    out[:, GRIPPER_COLUMNS] = 0.0
    return out


def run_expert(env: OpenArmEnv, episode_seed: int, dart_sigma: float = 0.0,
               max_steps: int = MAX_EPISODE_STEPS, on_step=None) -> EpisodeResult:
    env.reset(episode_seed)
    expert = Expert(env.model, env.scene)
    noise = dart_noise(episode_rng(env.scene.scene_seed, episode_seed, DART_STREAM), max_steps, dart_sigma)
    actions, labels, phases = [], [], []
    stage, success, held_for, touch_steps = 0, False, 0, 0
    for t in range(max_steps):
        label, info = expert.act(env.data)
        action = np.clip(label + noise[t], *env.ctrl_range.T)
        env.step(action)
        actions.append(action)
        labels.append(label)
        phases.append(info.phase)
        st = env.privileged()
        stage = max(stage, stage_of(st, env.scene))
        touch_steps += st.arms_touch
        if on_step is not None:
            on_step(env, info)
        held_for = held_for + 1 if is_success(st, env.scene) else 0
        if held_for >= SUCCESS_HOLD_STEPS:
            success = True
            break
    return EpisodeResult(env.scene.scene_seed, episode_seed, success, len(actions), stage, touch_steps,
                         np.array(actions), np.array(labels), phases)

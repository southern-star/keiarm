"""Running episodes: expert rollouts with DART noise, action-chunk labels, replay."""

from dataclasses import dataclass, field

import numpy as np

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.expert import Expert
from keiarm.scene import episode_rng, sample_cube_xy
from keiarm.task import is_success, stage_of

DART_STREAM = 1
DISTURB_STREAM = 2
CHUNK = 30  # K: action chunk length (1 s at 30 Hz)
SUCCESS_HOLD_STEPS = 10  # the success condition must hold this many consecutive steps


@dataclass
class EpisodeResult:
    scene_seed: int
    episode_seed: int
    success: bool
    steps: int
    stage: int  # furthest stage reached (index into task.STAGES)
    actions: np.ndarray  # (T, 6) executed actions; with the seeds this replays the episode
    labels: np.ndarray  # (T, 6) clean expert actions for the visited states
    phases: list[str] = field(default_factory=list)


def dart_noise(rng: np.random.Generator, steps: int, sigma: float, corr: float = 0.9) -> np.ndarray:
    """Temporally correlated (AR(1)) joint-target noise with stationary std `sigma` [rad]."""
    out = np.zeros((steps, 6))
    if sigma <= 0:
        return out
    x = rng.normal(0, sigma, 6)
    for t in range(steps):
        x = corr * x + np.sqrt(1 - corr**2) * rng.normal(0, sigma, 6)
        out[t] = x
    out[:, 5] *= 0.0  # the gripper command is binary; perturbing it only causes drops
    return out


@dataclass(frozen=True)
class Teleport:
    """Disturbance: at control step `step` the cube is put down at (x, y) with the given yaw."""

    step: int
    x: float
    y: float
    yaw: float

    def apply(self, env: KeiArmEnv) -> None:
        env.teleport_cube([self.x, self.y], self.yaw)


DISTURB_WINDOW = (10, 60)  # "random" disturbances happen at a step in this range


def planned_teleport(env: KeiArmEnv, mode: int | str | None) -> Teleport | None:
    """The episode's disturbance, a pure function of its seeds. Call after env.reset().

    mode: None (no disturbance), an int (teleport at that step) or "random" (at a
    seed-determined step inside DISTURB_WINDOW).
    """
    if mode is None:
        return None
    ep = env.episode
    rng = episode_rng(ep.scene_seed, ep.episode_seed, DISTURB_STREAM)
    xy = sample_cube_xy(rng, env.scene, ep.plate_xy)
    yaw = rng.uniform(-np.pi, np.pi)
    step = int(rng.integers(*DISTURB_WINDOW)) if mode == "random" else int(mode)
    return Teleport(step, float(xy[0]), float(xy[1]), float(yaw))


def run_expert(env: KeiArmEnv, episode_seed: int, dart_sigma: float = 0.0,
               max_steps: int = config.MAX_EPISODE_STEPS, on_step=None,
               teleport_at: int | str | None = None) -> EpisodeResult:
    """Roll out the expert. Executed action = expert action + DART noise; labels stay clean.

    `teleport_at`: a step number or "random"; the cube is moved to a new spot on the table at
    that step (the browser demo's "drag the cube" disturbance). See planned_teleport.
    """
    env.reset(episode_seed)
    expert = Expert(env.model, env.scene)
    noise = dart_noise(episode_rng(env.scene.scene_seed, episode_seed, DART_STREAM), max_steps, dart_sigma)
    teleport = planned_teleport(env, teleport_at)
    actions, labels, phases = [], [], []
    stage, success, held_for = 0, False, 0
    for t in range(max_steps):
        if teleport is not None and t == teleport.step:
            teleport.apply(env)
        label, info = expert.act(env.data)
        action = np.clip(label + noise[t], *env.ctrl_range.T)
        env.step(action)
        actions.append(action)
        labels.append(label)
        phases.append(info.phase)
        st = env.privileged()
        stage = max(stage, stage_of(st, env.scene))
        if on_step is not None:
            on_step(env, info)
        held_for = held_for + 1 if is_success(st, env.scene) else 0
        if held_for >= SUCCESS_HOLD_STEPS:
            success = True
            break
    return EpisodeResult(env.scene.scene_seed, episode_seed, success, len(actions), stage,
                         np.array(actions), np.array(labels), phases)


def expert_chunk(env: KeiArmEnv, expert: Expert, length: int = CHUNK) -> np.ndarray:
    """The next `length` expert actions from the current state, (length, 6).

    Rolled out on a copy of the simulator state; `env.data` is left untouched.
    """
    data = env.scratch_copy()
    out = np.empty((length, 6))
    for k in range(length):
        out[k], _ = expert.act(data)
        env.step_data(data, out[k])
    return out


def replay(env: KeiArmEnv, episode_seed: int, actions: np.ndarray,
           teleports: tuple[Teleport, ...] = ()) -> np.ndarray:
    """Re-execute a recorded action sequence (and its disturbances); returns the final [qpos, qvel]."""
    env.reset(episode_seed)
    for t, a in enumerate(actions):
        for tp in teleports:
            if tp.step == t:
                tp.apply(env)
        env.step(a)
    return np.concatenate([env.data.qpos, env.data.qvel])

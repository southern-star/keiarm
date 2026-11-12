"""Policies the OpenArm evaluation harness can run (milestone 2; mirrors keiarm/policies.py).

A policy sees the environment through `act(env)`. Learned policies must only use
`env.observe()`; privileged ones (the expert and its variants) read the simulator state.
"""

from typing import Protocol

import numpy as np

from keiarm.openarm.env import MAX_EPISODE_STEPS, OpenArmEnv
from keiarm.openarm.expert import Expert
from keiarm.openarm.rollout import DART_STREAM, dart_noise
from keiarm.openarm.scene import episode_rng


class Policy(Protocol):
    needs_images: bool

    def reset(self, env: OpenArmEnv) -> None: ...

    def act(self, env: OpenArmEnv) -> np.ndarray: ...


class ExpertPolicy:
    """The privileged expert, optionally with DART noise (the same noise as run_expert)."""

    needs_images = False

    def __init__(self, noise_sigma: float = 0.0):
        self.noise_sigma = noise_sigma

    def reset(self, env: OpenArmEnv) -> None:
        self.expert = Expert(env.model, env.scene)
        rng = episode_rng(env.scene.scene_seed, env.episode.episode_seed, DART_STREAM)
        self.noise = dart_noise(rng, MAX_EPISODE_STEPS, self.noise_sigma)

    def act(self, env: OpenArmEnv) -> np.ndarray:
        action, _ = self.expert.act(env.data)
        return action + self.noise[env.steps % len(self.noise)]


class HoldPolicy:
    """Keeps both arms at their start pose. A do-nothing baseline for testing the harness."""

    needs_images = False

    def reset(self, env: OpenArmEnv) -> None:
        self.target = env.joints()

    def act(self, env: OpenArmEnv) -> np.ndarray:
        return self.target


class FreezePolicy(ExpertPolicy):
    """Expert that freezes its last action once the right hand starts a given phase
    (default CARRY: the bar is lifted but never handed over). A harness-testing baseline."""

    def __init__(self, phase: str = "CARRY"):
        super().__init__()
        self.phase = phase

    def reset(self, env: OpenArmEnv) -> None:
        super().reset(env)
        self.frozen = None

    def act(self, env: OpenArmEnv) -> np.ndarray:
        if self.frozen is None:
            action, info = self.expert.act(env.data)
            if info.phase.split("/")[1] == self.phase:
                self.frozen = action
            return action
        return self.frozen


def make_policy(spec: str) -> Policy:
    """'expert', 'expert-noisy:<sigma>', 'hold', 'freeze[:<right phase>]',
    'keigrasp:<run name or .pt>[:<steps executed per inference>]',
    'keigrasp-blind:<run>[:<cameras, comma-separated; default all>]' (control: black images)."""
    name, _, arg = spec.partition(":")
    if name in ("keigrasp", "keigrasp-blind"):
        from keiarm.openarm.env import CAMERAS
        from keigrasp.policy import KeiGraspPolicy  # deferred: needs torch

        run, _, rest = arg.partition(":")
        if name == "keigrasp":
            return KeiGraspPolicy(run, execute=int(rest or 10), cameras=CAMERAS)
        return KeiGraspPolicy(run, cameras=CAMERAS, blind=tuple(rest.split(",")) if rest else CAMERAS)
    if name == "expert":
        return ExpertPolicy()
    if name == "expert-noisy":
        return ExpertPolicy(float(arg or 0.0065))
    if name == "hold":
        return HoldPolicy()
    if name == "freeze":
        return FreezePolicy(arg or "CARRY")
    raise ValueError(f"unknown policy {spec!r}")

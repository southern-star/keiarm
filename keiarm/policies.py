"""Policies the evaluation harness can run.

A policy sees the environment through `act(env)`. Learned policies must only use
`env.observe()`; privileged ones (the expert and its variants) read the simulator state.
"""

from typing import Protocol

import numpy as np

from keiarm.env import KeiArmEnv
from keiarm.expert import Expert
from keiarm.rollout import DART_STREAM, dart_noise
from keiarm.scene import episode_rng


class Policy(Protocol):
    needs_images: bool

    def reset(self, env: KeiArmEnv) -> None: ...

    def act(self, env: KeiArmEnv) -> np.ndarray: ...


class ExpertPolicy:
    """The privileged scripted expert, optionally with DART-style action noise."""

    needs_images = False

    def __init__(self, noise_sigma: float = 0.0):
        self.noise_sigma = noise_sigma

    def reset(self, env: KeiArmEnv) -> None:
        self.expert = Expert(env.model, env.scene)
        rng = episode_rng(env.scene.scene_seed, env.episode.episode_seed, DART_STREAM)
        self.noise = dart_noise(rng, 4096, self.noise_sigma)

    def act(self, env: KeiArmEnv) -> np.ndarray:
        action, _ = self.expert.act(env.data)
        return action + self.noise[env.steps % len(self.noise)]


class HoldPolicy:
    """Keeps the arm at its start pose. A do-nothing baseline for testing the harness."""

    needs_images = False

    def reset(self, env: KeiArmEnv) -> None:
        self.target = env.joints()

    def act(self, env: KeiArmEnv) -> np.ndarray:
        return self.target


class GraspOnlyPolicy(ExpertPolicy):
    """Expert that freezes once it has lifted the cube. Another harness-testing baseline."""

    def reset(self, env: KeiArmEnv) -> None:
        super().reset(env)
        self.frozen = None

    def act(self, env: KeiArmEnv) -> np.ndarray:
        if self.frozen is None:
            action, info = self.expert.act(env.data)
            if info.phase == "MOVE":
                self.frozen = action
            return action
        return self.frozen


def make_policy(spec: str) -> Policy:
    """Build a policy from a spec string: 'expert', 'expert-noisy:0.02', 'hold', 'grasp-only',
    'keigrasp:<run>'."""
    name, _, arg = spec.partition(":")
    if name == "expert":
        return ExpertPolicy()
    if name == "expert-noisy":
        return ExpertPolicy(float(arg or 0.01))
    if name == "hold":
        return HoldPolicy()
    if name == "grasp-only":
        return GraspOnlyPolicy()
    if name == "keigrasp":
        # keigrasp:<run name or .pt path>[:<steps executed per inference>]
        from keigrasp.policy import KeiGraspPolicy  # deferred: needs torch

        ckpt, _, execute = arg.rpartition(":") if arg.rsplit(":", 1)[-1].isdigit() else (arg, "", "")
        return KeiGraspPolicy(ckpt, execute=int(execute)) if execute else KeiGraspPolicy(ckpt)
    raise ValueError(f"unknown policy spec: {spec!r}")

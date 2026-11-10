"""Closed-loop environment: 30 Hz joint-target actions, two camera images + joint angles out."""

from dataclasses import dataclass

import mujoco
import numpy as np

from keiarm import config
from keiarm.scene import (
    EpisodeParams,
    SceneParams,
    build_model,
    reset_data,
    sample_episode,
    sample_scene,
)
from keiarm.task import TaskIndex, TaskState, read_state


@dataclass
class Observation:
    """What the policy is allowed to see."""

    images: dict[str, np.ndarray]  # camera name -> uint8 (H, W, 3)
    joints: np.ndarray  # (6,) joint angles [rad] in JOINT_NAMES order


class KeiArmEnv:
    def __init__(self, scene_seed: int, render: bool = True,
                 width: int = config.IMAGE_WIDTH, height: int = config.IMAGE_HEIGHT):
        self.scene: SceneParams = sample_scene(scene_seed)
        self.model = build_model(self.scene)
        self.data = mujoco.MjData(self.model)
        self.episode: EpisodeParams | None = None
        self.steps = 0
        self.ix = TaskIndex(self.model)
        self._scratch: mujoco.MjData | None = None
        self._renderer = None
        if render:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            flags = self._renderer.scene.flags
            flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

    @property
    def ctrl_range(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self.ix.act]

    def reset(self, episode_seed: int) -> None:
        self.episode = sample_episode(self.scene, episode_seed)
        reset_data(self.model, self.data, self.scene, self.episode)
        self.steps = 0

    def step(self, action: np.ndarray) -> None:
        """Advance one control period (1/30 s) holding `action` as the joint position targets."""
        lo, hi = self.ctrl_range.T
        self.data.ctrl[self.ix.act] = np.clip(action, lo, hi)
        mujoco.mj_step(self.model, self.data, nstep=config.SUBSTEPS)
        self.steps += 1

    def joints(self) -> np.ndarray:
        return self.data.qpos[self.ix.qadr].copy()

    def render(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("environment was created with render=False")
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def observe(self) -> Observation:
        return Observation({cam: self.render(cam) for cam in config.CAMERAS}, self.joints())

    def privileged(self) -> TaskState:
        """Ground truth for the expert and the evaluator only."""
        return read_state(self.model, self.ix, self.data)

    def teleport_cube(self, xy, yaw: float = 0.0) -> None:
        """Disturbance: put the cube flat on the table at `xy`, at rest."""
        a, v = self.ix.cube_qadr, self.ix.cube_dadr
        self.data.qpos[a : a + 3] = [xy[0], xy[1], self.scene.cube_size / 2 + 1e-3]
        self.data.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[v : v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def scratch_copy(self) -> mujoco.MjData:
        """A reusable copy of the current state for look-ahead rollouts (same model)."""
        if self._scratch is None:
            self._scratch = mujoco.MjData(self.model)
        mujoco.mj_copyData(self._scratch, self.model, self.data)
        return self._scratch

    def step_data(self, data: mujoco.MjData, action: np.ndarray) -> None:
        """`step` applied to another MjData of the same model (does not count steps)."""
        lo, hi = self.ctrl_range.T
        data.ctrl[self.ix.act] = np.clip(action, lo, hi)
        mujoco.mj_step(self.model, data, nstep=config.SUBSTEPS)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

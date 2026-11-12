"""Closed-loop OpenArm environment: 30 Hz, 16 joint targets in; three images + 16 joint angles out."""

from dataclasses import dataclass

import mujoco
import numpy as np

from keiarm import config
from keiarm.openarm.robot import wrist_camera
from keiarm.openarm.scene import (
    PLATE_HEIGHT,
    EpisodeParams,
    SceneParams,
    build_model,
    reset_data,
    sample_episode,
    sample_scene,
    set_bar_pose,
)
from keiarm.openarm.task import TaskIndex, TaskState, read_state

CAMERAS = ("cam_front", wrist_camera("left"), wrist_camera("right"))  # order of the policy's images
EPISODE_SECONDS = 30
MAX_EPISODE_STEPS = EPISODE_SECONDS * config.CONTROL_HZ


@dataclass
class Observation:
    """What the policy is allowed to see."""

    images: dict[str, np.ndarray]  # camera name -> uint8 (H, W, 3)
    joints: np.ndarray  # (16,) [rad], ACTUATOR_NAMES order (finger joint 1 for grippers)


class OpenArmEnv:
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
        """Advance one control period (1/30 s) holding `action` as the 16 position targets."""
        self.step_data(self.data, action)
        self.steps += 1

    def step_data(self, data: mujoco.MjData, action: np.ndarray) -> None:
        """`step` applied to another MjData of the same model (does not count steps)."""
        lo, hi = self.ctrl_range.T
        data.ctrl[self.ix.act] = np.clip(action, lo, hi)
        mujoco.mj_step(self.model, data, nstep=config.SUBSTEPS)

    def joints(self) -> np.ndarray:
        return self.data.qpos[self.ix.trn_qadr].copy()

    def render(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("environment was created with render=False")
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def observe(self) -> Observation:
        return Observation({cam: self.render(cam) for cam in CAMERAS}, self.joints())

    def privileged(self) -> TaskState:
        """Ground truth for the expert and the evaluator only."""
        return read_state(self.model, self.ix, self.data)

    def teleport_bar(self, xy, yaw: float) -> None:
        """Disturbance: put the bar flat on the table at `xy`, at rest."""
        set_bar_pose(self.model, self.data, xy, yaw, self.scene.bar_size[2] / 2 + 1e-3)
        mujoco.mj_forward(self.model, self.data)

    def move_plate(self, xy) -> None:
        """Disturbance: move the plate (a mocap body) to `xy` on the table."""
        self.data.mocap_pos[self.ix.plate_mocap] = [float(xy[0]), float(xy[1]), PLATE_HEIGHT / 2]
        mujoco.mj_forward(self.model, self.data)

    def scratch_copy(self) -> mujoco.MjData:
        """A reusable copy of the current state for look-ahead rollouts (same model)."""
        if self._scratch is None:
            self._scratch = mujoco.MjData(self.model)
        mujoco.mj_copyData(self._scratch, self.model, self.data)
        return self._scratch

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

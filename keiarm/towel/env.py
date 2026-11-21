"""Closed-loop towel environment: 30 Hz, 16 joint targets in; three images + 16 joint angles out.

Same interface as keiarm/openarm/env.py, plus the grasp rule (keiarm/towel/grasp.py), which
runs after every control step. Episode state = MjData state + grasp state (get_state /
set_state), because the grasp rule rewrites the model's constraint slots.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from keiarm import config
from keiarm.openarm.robot import ACTUATOR_NAMES, wrist_camera
from keiarm.towel.grasp import GraspRule, GraspState, HandState
from keiarm.towel.render import BacksideDrawer
from keiarm.towel.scene import (
    TOWEL_N,
    EpisodeParams,
    SceneParams,
    TowelIndex,
    build_model,
    reset_data,
    sample_episode,
    sample_scene,
)

CAMERAS = ("cam_front", wrist_camera("left"), wrist_camera("right"))  # order of the policy's images
EPISODE_SECONDS = 20
MAX_EPISODE_STEPS = EPISODE_SECONDS * config.CONTROL_HZ
_STATE = mujoco.mjtState.mjSTATE_INTEGRATION  # includes eq_active


@dataclass
class Observation:
    images: dict[str, np.ndarray]  # camera name -> uint8 (H, W, 3)
    joints: np.ndarray  # (16,) [rad], ACTUATOR_NAMES order (finger joint 1 for grippers)


@dataclass
class EpisodeState:
    physics: np.ndarray  # mj_getState(mjSTATE_INTEGRATION)
    grasp: GraspState
    steps: int


class TowelEnv:
    def __init__(self, scene_seed: int, render: bool = True, n: int = TOWEL_N,
                 width: int = config.IMAGE_WIDTH, height: int = config.IMAGE_HEIGHT):
        self.scene: SceneParams = sample_scene(scene_seed)
        self.model = build_model(self.scene, n)
        self.data = mujoco.MjData(self.model)
        self.towel = TowelIndex(self.model)
        self.grasp = GraspRule(self.model, self.towel)
        self.episode: EpisodeParams | None = None
        self.steps = 0
        m = self.model
        self.act = np.array([m.actuator(a).id for a in ACTUATOR_NAMES])
        # joint read back per actuator: the one it drives (finger joint 1 for the grippers)
        self.trn_qadr = m.jnt_qposadr[m.actuator_trnid[self.act, 0]]
        self._renderer = None
        self._backside = BacksideDrawer(m, self.towel, self.scene.towel_back_rgb)
        if render:
            self._renderer = mujoco.Renderer(m, height=height, width=width)
            flags = self._renderer.scene.flags
            flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

    @property
    def ctrl_range(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self.act]

    def reset(self, episode_seed: int) -> None:
        self.episode = sample_episode(self.scene, episode_seed)
        reset_data(self.model, self.data, self.towel, self.episode)
        self.grasp.reset(self.data)
        self.steps = 0

    def step(self, action: np.ndarray) -> list[tuple[str, str]]:
        """Advance one control period holding `action` as the 16 position targets, then apply
        the grasp rule. Returns the grasp events of this step."""
        lo, hi = self.ctrl_range.T
        self.data.ctrl[self.act] = np.clip(action, lo, hi)
        mujoco.mj_step(self.model, self.data, nstep=config.SUBSTEPS)
        self.steps += 1
        return self.grasp.update(self.data)

    # --- disturbances (keiarm/towel/eval.py) -----------------------------------------------

    def place_towel(self, xy, yaw: float) -> None:
        """Lay the towel flat at (xy, yaw), at rest. Only for a towel no hand holds."""
        self.towel.place(self.data, xy, yaw)
        mujoco.mj_forward(self.model, self.data)

    def slip(self, sides) -> None:
        """The towel slips out of these hands: their grab constraints go, the fingers stay
        closed and latched (they pick nothing up until they open, as after any close)."""
        for s in sides:
            self.grasp.state.hands[s] = HandState(latched=True)
        self.grasp.apply(self.data)
        mujoco.mj_forward(self.model, self.data)

    def kick(self, verts, velocity) -> None:
        """Add (len(verts), 3) velocities to these towel vertices."""
        self.data.qvel[self.towel.vert_dadr[np.asarray(verts)]] += np.asarray(velocity)
        mujoco.mj_forward(self.model, self.data)

    def joints(self) -> np.ndarray:
        return self.data.qpos[self.trn_qadr].copy()

    def towel_verts(self) -> np.ndarray:
        """Privileged: towel vertex positions (copy)."""
        return self.towel.verts(self.data).copy()

    def _update_scene(self, camera) -> mujoco.Renderer:
        if self._renderer is None:
            raise RuntimeError("environment was created with render=False")
        self._renderer.update_scene(self.data, camera=camera)
        self._backside.add(self._renderer.scene, self.data)
        return self._renderer

    def render(self, camera) -> np.ndarray:
        """RGB image (front checker, plain back). `camera` is a name or an MjvCamera."""
        return self._update_scene(camera).render()

    def towel_mask(self, camera) -> np.ndarray:
        """Privileged: boolean (H, W) mask of towel pixels (either side) from segmentation."""
        r = self._renderer
        r.enable_segmentation_rendering()
        try:
            seg = self._update_scene(camera).render()
        finally:
            r.disable_segmentation_rendering()
        return (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_FLEX)) & (seg[..., 0] == self.towel.flex)

    def observe(self) -> Observation:
        return Observation({cam: self.render(cam) for cam in CAMERAS}, self.joints())

    def get_state(self) -> EpisodeState:
        size = mujoco.mj_stateSize(self.model, _STATE)
        phys = np.empty(size)
        mujoco.mj_getState(self.model, self.data, phys, _STATE)
        return EpisodeState(phys, self.grasp.get_state(), self.steps)

    def set_state(self, state: EpisodeState) -> None:
        mujoco.mj_setState(self.model, self.data, state.physics, _STATE)
        self.grasp.set_state(self.data, state.grasp)
        self.steps = state.steps
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

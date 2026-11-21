"""Task "fold the towel in half": state readout, success, stages (milestone 3, Phase 2).

Privileged simulator state only. The policy never sees it. Everything is read from the
model and `data` (which hand holds which vertices is in the active grab slots, which the
grasp rule writes), so the expert and the success check work on any restored state.

Success (docs/milestone3.md), held for SUCCESS_HOLD_STEPS control steps:
  1. each far corner within CORNER_H horizontally and CORNER_V vertically of its near corner
  2. area seen from above (convex hull) 0.40-0.60 of the flat towel's
  3. layer order violations at most 10%
  4. neither hand holds the towel, no finger touches it
  5. no towel vertex faster than 5 cm/s
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from keiarm.openarm.robot import PINCH_OFFSET, SIDES, ee_site, finger_joints
from keiarm.towel import metrics
from keiarm.towel.grasp import FINGERTIP_OFFSET
from keiarm.towel.scene import GRAB_SLOTS, SceneParams, TowelIndex

STAGES = ("none", "reach", "grasp_one", "grasp_both", "lift", "carry", "placed", "released", "success")
CORNER_H = 0.03  # [m]
CORNER_V = 0.025
AREA_RANGE = (0.40, 0.60)
MAX_VIOLATION = 0.10
MAX_SPEED = 0.05  # [m/s]
SUCCESS_HOLD_STEPS = 10
STAGE_REACH = 0.05  # a fingertip this close to its far corner
STAGE_LIFT = 0.03  # a held far corner this high
STAGE_CARRY = 0.06  # held far corners this close (horizontally) to their near corners


class TaskIndex:
    """Model addresses needed to read the task state."""

    def __init__(self, model: mujoco.MjModel):
        self.towel = TowelIndex(model)
        self.slots = {s: np.array([model.equality(f"{s}_grab{k}").id for k in range(GRAB_SLOTS)]) for s in SIDES}
        self.body_to_vert = {int(b): k for k, b in enumerate(self.towel.vert_body)}
        self.site = {s: model.site(ee_site(s)).id for s in SIDES}
        self.finger_qadr = {s: model.joint(finger_joints(s)[0]).qposadr[0] for s in SIDES}
        self.finger_geoms = {s: {g for g in range(model.ngeom)
                                 if model.body(model.geom_bodyid[g]).name.startswith(f"openarm_{s}_ee_")
                                 and model.body(model.geom_bodyid[g]).name.endswith("_finger")} for s in SIDES}
        self.arm_geoms = {s: {g for g in range(model.ngeom)
                              if model.body(model.geom_bodyid[g]).name.startswith(f"openarm_{s}_")
                              and model.geom_contype[g]} for s in SIDES}
        n = self.towel.n
        self.far = {s: metrics.corner(self.towel, metrics.FAR_CORNER[s]) for s in SIDES}
        self.near = {s: metrics.corner(self.towel, metrics.NEAR_CORNER[s]) for s in SIDES}
        # The pair across the crease is left out: with the crease half a cell off (the far
        # edge put down 5 mm further), the row next to it is the crease's top and lies above
        # its partner, a whole row (1/8 of the pairs at N = 16) of "violations" on a good fold.
        self.layers = metrics.half_fold_layers(self.towel, skip_crease=True)
        self.n = n


@dataclass
class TowelState:
    verts: np.ndarray  # (nvert, 3)
    vert_vel: np.ndarray  # (nvert, 3) [m/s]
    vert_speed: np.ndarray  # (nvert,) [m/s]
    held: dict[str, list[int]]  # vertices each hand holds
    pinch_pos: dict[str, np.ndarray]
    pinch_mat: dict[str, np.ndarray]
    fingertip: dict[str, np.ndarray]
    finger_q: dict[str, float]
    finger_touch: dict[str, bool]  # a finger of that hand touches the towel
    arms_touch: bool  # a left-arm geom touches a right-arm geom

    def holds(self, side: str) -> bool:
        return bool(self.held[side])


def read_state(model: mujoco.MjModel, ix: TaskIndex, data: mujoco.MjData) -> TowelState:
    """Read the task state from `data` without modifying it."""
    tw = ix.towel
    verts = tw.verts(data).copy()
    vel = data.qvel[tw.vert_dadr].copy()
    speed = np.linalg.norm(vel, axis=1)
    held = {s: [ix.body_to_vert[int(model.eq_obj2id[k])] for k in ix.slots[s] if data.eq_active[k]] for s in SIDES}
    touch = dict.fromkeys(SIDES, False)
    arms_touch = False
    for c in data.contact[: data.ncon]:
        if c.dist > 0:
            continue
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        if c.flex[0] == tw.flex or c.flex[1] == tw.flex:
            g = g1 if g1 >= 0 else g2
            for s in SIDES:
                touch[s] |= g in ix.finger_geoms[s]
        elif (g1 in ix.arm_geoms["left"] and g2 in ix.arm_geoms["right"]) or (
                g2 in ix.arm_geoms["left"] and g1 in ix.arm_geoms["right"]):
            arms_touch = True
    offset = np.array(PINCH_OFFSET)
    pinch_pos, pinch_mat, tip = {}, {}, {}
    for s in SIDES:
        rot = data.site_xmat[ix.site[s]].reshape(3, 3).copy()
        pinch_mat[s] = rot
        pinch_pos[s] = data.site_xpos[ix.site[s]] + rot @ offset
        tip[s] = data.site_xpos[ix.site[s]] + rot @ FINGERTIP_OFFSET
    return TowelState(verts, vel, speed, held, pinch_pos, pinch_mat, tip,
                      {s: float(data.qpos[ix.finger_qadr[s]]) for s in SIDES}, touch, arms_touch)


@dataclass
class FoldCheck:
    corner_gap: dict[str, tuple[float, float]]  # side -> (horizontal, vertical) [m]
    area_ratio: float
    violation: float
    max_speed: float
    free: bool  # no hand holds the towel, no finger touches it

    @property
    def success(self) -> bool:
        return (all(h < CORNER_H and v < CORNER_V for h, v in self.corner_gap.values())
                and AREA_RANGE[0] <= self.area_ratio <= AREA_RANGE[1] and self.violation <= MAX_VIOLATION
                and self.free and self.max_speed <= MAX_SPEED)


def check_fold(st: TowelState, ix: TaskIndex, scene: SceneParams) -> FoldCheck:
    v = st.verts
    gaps = {}
    for s in SIDES:
        d = v[ix.far[s]] - v[ix.near[s]]
        gaps[s] = (float(np.hypot(d[0], d[1])), float(abs(d[2])))
    area0 = scene.towel_size[0] * scene.towel_size[1]
    return FoldCheck(gaps, metrics.convex_area(v) / area0, metrics.layer_violation(v, ix.layers),
                     float(st.vert_speed.max()),
                     not any(st.held[s] or st.finger_touch[s] for s in SIDES))


def stage_of(st: TowelState, ix: TaskIndex, scene: SceneParams) -> int:
    """Instantaneous progress stage (index into STAGES). Track the max over an episode."""
    chk = check_fold(st, ix, scene)
    if chk.success:
        return 8
    placed = all(h < CORNER_H and v < CORNER_V for h, v in chk.corner_gap.values())
    if placed and not any(st.held[s] for s in SIDES):
        return 7
    if placed:
        return 6
    holds = {s: st.holds(s) for s in SIDES}
    if all(holds.values()):
        if all(chk.corner_gap[s][0] < STAGE_CARRY for s in SIDES):
            return 5
        if all(st.verts[ix.far[s], 2] > STAGE_LIFT for s in SIDES):
            return 4
        return 3
    if any(holds.values()):
        return 2
    if any(np.linalg.norm(st.fingertip[s] - st.verts[ix.far[s]]) < STAGE_REACH for s in SIDES):
        return 1
    return 0

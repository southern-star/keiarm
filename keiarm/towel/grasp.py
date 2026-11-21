"""The grasp rule: closing a gripper attaches nearby towel vertices to that hand.

A simplification of pinching cloth between the fingers (docs/milestone3.md, spike E3). It is
part of the environment, not of the policy: the policy only commands the grippers.

- Close: on the control step where a hand's gripper command is past CLOSE_CMD (towards
  closed) and its fingers are closed past CLOSE_OPENING, the hand grabs up to GRAB_SLOTS
  vertices within GRAB_RADIUS of its fingertip midpoint, nearest first. Each is held by a
  `connect` constraint at its offset from the hand at that instant (no jump).
- The hand is then "latched": it adds no vertices until it opens, even if nothing was in
  reach (a closed gripper cannot pick anything up).
- Open: when the command goes past OPEN_CMD (towards open), the hand lets go and unlatches.
- A vertex held by the other hand is taken over by the hand that grabs later.

The constraint slots are compiled in (keiarm/towel/scene.py) and inactive; grabbing rewrites
their body2 (`model.eq_obj2id`), anchor (`model.eq_data`) and `data.eq_active`. Because the
model is modified, the rule's state (`GraspState`) is part of the episode state: restoring an
MjData snapshot must go with `GraspRule.set_state`.

The same rule is mirrored in the browser (Phase 4).
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from keiarm.openarm.robot import GRIPPER_OPEN, SIDES, ee_site, finger_joints
from keiarm.towel.scene import GRAB_SLOTS, TowelIndex

GRAB_RADIUS = 0.02  # [m] from the fingertip midpoint
# Fingertip midpoint in the ee site frame (the pads' pinch point is at z = -0.150 and the
# fingertips end at -0.170).
FINGERTIP_OFFSET = np.array([0.0, 0.0, -0.165])
# Gripper command and finger opening as a fraction of fully open (1 = open, 0 = jaws
# touching, negative = squeezing).
CLOSE_CMD = 0.25
CLOSE_OPENING = 0.35
OPEN_CMD = 0.5


@dataclass
class HandState:
    latched: bool = False
    verts: list[int] = field(default_factory=list)  # held towel vertices
    anchors: list[list[float]] = field(default_factory=list)  # their offsets in the hand frame

    def copy(self) -> "HandState":
        return HandState(self.latched, list(self.verts), [list(a) for a in self.anchors])


@dataclass
class GraspState:
    hands: dict[str, HandState] = field(default_factory=lambda: {s: HandState() for s in SIDES})

    def copy(self) -> "GraspState":
        return GraspState({s: h.copy() for s, h in self.hands.items()})

    def to_dict(self) -> dict:
        return {s: {"latched": h.latched, "verts": list(h.verts), "anchors": [list(a) for a in h.anchors]}
                for s, h in self.hands.items()}

    @staticmethod
    def from_dict(d: dict) -> "GraspState":
        return GraspState({s: HandState(bool(h["latched"]), [int(v) for v in h["verts"]],
                                        [list(map(float, a)) for a in h["anchors"]]) for s, h in d.items()})


class GraspRule:
    def __init__(self, model: mujoco.MjModel, ix: TowelIndex):
        self.model = model
        self.ix = ix
        self.slots = {s: [model.equality(f"{s}_grab{k}").id for k in range(GRAB_SLOTS)] for s in SIDES}
        self.hand = {s: model.body(f"openarm_{s}_ee_base_link").id for s in SIDES}
        self.site = {s: model.site(ee_site(s)).id for s in SIDES}
        self.grip_act = {s: model.actuator(f"{s}_finger1_ctrl").id for s in SIDES}
        self.finger_qadr = {s: model.joint(finger_joints(s)[0]).qposadr[0] for s in SIDES}
        self.state = GraspState()

    # --- queries ---------------------------------------------------------------------------

    def fingertip(self, data: mujoco.MjData, side: str) -> np.ndarray:
        rot = data.site_xmat[self.site[side]].reshape(3, 3)
        return data.site_xpos[self.site[side]] + rot @ FINGERTIP_OFFSET

    def command(self, data: mujoco.MjData, side: str) -> float:
        return float(data.ctrl[self.grip_act[side]] / GRIPPER_OPEN[side])

    def opening(self, data: mujoco.MjData, side: str) -> float:
        return float(data.qpos[self.finger_qadr[side]] / GRIPPER_OPEN[side])

    def held(self, side: str) -> list[int]:
        return list(self.state.hands[side].verts)

    def anchor_error(self, data: mujoco.MjData) -> float:
        """Largest distance between a held vertex and its anchor on the hand [m]."""
        err = 0.0
        for s in SIDES:
            h = self.state.hands[s]
            pos, rot = data.xpos[self.hand[s]], data.xmat[self.hand[s]].reshape(3, 3)
            for v, a in zip(h.verts, h.anchors):
                err = max(err, float(np.linalg.norm(pos + rot @ np.asarray(a) - self.ix.verts(data)[v])))
        return err

    # --- rule --------------------------------------------------------------------------------

    def reset(self, data: mujoco.MjData) -> None:
        self.state = GraspState()
        self.apply(data)

    def update(self, data: mujoco.MjData) -> list[tuple[str, str]]:
        """Apply the rule after a control step. Returns the events [(side, 'grab'|'release')]."""
        events = []
        for s in SIDES:
            h = self.state.hands[s]
            cmd = self.command(data, s)
            if h.latched:
                if cmd > OPEN_CMD:
                    self.state.hands[s] = HandState()
                    events.append((s, "release"))
            elif cmd < CLOSE_CMD and self.opening(data, s) < CLOSE_OPENING:
                self._grab(data, s)
                events.append((s, "grab"))
        if events:
            self.apply(data)
        return events

    def _grab(self, data: mujoco.MjData, side: str) -> None:
        verts = self.ix.verts(data)
        d = np.linalg.norm(verts - self.fingertip(data, side), axis=1)
        order = [int(k) for k in np.argsort(d, kind="stable")[:GRAB_SLOTS] if d[k] < GRAB_RADIUS]
        pos, rot = data.xpos[self.hand[side]], data.xmat[self.hand[side]].reshape(3, 3)
        anchors = [list(map(float, rot.T @ (verts[k] - pos))) for k in order]
        self.state.hands[side] = HandState(True, order, anchors)
        other = self.state.hands[next(s for s in SIDES if s != side)]
        keep = [(v, a) for v, a in zip(other.verts, other.anchors) if v not in order]
        other.verts, other.anchors = [v for v, _ in keep], [a for _, a in keep]

    def apply(self, data: mujoco.MjData) -> None:
        """Write the state into the model's constraint slots and data.eq_active."""
        m = self.model
        for s in SIDES:
            h = self.state.hands[s]
            for k, slot in enumerate(self.slots[s]):
                if k < len(h.verts):
                    m.eq_obj2id[slot] = self.ix.vert_body[h.verts[k]]
                    m.eq_data[slot, 0:3] = h.anchors[k]
                    m.eq_data[slot, 3:6] = 0.0
                    data.eq_active[slot] = 1
                else:
                    data.eq_active[slot] = 0

    def get_state(self) -> GraspState:
        return self.state.copy()

    def set_state(self, data: mujoco.MjData, state: GraspState) -> None:
        self.state = state.copy()
        self.apply(data)

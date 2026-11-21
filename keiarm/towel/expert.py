"""Privileged expert for the towel half fold (milestone 3, Phase 2).

Like the milestone 1 and 2 experts, `act(data)` is a pure function of the simulator state:
it reads the towel's vertices, which vertices each hand holds (the active grab slots) and
the fingers, decides each hand's phase and returns the 16 joint targets. No memory between
calls (DAgger labels arbitrary states; drops and slips are recovered by construction), and
`data` is not modified.

Per hand s (right: -y far corner, left: +y far corner), checked in order:

1. Holds the towel and its far corner is placed (next to the target), and so is the other
   hand's (or the other hand has let go of a placed corner) -> RELEASE: open, start lifting.
2. Holds the towel -> CARRY: servo the far corner along an arc to the target (the near
   corner + PLACE_SHORT along the fold), if the other hand also holds the towel or is done;
   otherwise WAIT at the arc point for the current distance (the other hand re-grasps).
   Both hands share one progress along the arc, set by the hand further behind.
3. Lets go and its far corner is at the target -> RISE (straight up while near the towel),
   then HOME (the start pinch point).
4. Otherwise -> grasp the far corner wherever it is: open if needed, funnel down onto the
   point GRASP_INSET inside the corner, close. A close that caught nothing opens again.

The geometry (jaw yaw, opening, place point, lift on release) is the scripted fold's, whose
reach over the whole spawn range is checked by tools/towel_reach.py. Targets are carrots,
and heights are absolute or relative to the towel, never "the current height".
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from keiarm.openarm.ik import ArmIK, topdown_rotation
from keiarm.openarm.robot import (
    ACTUATOR_NAMES,
    ARM_DOF,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_ARM_QPOS,
    HOME_PINCH,
    HOME_YAW,
    SIDES,
)
from keiarm.towel.grasp import CLOSE_CMD
from keiarm.towel.scene import TOWEL_RADIUS, SceneParams
from keiarm.towel.scripted import (
    GRASP_INSET,
    LIFT_RATIO,
    MIN_PINCH_Z,
    OPENED,
    PLACE_SHORT,
    RELEASE_LIFT,
    YAW_OFFSET,
)
from keiarm.towel.task import TaskIndex, TowelState, read_state

PLACE_PINCH_Z = MIN_PINCH_Z + 0.008  # pinch height when putting the corner down (as scripted)
# Pinch height when grasping. At MIN_PINCH_Z (fingertips 4 mm above the table) the hand
# settles 1-2 mm lower than commanded, the finger ends reach the cloth and close on it: the
# cloth just inside the grasp then rides between the fingers for the whole carry (~20
# contacts per step) and drags the middle of a stiff towel towards the robot. The scripted
# fold's hand settles 2 mm higher (its carrot stops at MIN_PINCH_Z and the arm lags).
GRASP_PINCH_Z = MIN_PINCH_Z + 0.003
GRASP_TOP_RADIUS = 0.015  # [m] cloth within this of the grasp point sets the grasp height
HOVER = 0.10  # pinch height above the grasp point while approaching
FLAT_Z = 0.008  # a towel with no vertex above this ...
FLAT_RMS = 0.004  # ... and within this of a rigid copy of its rest shape lies flat [m]
DEADBAND = 0.002  # horizontal grasp error that needs no extra funnel height [m]
FUNNEL = 3.0  # funnel height per metre of error beyond the deadband
# The jaws close within these errors. The grasp rule takes the 4 vertices nearest the
# fingertips; 1 cm inside the corner, the corner vertex is the 4th nearest with ~6 mm to
# spare, and closing 3.5 mm off (with CLOSE_XY = 6 mm) left it out: the free corner then
# rode between the fingers (~40 finger-cloth contacts per step) and dragged the towel.
CLOSE_XY, CLOSE_Z = 0.003, 0.002
# Once closing (the previous gripper command, data.ctrl, is closed), keep closing while the
# hand stays this close: the fingers take ~10 steps to close, and with DART noise (sigma
# 0.0065) the hand left the CLOSE window for a step, the command flipped back to open and the
# close never completed (the hand hovered and re-descended for the whole episode).
CLOSING_XY, CLOSING_Z = 0.006, 0.005
OPEN_ENOUGH = 0.5  # finger opening (fraction of fully open) that counts as open
MISSED = 0.3  # a hand that holds nothing with its fingers closed past this has missed
# The carry advances the *commanded* pose, not the actual one: the previous joint targets
# (data.ctrl, part of the state) give the commanded pinch point, the target moves CARRY_STEP
# further along the arc from there, and IK starts from the commanded joints. This is the
# scripted fold's carrot, read back from the state. Carrots taken from the actual pose
# (as in the other phases) did not control the speed here: the low-gain wrist joints lag,
# the hand ran ahead at up to 7 mm per step and overshot the place point by 3-4 cm,
# dragging the towel's middle along. If the command is more than RESYNC from the hand (a
# disturbance, a DAgger state), the carry restarts from the actual pose.
# [m] per control step along the arc at its middle; the step eases in and out with
# sin(pi * progress) (at least CARRY_MIN_FRAC of it), as the scripted fold's cosine timing.
# At a constant step the jaws lifted the corner off quickly and the cloth just inside the
# grasp stayed against the fingers for the whole carry (~20 contacts per step), dragging
# the middle of a stiff towel 4 cm towards the robot.
CARRY_STEP = 0.003
CARRY_MIN_FRAC = 0.33
# IK position tolerance [m]. ArmIK's default (0.2 mm) stops short of sub-millimetre steps,
# and with the carry continuing from the commanded pose the carry crept to a halt 9 mm
# short of the place point.
IK_TOL = 2e-5
RESYNC = 0.03
PLACED_XY, PLACED_Z = 0.006, 0.006  # far corner vs target, pinch vs PLACE_PINCH_Z: put down
# A let-go far corner this close to its near corner (horizontally, inside the success
# threshold of 3 cm) and at most this high above it is done. Measured against the target
# (1 cm past the near corner) with 1.5 cm, corners that landed beside it were folded again
# and again (Phase 2, 4 of 100 episodes).
DONE_XY, DONE_Z = 0.025, 0.020  # a curled corner can rest 16 mm up (success allows 25)
RELEASE_OPEN = 0.3  # after letting go, the hand keeps to the release lift until the fingers open this far
RELEASE_STEP = 0.006  # [m] per control step while letting go (the scripted fold's carrot)
SETTLED_SPEED = 0.03  # [m/s] far corner speed below which a put-down corner counts as at rest
RISE_Z = 0.08  # after letting go, rise to this height above the towel before going home
STEP_FAST = 0.008  # [m] per control step
STEP_SLOW = 0.005
YAW_STEP = np.radians(4.0)

PHASES = ("OPEN", "APPROACH", "DESCEND", "CLOSE", "MISSED", "WAIT", "CARRY", "RELEASE", "RISE", "HOME")


@dataclass
class ArmGoal:
    phase: str
    pos: np.ndarray  # pinch point goal (already a carrot)
    yaw: float
    gripper: float
    q_start: np.ndarray | None = None  # IK start (default: the actual arm joints)


@dataclass
class ExpertInfo:
    phase: str  # "<left phase>/<right phase>"
    goals: dict[str, ArmGoal]


def jaw_yaw(rot: np.ndarray) -> float:
    return float(np.arctan2(-rot[0, 1], rot[1, 1]))


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _carrot(pos: np.ndarray, goal: np.ndarray, step: float) -> np.ndarray:
    d = goal - pos
    n = float(np.linalg.norm(d))
    return goal.copy() if n <= step else pos + d * (step / n)


def fold_yaw(side: str) -> float:
    return HOME_YAW[side] + YAW_OFFSET[side]


class Expert:
    def __init__(self, model: mujoco.MjModel, scene: SceneParams):
        self.model = model
        self.scene = scene
        self.ix = TaskIndex(model)
        tw = self.ix.towel
        n = tw.n
        # the vertex diagonally inside each far corner: the grasp point lies towards it
        self.inner = {s: tw.vid(n - 2, 1 if s == "right" else n - 2) for s in SIDES}
        self.ik = {s: ArmIK(model, s) for s in SIDES}
        self.act_arm = {s: [ACTUATOR_NAMES.index(f"{s}_joint{i}_ctrl") for i in range(1, ARM_DOF + 1)] for s in SIDES}
        self.act_grip = {s: ACTUATOR_NAMES.index(f"{s}_finger1_ctrl") for s in SIDES}
        self.q_rest = {s: np.array(HOME_ARM_QPOS[s]) for s in SIDES}
        self.ctrl_arm = {s: np.array([model.actuator(f"{s}_joint{i}_ctrl").id for i in range(1, ARM_DOF + 1)])
                         for s in SIDES}
        self.ctrl_grip = {s: model.actuator(f"{s}_finger1_ctrl").id for s in SIDES}
        self._fk = mujoco.MjData(model)
        self._cmd = None  # set during act (see act)
        self.depth = scene.towel_size[0] - PLACE_SHORT  # far corner to target, flat towel
        self.lift = LIFT_RATIO * self.depth / 2

    # --- geometry --------------------------------------------------------------------------

    def _fold_axis(self, st: TowelState) -> np.ndarray:
        """Horizontal unit vector from the near edge towards the far edge (towel local +x),
        from the near corners (which stay on the table)."""
        w = st.verts[self.ix.near["left"], :2] - st.verts[self.ix.near["right"], :2]
        w /= np.linalg.norm(w) + 1e-12
        return np.array([w[1], -w[0]])

    def _target(self, st: TowelState, side: str) -> np.ndarray:
        """Where the far corner goes (horizontal): just past the near corner along the fold."""
        return st.verts[self.ix.near[side], :2] + PLACE_SHORT * self._fold_axis(st)

    def _corner_error(self, st: TowelState, side: str) -> float:
        return float(np.linalg.norm(st.verts[self.ix.far[side], :2] - self._target(st, side)))

    def _flat_pose(self, st: TowelState) -> tuple[np.ndarray, np.ndarray] | None:
        """(R, t) with verts ~ R @ rest + t if the towel lies flat (rigid 2D fit to the rest
        shape, residual below FLAT_RMS), else None."""
        rest, cur = self.ix.towel.rest[:, :2], st.verts[:, :2]
        if st.verts[:, 2].max() > FLAT_Z:
            return None
        mr, mc = rest.mean(axis=0), cur.mean(axis=0)
        u, _, vt = np.linalg.svd((cur - mc).T @ (rest - mr))
        rot = u @ np.diag([1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        t = mc - rot @ mr
        if np.sqrt(np.mean(np.sum((rest @ rot.T + t - cur) ** 2, axis=1))) > FLAT_RMS:
            return None
        return rot, t

    def _grasp_point(self, st: TowelState, side: str) -> np.ndarray:
        """GRASP_INSET inside the far corner, towards the diagonal neighbour. On a flat towel
        from the towel's fitted pose: the descending fingers nudge the corner vertex a few
        mm, and chasing it put the jaws 3 mm off, where the grasp rule left the corner out."""
        c = st.verts[self.ix.far[side]]
        pose = self._flat_pose(st)
        if pose is not None:
            rest = self.ix.towel.rest
            corner, inner = rest[self.ix.far[side], :2], rest[self.inner[side], :2]
            local = corner + GRASP_INSET * (inner - corner) / np.linalg.norm(inner - corner)
            xy = pose[0] @ local + pose[1]
        else:
            inward = st.verts[self.inner[side], :2] - c[:2]
            xy = c[:2] + GRASP_INSET * inward / (np.linalg.norm(inward) + 1e-12)
        # Height from the highest cloth under the jaws, not from the corner: a corner that
        # fell and curled under (slip, drop) lies below a ridge ~1 cm high, and descending
        # to the corner's height pressed the ridge through the curled edge (Phase 3).
        near = np.linalg.norm(st.verts[:, :2] - xy, axis=1) < GRASP_TOP_RADIUS
        top = float(st.verts[near, 2].max()) if near.any() else c[2]
        return np.array([*xy, max(GRASP_PINCH_Z, top - TOWEL_RADIUS + GRASP_PINCH_Z)])

    def _goal(self, st: TowelState, side: str, phase: str, pos_goal, gripper: float, step: float) -> ArmGoal:
        now = jaw_yaw(st.pinch_mat[side])
        yaw = now + float(np.clip(wrap(fold_yaw(side) - now), -YAW_STEP, YAW_STEP))
        return ArmGoal(phase, _carrot(st.pinch_pos[side], np.asarray(pos_goal, float), step), yaw, gripper)

    def _opening(self, st: TowelState, side: str) -> float:
        return st.finger_q[side] / GRIPPER_OPEN[side]

    def _grip_command(self, data: mujoco.MjData, side: str) -> float:
        """The previous gripper command (data.ctrl) as a fraction of open (the grasp rule's scale)."""
        return float(data.ctrl[self.ctrl_grip[side]] / GRIPPER_OPEN[side])

    # --- phases ------------------------------------------------------------------------------

    def _held_goal(self, st: TowelState, side: str) -> tuple[np.ndarray, np.ndarray, float]:
        """(held point now, where it goes, its distance to go when flat) [horizontal].

        The held point (the centroid of the held vertices) moves rigidly with the hand,
        unlike the far corner, which swings below it. Its goal is the corner's target plus
        the held point's offset from the corner in the towel's rest frame, as in the
        scripted fold (moving the grasp point by the corner-to-corner offset). Mirroring the
        offset for the turned-over layer puts the corner ~1 cm further out (21 mm instead
        of 10 from the near corner), with the flap at the corner bunched under the jaws, and
        the layer order breaks (Phase 2, scene 100001)."""
        held = st.held[side]
        h = st.verts[held, :2].mean(axis=0)
        t = self._target(st, side)
        far = self.ix.far[side]
        if far in held:  # the corner moves with the hand: keep its offset (the scripted fold's rule)
            goal = t + h - st.verts[far, :2]
        else:  # the corner hangs free: its rest offset along the fold axis, the current one across
            delta = self.ix.towel.rest[held, :2].mean(axis=0) - self.ix.towel.rest[far, :2]
            ex = self._fold_axis(st)
            ey = np.array([-ex[1], ex[0]])
            goal = t + delta[0] * ex + float((h - t) @ ey) * ey
        return h, goal, self.depth

    def _carry_error(self, st: TowelState, side: str) -> float:
        h, goal, _ = self._held_goal(st, side)
        return float(np.linalg.norm(h - goal))

    def _commanded(self, data: mujoco.MjData, side: str) -> tuple[np.ndarray, np.ndarray]:
        """(arm joint targets, pinch point they put the hand at) from data.ctrl (or the
        commanded targets given to act)."""
        if self._cmd is None:
            q = data.ctrl[self.ctrl_arm[side]].copy()
        else:
            q = np.asarray(self._cmd, float)[self.act_arm[side]].copy()
        fk = self._fk
        fk.qpos[:] = data.qpos
        fk.qpos[self.ik[side].qadr] = q
        mujoco.mj_kinematics(self.model, fk)
        return q, self.ik[side].pinch_pose(fk)[0].copy()

    def _carry_start(self, st: TowelState, side: str, data: mujoco.MjData):
        """(commanded held point, its goal, distance when flat, commanded joints or None,
        commanded pinch point): the carry continues from the commanded pose (see CARRY_STEP)."""
        h, goal, dist = self._held_goal(st, side)
        q_cmd, p_cmd = self._commanded(data, side)
        p = st.pinch_pos[side]
        if np.linalg.norm(p_cmd - p) > RESYNC:
            q_cmd, p_cmd = None, p
        return h + (p_cmd - p)[:2], goal, dist, q_cmd, p_cmd

    def _carry(self, st: TowelState, side: str, start, d_next: float, phase: str) -> ArmGoal:
        """Put the held point d_next from its goal along the line to it, at the arc's height
        for that distance."""
        h, goal, dist, q_cmd, p_cmd = start
        d = float(np.linalg.norm(h - goal))
        d_next = min(d, d_next)
        xy = goal + (h - goal) * (d_next / d if d > 1e-9 else 0.0)
        z = PLACE_PINCH_Z + self.lift * np.sin(np.pi * np.clip(d_next / dist, 0.0, 1.0))
        pos = _carrot(p_cmd, np.array([*(p_cmd[:2] + xy - h), z]), STEP_FAST)
        g = self._goal(st, side, phase, pos, GRIPPER_CLOSED[side], STEP_FAST)
        g.pos, g.q_start = pos, q_cmd
        return g

    def _next_distance(self, d: float, dist: float) -> float:
        ease = max(CARRY_MIN_FRAC, float(np.sin(np.pi * np.clip(1.0 - d / dist, 0.0, 1.0))))
        return max(0.0, d - ease * CARRY_STEP)

    def _placed(self, st: TowelState, side: str) -> bool:
        """The held point is at its goal, the hand down, the far corner at rest relative to
        the hand. Relative, not absolute: with DART noise the hand (and a held corner with
        it) moves 1-2 mm per step, 3-6 cm/s, and the corner never counted as at rest."""
        if not st.holds(side):
            return False
        far = self.ix.far[side]
        swing = float(np.linalg.norm(st.vert_vel[far] - st.vert_vel[st.held[side]].mean(axis=0)))
        return (self._carry_error(st, side) < PLACED_XY
                and st.pinch_pos[side][2] < PLACE_PINCH_Z + PLACED_Z
                and swing < SETTLED_SPEED)

    def _done(self, st: TowelState, side: str) -> bool:
        """A far corner that has been let go at its target."""
        c = st.verts[self.ix.far[side]]
        n = st.verts[self.ix.near[side]]
        falling = st.vert_speed[self.ix.far[side]] > SETTLED_SPEED  # just let go: still coming down
        return (not st.holds(side) and float(np.linalg.norm(c[:2] - n[:2])) < DONE_XY
                and (c[2] < n[2] + DONE_Z or falling))

    def _grasp(self, st: TowelState, side: str, data: mujoco.MjData) -> ArmGoal:
        target = self._grasp_point(st, side)
        p = st.pinch_pos[side]
        err = float(np.linalg.norm(p[:2] - target[:2]))
        opening = self._opening(st, side)
        if opening < MISSED:  # closed on nothing (or the vertices went to the other hand)
            return self._goal(st, side, "MISSED", [p[0], p[1], target[2] + HOVER / 2], OPENED[side], STEP_SLOW)
        closing = self._grip_command(data, side) < CLOSE_CMD
        if (err < CLOSE_XY and p[2] - target[2] < CLOSE_Z) or (
                closing and err < CLOSING_XY and p[2] - target[2] < CLOSING_Z):
            return self._goal(st, side, "CLOSE", target, GRIPPER_CLOSED[side], STEP_SLOW)
        if opening < OPEN_ENOUGH:  # still opening: wait above the grasp height
            return self._goal(st, side, "OPEN", [p[0], p[1], max(p[2], target[2] + 0.02)], OPENED[side], STEP_SLOW)
        lift = min(HOVER, FUNNEL * max(0.0, err - DEADBAND))
        goal = target + [0.0, 0.0, lift]
        if p[2] < target[2] + 0.03 and err > 0.03:  # low and off target: rise before travelling
            goal = np.array([p[0], p[1], target[2] + HOVER])
        phase = "DESCEND" if lift < 0.02 else "APPROACH"
        return self._goal(st, side, phase, goal, OPENED[side], STEP_SLOW if phase == "DESCEND" else STEP_FAST)

    def _home(self, st: TowelState, side: str) -> ArmGoal:
        p = st.pinch_pos[side]
        top = float(st.verts[:, 2].max())
        near = np.min(np.linalg.norm(st.verts[:, :2] - p[:2], axis=1)) < 0.05
        if near and p[2] < PLACE_PINCH_Z + RELEASE_LIFT - 0.003 and self._opening(st, side) < RELEASE_OPEN:
            # just let go: open while lifting RELEASE_LIFT, as the scripted fold. Opening in
            # place, one jaw sweeps the corner 2-3 cm along (Phase 1 and 2)
            return self._goal(st, side, "RELEASE", [p[0], p[1], PLACE_PINCH_Z + RELEASE_LIFT], OPENED[side],
                              RELEASE_STEP)
        if near and p[2] < top + RISE_Z - 0.01:
            return self._goal(st, side, "RISE", [p[0], p[1], top + RISE_Z], OPENED[side], STEP_SLOW)
        return self._goal(st, side, "HOME", HOME_PINCH[side], OPENED[side], STEP_FAST)

    def plan(self, st: TowelState, data: mujoco.MjData) -> dict[str, ArmGoal]:
        goals = {}
        for s in SIDES:
            o = next(x for x in SIDES if x != s)
            if st.holds(s):
                other_ready = st.holds(o) or self._done(st, o)
                if self._placed(st, s) and (self._placed(st, o) or self._done(st, o)):
                    p = st.pinch_pos[s]
                    goals[s] = self._goal(st, s, "RELEASE", [p[0], p[1], PLACE_PINCH_Z + RELEASE_LIFT], OPENED[s],
                                          RELEASE_STEP)
                    continue
                start = self._carry_start(st, s, data)
                d = float(np.linalg.norm(start[0] - start[1]))
                if not other_ready:
                    goals[s] = self._carry(st, s, start, d, "WAIT")
                    continue
                if st.holds(o):  # both carry: one shared progress, set by the hand further behind
                    other = self._carry_start(st, o, data)
                    d = max(d, float(np.linalg.norm(other[0] - other[1])))
                goals[s] = self._carry(st, s, start, self._next_distance(d, start[2]), "CARRY")
            elif self._done(st, s):
                goals[s] = self._home(st, s)
            else:
                goals[s] = self._grasp(st, s, data)
        return goals

    def act(self, data: mujoco.MjData, commanded: np.ndarray | None = None) -> tuple[np.ndarray, ExpertInfo]:
        """16 joint targets in ACTUATOR_NAMES order for the current state. Does not modify `data`.

        `commanded` (ACTUATOR_NAMES order) stands in for data.ctrl as the previous command the
        carry continues from. With DART, pass the previous clean label: data.ctrl holds the
        executed (noisy) action, and continuing from it adds the AR(1) noise *level* to the
        carry every step, a random walk of ~4 cm at sigma 0.00325 (milestone 3: the carry
        swung 12-45 mm around the place point and never released)."""
        self._cmd = commanded
        try:
            st = read_state(self.model, self.ix, data)
            goals = self.plan(st, data)
        finally:
            self._cmd = None
        action = np.empty(len(ACTUATOR_NAMES))
        for s in SIDES:
            g = goals[s]
            ik = self.ik[s]
            q0 = data.qpos[ik.qadr] if g.q_start is None else g.q_start
            q, _, _ = ik.solve(data.qpos, q0, g.pos, topdown_rotation(g.yaw), iters=30, tol_pos=IK_TOL,
                               q_rest=self.q_rest[s])
            action[self.act_arm[s]] = q
            action[self.act_grip[s]] = g.gripper
        return action, ExpertInfo(f"{goals['left'].phase}/{goals['right'].phase}", goals)

"""Privileged scripted expert for the bar hand-over (milestone 2, Phase 2).

Like keiarm/expert.py, the expert is a pure function of the simulator state: `act(data)`
reads where the bar, the plate and both hands are, decides which phase that is, and
returns the 16 joint targets. No memory between calls, so DAgger can ask it to label any
state, and slips and drops are recovered by construction.

Phases, checked in order:

1. The bar lies on the plate          -> the left hand lets go and backs off; right goes home.
2. Both hands hold the bar            -> hand-over: the right hand opens, both hold the
                                         hand-over pose (absolute targets, so the two
                                         position-controlled arms do not pull on the bar).
3. Only the left hand holds it        -> the left hand carries it over the plate and lowers
                                         it; the right hand backs off upwards, then goes home.
4. Only the right hand holds it       -> carry it to the hand-over pose (bar along y, centre
                                         HANDOVER); once it is there the left hand comes
                                         down on the +y end and closes.
5. Nobody holds it                    -> the hand on the bar's side grasps it (normally the
                                         right hand at the start; also the recovery after
                                         a drop); the other hand waits at its start pose.

Targets are carrots (at most STEP_* metres from the current pinch point towards the goal,
and the commanded jaw yaw turns at most YAW_STEP per step). Goals are absolute or relative
to the bar/plate, never "stay at the current height".
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from keiarm.openarm.ik import ArmIK, topdown_rotation
from keiarm.openarm.robot import (
    ARM_DOF,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_ARM_QPOS,
    HOME_PINCH,
    HOME_YAW,
    SIDES,
)
from keiarm.openarm.scene import BAR_GRIP_OFFSET, SceneParams
from keiarm.openarm.task import TaskIndex, TaskState, over_plate, read_state, rests_on_plate

HANDOVER = np.array([0.27, 0.0, 0.15])  # bar centre during the hand-over [m]
MIN_PINCH_Z = 0.024  # lowest pinch height (fingertips reach 2 cm below it)
HOVER = 0.08  # pinch height above the grasp point before descending
# Grasp tolerances, split along the bar and across it. Across, the jaws (open 14 cm) around
# a bar of at most 3.5 cm forgive a lot. Along, an error moves the grip towards the bar's
# centre and the two hands closer together at the hand-over (8 cm apart leaves 3 mm).
ALONG_DEADBAND, ACROSS_DEADBAND = 0.006, 0.015  # errors [m] that need no extra funnel height
ALONG_CLOSE, ACROSS_CLOSE = 0.010, 0.025  # the jaws close within these errors
YAW_DEADBAND = np.radians(4.0)
FUNNEL_LATERAL = 3.0  # funnel height per metre of error beyond the deadbands
FUNNEL_YAW = 0.15  # funnel height per radian of jaw yaw error beyond its deadband
CLOSE_HEIGHT = 0.010
CLOSE_YAW = np.radians(10.0)
PLACE_DEADBAND = 0.03  # bar centre vs plate centre [m] that needs no extra height (plate radius >= 7 cm)
HAND_SEPARATION = 0.115  # minimum pinch-to-pinch distance along the bar at the hand-over (9 cm: the wrists touch)
END_MARGIN = 0.012  # the left pinch stays this far inside the bar's end
MEET_SIDE = 0.07  # the left hand lines up this far on the +y side of its grip point before sliding in
CARRY_Z = 0.15  # bar centre height while carrying
PLACE_CLEARANCE = 0.001

STEP_FAST = 0.008  # [m] per control step (0.24 m/s)
STEP_SLOW = 0.005
YAW_STEP = np.radians(4.0)

OPEN_ENOUGH = 0.8  # fraction of the open finger angle that counts as "released"
OPEN_FULLY = 0.95  # after placing, the jaws open this far before the hand moves away
SETTLED_QVEL = 0.5  # [rad/s] finger speed below which a closed grip counts as settled

PHASES = ("APPROACH", "DESCEND", "CLOSE", "LIFT", "CARRY", "WAIT", "MEET", "HANDOVER", "RELEASE",
          "BACK_OFF", "PLACE_CARRY", "LOWER", "OPEN", "HOME")


@dataclass
class ArmGoal:
    phase: str
    pos: np.ndarray  # pinch point goal (already a carrot)
    yaw: float  # jaw yaw goal (already rate-limited)
    gripper: float


@dataclass
class ExpertInfo:
    phase: str  # "<left phase>/<right phase>"
    goals: dict[str, ArmGoal]


def jaw_yaw(rot: np.ndarray) -> float:
    """Yaw of the jaw closing direction (rot[:, 1] = (-sin, cos)(yaw))."""
    return float(np.arctan2(-rot[0, 1], rot[1, 1]))


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _carrot(pos: np.ndarray, goal: np.ndarray, step: float) -> np.ndarray:
    d = goal - pos
    n = float(np.linalg.norm(d))
    return goal.copy() if n <= step else pos + d * (step / n)


class Expert:
    def __init__(self, model: mujoco.MjModel, scene: SceneParams):
        self.model = model
        self.scene = scene
        self.ix = TaskIndex(model)
        self.ik = {s: ArmIK(model, s) for s in SIDES}
        self.act_arm = {s: np.array([model.actuator(f"{s}_joint{i}_ctrl").id for i in range(1, ARM_DOF + 1)])
                        for s in SIDES}
        self.act_grip = {s: model.actuator(f"{s}_finger1_ctrl").id for s in SIDES}
        self.act_order = {a: i for i, a in enumerate(self.ix.act)}
        self.q_rest = {s: np.array(HOME_ARM_QPOS[s]) for s in SIDES}

    # --- geometry --------------------------------------------------------------------------

    def _grip_point(self, st: TaskState, side: str) -> np.ndarray:
        """Where `side` grasps the bar lying flat: BAR_GRIP_OFFSET from its centre along the
        long axis (right: -y end, left: +y end), at the pinch height for its thickness."""
        sign = 1.0 if side == "left" else -1.0
        xy = st.bar_pos[:2] + sign * BAR_GRIP_OFFSET * st.bar_axis
        h = self.scene.bar_size[2]
        z = st.bar_pos[2] - h / 2 + max(h / 2, MIN_PINCH_Z)
        return np.array([*xy, z])

    def _bar_yaw_goal(self, side: str, axis_xy: np.ndarray) -> float:
        """`side`'s natural jaw yaw for a bar whose long axis is axis_xy (oriented towards +y):
        HOME_YAW[side] + the bar's angle from the y axis. Never the 180-deg flipped grasp:
        with it the hand cannot turn the bar to lie along y (wrist limits)."""
        phi = np.arctan2(-axis_xy[0], axis_xy[1])
        return float(HOME_YAW[side] + phi)

    def _goal(self, st: TaskState, side: str, phase: str, pos_goal, yaw_goal: float, gripper: float,
              step: float = STEP_FAST) -> ArmGoal:
        now = jaw_yaw(st.pinch_mat[side])
        yaw = now + float(np.clip(wrap(yaw_goal - now), -YAW_STEP, YAW_STEP))
        return ArmGoal(phase, _carrot(st.pinch_pos[side], np.asarray(pos_goal, float), step), yaw, gripper)

    def _home(self, st: TaskState, side: str) -> ArmGoal:
        p = st.pinch_pos[side]
        bar_near = np.linalg.norm(p[:2] - st.bar_pos[:2]) < 0.12 and p[2] < st.bar_pos[2] + 0.06
        if bar_near:  # rise clear of the bar before travelling: slowly, straight up, jaws kept as they are
            return self._goal(st, side, "BACK_OFF", [p[0], p[1], st.bar_pos[2] + 0.10], jaw_yaw(st.pinch_mat[side]),
                              GRIPPER_OPEN[side], STEP_SLOW)
        return self._goal(st, side, "HOME", HOME_PINCH[side], HOME_YAW[side], GRIPPER_OPEN[side])

    def _misalignment(self, st: TaskState, side: str, target: np.ndarray, yaw_goal: float):
        """(along-bar error, across-bar error, yaw error) of `side`'s pinch point vs target."""
        d = st.pinch_pos[side][:2] - target[:2]
        a = st.bar_axis
        along = abs(float(d @ a))
        across = abs(float(d[0] * a[1] - d[1] * a[0]))
        return along, across, abs(wrap(jaw_yaw(st.pinch_mat[side]) - yaw_goal))

    def _funnel(self, along: float, across: float, yaw_err: float) -> float:
        return (FUNNEL_LATERAL * (max(0.0, along - ALONG_DEADBAND) + max(0.0, across - ACROSS_DEADBAND))
                + FUNNEL_YAW * max(0.0, yaw_err - YAW_DEADBAND))

    def _grasp(self, st: TaskState, side: str) -> ArmGoal:
        """Top-down grasp of the bar at `side`'s grip point.

        Funnel: the pinch goal sits above the grip point by a height that grows with the
        along-bar, across-bar and yaw errors beyond their deadbands (capped at HOVER), so
        the hand comes down only as it lines up, without switching back and forth between
        "approach" and "descend".
        """
        target = self._grip_point(st, side)
        yaw_goal = self._bar_yaw_goal(side, st.bar_axis)
        p = st.pinch_pos[side]
        along, across, yaw_err = self._misalignment(st, side, target, yaw_goal)
        open_now = abs(st.finger_q[side]) > OPEN_ENOUGH * abs(GRIPPER_OPEN[side])
        if p[2] - target[2] < CLOSE_HEIGHT and along < ALONG_CLOSE and across < ACROSS_CLOSE and yaw_err < CLOSE_YAW:
            return self._goal(st, side, "CLOSE", target, yaw_goal, GRIPPER_CLOSED[side], STEP_SLOW)
        lift = min(HOVER, self._funnel(along, across, yaw_err))
        if not open_now:
            lift = HOVER
        goal = target + [0.0, 0.0, lift]
        if p[2] < st.bar_pos[2] + self.scene.bar_size[2] / 2 + 0.01 and (across > 0.04 or along > 0.03):
            goal[:2] = p[:2]  # fingertips beside the bar: rise in place before travelling
            goal[2] = target[2] + HOVER
        phase = "DESCEND" if lift < 0.03 else "APPROACH"
        return self._goal(st, side, phase, goal, yaw_goal, GRIPPER_OPEN[side],
                          STEP_SLOW if phase == "DESCEND" else STEP_FAST)

    def _meet(self, st: TaskState) -> ArmGoal:
        """The left hand takes the bar held by the right hand at the hand-over pose.

        It comes in sideways at the bar's height from the +y side, jaws open around the
        bar's end, and slides along the bar onto its grip point. (From above, its open
        fingers would sweep through the right hand's wrist housing.) The side offset shrinks
        to zero only as the hand lines up in x, z and yaw.
        """
        target = self._grip_point_held(st, "left")
        yaw_goal = self._bar_yaw_goal("left", st.bar_axis)
        p = st.pinch_pos["left"]
        along, across, yaw_err = self._misalignment(st, "left", target, yaw_goal)
        vertical = abs(p[2] - target[2])
        err_xz = float(np.hypot(across, vertical))
        if along < ALONG_CLOSE and across < ACROSS_CLOSE and vertical < 0.012 and yaw_err < CLOSE_YAW:
            return self._goal(st, "left", "CLOSE", target, yaw_goal, GRIPPER_CLOSED["left"], STEP_SLOW)
        side_offset = min(MEET_SIDE, FUNNEL_LATERAL * (max(0.0, across - ACROSS_DEADBAND) + max(0.0, vertical - 0.006))
                          + FUNNEL_YAW * max(0.0, yaw_err - YAW_DEADBAND))
        if p[1] < target[1] + side_offset - 0.01 and err_xz > 0.02:
            side_offset = MEET_SIDE  # still on the wrong side of the funnel: back out along +y first
        goal = target + [0.0, side_offset, 0.0]
        phase = "MEET" if side_offset > 0.02 else "SLIDE"
        return self._goal(st, "left", phase, goal, yaw_goal, GRIPPER_OPEN["left"],
                          STEP_SLOW if phase == "SLIDE" else STEP_FAST)

    def _along(self, st: TaskState, side: str) -> float:
        """Signed position of `side`'s pinch along the bar from its centre (+ towards +y)."""
        return float((st.pinch_pos[side][:2] - st.bar_pos[:2]) @ st.bar_axis)

    def _left_grip_along(self, st: TaskState) -> float:
        """Where the left hand takes the bar: BAR_GRIP_OFFSET, or further out if the right hand
        holds it closer to the centre, so the hands stay HAND_SEPARATION apart."""
        half = self.scene.bar_size[1] / 2
        return min(max(BAR_GRIP_OFFSET, self._along(st, "right") + HAND_SEPARATION), half - END_MARGIN)

    def _grip_point_held(self, st: TaskState, side: str) -> np.ndarray:
        """Grip point on a bar in the air (pinch at the same height above the bar centre as on the table)."""
        along = self._left_grip_along(st) if side == "left" else -BAR_GRIP_OFFSET
        h = self.scene.bar_size[2]
        xy = st.bar_pos[:2] + along * st.bar_axis
        return np.array([*xy, st.bar_pos[2] + max(h / 2, MIN_PINCH_Z) - h / 2])

    def _carry(self, st: TaskState, side: str, phase: str, bar_goal: np.ndarray, axis_goal, gripper: float,
               step: float) -> ArmGoal:
        """Servo the held bar's centre to bar_goal: move the pinch by the bar's error."""
        pos = st.pinch_pos[side] + (bar_goal - st.bar_pos)
        return self._goal(st, side, phase, pos, self._bar_yaw_goal(side, np.asarray(axis_goal)), gripper, step)

    def _handover_pose(self, st: TaskState, side: str) -> np.ndarray:
        """Pinch goal of `side` while both hands hold the bar at HANDOVER: where it actually
        holds the bar along it (so the two goals agree and the arms do not pull on the bar),
        at an absolute height."""
        h = self.scene.bar_size[2]
        return HANDOVER + [0.0, self._along(st, side), max(h / 2, MIN_PINCH_Z) - h / 2]

    # --- phase logic -----------------------------------------------------------------------

    def plan(self, st: TaskState) -> dict[str, ArmGoal]:
        scene, h = self.scene, self.scene.bar_size[2]
        y_axis = np.array([0.0, 1.0])
        held = {s: st.held(s) for s in SIDES}
        open_enough = {s: abs(st.finger_q[s]) > OPEN_ENOUGH * abs(GRIPPER_OPEN[s]) for s in SIDES}

        # 1. Bar on the plate: let go, back off.
        if rests_on_plate(st, scene, z_tol=0.008) and over_plate(st, scene, margin=0.005):
            goals = {}
            for s in SIDES:
                p = st.pinch_pos[s]
                near = np.linalg.norm(p[:2] - st.bar_pos[:2]) < 0.10 and p[2] < st.bar_pos[2] + 0.04
                if near and (st.touches(s) or abs(st.finger_q[s]) < OPEN_FULLY * abs(GRIPPER_OPEN[s])):
                    hold = self._grip_point(st, s) if s == "left" else p
                    goals[s] = self._goal(st, s, "OPEN", [p[0], p[1], hold[2]], jaw_yaw(st.pinch_mat[s]),
                                          GRIPPER_OPEN[s], STEP_SLOW)
                else:
                    goals[s] = self._home(st, s)
            return goals

        # 2. Both hold it: the right hand lets go; both keep the hand-over pose.
        if held["left"] and held["right"]:
            settled = abs(st.finger_qvel["left"]) < SETTLED_QVEL
            return {
                "left": self._goal(st, "left", "HANDOVER", self._handover_pose(st, "left"), HOME_YAW["left"],
                                   GRIPPER_CLOSED["left"], STEP_SLOW),
                "right": self._goal(st, "right", "RELEASE" if settled else "HANDOVER",
                                    self._handover_pose(st, "right"), HOME_YAW["right"],
                                    GRIPPER_OPEN["right"] if settled else GRIPPER_CLOSED["right"], STEP_SLOW),
            }

        # 3. Only the left hand: wait until the right fingers are off and open, then carry
        #    over the plate and lower.
        if held["left"]:
            if st.touches("right") or not open_enough["right"]:
                return {
                    "left": self._goal(st, "left", "HANDOVER", self._handover_pose(st, "left"), HOME_YAW["left"],
                                       GRIPPER_CLOSED["left"], STEP_SLOW),
                    "right": self._goal(st, "right", "RELEASE", self._handover_pose(st, "right"), HOME_YAW["right"],
                                        GRIPPER_OPEN["right"], STEP_SLOW),
                }
            rest_z = st.plate_top + h / 2 + PLACE_CLEARANCE
            to_plate = float(np.linalg.norm(st.plate_pos[:2] - st.bar_pos[:2]))
            # Funnel over the plate: carry height far away, down to resting height as the
            # bar's centre lines up with the plate's.
            z = rest_z + min(CARRY_Z - rest_z, FUNNEL_LATERAL * max(0.0, to_plate - PLACE_DEADBAND))
            phase = "LOWER" if z < CARRY_Z - 0.03 else "PLACE_CARRY"
            left = self._carry(st, "left", phase, np.array([*st.plate_pos[:2], z]), y_axis, GRIPPER_CLOSED["left"],
                               STEP_SLOW if phase == "LOWER" else STEP_FAST)
            return {"left": left, "right": self._home(st, "right")}

        # 4. Only the right hand: bring the bar to the hand-over pose; then the left hand meets it.
        if held["right"]:
            low = st.bar_pos[2] < CARRY_Z - 0.03
            far = np.linalg.norm(st.bar_pos[:2] - HANDOVER[:2]) > 0.03
            if low and far:
                right = self._carry(st, "right", "LIFT", np.array([*st.bar_pos[:2], CARRY_Z]), st.bar_axis,
                                    GRIPPER_CLOSED["right"], STEP_FAST)
            else:
                right = self._carry(st, "right", "CARRY", HANDOVER, y_axis, GRIPPER_CLOSED["right"], STEP_FAST)
            bar_yaw_err = abs(wrap(np.arctan2(-st.bar_axis[0], st.bar_axis[1])))
            bar_err = float(np.linalg.norm(st.bar_pos - HANDOVER))
            left_close = np.linalg.norm(st.pinch_pos["left"] - self._grip_point_held(st, "left")) < 0.12
            # Arrived; looser once the left hand is near, so a nudge from its fingers does not
            # send it back home.
            arrived = (bar_err < 0.03 and bar_yaw_err < np.radians(10.0)) or (
                left_close and bar_err < 0.05 and bar_yaw_err < np.radians(15.0))
            if arrived:
                left = self._meet(st)
            else:
                left = self._goal(st, "left", "WAIT", HOME_PINCH["left"], HOME_YAW["left"], GRIPPER_OPEN["left"])
            return {"left": left, "right": right}

        # 5. Nobody holds it: the hand on the bar's side grasps it, the other waits.
        grasper = "right" if st.bar_pos[1] < 0.0 else "left"
        other = "left" if grasper == "right" else "right"
        return {grasper: self._grasp(st, grasper), other: self._home(st, other)}

    def act(self, data: mujoco.MjData) -> tuple[np.ndarray, ExpertInfo]:
        """16 joint targets in ACTUATOR_NAMES order for the current state. Does not modify `data`."""
        st = read_state(self.model, self.ix, data)
        goals = self.plan(st)
        action = np.empty(len(self.ix.act))
        for s in SIDES:
            g = goals[s]
            ik = self.ik[s]
            q, _, _ = ik.solve(data.qpos, data.qpos[ik.qadr], g.pos, topdown_rotation(g.yaw), iters=30,
                               q_rest=self.q_rest[s])
            for a, qa in zip(self.act_arm[s], q):
                action[self.act_order[a]] = qa
            action[self.act_order[self.act_grip[s]]] = g.gripper
        return action, ExpertInfo(f"{goals['left'].phase}/{goals['right'].phase}", goals)

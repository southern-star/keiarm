"""Open-loop scripted half fold (Phase 0-1 check, from spike E3). Not the expert.

Both hands go above their far corner, descend, close (the grasp rule attaches the corner),
carry it along an arc to just short of the near corner, open while lifting (the rule lets
go) and go back to the start pinch point. The waypoints' reach over the whole spawn range
is checked by tools/towel_reach.py. Joint
targets come from ArmIK on a carrot. Everything goes through TowelEnv.step, so the gripper
commands, not direct constraint writes, trigger the grasp.

The plan is fixed when the episode starts (from the towel's initial corners): it does not
react to what happens, so it is a check of the physics and the grasp rule only.
"""

import time
from dataclasses import dataclass, field

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
from keiarm.towel import metrics
from keiarm.towel.env import TowelEnv

# Jaw yaw: the natural yaw (HOME_YAW) plus 45 deg (mirrored for the left hand). With the
# natural yaw the hands reach top-down only from x = 0.19; at +60 deg and more the two open
# grippers touch when the hands are 17 cm apart (tools/towel_reach.py). The grasp rule does
# not care about the jaw direction.
YAW_OFFSET = {"right": np.radians(45.0), "left": -np.radians(45.0)}
MIN_PINCH_Z = 0.024  # fingertips 4 mm above the table (keiarm/openarm/scripted.py)
GRASP_INSET = 0.006  # grasp this far inside the corner, towards the towel centre. At 1 cm the corner
# vertex is only just among the 4 the grasp rule takes (2 mm off and it is left out; Phase 2)
CARROT = 0.006  # [m] per control step outside the arc
CARRY_STEPS = 75  # 2.5 s
LIFT_RATIO = 0.55  # arc height / half the carried distance
# Gripper command while open: 60% of fully open. Far enough past the grasp rule's release
# threshold (OPEN_CMD = 50%); fully open, the two hands' fingers touch when the hands are
# less than about 17 cm apart (tools/towel_reach.py).
OPEN_FRACTION = 0.6
# Put the far corners down this far short of the near corners (+x, away from the robot): the
# hands meet the pedestal below x = 0.13 (tools/towel_reach.py). Within the success
# threshold (corners within 3 cm).
PLACE_SHORT = 0.01
# Letting go: open while lifting RELEASE_LIFT, then go back to the start pinch point
# (HOME_PINCH), out of cam_front's view of the folded towel.
RELEASE_LIFT = 0.03
OPENED = {s: OPEN_FRACTION * GRIPPER_OPEN[s] for s in SIDES}


@dataclass
class FoldResult:
    status: str = "ok"  # ok / broken kind / exception
    grabbed: dict = field(default_factory=dict)  # side -> number of vertices grabbed
    corner_gap_mm: dict = field(default_factory=dict)  # side -> (horizontal, vertical)
    violation: float | None = None
    crossings: int | None = None
    area_ratio: float | None = None
    max_anchor_err_mm: float = 0.0
    max_track_mm: float = 0.0
    steps: int = 0
    wall_physics: float = 0.0  # seconds inside env.step
    wall_total: float = 0.0

    @property
    def completed(self) -> bool:
        """The fold happened: no failure, both hands grabbed, both corners within 6 cm."""
        return (self.status == "ok" and all(self.grabbed.get(s, 0) > 0 for s in SIDES)
                and bool(self.corner_gap_mm) and all(h < 60.0 for h, _ in self.corner_gap_mm.values()))


def arc(p0, p1, s: float, lift: float) -> np.ndarray:
    u = 0.5 - 0.5 * np.cos(np.pi * s)
    p = p0 + u * (p1 - p0)
    p[2] += lift * np.sin(np.pi * u)
    return p


def scripted_fold(env: TowelEnv, on_step=None) -> FoldResult:
    """Run the scripted half fold from the current (just reset) state. `on_step(env)` is
    called after every control step (e.g. to render)."""
    m, d, tw = env.model, env.data, env.towel
    iks = {s: ArmIK(m, s) for s in SIDES}
    q_cmd = {s: d.qpos[iks[s].qadr].copy() for s in SIDES}
    carrot = {s: iks[s].pinch_pose(d)[0].copy() for s in SIDES}
    yaw = {s: HOME_YAW[s] + YAW_OFFSET[s] for s in SIDES}
    res = FoldResult()
    t_start = time.perf_counter()
    area0 = metrics.convex_area(env.towel_verts())
    act_index = {name: k for k, name in enumerate(ACTUATOR_NAMES)}

    def action(gripper: dict) -> np.ndarray:
        a = np.zeros(len(ACTUATOR_NAMES))
        for s in SIDES:
            q, _, _ = iks[s].solve(d.qpos, q_cmd[s], carrot[s], topdown_rotation(yaw[s]), iters=20,
                                   q_rest=np.array(HOME_ARM_QPOS[s]))
            q_cmd[s] = q
            for i in range(ARM_DOF):
                a[act_index[f"{s}_joint{i + 1}_ctrl"]] = q[i]
            a[act_index[f"{s}_finger1_ctrl"]] = gripper[s]
        return a

    def step(gripper: dict) -> bool:
        a = action(gripper)
        t = time.perf_counter()
        for side, ev in env.step(a):
            if ev == "grab":
                res.grabbed[side] = len(env.grasp.held(side))
        res.wall_physics += time.perf_counter() - t
        res.steps += 1
        res.max_anchor_err_mm = max(res.max_anchor_err_mm, env.grasp.anchor_error(d) * 1000)
        res.max_track_mm = max(res.max_track_mm, max(
            float(np.linalg.norm(iks[s].pinch_pose(d)[0] - carrot[s])) for s in SIDES) * 1000)
        if on_step:
            on_step(env)
        b = metrics.broken(d, tw.verts(d))
        if b:
            res.status = b
            return False
        return True

    def go(targets: dict, gripper: dict, steps: int) -> bool:
        for _ in range(steps):
            for s, tg in targets.items():
                delta = tg - carrot[s]
                dist = np.linalg.norm(delta)
                carrot[s] = carrot[s] + (delta if dist < CARROT else delta / dist * CARROT)
            if not step(gripper):
                return False
        return True

    opened = dict(OPENED)
    closed = dict(GRIPPER_CLOSED)
    try:
        v = env.towel_verts()
        far = {s: metrics.corner(tw, metrics.FAR_CORNER[s]) for s in SIDES}
        near = {s: metrics.corner(tw, metrics.NEAR_CORNER[s]) for s in SIDES}
        center = v.mean(axis=0)[:2]
        grasp_xy = {}
        for s in SIDES:
            inward = center - v[far[s], :2]
            grasp_xy[s] = v[far[s], :2] + GRASP_INSET * inward / np.linalg.norm(inward)
        above = {s: np.array([*grasp_xy[s], 0.12]) for s in SIDES}
        down = {s: np.array([*grasp_xy[s], MIN_PINCH_Z]) for s in SIDES}
        ok = go({}, opened, 30) and go(above, opened, 60) and go(down, opened, 40) and go(down, closed, 20)
        if ok:
            v = env.towel_verts()
            p0 = {s: carrot[s].copy() for s in SIDES}
            # carry the grasp point by the corner-to-corner offset (less PLACE_SHORT along the
            # fold), ending 8 mm higher
            p1 = {}
            for s in SIDES:
                shift = v[near[s], :2] - v[far[s], :2]
                shift -= PLACE_SHORT * shift / np.linalg.norm(shift)
                p1[s] = np.array([*(p0[s][:2] + shift), MIN_PINCH_Z + 0.008])
            for k in range(1, CARRY_STEPS + 1):
                for s in SIDES:
                    carrot[s] = arc(p0[s], p1[s], k / CARRY_STEPS, LIFT_RATIO * np.linalg.norm((p1[s] - p0[s])[:2]) / 2)
                if not step(closed):
                    ok = False
                    break
            ok = ok and go(p1, closed, 10)
        if ok:
            # open while lifting straight up, so the opening fingers leave the cloth instead
            # of sweeping it along (one jaw sweeps the corner by ~2 cm if the hand stays put)
            lift = {s: p1[s] + [0.0, 0.0, RELEASE_LIFT] for s in SIDES}
            home = {s: np.array(HOME_PINCH[s]) for s in SIDES}
            ok = go(lift, opened, 10) and go(home, opened, 45) and go(home, opened, 15)
        if ok:
            v = env.towel_verts()
            res.corner_gap_mm = {s: tuple(round(x * 1000, 1) for x in g) for s, g in metrics.corner_gaps(tw, v).items()}
            res.violation = round(metrics.layer_violation(v, metrics.half_fold_layers(tw)), 4)
            res.crossings = metrics.self_intersections(v, metrics.triangles(m, tw))
            res.area_ratio = round(metrics.convex_area(v) / area0, 4)
    except Exception as e:  # noqa: BLE001 - report, the caller counts failures
        res.status = f"exception: {e}"
    res.wall_total = time.perf_counter() - t_start
    return res

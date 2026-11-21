"""Closed-loop evaluation of the towel half fold: stages, infractions, disturbances, failures.

The towel counterpart of keiarm/openarm/eval.py (milestone 3, Phase 3). Per episode it
records the furthest stage and the first step each stage was reached, and four kinds of
infraction:
  - the towel leaving the table (a vertex beyond the table's edge or 1 cm into it)
  - layers passing through each other (the towel's mid-surface crossing itself)
  - the two arms touching
  - missed grasps (a hand closed and caught nothing) and early releases (a hand let go of
    the towel with its far corner not yet over its near corner)
Failed episodes are classified and probed with the expert ("could it still finish?").

Disturbance suite (one per episode, all decided by the seeds; docs/milestone3.md):
  shift  before the hands close, the towel jumps flat to another pose in the spawn range
  slip   while carrying, one hand loses the towel (its fingers stay closed): it re-grasps
  drop   while carrying, both hands lose it: the towel falls half folded and is folded again
         from where it lies (it is not spread out first)
  nudge  just after both hands let go, one corner of the upper layer is flicked up and back
         over the crease and lands a few cm from its place: it is put back
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from keiarm.openarm.robot import SIDES
from keiarm.openarm.scene import TABLE_CENTER, TABLE_HALF_SIZE
from keiarm.towel import metrics
from keiarm.towel.env import MAX_EPISODE_STEPS, TowelEnv
from keiarm.towel.expert import Expert
from keiarm.towel.scene import TOWEL_CENTER, TOWEL_CENTER_JITTER, TOWEL_YAW_RANGE, episode_rng
from keiarm.towel.task import (
    CORNER_H,
    STAGES,
    SUCCESS_HOLD_STEPS,
    TaskIndex,
    TowelState,
    check_fold,
    read_state,
    stage_of,
)

DISTURB_STREAM = 2
DISTURBANCES = ("none", "shift", "slip", "drop", "nudge")
EXPERT_PROBE_STEPS = 450  # 15 s: three times the expert's time for the whole fold
OFF_TABLE_Z = -0.01

# shift: fires 10-37 steps in (the hands come below 5 cm at about step 42 and close at
# 52-56), to a pose whose corners move at least SHIFT_MIN from where they were.
SHIFT_DELAY = (10, 38)
SHIFT_MIN = 0.008  # [m], the spawn range is only +-8 mm x +-10 mm x +-5 deg
# slip, drop: 0-19 steps after the far corners are lifted 3 cm (stage "lift", at about
# step 75; the carry ends at about step 100-120).
CARRY_DELAY = (0, 20)
# nudge: 3-10 steps after both hands have let go (stage "released"), the upper-layer
# vertices within NUDGE_RADIUS (rest distance) of one far corner get a velocity falling off
# linearly from NUDGE_SPEED at the corner, NUDGE_ELEVATION up and pointing back over the
# crease (the towel's +x, within +-NUDGE_SPREAD): the corner flips partly back and lands
# 3-5 cm from its place. A sideways flick does not move it for good: the flap cannot shear
# (the grid's edges keep their length) and springs back to within 2 cm even at 3 m/s.
# Above ~1.5 m/s the flap whips through the lower layer.
NUDGE_DELAY = (3, 11)
NUDGE_RADIUS = 0.05
NUDGE_SPEED = (0.8, 1.2)  # [m/s]
NUDGE_ELEVATION = np.radians(60.0)
NUDGE_SPREAD = np.radians(30.0)

FAILURE_CLASSES = ("sim_broken", "towel_off_table", "layers_crossed", "grasp_failed", "lift_failed",
                   "carry_failed", "place_failed", "release_failed", "shape_failed")


# --- Disturbances --------------------------------------------------------------------------


@dataclass
class Disturbance:
    """Fires once, `delay` steps after the episode first reaches `after_stage` (0: from the
    start). `step` is filled in when it fires; replaying the episode applies it at that step."""

    kind: str
    after_stage: int
    delay: int
    sides: list[str] = field(default_factory=list)  # slip, drop: the hands that lose the towel; nudge: the corner
    xy: list[float] = field(default_factory=list)  # shift: new centre
    yaw: float = 0.0  # shift: new yaw; nudge: horizontal direction of the flick
    speed: float = 0.0  # nudge [m/s]
    step: int | None = None

    def due(self, t: int, stage_step: dict[int, int]) -> bool:
        if self.step is not None:
            return t == self.step
        first = 0 if self.after_stage == 0 else stage_step.get(self.after_stage)
        return first is not None and t >= first + self.delay

    def apply(self, env: TowelEnv) -> None:
        if self.kind == "shift":
            env.place_towel(self.xy, self.yaw)
        elif self.kind in ("slip", "drop"):
            env.slip(self.sides)
        elif self.kind == "nudge":
            verts, weight = nudge_region(env, self.sides[0])
            c = np.cos(NUDGE_ELEVATION)
            v = self.speed * np.array([c * np.cos(self.yaw), c * np.sin(self.yaw), np.sin(NUDGE_ELEVATION)])
            env.kick(verts, weight[:, None] * v)
        else:
            raise ValueError(self.kind)


def nudge_region(env: TowelEnv, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Upper-layer vertices near the side's far corner and their weights (1 at the corner)."""
    tw = env.towel
    far = metrics.corner(tw, metrics.FAR_CORNER[side])
    upper = np.arange(tw.nvert) // tw.n >= tw.n // 2
    r = np.linalg.norm(tw.rest[:, :2] - tw.rest[far, :2], axis=1)
    sel = np.nonzero(upper & (r < NUDGE_RADIUS))[0]
    return sel, 1.0 - r[sel] / NUDGE_RADIUS


def _towel_corners(xy, yaw, tw) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rest = tw.rest[[0, tw.n - 1, tw.nvert - tw.n, tw.nvert - 1], :2]
    return rest @ np.array([[c, s], [-s, c]]) + np.asarray(xy)


def planned_disturbance(env: TowelEnv, kind: str) -> Disturbance | None:
    """The episode's disturbance of the given kind, a pure function of its seeds. Call after reset."""
    if kind == "none":
        return None
    ep = env.episode
    rng = episode_rng(ep.scene_seed, ep.episode_seed, DISTURB_STREAM)
    if kind == "shift":
        delay = int(rng.integers(*SHIFT_DELAY))
        old = _towel_corners(ep.towel_xy, ep.towel_yaw, env.towel)
        jitter = np.array(TOWEL_CENTER_JITTER)
        while True:
            xy = np.array(TOWEL_CENTER) + rng.uniform(-jitter, jitter)
            yaw = float(rng.uniform(-TOWEL_YAW_RANGE, TOWEL_YAW_RANGE))
            if np.linalg.norm(_towel_corners(xy, yaw, env.towel) - old, axis=1).min() >= SHIFT_MIN:
                return Disturbance(kind, 0, delay, xy=[float(v) for v in xy], yaw=yaw)
    if kind == "slip":
        delay = int(rng.integers(*CARRY_DELAY))
        return Disturbance(kind, STAGES.index("lift"), delay, sides=[str(rng.choice(SIDES))])
    if kind == "drop":
        delay = int(rng.integers(*CARRY_DELAY))
        return Disturbance(kind, STAGES.index("lift"), delay, sides=list(SIDES))
    if kind == "nudge":
        delay = int(rng.integers(*NUDGE_DELAY))
        side = str(rng.choice(SIDES))
        yaw = float(ep.towel_yaw + rng.uniform(-NUDGE_SPREAD, NUDGE_SPREAD))
        return Disturbance(kind, STAGES.index("released"), delay, sides=[side], yaw=yaw,
                           speed=float(rng.uniform(*NUDGE_SPEED)))
    raise ValueError(f"unknown disturbance {kind!r}")


# --- Episode -------------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    scene_seed: int
    episode_seed: int
    mesh: int  # towel vertices per side
    disturb: str
    success: bool
    steps: int
    stage: str  # furthest stage reached
    stage_steps: dict[str, int]  # stage -> first control step (1-based count) it was reached
    stage_after_disturbance: str | None  # furthest stage reached after the disturbance fired
    arms_touch_steps: int
    grabs: int  # grab events (2 for a clean fold)
    empty_grasps: int  # a hand closed and caught no vertex
    early_releases: int  # a hand let go with its far corner not over its near corner
    crossing_steps: int  # control steps ending with the mid-surface crossing itself
    max_crossings: int
    towel_off_table: bool
    broken: str | None  # simulation failure (metrics.broken)
    max_gap_after_mm: float  # largest corner gap after the disturbance fired (0 without one)
    final_corner_gap_mm: dict[str, list[float]]
    final_area_ratio: float
    final_violation: float
    failure: str | None = None  # one of FAILURE_CLASSES
    expert_could_finish: bool | None = None  # failed episodes only
    disturbance: Disturbance | None = None
    actions: np.ndarray | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("actions")
        return d


class Monitor:
    """Tracks stage progress and infractions over one episode."""

    def __init__(self, env: TowelEnv):
        self.env = env
        self.ix = TaskIndex(env.model)
        self.si = metrics.SelfIntersection(metrics.triangles(env.model, env.towel))
        self.stage = 0
        self.stage_step: dict[int, int] = {}
        self.stage_after: int | None = None
        self.arms_touch_steps = self.grabs = self.empty_grasps = self.early_releases = 0
        self.max_gap_after = 0.0  # largest far-to-near corner gap after the disturbance [m]
        self.crossing_steps = self.max_crossings = 0
        self.off_table = False
        self.broken: str | None = None
        self._success_run = 0
        self.st = read_state(env.model, self.ix, env.data)

    def disturbed(self) -> None:
        """Call right after a disturbance changed the state."""
        self.st = read_state(self.env.model, self.ix, self.env.data)
        self.stage_after = 0
        self._success_run = 0

    def update(self, t: int, events) -> tuple[TowelState, bool]:
        """Call after env.step at control step t (0-based) with its grasp events. Returns
        (state, success held)."""
        env, ix = self.env, self.ix
        before = self.st
        st = read_state(env.model, ix, env.data)
        self.st = st
        v = st.verts
        for side, ev in events:
            self.grabs += ev == "grab"
            if ev == "grab" and not st.held[side]:
                self.empty_grasps += 1
            if ev == "release" and before.held[side]:
                d = before.verts[ix.far[side], :2] - before.verts[ix.near[side], :2]
                self.early_releases += bool(np.hypot(*d) >= CORNER_H)
        self.arms_touch_steps += st.arms_touch
        k = self.si.count(v)
        self.crossing_steps += k > 0
        self.max_crossings = max(self.max_crossings, k)
        self.off_table |= bool(np.any(np.abs(v[:, 0] - TABLE_CENTER[0]) > TABLE_HALF_SIZE[0])
                               or np.any(np.abs(v[:, 1] - TABLE_CENTER[1]) > TABLE_HALF_SIZE[1])
                               or v[:, 2].min() < OFF_TABLE_Z)
        self.broken = self.broken or metrics.broken(env.data, v)

        chk = check_fold(st, ix, env.scene)
        if self.stage_after is not None:
            self.max_gap_after = max(self.max_gap_after, *(h for h, _ in chk.corner_gap.values()))
        self._success_run = self._success_run + 1 if chk.success else 0
        done = self._success_run >= SUCCESS_HOLD_STEPS
        # 'success' only counts once it has held
        now = len(STAGES) - 1 if done else min(stage_of(st, ix, env.scene), len(STAGES) - 2)
        for s in range(1, now + 1):
            self.stage_step.setdefault(s, t + 1)
        self.stage = max(self.stage, now)
        if self.stage_after is not None:
            self.stage_after = max(self.stage_after, now)
        return st, done


def expert_can_finish(env: TowelEnv, steps: int = EXPERT_PROBE_STEPS) -> bool:
    """From the current state, does the expert reach success within `steps`? The env's state
    is restored afterwards."""
    saved = env.get_state()
    expert = Expert(env.model, env.scene)
    ix = expert.ix
    run, ok = 0, False
    try:
        for _ in range(steps):
            action, _ = expert.act(env.data)
            env.step(action)
            run = run + 1 if check_fold(read_state(env.model, ix, env.data), ix, env.scene).success else 0
            if run >= SUCCESS_HOLD_STEPS:
                ok = True
                break
    finally:
        env.set_state(saved)
    return ok


_STAGE_FAILURE = {"none": "grasp_failed", "reach": "grasp_failed", "grasp_one": "grasp_failed",
                  "grasp_both": "lift_failed", "lift": "carry_failed", "carry": "place_failed",
                  "placed": "release_failed", "released": "shape_failed"}


def classify_failure(mon: Monitor, final_crossings: int) -> str:
    if mon.broken:
        return "sim_broken"
    if mon.off_table:
        return "towel_off_table"
    if final_crossings:
        return "layers_crossed"
    stage = mon.stage if mon.stage_after is None else mon.stage_after
    return _STAGE_FAILURE[STAGES[stage]]


def run_episode(env: TowelEnv, policy, episode_seed: int, disturb: str = "none",
                max_steps: int = MAX_EPISODE_STEPS, on_step=None, probe: bool = True) -> EpisodeRecord:
    env.reset(episode_seed)
    policy.reset(env)
    dist = planned_disturbance(env, disturb)
    mon = Monitor(env)
    actions = []
    success = False
    lo, hi = env.ctrl_range.T
    for t in range(max_steps):
        if dist is not None and dist.step in (None, t) and dist.due(t, mon.stage_step):
            dist.step = t
            dist.apply(env)
            mon.disturbed()
        action = np.clip(policy.act(env), lo, hi)
        events = env.step(action)
        actions.append(action)
        _st, success = mon.update(t, events)
        if on_step is not None:
            on_step(env)
        if mon.broken:
            success = False
            break
        # an episode that has a disturbance still to come is not over yet (nudge waits for the release)
        if success and (dist is None or dist.step is not None):
            break
        success = False
    chk = check_fold(mon.st, mon.ix, env.scene)
    crossings = mon.si.count(mon.st.verts)
    rec = EpisodeRecord(
        scene_seed=env.scene.scene_seed, episode_seed=episode_seed, mesh=env.towel.n, disturb=disturb,
        success=success, steps=len(actions), stage=STAGES[mon.stage],
        stage_steps={STAGES[s]: n for s, n in sorted(mon.stage_step.items())},
        stage_after_disturbance=None if mon.stage_after is None else STAGES[mon.stage_after],
        arms_touch_steps=int(mon.arms_touch_steps), grabs=int(mon.grabs), empty_grasps=int(mon.empty_grasps),
        early_releases=int(mon.early_releases), crossing_steps=int(mon.crossing_steps),
        max_crossings=int(mon.max_crossings), towel_off_table=bool(mon.off_table), broken=mon.broken,
        max_gap_after_mm=round(1000 * mon.max_gap_after, 1),
        final_corner_gap_mm={s: [round(1000 * x, 1) for x in g] for s, g in chk.corner_gap.items()},
        final_area_ratio=round(chk.area_ratio, 4), final_violation=round(chk.violation, 4),
        disturbance=dist, actions=np.array(actions),
    )
    if not success:
        rec.failure = classify_failure(mon, crossings)
        if probe and not mon.broken:
            rec.expert_could_finish = expert_can_finish(env)
    return rec


def replay(env: TowelEnv, rec: EpisodeRecord, on_step=None) -> TowelState:
    """Re-execute a recorded episode (actions and its disturbance); returns the final state."""
    env.reset(rec.episode_seed)
    d = rec.disturbance
    for t, a in enumerate(rec.actions):
        if d is not None and d.step == t:
            d.apply(env)
        env.step(a)
        if on_step is not None:
            on_step(env)
    return read_state(env.model, TaskIndex(env.model), env.data)


# --- Aggregation ---------------------------------------------------------------------------


def summarize(records: list[EpisodeRecord]) -> dict:
    n = len(records)
    ok = [r for r in records if r.success]
    steps = [r.steps for r in ok]
    fails = [r for r in records if not r.success]
    stage_time = {}
    for s in STAGES[1:]:
        firsts = [r.stage_steps[s] for r in records if s in r.stage_steps]
        stage_time[s] = float(np.median(firsts)) if firsts else None
    return {
        "episodes": n,
        "mesh": sorted({r.mesh for r in records}),
        "success": len(ok),
        "success_rate": len(ok) / n,
        "stage_reached": {s: sum(STAGES.index(r.stage) >= i for r in records) for i, s in enumerate(STAGES) if i},
        "stage_first_step_median": stage_time,
        "disturbance_applied": sum(r.disturbance is not None and r.disturbance.step is not None for r in records),
        "infractions": {
            "arms_touch_episodes": sum(r.arms_touch_steps > 0 for r in records),
            "arms_touch_steps": sum(r.arms_touch_steps for r in records),
            "crossing_episodes": sum(r.crossing_steps > 0 for r in records),
            "max_crossings": max(r.max_crossings for r in records),
            "towel_off_table": sum(r.towel_off_table for r in records),
            "empty_grasps": sum(r.empty_grasps for r in records),
            "early_releases": sum(r.early_releases for r in records),
            "sim_broken": sum(r.broken is not None for r in records),
        },
        "grabs_median": float(np.median([r.grabs for r in records])),
        "max_gap_after_mm_median": float(np.median([r.max_gap_after_mm for r in records])),
        "steps_to_success": {"median": float(np.median(steps)), "p90": float(np.percentile(steps, 90)),
                             "max": int(max(steps))} if steps else None,
        "success_corner_gap_mm_max": max(max(g[0] for g in r.final_corner_gap_mm.values()) for r in ok) if ok else None,
        "failures": {c: sum(r.failure == c for r in fails) for c in FAILURE_CLASSES if any(r.failure == c for r in fails)},
        "failed_expert_could_finish": sum(bool(r.expert_could_finish) for r in fails),
        "failed_stuck": sum(r.expert_could_finish is False for r in fails),
    }

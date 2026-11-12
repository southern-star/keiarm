"""Closed-loop evaluation of the bimanual hand-over: stages, infractions, disturbances, failures.

The OpenArm counterpart of keiarm/eval.py. Per episode it records the furthest stage, the
first step each stage was reached, arm-arm contact, hard hits on the table, drops and the
bar leaving the table; failed episodes are classified by stage and probed with the expert.

Disturbance suite (one disturbance per episode, all decided by the seeds):
  move   before the right hand closes, the bar jumps to a new spot on the right side
  knock  after the right hand has lifted it, the bar is knocked back onto the table (right side)
  drop   after the left hand has taken it over, the bar falls back onto the table (right
         side): the whole hand-over starts again
  plate  while the left hand carries the bar, the plate moves to a new spot
"""

from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from keiarm.openarm.env import MAX_EPISODE_STEPS, OpenArmEnv
from keiarm.openarm.expert import Expert
from keiarm.openarm.rollout import SUCCESS_HOLD_STEPS
from keiarm.openarm.scene import (
    TABLE_CENTER,
    TABLE_HALF_SIZE,
    bar_axis,
    episode_rng,
    grip_points,
    sample_bar_pose,
    sample_plate_xy,
)
from keiarm.openarm.task import STAGES, TaskState, is_success, over_plate, read_state, stage_of

DISTURB_STREAM = 2
DISTURBANCES = ("none", "move", "knock", "drop", "plate")

# Normal force between a robot arm and the table that counts as a hard hit (impacts only;
# the clean expert never touches the table, see docs/milestone2.md).
TABLE_HIT_FORCE = 20.0  # [N]
LIFTED = 0.02  # [m] bar centre this far above lying flat counts as lifted
DROP_STEPS = 3  # no finger on a carried bar for this many steps = it was dropped
EXPERT_PROBE_STEPS = 450  # how long the expert gets when probing a failed episode's final state

FAILURE_CLASSES = ("grasp_failed", "lift_failed", "handover_failed", "release_failed",
                   "carry_failed", "place_failed", "dropped", "bar_off_table")


# --- Disturbances --------------------------------------------------------------------------


@dataclass
class Disturbance:
    """Moves the bar (or the plate) once, `delay` steps after the episode first reaches
    `after_stage` (0: from the start). `step` is filled in when it fires; replaying the
    episode applies it at that step."""

    kind: str
    after_stage: int
    delay: int
    x: float
    y: float
    yaw: float
    step: int | None = None

    def due(self, t: int, stage_step: dict[int, int]) -> bool:
        if self.step is not None:
            return t == self.step
        first = 0 if self.after_stage == 0 else stage_step.get(self.after_stage)
        return first is not None and t >= first + self.delay

    def apply(self, env: OpenArmEnv) -> None:
        if self.kind == "plate":
            env.move_plate([self.x, self.y])
        else:
            env.teleport_bar([self.x, self.y], self.yaw)


def _clear_of(xy, yaw: float, size, point, radius: float) -> bool:
    """Is the bar's footprint (a rectangle) at least `radius` away from `point` in xy?"""
    a = bar_axis(yaw)
    d = np.asarray(point, float)[:2] - np.asarray(xy, float)
    along, across = abs(float(d @ a)), abs(float(d[0] * a[1] - d[1] * a[0]))
    gap = np.hypot(max(0.0, along - size[1] / 2), max(0.0, across - size[0] / 2))
    return bool(gap >= radius)


def planned_disturbance(env: OpenArmEnv, kind: str) -> Disturbance | None:
    """The episode's disturbance of the given kind, a pure function of its seeds. Call after reset."""
    if kind == "none":
        return None
    scene, ep = env.scene, env.episode
    rng = episode_rng(ep.scene_seed, ep.episode_seed, DISTURB_STREAM)
    old_grip = grip_points(ep.bar_xy, ep.bar_yaw)["right"]
    if kind == "move":
        # Fires while the right hand approaches (it closes at about step 45-55); the new spot
        # keeps the bar 4 cm from where the hand is heading.
        delay = int(rng.integers(10, 40))
        while True:
            xy, yaw = sample_bar_pose(rng, scene)
            if _clear_of(xy, yaw, scene.bar_size, old_grip, 0.04):
                return Disturbance(kind, 0, delay, float(xy[0]), float(xy[1]), float(yaw))
    if kind == "knock":
        delay = int(rng.integers(5, 30))
        xy, yaw = sample_bar_pose(rng, scene)
        return Disturbance(kind, STAGES.index("lift"), delay, float(xy[0]), float(xy[1]), float(yaw))
    if kind == "drop":
        delay = int(rng.integers(0, 20))
        xy, yaw = sample_bar_pose(rng, scene)
        return Disturbance(kind, STAGES.index("left_carry"), delay, float(xy[0]), float(xy[1]), float(yaw))
    if kind == "plate":
        delay = int(rng.integers(0, 20))
        while True:
            xy = sample_plate_xy(rng, scene)
            if np.linalg.norm(xy - np.asarray(ep.plate_xy)) >= 0.05:
                return Disturbance(kind, STAGES.index("left_carry"), delay, float(xy[0]), float(xy[1]), 0.0)
    raise ValueError(f"unknown disturbance {kind!r}")


# --- Episode -------------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    scene_seed: int
    episode_seed: int
    disturb: str
    success: bool
    steps: int
    stage: str  # furthest stage reached
    stage_steps: dict[str, int]  # stage -> first control step (1-based count) it was reached
    arms_touch_steps: int
    table_hits: int
    drops: int
    bar_off_table: bool
    max_table_force: float
    failure: str | None = None  # one of FAILURE_CLASSES
    # Failed episodes only: could the expert still finish from the final state?
    expert_could_finish: bool | None = None
    disturbance: Disturbance | None = None
    actions: np.ndarray | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("actions")
        return d


class Monitor:
    """Tracks stage progress and infractions over one episode."""

    def __init__(self, env: OpenArmEnv):
        self.env = env
        ix = env.ix
        self.robot_geoms = ix.arm_geoms["left"] | ix.arm_geoms["right"]
        self.table = env.model.geom("table").id
        self.stage = 0
        self.stage_step: dict[int, int] = {}
        self.arms_touch_steps = self.table_hits = self.drops = 0
        self.max_table_force = 0.0
        self.bar_left_table = False
        self._hitting = self._carrying = False
        self._free = 0  # consecutive steps with no finger on the bar
        self._success_run = 0

    def _table_force(self) -> float:
        m, d = self.env.model, self.env.data
        total, f = 0.0, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == self.table and c.geom2 in self.robot_geoms) or (
                    c.geom2 == self.table and c.geom1 in self.robot_geoms):
                mujoco.mj_contactForce(m, d, i, f)
                total += f[0]
        return total

    @staticmethod
    def bar_off_table(st: TaskState) -> bool:
        x, y, z = st.bar_pos
        return bool(abs(x - TABLE_CENTER[0]) > TABLE_HALF_SIZE[0] or abs(y - TABLE_CENTER[1]) > TABLE_HALF_SIZE[1]
                    or z < -0.01)

    def bar_was_moved(self) -> None:
        """A disturbance took the bar: do not count that as the policy dropping it."""
        self._carrying = False

    def update(self, t: int) -> tuple[TaskState, bool]:
        """Call after env.step at control step t (0-based). Returns (state, success held)."""
        scene = self.env.scene
        st = self.env.privileged()
        # 'success' only counts once it has held (below).
        now = min(stage_of(st, scene), len(STAGES) - 2)
        for s in range(1, now + 1):
            self.stage_step.setdefault(s, t + 1)
        self.stage = max(self.stage, now)
        self.arms_touch_steps += st.arms_touch

        force = self._table_force()
        self.max_table_force = max(self.max_table_force, force)
        hitting = force > TABLE_HIT_FORCE
        self.table_hits += hitting and not self._hitting
        self._hitting = hitting

        # A drop: the bar was held in the air and then no finger touched it for DROP_STEPS
        # steps, except when it was let go low over the plate (a deliberate release). One
        # finger's contact flickers for a step now and then while carrying; that is no drop.
        held = st.held("left") or st.held("right")
        lifted = st.bar_pos[2] > scene.bar_size[2] / 2 + LIFTED
        touched = st.touches("left") or st.touches("right")
        self._free = 0 if touched else self._free + 1
        if self._carrying and self._free >= DROP_STEPS:
            placed = over_plate(st, scene) and st.bar_pos[2] < st.plate_top + scene.bar_size[2] / 2 + 0.015
            self.drops += not placed
            self._carrying = False
        if held and lifted:
            self._carrying = True

        self.bar_left_table |= self.bar_off_table(st)

        self._success_run = self._success_run + 1 if is_success(st, scene) else 0
        done = self._success_run >= SUCCESS_HOLD_STEPS
        if done:
            self.stage = len(STAGES) - 1
            self.stage_step.setdefault(self.stage, t + 1)
        return st, done


def expert_can_finish(env: OpenArmEnv, steps: int = EXPERT_PROBE_STEPS) -> bool:
    """From the current state, does the expert reach success within `steps`? (env untouched)"""
    data = env.scratch_copy()
    expert = Expert(env.model, env.scene)
    run = 0
    for _ in range(steps):
        action, _ = expert.act(data)
        env.step_data(data, action)
        run = run + 1 if is_success(read_state(env.model, env.ix, data), env.scene) else 0
        if run >= SUCCESS_HOLD_STEPS:
            return True
    return False


_STAGE_FAILURE = {"none": "grasp_failed", "reach": "grasp_failed", "grasp_right": "lift_failed",
                  "lift": "handover_failed", "handover": "release_failed", "left_carry": "carry_failed",
                  "over_plate": "place_failed"}


def classify_failure(stage: int, drops: int, off_table: bool) -> str:
    if off_table:
        return "bar_off_table"
    name = STAGES[stage]
    if drops and name in ("lift", "left_carry"):
        return "dropped"
    return _STAGE_FAILURE[name]


def run_episode(env: OpenArmEnv, policy, episode_seed: int, disturb: str = "none",
                max_steps: int = MAX_EPISODE_STEPS, on_step=None, probe: bool = True) -> EpisodeRecord:
    env.reset(episode_seed)
    policy.reset(env)
    dist = planned_disturbance(env, disturb)
    mon = Monitor(env)
    actions = []
    success = False
    st = env.privileged()
    for t in range(max_steps):
        if dist is not None and dist.step in (None, t) and dist.due(t, mon.stage_step):
            dist.step = t
            dist.apply(env)
            mon.bar_was_moved()
        action = np.clip(policy.act(env), *env.ctrl_range.T)
        env.step(action)
        actions.append(action)
        st, success = mon.update(t)
        if on_step is not None:
            on_step(env)
        if success:
            break
    rec = EpisodeRecord(
        scene_seed=env.scene.scene_seed, episode_seed=episode_seed, disturb=disturb, success=success,
        steps=len(actions), stage=STAGES[mon.stage],
        stage_steps={STAGES[s]: n for s, n in sorted(mon.stage_step.items())},
        arms_touch_steps=int(mon.arms_touch_steps), table_hits=int(mon.table_hits), drops=int(mon.drops),
        bar_off_table=bool(mon.bar_left_table), max_table_force=round(float(mon.max_table_force), 2),
        disturbance=dist, actions=np.array(actions),
    )
    if not success:
        rec.failure = classify_failure(mon.stage, mon.drops, Monitor.bar_off_table(st))
        if probe:
            rec.expert_could_finish = expert_can_finish(env)
    return rec


def replay(env: OpenArmEnv, rec: EpisodeRecord, on_step=None) -> TaskState:
    """Re-execute a recorded episode (actions and its disturbance); returns the final state."""
    env.reset(rec.episode_seed)
    d = rec.disturbance
    for t, a in enumerate(rec.actions):
        if d is not None and d.step == t:
            d.apply(env)
        env.step(a)
        if on_step is not None:
            on_step(env)
    return env.privileged()


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
        "success": len(ok),
        "success_rate": len(ok) / n,
        "stage_reached": {s: sum(STAGES.index(r.stage) >= i for r in records) for i, s in enumerate(STAGES) if i},
        "stage_first_step_median": stage_time,
        "disturbance_applied": sum(r.disturbance is not None and r.disturbance.step is not None for r in records),
        "infractions_per_episode": {
            "arms_touch_episodes": sum(r.arms_touch_steps > 0 for r in records) / n,
            "table_hits": sum(r.table_hits for r in records) / n,
            "drops": sum(r.drops for r in records) / n,
            "bar_off_table": sum(r.bar_off_table for r in records) / n,
        },
        "arms_touch_steps": sum(r.arms_touch_steps for r in records),
        "max_table_force": max(r.max_table_force for r in records),
        "steps_to_success": {"median": float(np.median(steps)), "p90": float(np.percentile(steps, 90)),
                             "max": int(max(steps))} if steps else None,
        "failures": {c: sum(r.failure == c for r in fails) for c in FAILURE_CLASSES if any(r.failure == c for r in fails)},
        "failed_expert_could_finish": sum(bool(r.expert_could_finish) for r in fails),
        "failed_stuck": sum(r.expert_could_finish is False for r in fails),
    }


"""Closed-loop evaluation: success, stage progress, infractions, failure logs, paired sign test."""

import math
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from keiarm import config
from keiarm.env import KeiArmEnv
from keiarm.expert import Expert
from keiarm.policies import Policy
from keiarm.rollout import SUCCESS_HOLD_STEPS, Teleport, planned_teleport
from keiarm.scene import PAN_AXIS_XY, TABLE_CENTER, TABLE_HALF_SIZE
from keiarm.task import STAGES, TaskState, is_success, over_plate, read_state, stage_of

# --- Infraction thresholds -------------------------------------------------------------
# Normal force between the robot and the table that counts as a hit. The arm's torque limit
# caps steady pushing at about 15 N, so this only fires on impacts. The clean expert peaks at
# 17 N in one of 100 eval episodes; with 0.02 rad action noise impacts reach 100-240 N.
TABLE_HIT_FORCE = 20.0  # [N]
WORKSPACE_RADIUS = 0.28  # [m] from the shoulder pan axis; beyond this the cube is out of reach
LIFTED = 0.015  # [m] cube centre this far above its resting height counts as lifted
EXPERT_PROBE_STEPS = 300  # how long the expert gets when probing a timed-out state

FAILURE_CLASSES = ("approach_failed", "grasp_failed", "dropped_in_transit", "transit_timeout",
                   "place_failed", "out_of_workspace")


@dataclass
class EpisodeRecord:
    scene_seed: int
    episode_seed: int
    success: bool
    steps: int
    stage: str  # furthest stage reached
    table_hits: int
    drops: int
    workspace_exits: int
    max_table_force: float
    failure: str | None = None  # one of FAILURE_CLASSES
    # Timed-out episodes only: could the expert still finish from the final state?
    # True = the policy stalled or dithered; False = the situation itself was stuck.
    expert_could_finish: bool | None = None
    teleport: Teleport | None = None  # the disturbance applied, if any (needed to replay)
    actions: np.ndarray | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("actions")
        return d


class Monitor:
    """Tracks stage progress and infractions over one episode."""

    def __init__(self, env: KeiArmEnv):
        self.env = env
        m = env.model
        root = m.body("base").id
        self.robot_geoms = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root}
        self.table = m.geom("table").id
        self.stage = 0
        self.table_hits = self.drops = self.workspace_exits = 0
        self.max_table_force = 0.0
        self._hitting = self._outside = False
        self._carrying = False
        self._success_run = 0

    def _table_force(self) -> float:
        m, d = self.env.model, self.env.data
        total, f = 0.0, np.zeros(6)
        for i, c in enumerate(d.contact):
            if (c.geom1 == self.table and c.geom2 in self.robot_geoms) or (
                    c.geom2 == self.table and c.geom1 in self.robot_geoms):
                mujoco.mj_contactForce(m, d, i, f)
                total += f[0]
        return total

    def cube_outside(self, st: TaskState) -> bool:
        x, y, z = st.cube_pos
        off_table = (abs(x - TABLE_CENTER[0]) > TABLE_HALF_SIZE[0] or abs(y - TABLE_CENTER[1]) > TABLE_HALF_SIZE[1]
                     or z < -0.01)
        return bool(off_table or np.hypot(x - PAN_AXIS_XY[0], y - PAN_AXIS_XY[1]) > WORKSPACE_RADIUS)

    def cube_was_moved(self) -> None:
        """An external disturbance took the cube: do not count that as the policy dropping it."""
        self._carrying = False

    def update(self) -> tuple[TaskState, bool]:
        """Call once after every env.step. Returns (state, episode finished successfully)."""
        scene = self.env.scene
        st = self.env.privileged()
        # 'success' only counts once it has held (below); a momentary success is 'over_plate'.
        self.stage = max(self.stage, min(stage_of(st, scene), len(STAGES) - 2))

        force = self._table_force()
        self.max_table_force = max(self.max_table_force, force)
        hitting = force > TABLE_HIT_FORCE
        self.table_hits += hitting and not self._hitting
        self._hitting = hitting

        # A drop: the cube was being carried in the air and left the gripper anywhere but
        # low over the plate (which is a deliberate release).
        lifted = st.cube_pos[2] > scene.cube_size / 2 + LIFTED
        if self._carrying and not st.held:
            released_on_plate = over_plate(st, scene) and st.cube_pos[2] < st.plate_top + scene.cube_size / 2 + 0.01
            self.drops += not released_on_plate
        self._carrying = st.held and lifted

        outside = self.cube_outside(st)
        self.workspace_exits += outside and not self._outside
        self._outside = outside

        self._success_run = self._success_run + 1 if is_success(st, scene) else 0
        done = self._success_run >= SUCCESS_HOLD_STEPS
        if done:
            self.stage = len(STAGES) - 1
        return st, done


def expert_can_finish(env: KeiArmEnv, steps: int = EXPERT_PROBE_STEPS) -> bool:
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


def classify_failure(stage: int, drops: int, cube_outside: bool) -> str:
    if cube_outside:
        return "out_of_workspace"
    name = STAGES[stage]
    if name in ("none", "reach"):
        return "approach_failed"
    if name == "grasp":
        return "grasp_failed"
    if name == "lift":
        return "dropped_in_transit" if drops else "transit_timeout"
    return "place_failed"  # got the cube over the plate (or even momentarily placed) but did not finish


def run_episode(env: KeiArmEnv, policy: Policy, episode_seed: int,
                max_steps: int = config.MAX_EPISODE_STEPS, on_step=None,
                disturb: int | str | None = None) -> EpisodeRecord:
    """Run one episode. `disturb`: None, a step number, or "random" (see planned_teleport)."""
    env.reset(episode_seed)
    policy.reset(env)
    teleport = planned_teleport(env, disturb)
    mon = Monitor(env)
    actions = []
    success = False
    st = env.privileged()
    for t in range(max_steps):
        if teleport is not None and t == teleport.step:
            teleport.apply(env)
            mon.cube_was_moved()
        action = np.clip(policy.act(env), *env.ctrl_range.T)
        env.step(action)
        actions.append(action)
        st, success = mon.update()
        if on_step is not None:
            on_step(env)
        if success:
            break
    rec = EpisodeRecord(
        scene_seed=env.scene.scene_seed, episode_seed=episode_seed, success=success, steps=len(actions),
        stage=STAGES[mon.stage], table_hits=int(mon.table_hits), drops=int(mon.drops),
        workspace_exits=int(mon.workspace_exits), max_table_force=round(float(mon.max_table_force), 2),
        actions=np.array(actions), teleport=teleport,
    )
    if not success:
        rec.failure = classify_failure(mon.stage, mon.drops, mon.cube_outside(st))
        rec.expert_could_finish = expert_can_finish(env)
    return rec


# --- Aggregation -------------------------------------------------------------------------


def summarize(records: list[EpisodeRecord]) -> dict:
    n = len(records)
    ok = [r for r in records if r.success]
    steps = [r.steps for r in ok]
    fails = [r for r in records if not r.success]
    return {
        "episodes": n,
        "success": len(ok),
        "success_rate": len(ok) / n,
        "stage_reached": {s: sum(STAGES.index(r.stage) >= i for r in records) for i, s in enumerate(STAGES) if i},
        "infractions_per_episode": {
            "table_hits": sum(r.table_hits for r in records) / n,
            "drops": sum(r.drops for r in records) / n,
            "workspace_exits": sum(r.workspace_exits for r in records) / n,
        },
        "steps_to_success": {"median": float(np.median(steps)), "p90": float(np.percentile(steps, 90)),
                             "max": int(max(steps))} if steps else None,
        "failures": {c: sum(r.failure == c for r in fails) for c in FAILURE_CLASSES if any(r.failure == c for r in fails)},
        "timeouts_expert_could_finish": sum(bool(r.expert_could_finish) for r in fails),
        "timeouts_stuck": sum(r.expert_could_finish is False for r in fails),
    }


def sign_test(a: dict[tuple[int, int], bool], b: dict[tuple[int, int], bool]) -> dict:
    """Paired two-sided exact sign test on per-episode success over the seeds both runs share."""
    keys = sorted(set(a) & set(b))
    a_only = sum(a[k] and not b[k] for k in keys)
    b_only = sum(b[k] and not a[k] for k in keys)
    n = a_only + b_only
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n if n else 1.0
    return {"paired_episodes": len(keys), "both": sum(a[k] and b[k] for k in keys),
            "neither": sum(not a[k] and not b[k] for k in keys), "a_only": a_only, "b_only": b_only,
            "p_value": min(1.0, 2 * tail)}

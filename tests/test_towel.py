"""Milestone 3 (towel half fold), Phases 0-3: scene, grasp rule, episode state, determinism,
the scripted fold through the environment, scene randomization, reach and rendering."""

import colorsys
import copy
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from keiarm import config
from keiarm.openarm.robot import ACTUATOR_NAMES, GRIPPER_CLOSED, GRIPPER_OPEN, SIDES, finger_joints
from keiarm.towel import eval as towel_eval
from keiarm.towel import metrics
from keiarm.towel.env import CAMERAS, TowelEnv
from keiarm.towel.grasp import GRAB_RADIUS, GraspState
from keiarm.towel.policies import ExpertPolicy, make_policy
from keiarm.towel.scene import (
    GRAB_SLOTS,
    TOWEL_CENTER,
    TOWEL_CENTER_JITTER,
    TOWEL_DEPTH_RANGE,
    TOWEL_MASS_RANGE,
    TOWEL_N,
    TOWEL_RADIUS,
    TOWEL_SOLVER_ITERATIONS,
    TOWEL_WIDTH_RANGE,
    TOWEL_YAW_RANGE,
    TOWEL_YOUNG_RANGE,
    build_xml,
    sample_episode,
    sample_scene,
)
from keiarm.towel.scripted import scripted_fold

SCENE = 100000


# --- scene ---------------------------------------------------------------------------------


def test_seeds_determine_scene_and_episode():
    assert sample_scene(3) == sample_scene(3)
    assert sample_scene(3) != sample_scene(4)
    s = sample_scene(3)
    assert sample_episode(s, 7) == sample_episode(s, 7)
    assert sample_episode(s, 7) != sample_episode(s, 8)
    assert build_xml(s) == build_xml(sample_scene(3))


def test_scene_ranges():
    hue_gaps = []
    for seed in range(200):
        s = sample_scene(seed)
        assert TOWEL_DEPTH_RANGE[0] <= s.towel_size[0] <= TOWEL_DEPTH_RANGE[1]
        assert TOWEL_WIDTH_RANGE[0] <= s.towel_size[1] <= TOWEL_WIDTH_RANGE[1]
        assert TOWEL_MASS_RANGE[0] <= s.towel_mass <= TOWEL_MASS_RANGE[1]
        assert TOWEL_YOUNG_RANGE[0] <= s.towel_young <= TOWEL_YOUNG_RANGE[1] + 0.1
        e = sample_episode(s, seed)
        assert all(abs(e.towel_xy[k] - TOWEL_CENTER[k]) <= TOWEL_CENTER_JITTER[k] for k in range(2))
        assert abs(e.towel_yaw) <= TOWEL_YAW_RANGE
        front = colorsys.rgb_to_hsv(*s.towel_rgb)[0]
        back = colorsys.rgb_to_hsv(*s.towel_back_rgb)[0]
        hue_gaps.append(min(abs(front - back), 1 - abs(front - back)))
    assert min(hue_gaps) > 0.3  # the back is a clearly different colour
    sizes = np.array([sample_scene(seed).towel_size for seed in range(200)])
    assert sizes.std(axis=0).min() > 0.003  # it actually varies


@pytest.fixture(scope="module")
def env():
    e = TowelEnv(SCENE, render=False)
    yield e
    e.close()


def test_model_settings(env):
    m = env.model
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE
    assert m.opt.solver == mujoco.mjtSolver.mjSOL_CG
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert m.opt.iterations == TOWEL_SOLVER_ITERATIONS
    assert m.opt.timestep == config.TIMESTEP
    assert env.towel.nvert == TOWEL_N**2
    assert m.nflex == 1 and m.flex_edgeequality[0] == 1  # edge constraints
    slots = [m.equality(f"{s}_grab{k}").id for s in SIDES for k in range(GRAB_SLOTS)]
    assert len(slots) == 2 * GRAB_SLOTS and not m.eq_active0[slots].any()
    # the towel meets the towel floor and the arms, not the table box
    table, floor = m.geom("table").id, m.geom("towel_floor").id
    ct, ca = m.flex_contype[0], m.flex_conaffinity[0]
    assert not (ct & m.geom_conaffinity[table] or m.geom_contype[table] & ca)
    assert ct & m.geom_conaffinity[floor]


def test_model_follows_the_scene(env):
    m, sc = env.model, env.scene
    v = env.towel.rest
    np.testing.assert_allclose(np.ptp(v[:, :2], axis=0), sc.towel_size, rtol=1e-6)
    assert np.isclose(m.body_mass[env.towel.vert_body].sum(), sc.towel_mass)
    cam = m.camera("cam_front").id
    np.testing.assert_allclose(m.cam_pos[cam], sc.cam_front_pos, atol=1e-6)
    assert np.isclose(m.cam_fovy[cam], sc.cam_front_fovy, atol=1e-3)


def test_reset_lays_the_towel_flat_at_the_episode_pose(env):
    env.reset(5)
    v = env.towel_verts()
    ep = env.episode
    np.testing.assert_allclose(v[:, :2].mean(axis=0), ep.towel_xy, atol=1e-9)
    edge = v[metrics.corner(env.towel, (-1, 0))] - v[metrics.corner(env.towel, (0, 0))]
    assert np.isclose(np.arctan2(edge[1], edge[0]), ep.towel_yaw, atol=1e-9)
    for _ in range(15):  # 0.5 s with the arms holding still
        env.step(np.array(ep.start_ctrl))
    v = env.towel_verts()
    assert abs(v[:, 2].min() - TOWEL_RADIUS) < 5e-4 and v[:, 2].max() < TOWEL_RADIUS + 1e-3
    assert not any(env.grasp.state.hands[s].latched for s in SIDES)


# --- grasp rule (driven directly through MjData, no arm motion) -------------------------------


def _set_gripper(env, side, closed: bool):
    m, d = env.model, env.data
    d.ctrl[m.actuator(f"{side}_finger1_ctrl").id] = GRIPPER_CLOSED[side] if closed else GRIPPER_OPEN[side]
    q = 0.0 if closed else GRIPPER_OPEN[side]
    for name in finger_joints(side):
        d.qpos[m.joint(name).qposadr[0]] = q


def _put_vertex(env, k, pos):
    tw = env.towel
    env.data.qpos[tw.vert_qadr[k]] = np.asarray(pos) - tw.rest[k]


def _fresh(env):
    env.reset(0)
    mujoco.mj_forward(env.model, env.data)
    return env.grasp


def test_close_grabs_nearest_vertices_and_open_releases(env):
    g = _fresh(env)
    tip = g.fingertip(env.data, "right")
    near = [10, 11, 12, 13, 14]  # five vertices in reach; only GRAB_SLOTS are taken, nearest first
    for r, k in enumerate(near):
        _put_vertex(env, k, tip + [0.002 * (r + 1), 0.0, 0.0])
    _set_gripper(env, "right", closed=True)
    mujoco.mj_forward(env.model, env.data)
    assert g.update(env.data) == [("right", "grab")]
    assert g.held("right") == near[:GRAB_SLOTS]
    slots = g.slots["right"]
    assert env.data.eq_active[slots].all()
    assert [env.model.eq_obj2id[s] for s in slots] == [env.towel.vert_body[k] for k in near[:GRAB_SLOTS]]
    assert g.anchor_error(env.data) < 1e-9  # anchored where they are: no jump
    assert g.update(env.data) == []  # still closed: nothing new
    _set_gripper(env, "right", closed=False)
    assert g.update(env.data) == [("right", "release")]
    assert g.held("right") == [] and not env.data.eq_active[slots].any()


def test_closed_hand_does_not_pick_up_later(env):
    g = _fresh(env)
    _set_gripper(env, "left", closed=True)
    mujoco.mj_forward(env.model, env.data)
    assert g.update(env.data) == [("left", "grab")]
    assert g.held("left") == []  # nothing within GRAB_RADIUS of the raised hand
    _put_vertex(env, 3, g.fingertip(env.data, "left"))
    mujoco.mj_forward(env.model, env.data)
    assert g.update(env.data) == [] and g.held("left") == []


def test_vertex_out_of_reach_is_not_grabbed(env):
    g = _fresh(env)
    tip = g.fingertip(env.data, "right")
    _put_vertex(env, 20, tip + [GRAB_RADIUS * 1.05, 0.0, 0.0])
    _put_vertex(env, 21, tip + [GRAB_RADIUS * 0.95, 0.0, 0.0])
    _set_gripper(env, "right", closed=True)
    mujoco.mj_forward(env.model, env.data)
    g.update(env.data)
    assert g.held("right") == [21]


def test_later_hand_takes_over_a_shared_vertex(env):
    g = _fresh(env)
    _put_vertex(env, 40, g.fingertip(env.data, "right"))
    _set_gripper(env, "right", closed=True)
    mujoco.mj_forward(env.model, env.data)
    g.update(env.data)
    assert g.held("right") == [40]
    _put_vertex(env, 40, g.fingertip(env.data, "left"))
    _set_gripper(env, "left", closed=True)
    mujoco.mj_forward(env.model, env.data)
    assert g.update(env.data) == [("left", "grab")]
    assert g.held("left") == [40] and g.held("right") == []
    assert env.data.eq_active[g.slots["left"][0]] and not env.data.eq_active[g.slots["right"]].any()
    assert g.state.hands["right"].latched  # still closed: cannot grab again until it opens


def test_grasp_state_round_trips():
    st = GraspState()
    st.hands["right"].latched = True
    st.hands["right"].verts = [3, 4]
    st.hands["right"].anchors = [[0.1, 0.2, 0.3], [0.0, 0.0, -0.1]]
    assert GraspState.from_dict(st.to_dict()) == st


# --- the scripted fold through the environment (Phase 0 pass criterion, one episode here;
#     scripts/run_scripted_towel.py runs 20) ------------------------------------------------


@pytest.fixture(scope="module")
def folded():
    """Run the scripted fold once, recording every action and the state after the grab."""
    e = TowelEnv(SCENE, render=False)
    e.reset(0)
    actions, saved = [], {}
    step = e.step

    def recording_step(action):
        actions.append(np.array(action, copy=True))
        events = step(action)
        if not saved and all(e.grasp.held(s) for s in SIDES):
            saved["state"], saved["at"] = e.get_state(), len(actions)
        return events

    e.step = recording_step
    res = scripted_fold(e)
    final = e.data.qpos.copy(), e.towel_verts()
    e.close()
    return res, actions, saved, final


def test_scripted_fold_completes(folded):
    res = folded[0]
    assert res.completed, res
    assert res.crossings == 0
    assert res.violation < 0.05
    assert 0.40 < res.area_ratio < 0.60
    assert res.max_anchor_err_mm < 5.0
    assert all(res.grabbed[s] == GRAB_SLOTS for s in SIDES)


def test_replay_is_deterministic(folded):
    _, actions, _, (qpos, verts) = folded
    e = TowelEnv(SCENE, render=False)
    e.reset(0)
    for a in actions:
        e.step(a)
    np.testing.assert_array_equal(e.data.qpos, qpos)
    np.testing.assert_array_equal(e.towel_verts(), verts)
    e.close()


def test_restoring_state_after_the_grab_reproduces_the_rest(folded):
    _, actions, saved, (qpos, verts) = folded
    assert saved, "both hands never held the towel"
    e = TowelEnv(SCENE, render=False)
    e.reset(1)  # a different episode: everything must come from the saved state
    e.set_state(saved["state"])
    assert all(e.grasp.held(s) for s in SIDES)
    for a in actions[saved["at"]:]:
        e.step(a)
    np.testing.assert_array_equal(e.data.qpos, qpos)
    np.testing.assert_array_equal(e.towel_verts(), verts)
    e.close()


def test_observation(gl):
    e = TowelEnv(SCENE, render=True)
    e.reset(0)
    obs = e.observe()
    assert list(obs.images) == list(CAMERAS)
    assert all(img.shape == (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3) for img in obs.images.values())
    assert obs.joints.shape == (len(ACTUATOR_NAMES),)
    e.close()


# --- Phase 1: reach over the spawn range, two-sided rendering, segmentation -----------------


def _towel_reach():
    path = Path(__file__).parents[1] / "tools" / "towel_reach.py"
    spec = importlib.util.spec_from_file_location("towel_reach", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._init()
    return mod


def test_fold_path_reachable_at_the_tightest_spawns():
    """The spawn range's tightest corners (closest to the pedestal, farthest out) pass the
    reach check of tools/towel_reach.py, which runs the whole range."""
    tr = _towel_reach()
    (cx, _), (jx, jy), yaw = TOWEL_CENTER, TOWEL_CENTER_JITTER, TOWEL_YAW_RANGE
    deep, narrow, wide = TOWEL_DEPTH_RANGE[1], TOWEL_WIDTH_RANGE[0], TOWEL_WIDTH_RANGE[1]
    for case in [((cx - jx, -jy), yaw, (deep, narrow)), ((cx - jx, jy), -yaw, (deep, narrow)),
                 ((cx + jx, 0.0), yaw, (deep, wide)), ((cx + jx, -jy), -yaw, (deep, wide))]:
        r = tr._fold_case(case)
        assert r["ok"], (case, r)


def _folded_state():
    e = TowelEnv(SCENE, render=False)
    e.reset(0)
    saved = {}

    def keep(env):
        if env.steps == 230:  # carried and lowered, still holding
            saved["state"] = env.get_state()

    scripted_fold(e, on_step=keep)
    e.close()
    return saved["state"]


def test_two_sided_rendering_and_segmentation(gl):
    state = _folded_state()
    e = TowelEnv(SCENE, render=True)
    e.reset(0)
    # flat, seen from above: no back faces are added, the towel is in the mask
    e._update_scene("cam_front")
    assert e._renderer.scene.ngeom == _scene_geoms(e)
    flat = e.towel_mask("cam_front")
    assert flat.mean() > 0.12
    # folded over: the top layer shows its back colour
    e.set_state(state)
    img = e.render("cam_front").astype(float)
    mask = e.towel_mask("cam_front")
    assert e._backside.add(e._renderer.scene, e.data) > 50
    back = np.array(e.scene.towel_back_rgb)
    front = np.array([e.scene.towel_rgb, e.scene.towel_rgb2])
    px = img[mask] / 255.0
    chroma = px / (px.sum(axis=1, keepdims=True) + 1e-9)  # compare hues, not brightness
    to_back = np.linalg.norm(chroma - back / back.sum(), axis=1)
    to_front = np.min([np.linalg.norm(chroma - f / f.sum(), axis=1) for f in front], axis=0)
    assert np.mean(to_back < to_front) > 0.3
    e.close()


def _scene_geoms(env) -> int:
    """Geoms MuJoCo itself puts in the scene for cam_front (without the back faces)."""
    r = env._renderer
    r.update_scene(env.data, camera="cam_front")
    return r.scene.ngeom


# --- Phase 2: the expert ----------------------------------------------------------------------


def _snapshot(env) -> dict:
    m, d = env.model, env.data
    full = mujoco.mjtState.mjSTATE_FULLPHYSICS | mujoco.mjtState.mjSTATE_USER | mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(m, full))
    mujoco.mj_getState(m, d, state, full)
    return {"state": state, "xpos": d.xpos.copy(), "flexvert": d.flexvert_xpos.copy(), "ncon": d.ncon,
            "eq_obj2id": m.eq_obj2id.copy(), "eq_data": m.eq_data.copy()}


def test_expert_act_is_a_pure_function_of_the_state():
    """act() neither modifies the model or data nor keeps state between calls: the same
    state gives the same action, also from a fresh expert, and from a restored copy of
    the episode state in another environment (DAgger labels states it did not visit)."""
    from keiarm.towel.expert import Expert

    env = TowelEnv(SCENE, render=False)
    env.reset(0)
    expert = Expert(env.model, env.scene)
    checks = 0
    for t in range(120):  # through grasp and into the carry
        action, _ = expert.act(env.data)
        if t in (10, 50, 60, 90, 119):
            before = _snapshot(env)
            again, _ = expert.act(env.data)
            fresh, _ = Expert(env.model, env.scene).act(env.data)
            after = _snapshot(env)
            for key in before:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
            np.testing.assert_array_equal(action, again)
            np.testing.assert_array_equal(action, fresh)
            # restored elsewhere: positions are recomputed from qpos by mj_forward, while after
            # mj_step they are those of the last substep, so compare with a forwarded copy
            other = TowelEnv(SCENE, render=False)
            other.reset(3)
            other.set_state(env.get_state())
            restored, _ = Expert(other.model, other.scene).act(other.data)
            fwd = copy.copy(env.data)
            mujoco.mj_forward(env.model, fwd)
            np.testing.assert_allclose(restored, expert.act(fwd)[0], atol=1e-9)
            other.close()
            checks += 1
        env.step(action)
    assert checks == 5 and any(env.grasp.held(s) for s in SIDES)
    env.close()


@pytest.mark.parametrize("mesh", [16, 10])
def test_expert_folds_the_towel(mesh):
    from keiarm.towel.rollout import run_expert
    from keiarm.towel.task import STAGES

    env = TowelEnv(SCENE + 1, render=False, n=mesh)
    res = run_expert(env, 0)
    env.close()
    assert res.success, (STAGES[res.stage], res.final)
    assert res.arms_touch_steps == 0
    assert res.final.violation == 0.0


def test_success_check_needs_the_hands_off_and_the_towel_still():
    from keiarm.towel.task import TaskIndex, check_fold, read_state

    env = TowelEnv(SCENE, render=False)
    env.reset(0)
    ix = TaskIndex(env.model)
    chk = check_fold(read_state(env.model, ix, env.data), ix, env.scene)
    assert not chk.success and chk.area_ratio > 0.9  # flat: corners far apart
    assert all(h > 0.1 for h, _ in chk.corner_gap.values())
    env.close()


# --- Phase 3: evaluation harness -----------------------------------------------------------

def test_fast_self_intersection_matches_the_reference():
    env = TowelEnv(SCENE, render=False)
    env.reset(0)
    tris = metrics.triangles(env.model, env.towel)
    si = metrics.SelfIntersection(tris)
    rng = np.random.default_rng(0)
    v0 = env.towel_verts()
    # fold the towel onto itself and poke vertices through the other layer
    half = np.arange(env.towel.nvert) // env.towel.n >= env.towel.n // 2
    cases = 0
    for _ in range(20):
        v = v0.copy()
        x0 = v0[:, 0].mean()
        v[half, 0] = 2 * x0 - v[half, 0]
        v[half, 2] += 0.004
        v[rng.integers(env.towel.nvert, size=3), 2] += rng.uniform(-0.01, 0.01, 3)
        ref = metrics.self_intersections(v, tris)
        cases += ref > 0
        assert si.count(v) == ref
    assert cases > 0
    env.close()


def test_disturbances_are_decided_by_the_seeds():
    env = TowelEnv(SCENE, render=False)
    for ep in range(5):
        env.reset(ep)
        for kind in towel_eval.DISTURBANCES[1:]:
            a, b = towel_eval.planned_disturbance(env, kind), towel_eval.planned_disturbance(env, kind)
            assert a == b and a.step is None
        d = towel_eval.planned_disturbance(env, "shift")
        assert towel_eval.SHIFT_DELAY[0] <= d.delay < towel_eval.SHIFT_DELAY[1]
        assert np.all(np.abs(np.array(d.xy) - TOWEL_CENTER) <= TOWEL_CENTER_JITTER)
        assert abs(d.yaw) <= TOWEL_YAW_RANGE
        old = towel_eval._towel_corners(env.episode.towel_xy, env.episode.towel_yaw, env.towel)
        new = towel_eval._towel_corners(d.xy, d.yaw, env.towel)
        assert np.linalg.norm(new - old, axis=1).min() >= towel_eval.SHIFT_MIN
        assert towel_eval.planned_disturbance(env, "drop").sides == list(SIDES)
        assert towel_eval.planned_disturbance(env, "slip").sides[0] in SIDES
    assert towel_eval.planned_disturbance(env, "none") is None
    env.close()


def test_baselines_get_the_right_stage_and_class():
    env = TowelEnv(SCENE, render=False)
    hold = towel_eval.run_episode(env, make_policy("hold"), 0, max_steps=40)
    assert not hold.success and hold.stage == "none" and hold.failure == "grasp_failed"
    assert hold.expert_could_finish
    frozen = towel_eval.run_episode(env, make_policy("freeze"), 0, max_steps=120)
    assert not frozen.success and frozen.stage == "grasp_both" and frozen.failure == "lift_failed"
    assert frozen.grabs == 2 and frozen.empty_grasps == 0
    assert frozen.expert_could_finish
    env.close()


class _LetGoEarly(ExpertPolicy):
    """The expert, but both hands open once the far corners are lifted."""

    def act(self, env):
        action = super().act(env)
        st = towel_eval.read_state(env.model, self.expert.ix, env.data)
        if towel_eval.stage_of(st, self.expert.ix, env.scene) >= towel_eval.STAGES.index("lift"):
            self.opened = True
        if getattr(self, "opened", False):
            for s in SIDES:
                action[ACTUATOR_NAMES.index(f"{s}_finger1_ctrl")] = GRIPPER_OPEN[s]
        return action


def test_early_release_and_missed_grasp_are_counted():
    env = TowelEnv(SCENE, render=False)
    rec = towel_eval.run_episode(env, _LetGoEarly(), 0, max_steps=110, probe=False)
    assert rec.early_releases == 2 and not rec.success
    env.close()


@pytest.mark.parametrize("kind", ["shift", "slip", "drop", "nudge"])
def test_expert_recovers_from_each_disturbance(kind):
    env = TowelEnv(SCENE, render=False)
    rec = towel_eval.run_episode(env, make_policy("expert"), 0, disturb=kind)
    d = rec.disturbance
    assert d is not None and d.step is not None
    assert rec.success, rec.to_json()
    assert rec.arms_touch_steps == 0 and rec.crossing_steps == 0 and not rec.towel_off_table
    if kind in ("slip", "drop"):
        assert rec.grabs == 2 + len(d.sides)  # each hand that lost the towel grasped it again
    if kind == "nudge":
        assert rec.max_gap_after_mm > 1000 * towel_eval.CORNER_H  # the corner was knocked off its place
        assert rec.grabs >= 3  # and grasped again
    if kind == "slip":  # replaying the actions and the disturbance gives the same final state
        final = env.towel_verts()
        towel_eval.replay(env, rec)
        np.testing.assert_array_equal(env.towel_verts(), final)
    env.close()


# --- Phase 4: DART, the browser export, the N = 16 records -------------------------------------


def test_expert_succeeds_under_dart_noise():
    """With DART noise the carry continues from the previous clean label, not from the noisy
    ctrl (Expert.act's `commanded`): continuing from the noisy ctrl integrated the noise and
    the expert never let go (0/100 at sigma 0.0065 before)."""
    env = TowelEnv(SCENE, render=False)
    try:
        rec = towel_eval.run_episode(env, make_policy("expert-noisy:0.0065"), 0, probe=False)
    finally:
        env.close()
    assert rec.success and rec.arms_touch_steps == 0 and rec.crossing_steps == 0


def test_layer_order_leaves_the_crease_pair_out():
    from keiarm.towel.task import TaskIndex

    env = TowelEnv(SCENE, render=False)
    lo, hi = TaskIndex(env.model).layers
    n = env.towel.n
    assert len(lo) == (n // 2 - 1) * n
    assert not set((lo // n).tolist()) & {n // 2 - 1} and not set((hi // n).tolist()) & {n // 2}


def test_record_frames_reproduce_the_python_outcome():
    """The browser recomputes the success check from a record's quantized frames (held counts,
    finger contact and vertex speed are recorded with them); it must agree with Python. This
    is the same computation in numpy."""
    import base64

    from keiarm.towel import export
    from keiarm.towel.task import (
        AREA_RANGE,
        CORNER_H,
        CORNER_V,
        MAX_SPEED,
        MAX_VIOLATION,
        SUCCESS_HOLD_STEPS,
        TaskIndex,
    )

    rec = export.record_frames(SCENE, 0, "none", "expert", n=10)
    env = TowelEnv(SCENE, render=False, n=10)
    ix = TaskIndex(env.model)
    size = env.scene.towel_size
    verts = np.frombuffer(base64.b64decode(rec["verts"]), np.int16).reshape(rec["frames"], -1, 3) * rec["vert_scale"]
    touch = np.frombuffer(base64.b64decode(rec["touch"]), np.uint8)
    speed = np.frombuffer(base64.b64decode(rec["speed"]), np.uint16) / 1000
    run, step = 0, None
    for k in range(1, rec["frames"]):
        v = verts[k]
        gaps = [(np.hypot(*(v[ix.far[s], :2] - v[ix.near[s], :2])), abs(v[ix.far[s], 2] - v[ix.near[s], 2]))
                for s in SIDES]
        area = metrics.convex_area(v) / (size[0] * size[1])
        ok = (all(h < CORNER_H and z < CORNER_V for h, z in gaps) and AREA_RANGE[0] <= area <= AREA_RANGE[1]
              and metrics.layer_violation(v, ix.layers) <= MAX_VIOLATION and not any(rec["held"][k])
              and not touch[k] and speed[k] <= MAX_SPEED)
        run = run + 1 if ok else 0
        if run >= SUCCESS_HOLD_STEPS:
            step = k
            break
    assert rec["python"]["success"] and step == rec["python"]["steps"]


def test_web_bundle_has_what_the_browser_needs():
    from keiarm.towel import export

    b = export.scene_bundle(SCENE, 8, [0], [(0, "nudge")])
    c = b["constants"]
    assert b["n"] == 8 and 'meshdir="assets"' in b["xml"] and len(b["episodes"]) == 2
    assert len(c["layers"][0]) == (8 // 2 - 1) * 8 and c["back_offset"] > 0
    nudge = b["episodes"][1]["planned_disturbances"]["nudge"]
    assert len(nudge["kick_verts"]) == len(nudge["kick_vel"]) > 0
    model = mujoco.MjModel.from_xml_string(b["xml"].replace('meshdir="assets"', f'meshdir="{_mesh_root()}"'))
    assert model.flex_vertnum[0] == 64


def _mesh_root() -> str:
    from keiarm.paths import OPENARM_XML

    return str(OPENARM_XML.parent / "assets")


# --- Phase 5: collection -------------------------------------------------------------------


def test_chunk_label_leaves_the_episode_untouched():
    """The look-ahead runs on the environment (the grasp rule writes the model) and must
    restore it bit-exactly, including the grasp state."""
    from keiarm.towel.expert import Expert
    from keigrasp.towel import expert_chunk

    def run(look):
        e = TowelEnv(5, render=False)
        e.reset(0)
        ex = Expert(e.model, e.scene)
        chunks = []
        for t in range(90):
            if look and t % 30 == 29:
                chunks.append(expert_chunk(e, ex, None, length=10))
            a = ex.act(e.data)[0]
            e.step(a if not chunks or t % 30 != 29 else chunks[-1][0])
        out = e.data.qpos.copy(), e.grasp.get_state(), chunks
        e.close()
        return out

    q0, g0, _ = run(False)
    q1, g1, chunks = run(True)
    assert len(chunks) == 3 and chunks[0].shape == (10, len(ACTUATOR_NAMES))
    np.testing.assert_allclose(q0, q1, atol=1e-6)  # the expert reads fresher kinematics after a restore
    assert [h.verts for h in g0.hands.values()] == [h.verts for h in g1.hands.values()]


def test_collection_refuses_eval_seeds():
    from keigrasp.towel import collect_episode

    with pytest.raises(ValueError):
        collect_episode(SCENE, 0, max_steps=1)

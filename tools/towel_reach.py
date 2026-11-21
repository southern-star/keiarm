"""Towel half fold: reach of both hands over the whole spawn range (milestone 3, Phase 1).

Two checks, both top-down with the fold's jaw yaw (HOME_YAW + YAW_OFFSET, keiarm/towel/scripted.py):

1. Map: for each hand, the cells (x, y) of the table where the pinch point reaches the grasp
   height (MIN_PINCH_Z) and the carry height (0.12 m) with a collision-free pose, the other
   arm at its start pose. The spawn range's grasp and place points are drawn on top.
2. Fold paths: for every corner of the spawn range (towel centre x, y, yaw and side length
   at their extremes, 3 x 3 x 3 x 2 x 2 = 108) plus random samples inside it, both arms follow the
   fold's waypoints together (above the far corners, down, the carry arc, place, lift while
   letting go, back to the start pinch point),
   each IK warm-started from the previous waypoint. A waypoint passes if both arms reach it
   (position error < 2 mm, rotation error < 0.02 rad) and no arm comes within 2 mm of the table, the
   pedestal, the floor or the other arm, with the fingers open (as far as the fold opens
   them, scripted.OPENED) and closed.

The towel is moved out of the way: contact with it is allowed (it is what the hands grab).
Output: docs/towel/reach.{png,json}. Exit status 1 if any fold path fails.

Usage: uv run python tools/towel_reach.py [--random 200] [--workers 10]
"""

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from keiarm.openarm.ik import ArmIK, topdown_rotation
from keiarm.openarm.robot import HOME_ARM_QPOS, HOME_PINCH, HOME_YAW, SIDES, finger_joints
from keiarm.towel.scene import (
    TOWEL_CENTER,
    TOWEL_CENTER_JITTER,
    TOWEL_DEPTH_RANGE,
    TOWEL_WIDTH_RANGE,
    TOWEL_YAW_RANGE,
    TowelIndex,
    build_model,
    reset_data,
    sample_episode,
    sample_scene,
)
from keiarm.towel.scripted import (
    GRASP_INSET,
    LIFT_RATIO,
    MIN_PINCH_Z,
    OPENED,
    PLACE_SHORT,
    RELEASE_LIFT,
    YAW_OFFSET,
    arc,
)

OUT = Path("docs/towel")
CARRY_Z = 0.12  # above the far corners before descending
POS_TOL, ROT_TOL = 2e-3, 0.02
CLEARANCE = 0.002  # an arm closer than this to anything counts as a contact (geom margin)
# local towel coordinates (+x = far from the robot, +y = left) of each hand's corners
FAR = {"right": (0.5, -0.5), "left": (0.5, 0.5)}
NEAR = {"right": (-0.5, -0.5), "left": (-0.5, 0.5)}

_model = _data = _iks = None


def _init():
    global _model, _data, _iks
    scene = sample_scene(0)
    _model = build_model(scene)
    _model.geom_margin[:] = CLEARANCE
    _data = mujoco.MjData(_model)
    reset_data(_model, _data, TowelIndex(_model), sample_episode(scene, 0))
    TowelIndex(_model).place(_data, (3.0, 3.0), 0.0)  # towel out of the way
    for s in SIDES:
        _data.qpos[ArmIK(_model, s).qadr] = HOME_ARM_QPOS[s]
    _iks = {s: ArmIK(_model, s) for s in SIDES}


def _set_fingers(d, value: dict) -> None:
    for s in SIDES:
        for name in finger_joints(s):
            d.qpos[_model.joint(name).qposadr[0]] = value[s]


def _arm_contacts(m, d) -> list[str]:
    """Contacts of an arm with anything but the towel."""
    bad = []
    for c in d.contact[: d.ncon]:
        if c.geom1 < 0 or c.geom2 < 0:  # flex contact
            continue
        n1, n2 = (m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
        if n1.startswith("towel") or n2.startswith("towel"):
            continue
        if n1.startswith("openarm_") or n2.startswith("openarm_"):
            same_arm = n1.split("_")[1:2] == n2.split("_")[1:2] and n1.startswith("openarm_") and n2.startswith("openarm_")
            if not same_arm:
                bad.append(f"{n1}|{n2}")
    return bad


def _collision_free(d) -> list[str]:
    bad = []
    for fingers in (OPENED, {s: 0.0 for s in SIDES}):  # open, jaws closed
        _set_fingers(d, fingers)
        mujoco.mj_forward(_model, d)
        bad += _arm_contacts(_model, d)
    return sorted(set(bad))


def _yaw(side: str) -> float:
    return HOME_YAW[side] + YAW_OFFSET[side]


# --- 1. map -----------------------------------------------------------------------------------


def _map_cell(args):
    side, x, y, z = args
    d = mujoco.MjData(_model)
    d.qpos[:] = _data.qpos
    ik = _iks[side]
    q, ep, er = ik.solve(d.qpos, np.array(HOME_ARM_QPOS[side]), [x, y, z], topdown_rotation(_yaw(side)), iters=250,
                         q_rest=np.array(HOME_ARM_QPOS[side]))
    if ep > POS_TOL or er > ROT_TOL:
        return 0
    d.qpos[ik.qadr] = q
    return 0 if _collision_free(d) else 1


# --- 2. fold paths ----------------------------------------------------------------------------


def towel_point(center, yaw, size, local, inset=0.0):
    """World xy of a towel point given in units of its (depth, width), moved `inset` towards the centre."""
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.array(local) * np.array(size)
    p = p - inset * p / np.linalg.norm(p)
    return np.array(center) + [c * p[0] - s * p[1], s * p[0] + c * p[1]]


def fold_waypoints(center, yaw, size) -> list[tuple[str, dict]]:
    """(name, {side: pinch target}) along the scripted fold."""
    grasp = {s: towel_point(center, yaw, size, FAR[s], GRASP_INSET) for s in SIDES}
    shift = {s: towel_point(center, yaw, size, NEAR[s]) - towel_point(center, yaw, size, FAR[s]) for s in SIDES}
    shift = {s: v - PLACE_SHORT * v / np.linalg.norm(v) for s, v in shift.items()}
    p0 = {s: np.array([*grasp[s], MIN_PINCH_Z]) for s in SIDES}
    p1 = {s: np.array([*(grasp[s] + shift[s]), MIN_PINCH_Z + 0.008]) for s in SIDES}
    wps = [("above", {s: np.array([*grasp[s], CARRY_Z]) for s in SIDES}), ("grasp", p0)]
    for k in range(1, 9):
        u = k / 8
        wps.append((f"carry{k}", {s: arc(p0[s], p1[s], u, LIFT_RATIO * np.linalg.norm((p1[s] - p0[s])[:2]) / 2)
                                  for s in SIDES}))
    lift = {s: p1[s] + [0.0, 0.0, RELEASE_LIFT] for s in SIDES}
    wps.append(("release", lift))
    for k in range(1, 5):  # straight back to the start pinch point
        wps.append((f"retreat{k}", {s: lift[s] + k / 4 * (np.array(HOME_PINCH[s]) - lift[s]) for s in SIDES}))
    return wps


def _fold_case(case):
    center, yaw, size = case
    d = mujoco.MjData(_model)
    d.qpos[:] = _data.qpos
    q = {s: np.array(HOME_ARM_QPOS[s]) for s in SIDES}
    worst = {"pos_mm": 0.0, "rot": 0.0}
    for name, targets in fold_waypoints(center, yaw, size):
        for s in SIDES:
            ik = _iks[s]
            q[s], ep, er = ik.solve(d.qpos, q[s], targets[s], topdown_rotation(_yaw(s)), iters=250,
                                    q_rest=np.array(HOME_ARM_QPOS[s]))
            worst["pos_mm"] = max(worst["pos_mm"], ep * 1000)
            worst["rot"] = max(worst["rot"], er)
            if ep > POS_TOL or er > ROT_TOL:
                return {"ok": False, "at": name, "why": f"{s} IK {ep * 1000:.1f} mm {er:.3f} rad", **worst}
            d.qpos[ik.qadr] = q[s]
        bad = _collision_free(d)
        if bad:
            return {"ok": False, "at": name, "why": "contact " + ", ".join(bad), **worst}
    return {"ok": True, **worst}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=200, help="random spawn samples besides the 108 extremes")
    ap.add_argument("--step", type=float, default=0.02, help="map cell [m]")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    cx, cy = TOWEL_CENTER
    jx, jy = TOWEL_CENTER_JITTER
    extremes = [((x, y), yaw, (sx, sy)) for x, y, yaw, sx, sy in itertools.product(
        (cx - jx, cx, cx + jx), (cy - jy, cy, cy + jy), (-TOWEL_YAW_RANGE, 0.0, TOWEL_YAW_RANGE),
        TOWEL_DEPTH_RANGE, TOWEL_WIDTH_RANGE)]
    rng = np.random.default_rng(0)
    randoms = [((cx + rng.uniform(-jx, jx), cy + rng.uniform(-jy, jy)), rng.uniform(-TOWEL_YAW_RANGE, TOWEL_YAW_RANGE),
                (rng.uniform(*TOWEL_DEPTH_RANGE), rng.uniform(*TOWEL_WIDTH_RANGE))) for _ in range(args.random)]
    cases = extremes + randoms
    xs = np.round(np.arange(0.04, 0.40 + 1e-9, args.step), 4)
    ys = np.round(np.arange(-0.30, 0.30 + 1e-9, args.step), 4)
    heights = {"grasp": MIN_PINCH_Z, "carry": CARRY_Z}
    cells = [(s, x, y, z) for s in SIDES for z in heights.values() for y in ys for x in xs]
    with Pool(args.workers, initializer=_init) as pool:
        results = pool.map(_fold_case, cases, chunksize=4)
        flat = pool.map(_map_cell, cells, chunksize=16)
    reach = np.array(flat).reshape(len(SIDES), len(heights), len(ys), len(xs))

    fails = [dict(center=[round(float(v), 4) for v in c[0]], yaw_deg=round(float(np.degrees(c[1])), 2),
                  size=[round(float(v), 4) for v in c[2]], **r) for c, r in zip(cases, results) if not r["ok"]]
    summary = {
        "cases": len(cases), "extremes": len(extremes), "random": len(randoms),
        "passed": sum(r["ok"] for r in results),
        "worst_pos_mm": round(max(r["pos_mm"] for r in results), 3),
        "worst_rot_rad": round(max(r["rot"] for r in results), 4),
        "jaw_yaw_deg": {s: round(float(np.degrees(_yaw(s))), 1) for s in SIDES},
        "spawn": {"center": TOWEL_CENTER, "jitter": TOWEL_CENTER_JITTER, "yaw_deg": round(float(np.degrees(TOWEL_YAW_RANGE)), 2),
                  "depth": TOWEL_DEPTH_RANGE, "width": TOWEL_WIDTH_RANGE},
        "failures": fails[:20],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "reach.json", "w") as f:
        json.dump({"summary": summary, "xs": xs.tolist(), "ys": ys.tolist(), "heights": heights,
                   "reachable": {s: {h: reach[i, k].astype(int).tolist() for k, h in enumerate(heights)}
                                 for i, s in enumerate(SIDES)}}, f)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=1))
    for fl in fails[:10]:
        print("FAIL", fl)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    ext = [xs[0] - args.step / 2, xs[-1] + args.step / 2, ys[0] - args.step / 2, ys[-1] + args.step / 2]
    cmap = matplotlib.colors.ListedColormap(["#eeeeee", "#e07b39", "#3a7bd5", "#6a3d9a"])
    for ax, (k, h) in zip(axes, enumerate(heights)):
        img = reach[SIDES.index("right"), k] + 2 * reach[SIDES.index("left"), k]
        ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=3)
        ax.set_title(f"{h}: pinch z = {heights[h]} m")
        ax.set_xlabel("x [m]")
        for (center, yaw, size), r in zip(cases, results):
            for s in SIDES:
                g = towel_point(center, yaw, size, FAR[s], GRASP_INSET)
                p = fold_waypoints(center, yaw, size)[-6][1][s][:2]
                ax.plot(*g, ".", ms=2, color="k" if r["ok"] else "r")
                ax.plot(*p, ".", ms=2, color="0.4" if r["ok"] else "r")
        for size in zip(TOWEL_DEPTH_RANGE, TOWEL_WIDTH_RANGE):  # smallest and largest towel
            corners = [towel_point(TOWEL_CENTER, 0.0, size, c) for c in ((.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5), (.5, .5))]
            ax.plot(*np.array(corners).T, "k-", lw=0.6)
        ax.plot([0, 0], [-0.152, 0.152], "ko")
    axes[0].set_ylabel("y [m]")
    fig.suptitle(f"Towel fold reach, jaw yaw = start yaw ± 45° (orange = right, blue = left, purple = both). "
                 f"Dots: grasp (black) and place (grey) points of {len(cases)} spawns; "
                 f"fold paths passed {summary['passed']}/{len(cases)}", fontsize=9)
    fig.savefig(OUT / "reach.png", dpi=100, bbox_inches="tight")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

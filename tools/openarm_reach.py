"""OpenArm top-down reachability over the table, with collisions (milestone 2, Phase 1).

A cell (x, y) is reachable for one arm at height z if, for every jaw yaw psi in the
object's range, some equivalent yaw psi + k * period has an IK solution with the fingers
straight down whose pose touches nothing: not the table, the pedestal, the floor, nor the
other arm (held at the start pose).

  --object cube: psi = start yaw + [0, 90 deg), period 90 deg (faces repeat every 90 deg)
  --object bar:  psi = start yaw + phi with the bar's long axis phi in [-20, 45] deg from the
                 world y axis (jaws close across the bar), and no equivalent yaw: the hand
                 must hold the bar the way that turns it to lie along y for the hand-over
                 (with the jaws turned 180 deg it would have to rotate past a joint limit)

Heights (pinch point above the table): grasp (centre of a 4 cm object) and carry.
Output: docs/openarm/reach_<object>.{png,json}.

Usage: uv run python tools/openarm_reach.py --object bar [--step 0.025] [--workers 6]
"""

import argparse
import json
from multiprocessing import Pool

import mujoco
import numpy as np

from keiarm.openarm.ik import ArmIK, topdown_rotation
from keiarm.openarm.robot import HOME_ARM_QPOS, HOME_YAW, SIDES
from keiarm.openarm.scene import build_model, reset_data, sample_episode, sample_scene, set_bar_pose

HEIGHTS = {"grasp": 0.02, "carry": 0.12}
# (jaw yaws relative to the side's start yaw, period of equivalent yaws)
OBJECTS = {
    "cube": (np.radians(np.arange(0.0, 90.0, 15.0)), np.pi / 2),
    "bar": (np.radians(np.linspace(-20.0, 45.0, 14)), 2 * np.pi),
}

_model = None


_scene = None


def _init():
    global _model, _scene
    _scene = sample_scene(0)
    _model = build_model(_scene)


def _collides(model, data) -> bool:
    for c in data.contact[: data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        n1, n2 = model.body(b1).name, model.body(b2).name
        if {"bar", "plate"} & {n1, n2}:
            continue
        if n1.startswith("openarm_") or n2.startswith("openarm_"):
            return True
    return False


def _solve_cell(args):
    obj, side, x, y, z = args
    psis, period = OBJECTS[obj]
    m = _model
    d = mujoco.MjData(m)
    reset_data(m, d, _scene, sample_episode(_scene, 0))
    set_bar_pose(m, d, (2.0, 2.0), 0.0, 1.0)  # objects out of the way
    d.mocap_pos[m.body("plate").mocapid[0]] = [2.0, -2.0, 1.0]
    ik = ArmIK(m, side)
    home = np.array(HOME_ARM_QPOS[side])
    mirror = 1.0 if side == "right" else -1.0  # the left hand sees the mirror image of the bar
    for psi in HOME_YAW[side] + mirror * psis:
        # equivalent yaws, nearest to the start yaw first
        n_equiv = round(2 * np.pi / period)
        yaws = sorted((psi + k * period for k in range(n_equiv)), key=lambda a: abs(np.angle(np.exp(1j * (a - HOME_YAW[side])))))
        ok = False
        for yaw in yaws:
            q, ep, er = ik.solve(d.qpos, home, [x, y, z], topdown_rotation(yaw), iters=250)
            if ep > 2e-3 or er > 0.02:
                continue
            d.qpos[ik.qadr] = q
            mujoco.mj_forward(m, d)
            free = not _collides(m, d)
            d.qpos[ik.qadr] = home
            if free:
                ok = True
                break
        if not ok:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", choices=sorted(OBJECTS), default="bar")
    ap.add_argument("--step", type=float, default=0.025)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    xs = np.round(np.arange(0.075, 0.55 + 1e-9, args.step), 4)
    ys = np.round(np.arange(-0.50, 0.50 + 1e-9, args.step), 4)
    jobs = [(args.object, side, x, y, z) for side in SIDES for z in HEIGHTS.values() for y in ys for x in xs]
    with Pool(args.workers, initializer=_init) as pool:
        flat = pool.map(_solve_cell, jobs, chunksize=8)
    res = np.array(flat).reshape(len(SIDES), len(HEIGHTS), len(ys), len(xs))
    out = {"xs": xs.tolist(), "ys": ys.tolist(), "heights": HEIGHTS, "object": args.object,
           "psi_deg": np.degrees(OBJECTS[args.object][0]).tolist(),
           "reachable": {side: {h: res[i, j].astype(int).tolist() for j, h in enumerate(HEIGHTS)}
                         for i, side in enumerate(SIDES)}}
    with open(f"docs/openarm/reach_{args.object}.json", "w") as f:
        json.dump(out, f)
    both = {side: res[i, 0] & res[i, 1] for i, side in enumerate(SIDES)}
    for side in SIDES:
        print(side, {h: int(res[SIDES.index(side), j].sum()) for j, h in enumerate(HEIGHTS)},
              "grasp&carry", int(both[side].sum()))
    shared = both["left"] & both["right"]
    print("both arms (grasp&carry):", int(shared.sum()), "cells")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    ext = [xs[0] - args.step / 2, xs[-1] + args.step / 2, ys[0] - args.step / 2, ys[-1] + args.step / 2]
    cmap = matplotlib.colors.ListedColormap(["#eeeeee", "#e07b39", "#3a7bd5", "#6a3d9a"])
    for ax, (j, h) in zip(axes[:2], enumerate(HEIGHTS)):
        img = res[SIDES.index("right"), j].astype(int) + 2 * res[SIDES.index("left"), j].astype(int)
        ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=3)
        ax.set_title(f"{h}: pinch z = {HEIGHTS[h]} m")
    img = both["right"].astype(int) + 2 * both["left"].astype(int)
    axes[2].imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=3)
    axes[2].set_title("grasp and carry")
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.plot([0, 0], [-0.152, 0.152], "ko")
    axes[0].set_ylabel("y [m]")
    fig.suptitle(f"OpenArm top-down reach for a {args.object} (all yaws in range), collision-free "
                 "(orange = right only, blue = left only, purple = both)")
    fig.savefig(f"docs/openarm/reach_{args.object}.png", dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()

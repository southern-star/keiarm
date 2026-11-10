"""Brute-force reachability of top-down grasp poses over the table (design doc 5.5).

For every grid point and height, solve IK for the gripper site with the approach axis
pointing straight down. If that fails, retry with a weak orientation cost and record how
far from vertical the gripper ends up. Poses where the arm penetrates the table are
rejected. Kinematics only.

Usage: uv run python tools/reachability.py   (writes docs/reachability.png and .json)
"""

import json
import time
from multiprocessing import Pool

import mujoco
import numpy as np

from keiarm.ik import initial_guesses, solve_arm_ik
from keiarm.paths import REPO_ROOT
from keiarm.robot import so101_spec

STEP = 0.02
XS = np.round(np.arange(-0.10, 0.50 + 1e-9, STEP), 3)
YS = np.round(np.arange(-0.40, 0.40 + 1e-9, STEP), 3)
# Heights of the gripper site above the table. The first two (grasp, hover/carry) define the
# spawn region; the higher ones show how quickly strict top-down reach disappears.
HEIGHTS = (0.015, 0.06, 0.09, 0.12)
N_CORE = 2
POS_TOL = 1e-3
STRICT_TILT = np.radians(2.0)
GRIPPER_OPEN = 0.6

_model = None


def _get_model() -> mujoco.MjModel:
    global _model
    if _model is None:
        spec = so101_spec()
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
        _model = spec.compile()
    return _model


def _hits_table(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> bool:
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    table = model.geom("table").id
    return any(table in (c.geom1, c.geom2) and c.dist < -1e-3 for c in data.contact)


def solve_point(p: tuple[float, float, float]) -> float:
    """Return tilt from vertical in degrees (0 = strict top-down), or nan if unreachable."""
    model = _get_model()
    data = mujoco.MjData(model)
    target = np.array(p)
    best = np.nan
    for cost in (1.0, 0.02):
        for q0 in initial_guesses(model, target):
            q0[model.joint("gripper").qposadr[0]] = GRIPPER_OPEN
            r = solve_arm_ik(model, target, q0, orientation_cost=cost, max_iters=80)
            if r.pos_err > POS_TOL or _hits_table(model, data, r.q):
                continue
            tilt = np.degrees(r.tilt)
            if r.tilt < STRICT_TILT:
                return 0.0
            best = tilt if np.isnan(best) else min(best, tilt)
    return best


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [(x, y, z) for z in HEIGHTS for x in XS for y in YS]
    t0 = time.perf_counter()
    with Pool() as pool:
        tilts = np.array(pool.map(solve_point, points, chunksize=32))
    tilts = tilts.reshape(len(HEIGHTS), len(XS), len(YS))
    print(f"{len(points)} IK problems in {time.perf_counter() - t0:.0f}s")

    strict = tilts == 0.0
    all_strict = strict[:N_CORE].all(axis=0)
    model = _get_model()
    pan_xy = model.body("shoulder").pos[:2]
    gx, gy = np.meshgrid(XS, YS, indexing="ij")
    radius = np.hypot(gx - pan_xy[0], gy - pan_xy[1])
    azimuth = np.degrees(np.arctan2(gy - pan_xy[1], gx - pan_xy[0]))
    for i, z in enumerate(HEIGHTS):
        r = radius[strict[i]]
        span = f"radius {r.min():.3f}-{r.max():.3f} m" if r.size else "nowhere"
        print(f"z={z:.3f}: strict top-down at {strict[i].sum()} / {strict[i].size} points, {span}; "
              f"any tilt: {np.isfinite(tilts[i]).sum()}")
    r = radius[all_strict]
    print(f"heights {HEIGHTS[:N_CORE]}: {all_strict.sum()} points, radius {r.min():.3f}-{r.max():.3f} m, "
          f"azimuth {azimuth[all_strict].min():.0f}..{azimuth[all_strict].max():.0f} deg")
    # Largest annulus fully covered within |azimuth| <= 60 deg, for the spawn region.
    sector = np.abs(azimuth) <= 60
    radii = np.round(np.arange(0.08, 0.45, 0.01), 2)
    ok = [bool(all_strict[sector & (np.abs(radius - rr) <= STEP)].all()) for rr in radii]
    print("radius -> fully covered within +-60deg:", {float(a): b for a, b in zip(radii, ok)})

    out = REPO_ROOT / "docs"
    (out / "reachability.json").write_text(json.dumps({
        "step": STEP, "xs": XS.tolist(), "ys": YS.tolist(), "heights": list(HEIGHTS),
        "pan_axis_xy": pan_xy.tolist(), "core_heights": list(HEIGHTS[:N_CORE]), "strict_tilt_deg": 2.0,
        "tilt_deg": [[[None if np.isnan(v) else round(float(v), 1) for v in row] for row in layer]
                     for layer in tilts],
    }))

    fig, axes = plt.subplots(1, len(HEIGHTS) + 1, figsize=(4.4 * (len(HEIGHTS) + 1), 5.2), sharey=True)
    extent = [YS[0] - STEP / 2, YS[-1] + STEP / 2, XS[0] - STEP / 2, XS[-1] + STEP / 2]
    for ax, layer, z in zip(axes, tilts, HEIGHTS):
        im = ax.imshow(np.ma.masked_invalid(layer), origin="lower", extent=extent, cmap="viridis_r",
                       vmin=0, vmax=90, aspect="equal")
        ax.set_title(f"site height {z * 100:.1f} cm")
    axes[-1].imshow(np.where(all_strict, 1.0, np.nan), origin="lower", extent=extent, cmap="Greens",
                    vmin=0, vmax=1.3, aspect="equal")
    axes[-1].set_title(f"strict top-down at {HEIGHTS[0] * 100:.1f} and {HEIGHTS[1] * 100:.0f} cm")
    for ax in axes:
        ax.plot(pan_xy[1], pan_xy[0], "r+", ms=10)
        ax.set_xlabel("y [m]")
        ax.invert_xaxis()  # +y is to the robot's left when looking along +x
        ax.set_facecolor("0.85")
    axes[0].set_ylabel("x [m] (forward)")
    fig.colorbar(im, ax=axes[:-1], label="tilt from vertical [deg] (grey = unreachable)", shrink=0.8)
    fig.suptitle("SO-101 top-down grasp reachability (red + = shoulder pan axis)")
    fig.savefig(out / "reachability.png", dpi=110, bbox_inches="tight")
    print(f"wrote {out / 'reachability.png'}")


if __name__ == "__main__":
    main()

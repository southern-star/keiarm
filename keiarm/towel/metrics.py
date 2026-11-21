"""Measurements of the towel's shape (from spikes/cloth/cloth.py). Privileged: evaluation only.

The fold in milestone 3 brings the far edge (local i = n - 1) onto the near edge (i = 0):
the right hand carries corner (n-1, 0) onto (0, 0) and the left hand (n-1, n-1) onto (0, n-1).
"""

import mujoco
import numpy as np

from keiarm.towel.scene import TowelIndex

FAR_CORNER = {"right": (-1, 0), "left": (-1, -1)}  # (i, j) with -1 = n - 1
NEAR_CORNER = {"right": (0, 0), "left": (0, -1)}


def corner(ix: TowelIndex, ij: tuple[int, int]) -> int:
    i, j = ((c + ix.n) % ix.n for c in ij)
    return ix.vid(i, j)


def half_fold_layers(ix: TowelIndex, skip_crease: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) vertex pairs of the ideal half fold."""
    n, h = ix.n, ix.n // 2
    lo, hi = [], []
    for i in range(h - 1 if skip_crease else h):
        for j in range(n):
            lo.append(ix.vid(i, j))
            hi.append(ix.vid(n - 1 - i, j))
    return np.array(lo), np.array(hi)


def layer_violation(verts: np.ndarray, pairs: tuple[np.ndarray, np.ndarray], tol: float = 0.0) -> float:
    """Fraction of pairs whose upper vertex is below its lower one (by more than tol)."""
    lo, hi = pairs
    return float(np.mean(verts[hi, 2] < verts[lo, 2] - tol))


def corner_gaps(ix: TowelIndex, verts: np.ndarray) -> dict[str, tuple[float, float]]:
    """Per hand: (horizontal, vertical) distance from the far corner to its near corner [m]."""
    out = {}
    for side in FAR_CORNER:
        d = verts[corner(ix, FAR_CORNER[side])] - verts[corner(ix, NEAR_CORNER[side])]
        out[side] = (float(np.hypot(d[0], d[1])), float(abs(d[2])))
    return out


def convex_area(verts: np.ndarray) -> float:
    """Area of the convex hull seen from above [m^2]."""
    pts = verts[:, :2]
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def triangles(model: mujoco.MjModel, ix: TowelIndex) -> np.ndarray:
    adr, n = model.flex_elemdataadr[ix.flex], model.flex_elemnum[ix.flex]
    return model.flex_elem[adr : adr + 3 * n].reshape(n, 3).copy()


def self_intersections(verts: np.ndarray, tris: np.ndarray) -> int:
    """Crossings of the towel's mid-surface with itself (edges piercing triangles that share
    no vertex with them). 0 means no layer has passed through another."""
    edges = np.unique(np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    e1, e2 = b - a, c - a
    tmin, tmax = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
    count = 0
    for i, j in edges:
        p, q = verts[i], verts[j]
        cand = np.all((tmax >= np.minimum(p, q)) & (tmin <= np.maximum(p, q)), axis=1)
        cand &= ~np.any((tris == i) | (tris == j), axis=1)
        if not cand.any():
            continue
        d = q - p
        h = np.cross(d, e2[cand])
        det = np.einsum("ij,ij->i", e1[cand], h)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = p - a[cand]
        u = inv * np.einsum("ij,ij->i", sv, h)
        qv = np.cross(sv, e1[cand])
        v = inv * (qv @ d)
        t = inv * np.einsum("ij,ij->i", e2[cand], qv)
        count += int(np.sum(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)))
    return count


class SelfIntersection:
    """`self_intersections` for one mesh, fast enough to run every control step (~1 ms at
    N = 16 instead of ~50): the edges and the edge-triangle pairs that share a vertex are
    precomputed, and the bounding-box test runs on all pairs at once."""

    def __init__(self, tris: np.ndarray):
        self.tris = tris
        self.edges = np.unique(np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1),
                               axis=0)
        e, t = self.edges, tris
        self.shared = ((t[None, :, :] == e[:, None, 0:1]) | (t[None, :, :] == e[:, None, 1:2])).any(axis=2)

    def count(self, verts: np.ndarray) -> int:
        t, e = self.tris, self.edges
        a, b, c = verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
        tmin, tmax = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
        p, q = verts[e[:, 0]], verts[e[:, 1]]
        emin, emax = np.minimum(p, q), np.maximum(p, q)
        cand = ~self.shared
        for k in range(3):
            cand &= (tmax[None, :, k] >= emin[:, None, k]) & (tmin[None, :, k] <= emax[:, None, k])
        ei, ti = np.nonzero(cand)
        if len(ei) == 0:
            return 0
        p, d = p[ei], q[ei] - p[ei]
        a, e1, e2 = a[ti], b[ti] - a[ti], c[ti] - a[ti]
        h = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = p - a
        u = inv * np.einsum("ij,ij->i", sv, h)
        qv = np.cross(sv, e1)
        v = inv * np.einsum("ij,ij->i", qv, d)
        tt = inv * np.einsum("ij,ij->i", e2, qv)
        return int(np.sum(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (tt >= 0) & (tt <= 1)))


def broken(data: mujoco.MjData, verts: np.ndarray) -> str | None:
    """Simulation failure: NaN, explosion, or the towel more than 1 cm into the table."""
    if not np.isfinite(verts).all() or not np.isfinite(data.qacc).all():
        return "nan"
    if np.abs(verts).max() > 2.0:
        return "explode"
    if verts[:, 2].min() < -0.01:
        return "through_table"
    return None

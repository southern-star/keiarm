"""Two-sided towel rendering for mujoco.Renderer.

A flex has one material, so MuJoCo draws both sides of the towel with the front's checker.
To give the back its own plain colour, after `update_scene` we append one `mjGEOM_TRIANGLE`
per towel triangle to the `mjvScene`, offset just below the back surface along the
triangle's normal and wound to face away from the front (spike E4). Seen from the front the
real surface hides them; seen from the back they hide the real surface.

Only triangles whose back faces the camera are added: where the front faces the camera, the
front surface is in front of the offset back triangle anyway. A flat towel seen from above
adds none.

The added triangles are labelled as the towel's flex in segmentation rendering
(objtype mjOBJ_FLEX, objid = the flex id), so a segmentation mask covers both sides.

The browser draws the back with a `BackSide` material instead (Phase 4).
"""

import mujoco
import numpy as np

from keiarm.towel.scene import TOWEL_RADIUS, TowelIndex

# Below the back surface: the flex is drawn as a shell `radius` thick on each side of the
# mid-surface. At 1.05 * radius (0.1 mm under the shell) the shell's checker showed through
# the back in specks at cam_front's distance (depth precision, Phase 4); 1 mm under is clean.
BACK_OFFSET = 1.5 * TOWEL_RADIUS


def front_triangles(model: mujoco.MjModel, ix: TowelIndex) -> np.ndarray:
    """(ntri, 3) vertex indices wound so that the normal points up (+z) at rest: the front
    of the towel is the side that faces up when it is laid out."""
    adr, n = model.flex_elemdataadr[ix.flex], model.flex_elemnum[ix.flex]
    tris = model.flex_elem[adr : adr + 3 * n].reshape(n, 3).copy()
    r = ix.rest
    nz = np.cross(r[tris[:, 1]] - r[tris[:, 0]], r[tris[:, 2]] - r[tris[:, 0]])[:, 2]
    tris[nz < 0] = tris[nz < 0][:, [0, 2, 1]]
    return tris


class BacksideDrawer:
    """Appends the towel's back faces to a renderer's scene."""

    def __init__(self, model: mujoco.MjModel, ix: TowelIndex, rgb):
        self.ix = ix
        self.tris = front_triangles(model, ix)
        self.rgba = np.array([*rgb, 1.0], np.float32)
        self.size = np.ones(3)

    def add(self, scene: mujoco.MjvScene, data: mujoco.MjData) -> int:
        """Append the back faces for the towel's current shape, as seen from the scene's
        camera (call after update_scene). Returns the number added."""
        v = self.ix.verts(data)
        t = self.tris
        a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        cr = np.cross(b - a, c - a)
        nrm = cr / (np.linalg.norm(cr, axis=1, keepdims=True) + 1e-12)
        # Offset along the vertex normals (area-weighted), not each triangle's own: offset
        # triangle by triangle, cracks opened between neighbours where the towel bends and the
        # front's checker showed through the back in lines (Phase 4, the browser comparison).
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, t[:, k], cr)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        eye = 0.5 * (scene.camera[0].pos + scene.camera[1].pos)
        seen = np.einsum("ij,ij->i", nrm, eye - a) < 0.0  # the back faces the camera
        t, nrm = t[seen], nrm[seen]
        off = v - vn * BACK_OFFSET
        a, b, c = off[t[:, 0]], off[t[:, 1]], off[t[:, 2]]
        pos = a
        # The unit triangle (0,0,0), (1,0,0), (0,1,0) mapped onto a, c, b: wound the other way,
        # with its local +z along -n (the back's outward normal, used for lighting: flat per
        # triangle, which the browser reproduces).
        mats = np.stack([c - a, b - a, -nrm], axis=2).reshape(-1, 9)
        start = scene.ngeom
        count = min(len(a), scene.maxgeom - start)
        flex = self.ix.flex
        geoms = scene.geoms
        for k in range(count):
            g = geoms[start + k]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_TRIANGLE, self.size, pos[k], mats[k], self.rgba)
            g.objtype = mujoco.mjtObj.mjOBJ_FLEX
            g.objid = flex
            g.segid = start + k
        scene.ngeom = start + count
        return count

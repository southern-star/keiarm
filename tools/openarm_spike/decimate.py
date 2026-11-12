import pathlib
import shutil
import sys

import fast_simplification
import numpy as np
import trimesh

src = pathlib.Path(sys.argv[1])
keep = float(sys.argv[2])
dst = pathlib.Path(sys.argv[3])
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("cell"))
tot0 = tot1 = f0 = f1 = 0
for p in sorted((dst / "assets" / "visual").rglob("*")):
    if p.suffix not in (".stl", ".obj"):
        continue
    mesh = trimesh.load(p, force="mesh", process=False)
    mesh.merge_vertices()
    n0 = len(mesh.faces)
    tot0 += p.stat().st_size
    f0 += n0
    if n0 > 200:
        v, f = fast_simplification.simplify(
            np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces), target_reduction=1 - keep
        )
        mesh = trimesh.Trimesh(v, f, process=False)
    f1 += len(mesh.faces)
    if p.suffix == ".obj":
        p.write_text(
            trimesh.exchange.obj.export_obj(
                mesh, include_normals=False, include_texture=False, include_color=False
            )
        )
    else:
        mesh.export(p)
    tot1 += p.stat().st_size
stl = f1 * 50
print(
    f"keep={keep}: visual faces {f0} -> {f1}; files {tot0 / 1e6:.1f} MB -> {tot1 / 1e6:.2f} MB (as binary STL ~{stl / 1e6:.2f} MB)"
)

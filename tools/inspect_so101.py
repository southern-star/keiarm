"""Print the facts about the SO-101 MJCF that docs/so101_model.md records."""

import mujoco
import numpy as np

from keiarm.paths import SO101_XML


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(SO101_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    o = m.opt
    print(f"mujoco {mujoco.__version__}")
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} nmesh={m.nmesh}")
    print(f"timestep={o.timestep} integrator={mujoco.mjtIntegrator(o.integrator).name} "
          f"cone={mujoco.mjtCone(o.cone).name} solver={mujoco.mjtSolver(o.solver).name} "
          f"iterations={o.iterations} ls_iterations={o.ls_iterations} impratio={o.impratio}")
    print(f"total mass={m.body_mass.sum():.4f} kg  nkey={m.nkey} ncam={m.ncam} nsite={m.nsite}")
    print("\njoints (qpos order):")
    for j in range(m.njnt):
        print(f"  {j} {m.joint(j).name:14s} range={np.round(m.jnt_range[j], 4)} "
              f"deg={np.round(np.degrees(m.jnt_range[j]), 1)} damping={m.dof_damping[m.jnt_dofadr[j]]:.3f} "
              f"armature={m.dof_armature[m.jnt_dofadr[j]]:.3f} frictionloss={m.dof_frictionloss[m.jnt_dofadr[j]]:.3f}")
    print("\nactuators (ctrl order):")
    for a in range(m.nu):
        print(f"  {a} {m.actuator(a).name:14s} joint={m.joint(m.actuator_trnid[a, 0]).name:14s} "
              f"ctrlrange={np.round(m.actuator_ctrlrange[a], 4)} kp={m.actuator_gainprm[a, 0]:.2f} "
              f"kv={-m.actuator_biasprm[a, 2]:.3f} forcerange={m.actuator_forcerange[a]}")
    print("\ncameras:")
    for c in range(m.ncam):
        print(f"  {m.camera(c).name} body={m.body(m.cam_bodyid[c]).name} fovy={m.cam_fovy[c]}")
    print("\nsites (world pos at qpos=0):")
    for s in range(m.nsite):
        print(f"  {m.site(s).name} body={m.body(m.site_bodyid[s]).name} pos={np.round(d.site_xpos[s], 4)}")
        print(f"    xmat rows={np.round(d.site_xmat[s].reshape(3, 3), 3).tolist()}")
    print("\ncollision geoms (contype|conaffinity != 0):")
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            print(f"  {m.geom(g).name or '-':20s} body={m.body(m.geom_bodyid[g]).name:20s} "
                  f"type={mujoco.mjtGeom(m.geom_type[g]).name[7:]:8s} group={m.geom_group[g]} "
                  f"condim={m.geom_condim[g]} friction={m.geom_friction[g]} priority={m.geom_priority[g]}")
    # Jaw opening: distance between fixed and moving fingertip spheres over the gripper range.
    fid = m.geom("fixed_jaw_sph_tip1").id
    mid = m.geom("moving_jaw_sph_tip1").id
    qadr = m.joint("gripper").qposadr[0]
    print("\njaw tip distance vs gripper angle:")
    for q in np.linspace(*m.jnt_range[m.joint("gripper").id], 8):
        d.qpos[:] = 0
        d.qpos[qadr] = q
        mujoco.mj_forward(m, d)
        print(f"  q={q:+.3f} rad  tip gap={np.linalg.norm(d.geom_xpos[fid] - d.geom_xpos[mid]) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
